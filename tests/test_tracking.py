from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

from molgap.tracking import feishu, resources
from molgap.tracking.__main__ import SyncWorker, monitor
from molgap.tracking.feishu import FeishuClient, FeishuError, Settings
from molgap.tracking.records import SCHEMA, build_record, find_runs, read_json, write_json


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "graph_model" / "seed42"
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "graph_model", "protocol_id": "paired_split_gap_mae_v2"},
                "training": {"seed": 42, "batch_size": 256, "epochs": 100},
                "data": {"split": {"seed": 42}, "label_unit": "hartree"},
            }
        )
    )
    return path


@pytest.mark.parametrize("nested", [False, True])
def test_record_handles_model_metric_formats_and_backfills_mse(run_dir, nested):
    metrics = {"mae": 0.2, "rmse": 0.3, "r2": float("nan"), "mape": 1.5}
    payload = (
        {"targets": {"delta_e": metrics}, "rows": 10}
        if nested
        else {"target": "delta_e", "n": 10, **metrics}
    )
    (run_dir / "metrics.json").write_text(json.dumps(payload))
    write_json(run_dir / "training_timing.json", {"fit_seconds": 7.5})
    (run_dir / "history.csv").write_text("epoch,train_loss,val_mae\n1,0.9,0.4\n2,0.6,0.25\n")
    record = build_record(run_dir)
    fields = record["fields"]
    assert fields["测试MSE_Hartree2"] == pytest.approx(0.09)
    assert fields["测试R2"] is None
    assert fields["测试样本数"] == 10
    assert fields["当前epoch"] == 2
    assert fields["验证MAE_Hartree"] == 0.25
    assert fields["训练耗时_s"] == 7.5
    assert fields["峰值RSS_MiB"] is None
    assert fields["状态"] == "completed"
    assert fields["模型"] == "graph_model"
    assert fields["训练seed"] == fields["划分seed"] == 42
    assert record["mse_source"] == "rmse_squared"
    assert set(fields) == set(SCHEMA)
    assert build_record(run_dir)["run_id"] == record["run_id"]
    assert find_runs([run_dir.parent, run_dir]) == [run_dir.resolve()]


def test_saved_failure_wins_over_metrics_and_explicit_mse_is_preserved(run_dir):
    write_json(run_dir / "metrics.json", {"mse": 0.08, "rmse": 0.3})
    write_json(
        run_dir / "tracking/state.json",
        {"status": "failed", "coverage": "attached_partial_sampled"},
    )
    record = build_record(run_dir)
    assert record["fields"]["状态"] == "failed"
    assert record["fields"]["测试MSE_Hartree2"] == 0.08
    assert record["fields"]["采样覆盖"] == "attached_partial_sampled"
    assert record["mse_source"] == "metrics"


def test_metrics_with_a_different_unit_cannot_be_mislabeled(run_dir):
    write_json(run_dir / "metrics.json", {"unit": "eV", "mae": 0.2})
    with pytest.raises(ValueError, match="Hartree"):
        build_record(run_dir)


def test_lightning_progress_keeps_metrics_across_sparse_rows(run_dir):
    logs = run_dir / "logs/lightning/version_0"
    logs.mkdir(parents=True)
    (logs / "metrics.csv").write_text(
        "epoch,train_loss,val/gap_mae,test/mae\n0,,0.5,\n0,2,,\n1,,0.4,\n2,,,0.3\n"
    )
    fields = build_record(run_dir)["fields"]
    assert fields["当前epoch"] == 2
    assert fields["训练loss"] == 2
    assert fields["验证MAE_Hartree"] == 0.4


def test_settings_parse_domestic_base_url_without_logging_secret(monkeypatch):
    for name in ("FEISHU_APP_TOKEN", "FEISHU_TABLE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "super_secret")
    monkeypatch.setenv(
        "FEISHU_BITABLE_URL", "https://tenant.feishu.cn/base/base_test?table=tbl_test&view=vew_test"
    )
    settings = Settings.from_env()
    assert settings.app_token == "base_test"
    assert settings.table_id == "tbl_test"
    assert "super_secret" not in repr(settings)
    monkeypatch.setenv(
        "FEISHU_BITABLE_URL", "https://tenant.feishu.cn/wiki/not_a_base?table=tbl_test"
    )
    with pytest.raises(ValueError, match="wiki"):
        Settings.from_env()


@pytest.fixture
def api(monkeypatch):
    requests, rows, tokens = [], {}, []
    create_tokens = {}

    def transport(request, timeout):
        path = urlparse(request.full_url).path
        body = json.loads(request.data) if request.data else {}
        requests.append(
            (request.get_method(), path, body, parse_qs(urlparse(request.full_url).query))
        )
        assert timeout == 10
        if path.endswith("/internal"):
            tokens.append(body)
            result = {"code": 0, "tenant_access_token": "secret_access_token", "expire": 7200}
        else:
            assert request.get_header("Authorization") == "Bearer secret_access_token"
            if path.endswith("/fields"):
                data = {
                    "items": [{"field_name": name, "type": kind} for name, kind in SCHEMA.items()]
                }
            elif path.endswith("/search"):
                run_id = body["filter"]["conditions"][0]["value"][0]
                assert body["filter"]["conditions"][0]["operator"] == "is"
                data = {
                    "items": [
                        {"record_id": key}
                        for key, fields in rows.items()
                        if fields["运行ID"] == run_id
                    ]
                }
            elif request.get_method() == "POST":
                assert path.endswith("/records")
                token = requests[-1][3]["client_token"][0]
                assert len(token) == 36
                record_id = create_tokens.get(token)
                if record_id is None:
                    record_id = f"rec{len(rows) + 1}"
                    create_tokens[token] = record_id
                    rows[record_id] = body["fields"]
                data = {"record": {"record_id": record_id}}
            else:
                assert request.get_method() == "PUT"
                rows[path.rsplit("/", 1)[1]] = body["fields"]
                data = {}
            result = {"code": 0, "data": data}
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(feishu, "urlopen", transport)
    return (
        FeishuClient(Settings("cli_test", "app_secret", "base_test", "tbl_test")),
        requests,
        rows,
        tokens,
    )


def test_sync_creates_once_and_updates_on_retry_even_without_receipt(run_dir, api):
    client, calls, rows, tokens = api
    record = build_record(run_dir)
    client.sync(run_dir, record)
    client.sync(run_dir, {**record, "fields": {**record["fields"], "状态": "failed"}})
    (run_dir / "tracking/feishu_sync.json").unlink()
    client.sync(run_dir, record)
    assert len(rows) == 1
    assert len(tokens) == 1
    assert sum(method == "POST" and path.endswith("/records") for method, path, _, _ in calls) == 1
    assert read_json(run_dir / "tracking/feishu_sync.json")["record_id"] == "rec1"
    assert "app_secret" not in (run_dir / "tracking/feishu_sync.json").read_text()


def test_expired_token_refreshes_once(monkeypatch, api):
    client, _, _, tokens = api
    transport = feishu.urlopen
    failed = False

    def expire_once(request, timeout):
        nonlocal failed
        if not request.full_url.endswith("/internal") and not failed:
            failed = True
            return io.BytesIO(b'{"code":99991663,"msg":"expired"}')
        return transport(request, timeout)

    monkeypatch.setattr(feishu, "urlopen", expire_once)
    client.ensure_schema()
    assert len(tokens) == 2


def test_lost_create_response_reuses_durable_idempotency_key(run_dir, api, monkeypatch):
    client, calls, rows, _ = api
    transport = feishu.urlopen
    response_lost = False
    stale_search = False

    def lose_response_once(request, timeout):
        nonlocal response_lost, stale_search
        if request.get_method() == "POST" and request.full_url.split("?")[0].endswith("/records"):
            response = transport(request, timeout)
            if not response_lost:
                response_lost = True
                raise OSError("simulated lost response")
            return response
        if (
            response_lost
            and request.full_url.split("?")[0].endswith("/search")
            and not stale_search
        ):
            stale_search = True
            return io.BytesIO(b'{"code":0,"data":{"items":[]}}')
        return transport(request, timeout)

    monkeypatch.setattr(feishu, "urlopen", lose_response_once)
    record = build_record(run_dir)
    with pytest.raises(FeishuError, match="connection failed"):
        client.sync(run_dir, record)
    assert "create_token" in read_json(run_dir / "tracking/feishu_sync.json")
    latest = {**record, "fields": {**record["fields"], "状态": "failed"}}
    client.sync(run_dir, latest)
    created = [
        query["client_token"]
        for method, path, _, query in calls
        if method == "POST" and path.endswith("/records")
    ]
    assert len(created) == 2 and created[0] == created[1]
    assert len(rows) == 1 and rows["rec1"]["状态"] == "failed"


def test_business_and_http_errors_do_not_expose_response_or_credentials(monkeypatch, api):
    client = api[0]
    monkeypatch.setattr(
        feishu,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            b'{"code":91403,"msg":"app_secret secret_access_token"}'
        ),
    )
    with pytest.raises(FeishuError, match="91403") as error:
        client.ensure_schema()
    assert "app_secret" not in str(error.value)

    def forbidden(request, timeout):
        raise HTTPError(request.full_url, 403, "secret_access_token", {}, None)

    monkeypatch.setattr(feishu, "urlopen", forbidden)
    with pytest.raises(FeishuError, match="HTTP 403") as error:
        client.ensure_schema()
    assert "secret_access_token" not in str(error.value)


def test_schema_conflicts_and_duplicate_rows_fail_before_writing(monkeypatch, api, run_dir):
    client = api[0]
    monkeypatch.setattr(client, "list_fields", lambda: {**SCHEMA, "运行ID": 2})
    with pytest.raises(FeishuError, match="Incompatible"):
        client.ensure_schema(create=True)
    monkeypatch.setattr(
        client, "request", lambda *args: {"items": [{"record_id": "r1"}, {"record_id": "r2"}]}
    )
    with pytest.raises(FeishuError, match="Duplicate"):
        client.find_record("run")


def test_process_tree_memory_has_rss_fallback_and_missing_pss(tmp_path):
    for pid, parent, rss in ((10, 1, 1024), (11, 10, 2048), (12, 11, 3072)):
        directory = tmp_path / str(pid)
        directory.mkdir()
        fields = ["S", str(parent)] + ["0"] * 17 + ["123"]
        (directory / "stat").write_text(f"{pid} (a (process)) " + " ".join(fields))
        (directory / "status").write_text(f"VmRSS:\t{rss} kB\n")
    pids, result = resources.process_memory(10, tmp_path)
    assert pids == {10, 11, 12}
    assert result["rss_mib"] == 6
    assert result["pss_mib"] is None
    assert result["rss_processes_measured"] == 3
    assert resources.process_identity(10, tmp_path) == "10:123"


def test_gpu_query_keeps_device_memory_when_per_process_memory_is_unavailable(monkeypatch):
    def query(command, **kwargs):
        return "400\n" if "--query-gpu=memory.used" in command else "10, [N/A]\n"

    monkeypatch.setattr(resources.subprocess, "check_output", query)
    assert resources.gpu_memory({10}) == {"gpu_process_mib": None, "gpu_device_mib": 400}


@pytest.mark.parametrize("exit_code,expected", [(0, "completed"), (7, "failed")])
def test_monitor_persists_resources_and_child_exit_status(
    run_dir, monkeypatch, exit_code, expected
):
    monkeypatch.setattr(
        "molgap.tracking.__main__.sample_resources",
        lambda pid: {
            "rss_mib": 20.0,
            "pss_mib": 15.0,
            "gpu_process_mib": None,
            "gpu_device_mib": None,
        },
    )
    write_json(run_dir / "metrics.json", {"mae": 0.1, "mse": 0.04, "rmse": 0.2})
    child = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep(.15); raise SystemExit({exit_code})"],
        start_new_session=True,
    )
    try:
        assert monitor(run_dir, child.pid, 0.02, 60, None, child) == exit_code
    finally:
        child.wait()
    record = read_json(run_dir / "tracking/summary.json")
    assert record["fields"]["状态"] == expected
    assert record["fields"]["峰值RSS_MiB"] == 20
    assert record["fields"]["峰值进程显存_MiB"] is None
    assert read_json(run_dir / "tracking/state.json")["sample_count"] > 0
    assert (run_dir / "tracking/samples.jsonl").is_file()


def test_background_network_failure_preserves_latest_local_record(run_dir):
    class Offline:
        def sync(self, path, record):
            time.sleep(0.1)
            raise FeishuError("offline")

    worker = SyncWorker(Offline(), run_dir)
    try:
        worker.submit(build_record(run_dir))
        write_json(run_dir / "tracking/summary.json", build_record(run_dir, {"status": "failed"}))
        worker.submit(build_record(run_dir, {"status": "failed"}))
    finally:
        worker.close()
    assert read_json(run_dir / "tracking/sync_status.json")["status"] == "pending"
    assert read_json(run_dir / "tracking/summary.json")["fields"]["状态"] == "failed"
