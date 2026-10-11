from __future__ import annotations

import csv
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

from molgap.scheduling.hardware import admission
from molgap.scheduling.plan import create_plan
from molgap.scheduling.queue import check_config, run_queue
from molgap.tracking.bridge import remote_file, sync_report
from molgap.tracking.feishu import FeishuError
from molgap.tracking.inference import measure_prediction
from molgap.tracking.lark_cli import LarkCLIClient
from molgap.tracking.records import read_json, write_json
from molgap.tracking.resources import process_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return yaml.safe_load((ROOT / "configs/auto_training.yaml").read_text())


def hardware():
    return {
        "cpus": 15,
        "ram_total_mib": 92160,
        "ram_available_mib": 85000,
        "disk_free_gib": 25,
        "gpu_total_mib": 24576,
        "gpu_free_mib": 24575,
        "gpu_used_mib": 1,
        "gpu_util_pct": 0,
    }


def test_full_plan_is_63_paired_jobs_and_excludes_paused(policy):
    parent = ROOT / "artifacts/test_scheduler"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        plan = create_plan(ROOT, Path(directory), policy)
        assert len(plan["jobs"]) == 63
        assert not {"cormorant", "lie_transformer"} & {j["model"] for j in plan["jobs"]}
        for job in plan["jobs"]:
            cfg = yaml.safe_load(Path(job["config"]).read_text())
            settings = cfg.get("training", cfg.get("pretraining"))
            assert settings["seed"] == cfg["data"]["split"]["seed"] == job["seed"]
            if job["kind"] != "ml":
                assert (settings["batch_size"], settings["epochs"]) == (256, 100)
            if job["model"] == "3dinfomax":
                assert job["depends_on"] == [f"3dinfomax_pretrain_seed{job['seed']}"]
            assert admission(job, [], hardware(), policy) is None


def job(gpu=1024, exclusive=False):
    return {"gpu_mib": gpu, "ram_mib": 2048, "cpu_slots": 6, "exclusive": exclusive}


def test_admission_reserves_startup_peaks_and_checks_live_hardware(policy):
    hw = hardware()
    assert admission(job(14000), [job(11000)], hw, policy) == "GPU reservation"
    assert admission(job(1000), [job(22000, True)], hw, policy) == "exclusive GPU job"
    hw["gpu_util_pct"] = 98
    assert admission(job(), [], hw, policy) == "GPU busy"
    hw["gpu_util_pct"] = 0
    hw["gpu_free_mib"] = 2200
    assert admission(job(), [], hw, policy) == "available GPU memory"
    hw["disk_free_gib"] = 5
    assert admission(job(), [], hw, policy) == "disk reserve"


def test_exclusive_alignn_fits_and_external_allocation_blocks(policy):
    hw = hardware()
    hw["gpu_free_mib"] = 24135  # Actual server includes driver-reserved memory.
    assert admission(job(22.234 * 1024, True), [], hw, policy) is None
    hw["gpu_used_mib"] = 600
    assert admission(job(22.234 * 1024, True), [], hw, policy) == "external GPU allocation"
    hw["gpu_used_mib"] = 1
    hw["ram_available_mib"] = 22000
    assert admission(job(), [], hw, policy) == "available RAM"


def test_queue_isolates_failure_and_resumes_without_rerun(tmp_path, policy, monkeypatch):
    from molgap.scheduling import queue

    policy = {**policy, "sample_seconds": 0.01, "launch_spacing_seconds": 0}
    monkeypatch.setattr(queue, "snapshot", lambda *a: hardware())
    monkeypatch.setattr(queue, "sample_resources", lambda *a: {"pss_mib": 10})
    original_reader = queue.build_record
    attempted = [False]

    def transient_read(run_dir, state):
        if not attempted[0]:
            attempted[0] = True
            raise ValueError("Trainer is writing JSON")
        return original_reader(run_dir, state)

    monkeypatch.setattr(queue, "build_record", transient_read)
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.csv").write_text("sample_id,smiles,delta_e\na,C,1\nb,CC,2\n")
    jobs = []
    for name, fail in (("failure", True), ("success", False)):
        run = tmp_path / "runs" / name
        cfg = tmp_path / (name + ".yaml")
        cfg.write_text(
            yaml.safe_dump(
                {"data": {"processed_dir": "data", "split": {"seed": 42}}, "training": {"seed": 42}}
            )
        )
        code = (
            "raise SystemExit(2)"
            if fail
            else (
                "from pathlib import Path; import json; "
                f"p=Path({str(run)!r}); (p/'predictions').mkdir(parents=True); "
                "(p/'predictions/test_predictions.csv').write_text("
                "'sample_id,delta_e_pred\\na,1\\nb,2\\n'); "
                "(p/'metrics.json').write_text(json.dumps({'mae':0})); "
                "(p/'model.json').write_text('{}')"
            )
        )
        jobs.append(
            {
                "id": name,
                "kind": "ml",
                "model": name,
                "seed": 42,
                "config": str(cfg),
                "run_dir": str(run),
                "commands": [[sys.executable, "-c", code]],
                "gpu_mib": 0,
                "ram_mib": 100,
                "cpu_slots": 1,
                "exclusive": False,
                "priority": 2 if fail else 1,
                "depends_on": [],
                "status": "pending",
            }
        )
    write_json(tmp_path / "plan.json", {"root": str(tmp_path), "policy": policy, "jobs": jobs})
    assert run_queue(tmp_path) == 1
    state = read_json(tmp_path / "runtime/state.json")
    assert [j["status"] for j in state["jobs"]] == ["failed", "completed"]
    assert state["jobs"][1]["verification"]["test_rows"] == 2
    assert any(j.get("monitor_warning") for j in state["jobs"])
    old_start = state["jobs"][1]["started_at"]
    assert run_queue(tmp_path) == 1
    assert read_json(tmp_path / "runtime/state.json")["jobs"][1]["started_at"] == old_start


def test_stop_terminates_worker_and_training_child_and_updates_csv(tmp_path, policy):
    directory = tmp_path / "queue"
    directory.mkdir()
    config = directory / "config.yaml"
    config.write_text("data: {split: {seed: 42}}\ntraining: {seed: 42}\n")
    child_pid = directory / "child.pid"
    code = (
        "from pathlib import Path; import os, time; "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    task = {
        "id": "sleep",
        "model": "sleep",
        "kind": "ml",
        "seed": 42,
        "config": str(config),
        "run_dir": str(directory / "run"),
        "commands": [[sys.executable, "-c", code]],
        "gpu_mib": 0,
        "ram_mib": 1,
        "cpu_slots": 1,
        "exclusive": False,
        "priority": 1,
        "depends_on": [],
        "status": "pending",
    }
    policy = {
        **policy,
        "sample_seconds": 0.05,
        "launch_spacing_seconds": 0,
        "ram_reserve_mib": 0,
        "disk_reserve_gib": 0,
        "cpu_reserve": 0,
    }
    write_json(directory / "plan.json", {"root": str(ROOT), "policy": policy, "jobs": [task]})
    with (directory / "controller.log").open("w") as log:
        controller = subprocess.Popen(
            [sys.executable, "-m", "molgap.scheduling", "run", "--directory", str(directory)],
            cwd=ROOT,
            env={**os.environ, "PATH": "/nonexistent"},
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while not child_pid.exists() and time.monotonic() < deadline:
                assert controller.poll() is None
                time.sleep(0.05)
            assert child_pid.exists()
            os.kill(controller.pid, signal.SIGTERM)
            assert controller.wait(timeout=10) == 1
            job_state = read_json(directory / "runtime/state.json")["jobs"][0]
            assert job_state["status"] == "interrupted"
            assert process_identity(job_state["pid"]) is None
            assert process_identity(int(child_pid.read_text())) is None
            report = read_json(directory / "runtime/report.json")
            assert report["terminal"] and report["stopped"]
            with (directory / "runtime/summary.csv").open() as stream:
                assert next(csv.DictReader(stream))["状态"] == "interrupted"
        finally:
            if controller.poll() is None:
                controller.terminate()
                controller.wait(timeout=10)


def test_frozen_config_edit_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\n")
    task = {
        "id": "model",
        "config": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    check_config(task)
    path.write_text("seed: 2026\n")
    with pytest.raises(ValueError, match="Frozen configuration changed"):
        check_config(task)


def test_remote_bridge_rejects_parent_traversal_before_ssh():
    with pytest.raises(ValueError, match="inside the configured project"):
        remote_file({"root": "/root/project"}, "/root/project/../../etc/passwd")


def test_cloud_failure_does_not_prevent_other_records(tmp_path):
    class Client:
        destination = {"hardware_table_id": "hardware"}
        accepted = []

        def sync_record(self, record):
            if record["run_id"] == "failed_sync":
                raise FeishuError("Temporary network failure")
            self.accepted.append(record["run_id"])
            return "rec", tmp_path / "receipt.json"

    client = Client()
    report = {
        "records": [
            {"run_id": name, "fields": {"状态": "running"}} for name in ("failed_sync", "good_sync")
        ]
    }
    failures = sync_report(report, client, tmp_path)
    assert len(failures) == 1
    assert client.accepted == ["good_sync"]


def test_new_field_schema_supports_platform_precision():
    from molgap.tracking.lark_cli import table_fields

    fields = {f["name"]: f for f in table_fields()}
    assert fields["测试MSE_Hartree2"]["style"]["precision"] == 4
    assert fields["指标完整精度_JSON"]["type"] == "text"


def test_remote_launcher_checks_cloud_configuration_before_ssh(monkeypatch):
    from molgap.scheduling import remote

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["remote", "--directory", "artifacts/never_started"])
    monkeypatch.setattr(remote, "read_json", lambda *a: {})
    calls = []
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(SystemExit) as exc:
        remote.main()
    assert exc.value.code == 2
    assert not calls


def test_remote_launcher_forwards_custom_policy_to_queue_and_bridge(monkeypatch):
    from molgap.scheduling import remote

    monkeypatch.chdir(ROOT)
    directory = ROOT / "artifacts/test_scheduler"
    directory.mkdir(parents=True, exist_ok=True)
    calls = []

    class Client:
        def __init__(self, *a):
            pass

        def ensure_schema(self):
            calls.append("schema")

    with tempfile.TemporaryDirectory(dir=directory) as work:
        custom = Path(work) / "custom.yaml"
        custom.write_bytes((ROOT / "configs/auto_training.yaml").read_bytes())
        relative = str(custom.relative_to(ROOT))
        monkeypatch.setattr(sys, "argv", ["remote", "--config", relative])
        monkeypatch.setattr(
            remote,
            "read_json",
            lambda *a: {"base_token": "base", "table_id": "tbl", "hardware_table_id": "hw"},
        )
        monkeypatch.setattr(remote, "LarkCLIClient", Client)
        monkeypatch.setattr(remote.subprocess, "run", lambda argv, **k: calls.append(argv))
        remote.main()
    assert calls[0] == "schema"
    assert "--config " + relative in calls[1][-1]
    assert calls[2][-2:] == ["--policy", relative]


def test_inference_scope_and_no_invented_flops(tmp_path):
    with measure_prediction(tmp_path, 2, "model_predict_precomputed_features"):
        sum(range(1000))
    timing = read_json(tmp_path / "inference_timing.json")
    assert timing["samples_per_second"] > 0
    assert timing["flops"] is None


def test_cli_recovers_record_after_lost_response(tmp_path, monkeypatch):
    from molgap.tracking import lark_cli

    calls = []

    def fake(arguments, **kwargs):
        calls.append(arguments)
        if "+record-list" in arguments:
            return {"record_id_list": ["rec_saved"], "has_more": False}
        return {"record_id": "rec_saved"}

    monkeypatch.setattr(lark_cli, "cli", fake)
    client = LarkCLIClient({"base_token": "base", "table_id": "tbl"}, tmp_path)
    record_id, receipt = client.upsert({"运行ID": "run", "模型": "test"}, "运行ID")
    assert record_id == "rec_saved"
    assert "--record-id" in calls[-1]
    assert read_json(receipt)["run_id"] == "run"
