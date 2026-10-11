"""Check diagnostic configs and resource reports use the batch actually selected."""

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("override", [None, 16, 32])
def test_smoke_batch_is_configured_and_formal_source_is_unchanged(tmp_path, override):
    runner = _script("run_one_epoch_smoke")
    source = ROOT / "configs/qm9_models/cormorant.yaml"
    original = source.read_bytes()
    path, _, config = runner.prepare(ROOT, tmp_path, "cormorant", "qm9", source, override)
    expected = yaml.safe_load(original)["training"]["batch_size"] if override is None else override
    assert config["training"]["batch_size"] == config["smoke_suite"]["batch_size"] == expected
    assert config["training"]["epochs"] == 1
    assert config["data"] == yaml.safe_load(original)["data"]
    assert yaml.safe_load(path.read_text())["training"]["batch_size"] == expected
    assert source.read_bytes() == original


def test_resource_csv_reports_per_model_batch_and_traditional_na(tmp_path):
    monitor = _script("monitor_smoke_resources")
    report = {
        "models": [
            {
                "model": "cormorant",
                "kind": "qm9",
                "status": "success",
                "batch_size": 32,
                "epochs": 1,
                "run_dir": "test",
            },
            {"model": "training_mean", "kind": "ml", "status": "success", "run_dir": "test"},
        ]
    }
    monitor.export(tmp_path, report, {}, 1.0)
    with (tmp_path / "resources.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["batch_size"] == "32"
    assert rows[1]["batch_size"] == rows[1]["epochs"] == "N/A"


def test_protocol_audit_rejects_unapproved_batch(monkeypatch):
    protocol = _script("audit_paired_seed_protocol")
    assert protocol.audit(ROOT)["status"] == "pass"
    original = protocol._yaml

    def altered(path):
        payload = original(path)
        if path.name == "cormorant.yaml":
            payload["training"]["batch_size"] *= 2
        return payload

    monkeypatch.setattr(protocol, "_yaml", altered)
    report = protocol.audit(ROOT)
    assert report["status"] == "fail"
    assert not report["neural_configs"]["qm9_models/cormorant.yaml"]["checks"][
        "approved_batch_size"
    ]
