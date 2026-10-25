"""Smoke orchestration must isolate failures and never reuse stale-source passes."""

import argparse
import csv
import json
from pathlib import Path

from scripts import smoke_qm9_epoch200 as smoke


def test_smoke_queue_continues_and_retests_source_changes(tmp_path, monkeypatch):
    script = tmp_path / "scripts/smoke_qm9_epoch200.py"
    script.parent.mkdir()
    script.write_text("# smoke runner fixture\n")
    module = tmp_path / "src/molgap/model.py"
    module.parent.mkdir(parents=True)
    module.write_text("revision = 1\n")
    monkeypatch.setattr(smoke, "__file__", str(script))
    queue = tmp_path / "docs/features/qm9-epoch200-modular-study/training_queue.csv"
    queue.parent.mkdir(parents=True)
    configs = tmp_path / "configs"
    configs.mkdir()
    with queue.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "model", "variant", "seed", "config"])
        writer.writeheader()
        for i in range(smoke.EXPECTED_QUEUE_JOBS):
            (configs / f"{i}.yaml").write_text("fixture: true\n")
            writer.writerow(
                dict(
                    id=f"job{i}",
                    model="chemprop",
                    variant="single_task",
                    seed=42,
                    config=f"configs/{i}.yaml",
                )
            )
    calls = []

    class Process:
        pid = 1

        def __init__(self, argv, **kwargs):
            run = Path(argv[argv.index("--worker-run") + 1])
            calls.append(run)
            self.returncode = 1 if len(calls) == 1 else 0
            if self.returncode == 0:
                run.mkdir(parents=True)
                (run / "smoke_audit.json").write_text(
                    json.dumps(
                        dict(
                            timing=dict(peak_gpu_allocated_bytes=1024, peak_gpu_reserved_bytes=2048)
                        )
                    )
                )
                (run / "training_log.csv").write_text("epoch_seconds\n0.1\n")

        def poll(self):
            return self.returncode

    monkeypatch.setattr(smoke.subprocess, "Popen", Process)
    args = argparse.Namespace(
        model=None,
        job=["job0", "job1", "job2"],
        output=tmp_path / "results",
        rows=500,
        prune_successful_checkpoints=True,
        rerun_passed=False,
    )
    assert smoke.campaign(args) == 1
    assert len(calls) == 3  # First failure cannot cancel the remaining two jobs.
    assert smoke.campaign(args) == 0
    assert len(calls) == 4  # Only the failed job is retried on the same source.
    module.write_text("revision = 2\n")
    assert smoke.campaign(args) == 0
    assert len(calls) == 7  # Previous passes do not certify changed source code.
    report = json.loads((args.output / "results.json").read_text())
    assert report["passed"] == 3 and report["failed"] == 0 and len(report["attempts"]) == 7
