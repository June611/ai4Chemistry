"""Cross-machine queue membership, safe script routing, resume and report exchange."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from molgap.data.neural_properties import NEURAL_PROPERTY_PROTOCOL
from molgap.data.qm9_properties import NEW_TARGETS, PROPERTY_UNITS, SEEDS, digest
from molgap.experiments import neural_groups as groups
from molgap.tracking.records import read_json, write_json

ROOT = Path(__file__).parents[1]


def test_three_profiles_cover_180_experiments_without_overlap():
    result = groups.audit_profiles(ROOT)
    assert {g: r["downstream_jobs"] for g, r in result["groups"].items()} == {
        "high": 54,
        "medium": 72,
        "low": 54,
    }
    assert result["groups"]["high"]["models"] == ["dimenet_pp", "3dinfomax", "torchmd_net"]
    assert result["groups"]["medium"]["models"] == ["esa", "painn", "egnn", "schnet"]
    assert result["groups"]["low"]["models"] == [
        "chemprop_single_gap",
        "smiles_transformer",
        "spe_transformer",
    ]


@pytest.mark.parametrize("case", ["overlap", "seed", "budget", "timing"])
def test_modified_profiles_reject_protocol_drift(tmp_path, case):
    shutil.copytree(ROOT / groups.PROFILE_ROOT, tmp_path / groups.PROFILE_ROOT)
    source = Path("docs/features/qm9-one-epoch-smoke/verification/resources_complete.csv")
    (tmp_path / source).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / source, tmp_path / source)
    path = tmp_path / groups.PROFILE_ROOT / "medium.yaml"
    cfg = yaml.safe_load(path.read_text())
    if case == "overlap":
        high = yaml.safe_load((tmp_path / groups.PROFILE_ROOT / "high.yaml").read_text())
        cfg["models"]["dimenet_pp"] = high["models"]["dimenet_pp"]
    elif case == "seed":
        cfg["seeds"] = [42]
    elif case == "budget":
        cfg["defaults"]["batch_size"] = 128
    else:
        cfg["models"]["esa"]["smoke_fit_seconds"] = 1.0
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        groups.audit_profiles(tmp_path)


def _batch(root):
    directory = root / "high_test"
    snapshot = dict(
        queue_group="high",
        models={"dimenet_pp": {}},
        data_root="data",
        defaults=dict(batch_size=256, epochs=100, patience=20, min_delta=0.0),
    )
    directory.mkdir()
    (directory / "tasks.yaml").write_text(yaml.safe_dump(snapshot))
    lane = directory / "lanes/gpu0"
    write_json(
        lane / "plan.json",
        dict(
            jobs=[
                dict(
                    model="dimenet_pp",
                    target="mu",
                    seed=42,
                    kind="qm9",
                    id="dimenet_pp_mu_seed42",
                    status="pending",
                ),
            ]
        ),
    )
    write_json(
        directory / "plan.json",
        dict(
            root=str(root),
            tasks_sha256=digest(directory / "tasks.yaml"),
            models=["dimenet_pp"],
            targets=["mu"],
            seeds=[42],
            gpus=[0],
            lanes=[str(lane)],
        ),
    )
    return directory


def test_resume_uses_disk_plan_and_does_not_prepare_again(tmp_path, monkeypatch):
    directory = _batch(tmp_path)

    def forbidden(*a, **kw):
        pytest.fail("Existing plan must not be regenerated")

    monkeypatch.setattr(groups, "prepare_group", forbidden)
    monkeypatch.setattr(groups.queues, "run", lambda path: 17)
    assert groups.start_group(tmp_path, "high", directory, gpus=[0]) == 17
    with pytest.raises(ValueError, match="Frozen gpus"):
        groups.start_group(tmp_path, "high", directory, gpus=[1])
    with pytest.raises(ValueError, match="another group"):
        groups.start_group(tmp_path, "low", directory)


def test_batch_rejects_snapshot_tampering_and_machine_path_change(tmp_path):
    directory = _batch(tmp_path)
    with pytest.raises(ValueError, match="destination"):
        groups.check_batch(ROOT, directory, "high")
    path = directory / "tasks.yaml"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        groups.check_batch(tmp_path, directory, "high")


@pytest.mark.parametrize("group", groups.GROUPS)
def test_shell_wrappers_forward_prepare_and_quoted_paths(tmp_path, group):
    interpreter = tmp_path / "fake_python"
    interpreter.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    interpreter.chmod(0o755)
    directory = tmp_path / f"{group}_path with spaces"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / f"train_qm9_{group}.sh"),
            "prepare",
            "--directory",
            str(directory),
            "--gpus",
            "0",
            "1",
        ],
        cwd=tmp_path,
        env={**os.environ, "QM9_PYTHON": str(interpreter)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        "-m",
        "molgap.experiments.neural_groups",
        group,
        "prepare",
        "--directory",
        str(directory),
        "--gpus",
        "0",
        "1",
    ]


@pytest.fixture
def reports(tmp_path):
    outputs = []
    for group in groups.GROUPS:
        cfg = yaml.safe_load((ROOT / groups.PROFILE_ROOT / f"{group}.yaml").read_text())
        output = tmp_path / group
        output.mkdir()
        rows = []
        for index, model in enumerate(cfg["models"]):
            for target in NEW_TARGETS:
                for i, seed in enumerate(SEEDS):
                    error = 0.1 + index * 0.01 + i * 0.001
                    rows.append(
                        dict(
                            model=model,
                            target=target,
                            seed=seed,
                            status="completed",
                            unit=PROPERTY_UNITS[target],
                            mae=error,
                            mse=error**2,
                            rmse=error,
                            r2=0.9 - i * 0.01,
                            fit_seconds=100 + i,
                            gpu_process_mib=1000 + i,
                        )
                    )
        runs = pd.DataFrame(rows)
        summary = []
        for (model, target), frame in runs.groupby(["model", "target"]):
            row = dict(
                model=model,
                target=target,
                unit=PROPERTY_UNITS[target],
                n_seeds=3,
                complete_three_seeds=True,
                target_scaling="standard",
            )
            for metric in groups.queues.METRICS:
                row[metric + "_mean"] = frame[metric].mean()
                row[metric + "_std"] = frame[metric].std(ddof=1)
            summary.append(row)
        runs.to_csv(output / "runs.csv", index=False)
        pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
        write_json(output / "verification.json", dict(complete=True))
        write_json(
            output / "group_report.json",
            dict(
                group=group,
                models=list(cfg["models"]),
                seeds=list(SEEDS),
                targets=list(NEW_TARGETS),
                protocol_id=NEURAL_PROPERTY_PROTOCOL,
                defaults=cfg["defaults"],
                data_identity_sha256="same-qm9-data",
                files={
                    p: digest(output / p) for p in ("runs.csv", "summary.csv", "verification.json")
                },
            ),
        )
        outputs.append(output)
    return outputs


def test_merge_produces_180_runs_and_60_sorted_property_summaries(reports, tmp_path):
    output = tmp_path / "merged"
    groups.merge_reports(reports, output)
    assert len(pd.read_csv(output / "runs.csv")) == 180
    assert len(pd.read_csv(output / "summary.csv")) == 60
    assert read_json(output / "verification.json")["complete"]
    for target in NEW_TARGETS:
        assert pd.read_csv(output / f"{target}_comparison.csv").rmse_mean.is_monotonic_decreasing


@pytest.mark.parametrize(
    "case", ["duplicate", "hash", "data", "statistics", "unit", "unfinished", "seedcount"]
)
def test_merge_rejects_inconsistent_or_duplicate_reports(reports, tmp_path, case):
    if case == "duplicate":
        reports = [reports[0], reports[0], reports[2]]
    else:
        report = reports[1]
        meta = read_json(report / "group_report.json")
        if case in {"hash", "statistics", "seedcount"}:
            path = report / "summary.csv"
            frame = pd.read_csv(path)
            if case == "seedcount":
                frame.loc[0, "n_seeds"] = 2
            else:
                frame.loc[0, "rmse_mean"] = 999.0
            frame.to_csv(path, index=False)
            if case in {"statistics", "seedcount"}:
                meta["files"]["summary.csv"] = digest(path)
        elif case == "data":
            meta["data_identity_sha256"] = "different-data"
        elif case == "unit":
            path = report / "runs.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "unit"] = "wrong-unit"
            frame.to_csv(path, index=False)
            meta["files"]["runs.csv"] = digest(path)
        else:
            write_json(report / "verification.json", dict(complete=False))
            meta["files"]["verification.json"] = digest(report / "verification.json")
        write_json(report / "group_report.json", meta)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        groups.merge_reports(reports, output)
    assert not output.exists()


def test_export_seals_only_current_group_report(tmp_path, reports, monkeypatch):
    directory = _batch(tmp_path)
    write_json(tmp_path / "data/data_identity.json", dict(fixture=True))
    monkeypatch.setattr(groups.queues, "summarize", lambda *a, **kw: reports[0])
    output = groups.export_group(tmp_path, "high", directory)
    meta = read_json(output / "group_report.json")
    assert meta["group"] == "high"
    assert meta["files"]["runs.csv"] == digest(output / "runs.csv")


def test_merge_cli_can_fill_original_latex_without_server_paths(reports, tmp_path, monkeypatch):
    source = tmp_path / "original.tex"
    source.write_text(
        "\n".join(
            " & ".join([label, "--", "--", "OLD_GAP", "--", "--", "--", "--", "OLD_TIME"]) + r" \\"
            for label in groups.queues.LABELS.values()
        )
    )
    output = tmp_path / "combined"
    tex = tmp_path / "filled.tex"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neural_groups",
            "merge",
            "--reports",
            *map(str, reports),
            "--output",
            str(output),
            "--source-tex",
            str(source),
            "--output-tex",
            str(tex),
        ],
    )
    groups.main()
    text = tex.read_text()
    assert text.count("OLD_GAP") == 10 and text.count("OLD_TIME") == 10
    assert text.count(r"\shortstack") == 60
