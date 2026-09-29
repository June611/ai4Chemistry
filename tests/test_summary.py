import json
from pathlib import Path

import pytest
import yaml

from molgap.experiments.summary import (
    find_metric_files,
    load_records,
    summarize_records,
    validate_expected_seeds,
    write_summary,
)


def _write_run(root: Path, model: str, seed: int, metrics: dict) -> Path:
    run_dir = root / "outputs" / model / f"seed{seed}"
    run_dir.mkdir(parents=True)
    config = {
        "experiment": {"name": model},
        "training": {"seed": seed},
        "data": {
            "processed_dir": f"data/processed/tiny/random/seed{seed}",
            "smiles_column": "smiles",
            "id_column": "sample_id",
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def test_summary_reads_both_metric_schemas_and_writes_figures(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='summary-test'\nversion='0.0.0'\n", encoding="utf-8"
    )
    nested = lambda value: {  # noqa: E731
        "targets": {"delta_e": {"mae": value, "rmse": value + 1, "mape": value + 2, "r2": 0.9}}
    }
    flat = lambda value: {  # noqa: E731
        "target": "delta_e",
        "mae": value,
        "rmse": value + 1,
        "mape": value + 2,
        "r2": 0.8,
    }
    _write_run(root, "graph", 1, nested(1.0))
    _write_run(root, "graph", 2, nested(3.0))
    _write_run(root, "sequence", 1, flat(2.0))
    _write_run(root, "sequence", 2, flat(4.0))

    paths = find_metric_files([root / "outputs"])
    records = load_records(paths)
    validate_expected_seeds(records, [1, 2])
    summaries = summarize_records(records)
    graph = next(item for item in summaries if item["model"] == "graph")
    assert graph["mae"]["mean"] == pytest.approx(2.0)
    assert graph["mae"]["std"] == pytest.approx(2**0.5)

    artifacts = write_summary(records, root / "summary")
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    assert "graph" in artifacts["summary_markdown"].read_text(encoding="utf-8")


def test_summary_rejects_incomplete_seed_sets() -> None:
    records = [
        {"model": "a", "seed": 1},
        {"model": "a", "seed": 2},
        {"model": "b", "seed": 1},
    ]
    with pytest.raises(ValueError, match="incomplete"):
        validate_expected_seeds(records, [1, 2])
