"""Summarize repeated seeds and compare molecular-gap models with tables and plots."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from molgap.config import find_project_root, resolve_path
from molgap.training.evaluate import evaluate_files

METRICS = ("mae", "rmse", "mape", "r2")
METRIC_LABELS = {
    "mae": "MAE (Hartree)",
    "rmse": "RMSE (Hartree)",
    "mape": "MAPE (%)",
    "r2": "R²",
}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def find_metric_files(paths: list[str | Path]) -> list[Path]:
    """Find direct or recursively nested metrics.json files."""
    found: set[Path] = set()
    for item in paths:
        path = Path(item).resolve()
        if path.is_file():
            if path.name != "metrics.json":
                raise ValueError(f"Expected metrics.json, got: {path}")
            found.add(path)
        elif path.is_dir():
            local: set[Path] = set()
            direct = path / "metrics.json"
            if direct.is_file():
                local.add(direct)
            local.update(path.rglob("metrics.json"))
            if not local:
                raise FileNotFoundError(f"No metrics.json found under: {path}")
            found.update(local)
        else:
            raise FileNotFoundError(path)
    return sorted(found)


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Run configuration is required for traceability: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid run configuration: {path}")
    return config


def _extract_metrics(payload: dict[str, Any], target: str) -> dict[str, float | None]:
    source = (
        payload.get("targets", {}).get(target) if isinstance(payload.get("targets"), dict) else None
    )
    if not isinstance(source, dict):
        source = payload
    return {metric: _finite_float(source.get(metric)) for metric in METRICS}


def _recompute_metrics(run_dir: Path, config: dict[str, Any], target: str) -> dict[str, float]:
    root = find_project_root(run_dir)
    truth = resolve_path(root, Path(config["data"]["processed_dir"]) / "test.csv")
    predictions = run_dir / "predictions" / "test_predictions.csv"
    evaluated = evaluate_files(
        truth,
        predictions,
        [target],
        config["data"]["smiles_column"],
        config["data"]["id_column"],
    )
    result = evaluated["targets"][target]
    return {metric: float(result[metric]) for metric in METRICS}


def load_records(metric_paths: list[Path], target: str = "delta_e") -> list[dict[str, Any]]:
    """Load standardized records from Chemprop or Transformer metric schemas."""
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for path in metric_paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Metrics root must be a mapping: {path}")
        run_dir = path.parent
        config = _load_config(run_dir)
        model = str(config["experiment"]["name"])
        seed = int(config["training"]["seed"])
        identity = (model, seed)
        if identity in identities:
            raise ValueError(f"Duplicate metrics for model={model}, seed={seed}.")
        identities.add(identity)
        values = _extract_metrics(payload, target)
        if any(values[metric] is None for metric in METRICS):
            values = _recompute_metrics(run_dir, config, target)
        records.append(
            {
                "model": model,
                "seed": seed,
                "run_mode": config["training"].get("run_mode", "repeated"),
                "target": target,
                "unit": "hartree",
                "metrics_path": str(path),
                **values,
            }
        )
    validate_expected_seeds(records, None)
    return sorted(records, key=lambda item: (item["model"], item["seed"]))


def validate_expected_seeds(
    records: list[dict[str, Any]], expected_seeds: list[int] | None
) -> None:
    """Validate repeated seeds, allowing explicitly declared single runs at seed3407."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)
    problems = []
    for model, items in sorted(grouped.items()):
        modes = {item.get("run_mode", "repeated") for item in items}
        if len(modes) != 1 or not modes <= {"single", "repeated"}:
            raise ValueError(f"Inconsistent run_mode for {model}: {modes}")
        seeds = {int(item["seed"]) for item in items}
        if len(seeds) != len(items):
            raise ValueError(f"Duplicate seeds for {model}")
        if modes == {"single"}:
            expected = {3407}
        elif expected_seeds:
            expected = set(map(int, expected_seeds))
        else:
            continue
        if seeds != expected:
            problems.append(
                f"{model}: missing={sorted(expected - seeds)}, "
                f"unexpected={sorted(seeds - expected)}"
            )
    if problems:
        raise ValueError("Seed sets are incomplete or inconsistent: " + "; ".join(problems))


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate per-model sample mean/std and ranges."""
    if not records:
        raise ValueError("No metrics records were found.")
    validate_expected_seeds(records, None)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model"])].append(record)
    summaries: list[dict[str, Any]] = []
    for model, items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "model": model,
            "runs": len(items),
            "seeds": [int(item["seed"]) for item in sorted(items, key=lambda item: item["seed"])],
            "target": items[0]["target"],
            "unit": items[0]["unit"],
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            row[metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        summaries.append(row)
    return summaries


def _flat_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        row: dict[str, Any] = {
            "model": summary["model"],
            "runs": summary["runs"],
            "seeds": ",".join(map(str, summary["seeds"])),
            "target": summary["target"],
            "unit": summary["unit"],
        }
        for metric in METRICS:
            for statistic in ("mean", "std", "min", "max", "count"):
                row[f"{metric}_{statistic}"] = summary[metric][statistic]
        rows.append(row)
    return rows


def _format_metric(summary: dict[str, Any], metric: str) -> str:
    value = summary[metric]
    if value["std"] is None:
        return f"{value['mean']:.6g} (SD N/A)"
    return f"{value['mean']:.6g} ± {value['std']:.3g}"


def _write_markdown(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Model comparison",
        "",
        "Repeated runs: mean ± sample SD. Single runs: one value; SD is not applicable.",
        "",
        "| Model | Runs | MAE (Hartree) | RMSE (Hartree) | MAPE (%) | R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        formatted = {metric: _format_metric(summary, metric) for metric in METRICS}
        lines.append(
            f"| {summary['model']} | {summary['runs']} | {formatted['mae']} | "
            f"{formatted['rmse']} | {formatted['mape']} | {formatted['r2']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metrics(summaries: list[dict[str, Any]], path: Path, title: str) -> None:
    models = [f"{summary['model']} (n={summary['runs']})" for summary in summaries]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.arange(len(models)))
    for axis, metric in zip(axes.flat, METRICS, strict=True):
        means = [summary[metric]["mean"] for summary in summaries]
        positions = np.arange(len(models))
        axis.bar(positions, means, color=colors, alpha=0.85)
        repeated = [i for i, summary in enumerate(summaries) if summary[metric]["std"] is not None]
        if repeated:
            axis.errorbar(
                positions[repeated],
                [means[i] for i in repeated],
                yerr=[summaries[i][metric]["std"] for i in repeated],
                fmt="none",
                ecolor="black",
                capsize=5,
            )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(positions, models, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_table(summaries: list[dict[str, Any]], path: Path) -> None:
    columns = ["Model", "Runs", *[METRIC_LABELS[metric] for metric in METRICS]]
    cells = []
    for summary in summaries:
        cells.append(
            [
                summary["model"],
                str(summary["runs"]),
                *[_format_metric(summary, metric) for metric in METRICS],
            ]
        )
    height = max(2.6, 1.1 + 0.55 * len(cells))
    figure, axis = plt.subplots(figsize=(14, height))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    title: str = "QM9 delta_e model comparison",
) -> dict[str, Path]:
    """Write raw records, aggregate tables, JSON, Markdown, and figures."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_records(records)
    raw_path = output_dir / "runs.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    chart_path = output_dir / "model_comparison.png"
    table_path = output_dir / "model_comparison_table.png"
    pd.DataFrame(records).to_csv(raw_path, index=False)
    pd.DataFrame(_flat_summary_rows(summaries)).to_csv(summary_csv, index=False)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": list(METRICS),
        "mape_unit": "percent",
        "std_definition": "sample standard deviation (ddof=1; null/not applicable for one run)",
        "records": records,
        "models": summaries,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_markdown(summaries, summary_md)
    _plot_metrics(summaries, chart_path, title)
    _plot_table(summaries, table_path)
    return {
        "runs_csv": raw_path,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "summary_markdown": summary_md,
        "comparison_chart": chart_path,
        "comparison_table": table_path,
    }


def summarize_paths(
    paths: list[str | Path],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | None = None,
    target: str = "delta_e",
    title: str = "QM9 delta_e model comparison",
) -> dict[str, Path]:
    metric_paths = find_metric_files(paths)
    records = load_records(metric_paths, target)
    if expected_seeds:
        validate_expected_seeds(records, expected_seeds)
    outputs = write_summary(records, output_dir, title=title)
    print(
        f"Loaded {len(records)} runs across {len(set(item['model'] for item in records))} models."
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", help="Model directories, run directories, or metrics.json"
    )
    parser.add_argument("--output-dir", required=True, help="Summary artifact directory")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Required repeated-run seeds; run_mode=single models require only seed3407",
    )
    parser.add_argument("--target", default="delta_e")
    parser.add_argument("--title", default="QM9 delta_e model comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize_paths(
        args.paths,
        args.output_dir,
        expected_seeds=args.seeds,
        target=args.target,
        title=args.title,
    )


if __name__ == "__main__":
    main()
