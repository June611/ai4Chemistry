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
SUPPORTED_METRICS = ("mae", "mse", "rmse", "mape", "r2")
METRIC_LABELS = {
    "mae": "MAE (Hartree)",
    "mse": "MSE (Hartree²)",
    "rmse": "RMSE (Hartree)",
    "mape": "MAPE (%)",
    "r2": "R²",
}
MODEL_LABELS = {
    "lightgbm_rdkit29": "LightGBM: RDKit29",
    "linear_regression_rdkit29": "Linear regression: RDKit29",
    "ridge_rdkit29": "Ridge: RDKit29",
    "extra_trees_rdkit29": "ExtraTrees: RDKit29",
    "hist_gradient_boosting_rdkit29": "HistGradientBoosting: RDKit29",
    "chemprop_single_gap": "Chemprop: gap",
    "chemprop_multitask": "Chemprop: multitask",
    "chemprop_consistency": "Chemprop: consistency",
    "spe_transformer": "SPE: gap",
    "spe_multitask": "SPE: multitask",
    "spe_multitask_affinity": "SPE: affinity",
    "spe_multitask_consistency": "SPE: consistency",
    "spe_multitask_affinity_consistency": "SPE: affinity + consistency",
}


def _validate_metrics(metrics) -> None:
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(SUPPORTED_METRICS):
        raise ValueError("Choose distinct supported metrics: " + ", ".join(SUPPORTED_METRICS))


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


def _extract_metrics(
    payload: dict[str, Any], target: str, metrics=METRICS
) -> dict[str, float | None]:
    source = (
        payload.get("targets", {}).get(target) if isinstance(payload.get("targets"), dict) else None
    )
    if not isinstance(source, dict):
        source = payload
    return {metric: _finite_float(source.get(metric)) for metric in metrics}


def _recompute_metrics(
    run_dir: Path, config: dict[str, Any], target: str, metrics=METRICS
) -> dict[str, float]:
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
    return {metric: float(result[metric]) for metric in metrics}


def load_records(
    metric_paths: list[Path], target: str = "delta_e", *, metrics=METRICS
) -> list[dict[str, Any]]:
    """Load standardized records from Chemprop or Transformer metric schemas."""
    records: list[dict[str, Any]] = []
    _validate_metrics(metrics)
    identities: set[tuple[str, int]] = set()
    for path in metric_paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Metrics root must be a mapping: {path}")
        run_dir = path.parent
        config = _load_config(run_dir)
        if config["experiment"].get("protocol_id") == "qm9_modular_fresh200_v1":
            if payload.get("diagnostic"):
                raise ValueError(f"Diagnostic modular run cannot enter a formal comparison: {path}")
            if target not in config["data"]["targets"]:
                continue
        model = str(config["experiment"].get("model_id", config["experiment"]["name"]))
        seed = int(config["training"]["seed"])
        identity = (model, seed)
        if identity in identities:
            raise ValueError(f"Duplicate metrics for model={model}, seed={seed}.")
        identities.add(identity)
        values = _extract_metrics(payload, target, metrics)
        if any(values[metric] is None for metric in metrics):
            values = _recompute_metrics(run_dir, config, target, metrics)
        records.append(
            {
                "model": model,
                "seed": seed,
                "split_seed": int(config["data"].get("split", {}).get("seed", seed)),
                "run_mode": config["training"].get("run_mode", "repeated"),
                "protocol_id": config["experiment"].get("protocol_id"),
                "target": target,
                "unit": config["data"].get("units", {}).get(target, "hartree"),
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
    protocols = {record.get("protocol_id") for record in records}
    if len(protocols) > 1:
        raise ValueError(f"Cannot mix different experiment protocols: {protocols}")
    for record in records:
        grouped[record["model"]].append(record)
    problems = []
    for model, items in sorted(grouped.items()):
        modes = {item.get("run_mode", "repeated") for item in items}
        if len(modes) != 1 or not modes <= {"single", "repeated", "paired_split_seed"}:
            raise ValueError(f"Inconsistent run_mode for {model}: {modes}")
        seeds = {int(item["seed"]) for item in items}
        if len(seeds) != len(items):
            raise ValueError(f"Duplicate seeds for {model}")
        if modes == {"single"}:
            expected = {3407}
        elif modes == {"paired_split_seed"}:
            mismatched = [
                item for item in items if int(item["seed"]) != int(item.get("split_seed", -1))
            ]
            if mismatched:
                raise ValueError(f"Paired split/training seeds differ for {model}.")
            if not expected_seeds:
                continue
            expected = set(map(int, expected_seeds))
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


def summarize_records(records: list[dict[str, Any]], *, metrics=METRICS) -> list[dict[str, Any]]:
    """Calculate per-model sample mean/std and ranges."""
    if not records:
        raise ValueError("No metrics records were found.")
    _validate_metrics(metrics)
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
            "run_mode": items[0].get("run_mode", "repeated"),
            "target": items[0]["target"],
            "unit": items[0]["unit"],
        }
        for metric in metrics:
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


def _flat_summary_rows(summaries: list[dict[str, Any]], metrics=METRICS) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        row: dict[str, Any] = {
            "model": summary["model"],
            "runs": summary["runs"],
            "seeds": ",".join(map(str, summary["seeds"])),
            "run_mode": summary["run_mode"],
            "target": summary["target"],
            "unit": summary["unit"],
        }
        for metric in metrics:
            for statistic in ("mean", "std", "min", "max", "count"):
                row[f"{metric}_{statistic}"] = summary[metric][statistic]
        rows.append(row)
    return rows


def _format_metric(summary: dict[str, Any], metric: str) -> str:
    value = summary[metric]
    if value["std"] is None:
        return f"{value['mean']:.6g} (SD N/A)"
    return f"{value['mean']:.6g} ± {value['std']:.3g}"


def _write_markdown(
    summaries: list[dict[str, Any]], path: Path, metrics=METRICS, labels=METRIC_LABELS
) -> None:
    lines = [
        "# Model comparison",
        "",
        "Repeated runs: mean ± sample SD. Single runs: one value; SD is not applicable.",
        "Paired split-seed runs change both data partition and training randomness; "
        "their SD is not initialization variance on a fixed split.",
        "",
        "| Model | Runs | " + " | ".join(labels[m] for m in metrics) + " |",
        "|---|---:|" + "---:|" * len(metrics),
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['runs']} | "
            + " | ".join(_format_metric(summary, metric) for metric in metrics)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(figure, path: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix("." + suffix), dpi=200, bbox_inches="tight")


def _plot_metrics(
    summaries: list[dict[str, Any]], path: Path, title: str, metrics=METRICS, labels=METRIC_LABELS
) -> None:
    models = [f"{MODEL_LABELS.get(s['model'], s['model'])} (n={s['runs']})" for s in summaries]
    panels = math.ceil(len(metrics) / 2)
    height = max(3.5, 1.4 + 0.4 * len(models))
    figure, axes = plt.subplots(
        panels, 2, figsize=(16, height * panels), constrained_layout=True, squeeze=False
    )
    colors = [
        "#0072B2"
        if s["model"].startswith("chemprop")
        else "#E69F00"
        if s["model"].startswith("spe_")
        else "#777777"
        for s in summaries
    ]
    for axis, metric in zip(list(axes.flat)[: len(metrics)], metrics, strict=True):
        means = [summary[metric]["mean"] for summary in summaries]
        positions = np.arange(len(models))
        axis.barh(positions, means, color=colors, alpha=0.85)
        repeated = [i for i, summary in enumerate(summaries) if summary[metric]["std"] is not None]
        if repeated:
            axis.errorbar(
                [means[i] for i in repeated],
                positions[repeated],
                xerr=[summaries[i][metric]["std"] for i in repeated],
                fmt="none",
                ecolor="black",
                capsize=5,
            )
        axis.set_xlabel(labels[metric])
        axis.set_yticks(positions, models)
        axis.invert_yaxis()
        lower = min([0.0, *[s[metric]["mean"] - (s[metric]["std"] or 0.0) for s in summaries]])
        axis.set_xlim(left=lower * 1.1)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(metrics) :]:
        axis.set_axis_off()
    figure.suptitle(title + "\nMean ± sample SD (ddof=1)", fontsize=15)
    _save_figure(figure, path)
    plt.close(figure)


def _plot_table(
    summaries: list[dict[str, Any]], path: Path, metrics=METRICS, labels=METRIC_LABELS
) -> None:
    columns = ["Model", "Runs", *[labels[metric] for metric in metrics]]
    cells = []
    for summary in summaries:
        cells.append(
            [
                summary["model"],
                str(summary["runs"]),
                *[
                    f"{summary[m]['mean']:.4g} ± {summary[m]['std']:.2g}"
                    if summary[m]["std"] is not None
                    else f"{summary[m]['mean']:.4g} (SD N/A)"
                    for m in metrics
                ],
            ]
        )
    height = max(2.6, 1.1 + 0.55 * len(cells))
    figure, axis = plt.subplots(figsize=(16, height))
    axis.axis("off")
    widths = [0.26, 0.05, *[0.69 / len(metrics)] * len(metrics)]
    table = axis.table(
        cellText=cells, colLabels=columns, cellLoc="center", loc="center", colWidths=widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    _save_figure(figure, path)
    plt.close(figure)


def write_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    title: str = "QM9 delta_e model comparison",
    metrics=METRICS,
) -> dict[str, Path]:
    """Write raw records, aggregate tables, JSON, Markdown, and figures."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_records(records, metrics=metrics)
    targets = {r.get("target", "delta_e") for r in records}
    if len(targets) > 1:
        raise ValueError(f"Cannot merge different prediction targets: {targets}")
    if title == "QM9 delta_e model comparison" and targets != {"delta_e"}:
        title = f"QM9 {next(iter(targets))} model comparison"
    units = {r.get("unit", "hartree") for r in records}
    if len(units) > 1:
        raise ValueError(f"Cannot compare raw error metrics in different units: {units}")
    unit = next(iter(units), "hartree")
    labels = dict(METRIC_LABELS)
    if unit != "hartree":
        labels.update(mae=f"MAE ({unit})", rmse=f"RMSE ({unit})", mse=f"MSE (({unit})²)")
    raw_path = output_dir / "runs.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    chart_path = output_dir / "model_comparison.png"
    table_path = output_dir / "model_comparison_table.png"
    pd.DataFrame(records).to_csv(raw_path, index=False)
    pd.DataFrame(_flat_summary_rows(summaries, metrics)).to_csv(summary_csv, index=False)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": list(metrics),
        "mape_unit": "percent",
        "std_definition": "sample standard deviation (ddof=1; null/not applicable for one run)",
        "records": records,
        "models": summaries,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_markdown(summaries, summary_md, metrics, labels)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        _plot_metrics(summaries, chart_path, title, metrics, labels)
        _plot_table(summaries, table_path, metrics, labels)
    return {
        "runs_csv": raw_path,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "summary_markdown": summary_md,
        "comparison_chart": chart_path,
        "comparison_table": table_path,
        "comparison_chart_pdf": chart_path.with_suffix(".pdf"),
        "comparison_chart_svg": chart_path.with_suffix(".svg"),
        "comparison_table_pdf": table_path.with_suffix(".pdf"),
        "comparison_table_svg": table_path.with_suffix(".svg"),
    }


def summarize_paths(
    paths: list[str | Path],
    output_dir: str | Path,
    *,
    expected_seeds: list[int] | None = None,
    target: str = "delta_e",
    title: str = "QM9 delta_e model comparison",
    metrics=METRICS,
) -> dict[str, Path]:
    metric_paths = find_metric_files(paths)
    records = load_records(metric_paths, target, metrics=metrics)
    if expected_seeds:
        validate_expected_seeds(records, expected_seeds)
    outputs = write_summary(records, output_dir, title=title, metrics=metrics)
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
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        default=METRICS,
        help="Metrics to summarize; add mse for Hartree-squared error",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize_paths(
        args.paths,
        args.output_dir,
        expected_seeds=args.seeds,
        target=args.target,
        title=args.title,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
