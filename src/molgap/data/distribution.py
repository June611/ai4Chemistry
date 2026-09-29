"""Audit and visualize target distributions across prepared data splits."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from molgap.data.audit import sha256_file

SPLIT_NAMES = ("train", "val", "test")
COLORS = {"train": "#0072B2", "val": "#D55E00", "test": "#009E73"}


def _describe(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(values.max()),
    }


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate the two-sample Kolmogorov-Smirnov distance without p-value heuristics."""
    left = np.sort(left)
    right = np.sort(right)
    points = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, points, side="right") / left.size
    right_cdf = np.searchsorted(right, points, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    pooled_variance = (left.var(ddof=1) + right.var(ddof=1)) / 2
    if pooled_variance == 0:
        return 0.0 if left.mean() == right.mean() else float("inf")
    return float(abs(left.mean() - right.mean()) / np.sqrt(pooled_variance))


def _jensen_shannon_distance(left: np.ndarray, right: np.ndarray, bins: np.ndarray) -> float:
    left_hist = np.histogram(left, bins=bins, density=False)[0].astype(float)
    right_hist = np.histogram(right, bins=bins, density=False)[0].astype(float)
    left_prob = left_hist / left_hist.sum()
    right_prob = right_hist / right_hist.sum()
    midpoint = (left_prob + right_prob) / 2

    def divergence(distribution: np.ndarray) -> float:
        valid = distribution > 0
        return float(np.sum(distribution[valid] * np.log(distribution[valid] / midpoint[valid])))

    return float(np.sqrt((divergence(left_prob) + divergence(right_prob)) / 2))


def _plot_density(
    values: dict[str, np.ndarray],
    target: str,
    dataset_name: str,
    stage: str,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11, 7))
    for split_name in SPLIT_NAMES:
        sns.kdeplot(
            x=values[split_name],
            ax=axis,
            color=COLORS[split_name],
            linewidth=2.2,
            label=f"{split_name} (n={len(values[split_name]):,})",
            common_norm=False,
            warn_singular=True,
        )
    axis.set_title(f"{dataset_name}: {target} density ({stage})")
    axis.set_xlabel(f"{target} (Hartree)")
    axis.set_ylabel("Probability density")
    axis.legend(frameon=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def audit_split_distribution(
    processed_dir: Path,
    output_dir: Path,
    dataset_name: str,
    stage: str,
    target: str = "delta_e",
) -> dict[str, Any]:
    """Write distribution statistics and a three-way KDE plot for prepared CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(processed_dir / f"{name}.csv") for name in SPLIT_NAMES}
    values: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        if target not in frame:
            raise ValueError(f"{processed_dir / f'{name}.csv'} has no target column '{target}'.")
        array = frame[target].to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name}.{target} contains missing or non-finite values.")
        values[name] = array

    all_values = np.concatenate(list(values.values()))
    histogram_bins = np.histogram_bin_edges(all_values, bins="fd")
    if len(histogram_bins) < 3:
        histogram_bins = np.linspace(all_values.min(), all_values.max(), 11)
    pairwise: dict[str, dict[str, float]] = {}
    for left_name, right_name in combinations(SPLIT_NAMES, 2):
        key = f"{left_name}_vs_{right_name}"
        pairwise[key] = {
            "ks_distance": _ks_distance(values[left_name], values[right_name]),
            "standardized_mean_difference": _standardized_mean_difference(
                values[left_name], values[right_name]
            ),
            "jensen_shannon_distance": _jensen_shannon_distance(
                values[left_name], values[right_name], histogram_bins
            ),
        }
    maximums = {
        metric: max(comparison[metric] for comparison in pairwise.values())
        for metric in ("ks_distance", "standardized_mean_difference", "jensen_shannon_distance")
    }
    report = {
        "dataset": dataset_name,
        "stage": stage,
        "target": target,
        "unit": "hartree",
        "processed_dir": str(processed_dir),
        "files": {
            name: {
                "path": str(processed_dir / f"{name}.csv"),
                "sha256": sha256_file(processed_dir / f"{name}.csv"),
            }
            for name in SPLIT_NAMES
        },
        "descriptive_statistics": {name: _describe(array) for name, array in values.items()},
        "pairwise_distances": pairwise,
        "maximum_pairwise_distances": maximums,
        "diagnostic_thresholds": {
            "ks_distance": 0.03,
            "standardized_mean_difference": 0.1,
        },
        "approximately_aligned": maximums["ks_distance"] <= 0.03
        and maximums["standardized_mean_difference"] <= 0.1,
    }
    with (output_dir / "distribution_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _plot_density(
        values,
        target,
        dataset_name,
        stage,
        output_dir / f"{target}_density.png",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--target", default="delta_e")
    args = parser.parse_args()
    report = audit_split_distribution(
        args.processed_dir,
        args.output_dir,
        args.dataset_name,
        args.stage,
        args.target,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
