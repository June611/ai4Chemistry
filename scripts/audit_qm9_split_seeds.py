#!/usr/bin/env python3
"""Audit membership, labels, lineage, and geometry for all prepared QM9 split seeds."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def molecule_equal(
    reference: dict[str, np.ndarray],
    reference_index: dict[str, int],
    candidate: dict[str, np.ndarray],
) -> bool:
    for candidate_index, sample_id in enumerate(candidate["sample_ids"].astype(str)):
        source_index = reference_index[sample_id]
        source_start, source_end = reference["offsets"][source_index : source_index + 2]
        target_start, target_end = candidate["offsets"][candidate_index : candidate_index + 2]
        if not np.array_equal(
            reference["atomic_numbers"][source_start:source_end],
            candidate["atomic_numbers"][target_start:target_end],
        ) or not np.array_equal(
            reference["positions"][source_start:source_end],
            candidate["positions"][target_start:target_end],
        ):
            return False
    return True


def audit(root: Path, seeds: list[int]) -> dict:
    base = root / "data/processed/qm9_full/random"
    manifests: dict[int, pd.DataFrame] = {}
    datasets: dict[int, pd.DataFrame] = {}
    result: dict = {"status": "pass", "seeds": {}}

    reference_geometry = load_geometry(root / f"data/geometry/qm9_full_seed{seeds[0]}.npz")
    reference_geometry_index = {
        sample_id: index
        for index, sample_id in enumerate(reference_geometry["sample_ids"].astype(str))
    }

    for seed in seeds:
        directory = base / f"seed{seed}"
        manifest = pd.read_csv(directory / "split_manifest.csv")
        dataset = pd.read_csv(directory / "dataset.csv")
        manifests[seed] = manifest.set_index("sample_id")
        datasets[seed] = dataset.set_index("sample_id").sort_index()
        if not manifests[seed].index.is_unique or not datasets[seed].index.is_unique:
            raise ValueError(f"Duplicate sample IDs for split seed {seed}")
        split_frames = {name: pd.read_csv(directory / f"{name}.csv") for name in SPLITS}
        split_sets = {
            name: set(frame.sample_id.astype(str)) for name, frame in split_frames.items()
        }
        ordered_ids = [sample_id for name in SPLITS for sample_id in split_frames[name].sample_id]
        expected_ids = [
            sample_id
            for name in SPLITS
            for sample_id in manifest.loc[manifest.split == name]
            .sort_values("split_position", kind="stable")
            .sample_id
        ]
        geometry_path = root / f"data/geometry/qm9_full_seed{seed}.npz"
        geometry = load_geometry(geometry_path)
        metadata = json.loads(geometry_path.with_suffix(".json").read_text())
        report = json.loads((directory / "preprocess_report.json").read_text())
        pairwise_disjoint = all(
            not split_sets[left] & split_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        )
        checks = {
            "preprocess_complete": report["status"] == "complete",
            "manifest_seed": set(manifest.seed.astype(int)) == {seed},
            "manifest_strategy": set(manifest.split_strategy.astype(str))
            == {"quantile_stratified"},
            "manifest_bins": set(manifest.stratify_bins_requested.astype(int)) == {200},
            "split_rows": {name: len(frame) for name, frame in split_frames.items()}
            == {"train": 107038, "val": 13380, "test": 13380},
            "pairwise_disjoint": pairwise_disjoint,
            "union_matches_dataset": set.union(*split_sets.values())
            == set(datasets[seed].index.astype(str)),
            "split_order_matches_manifest": ordered_ids == expected_ids,
            "labels_finite": all(
                np.isfinite(frame[["homo", "lumo", "delta_e"]].to_numpy()).all()
                for frame in split_frames.values()
            ),
            "geometry_hash_matches_metadata": file_hash(geometry_path)
            == metadata["geometry_sha256"],
            "geometry_order_matches_splits": geometry["sample_ids"].astype(str).tolist()
            == ordered_ids,
            "geometry_content_matches_reference": molecule_equal(
                reference_geometry, reference_geometry_index, geometry
            ),
        }
        result["seeds"][str(seed)] = {
            "checks": checks,
            "manifest_sha256": file_hash(directory / "split_manifest.csv"),
            "geometry_sha256": file_hash(geometry_path),
            "delta_e": {
                name: {
                    "mean": float(frame.delta_e.mean()),
                    "std": float(frame.delta_e.std(ddof=0)),
                }
                for name, frame in split_frames.items()
            },
        }
        if not all(value for value in checks.values() if isinstance(value, bool)):
            result["status"] = "fail"

    reference_seed = seeds[0]
    result["cross_seed"] = {
        "same_sample_universe": all(
            set(datasets[reference_seed].index) == set(datasets[seed].index) for seed in seeds[1:]
        ),
        "same_preprocessed_dataset": all(
            datasets[reference_seed].equals(datasets[seed]) for seed in seeds[1:]
        ),
        "assignment_changes_vs_reference": {
            str(seed): int(
                (
                    manifests[reference_seed].loc[datasets[reference_seed].index, "split"]
                    != manifests[seed].loc[datasets[reference_seed].index, "split"]
                ).sum()
            )
            for seed in seeds[1:]
        },
    }
    if (
        not result["cross_seed"]["same_sample_universe"]
        or not result["cross_seed"]["same_preprocessed_dataset"]
    ):
        result["status"] = "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 42, 2026])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.seeds)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
