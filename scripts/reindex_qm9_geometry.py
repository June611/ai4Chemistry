#!/usr/bin/env python3
"""Reorder an existing QM9 geometry artifact for another fixed split seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.utils.files import sha256


def reindex_geometry(source: Path, processed_dir: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite geometry artifact: {output}")

    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    source_ids = arrays["sample_ids"].astype(str).tolist()
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source geometry contains duplicate sample IDs")
    source_index = {sample_id: index for index, sample_id in enumerate(source_ids)}

    target_ids: list[str] = []
    split_rows: dict[str, int] = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(processed_dir / f"{split}.csv", usecols=["sample_id"])
        ids = frame["sample_id"].astype(str).tolist()
        target_ids.extend(ids)
        split_rows[split] = len(ids)
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target split files contain duplicate sample IDs")
    if set(target_ids) != set(source_ids):
        missing = sorted(set(target_ids) - set(source_ids))[:3]
        extra = sorted(set(source_ids) - set(target_ids))[:3]
        raise ValueError(f"Source/target sample universes differ: missing={missing}, extra={extra}")

    permutation = np.asarray([source_index[sample_id] for sample_id in target_ids], dtype=np.int64)
    source_offsets = arrays["offsets"]
    counts = np.diff(source_offsets)[permutation]
    target_offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    target_atomic_numbers = np.empty(target_offsets[-1], dtype=arrays["atomic_numbers"].dtype)
    target_positions = np.empty(
        (target_offsets[-1], arrays["positions"].shape[1]), dtype=arrays["positions"].dtype
    )
    for target_index, source_index_value in enumerate(permutation):
        source_start, source_end = source_offsets[source_index_value : source_index_value + 2]
        target_start, target_end = target_offsets[target_index : target_index + 2]
        target_atomic_numbers[target_start:target_end] = arrays["atomic_numbers"][
            source_start:source_end
        ]
        target_positions[target_start:target_end] = arrays["positions"][source_start:source_end]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(target_ids),
            offsets=target_offsets,
            atomic_numbers=target_atomic_numbers,
            positions=target_positions,
        )
    report = {
        "source": "reindexed official QM9 geometry artifact",
        "source_geometry": str(source),
        "source_geometry_sha256": sha256(source),
        "processed_dir": str(processed_dir),
        "split_manifest_sha256": sha256(processed_dir / "split_manifest.csv"),
        "coordinate_unit": "angstrom",
        "target_unit": "hartree",
        "rows": len(target_ids),
        "split_rows": split_rows,
        "geometry_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reindex_geometry(args.source, args.processed_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
