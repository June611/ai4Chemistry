"""Attach original QM9 XYZ geometry to the project's immutable molecular split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from molgap.config import find_project_root
from molgap.qm9_models.geometry import (
    KNOWN_QM9_ATOM_SMILES_CONFLICTS,
    read_sdf,
    read_xyz,
    validate_identity,
    xyz_path,
)
from molgap.transformer.dataset import _read_and_validate_splits
from molgap.utils.files import sha256


def prepare(config_path: Path, geometry_source: Path, output: Path, use_sdf: bool = False) -> dict:
    """Prepare geometry data from XYZ directory or SDF file."""
    root = find_project_root(config_path.resolve())
    config = yaml.safe_load(config_path.read_text())
    config = copy.deepcopy(config)
    config["data"].setdefault("chemprop_reference_config", "configs/baseline.yaml")
    frames, identity = _read_and_validate_splits(config, root)
    processed = root / config["data"]["processed_dir"]
    lineage = pd.read_csv(processed / "dataset.csv").set_index("sample_id", verify_integrity=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite geometry artifact: {output}")

    # Load geometries based on source type
    if use_sdf:
        print(f"Loading SDF file: {geometry_source}")
        geometries = read_sdf(geometry_source)
        source_type = "QM9 SDF file"
        source_path = str(geometry_source.resolve())
    else:
        geometries = None
        source_type = "original QM9 extended XYZ"
        source_path = str(geometry_source.resolve())

    ids, elements, coordinates, hashes = [], [], [], []
    offsets = [0]

    for frame in frames.values():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="QM9 geometry"):
            source_ids = str(lineage.loc[row.sample_id, "source_id"]).split("|")
            failures = []
            for source_id in source_ids:
                try:
                    if use_sdf:
                        if source_id not in geometries:
                            raise ValueError(f"Molecule {source_id} not found in SDF")
                        record = geometries[source_id]
                        # SDF has no QM9 targets; target validation is supplied by XYZ only.
                        record.gap_hartree = row.delta_e
                        file_hash = sha256(geometry_source) if len(hashes) < 1 else None
                    else:
                        path = xyz_path(geometry_source, source_id)
                        record = read_xyz(path.read_text())
                        file_hash = sha256(path) if len(hashes) < 5 else None
                    validate_identity(record, source_id, row.smiles, row.delta_e)
                    break
                except (FileNotFoundError, ValueError) as error:
                    failures.append(f"{source_id}: {error}")
            else:
                raise ValueError(
                    f"No source geometry matches {row.sample_id}: {'; '.join(failures)}"
                )
            ids.append(row.sample_id)
            elements.append(record.atomic_numbers)
            coordinates.append(record.positions)
            offsets.append(offsets[-1] + len(record.atomic_numbers))
            if file_hash is not None:
                hashes.append(
                    {"sample_id": row.sample_id, "source_id": source_id, "sha256": file_hash}
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    # Numeric/Unicode arrays only: loading requires no pickle or external object execution.
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(ids),
            offsets=np.asarray(offsets),
            atomic_numbers=np.concatenate(elements),
            positions=np.concatenate(coordinates),
        )
    report = {
        "source": source_type,
        "geometry_path": source_path,
        "coordinate_unit": "angstrom",
        "target_unit": "hartree",
        "rows": len(ids),
        "split_identity": identity,
        "geometry_sha256": sha256(output),
        "lineage_sha256": sha256(processed / "dataset.csv"),
        "known_official_record_conflicts": sorted(KNOWN_QM9_ATOM_SMILES_CONFLICTS),
        "sample_file_hashes": hashes[:5],
    }  # Only store first 5 for brevity
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/spe_transformer.yaml"))
    parser.add_argument("--xyz-dir", type=Path, help="Directory containing XYZ files")
    parser.add_argument("--sdf-file", type=Path, help="SDF file path (alternative to xyz-dir)")
    parser.add_argument("--output", type=Path, default=Path("data/geometry/qm9_full_seed3407.npz"))
    args = parser.parse_args()

    if args.sdf_file:
        prepare(args.config, args.sdf_file, args.output, use_sdf=True)
    elif args.xyz_dir:
        prepare(args.config, args.xyz_dir, args.output, use_sdf=False)
    else:
        parser.error("Either --xyz-dir or --sdf-file must be provided")


if __name__ == "__main__":
    main()
