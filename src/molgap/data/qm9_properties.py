"""Extend the frozen QM9 molecular identities with original-unit scalar labels."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROPERTY_UNITS = {
    "homo": "hartree",
    "lumo": "hartree",
    "delta_e": "hartree",
    "alpha": "bohr^3",
    "mu": "debye",
    "r2": "bohr^2",
    "zpve": "hartree",
}
NEW_TARGETS = ("homo", "lumo", "alpha", "mu", "r2", "zpve")
EXTRA_LABELS = ("alpha", "mu", "r2", "zpve")
SEEDS = (42, 2026, 3407)
SINGLE_PROPERTY_PROTOCOL = "paired_split_qm9_rdkit29_single_property_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def attach_original_labels(raw: pd.DataFrame, canonical: pd.DataFrame):
    """Reuse exact retained IDs and source_id groups, including their mean policy."""
    raw = raw.copy()
    required = ["mol_id", "homo", "lumo", "delta_E", *EXTRA_LABELS]
    if not set(required).issubset(raw) or raw.mol_id.isna().any() or raw.mol_id.duplicated().any():
        raise ValueError("Raw QM9 properties require complete, unique molecular identities.")
    if not np.isfinite(raw[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Raw QM9 properties must all be finite.")
    if canonical.sample_id.duplicated().any() or canonical.smiles.duplicated().any():
        raise ValueError("Retained canonical identities must be unique.")
    members = canonical[["sample_id", "source_id"]].copy()
    members["mol_id"] = members.source_id.str.split("|")
    members = members.explode("mol_id").drop(columns="source_id")
    if members.mol_id.duplicated().any() or set(members.mol_id) != set(raw.mol_id):
        raise ValueError("The frozen source lineage must cover every raw identity exactly once.")
    joined = members.merge(raw[required], on="mol_id", how="left", validate="many_to_one")
    grouped = joined.groupby("sample_id", sort=False)
    means = grouped[required[1:]].mean().loc[canonical.sample_id]
    for old, source in [("homo", "homo"), ("lumo", "lumo"), ("delta_e", "delta_E")]:
        np.testing.assert_allclose(
            means[source],
            canonical[old],
            rtol=0,
            atol=1e-12,
            err_msg=f"Frozen {old} disagrees with raw-label lineage means",
        )
    result = canonical.copy()
    for target in EXTRA_LABELS:
        result[target] = means[target].to_numpy()
    duplicate_ids = grouped.size().loc[lambda n: n > 1].index
    audit = joined[joined.sample_id.isin(duplicate_ids)].copy()
    for target in EXTRA_LABELS:
        group_values = audit.groupby("sample_id")[target]
        audit[f"{target}_min"] = group_values.transform("min")
        audit[f"{target}_max"] = group_values.transform("max")
        audit[f"{target}_range"] = audit[f"{target}_max"] - audit[f"{target}_min"]
        audit[f"{target}_mean"] = group_values.transform("mean")
    return result, audit


def prepare_property_data(root: Path, destination: Path, *, seeds=SEEDS) -> dict:
    raw_path = root / "data/raw/qm9_chn.csv"
    source_base = root / "data/processed/qm9_full/random"
    first = source_base / f"seed{seeds[0]}"
    canonical = pd.read_csv(first / "dataset.csv", float_precision="round_trip")
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    expanded, duplicates = attach_original_labels(raw, canonical)
    if destination.exists():
        raise FileExistsError(f"Retain existing extended data: {destination}")
    destination.mkdir(parents=True)
    expanded.to_csv(destination / "dataset.csv", index=False)
    duplicates.to_csv(destination / "duplicate_label_audit.csv", index=False)
    source_files, output_files = {}, {}
    columns = ["sample_id", "smiles", "homo", "lumo", "delta_e", *EXTRA_LABELS]
    index = expanded.set_index("sample_id", drop=False)
    for seed in seeds:
        source = source_base / f"seed{seed}"
        current = pd.read_csv(source / "dataset.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(current, canonical)
        dest = destination / "random" / f"seed{seed}"
        dest.mkdir(parents=True)
        sizes = {}
        sets = []
        manifest = pd.read_csv(source / "split_manifest.csv")
        for split in ("train", "val", "test"):
            original_path = source / f"{split}.csv"
            original = pd.read_csv(original_path, float_precision="round_trip")
            new = index.loc[original.sample_id, columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(new[original.columns], original)
            expected_ids = manifest.loc[manifest.split.eq(split)].sort_values("split_position")
            if expected_ids.sample_id.tolist() != original.sample_id.tolist():
                raise ValueError("Original split order differs from its fixed manifest")
            new_path = dest / f"{split}.csv"
            new.to_csv(new_path, index=False)
            saved = pd.read_csv(new_path, float_precision="round_trip")
            pd.testing.assert_frame_equal(saved, new)
            source_files[str(original_path.relative_to(root))] = digest(original_path)
            output_files[str(new_path.relative_to(destination))] = digest(new_path)
            sizes[split] = len(saved)
            sets.append(set(saved.sample_id))
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
            raise ValueError("Split identities overlap")
        if len(set.union(*sets)) != len(canonical):
            raise ValueError("Splits must cover the retained dataset")
        shutil.copy2(source / "split_manifest.csv", dest / "split_manifest.csv")
        source_files[str((source / "split_manifest.csv").relative_to(root))] = digest(
            source / "split_manifest.csv"
        )
    audit = {
        "protocol_id": SINGLE_PROPERTY_PROTOCOL,
        "raw_sha256": digest(raw_path),
        "rows_raw": len(raw),
        "rows_retained": len(expanded),
        "targets": PROPERTY_UNITS,
        "target_scaling": "none",
        "split_seeds": list(seeds),
        "split_rows": sizes,
        "preserved_original_membership_order_and_energy_labels": True,
        "source_files": source_files,
        "extended_files": output_files,
        "duplicate_groups": duplicates.sample_id.nunique(),
        "duplicate_policy": "reuse frozen source_id groups; equal-weight mean; no rows removed",
        "duplicate_max_ranges": {
            target: float(duplicates[f"{target}_range"].max()) if len(duplicates) else 0.0
            for target in EXTRA_LABELS
        },
        "unit_evidence": (
            "QM9 original publication Table 3; reports/qm9_seven_target_label_audit.json"
        ),
        "unit_source_url": "https://www.nature.com/articles/sdata201422",
    }
    (destination / "data_identity.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
