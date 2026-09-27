"""Validation and traceable transformations for molecular tabular data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


class DataValidationError(ValueError):
    """Raised when molecular input data are invalid or scientifically ambiguous."""


@dataclass(frozen=True)
class DuplicateResult:
    """Result of duplicate inspection and tolerance-bounded aggregation."""

    frame: pd.DataFrame
    audit: pd.DataFrame
    changes: list[dict[str, Any]]
    conflict_smiles: list[str]


def canonicalize_smiles(smiles: str, preserve_stereo: bool = True) -> str | None:
    """Return canonical SMILES, preserving stereochemistry by default."""
    if pd.isna(smiles):
        return None
    value = str(smiles).strip()
    if not value:
        return None
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=preserve_stereo)


def canonicalize_contract(
    frame: pd.DataFrame, preserve_stereo: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Canonicalize SMILES and return valid rows, rejected rows, and field-level changes."""
    result = frame.copy()
    canonical: list[str | None] = []
    changes: list[dict[str, Any]] = []
    for row in result.itertuples(index=False):
        normalized = canonicalize_smiles(row.smiles_raw, preserve_stereo)
        canonical.append(normalized)
        if normalized is not None and str(row.smiles_raw).strip() != normalized:
            changes.append(
                {
                    "sample_id": row.sample_id,
                    "source_id": row.source_id,
                    "source_row": row.source_row,
                    "field": "smiles",
                    "old_value": row.smiles_raw,
                    "new_value": normalized,
                    "reason": "rdkit_canonicalization",
                }
            )
    result["smiles"] = canonical
    invalid = result["smiles"].isna()
    rejected = result.loc[invalid].copy()
    if not rejected.empty:
        rejected["rejection_reason"] = "invalid_smiles"
    return result.loc[~invalid].reset_index(drop=True), rejected, changes


def coerce_targets(frame: pd.DataFrame, targets: Iterable[str]) -> pd.DataFrame:
    """Convert target columns to floats and reject missing or non-finite values."""
    result = frame.copy()
    target_list = list(targets)
    for target in target_list:
        if target not in result:
            raise DataValidationError(f"Missing required target column: {target}")
        result[target] = pd.to_numeric(result[target], errors="coerce")
    invalid = result[target_list].isna().any(axis=1) | ~np.isfinite(
        result[target_list].to_numpy(dtype=float)
    ).all(axis=1)
    if invalid.any():
        rows = result.loc[invalid, "source_row"].astype(int).tolist()[:10]
        raise DataValidationError(
            f"Targets must be complete finite numbers; invalid source rows include {rows}."
        )
    return result


def aggregate_duplicates(
    frame: pd.DataFrame,
    targets: list[str],
    tolerance: float,
) -> DuplicateResult:
    """Average duplicate labels only when every target spread is within tolerance."""
    rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    conflicts: list[str] = []

    for smiles, group in frame.groupby("smiles", sort=False, dropna=False):
        first = group.iloc[0].copy()
        if len(group) == 1:
            rows.append(first)
            continue

        spread = group[targets].max() - group[targets].min()
        means = group[targets].mean()
        conflict = bool((spread > tolerance + 1e-12).any())
        status = "conflict" if conflict else "aggregated"
        if conflict:
            conflicts.append(str(smiles))

        for original in group.itertuples(index=False):
            detail = {
                "canonical_smiles": smiles,
                "group_size": len(group),
                "status": status,
                "sample_id": original.sample_id,
                "source": original.source,
                "source_id": original.source_id,
                "source_row": original.source_row,
                "smiles_raw": original.smiles_raw,
                "smiles": original.smiles,
            }
            for target in targets:
                detail[target] = getattr(original, target)
                detail[f"{target}_min"] = float(group[target].min())
                detail[f"{target}_max"] = float(group[target].max())
                detail[f"{target}_range"] = float(spread[target])
                detail[f"{target}_aggregate"] = float(means[target])
            audit_rows.append(detail)

        if conflict:
            continue

        aggregate_id = str(first["sample_id"])
        first["source_id"] = "|".join(group["source_id"].astype(str))
        for target in targets:
            old_value = float(first[target])
            new_value = float(means[target])
            first[target] = new_value
            if old_value != new_value:
                changes.append(
                    {
                        "sample_id": aggregate_id,
                        "source_id": first["source_id"],
                        "source_row": int(first["source_row"]),
                        "field": target,
                        "old_value": old_value,
                        "new_value": new_value,
                        "reason": "duplicate_group_mean",
                    }
                )
        for duplicate in group.iloc[1:].itertuples(index=False):
            changes.append(
                {
                    "sample_id": duplicate.sample_id,
                    "source_id": duplicate.source_id,
                    "source_row": duplicate.source_row,
                    "field": "row_status",
                    "old_value": "active",
                    "new_value": f"aggregated_into:{aggregate_id}",
                    "reason": "duplicate_canonical_smiles",
                }
            )
        rows.append(first)

    deduplicated = pd.DataFrame(rows, columns=frame.columns).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return DuplicateResult(deduplicated, audit, changes, conflicts)


def resolve_duplicates(
    frame: pd.DataFrame,
    smiles_column: str,
    targets: list[str],
    policy: str,
    tolerance: float,
) -> tuple[pd.DataFrame, int]:
    """Compatibility wrapper for the original public helper."""
    working = (
        frame.rename(columns={smiles_column: "smiles"}) if smiles_column != "smiles" else frame
    )
    required_trace = {"sample_id", "source_id", "source_row"}
    if not required_trace.issubset(working):
        working = working.copy()
        working["sample_id"] = working.index.astype(str)
        working["source_id"] = working.index.astype(str)
        working["source_row"] = working.index.to_numpy() + 2
    if policy not in {"mean", "mean_within_tolerance", "error"}:
        raise DataValidationError(
            "duplicate_policy must be 'mean_within_tolerance' for production preprocessing."
        )
    outcome = aggregate_duplicates(working, targets, tolerance)
    if outcome.conflict_smiles or (policy == "error" and not outcome.audit.empty):
        raise DataValidationError(
            f"Duplicate canonical SMILES require review: {outcome.conflict_smiles[:5]}"
        )
    result = outcome.frame
    if smiles_column != "smiles":
        result = result.rename(columns={"smiles": smiles_column})
    return result[frame.columns].reset_index(drop=True), len(frame) - len(result)
