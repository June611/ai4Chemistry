"""Leakage-resistant random and Bemis–Murcko scaffold splitting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from molgap.data.dataset import DataValidationError

SPLIT_NAMES = ("train", "val", "test")


def _split_counts(n_items: int, fractions: list[float]) -> list[int]:
    if n_items < 3:
        raise DataValidationError(
            "At least three unique molecules are required for train/val/test."
        )
    raw = np.asarray(fractions, dtype=float) * n_items
    counts = np.floor(raw).astype(int)
    counts[np.argsort(raw - counts)[::-1][: n_items - int(counts.sum())]] += 1
    for index in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise DataValidationError("Dataset is too small for three non-empty splits.")
        counts[donor] -= 1
        counts[index] += 1
    return counts.tolist()


def random_split(frame: pd.DataFrame, fractions: list[float], seed: int) -> dict[str, pd.DataFrame]:
    """Split unique rows deterministically using a local random generator."""
    counts = _split_counts(len(frame), fractions)
    indices = np.random.default_rng(seed).permutation(len(frame))
    boundaries = np.cumsum([0, *counts])
    return {
        name: frame.iloc[indices[boundaries[i] : boundaries[i + 1]]].reset_index(drop=True)
        for i, name in enumerate(SPLIT_NAMES)
    }


def scaffold_key(smiles: str) -> str:
    """Return a stereochemistry-aware Bemis–Murcko scaffold key."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise DataValidationError(f"Invalid canonical SMILES reached splitting: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)


def scaffold_split(
    frame: pd.DataFrame,
    smiles_column: str,
    fractions: list[float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Keep each Bemis–Murcko scaffold in exactly one split."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, smiles in enumerate(frame[smiles_column]):
        groups[scaffold_key(smiles)].append(index)

    rng = np.random.default_rng(seed)
    scaffold_groups = list(groups.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)
    target_sizes = np.asarray(fractions, dtype=float) * len(frame)
    assigned: list[list[int]] = [[], [], []]
    for group in scaffold_groups:
        fill = np.asarray([len(values) for values in assigned], dtype=float) / target_sizes
        destination = int(np.argmin(fill))
        assigned[destination].extend(group)

    if any(not values for values in assigned):
        raise DataValidationError(
            "Scaffold split produced an empty subset. "
            "Use more scaffold-diverse data or random split."
        )
    return {
        name: frame.iloc[indices].reset_index(drop=True)
        for name, indices in zip(SPLIT_NAMES, assigned, strict=True)
    }


def split_frame(
    frame: pd.DataFrame,
    smiles_column: str,
    method: str,
    fractions: list[float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Dispatch to a supported split implementation and assert no SMILES leakage."""
    if method == "random":
        splits = random_split(frame, fractions, seed)
    elif method in {"scaffold", "scaffold_balanced"}:
        splits = scaffold_split(frame, smiles_column, fractions, seed)
    else:
        raise DataValidationError(f"Unsupported split method: {method}")

    smiles_sets = [set(splits[name][smiles_column]) for name in SPLIT_NAMES]
    if any(smiles_sets[i] & smiles_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Internal error: a canonical SMILES appears in multiple splits.")
    if method in {"scaffold", "scaffold_balanced"}:
        scaffold_sets = [
            {scaffold_key(smiles) for smiles in splits[name][smiles_column]} for name in SPLIT_NAMES
        ]
        if any(scaffold_sets[i] & scaffold_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("Internal error: a scaffold appears in multiple splits.")
    return splits


def split_with_manifest(
    frame: pd.DataFrame,
    smiles_column: str,
    method: str,
    fractions: list[float],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Split data and return a row-level, deterministic assignment manifest."""
    if frame["sample_id"].duplicated().any():
        raise DataValidationError("sample_id must be unique before splitting.")
    splits = split_frame(frame, smiles_column, method, fractions, seed)
    assignments: dict[str, tuple[str, int]] = {}
    for split_name in SPLIT_NAMES:
        for position, sample_id in enumerate(splits[split_name]["sample_id"]):
            assignments[str(sample_id)] = (split_name, position)

    rows = []
    for row in frame.itertuples(index=False):
        split_name, position = assignments[str(row.sample_id)]
        rows.append(
            {
                "sample_id": row.sample_id,
                "smiles": getattr(row, smiles_column),
                "scaffold": scaffold_key(getattr(row, smiles_column)),
                "split": split_name,
                "split_position": position,
                "split_method": method,
                "seed": seed,
            }
        )
    return splits, pd.DataFrame(rows)
