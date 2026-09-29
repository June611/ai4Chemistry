"""Leakage-resistant random, regression-stratified, and scaffold splitting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedShuffleSplit

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


def _regression_stratified_indices(
    values: pd.Series,
    fractions: list[float],
    seed: int,
    requested_bins: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Create exact-size split indices after deterministic quantile binning.

    The number of bins is reduced only when ties or dataset size make a requested
    stratification mathematically infeasible. Failure is explicit if even two bins
    cannot populate all three splits.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataValidationError("The stratification target must contain finite numbers only.")
    if requested_bins < 2:
        raise DataValidationError("Regression stratification requires at least two bins.")
    unique_values = int(np.unique(numeric).size)
    if unique_values < 2:
        raise DataValidationError("Regression stratification requires at least two unique values.")

    counts = _split_counts(len(numeric), fractions)
    attempted_effective_bins: set[int] = set()
    last_error: ValueError | None = None
    for candidate_bins in range(min(requested_bins, unique_values), 1, -1):
        labels = pd.qcut(numeric, q=candidate_bins, labels=False, duplicates="drop")
        label_array = np.asarray(labels, dtype=int)
        effective_bins = int(np.unique(label_array).size)
        if effective_bins < 2 or effective_bins in attempted_effective_bins:
            continue
        attempted_effective_bins.add(effective_bins)
        try:
            first = StratifiedShuffleSplit(
                n_splits=1,
                train_size=counts[0],
                test_size=counts[1] + counts[2],
                random_state=seed,
            )
            train_indices, remaining_indices = next(
                first.split(np.zeros(len(numeric)), label_array)
            )
            remaining_labels = label_array[remaining_indices]
            second = StratifiedShuffleSplit(
                n_splits=1,
                train_size=counts[1],
                test_size=counts[2],
                random_state=seed + 1,
            )
            val_local, test_local = next(
                second.split(np.zeros(len(remaining_indices)), remaining_labels)
            )
        except ValueError as error:
            last_error = error
            continue
        return (
            {
                "train": train_indices,
                "val": remaining_indices[val_local],
                "test": remaining_indices[test_local],
            },
            label_array,
            effective_bins,
        )
    detail = f" Last splitter error: {last_error}" if last_error is not None else ""
    raise DataValidationError(
        "Unable to construct regression-stratified train/val/test splits. "
        "Use fewer bins or a larger dataset." + detail
    )


def regression_stratified_random_split(
    frame: pd.DataFrame,
    target_column: str,
    fractions: list[float],
    seed: int,
    bins: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], int]:
    """Split by quantile bins while preserving exact requested subset sizes."""
    if target_column not in frame:
        raise DataValidationError(f"Stratification column does not exist: {target_column}")
    indices, labels, effective_bins = _regression_stratified_indices(
        frame[target_column], fractions, seed, bins
    )
    split_frames = {name: frame.iloc[indices[name]].reset_index(drop=True) for name in SPLIT_NAMES}
    bins_by_sample = {
        str(sample_id): int(label)
        for sample_id, label in zip(frame["sample_id"], labels, strict=True)
    }
    return split_frames, bins_by_sample, effective_bins


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
    stratify_column: str | None = None,
    stratify_bins: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Dispatch to a supported split implementation and assert no SMILES leakage."""
    if method == "random" and stratify_column is not None:
        if stratify_bins is None:
            raise DataValidationError("stratify_bins is required with stratify_column.")
        splits, _, _ = regression_stratified_random_split(
            frame, stratify_column, fractions, seed, stratify_bins
        )
    elif method == "random":
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
    stratify_column: str | None = None,
    stratify_bins: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Split data and return a row-level, deterministic assignment manifest."""
    if frame["sample_id"].duplicated().any():
        raise DataValidationError("sample_id must be unique before splitting.")
    bins_by_sample: dict[str, int] = {}
    effective_bins: int | None = None
    if method == "random" and stratify_column is not None:
        if stratify_bins is None:
            raise DataValidationError("stratify_bins is required with stratify_column.")
        splits, bins_by_sample, effective_bins = regression_stratified_random_split(
            frame, stratify_column, fractions, seed, stratify_bins
        )
        split_strategy = "quantile_stratified"
    else:
        splits = split_frame(frame, smiles_column, method, fractions, seed)
        split_strategy = "plain_random" if method == "random" else "scaffold_balanced"

    smiles_sets = [set(splits[name][smiles_column]) for name in SPLIT_NAMES]
    if any(smiles_sets[i] & smiles_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Internal error: a canonical SMILES appears in multiple splits.")
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
                "split_strategy": split_strategy,
                "seed": seed,
                "stratify_column": stratify_column,
                "stratify_bins_requested": stratify_bins,
                "stratify_bins_effective": effective_bins,
                "stratify_bin": bins_by_sample.get(str(row.sample_id)),
            }
        )
    return splits, pd.DataFrame(rows)
