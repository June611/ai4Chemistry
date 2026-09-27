"""Read source-specific CSV files into the project's canonical data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CONTRACT_FIELDS = ("sample_id", "source", "source_id", "smiles_raw", "homo", "lumo", "delta_e")
REQUIRED_MAPPINGS = ("smiles", "homo", "lumo", "delta_e")
PSEUDO_MISSING = {"", "-", "--", "na", "n/a", "nan", "none", "null", "999", "-999"}


class DataSpecificationError(ValueError):
    """Raised when a raw source contradicts its declared schema."""


@dataclass(frozen=True)
class AdapterResult:
    """Adapted rows and the non-mutating source audit produced while reading them."""

    frame: pd.DataFrame
    audit: dict[str, Any]


def _pseudo_missing_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for column in columns:
        values = frame[column].astype("string").str.strip().str.lower()
        counts = values[values.isin(PSEUDO_MISSING)].value_counts(dropna=False)
        if not counts.empty:
            result[column] = {str(value): int(count) for value, count in counts.items()}
    return result


def read_source(root: Path, config: dict[str, Any]) -> AdapterResult:
    """Read one configured source and map it to stable, source-independent columns."""
    dataset = config["dataset"]
    source = dataset["source"]
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {source_path}")

    encoding = str(source["encoding"])
    frame = pd.read_csv(source_path, encoding=encoding)
    expected_rows = int(dataset["expected_rows"])
    if len(frame) != expected_rows:
        raise DataSpecificationError(
            f"Expected {expected_rows} rows in {source_path.name}, found {len(frame)}."
        )

    column_map = source["columns"]
    if set(REQUIRED_MAPPINGS) - set(column_map):
        missing = sorted(set(REQUIRED_MAPPINGS) - set(column_map))
        raise DataSpecificationError(f"Missing canonical column mappings: {missing}")
    id_column = source["id_column"]
    required_source_columns = [id_column, *(column_map[name] for name in REQUIRED_MAPPINGS)]
    missing_columns = [column for column in required_source_columns if column not in frame]
    if missing_columns:
        raise DataSpecificationError(
            f"Source {source_path.name} is missing declared columns: {missing_columns}"
        )

    null_counts = {column: int(frame[column].isna().sum()) for column in required_source_columns}
    pseudo_missing = _pseudo_missing_counts(frame, required_source_columns)
    dataset_name = str(dataset["name"])
    source_ids = frame[id_column].astype("string").str.strip()
    adapted = pd.DataFrame(
        {
            "sample_id": dataset_name + ":" + source_ids,
            "source": dataset_name,
            "source_id": source_ids,
            "source_row": frame.index.to_numpy() + 2,
            "smiles_raw": frame[column_map["smiles"]].astype("string"),
            "homo": pd.to_numeric(frame[column_map["homo"]], errors="coerce"),
            "lumo": pd.to_numeric(frame[column_map["lumo"]], errors="coerce"),
            "delta_e": pd.to_numeric(frame[column_map["delta_e"]], errors="coerce"),
        }
    )
    audit = {
        "path": str(source_path.relative_to(root)),
        "encoding": encoding,
        "rows": len(frame),
        "columns": list(frame.columns),
        "column_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "required_source_columns": required_source_columns,
        "null_counts": null_counts,
        "pseudo_missing": pseudo_missing,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "raw_unique_smiles": int(frame[column_map["smiles"]].nunique(dropna=True)),
    }
    return AdapterResult(adapted, audit)
