"""Audit artifact helpers for the preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write stable, human-readable UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write one stable JSON object per line."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_targets(
    datasets: dict[str, pd.DataFrame], targets: list[str], unit: str
) -> dict[str, Any]:
    """Return descriptive statistics without changing labels."""
    result: dict[str, Any] = {"unit": unit, "scaling": "none", "datasets": {}}
    for dataset_name, frame in datasets.items():
        result["datasets"][dataset_name] = {}
        for target in targets:
            values = frame[target].astype(float)
            result["datasets"][dataset_name][target] = {
                "count": int(values.count()),
                "min": float(values.min()),
                "q01": float(values.quantile(0.01)),
                "mean": float(values.mean()),
                "std_population": float(values.std(ddof=0)),
                "median": float(values.median()),
                "q99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
    return result


def artifact_record(path: Path, root: Path, version: str, created_at: str) -> dict[str, Any]:
    """Build one versioned/checksummed artifact manifest entry."""
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "artifact_version": version,
        "created_at": created_at,
    }
