"""Build auditable, deterministic Chemprop CSV splits from raw QM9 tables."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from molgap.config import find_project_root
from molgap.data.adapters import DataSpecificationError, read_source
from molgap.data.audit import (
    artifact_record,
    sha256_file,
    summarize_targets,
    write_json,
    write_jsonl,
)
from molgap.data.dataset import aggregate_duplicates, canonicalize_contract
from molgap.data.split import SPLIT_NAMES, split_with_manifest

PIPELINE_VERSION = "1.1.0"
TARGETS = ["homo", "lumo", "delta_e"]
CONTRACT_COLUMNS = [
    "sample_id",
    "source",
    "source_id",
    "smiles_raw",
    "smiles",
    *TARGETS,
]
CHANGE_COLUMNS = [
    "sample_id",
    "source_id",
    "source_row",
    "field",
    "old_value",
    "new_value",
    "reason",
]


class ConfirmationRequired(RuntimeError):
    """Raised after audit artifacts are written for a blocking ambiguity."""


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Preprocessing config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DataSpecificationError("Preprocessing config root must be a mapping.")
    root = find_project_root(config_path)
    _validate_config(config)
    return config, root, config_path


def _validate_config(config: dict[str, Any]) -> None:
    for section in ("dataset", "labels", "processing", "split", "chemprop", "output"):
        if not isinstance(config.get(section), dict):
            raise DataSpecificationError(f"Missing config mapping: {section}")
    if config.get("schema_version") != 1:
        raise DataSpecificationError("schema_version must be 1.")
    source = config["dataset"].get("source", {})
    if not isinstance(source, dict) or not isinstance(source.get("columns"), dict):
        raise DataSpecificationError("dataset.source and dataset.source.columns are required.")
    if str(config["labels"].get("unit", "")).lower() != "hartree":
        raise DataSpecificationError("This pipeline requires labels.unit: hartree.")
    processing = config["processing"]
    if processing.get("duplicate_policy") != "mean_within_tolerance":
        raise DataSpecificationError("duplicate_policy must be mean_within_tolerance.")
    if float(processing.get("duplicate_tolerance", -1)) != 1.1e-4:
        raise DataSpecificationError("duplicate_tolerance must be 1.1e-4.")
    split = config["split"]
    if split.get("method") not in {"random", "scaffold_balanced"}:
        raise DataSpecificationError("split.method must be random or scaffold_balanced.")
    sizes = split.get("sizes")
    if not isinstance(sizes, list) or len(sizes) != 3 or not np.isclose(sum(sizes), 1.0):
        raise DataSpecificationError("split.sizes must contain three fractions summing to 1.")
    stratify = split.get("stratify")
    if stratify is not None:
        if split["method"] != "random":
            raise DataSpecificationError(
                "split.stratify is supported only for random splitting; scaffold groups "
                "must remain indivisible."
            )
        if not isinstance(stratify, dict):
            raise DataSpecificationError("split.stratify must be a mapping.")
        if stratify.get("strategy") != "quantile":
            raise DataSpecificationError("split.stratify.strategy must be quantile.")
        if stratify.get("column") not in TARGETS:
            raise DataSpecificationError(f"split.stratify.column must be one of {TARGETS}.")
        bins = stratify.get("bins")
        if not isinstance(bins, int) or bins < 2:
            raise DataSpecificationError("split.stratify.bins must be an integer >= 2.")
    target_columns = config["chemprop"].get("target_columns")
    if not isinstance(target_columns, list) or not target_columns:
        raise DataSpecificationError("chemprop.target_columns must be a non-empty list.")
    if set(target_columns) - set(TARGETS):
        raise DataSpecificationError(f"Unsupported Chemprop targets: {target_columns}")


def _output_dir(config: dict[str, Any], root: Path) -> Path:
    output_root = Path(config["output"]["root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    split = config["split"]
    return output_root / config["dataset"]["name"] / split["method"] / f"seed{split['seed']}"


def _log_event(
    records: list[dict[str, Any]],
    operation: str,
    before_rows: int,
    after_rows: int,
    affected_columns: list[str],
    parameters: dict[str, Any],
) -> None:
    records.append(
        {
            "timestamp": _timestamp(),
            "pipeline_version": PIPELINE_VERSION,
            "operation": operation,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "affected_rows": abs(after_rows - before_rows),
            "affected_columns": affected_columns,
            "parameters": parameters,
        }
    )


def _append_rejection_changes(rejected: pd.DataFrame, changes: list[dict[str, Any]]) -> None:
    for row in rejected.itertuples(index=False):
        changes.append(
            {
                "sample_id": row.sample_id,
                "source_id": row.source_id,
                "source_row": row.source_row,
                "field": "row_status",
                "old_value": "active",
                "new_value": "rejected",
                "reason": row.rejection_reason,
            }
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def _rollback_matches_adapted_contract(
    adapted: pd.DataFrame,
    deduplicated: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    rejected: pd.DataFrame,
) -> bool:
    """Reconstruct the adapted contract from final/audit rows and compare it to source order."""
    compare_columns = [
        "sample_id",
        "source",
        "source_id",
        "source_row",
        "smiles_raw",
        *TARGETS,
    ]
    duplicate_smiles = set(duplicate_audit.get("canonical_smiles", pd.Series(dtype=str)))
    unique_rows = deduplicated.loc[~deduplicated["smiles"].isin(duplicate_smiles), compare_columns]
    duplicate_rows = duplicate_audit.reindex(columns=compare_columns)
    rejected_rows = rejected.reindex(columns=compare_columns)
    reconstructed = pd.concat(
        [unique_rows, duplicate_rows, rejected_rows], ignore_index=True
    ).sort_values("source_row")
    expected = adapted[compare_columns].sort_values("source_row")
    reconstructed = reconstructed.reset_index(drop=True).astype({"source_id": "string"})
    expected = expected.reset_index(drop=True).astype({"source_id": "string"})
    try:
        pd.testing.assert_frame_equal(
            reconstructed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError:
        return False
    return True


def _write_lineage(
    root: Path,
    output_dir: Path,
    source_path: Path,
    artifacts: list[Path],
    version: str,
    created_at: str,
) -> None:
    source_record = artifact_record(source_path, root, version, created_at)
    artifact_records = [artifact_record(path, root, version, created_at) for path in artifacts]
    lineage_path = output_dir / "data_lineage.json"
    write_json(
        lineage_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "artifact_version": version,
            "created_at": created_at,
            "source": source_record,
            "transformations": [
                "source_adapter",
                "rdkit_canonicalization",
                "label_consistency_filter",
                "tolerance_bounded_duplicate_aggregation",
                "deterministic_split",
                "chemprop_csv_projection",
            ],
            "artifacts": artifact_records,
        },
    )
    manifest_records = [
        source_record,
        *artifact_records,
        artifact_record(lineage_path, root, version, created_at),
    ]
    manifest_path = output_dir / "artifact_manifest.json"
    write_json(
        manifest_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "artifact_version": version,
            "created_at": created_at,
            "files": manifest_records,
        },
    )
    checksum_paths = [source_path, *artifacts, lineage_path, manifest_path]
    with (output_dir / "lineage.sha256").open("w", encoding="utf-8") as handle:
        handle.write(f"# artifact_version={version}\n")
        handle.write(f"# created_at={created_at}\n")
        for path in checksum_paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")


def prepare_dataset(config_path: str | Path) -> dict[str, Any]:
    """Execute the configured preprocessing pipeline without training a model."""
    config, root, resolved_config_path = _load_config(config_path)
    created_at = _timestamp()
    dataset_config = config["dataset"]
    processing = config["processing"]
    labels = config["labels"]
    split_config = config["split"]
    version = str(dataset_config["artifact_version"])
    output_dir = _output_dir(config, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(dataset_config["source"]["path"])
    if not source_path.is_absolute():
        source_path = root / source_path

    logs: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    adapter_result = read_source(root, config)
    adapted = adapter_result.frame
    input_rows = len(adapted)
    _log_event(
        logs,
        "data_probe",
        input_rows,
        input_rows,
        list(adapted.columns),
        adapter_result.audit,
    )

    nulls = {key: value for key, value in adapter_result.audit["null_counts"].items() if value}
    if nulls:
        pending.append({"type": "missing_values", "details": nulls})
    if adapter_result.audit["pseudo_missing"]:
        pending.append(
            {"type": "ambiguous_pseudo_missing", "details": adapter_result.audit["pseudo_missing"]}
        )
    numeric_invalid = adapted[TARGETS].isna().any(axis=1) | ~np.isfinite(
        adapted[TARGETS].to_numpy(dtype=float)
    ).all(axis=1)
    if numeric_invalid.any():
        pending.append(
            {
                "type": "missing_or_non_numeric_targets",
                "rows": adapted.loc[numeric_invalid, "source_row"].astype(int).tolist(),
            }
        )

    canonical, invalid_smiles, canonical_changes = canonicalize_contract(
        adapted, preserve_stereo=bool(processing["preserve_stereo"])
    )
    changes.extend(canonical_changes)
    _log_event(
        logs,
        "rdkit_canonicalization",
        input_rows,
        len(canonical),
        ["smiles_raw", "smiles"],
        {"preserve_stereo": bool(processing["preserve_stereo"])},
    )

    residual = (canonical["delta_e"] - (canonical["lumo"] - canonical["homo"])).abs()
    inconsistent = residual > float(labels["consistency_tolerance"])
    inconsistent_rows = canonical.loc[inconsistent].copy()
    if not inconsistent_rows.empty:
        inconsistent_rows["rejection_reason"] = "orbital_gap_inconsistency"
        inconsistent_rows["consistency_abs_error"] = residual.loc[inconsistent].to_numpy()
    consistency_fraction = len(inconsistent_rows) / input_rows
    if consistency_fraction > 0.001:
        pending.append(
            {
                "type": "excessive_orbital_gap_inconsistency",
                "rows": len(inconsistent_rows),
                "fraction": consistency_fraction,
                "threshold": 0.001,
            }
        )
    rejected = pd.concat([invalid_smiles, inconsistent_rows], ignore_index=True, sort=False)
    _append_rejection_changes(rejected, changes)
    canonical = canonical.loc[~inconsistent].reset_index(drop=True)
    rejected_fraction = len(rejected) / input_rows
    if rejected_fraction > float(processing["max_rejected_fraction"]):
        pending.append(
            {
                "type": "excessive_row_rejection",
                "rows": len(rejected),
                "fraction": rejected_fraction,
                "threshold": float(processing["max_rejected_fraction"]),
            }
        )
    _log_event(
        logs,
        "label_consistency_filter",
        len(canonical) + len(inconsistent_rows),
        len(canonical),
        TARGETS,
        {"tolerance": float(labels["consistency_tolerance"])},
    )

    duplicate_result = aggregate_duplicates(
        canonical, TARGETS, float(processing["duplicate_tolerance"])
    )
    changes.extend(duplicate_result.changes)
    if duplicate_result.conflict_smiles:
        pending.append(
            {
                "type": "conflicting_duplicate_labels",
                "count": len(duplicate_result.conflict_smiles),
                "smiles": duplicate_result.conflict_smiles,
                "tolerance": float(processing["duplicate_tolerance"]),
            }
        )
    _log_event(
        logs,
        "duplicate_aggregation",
        len(canonical),
        len(duplicate_result.frame),
        ["smiles", *TARGETS],
        {
            "policy": processing["duplicate_policy"],
            "tolerance": float(processing["duplicate_tolerance"]),
            "duplicate_rows": len(duplicate_result.audit),
        },
    )

    # Always persist the audit needed to resolve a blocking ambiguity.
    shutil.copy2(resolved_config_path, output_dir / "config.yaml")
    duplicate_columns = [
        "canonical_smiles",
        "group_size",
        "status",
        "sample_id",
        "source",
        "source_id",
        "source_row",
        "smiles_raw",
        "smiles",
        *[
            column
            for target in TARGETS
            for column in (
                target,
                f"{target}_min",
                f"{target}_max",
                f"{target}_range",
                f"{target}_aggregate",
            )
        ],
    ]
    duplicate_audit = duplicate_result.audit.reindex(columns=duplicate_columns)
    _write_csv(duplicate_audit, output_dir / "duplicates.csv")
    rejected_columns = [*adapted.columns, "smiles", "rejection_reason", "consistency_abs_error"]
    _write_csv(rejected.reindex(columns=rejected_columns), output_dir / "rejected_rows.csv")
    _write_csv(pd.DataFrame(changes, columns=CHANGE_COLUMNS), output_dir / "change_map.csv")
    write_json(output_dir / "pending_confirmations.json", {"items": pending})
    write_jsonl(output_dir / "processing_log.jsonl", logs)

    if pending:
        report = {
            "status": "confirmation_required",
            "pipeline_version": PIPELINE_VERSION,
            "artifact_version": version,
            "created_at": created_at,
            "dataset": dataset_config["name"],
            "source_audit": adapter_result.audit,
            "pending_confirmations": pending,
        }
        write_json(output_dir / "preprocess_report.json", report)
        raise ConfirmationRequired(
            f"Preprocessing stopped with {len(pending)} pending item(s); see {output_dir}."
        )

    dataset = duplicate_result.frame[CONTRACT_COLUMNS].reset_index(drop=True)
    if bool(labels.get("derive_ev", False)):
        dataset["delta_e_ev"] = dataset["delta_e"] * 27.211386245988
    stratify_config = split_config.get("stratify")
    stratify_column = stratify_config.get("column") if stratify_config else None
    stratify_bins = int(stratify_config["bins"]) if stratify_config else None
    splits, manifest = split_with_manifest(
        dataset,
        smiles_column="smiles",
        method=str(split_config["method"]),
        fractions=[float(value) for value in split_config["sizes"]],
        seed=int(split_config["seed"]),
        stratify_column=stratify_column,
        stratify_bins=stratify_bins,
    )
    chemprop_columns = ["sample_id", "smiles", *config["chemprop"]["target_columns"]]
    _write_csv(dataset, output_dir / "dataset.csv")
    for split_name in SPLIT_NAMES:
        _write_csv(splits[split_name][chemprop_columns], output_dir / f"{split_name}.csv")
    _write_csv(manifest, output_dir / "split_manifest.csv")
    split_manifest_sha256 = sha256_file(output_dir / "split_manifest.csv")
    _log_event(
        logs,
        "deterministic_split",
        len(dataset),
        len(dataset),
        ["split"],
        {
            "method": split_config["method"],
            "sizes": split_config["sizes"],
            "seed": int(split_config["seed"]),
            "stratify": stratify_config,
            "stratify_bins_effective": (
                int(manifest["stratify_bins_effective"].dropna().iloc[0])
                if stratify_config
                else None
            ),
            "split_rows": {name: len(splits[name]) for name in SPLIT_NAMES},
        },
    )
    _log_event(
        logs,
        "chemprop_csv_projection",
        len(dataset),
        len(dataset),
        chemprop_columns,
        {"target_scaling": "none", "label_unit": "hartree"},
    )
    write_jsonl(output_dir / "processing_log.jsonl", logs)

    target_statistics = summarize_targets({"all": dataset, **splits}, TARGETS, unit="hartree")
    target_statistics.update(
        {
            "pipeline_version": PIPELINE_VERSION,
            "artifact_version": version,
            "created_at": created_at,
        }
    )
    write_json(output_dir / "target_statistics.json", target_statistics)

    split_sets = {name: set(splits[name]["smiles"]) for name in SPLIT_NAMES}
    no_smiles_leakage = not any(
        split_sets[SPLIT_NAMES[i]] & split_sets[SPLIT_NAMES[j]]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    rollback_matches = _rollback_matches_adapted_contract(
        adapted, duplicate_result.frame, duplicate_result.audit, rejected
    )
    report = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "artifact_version": version,
        "created_at": created_at,
        "dataset": dataset_config["name"],
        "label_unit": "hartree",
        "target_scaling": "none",
        "source_audit": adapter_result.audit,
        "input_rows": input_rows,
        "invalid_smiles_rows": len(invalid_smiles),
        "inconsistent_gap_rows": len(inconsistent_rows),
        "rejected_rows": len(rejected),
        "duplicate_rows": len(duplicate_result.audit),
        "duplicate_groups": int(duplicate_result.audit["canonical_smiles"].nunique()),
        "final_unique_rows": len(dataset),
        "split": {
            "method": split_config["method"],
            "sizes": split_config["sizes"],
            "seed": int(split_config["seed"]),
            "stratify": stratify_config,
            "stratify_bins_effective": (
                int(manifest["stratify_bins_effective"].dropna().iloc[0])
                if stratify_config
                else None
            ),
            "rows": {name: len(splits[name]) for name in SPLIT_NAMES},
            "manifest_sha256": split_manifest_sha256,
        },
        "checks": {
            "no_smiles_leakage": no_smiles_leakage,
            "sample_ids_unique": not dataset["sample_id"].duplicated().any(),
            "labels_finite": bool(np.isfinite(dataset[TARGETS].to_numpy()).all()),
            "chemprop_columns": chemprop_columns,
            "adapted_contract_rollback_matches_source": rollback_matches,
        },
        "pending_confirmations": [],
    }
    write_json(output_dir / "preprocess_report.json", report)

    artifacts = [
        output_dir / name
        for name in (
            "config.yaml",
            "dataset.csv",
            "train.csv",
            "val.csv",
            "test.csv",
            "split_manifest.csv",
            "preprocess_report.json",
            "target_statistics.json",
            "duplicates.csv",
            "rejected_rows.csv",
            "change_map.csv",
            "processing_log.jsonl",
            "pending_confirmations.json",
        )
    ]
    _write_lineage(root, output_dir, source_path, artifacts, version, created_at)
    return report


def build_parser(default_config: str = "configs/data/qm9_full.yaml") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, help="Preprocessing YAML file")
    return parser


def main(default_config: str = "configs/data/qm9_full.yaml") -> None:
    args = build_parser(default_config).parse_args()
    report = prepare_dataset(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
