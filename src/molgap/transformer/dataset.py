"""Dataset preparation that reuses and verifies the Chemprop split files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from molgap.config import ConfigError, resolve_path
from molgap.transformer.tokenizer import SmilesTokenizer
from molgap.utils.files import sha256

SPLIT_NAMES = ("train", "val", "test")


class SmilesRegressionDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-encoded fixed-length SMILES and physical regression labels."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not (len(input_ids) == len(attention_mask) == len(targets)):
            raise ValueError("Encoded inputs, masks, and targets must have equal lengths.")
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "target": self.targets[index],
        }


@dataclass
class PreparedTransformerData:
    """Validated frames, encoded datasets, tokenizer, and train-only scaler."""

    frames: dict[str, pd.DataFrame]
    datasets: dict[str, SmilesRegressionDataset]
    tokenizer: SmilesTokenizer
    target_mean: float
    target_scale: float
    split_identity: dict[str, Any]
    tokenizer_report: dict[str, Any]


def _load_reference_config(config: dict[str, Any], root: Path) -> tuple[Path, dict[str, Any]]:
    reference_path = resolve_path(root, config["data"]["chemprop_reference_config"])
    if not reference_path.is_file():
        raise FileNotFoundError(f"Chemprop reference config does not exist: {reference_path}")
    with reference_path.open(encoding="utf-8") as handle:
        reference = yaml.safe_load(handle)
    if not isinstance(reference, dict) or not isinstance(reference.get("data"), dict):
        raise ConfigError(f"Invalid Chemprop reference config: {reference_path}")
    return reference_path, reference


def _validate_chemprop_reference(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Prove that the sequence experiment points at Chemprop's exact data contract."""
    reference_path, reference = _load_reference_config(config, root)
    data = config["data"]
    reference_data = reference["data"]
    actual_dir = resolve_path(root, data["processed_dir"]).resolve()
    reference_dir = resolve_path(root, reference_data.get("processed_dir", "")).resolve()
    comparisons = {
        "processed_dir": actual_dir == reference_dir,
        "id_column": data["id_column"] == reference_data.get("id_column"),
        "smiles_column": data["smiles_column"] == reference_data.get("smiles_column"),
        "target": data["target_column"] in reference_data.get("targets", []),
        "split": data["split"] == reference_data.get("split"),
    }
    failed = [name for name, matched in comparisons.items() if not matched]
    if failed:
        raise ConfigError(
            "Transformer data contract differs from the Chemprop reference config: "
            + ", ".join(failed)
        )
    return {
        "path": str(reference_path),
        "sha256": sha256(reference_path),
        "comparisons": comparisons,
    }


def _read_and_validate_splits(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    data = config["data"]
    processed_dir = resolve_path(root, data["processed_dir"])
    manifest_path = processed_dir / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest_required = {
        data["id_column"],
        data["smiles_column"],
        "split",
        "split_position",
        "split_method",
        "split_strategy",
        "seed",
        "stratify_column",
        "stratify_bins_requested",
    }
    missing_manifest = sorted(manifest_required - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Split manifest is missing columns: {missing_manifest}")
    if manifest[data["id_column"]].astype(str).duplicated().any():
        raise ValueError("split_manifest.csv contains duplicate sample IDs.")
    if set(manifest["split"].astype(str)) != set(SPLIT_NAMES):
        raise ValueError("split_manifest.csv must contain exactly train, val, and test rows.")

    split_config = data["split"]
    expected_strategy = "quantile_stratified"
    checks = {
        "method": set(manifest["split_method"].astype(str)) == {split_config["method"]},
        "strategy": set(manifest["split_strategy"].astype(str)) == {expected_strategy},
        "seed": set(manifest["seed"].astype(int)) == {int(split_config["seed"])},
        "stratify_column": set(manifest["stratify_column"].astype(str))
        == {split_config["stratify"]["column"]},
        "stratify_bins": set(manifest["stratify_bins_requested"].astype(int))
        == {int(split_config["stratify"]["bins"])},
    }
    failed_checks = [name for name, matched in checks.items() if not matched]
    if failed_checks:
        raise ValueError("Split manifest conflicts with YAML: " + ", ".join(failed_checks))

    frames: dict[str, pd.DataFrame] = {}
    split_files: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    for split_name in SPLIT_NAMES:
        path = processed_dir / f"{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed split: {path}")
        frame = pd.read_csv(path)
        required = [data["id_column"], data["smiles_column"], data["target_column"]]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if frame[data["id_column"]].astype(str).duplicated().any():
            raise ValueError(f"{path} contains duplicate sample IDs.")
        targets = frame[data["target_column"]].to_numpy(dtype=float)
        if not np.isfinite(targets).all():
            raise ValueError(f"{path} contains missing or non-finite delta_e values.")

        expected = (
            manifest.loc[manifest["split"] == split_name]
            .sort_values("split_position", kind="stable")
            .reset_index(drop=True)
        )
        if expected["split_position"].astype(int).tolist() != list(range(len(expected))):
            raise ValueError(f"Manifest positions for {split_name} are not contiguous from zero.")
        actual_ids = frame[data["id_column"]].astype(str).reset_index(drop=True)
        expected_ids = expected[data["id_column"]].astype(str).reset_index(drop=True)
        actual_smiles = frame[data["smiles_column"]].astype(str).reset_index(drop=True)
        expected_smiles = expected[data["smiles_column"]].astype(str).reset_index(drop=True)
        if not actual_ids.equals(expected_ids):
            raise ValueError(f"{path} sample membership/order differs from split_manifest.csv.")
        if not actual_smiles.equals(expected_smiles):
            raise ValueError(f"{path} SMILES/order differs from split_manifest.csv.")
        overlap = all_ids.intersection(actual_ids)
        if overlap:
            raise ValueError(f"Sample IDs leak across splits; first overlap: {next(iter(overlap))}")
        all_ids.update(actual_ids)
        frames[split_name] = frame
        split_files[split_name] = {"path": str(path), "sha256": sha256(path), "rows": len(frame)}

    manifest_ids = set(manifest[data["id_column"]].astype(str))
    if all_ids != manifest_ids:
        raise ValueError("The train/val/test union differs from split_manifest.csv.")
    identity = {
        "verified_same_split_as_chemprop": True,
        "chemprop_reference": _validate_chemprop_reference(config, root),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "rows": len(manifest),
        },
        "split_files": split_files,
        "manifest_checks": checks,
        "no_sample_overlap": True,
        "row_order_matches_manifest": True,
    }
    return frames, identity


def prepare_transformer_data(config: dict[str, Any], root: Path) -> PreparedTransformerData:
    """Validate shared splits, fit the train-only tokenizer, and encode all rows."""
    frames, identity = _read_and_validate_splits(config, root)
    data = config["data"]
    tokenizer_config = config["tokenizer"]
    tokenizer = SmilesTokenizer(
        pattern=tokenizer_config["pattern"],
        max_length=int(tokenizer_config["max_length"]),
        special_tokens=tokenizer_config["special_tokens"],
    )
    tokenizer.fit(frames["train"][data["smiles_column"]].astype(str))

    datasets: dict[str, SmilesRegressionDataset] = {}
    split_reports: dict[str, Any] = {}
    for split_name, frame in frames.items():
        encoded = [tokenizer.encode(smiles) for smiles in frame[data["smiles_column"]].astype(str)]
        input_ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long)
        attention_mask = torch.tensor([item.attention_mask for item in encoded], dtype=torch.bool)
        targets = torch.tensor(frame[data["target_column"]].to_numpy(), dtype=torch.float32)
        datasets[split_name] = SmilesRegressionDataset(input_ids, attention_mask, targets)
        token_counts = [item.token_count for item in encoded]
        unknown_count = sum(item.unknown_count for item in encoded)
        total_tokens = sum(token_counts)
        split_reports[split_name] = {
            "rows": len(frame),
            "min_tokens_without_cls": min(token_counts),
            "max_tokens_without_cls": max(token_counts),
            "mean_tokens_without_cls": float(np.mean(token_counts)),
            "unknown_tokens": unknown_count,
            "unknown_rate": 0.0 if total_tokens == 0 else unknown_count / total_tokens,
            "overlength_rows": 0,
        }

    train_targets = frames["train"][data["target_column"]].to_numpy(dtype=np.float64)
    target_mean = float(np.mean(train_targets))
    target_scale = float(np.std(train_targets, ddof=0))
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Training delta_e standard deviation must be finite and positive.")
    report = {
        "vocabulary_source": "train",
        "vocabulary_size": len(tokenizer.vocabulary),
        "max_length_including_cls": tokenizer.max_length,
        "special_token_ids": {
            "pad": tokenizer.pad_id,
            "unk": tokenizer.unk_id,
            "cls": tokenizer.cls_id,
        },
        "splits": split_reports,
        "target_scaler": {
            "fit_split": "train",
            "method": "standard",
            "mean_hartree": target_mean,
            "scale_hartree": target_scale,
        },
    }
    return PreparedTransformerData(
        frames=frames,
        datasets=datasets,
        tokenizer=tokenizer,
        target_mean=target_mean,
        target_scale=target_scale,
        split_identity=identity,
        tokenizer_report=report,
    )


def write_data_artifacts(prepared: PreparedTransformerData, run_dir: Path) -> None:
    """Persist the exact split proof, tokenizer, and audit statistics."""
    artifacts = {
        "split_identity.json": prepared.split_identity,
        "tokenizer_report.json": prepared.tokenizer_report,
        "vocabulary.json": prepared.tokenizer.to_dict(),
    }
    for name, payload in artifacts.items():
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
