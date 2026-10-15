"""Validated fixed-split datasets backed by an identity-checked geometry artifact."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from molgap.qm9_models.batch import Molecule
from molgap.transformer.dataset import _read_and_validate_splits
from molgap.utils.files import sha256


class GeometryDataset(Dataset):
    def __init__(self, frame, arrays: dict, index: dict[str, int], target="delta_e") -> None:
        self.frame = frame.reset_index(drop=True)
        self.arrays = arrays
        self.index = index
        self.target = target

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        geo_index = self.index[row.sample_id]
        start, end = self.arrays["offsets"][geo_index : geo_index + 2]
        return Molecule(
            row.sample_id,
            row.smiles,
            torch.from_numpy(self.arrays["atomic_numbers"][start:end].copy()).long(),
            torch.from_numpy(self.arrays["positions"][start:end].copy()).float(),
            float(row[self.target]),
        )


def load_data(config: dict, root: Path, smoke: bool = False):
    """Load QM9 geometry data without Transformer-specific validation."""
    data_config = config["data"]

    # Reuse the same manifest/order validation as the Chemprop and Transformer
    # pipelines. QM9 adapter configs point at the same immutable split contract.
    validation_config = json.loads(json.dumps(config))
    validation_config["data"].setdefault("chemprop_reference_config", "configs/baseline.yaml")
    frames, identity = _read_and_validate_splits(validation_config, root)

    # Load geometry
    geom_path = root / data_config["geometry"]
    if not geom_path.exists():
        raise FileNotFoundError(
            f"Official QM9 geometry missing: {geom_path}; run geometry preparation first"
        )

    metadata_path = geom_path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["geometry_sha256"] != sha256(geom_path):
            raise ValueError("Geometry content hash mismatch")
    else:
        metadata = {"geometry_sha256": sha256(geom_path)}

    # Load geometry arrays
    with np.load(geom_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}

    ids = arrays["sample_ids"].tolist()
    expected = [sid for frame in frames.values() for sid in frame.sample_id]

    if len(set(ids)) != len(ids) or ids != expected:
        raise ValueError("Geometry identities/order do not match split files")

    offsets = arrays["offsets"]
    if len(offsets) != len(ids) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid packed geometry offsets")

    if offsets[-1] != len(arrays["positions"]) or offsets[-1] != len(arrays["atomic_numbers"]):
        raise ValueError("Packed coordinate/atom counts differ")

    index = {sid: i for i, sid in enumerate(ids)}

    if smoke:
        smoke_config = config.get("smoke", {})
        frames = {
            key: frame.head(
                smoke_config.get("train_rows", 128)
                if key == "train"
                else smoke_config.get("eval_rows", 32)
            ).copy()
            for key, frame in frames.items()
        }

    target = data_config.get("target_column", "delta_e")
    datasets = {key: GeometryDataset(frame, arrays, index, target) for key, frame in frames.items()}
    if data_config.get("topology"):
        from molgap.qm9_models.topology import load_topology

        datasets = load_topology(config, root, metadata, frames, arrays, index)

    return frames, datasets, identity, metadata
