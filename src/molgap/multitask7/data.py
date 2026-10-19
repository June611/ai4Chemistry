"""Seven original-unit labels, immutable split lineage, train-only scalers."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.data.neural_properties import scaler_record, validate_property_data
from molgap.multitask7.config import TASK_NAMES, TASK_UNITS


class MultiTask7Dataset:
    def __init__(self, csv_path, *, split=None, scalers=None, input_type="graph", tokenizer=None):
        self.path = Path(csv_path)
        self.split = split or self.path.stem
        self.df = pd.read_csv(self.path, float_precision="round_trip")
        required = {"sample_id", "smiles", *TASK_NAMES}
        if not required <= set(self.df):
            raise ValueError(f"Missing columns: {required - set(self.df)}")
        self.targets = self.df[list(TASK_NAMES)].to_numpy(dtype=np.float64)
        if not len(self.df) or not np.isfinite(self.targets).all():
            raise ValueError("Seven targets must be non-empty and finite")
        if self.df.sample_id.isna().any() or self.df.sample_id.duplicated().any():
            raise ValueError("Sample identities must be unique and present")
        if self.df.smiles.isna().any():
            raise ValueError("SMILES must be present")
        if input_type not in {"graph", "smiles", "spe"}:
            raise ValueError("Unknown molecular input")
        self.input_type, self.tokenizer, self.scalers = input_type, tokenizer, scalers

    def fit_scalers(self):
        if self.split != "train":
            raise ValueError("Only the training split may fit scalers")
        self.scalers = {
            t: scaler_record(self.targets[:, i], t, TASK_UNITS[t]) for i, t in enumerate(TASK_NAMES)
        }
        return self.scalers

    def _vectors(self):
        if self.scalers is None:
            raise ValueError("Training scalers are required")
        return (
            np.array([self.scalers[t]["mean"] for t in TASK_NAMES]),
            np.array([self.scalers[t]["scale"] for t in TASK_NAMES]),
        )

    def transform(self, targets):
        means, scales = self._vectors()
        return (np.asarray(targets, dtype=np.float64) - means) / scales

    def inverse_transform(self, normalized):
        means, scales = self._vectors()
        return np.asarray(normalized, dtype=np.float64) * scales + means

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        value = str(row.smiles)
        if self.input_type != "graph":
            if self.tokenizer is None:
                raise ValueError("Token input requires a training-fitted tokenizer")
            value = self.tokenizer.encode(value)
        target = self.targets[index]
        if self.scalers is not None:
            target = self.transform(target)
        return {"input": value, "targets": target, "sample_id": row.sample_id}


def load_splits(config, root):
    # Reuse the existing immutable lineage validator through a single-target proxy.
    # This does not relax or change its single-property public contract.
    proxy = copy.deepcopy(config)
    proxy["data"].update(targets=["homo"], target_column="homo", label_unit="hartree")
    identity = validate_property_data(proxy, root)
    data_root = Path(config["data"]["processed_dir"])
    if not data_root.is_absolute():
        data_root = root / data_root
    datasets = {
        s: MultiTask7Dataset(data_root / f"{s}.csv", split=s) for s in ("train", "val", "test")
    }
    sets = [set(d.df.sample_id.astype(str)) for d in datasets.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
        raise ValueError("Fixed split identities overlap")
    manifest = pd.read_csv(data_root / "split_manifest.csv")
    for split, dataset in datasets.items():
        expected = manifest.loc[manifest.split.eq(split)].sort_values("split_position")
        if expected.sample_id.astype(str).tolist() != dataset.df.sample_id.astype(str).tolist():
            raise ValueError("Split membership/order differs from the frozen manifest")
    scalers = datasets["train"].fit_scalers()
    for dataset in datasets.values():
        dataset.scalers = scalers
    return datasets, identity


def make_graph_loaders(datasets, config):
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    training = config["training"]
    loaders = {}
    for split, ds in datasets.items():
        normalized = ds.transform(ds.targets).astype(np.float32)
        points = [
            MoleculeDatapoint.from_smi(smi, y=y, name=str(name))
            for smi, y, name in zip(ds.df.smiles, normalized, ds.df.sample_id, strict=True)
        ]
        graph = MoleculeDataset(points, SimpleMoleculeMolGraphFeaturizer())
        graph.cache = True
        loaders[split] = build_dataloader(
            graph,
            batch_size=training["batch_size"],
            num_workers=training["num_workers"],
            seed=training["seed"],
            shuffle=split == "train",
            drop_last=False,
            pin_memory=training.get("pin_memory", True),
            persistent_workers=training["num_workers"] > 0,
        )
    return loaders
