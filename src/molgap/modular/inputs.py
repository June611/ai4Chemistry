"""Reuse existing graph, token and identity-checked 3D input preparation."""

import copy
import json
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from molgap.modular.data import InputLabels


def collate_inputs(rows, kind):
    values, labels, ids = zip(*rows, strict=True)
    if kind == "chemprop":
        from chemprop.data.collate import collate_batch

        batch = collate_batch(values)
        inputs = dict(graph=batch.bmg, V_d=batch.V_d)
    elif kind == "sequence":
        inputs = {k: torch.stack([v[k] for v in values]) for k in ("input_ids", "attention_mask")}
    elif kind == "smiles":
        inputs = dict(smiles=list(values))
    else:
        from molgap.qm9_models.batch import collate_molecules

        inputs = collate_molecules(values)
        inputs.pop("target", None)
    return inputs, torch.from_numpy(np.stack(labels)), list(ids)


def source_for_seed(config):
    src = copy.deepcopy(config["source"])
    seed = config["training"]["seed"]
    src["data"]["processed_dir"] = f"data/processed/qm9_full/random/seed{seed}"
    src["data"]["split"]["seed"] = seed
    src["data"]["chemprop_reference_config"] = (
        "configs/baseline.yaml"
        if seed == 3407
        else f"configs/qm9_models/split_references/seed{seed}.yaml"
    )
    if "geometry" in src["data"]:
        src["data"]["geometry"] = f"data/geometry/qm9_full_seed{seed}.npz"
    src["training"]["seed"] = seed
    return src


def make_inputs(config, root, frames, identity, run_dir=None, saved_tokenizer=None):
    name = config["backbone"]["model"]
    source = source_for_seed(config)
    input_sets, tokenizer = {}, None
    if name == "chemprop":
        from chemprop.data import MoleculeDatapoint, MoleculeDataset
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        for split, frame in frames.items():
            points = [MoleculeDatapoint.from_smi(s, y=np.array([0.0])) for s in frame.smiles]
            ds = MoleculeDataset(points, SimpleMoleculeMolGraphFeaturizer())
            ds.cache = True
            input_sets[split] = ds
        kind = "chemprop"
    elif name in {"smiles_transformer", "spe_transformer"}:
        from molgap.transformer.dataset import (
            SmilesRegressionDataset,
            _read_and_validate_splits,
            prepare_transformer_data,
        )

        if saved_tokenizer is None:
            prepared = prepare_transformer_data(source, root)
            tokenizer = prepared.tokenizer
            full_frames, datasets = prepared.frames, prepared.datasets
        else:
            from molgap.transformer.tokenizer import SmilesTokenizer

            full_frames, _ = _read_and_validate_splits(source, root)
            tokenizer = SmilesTokenizer.from_dict(saved_tokenizer)
            datasets = {}
            for split, frame in frames.items():
                encoded = [tokenizer.encode(s) for s in frame.smiles]
                datasets[split] = SmilesRegressionDataset(
                    torch.tensor([e.input_ids for e in encoded]),
                    torch.tensor([e.attention_mask for e in encoded], dtype=torch.bool),
                    torch.zeros(len(frame)),
                )
        for split, ds in datasets.items():
            # Use original train-only vocabulary and validate exact row alignment.
            if (
                full_frames[split].head(len(frames[split])).sample_id.tolist()
                != frames[split].sample_id.tolist()
            ):
                raise ValueError("Token inputs and labels differ in membership/order")
            input_sets[split] = ds
        identity["tokenizer"] = tokenizer.to_dict()
        if run_dir:
            (run_dir / "vocabulary.json").write_text(
                json.dumps(tokenizer.to_dict(), indent=2) + "\n"
            )
        kind = "sequence"
    elif name == "esa":
        input_sets = {split: f.smiles.astype(str).tolist() for split, f in frames.items()}
        kind = "smiles"
    else:
        from molgap.qm9_models.data import load_data
        from molgap.qm9_models.train import _validate_infomax_checkpoint
        from molgap.utils.files import sha256

        original_frames, input_sets, original_identity, geometry = load_data(source, root)
        # Original data includes all samples; a diagnostic subset must keep order.
        for split, frame in frames.items():
            original = original_frames[split]
            if original.sample_id.head(len(frame)).tolist() != frame.sample_id.tolist():
                raise ValueError("Geometry inputs and labels differ in membership/order")
        identity["geometry"] = geometry
        if name == "3dinfomax":
            _validate_infomax_checkpoint(source, root, original_identity, geometry)
            p = root / source["model"]["params"]["pretrain_checkpoint"]
            identity["pretrain_checkpoint_sha256"] = sha256(p)
        kind = "geometry"
        if name == "3dinfomax":
            # Identity/checkpoint validation above remains mandatory, but the
            # downstream 2D encoder consumes only SMILES, not padded geometry.
            input_sets = {split: f.smiles.astype(str).tolist() for split, f in frames.items()}
            kind = "smiles"
    # Prefix slicing of the backbone inputs is handled by InputLabels length.
    return input_sets, kind, tokenizer


def make_loaders(config, inputs, kind, frames, scalers):
    t = config["training"]
    loaders = {}
    for i, split in enumerate(("train", "val", "test")):
        ds = InputLabels(inputs[split], frames[split], config["data"]["targets"], scalers)
        loaders[split] = DataLoader(
            ds,
            batch_size=t["batch_size"],
            shuffle=split == "train",
            num_workers=t["num_workers"],
            pin_memory=t.get("pin_memory", True),
            generator=torch.Generator().manual_seed(t["seed"] + i),
            # Recreate deterministic workers each epoch, so resume uses the same
            # base-seed progression. Current datasets have no random transforms.
            persistent_workers=False,
            drop_last=False,
            collate_fn=partial(collate_inputs, kind=kind),
        )
    return loaders
