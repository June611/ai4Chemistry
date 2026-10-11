"""Pretrain the official 3D Infomax PNA/Net3D pair on the fixed QM9 train split."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from molgap.config import find_project_root
from molgap.qm9_models.batch import collate_molecules
from molgap.qm9_models.data import load_data
from molgap.qm9_models.infomax import _molecular_graph
from molgap.qm9_models.sources import activate_source


def _activate(root: Path):
    activate_source(root, "3dinfomax")
    for module_name in list(sys.modules):
        if module_name in {"models", "commons"} or module_name.startswith(("models.", "commons.")):
            del sys.modules[module_name]
    from commons.losses import NTXent
    from models.net3d import Net3D
    from models.pna import PNA

    return PNA, Net3D, NTXent


def _complete_graphs(batch: dict, device: torch.device):
    import dgl

    graphs = []
    for positions, mask in zip(batch["positions"], batch["atom_mask"], strict=True):
        positions = positions[mask].to(device)
        count = len(positions)
        nodes = torch.arange(count, device=device)
        src = nodes.repeat_interleave(count)
        dst = nodes.repeat(count)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        graph = dgl.graph((src, dst), num_nodes=count, device=device)
        graph.edata["d"] = torch.linalg.vector_norm(
            positions[src] - positions[dst], dim=-1, keepdim=True
        ).detach()
        graphs.append(graph)
    return dgl.batch(graphs)


def _forward(model2d, model3d, batch: dict, device: torch.device):
    import dgl

    graph2d = dgl.batch([_molecular_graph(smiles, device) for smiles in batch["smiles"]])
    graph3d = _complete_graphs(batch, device)
    return model2d(graph2d), model3d(graph3d)


def _mean_loss(model2d, model3d, loss_fn, loader, device):
    model2d.eval()
    model3d.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for batch in loader:
            z2d, z3d = _forward(model2d, model3d, batch, device)
            loss = loss_fn(z2d, z3d)
            total += float(loss) * len(batch["smiles"])
            rows += len(batch["smiles"])
    return total / rows


def run(
    config_path: Path,
    *,
    smoke: bool = False,
    split_seed_override: int | None = None,
    seed_override: int | None = None,
) -> Path:
    root = find_project_root(config_path)
    config = yaml.safe_load(config_path.read_text())
    config = copy.deepcopy(config)
    if split_seed_override is not None:
        split_seed = int(split_seed_override)
        reference = (
            "configs/baseline.yaml"
            if split_seed == 3407
            else f"configs/qm9_models/split_references/seed{split_seed}.yaml"
        )
        if not (root / reference).is_file():
            raise ValueError(f"Unsupported split seed {split_seed}: missing {reference}")
        config["data"].update(
            {
                "processed_dir": f"data/processed/qm9_full/random/seed{split_seed}",
                "geometry": f"data/geometry/qm9_full_seed{split_seed}.npz",
                "chemprop_reference_config": reference,
            }
        )
        config["data"]["split"]["seed"] = split_seed
    settings = config["pretraining"]
    if seed_override is not None:
        settings["seed"] = int(seed_override)
    seed = int(settings["seed"])
    split_seed = int(config["data"]["split"]["seed"])
    if seed != split_seed:
        raise ValueError(
            "3D Infomax pretraining requires its seed to equal the split seed "
            f"(training={seed}, split={split_seed})."
        )
    config["protocol"] = {
        "id": "paired_split_gap_mae_v2",
        "run_mode": "paired_split_seed",
        "split_seed": split_seed,
        "training_seed": seed,
        "seed_pairing": "matched",
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, datasets, identity, geometry = load_data(config, root, smoke=False)
    train_count = min(int(settings["num_train"]), len(datasets["train"]))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(datasets["train"]), generator=generator)[:train_count]
    train_data = Subset(datasets["train"], indices.tolist())
    val_data = datasets["val"]
    epochs = 1 if smoke else int(settings["epochs"])
    batch_size = min(8, int(settings["batch_size"])) if smoke else int(settings["batch_size"])
    if smoke:
        train_data = Subset(train_data, range(min(32, len(train_data))))
        val_data = Subset(val_data, range(16))
    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_molecules,
            generator=generator,
            num_workers=0 if smoke else int(settings["num_workers"]),
        ),
        "val": DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_molecules,
            num_workers=0 if smoke else int(settings["num_workers"]),
        ),
    }
    PNA, Net3D, NTXent = _activate(root)
    model2d = PNA(**config["model2d"]).to(device)
    model3d = Net3D(**config["model3d"]).to(device)
    loss_fn = NTXent(tau=float(settings["tau"]))
    optimizer = torch.optim.Adam(
        [*model2d.parameters(), *model3d.parameters()], lr=float(settings["lr"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(settings["lr_factor"]),
        patience=int(settings["lr_patience"]),
        cooldown=int(settings["lr_cooldown"]),
        min_lr=float(settings["min_lr"]),
        threshold=float(settings["min_delta"]),
        threshold_mode="abs",
    )
    base_lr = float(settings["lr"])
    warmup_steps = int(settings["warmup_steps"])
    global_step = 0
    output_root = root / config["output"]["root"]
    if split_seed_override is not None:
        output_root = output_root / f"splitseed{int(split_seed_override)}"
    output = output_root / f"seed{seed}"
    if smoke:
        output = output.with_name(output.name + "_smoke")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pretraining run: {output}")
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "data_identity.json").write_text(
        json.dumps(
            {
                "split_identity": identity,
                "geometry_sha256": geometry["geometry_sha256"],
                "train_rows_available": len(frames["train"]),
                "train_rows_sampled": train_count,
            },
            indent=2,
        )
        + "\n"
    )
    best = float("inf")
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model2d.train()
        model3d.train()
        total = 0.0
        rows = 0
        for batch in tqdm(loaders["train"], desc=f"3D Infomax {epoch + 1}", mininterval=2):
            global_step += 1
            if global_step <= warmup_steps:
                warmup_lr = base_lr * global_step / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr
            optimizer.zero_grad()
            z2d, z3d = _forward(model2d, model3d, batch, device)
            loss = loss_fn(z2d, z3d)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite 3D Infomax pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model2d.parameters(), *model3d.parameters()],
                float(settings["gradient_clip_val"]),
            )
            optimizer.step()
            total += float(loss.detach()) * len(batch["smiles"])
            rows += len(batch["smiles"])
        val_loss = _mean_loss(model2d, model3d, loss_fn, loaders["val"], device)
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite 3D Infomax validation loss")
        if global_step > warmup_steps:
            scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / rows,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if val_loss < best - float(settings["min_delta"]):
            best = val_loss
            stale = 0
            torch.save(
                {
                    "model_state_dict": model2d.state_dict(),
                    "model3d_state_dict": model3d.state_dict(),
                    "epoch": epoch + 1,
                    "best_validation_loss": best,
                    "training_seed": seed,
                    "split_manifest_sha256": identity["manifest"]["sha256"],
                    "geometry_sha256": geometry["geometry_sha256"],
                },
                output / "best_checkpoint.pt",
            )
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    (output / "timing.json").write_text(
        json.dumps({"seconds": time.perf_counter() - started}, indent=2) + "\n"
    )
    return output / "best_checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qm9_models/3dinfomax_pretrain.yaml")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split-seed", type=int, help="Use a prepared QM9 data split")
    parser.add_argument("--seed", type=int, help="Override the pretraining random seed")
    args = parser.parse_args()
    print(
        run(
            args.config,
            smoke=args.smoke,
            split_seed_override=args.split_seed,
            seed_override=args.seed,
        )
    )


if __name__ == "__main__":
    main()
