"""New budget must not silently change historical contracts or fake a resume."""

import copy
from pathlib import Path

import pytest
import torch

from molgap.config import ConfigError
from molgap.multitask7.config import (
    PROTOCOL_ID_200,
    load_multitask7_config,
    validate_multitask7_config,
)
from scripts.resume_qm9_epoch200 import keep_original_schedule, validate_payload

ROOT = Path(__file__).resolve().parents[1]


def test_separate_100_and_200_contracts():
    old, _ = load_multitask7_config(ROOT / "configs/qm9_multitask7/models/chemprop_mtl7.yaml")
    new = copy.deepcopy(old)
    new["training"]["epochs"] = 200
    with pytest.raises(ConfigError):
        validate_multitask7_config(new)
    new["experiment"]["protocol_id"] = PROTOCOL_ID_200
    validate_multitask7_config(new)
    for epochs in (10, 100, 199, 201):
        bad = copy.deepcopy(new)
        bad["training"]["epochs"] = epochs
        with pytest.raises(ConfigError):
            validate_multitask7_config(bad)


def test_reject_weights_only_and_nonterminal_checkpoints():
    with pytest.raises(ValueError):
        validate_payload(dict(state_dict={"a": torch.ones(1)}, epoch=99))
    full = dict(
        state_dict={"a": torch.ones(1)},
        optimizer_states=[{"state": {}}],
        lr_schedulers=[{"last_epoch": 41900}],
        loops={"fit_loop": {"epoch_progress": {"total": {"completed": 100}}}},
        epoch=99,
        global_step=41900,
    )
    validate_payload(full)
    for change in [dict(epoch=98), dict(global_step=41899), dict(optimizer_states=[])]:
        with pytest.raises(ValueError):
            validate_payload(dict(full, **change))


def test_optimizer_scheduler_resume_to_200_without_restarting_warmup(tmp_path):
    import lightning.pytorch as pl
    from chemprop.schedulers import build_NoamLike_LRSched
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    from molgap.training.terminal_checkpoint import TerminalFitCheckpoint

    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def training_step(self, batch, idx):
            return (self.weight - batch[0]).square().mean()

        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.001)
            scheduler = build_NoamLike_LRSched(
                opt, 2, self.trainer.max_epochs - 2, 0.001, 0.01, 0.001
            )
            return dict(optimizer=opt, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    loader = DataLoader(TensorDataset(torch.ones(1)), batch_size=1)
    cp = ModelCheckpoint(dirpath=tmp_path / "old", save_last=True, save_top_k=0)
    first = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        logger=False,
        callbacks=[cp],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    first.fit(Toy(), loader)
    payload = torch.load(cp.last_model_path, weights_only=False)
    validate_payload(payload)
    cp2 = ModelCheckpoint(dirpath=tmp_path / "new", save_last=True, save_top_k=0)
    second = pl.Trainer(
        max_epochs=200,
        accelerator="cpu",
        logger=False,
        callbacks=[cp2, TerminalFitCheckpoint()],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    model = Toy()
    keep_original_schedule(model)
    second.fit(model, loader, ckpt_path=cp.last_model_path)
    final = torch.load(cp2.last_model_path, weights_only=False)
    assert final["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"] == 200
    assert final["global_step"] == 200
    assert final["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(0.001)
    assert final["optimizer_states"][0]["state"][0]["step"].item() == 200
    assert final["lr_schedulers"][0]["last_epoch"] == 200
    terminal = torch.load(tmp_path / "new/terminal_fit.ckpt", weights_only=False)
    assert terminal["global_step"] == 200 and "rng_snapshot" in terminal


def test_review_queue_and_configs_are_200_with_no_implicit_stage2_execution():
    import csv

    from molgap.modular.config import load
    from scripts.prepare_qm9_epoch200_review import BACKBONES

    folder = ROOT / "docs/features/qm9-epoch200-modular-study"
    rows = list(csv.DictReader((folder / "training_queue.csv").open()))
    assert len({r["id"] for r in rows}) == len(rows) == 114
    assert sum(r["stage"] == "1" for r in rows) == 33
    assert all(r["epochs"] == "200" and r["command"] for r in rows)
    assert not {"B", "C", "mtl7", "physics_gap7", "mtl10_control"} & {r["variant"] for r in rows}
    assert not any(
        BACKBONES[r["model"]][0] == "transformer" and r["variant"] == "single_task" for r in rows
    )
    plugin_rows = [r for r in rows if r["variant"] in {"task_attention7", "physics_thermo10"}]
    assert len(plugin_rows) == 72 and len({r["model"] for r in plugin_rows}) == 12
    for row in rows:
        cfg, _ = load(ROOT / row["config"])
        assert cfg["training"]["seed"] == cfg["data"]["split"]["seed"] == int(row["seed"])
        assert cfg["training"]["checkpoint_every_n_epochs"] == 50
        assert cfg["training"]["selection_metric"] == row["selection_metric"]
        assert "molgap.modular.train" in row["command"]
    for stage, filename in [("1", "fresh_commands.sh"), ("2", "stage2_commands.sh")]:
        commands = [
            line[len("run_job ") :]
            for line in (folder / filename).read_text().splitlines()
            if line.startswith("run_job ")
        ]
        assert commands == [r["command"] for r in rows if r["stage"] == stage]
