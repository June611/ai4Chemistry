from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from lightning.pytorch import Trainer
from test_transformer import _write_split_fixture
from torch.utils.data import DataLoader

from molgap.config import ConfigError
from molgap.training.spe_multitask import (
    SpeMultiTaskLightningModule,
    gap_consistency_residual,
)
from molgap.transformer.dataset import SmilesRegressionDataset, prepare_multitask_transformer_data
from molgap.transformer.multitask import SpeAffinityMultiTaskRegressor
from molgap.transformer.multitask_config import load_spe_multitask_config


def _model_config(use_affinity: bool) -> dict:
    return {
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "activation": "gelu",
        "pooling": "cls",
        "regression_hidden_dim": 8,
        "use_affinity": use_affinity,
        "affinity_temperature": 1.0,
        "affinity_identity_bias": 2.0,
    }


def _multitask_fixture(tmp_path: Path) -> tuple[dict, Path]:
    config, root = _write_split_fixture(tmp_path)
    processed = root / config["data"]["processed_dir"]
    for name in ("train", "val", "test"):
        path = processed / f"{name}.csv"
        frame = pd.read_csv(path)
        frame["homo"] = [-0.2 - 0.01 * index for index in range(len(frame))]
        frame["lumo"] = frame["homo"] + frame["delta_e"]
        frame.to_csv(path, index=False)
    reference_path = root / "configs" / "baseline.yaml"
    reference = yaml.safe_load(reference_path.read_text())
    reference["data"]["targets"] = ["homo", "lumo", "delta_e"]
    reference_path.write_text(yaml.safe_dump(reference))
    config["data"].update(
        target_columns=["homo", "lumo", "delta_e"],
        targets=["homo", "lumo", "delta_e"],
    )
    config["tokenizer"].update(type="spe", num_merges=10, min_frequency=2)
    return config, root


def test_affinity_matrix_is_row_normalized_trainable_and_portable(tmp_path: Path) -> None:
    config = _model_config(True)
    model = SpeAffinityMultiTaskRegressor(12, 8, config)
    ids = torch.randint(1, 12, (5, 8))
    mask = torch.ones_like(ids, dtype=torch.bool)
    prediction = model(ids, mask)
    assert prediction.shape == (5, 3)
    matrix = model.affinity_matrix()
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(3))
    assert torch.all(matrix >= 0)
    prediction.square().mean().backward()
    assert model.affinity_logits.grad is not None
    assert torch.isfinite(model.affinity_logits.grad).all()
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "state_dict": model.state_dict()}, path)
    restored = SpeAffinityMultiTaskRegressor(12, 8, config)
    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, mask), model(ids, mask))


def test_disabled_affinity_is_exact_identity() -> None:
    model = SpeAffinityMultiTaskRegressor(12, 8, _model_config(False))
    torch.testing.assert_close(model.affinity_matrix(), torch.eye(3))
    assert "affinity_logits" not in dict(model.named_parameters())


def test_physics_residual_uses_delta_e_equals_lumo_minus_homo() -> None:
    consistent = torch.tensor([[-0.3, 0.2, 0.5], [-0.1, 0.1, 0.2]])
    torch.testing.assert_close(gap_consistency_residual(consistent), torch.zeros(2))
    inconsistent = consistent.clone()
    inconsistent[:, 2] += 0.05
    torch.testing.assert_close(gap_consistency_residual(inconsistent), torch.full((2,), 0.05))
    with pytest.raises(ValueError, match="shape"):
        gap_consistency_residual(torch.zeros(2, 2))


def test_loss_switches_consistency_term_without_changing_supervision() -> None:
    model = SpeAffinityMultiTaskRegressor(12, 8, _model_config(False))
    common = {
        "model": model,
        "target_means": (0.0, 0.0, 0.0),
        "target_scales": (1.0, 1.0, 1.0),
        "training_config": {
            "lr": 3e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
        },
    }
    no_consistency = SpeMultiTaskLightningModule(
        **common,
        loss_config={
            "task_weights": {"homo": 1.0, "lumo": 1.0, "delta_e": 1.0},
            "consistency_weight": 0.0,
        },
    )
    with_consistency = SpeMultiTaskLightningModule(
        **common,
        loss_config={
            "task_weights": {"homo": 1.0, "lumo": 1.0, "delta_e": 1.0},
            "consistency_weight": 0.5,
        },
    )
    batch = {
        "input_ids": torch.randint(1, 12, (4, 8)),
        "attention_mask": torch.ones((4, 8), dtype=torch.bool),
        "target": torch.randn(4, 3),
    }
    total0, supervised0, consistency0 = no_consistency._losses(batch)
    total1, supervised1, consistency1 = with_consistency._losses(batch)
    torch.testing.assert_close(supervised0, supervised1)
    torch.testing.assert_close(consistency0, consistency1)
    torch.testing.assert_close(total0, supervised0)
    torch.testing.assert_close(total1, supervised1 + 0.5 * consistency1)


def test_tiny_multitask_train_and_predict_smoke() -> None:
    torch.manual_seed(7)
    count = 12
    homo = torch.linspace(-0.4, -0.2, count)
    lumo = torch.linspace(0.05, 0.15, count)
    targets = torch.stack((homo, lumo, lumo - homo), dim=1)
    dataset = SmilesRegressionDataset(
        torch.randint(1, 12, (count, 8)),
        torch.ones((count, 8), dtype=torch.bool),
        targets,
    )
    loader = DataLoader(dataset, batch_size=4)
    model = SpeAffinityMultiTaskRegressor(12, 8, _model_config(True))
    module = SpeMultiTaskLightningModule(
        model,
        tuple(float(value) for value in targets.mean(dim=0)),
        tuple(float(value) for value in targets.std(dim=0, unbiased=False)),
        {"lr": 3e-4, "weight_decay": 0.0, "warmup_ratio": 0.1, "min_lr_ratio": 0.1},
        {
            "task_weights": {"homo": 1.0, "lumo": 1.0, "delta_e": 1.0},
            "consistency_weight": 0.1,
        },
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(module, loader, loader)
    predictions = torch.cat(trainer.predict(module, loader))
    assert predictions.shape == (count, 3)
    assert torch.isfinite(predictions).all()


def test_multitask_data_uses_train_only_scalers(tmp_path: Path) -> None:
    config, root = _multitask_fixture(tmp_path)
    prepared = prepare_multitask_transformer_data(config, root)
    train = prepared.frames["train"][["homo", "lumo", "delta_e"]]
    assert prepared.target_means == pytest.approx(tuple(train.mean()))
    assert prepared.target_scales == pytest.approx(tuple(train.std(ddof=0)))
    assert prepared.datasets["train"][0]["target"].shape == (3,)
    assert prepared.tokenizer_report["target_scaler"]["fit_split"] == "train"


def test_all_ablation_configs_are_valid_and_distinct() -> None:
    root = Path(__file__).parents[1]
    expected = {
        "spe_multitask.yaml": (False, 0.0),
        "spe_multitask_affinity.yaml": (True, 0.0),
        "spe_multitask_consistency.yaml": (False, 0.1),
        "spe_multitask_affinity_consistency.yaml": (True, 0.1),
    }
    for name, setting in expected.items():
        config, _ = load_spe_multitask_config(root / "configs" / name)
        assert (config["model"]["use_affinity"], config["loss"]["consistency_weight"]) == setting
    config, _ = load_spe_multitask_config(root / "configs" / "spe_multitask.yaml")
    config["data"]["target_columns"] = ["lumo", "homo", "delta_e"]
    from molgap.transformer.multitask_config import validate_spe_multitask_config

    with pytest.raises(ConfigError, match="target_columns"):
        validate_spe_multitask_config(config)
