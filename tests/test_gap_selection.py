from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from chemprop.models.utils import load_model

from molgap.config import load_config
from molgap.training.gap_metric import GapMAE, GapMetricScale
from molgap.training.predict import predict
from molgap.training.train import train


def test_gap_mae_ignores_auxiliary_targets_and_uses_sample_count() -> None:
    metric = GapMAE(task_weights=[100, 100, 0])
    metric.gap_scale.fill_(0.1)
    metric.update(torch.tensor([[100.0, -100.0, 1.0], [0.0, 0.0, 3.0]]), torch.zeros(2, 3))
    metric.update(torch.tensor([[float("nan"), float("nan"), 2.0]]), torch.zeros(1, 3))
    assert metric.compute().item() == pytest.approx(0.2)
    assert metric.num_samples.item() == 3


def test_gap_only_mask_and_inverse_scale() -> None:
    metric = GapMAE()
    metric.gap_scale.fill_(0.02)
    metric.update(
        torch.tensor([[20.0, 30.0, 1.0], [20.0, 30.0, 10.0]]),
        torch.zeros(2, 3),
        torch.tensor([[False, False, True], [True, True, False]]),
    )
    assert metric.compute().item() == pytest.approx(0.02)
    # Test labels and predictions are raw Hartree; the callback restores unit scale.
    from types import SimpleNamespace

    GapMetricScale().on_test_epoch_start(None, SimpleNamespace(metrics=[metric]))
    metric.reset()
    metric.update(torch.tensor([[0.02]]), torch.zeros(1, 1))
    assert metric.compute().item() == pytest.approx(0.02)


def test_checkpoint_and_early_stop_follow_gap_instead_of_auxiliary_error(tmp_path: Path) -> None:
    import lightning.pytorch as pl
    from chemprop.nn.transforms import UnscaleTransform
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.predictor = torch.nn.Module()
            self.predictor.output_transform = UnscaleTransform([0, 0, 0], [1, 1, 0.1])
            self.metrics = torch.nn.ModuleList([GapMAE()])

        def training_step(self, batch, batch_idx):
            return self.anchor.square()

        def on_validation_model_eval(self):
            self.eval()
            self.predictor.output_transform.train()

        def validation_step(self, batch, batch_idx):
            # Auxiliary errors improve while gap deteriorates: selected epoch must be 0.
            values = [100.0, 100.0, 0.1] if self.current_epoch == 0 else [0.0, 0.0, 0.2]
            prediction = torch.tensor(values).repeat(len(batch), 1)
            self.metrics[0].update(prediction, torch.zeros_like(prediction))
            self.log("val/gap_mae", self.metrics[0], on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val/gap_mae", mode="min")
    stopping = EarlyStopping(monitor="val/gap_mae", mode="min", patience=1, min_delta=1e-5)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=8,
        num_sanity_val_steps=0,
        callbacks=[GapMetricScale(), checkpoint, stopping],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(Model(), DataLoader(torch.zeros(2, 1)), DataLoader(torch.zeros(3, 1), batch_size=2))
    payload = torch.load(checkpoint.best_model_path, weights_only=False)
    assert payload["epoch"] == 0
    assert float(checkpoint.best_model_score) == pytest.approx(0.01)
    assert stopping.stopped_epoch == 1


def test_validation_physical_diagnostic_uses_hartree_then_test_uses_raw_values() -> None:
    from types import SimpleNamespace

    from chemprop.nn.transforms import UnscaleTransform

    from molgap.models.losses import PhysicsConsistencyMSE, gap_consistency_residual

    means = [-0.3, 0.05, 0.35]
    scales = [0.02, 0.03, 0.04]
    criterion = PhysicsConsistencyMSE([1, 1, 1], 0.1, means, scales).eval()
    transform = UnscaleTransform(means, scales)
    model = SimpleNamespace(
        predictor=SimpleNamespace(output_transform=transform), metrics=[criterion]
    )
    callback = GapMetricScale()
    prediction = torch.tensor([[1.0, 2.0, 3.0]])
    targets = prediction.clone()
    raw = prediction * torch.tensor(scales) + torch.tensor(means)
    expected = 0.1 * gap_consistency_residual(raw).square() / 3
    callback.on_validation_epoch_start(None, model)
    assert criterion._calc_unreduced_loss(prediction, targets).flatten().tolist() == pytest.approx(
        [expected.item()] * 3
    )
    callback.on_test_epoch_start(None, model)
    assert criterion._calc_unreduced_loss(raw, raw).flatten().tolist() == pytest.approx(
        [expected.item()] * 3
    )


@pytest.mark.parametrize("config_name", ["baseline.yaml", "multitask.yaml"])
def test_official_cli_gap_checkpoint_matches_raw_validation_predictions(
    tmp_path: Path, config_name: str
) -> None:
    config, _ = load_config(Path(__file__).resolve().parents[1] / "configs" / config_name)
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='gap-smoke'\nversion='0.0.0'\n")
    processed = project / "data/tiny"
    processed.mkdir(parents=True)
    smiles = ["C", "CC", "CCC", "CO", "CN", "O", "N", "CF", "CCl", "CBr"]
    frame = pd.DataFrame(
        {
            "sample_id": range(10),
            "smiles": smiles,
            "homo": np.linspace(-0.3, -0.5, 10),
            "lumo": np.linspace(0.05, 0.02, 10),
        }
    )
    frame["delta_e"] = frame.lumo - frame.homo
    for split, subset in (("train", frame[:6]), ("val", frame[6:8]), ("test", frame[8:])):
        subset.to_csv(processed / f"{split}.csv", index=False)
    config["data"]["processed_dir"] = "data/tiny"
    config["experiment"]["name"] = "smoke"
    config["training"].update(
        epochs=2, batch_size=2, num_workers=0, accelerator="cpu", devices=1, warmup_epochs=1
    )
    config["model"].update(message_hidden_dim=16, ffn_hidden_dim=16, depth=2, dropout=0.0)
    config["output"]["root"] = "outputs"
    path = project / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    train(path)
    run_dir = project / "outputs/smoke/seed3407"
    checkpoint = next((run_dir / "chemprop").rglob("best*.ckpt"))
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    checkpoint_state = next(
        v for k, v in payload["callbacks"].items() if k.startswith("ModelCheckpoint")
    )
    assert "val/gap_mae" in str(payload["callbacks"])
    assert isinstance(load_model(next((run_dir / "chemprop").rglob("*.pt"))), torch.nn.Module)
    predictions_path = predict(path, input_override="data/tiny/val.csv", output_override="val.csv")
    predictions = pd.read_csv(predictions_path)
    raw_mae = np.abs(frame[6:8].delta_e.to_numpy() - predictions.delta_e.to_numpy()).mean()
    assert raw_mae == pytest.approx(float(checkpoint_state["best_model_score"]), rel=2e-5, abs=1e-7)
