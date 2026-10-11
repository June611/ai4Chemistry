import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from molgap.metrics.regression import regression_metrics
from molgap.transformer.model import SmilesTransformerRegressor
from molgap.transformer.tokenizer import SmilesTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(2)
run = Path("outputs/spe_transformer_10epoch/seed3407")
out = Path("reports/spe_transformer/10epoch")
out.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((run / "config.yaml").read_text())
data = Path(config["data"]["processed_dir"])
train = pd.read_csv(data / "train.csv")
test = pd.read_csv(data / "test.csv")
pred = pd.read_csv(run / "predictions/test_predictions.csv")
assert test.sample_id.tolist() == pred.sample_id.tolist()
assert test.smiles.tolist() == pred.smiles.tolist()
assert np.isfinite(pred.delta_e_pred).all()
assert np.allclose(test.delta_e, pred.delta_e, rtol=0, atol=1e-9)
computed = regression_metrics(test.delta_e, pred.delta_e_pred)
metrics = json.loads((run / "metrics.json").read_text())
for key in ("mae", "rmse", "mape", "r2"):
    assert np.isclose(computed[key], metrics[key], atol=1e-8, rtol=0), key
payload = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
tok = SmilesTokenizer.from_dict(payload["tokenizer"])
model = SmilesTransformerRegressor(len(tok.vocabulary), tok.max_length, payload["model_config"])
model.load_state_dict(payload["state_dict"])
model.eval()
indices = np.linspace(0, len(test) - 1, 64, dtype=int)
enc = [tok.encode(s) for s in test.iloc[indices].smiles]
with torch.inference_mode():
    normalized = model(
        torch.tensor([x.input_ids for x in enc]), torch.tensor([x.attention_mask for x in enc])
    )
    scaler = payload["target_scaler"]
    restored = (normalized * scaler["scale_hartree"] + scaler["mean_hartree"]).numpy()
cpu_error = float(np.max(np.abs(restored - pred.iloc[indices].delta_e_pred.to_numpy())))
# Match the original inference device and batch size before testing serialization.
# CPU and CUDA attention kernels have different float32 accumulation behavior.
assert torch.cuda.is_available(), "This audit requires the training CUDA device."
model.cuda()
gpu_predictions = []
with torch.inference_mode():
    for start in range(0, len(test), int(config["training"]["batch_size"])):
        batch = test.iloc[start : start + int(config["training"]["batch_size"])]
        encoded = [tok.encode(s) for s in batch.smiles]
        normalized = model(
            torch.tensor([x.input_ids for x in encoded], device="cuda"),
            torch.tensor([x.attention_mask for x in encoded], device="cuda"),
        )
        gpu_predictions.extend(
            (normalized * scaler["scale_hartree"] + scaler["mean_hartree"]).cpu().tolist()
        )
max_error = float(np.max(np.abs(np.asarray(gpu_predictions) - pred.delta_e_pred.to_numpy())))
assert max_error < 1e-6, max_error
comparisons = []
for name, values in [
    ("Training mean", np.full(len(test), train.delta_e.mean())),
    ("Training median", np.full(len(test), train.delta_e.median())),
    ("SPE + Transformer (10 epochs)", pred.delta_e_pred),
]:
    comparisons.append({"model": name, **regression_metrics(test.delta_e, values)})
pd.DataFrame(comparisons).to_csv(out / "comparison.csv", index=False)
ea = EventAccumulator(str(run / "logs/transformer/version_0"), size_guidance={"scalars": 0})
ea.Reload()
series = {
    tag: [{"step": e.step, "value": e.value} for e in ea.Scalars(tag)]
    for tag in ("train_loss", "val_mae", "val_r2")
}
curves = []
for i, e in enumerate(series["val_mae"]):
    curves.append(
        {
            "epoch": i + 1,
            "step": e["step"],
            "val_mae_hartree": e["value"],
            "train_normalized_mse": series["train_loss"][i]["value"],
            "val_r2": series["val_r2"][i]["value"],
        }
    )
pd.DataFrame(curves).to_csv(out / "learning_curve.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].plot([x["epoch"] for x in curves], [x["train_normalized_mse"] for x in curves], marker="o")
axes[0].set_ylabel("Training normalized MSE")
axes[1].plot([x["epoch"] for x in curves], [x["val_mae_hartree"] for x in curves], marker="o")
axes[1].set_ylabel("Validation MAE (Hartree)")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(out / "learning_curve.png", dpi=180)
plt.close(fig)
summary = {
    "test_rows": len(test),
    "identity_and_finiteness_checks": "passed",
    "recomputed_csv_metrics": computed,
    "portable_prediction_max_absolute_difference_hartree": max_error,
    "portable_prediction_samples": len(test),
    "cpu_vs_original_cuda_max_absolute_difference_hartree_64_samples": cpu_error,
    "comparisons": comparisons,
    "learning_curves": curves,
    "fit": json.loads((run / "training_timing.json").read_text()),
    "best_validation_mae_hartree": metrics["best_validation_mae"],
    "best_checkpoint": metrics["best_checkpoint"],
    "mae_reduction_vs_training_mean_percent": 100 * (1 - computed["mae"] / comparisons[0]["mae"]),
}
(out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "learning_curves"}, indent=2))
