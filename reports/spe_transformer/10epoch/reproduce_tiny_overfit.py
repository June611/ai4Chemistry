import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from molgap.transformer.model import SmilesTransformerRegressor
from molgap.transformer.tokenizer import SmilesTokenizer

torch.set_num_threads(2)
torch.manual_seed(3407)
run = Path("outputs/spe_transformer_10epoch/seed3407")
config = yaml.safe_load((run / "config.yaml").read_text())
tokenizer = SmilesTokenizer.from_dict(json.loads((run / "vocabulary.json").read_text()))
frame = pd.read_csv(Path(config["data"]["processed_dir"]) / "train.csv").sample(
    n=16, random_state=3407
)
encoded = [tokenizer.encode(s) for s in frame.smiles]
ids = torch.tensor([e.input_ids for e in encoded])
masks = torch.tensor([e.attention_mask for e in encoded])
scaler = json.loads((run / "tokenizer_report.json").read_text())["target_scaler"]
y = (torch.tensor(frame.delta_e.to_numpy(), dtype=torch.float32) - scaler["mean_hartree"]) / scaler[
    "scale_hartree"
]
model_config = dict(config["model"])
model_config["dropout"] = 0.0
model = SmilesTransformerRegressor(len(tokenizer.vocabulary), tokenizer.max_length, model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
losses = []
for step in range(201):
    optimizer.zero_grad()
    loss = (model(ids, masks) - y).square().mean()
    losses.append(float(loss.detach()))
    if step < 200:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
report = {
    "purpose": "Capacity/gradient sanity check on 16 training rows only, not a test score",
    "device": "cpu",
    "seed": 3407,
    "steps": 200,
    "diagnostic_overrides": {"dropout": 0.0, "weight_decay": 0.0, "schedule": "constant lr 3e-4"},
    "sample_ids": frame.sample_id.tolist(),
    "initial_normalized_mse": losses[0],
    "final_normalized_mse": losses[-1],
    "final_to_initial_ratio": losses[-1] / losses[0],
    "loss_history": losses,
}
p = Path("reports/spe_transformer/10epoch")
p.mkdir(parents=True, exist_ok=True)
(p / "tiny_overfit.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k not in ("loss_history", "sample_ids")})
