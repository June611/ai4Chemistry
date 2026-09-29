from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from molgap.training.transformer import TransformerLightningModule
from molgap.transformer.dataset import SmilesRegressionDataset, prepare_transformer_data
from molgap.transformer.model import SmilesTransformerRegressor, count_trainable_parameters
from molgap.transformer.tokenizer import SmilesTokenizer, TokenizationError

PATTERN = (
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|"
    r"\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
SPECIALS = {"pad": "[PAD]", "unk": "[UNK]", "cls": "[CLS]"}


def test_tokenizer_examples_full_coverage_and_padding() -> None:
    tokenizer = SmilesTokenizer(PATTERN, max_length=10, special_tokens=SPECIALS)
    assert tokenizer.tokenize("C=C") == ["C", "=", "C"]
    assert tokenizer.tokenize("c1cc[nH]c1") == ["c", "1", "c", "c", "[nH]", "c", "1"]
    tokenizer.fit(["C=C", "O"])

    encoded = tokenizer.encode("C=C")
    assert encoded.input_ids[0] == tokenizer.cls_id
    assert sum(encoded.attention_mask) == 4
    assert encoded.input_ids[-1] == tokenizer.pad_id


def test_tokenizer_rejects_unmatched_and_overlength_smiles() -> None:
    tokenizer = SmilesTokenizer(PATTERN, max_length=3, special_tokens=SPECIALS)
    tokenizer.fit(["C"])
    with pytest.raises(TokenizationError, match="Unmatched SMILES fragment"):
        tokenizer.tokenize("C_C")
    with pytest.raises(TokenizationError, match="exceeding max_length"):
        tokenizer.encode("CCC")


def _write_split_fixture(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "project"
    processed = root / "data" / "processed" / "tiny" / "random" / "seed7"
    processed.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='tiny-transformer'\nversion='0.0.0'\n", encoding="utf-8"
    )
    split = {
        "method": "random",
        "sizes": [0.5, 0.25, 0.25],
        "seed": 7,
        "stratify": {"strategy": "quantile", "column": "delta_e", "bins": 2},
    }
    baseline = {
        "data": {
            "processed_dir": "data/processed/tiny/random/seed7",
            "id_column": "sample_id",
            "smiles_column": "smiles",
            "targets": ["delta_e"],
            "split": split,
        }
    }
    (root / "configs" / "baseline.yaml").write_text(yaml.safe_dump(baseline), encoding="utf-8")
    rows = [
        ("a", "C", 0.1, "train", 0, 0),
        ("b", "CC", 0.2, "train", 1, 1),
        ("c", "O", 0.3, "val", 0, 0),
        ("d", "N", 0.4, "test", 0, 1),
    ]
    frame = pd.DataFrame(
        rows, columns=["sample_id", "smiles", "delta_e", "split", "split_position", "bin"]
    )
    for name in ("train", "val", "test"):
        frame.loc[frame["split"] == name, ["sample_id", "smiles", "delta_e"]].to_csv(
            processed / f"{name}.csv", index=False
        )
    manifest = frame[["sample_id", "smiles", "split", "split_position", "bin"]].rename(
        columns={"bin": "stratify_bin"}
    )
    manifest["split_method"] = "random"
    manifest["split_strategy"] = "quantile_stratified"
    manifest["seed"] = 7
    manifest["stratify_column"] = "delta_e"
    manifest["stratify_bins_requested"] = 2
    manifest.to_csv(processed / "split_manifest.csv", index=False)
    config = {
        "data": {
            "processed_dir": "data/processed/tiny/random/seed7",
            "chemprop_reference_config": "configs/baseline.yaml",
            "id_column": "sample_id",
            "smiles_column": "smiles",
            "target_column": "delta_e",
            "split": split,
        },
        "tokenizer": {
            "pattern": PATTERN,
            "max_length": 10,
            "special_tokens": SPECIALS,
        },
    }
    return config, root


def test_transformer_reuses_manifest_membership_and_order(tmp_path: Path) -> None:
    config, root = _write_split_fixture(tmp_path)
    prepared = prepare_transformer_data(config, root)

    assert prepared.split_identity["verified_same_split_as_chemprop"] is True
    assert prepared.split_identity["row_order_matches_manifest"] is True
    assert prepared.frames["train"]["sample_id"].tolist() == ["a", "b"]
    assert prepared.tokenizer_report["target_scaler"]["fit_split"] == "train"

    train_path = root / config["data"]["processed_dir"] / "train.csv"
    pd.read_csv(train_path).iloc[::-1].to_csv(train_path, index=False)
    with pytest.raises(ValueError, match="membership/order"):
        prepare_transformer_data(config, root)


def test_default_transformer_shape_and_parameter_budget() -> None:
    model_config = {
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "activation": "gelu",
        "pooling": "cls",
        "regression_hidden_dim": 128,
    }
    model = SmilesTransformerRegressor(vocab_size=33, max_length=40, config=model_config)
    result = model(torch.ones((3, 40), dtype=torch.long), torch.ones((3, 40), dtype=torch.bool))

    assert result.shape == (3,)
    assert 1_000_000 <= count_trainable_parameters(model) <= 5_000_000


def test_tiny_cpu_train_and_predict_smoke() -> None:
    torch.manual_seed(7)
    input_ids = torch.randint(1, 8, (12, 6))
    masks = torch.ones((12, 6), dtype=torch.bool)
    targets = torch.linspace(0.1, 0.4, 12)
    dataset = SmilesRegressionDataset(input_ids, masks, targets)
    loader = DataLoader(dataset, batch_size=4)
    model = SmilesTransformerRegressor(
        vocab_size=8,
        max_length=6,
        config={
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 4,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "activation": "gelu",
            "pooling": "cls",
            "regression_hidden_dim": 8,
        },
    )
    module = TransformerLightningModule(
        model,
        target_mean=float(targets.mean()),
        target_scale=float(targets.std(unbiased=False)),
        training_config={"lr": 3e-4, "weight_decay": 0.0, "warmup_ratio": 0.1, "min_lr_ratio": 0.1},
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

    assert predictions.shape == (12,)
    assert torch.isfinite(predictions).all()
