import io
from pathlib import Path

import pytest
import torch
import yaml
from SmilesPE.tokenizer import SPE_Tokenizer
from test_transformer import PATTERN, SPECIALS, _write_split_fixture

from molgap.config import ConfigError
from molgap.transformer.config import validate_transformer_config
from molgap.transformer.dataset import prepare_transformer_data
from molgap.transformer.model import SmilesTransformerRegressor
from molgap.transformer.spe import SpeSmilesTokenizer
from molgap.transformer.tokenizer import SmilesTokenizer, TokenizationError


def test_official_merges_roundtrip_and_unseen_context() -> None:
    tokenizer = SpeSmilesTokenizer(PATTERN, 20, SPECIALS, num_merges=10, min_frequency=2)
    tokenizer.fit(["CCCC", "CCCO", "CCCN", "CCO", "CCN"])
    assert tokenizer.merge_codes
    official = SPE_Tokenizer(io.StringIO(tokenizer.merge_codes))
    restored = SmilesTokenizer.from_dict(tokenizer.to_dict())
    for smiles in ["CCCC", "NCCO", "C", "CCN", "ClC"]:
        assert tokenizer.tokenize(smiles) == official.tokenize(smiles).split()
        assert "".join(tokenizer.tokenize(smiles)) == smiles
        assert restored.encode(smiles) == tokenizer.encode(smiles)
    assert tokenizer.encode("ClC").unknown_count == 1
    assert tokenizer.encode("C").unknown_count == 0
    with pytest.raises(TokenizationError):
        tokenizer.encode("C_C")
    with pytest.raises(TokenizationError, match="exceeding"):
        tokenizer.encode("N" * 21)


def test_single_atom_training_and_repeated_fit() -> None:
    tokenizer = SpeSmilesTokenizer(PATTERN, 10, SPECIALS, min_frequency=2)
    tokenizer.fit(["CCCC", "CCCC"])
    tokenizer.fit(["C", "O"])
    assert tokenizer.merge_codes == ""
    assert tokenizer.tokenize("CC") == ["C", "C"]
    assert set(tokenizer.vocabulary) == {*SPECIALS.values(), "C", "O"}


def test_spe_model_optimizer_and_portable_prediction_roundtrip(tmp_path: Path) -> None:
    tokenizer = SpeSmilesTokenizer(PATTERN, 12, SPECIALS, min_frequency=2)
    smiles = ["CCO", "CCN", "CCCC", "CCCO"]
    tokenizer.fit(smiles)
    model_config = {
        "d_model": 16, "num_layers": 1, "num_heads": 4,
        "dim_feedforward": 32, "dropout": 0.0, "activation": "gelu",
        "pooling": "cls", "regression_hidden_dim": 8,
    }
    model = SmilesTransformerRegressor(len(tokenizer.vocabulary), 12, model_config)
    encoded = [tokenizer.encode(value) for value in smiles]
    ids = torch.tensor([value.input_ids for value in encoded])
    masks = torch.tensor([value.attention_mask for value in encoded])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.regression_head[-1].weight.detach().clone()
    loss = (model(ids, masks) - torch.tensor([0.1, 0.2, 0.3, 0.4])).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.regression_head[-1].weight)
    path = tmp_path / "model.pt"
    torch.save({"tokenizer": tokenizer.to_dict(), "state_dict": model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    restored_tokenizer = SmilesTokenizer.from_dict(payload["tokenizer"])
    restored = SmilesTransformerRegressor(len(restored_tokenizer.vocabulary), 12, model_config)
    restored.load_state_dict(payload["state_dict"])
    model.eval()
    restored.eval()
    restored_encoded = [restored_tokenizer.encode(value) for value in smiles]
    with torch.inference_mode():
        actual = restored(torch.tensor([value.input_ids for value in restored_encoded]), masks)
        torch.testing.assert_close(actual, model(ids, masks))


def test_spe_validation_and_test_cannot_change_vocabulary(tmp_path: Path) -> None:
    config, root = _write_split_fixture(tmp_path)
    config["tokenizer"].update(type="spe", num_merges=10, min_frequency=2)
    prepared = prepare_transformer_data(config, root)
    assert "O" not in prepared.tokenizer.vocabulary
    assert "N" not in prepared.tokenizer.vocabulary
    assert prepared.tokenizer_report["splits"]["val"]["unknown_tokens"] == 1
    assert prepared.tokenizer_report["splits"]["test"]["unknown_tokens"] == 1
    assert prepared.split_identity["verified_same_split_as_chemprop"]


def test_spe_config_shares_architecture_and_training_contract() -> None:
    root = Path(__file__).parents[1]
    base = yaml.safe_load((root / "configs/transformer.yaml").read_text())
    spe = yaml.safe_load((root / "configs/spe_transformer.yaml").read_text())
    validate_transformer_config(spe)
    for section in ("model", "training", "data", "evaluation"):
        assert spe[section] == base[section]
    spe["tokenizer"]["type"] = "typo"
    with pytest.raises(ConfigError, match="tokenizer.type"):
        validate_transformer_config(spe)
