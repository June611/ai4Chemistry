from pathlib import Path

from molgap.config import load_config
from molgap.training.commands import build_train_command


def test_baseline_command_uses_explicit_splits_and_seeds() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "baseline.yaml"
    config, root = load_config(config_path)

    command = build_train_command(config, root)

    assert Path(command[0]).name == "chemprop"
    assert command[1] == "train"
    assert "--target-columns" in command
    assert "delta_e" in command
    assert command[command.index("--data-seed") + 1] == "3407"
    assert command[command.index("--pytorch-seed") + 1] == "3407"
    data_index = command.index("--data-path")
    assert command[data_index + 1].endswith("train.csv")
    assert command[data_index + 2].endswith("val.csv")
    assert command[data_index + 3].endswith("test.csv")


def test_multitask_command_uses_equal_weights_and_all_targets() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "multitask.yaml"
    config, root = load_config(config_path)

    command = build_train_command(config, root)

    target_index = command.index("--target-columns")
    assert command[target_index + 1 : target_index + 4] == ["homo", "lumo", "delta_e"]
    weight_index = command.index("--task-weights")
    assert command[weight_index + 1 : weight_index + 4] == ["1.0", "1.0", "1.0"]
    assert "--show-individual-scores" in command
