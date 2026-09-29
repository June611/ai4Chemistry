from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_four_primary_configs_share_training_budget_and_fixed_split() -> None:
    configs = [
        _load("baseline.yaml"),
        _load("multitask.yaml"),
        _load("consistency.yaml"),
        _load("transformer.yaml"),
    ]

    assert {config["data"]["processed_dir"] for config in configs} == {
        "data/processed/qm9_full/random/seed3407"
    }
    assert {config["data"]["split"]["seed"] for config in configs} == {3407}
    assert {config["training"]["batch_size"] for config in configs} == {256}
    assert {config["training"]["epochs"] for config in configs} == {100}
    assert {config["training"]["num_workers"] for config in configs} == {8}

    chemprop = configs[:3]
    assert {config["training"]["patience"] for config in chemprop} == {20}
    assert {
        (
            config["training"]["init_lr"],
            config["training"]["max_lr"],
            config["training"]["final_lr"],
        )
        for config in chemprop
    } == {(0.0004, 0.004, 0.0004)}

    transformer = configs[3]
    assert transformer["training"]["early_stopping_patience"] == 20
    assert transformer["training"]["lr"] == 0.0003
    assert transformer["training"]["precision"] == "32-true"
    assert transformer["training"]["pin_memory"] is True


def test_chemprop_cli_configs_expose_throughput_values_in_commands() -> None:
    from molgap.config import load_config
    from molgap.training.commands import build_train_command

    for name in ("baseline.yaml", "multitask.yaml"):
        config, root = load_config(ROOT / "configs" / name)
        command = build_train_command(config, root)
        assert command[command.index("--batch-size") + 1] == "256"
        assert command[command.index("--num-workers") + 1] == "8"
        assert command[command.index("--patience") + 1] == "20"
        assert command[command.index("--max-lr") + 1] == "0.004"
