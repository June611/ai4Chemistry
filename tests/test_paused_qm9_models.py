"""Verify paused models cannot start through direct or batch training entrypoints."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from molgap.qm9_models import train
from molgap.qm9_models.policy import paused_models

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["cormorant", "lie_transformer"])
def test_direct_training_refuses_paused_model_before_loading(monkeypatch, model):
    def unexpected(*args, **kwargs):
        pytest.fail("Paused models must fail before loading data or building models")

    monkeypatch.setattr(train, "load_data", unexpected)
    monkeypatch.setattr(train, "build_model", unexpected)
    with pytest.raises(RuntimeError, match=f"{model} is paused by user request"):
        train.run(ROOT / f"configs/qm9_models/{model}.yaml")


def test_default_smoke_preparation_excludes_paused_models(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_one_epoch_smoke.py"),
            "--output",
            str(tmp_path),
            "--prepare-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "status.json").read_text())
    assert set(report["paused_models"]) == {"cormorant", "lie_transformer"}
    assert len(report["models"]) == 21
    assert not ({entry["model"] for entry in report["models"]} & paused_models(ROOT).keys())


@pytest.mark.parametrize("model", ["cormorant", "lie_transformer"])
def test_explicit_smoke_selection_is_also_blocked(tmp_path, model):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_one_epoch_smoke.py"),
            "--prepare-only",
            "--output",
            str(tmp_path),
            "--models",
            model,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "paused by user request" in result.stderr
    assert not (tmp_path / "status.json").exists()


def test_formal_shell_entry_skips_paused_models(tmp_path):
    commands = tmp_path / "commands.txt"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == "run python -m molgap.qm9_models.policy" ]]; then\n'
        '  echo "cormorant lie_transformer"\n'
        "else\n"
        '  printf "%s\\n" "$*" >> "$TEST_COMMANDS"\n'
        "fi\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_qm9_models.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "TEST_COMMANDS": str(commands),
            "QM9_PAIRED_SEEDS": "3407",
            "INFOMAX_CHECKPOINT": "",
            "INFOMAX_CHECKPOINT_TEMPLATE": "",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    actual = commands.read_text().splitlines()
    assert len(actual) == 7
    assert not any(
        model in command for model in ("cormorant", "lie_transformer") for command in actual
    )
    assert "Skipping cormorant" in result.stdout
    assert "Skipping lie_transformer" in result.stdout
