import json

import pytest
import torch
import yaml
from test_modular_pipeline import project

from molgap.modular.train import archive_outputs, train
from scripts.prepare_qm9_epoch200_review import make_executable_config


def test_terminal_finalize_without_optimizer_step(tmp_path, monkeypatch):
    root = project(tmp_path)
    cfg = make_executable_config("chemprop", "single_task", 42, "delta_e")
    cfg["source"]["model"].update(message_hidden_dim=8, ffn_hidden_dim=8, depth=2)
    cfg["training"].update(num_workers=0, pin_memory=False)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = root / "run"
    expected = train(path, output_dir=run, diagnostic_epochs=1, device_override="cpu")
    checkpoint = run / "checkpoints/last.ckpt"
    original = torch.load(checkpoint, weights_only=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Finalization must not take optimizer steps")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    actual = train(
        path,
        resume=checkpoint,
        output_dir=run,
        diagnostic_epochs=1,
        device_override="cpu",
        finalize_only=True,
    )
    assert actual == expected
    assert json.loads((run / "COMPLETED.json").read_text())["epochs_completed"] == 1
    latest = torch.load(run / "checkpoints/terminal_fit.ckpt", weights_only=False)
    assert latest["global_step"] == original["global_step"]
    assert list((run / "resume_backups").glob("*/COMPLETED.json"))


def test_archive_invalidates_completion(tmp_path):
    (tmp_path / "COMPLETED.json").write_text("{}")
    (tmp_path / "test_metrics.json").write_text("{}")
    backup = archive_outputs(tmp_path)
    assert not (tmp_path / "COMPLETED.json").exists()
    assert not (tmp_path / "test_metrics.json").exists()
    assert (backup / "COMPLETED.json").exists()


def test_archive_rejects_external_symlinks(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "COMPLETED.json").symlink_to(tmp_path / "external.json")
    with pytest.raises(ValueError, match="escapes"):
        archive_outputs(run)
