"""End-to-end data lineage, saved predictions and exact epoch-boundary recovery."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from molgap.data.qm9_properties import PROPERTY_UNITS, digest
from molgap.modular.config import PHYSICS
from molgap.modular.data import prepare_thermo_data
from molgap.modular.physics import TASKS7
from molgap.modular.train import train
from scripts.prepare_qm9_epoch200_review import make_executable_config


def project(root):
    (root / "pyproject.toml").write_text('[project]\nname="modular-fixture"\nversion="0.0.0"\n')
    x = np.arange(20, dtype=float)
    f = pd.DataFrame(
        dict(
            sample_id=[f"id{i}" for i in range(20)],
            smiles=["C", "CC", "CO", "O"] * 5,
            homo=-0.5 + x * 0.001,
            lumo=0.1 + x * 0.002,
            alpha=1 + x * 0.2,
            mu=0.2 + x * 0.03,
            r2=2 + x * 0.3,
            zpve=0.01 + x * 0.001,
        )
    )
    f["delta_e"] = f.lumo - f.homo
    original = root / "data/processed/qm9_full/random/seed42"
    original.mkdir(parents=True)
    canonical = f[["sample_id", "smiles", "homo", "lumo", "delta_e"]].copy()
    canonical["source_id"] = [f"gdb_{i}" for i in range(20)]
    canonical.to_csv(original / "dataset.csv", index=False)
    props = root / "data/processed/qm9_single_properties_v1"
    target = props / "random/seed42"
    target.mkdir(parents=True)
    manifest = []
    hashes = {}
    for split, a, b in [("train", 0, 12), ("val", 12, 16), ("test", 16, 20)]:
        subset = f.iloc[a:b].copy()
        subset[["sample_id", "smiles", "homo", "lumo", "delta_e"]].to_csv(
            original / f"{split}.csv", index=False
        )
        subset[["sample_id", "smiles", *TASKS7]].to_csv(target / f"{split}.csv", index=False)
        hashes[f"random/seed42/{split}.csv"] = digest(target / f"{split}.csv")
        manifest.extend(
            dict(sample_id=s, split=split, split_position=j) for j, s in enumerate(subset.sample_id)
        )
    for d in [original, target]:
        pd.DataFrame(manifest).to_csv(d / "split_manifest.csv", index=False)
    identity = dict(
        targets=PROPERTY_UNITS,
        preserved_original_membership_order_and_energy_labels=True,
        extended_files=hashes,
    )
    (props / "data_identity.json").write_text(json.dumps(identity))
    raw = root / "data/raw"
    raw.mkdir()
    energy = pd.DataFrame(
        dict(
            mol_id=canonical.source_id,
            u0=-100 + x * 0.2,
            u298=-100 + x * 0.2 + 0.01,
            h298=-100 + x * 0.2 + 0.01 + PHYSICS["kbt_hartree"],
        )
    )
    energy.to_csv(raw / "qm9_chn.csv", index=False)
    prepare_thermo_data(root, root / "data/processed/qm9_thermo10_v1", seeds=(42,))
    return root


@pytest.mark.parametrize(
    "variant", ["single_task", "D1", "D2", "task_attention7", "physics_thermo10"]
)
def test_continuous_vs_resume_full_state_and_saved_metrics(tmp_path, variant):
    root = project(tmp_path)
    cfg = make_executable_config(
        "chemprop", variant, 42, "delta_e" if variant == "single_task" else None
    )
    cfg["source"]["model"].update(message_hidden_dim=8, ffn_hidden_dim=8, depth=2)
    cfg["module"].update(hidden_dim=8, num_heads=2, dropout=0.1)
    cfg["module"]["physics"].update(warmup_epochs=0, ramp_epochs=1)
    cfg["training"].update(num_workers=0, pin_memory=False)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    full = root / "full"
    split = root / "split"
    train(path, output_dir=full, diagnostic_epochs=4, device_override="cpu")
    train(path, output_dir=split, diagnostic_epochs=2, device_override="cpu")
    checkpoint = split / "checkpoints/last.ckpt"
    train(path, resume=checkpoint, output_dir=split, diagnostic_epochs=4, device_override="cpu")
    a = torch.load(full / "checkpoints/last.ckpt", weights_only=False)
    b = torch.load(split / "checkpoints/last.ckpt", weights_only=False)
    assert a["completed_epochs"] == b["completed_epochs"] == 4
    assert a["global_step"] == b["global_step"] == 4
    assert a["lr_schedulers"] == b["lr_schedulers"]
    assert a["best_score"] == b["best_score"] and a["bad_epochs"] == b["bad_epochs"]
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    for key in a["best_inference_state"]:
        torch.testing.assert_close(
            a["best_inference_state"][key], b["best_inference_state"][key], rtol=0, atol=0
        )
    pd.testing.assert_frame_equal(
        pd.read_csv(full / "test_predictions.csv"), pd.read_csv(split / "test_predictions.csv")
    )
    result = json.loads((split / "test_metrics.json").read_text())
    assert result["diagnostic"] and result["best_validation_mae"] == b["best_score"]
    frame = pd.read_csv(split / "test_predictions.csv", float_precision="round_trip")
    assert result["targets"]["delta_e"]["mae"] == pytest.approx(
        abs(frame.delta_e - frame.true_delta_e).mean()
    )
    scalers = json.loads((split / "scalers.json").read_text())
    assert all(v["fit_split"] == "train" and v["train_rows"] == 12 for v in scalers.values())
    assert len(scalers) == (
        10 if variant == "physics_thermo10" else 1 if variant == "single_task" else 7
    )
    # The saved metadata/scalers/context reject silently changing a resumed experiment.
    bad = copy.deepcopy(cfg)
    bad["source"]["training"]["max_lr"] *= 2
    badpath = root / "bad.yaml"
    badpath.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="context differs"):
        train(
            badpath, resume=checkpoint, output_dir=split, diagnostic_epochs=5, device_override="cpu"
        )

    with pytest.raises(ValueError, match="context differs"):
        train(
            badpath,
            resume=checkpoint,
            output_dir=split,
            diagnostic_epochs=5,
            device_override="cpu",
            dry_run=True,
        )
    if variant == "single_task":
        backups = list((split / "resume_backups").iterdir())
        with pytest.raises(ValueError, match="already reached"):
            train(
                path,
                resume=checkpoint,
                output_dir=split,
                diagnostic_epochs=4,
                device_override="cpu",
            )
        assert list((split / "resume_backups").iterdir()) == backups


def test_thermo_extension_keeps_duplicate_source_mean_and_original_splits(tmp_path):
    root = project(tmp_path)
    f = pd.read_csv(root / "data/processed/qm9_thermo10_v1/random/seed42/train.csv")
    original = pd.read_csv(root / "data/processed/qm9_full/random/seed42/train.csv")
    pd.testing.assert_frame_equal(f[original.columns], original)
    np.testing.assert_allclose(f.h298 - f.u298, PHYSICS["kbt_hartree"], atol=1e-12)
    with pytest.raises(FileExistsError):
        prepare_thermo_data(root, root / "data/processed/qm9_thermo10_v1", seeds=(42,))


def small_config(root, variant="single_task", workers=0):
    cfg = make_executable_config(
        "chemprop", variant, 42, "delta_e" if variant == "single_task" else None
    )
    cfg["source"]["model"].update(message_hidden_dim=8, ffn_hidden_dim=8, depth=2)
    cfg["module"].update(hidden_dim=8, num_heads=2, dropout=0.1)
    cfg["training"].update(num_workers=workers, pin_memory=False)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_epoch50_snapshot_recovers_optimizer_rng_and_historical_best(tmp_path, monkeypatch):
    import molgap.modular.train as engine

    root = project(tmp_path)
    path = small_config(root)
    real_evaluate = engine.evaluate
    counter = [0]

    def controlled(model, loader, frame, tasks, scalers, device):
        pred, values, attention = real_evaluate(model, loader, frame, tasks, scalers, device)
        if frame.sample_id.iloc[0] == "id12":
            counter[0] += 1
            # Controlled validation score tests checkpoint cadence/history only;
            # it is not evidence about molecular prediction accuracy.
            values["delta_e"]["mae"] = 1 / (1 + counter[0]) if counter[0] <= 50 else 1.0
        return pred, values, attention

    monkeypatch.setattr(engine, "evaluate", controlled)
    full = root / "full52"
    interrupted = root / "interrupted50"
    engine.train(path, output_dir=full, diagnostic_epochs=52, device_override="cpu")
    counter[0] = 0
    engine.train(path, output_dir=interrupted, diagnostic_epochs=50, device_override="cpu")
    periodic = interrupted / "checkpoints/epoch0050.ckpt"
    state = torch.load(periodic, weights_only=False)
    assert state["global_step"] == state["completed_epochs"] == 50
    assert state["optimizer_states"] and state["lr_schedulers"] and state["rng_snapshot"]
    # The historical best must remain recoverable even if a later branch's
    # best file is wrong: every full checkpoint embeds its own best weights.
    torch.save(dict(not_a_training_checkpoint=True), interrupted / "checkpoints/best.ckpt")
    engine.train(
        path, resume=periodic, output_dir=interrupted, diagnostic_epochs=52, device_override="cpu"
    )
    a = torch.load(full / "checkpoints/last.ckpt", weights_only=False)
    b = torch.load(interrupted / "checkpoints/last.ckpt", weights_only=False)
    assert a["best_epoch"] == b["best_epoch"] == 50 and a["bad_epochs"] == b["bad_epochs"] == 2
    assert b["global_step"] == 52
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    pd.testing.assert_frame_equal(
        pd.read_csv(full / "test_predictions.csv"),
        pd.read_csv(interrupted / "test_predictions.csv"),
    )
    assert list((interrupted / "resume_backups").glob("*/best.ckpt"))


def test_two_worker_sampler_state_survives_epoch_boundary_resume(tmp_path):
    root = project(tmp_path)
    path = small_config(root, "task_attention7", workers=2)
    full = root / "full"
    partial = root / "partial"
    train(path, output_dir=full, diagnostic_epochs=2, device_override="cpu")
    train(path, output_dir=partial, diagnostic_epochs=1, device_override="cpu")
    train(
        path,
        output_dir=partial,
        resume=partial / "checkpoints/last.ckpt",
        diagnostic_epochs=2,
        device_override="cpu",
    )
    a = torch.load(full / "checkpoints/last.ckpt", weights_only=False)
    b = torch.load(partial / "checkpoints/last.ckpt", weights_only=False)
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)


def test_duplicate_thermo_targets_use_frozen_source_group_mean(tmp_path):
    root = project(tmp_path)
    file = root / "data/processed/qm9_full/random/seed42/dataset.csv"
    canonical = pd.read_csv(file)
    canonical.loc[0, "source_id"] = "gdb_0|gdb_extra"
    canonical.to_csv(file, index=False)
    raw_path = root / "data/raw/qm9_chn.csv"
    raw = pd.read_csv(raw_path)
    extra = raw.iloc[[0]].copy()
    extra["mol_id"] = "gdb_extra"
    extra[["u0", "u298", "h298"]] += 0.4
    pd.concat([raw, extra]).to_csv(raw_path, index=False)
    dest = root / "data/processed/thermo_duplicate"
    prepare_thermo_data(root, dest, seeds=(42,))
    result = pd.read_csv(dest / "random/seed42/train.csv")
    assert result.loc[0, "u0"] == pytest.approx(raw.loc[0, "u0"] + 0.2)
    assert len(result) == 12


def test_thermo_rejects_changed_original_seven_labels(tmp_path):
    from molgap.modular.data import load_labels

    root = project(tmp_path)
    cfg = make_executable_config("chemprop", "physics_thermo10", 42)
    path = root / "data/processed/qm9_single_properties_v1/random/seed42/train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "alpha"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source labels changed"):
        load_labels(cfg, root)


@pytest.mark.parametrize("name", ["smiles_transformer", "spe_transformer"])
def test_sequence_prefix_training_and_resume_restore_saved_tokenizer(tmp_path, monkeypatch, name):
    from types import SimpleNamespace

    import molgap.transformer.dataset as legacy
    from molgap.transformer.spe import SpeSmilesTokenizer
    from molgap.transformer.tokenizer import SmilesTokenizer

    root = project(tmp_path)
    cfg = make_executable_config(name, "task_attention7", 42)
    cfg["source"]["model"].update(d_model=16, num_layers=1, num_heads=4, dim_feedforward=32)
    cfg["module"].update(hidden_dim=8, num_heads=2)
    cfg["training"].update(num_workers=0, pin_memory=False)
    frames = {
        s: pd.read_csv(root / f"data/processed/qm9_full/random/seed42/{s}.csv")
        for s in ["train", "val", "test"]
    }
    fits = []

    def prepare(config, project):
        t = config["tokenizer"]
        if name == "spe_transformer":
            tokenizer = SpeSmilesTokenizer(
                t["pattern"], t["max_length"], t["special_tokens"], num_merges=8, min_frequency=2
            )
        else:
            tokenizer = SmilesTokenizer(t["pattern"], t["max_length"], t["special_tokens"])
        tokenizer.fit(frames["train"].smiles)
        fits.append(True)
        datasets = {}
        for split, frame in frames.items():
            encoded = [tokenizer.encode(s) for s in frame.smiles]
            datasets[split] = legacy.SmilesRegressionDataset(
                torch.tensor([e.input_ids for e in encoded]),
                torch.tensor([e.attention_mask for e in encoded], dtype=torch.bool),
                torch.zeros(len(frame)),
            )
        return SimpleNamespace(tokenizer=tokenizer, frames=frames, datasets=datasets)

    monkeypatch.setattr(legacy, "prepare_transformer_data", prepare)
    monkeypatch.setattr(legacy, "_read_and_validate_splits", lambda *args: (frames, {}))
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = root / "run"
    train(config, output_dir=output, diagnostic_epochs=1, limit_rows=2, device_override="cpu")
    train(
        config,
        resume=output / "checkpoints/last.ckpt",
        output_dir=output,
        diagnostic_epochs=2,
        limit_rows=2,
        device_override="cpu",
    )
    assert len(fits) == 1
    saved = json.loads((output / "vocabulary.json").read_text())
    assert "O" in saved["vocabulary"]  # Train prefix is C,CC; fit used full train.
    assert len(pd.read_csv(output / "test_predictions.csv")) == 2
