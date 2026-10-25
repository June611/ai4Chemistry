"""Property lineage, scaling, inverse evaluation, persisted plans and failure isolation."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from molgap.config import validate_config as validate_chemprop
from molgap.data.neural_properties import (
    scaler_record,
    validate_property_data,
)
from molgap.data.qm9_properties import NEW_TARGETS, PROPERTY_UNITS, SEEDS, digest
from molgap.experiments.neural_properties import MODELS, fill_latex, prepare, retry
from molgap.qm9_models.data import GeometryDataset
from molgap.scheduling.queue import check_config, run_queue
from molgap.tracking.records import read_json, write_json
from molgap.training.commands import build_train_command
from molgap.training.transformer import TransformerLightningModule
from molgap.transformer.dataset import prepare_transformer_data
from molgap.transformer.model import SmilesTransformerRegressor

ROOT = Path(__file__).parents[1]


@pytest.fixture
def property_fixture(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n")
    manifest = yaml.safe_load((ROOT / "configs/qm9_six_properties/tasks.yaml").read_text())
    manifest["data_root"] = "data/processed/properties"
    for name, item in manifest["models"].items():
        item["config"] = str(ROOT / "configs/qm9_six_properties/models" / f"{name}.yaml")
    manifest["pretraining"]["config"] = str(ROOT / manifest["pretraining"]["config"])
    ids = [f"s{i}" for i in range(8)]
    smiles = ["CC", "CO", "CN", "CCC", "CCO", "CCN", "CCCO", "CCCN"]
    frame = pd.DataFrame(dict(sample_id=ids, smiles=smiles, delta_e=np.arange(8) * 0.01 + 0.1))
    for k, target in enumerate(NEW_TARGETS):
        frame[target] = (np.arange(8) + 1.0) * (k + 1)
    extended_files, source_files = {}, {}
    data_root = root / manifest["data_root"]
    for seed in SEEDS:
        out = data_root / "random" / f"seed{seed}"
        out.mkdir(parents=True)
        splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
        positions = [0, 1, 2, 3, 0, 1, 0, 1]
        frozen = pd.DataFrame(
            dict(
                sample_id=ids,
                smiles=smiles,
                split=splits,
                split_position=positions,
                split_method="random",
                split_strategy="quantile_stratified",
                seed=seed,
                stratify_column="delta_e",
                stratify_bins_requested=200,
            )
        )
        frozen.to_csv(out / "split_manifest.csv", index=False)
        source_files[f"data/processed/qm9_full/random/seed{seed}/split_manifest.csv"] = digest(
            out / "split_manifest.csv"
        )
        for split in ("train", "val", "test"):
            frame.loc[np.array(splits) == split].to_csv(out / f"{split}.csv", index=False)
            relative = f"random/seed{seed}/{split}.csv"
            extended_files[relative] = digest(data_root / relative)
    write_json(
        data_root / "data_identity.json",
        dict(
            targets=PROPERTY_UNITS,
            preserved_original_membership_order_and_energy_labels=True,
            source_files=source_files,
            extended_files=extended_files,
        ),
    )
    tasks = root / "tasks.yaml"
    tasks.write_text(yaml.safe_dump(manifest))
    return root, tasks, frame


def _config(root, tasks, model, target="alpha"):
    plan = prepare(
        root, tasks, root / f"batch_{model}_{target}", models=[model], targets=[target], seeds=[42]
    )
    jobs = read_json(Path(plan["lanes"][0]) / "plan.json")["jobs"]
    job = next(j for j in jobs if j["kind"] != "pretrain")
    return yaml.safe_load(Path(job["config"]).read_text()), job


def test_full_180_matrix_is_persisted_frozen_and_does_not_launch(property_fixture):
    root, tasks, _ = property_fixture
    plan = prepare(root, tasks, root / "batch", gpus=[0, 1])
    jobs = [j for lane in plan["lanes"] for j in read_json(Path(lane) / "plan.json")["jobs"]]
    downstream = [j for j in jobs if j["kind"] != "pretrain"]
    assert len(downstream) == len({j["id"] for j in downstream}) == 180
    assert plan["pretraining_jobs"] == 3
    assert {(j["model"], j["target"], j["seed"]) for j in downstream} == {
        (m, t, s) for m in MODELS for t in NEW_TARGETS for s in SEEDS
    }
    for job in downstream:
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        assert cfg["training"]["batch_size"] == 256
        assert cfg["training"]["epochs"] == 100
        assert cfg["training"]["seed"] == cfg["data"]["split"]["seed"]
        assert cfg["training"]["target_scaling"] == "standard"
        assert not Path(job["run_dir"]).exists()
        assert job["status"] == "pending"
    path = Path(downstream[0]["config"])
    path.write_text(path.read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="Frozen configuration changed"):
        check_config(downstream[0])


@pytest.mark.parametrize("target", NEW_TARGETS)
@pytest.mark.parametrize("model", ["smiles_transformer", "spe_transformer"])
def test_each_tokenizer_property_uses_only_training_scaler(property_fixture, model, target):
    root, tasks, frame = property_fixture
    cfg, _ = _config(root, tasks, model, target)
    prepared = prepare_transformer_data(cfg, root)
    train = frame[target].iloc[:4]
    assert prepared.target_mean == pytest.approx(train.mean())
    assert prepared.target_scale == pytest.approx(train.std(ddof=0))
    assert prepared.target_mean != pytest.approx(frame[target].mean())
    assert prepared.tokenizer_report["target_scaler"]["unit"] == PROPERTY_UNITS[target]
    assert prepared.datasets["test"].targets.tolist() == frame[target].iloc[6:].tolist()


@pytest.mark.parametrize("target", NEW_TARGETS)
def test_chemprop_tracks_original_unit_property_mae_and_rejects_wrong_units(
    property_fixture, target
):
    root, tasks, _ = property_fixture
    cfg, _ = _config(root, tasks, "chemprop_single_gap", target)
    command = build_train_command(cfg, root)
    assert command[command.index("--tracking-metric") + 1] == "property_mae"
    assert command[command.index("--task-weights") + 1] == "1.0"
    cfg["data"]["label_unit"] = "wrong-unit"
    with pytest.raises(ValueError, match="label_unit"):
        validate_chemprop(cfg)


def test_geometry_reads_requested_property_without_changing_positions():
    f = pd.DataFrame(dict(sample_id=["x"], smiles=["CC"], delta_e=[0.2], mu=[3.4]))
    arrays = dict(
        offsets=np.array([0, 2]),
        atomic_numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
    )
    d = GeometryDataset(f, arrays, {"x": 0}, "mu")
    assert d[0].target == 3.4
    assert np.array_equal(d[0].positions.numpy(), arrays["positions"])


def test_modified_augmented_labels_fail_integrity_check(property_fixture):
    root, tasks, _ = property_fixture
    cfg, _ = _config(root, tasks, "smiles_transformer")
    p = root / cfg["data"]["processed_dir"] / "test.csv"
    p.write_text(p.read_text().replace("7.0", "70.0"))
    with pytest.raises(ValueError, match="data changed"):
        validate_property_data(cfg, root)


def test_transformer_forward_restores_physical_units_and_checkpoint_buffers(tmp_path):
    from torch import nn

    torch.set_num_threads(2)

    class Constant(nn.Module):
        def forward(self, ids, mask):
            return torch.full((len(ids),), 2.0)

    module = TransformerLightningModule(Constant(), 10.0, 3.0, {})
    assert module(torch.ones(2, 2), torch.ones(2, 2)).tolist() == [16.0, 16.0]
    torch.save(module.state_dict(), tmp_path / "weights.pt")
    clone = TransformerLightningModule(Constant(), 0.0, 1.0, {})
    clone.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    assert clone(torch.ones(2, 2), torch.ones(2, 2)).tolist() == [16.0, 16.0]
    assert scaler_record([1.0, 3.0], "mu", "debye")["scale"] == 1.0


def test_property_mae_scales_residual_without_offset():
    from molgap.training.gap_metric import PropertyMAE

    metric = PropertyMAE()
    metric.gap_scale.fill_(3.0)
    metric.update(torch.tensor([[2.0], [4.0]]), torch.tensor([[1.0], [2.0]]))
    assert float(metric.compute()) == pytest.approx(4.5)


def test_existing_gap_target_restriction_stays_strict():
    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    cfg["data"]["targets"] = ["alpha"]
    with pytest.raises(ValueError):
        validate_chemprop(cfg)


def test_failed_task_continues_next_and_restart_skips_complete(tmp_path, monkeypatch):
    import molgap.scheduling.queue as queue

    directory = tmp_path / "queue"
    directory.mkdir()
    jobs = []
    for k in ("bad", "good"):
        jobs.append(
            dict(
                id=k,
                model=k,
                kind="ml",
                run_dir=str(tmp_path / k),
                commands=[],
                status="pending",
                gpu_mib=0,
                ram_mib=1,
                cpu_slots=1,
                exclusive=False,
                priority=1,
                depends_on=[],
                local_property_report=True,
            )
        )
    write_json(
        directory / "plan.json",
        dict(
            root=str(ROOT),
            jobs=jobs,
            policy=dict(
                gpu=0,
                cpu_reserve=0,
                ram_reserve_mib=0,
                disk_reserve_gib=0,
                max_cpu_jobs=1,
                max_gpu_jobs=0,
                gpu_reserve_mib=0,
                exclusive_gpu_reserve_mib=0,
                launch_spacing_seconds=0,
                sample_seconds=0.01,
                gpu_admission_utilization=95,
                threads_per_job=1,
            ),
        ),
    )
    monkeypatch.setattr(
        queue,
        "snapshot",
        lambda *a: dict(
            cpus=4,
            ram_total_mib=100,
            ram_available_mib=100,
            gpu_util_pct=0,
            gpu_used_mib=0,
            gpu_total_mib=None,
            gpu_free_mib=None,
            disk_free_gib=100,
        ),
    )
    monkeypatch.setattr(queue, "sample_resources", lambda *a: {})
    monkeypatch.setattr(queue, "verify", lambda *a: {"passed": True})
    original = subprocess.Popen
    launched = []

    def launch(argv, **kwargs):
        name = argv[-1]
        launched.append(name)
        return original(
            [sys.executable, "-c", f"raise SystemExit({1 if name == 'bad' else 0})"], **kwargs
        )

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    assert run_queue(directory) == 1
    state = read_json(directory / "runtime/state.json")
    assert [j["status"] for j in state["jobs"]] == ["failed", "completed"]
    assert launched == ["bad", "good"]
    assert run_queue(directory) == 1
    assert launched == ["bad", "good"]


def test_retry_unknown_id_does_not_change_files(property_fixture):
    root, tasks, _ = property_fixture
    plan = prepare(
        root, tasks, root / "retry", models=["smiles_transformer"], targets=["mu"], seeds=[42]
    )
    path = Path(plan["lanes"][0]) / "plan.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unknown"):
        retry(root / "retry", ["not-a-job"])
    assert path.read_bytes() == before


def test_latex_refuses_incomplete_or_wrong_units(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame([dict(model="schnet", target="alpha")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="all 60"):
        fill_latex(path, tmp_path / "source.tex", tmp_path / "out.tex")


@pytest.mark.parametrize("model", ["smiles_transformer", "spe_transformer"])
def test_nonenergy_transformer_train_export_and_metric_recompute(property_fixture, model):
    from molgap.metrics import regression_metrics
    from molgap.training.transformer import train_transformer

    root, tasks, _ = property_fixture
    cfg, job = _config(root, tasks, model, "alpha")
    cfg["training"].update(epochs=1, batch_size=4, num_workers=0, accelerator="cpu")
    path = Path(job["config"])
    path.write_text(yaml.safe_dump(cfg))
    result = train_transformer(path)
    run = Path(job["run_dir"])
    predictions = pd.read_csv(run / "predictions/test_predictions.csv")
    expected = regression_metrics(predictions.alpha, predictions.alpha_pred)
    assert result["metrics"]["unit"] == "bohr^3"
    assert expected["rmse"] == pytest.approx(result["metrics"]["rmse"], abs=1e-8)
    portable = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    assert portable["target_scaler"]["target"] == "alpha"
    assert "mean_hartree" not in portable["target_scaler"]
    restored = SmilesTransformerRegressor(
        len(portable["tokenizer"]["vocabulary"]),
        portable["tokenizer"]["max_length"],
        portable["model_config"],
    )
    restored.load_state_dict(portable["state_dict"])
    restored.eval()
    data = prepare_transformer_data(cfg, root).datasets["test"]
    with torch.no_grad():
        pred = restored(data.input_ids, data.attention_mask).numpy()
    scale = portable["target_scaler"]
    np.testing.assert_allclose(
        pred * scale["scale"] + scale["mean"], predictions.alpha_pred, rtol=1e-6
    )


def test_nonenergy_chemprop_cli_train_predict_evaluate(property_fixture):
    from molgap.training.evaluate import evaluate
    from molgap.training.predict import predict
    from molgap.training.train import train

    root, tasks, frame = property_fixture
    cfg, job = _config(root, tasks, "chemprop_single_gap", "mu")
    cfg["training"].update(epochs=1, warmup_epochs=0, num_workers=0, accelerator="cpu")
    cfg["model"].update(message_hidden_dim=16, ffn_hidden_dim=16, depth=2)
    path = Path(job["config"])
    path.write_text(yaml.safe_dump(cfg))
    train(path)
    predict(path)
    result = evaluate(path)
    assert result["unit"] == "debye"
    assert result["targets"]["mu"]["n"] == 2
    scaler = read_json(Path(job["run_dir"]) / "target_scaler.json")
    assert scaler["mean"] == pytest.approx(frame.mu.iloc[:4].mean())
    assert read_json(Path(job["run_dir"]) / "training_timing.json")["epochs_completed"] == 1
    # Confirm the library model really contains the same train-only scaler.
    from chemprop.models import load_model

    model = load_model(next((Path(job["run_dir"]) / "chemprop").rglob("best.pt")))
    transform = model.predictor.output_transform
    assert float(transform.mean.reshape(-1)[0]) == pytest.approx(scaler["mean"])
    assert float(transform.scale.reshape(-1)[0]) == pytest.approx(scaler["scale"])


@pytest.mark.parametrize(
    "name", ["schnet", "painn", "egnn", "dimenet_pp", "esa", "torchmd_net", "3dinfomax"]
)
def test_real_backbone_nonenergy_gradient_and_weight_roundtrip(name, tmp_path):
    from molgap.qm9_models.batch import Molecule, collate_molecules
    from molgap.qm9_models.factory import build_model

    torch.set_num_threads(2)
    cfg = yaml.safe_load((ROOT / f"configs/qm9_six_properties/models/{name}.yaml").read_text())
    params = cfg["model"]["params"]
    if name == "3dinfomax":
        # This test isolates the prediction head. Production configs always require pretraining.
        params.update(require_pretrained=False, pretrain_checkpoint=None)
    device = torch.device("cpu")
    model = build_model(name, params, ROOT, device)
    molecules = [
        Molecule(
            "ethane",
            "CC",
            torch.tensor([6, 6, 1, 1]),
            torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0], [1.5, 1.0, 0.0]]),
            8.0,
        ),
        Molecule(
            "ethanol",
            "CCO",
            torch.tensor([6, 6, 8, 1]),
            torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            14.0,
        ),
    ]
    batch = collate_molecules(molecules)
    prediction = model(batch)
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()
    loss = (prediction - (batch["target"] - 11.0) / 3.0).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    model.eval()
    with torch.no_grad():
        reference = model(batch)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    clone = build_model(name, params, ROOT, device)
    clone.load_state_dict(torch.load(tmp_path / "model.pt", map_location=device, weights_only=True))
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(clone(batch), reference, rtol=1e-5, atol=1e-5)


def test_qm9_generic_training_uses_nonenergy_target_end_to_end(property_fixture):
    from molgap.metrics import regression_metrics
    from molgap.qm9_models.train import run

    root, tasks, frame = property_fixture
    cfg, job = _config(root, tasks, "schnet", "alpha")
    cfg["training"].update(
        epochs=1,
        batch_size=4,
        num_workers=0,
        accelerator="cpu",
        formal_protocol=False,
    )
    cfg["model"]["params"].update(n_atom_basis=16, n_interactions=1, n_rbf=8)
    geometry_path = root / cfg["data"]["geometry"]
    geometry_path.parent.mkdir(parents=True)
    np.savez(
        geometry_path,
        sample_ids=frame.sample_id.to_numpy(dtype=str),
        offsets=np.arange(len(frame) + 1) * 2,
        atomic_numbers=np.tile([6, 6], len(frame)),
        positions=np.tile(
            np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32), (len(frame), 1)
        ),
    )
    lock = root / "configs/qm9_models/sources.lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text((ROOT / "configs/qm9_models/sources.lock.json").read_text())
    path = Path(job["config"])
    path.write_text(yaml.safe_dump(cfg))
    result = run(path)
    assert result["target"] == "alpha" and result["unit"] == "bohr^3"
    prediction = pd.read_csv(Path(job["run_dir"]) / "predictions/test_predictions.csv")
    expected = regression_metrics(frame.alpha.iloc[6:], prediction.alpha_pred)
    assert result["rmse"] == pytest.approx(expected["rmse"])
    assert read_json(Path(job["run_dir"]) / "target_scaler.json")["mean"] == pytest.approx(7.5)


def test_retry_archives_failed_output_and_keeps_completed_jobs(property_fixture):
    root, tasks, _ = property_fixture
    plan = prepare(
        root,
        tasks,
        root / "retries",
        models=["smiles_transformer"],
        targets=["mu", "alpha"],
        seeds=[42],
    )
    lane = Path(plan["lanes"][0])
    state = read_json(lane / "plan.json")
    for job, status in zip(state["jobs"], ["failed", "completed"], strict=True):
        job["status"] = status
        path = Path(job["run_dir"])
        path.mkdir(parents=True)
        (path / "evidence.txt").write_text(status)
    write_json(lane / "runtime/state.json", state)
    failed, completed = state["jobs"]
    assert retry(root / "retries", [failed["id"]]) == [failed["id"]]
    after = read_json(lane / "runtime/state.json")
    assert [j["status"] for j in after["jobs"]] == ["pending", "completed"]
    assert not Path(failed["run_dir"]).exists()
    assert len(list(Path(failed["run_dir"]).parent.glob("seed42.attempt_*/evidence.txt"))) == 1
    assert (Path(completed["run_dir"]) / "evidence.txt").read_text() == "completed"


def test_summary_recomputes_units_scaler_and_sample_sd(property_fixture):
    from molgap.data.neural_properties import write_scaler
    from molgap.experiments.neural_properties import summarize
    from molgap.metrics import regression_metrics

    root, tasks, frame = property_fixture
    plan = prepare(root, tasks, root / "summary", models=["smiles_transformer"], targets=["mu"])
    lane = Path(plan["lanes"][0])
    state = read_json(lane / "plan.json")
    for index, job in enumerate(state["jobs"]):
        out = Path(job["run_dir"])
        (out / "predictions").mkdir(parents=True)
        truth = frame.iloc[6:][["sample_id", "smiles", "mu"]].copy()
        truth["mu_pred"] = truth.mu + index + 1
        truth.to_csv(out / "predictions/test_predictions.csv", index=False)
        write_json(
            out / "metrics.json",
            dict(target="mu", unit="debye", **regression_metrics(truth.mu, truth.mu_pred)),
        )
        write_scaler(out, scaler_record(frame.mu.iloc[:4], "mu", "debye"), "tokens")
        torch.save({"test_fixture": torch.ones(1)}, out / "model.pt")
        job.update(status="completed", peaks=dict(rss_mib=100 + index, gpu_process_mib=200 + index))
    write_json(lane / "runtime/state.json", state)
    out = summarize(root / "summary", strict=True)
    stats = pd.read_csv(out / "summary.csv").iloc[0]
    assert stats.rmse_mean == 2.0 and stats.rmse_std == 1.0
    assert stats.rss_mib_max == 102 and stats.gpu_process_mib_max == 202
    assert stats.unit == "debye" and stats.complete_three_seeds


def test_infomax_reuse_rejects_wrong_seed_and_changed_content(tmp_path):
    from molgap.qm9_models.train import _validate_infomax_checkpoint

    path = tmp_path / "weights.pt"
    torch.save(
        dict(split_manifest_sha256="split", geometry_sha256="geometry", training_seed=42), path
    )
    cfg = dict(
        model=dict(name="3dinfomax", params=dict(pretrain_checkpoint=str(path))),
        data=dict(split=dict(seed=2026)),
    )
    with pytest.raises(ValueError, match="seed"):
        _validate_infomax_checkpoint(
            cfg, tmp_path, {"manifest": {"sha256": "split"}}, {"geometry_sha256": "geometry"}
        )
    cfg["data"]["split"]["seed"] = 42
    cfg["model"]["params"]["pretrain_checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="content changed"):
        _validate_infomax_checkpoint(
            cfg, tmp_path, {"manifest": {"sha256": "split"}}, {"geometry_sha256": "geometry"}
        )


def test_complete_latex_fills_sixty_cells_and_preserves_existing_values(tmp_path):
    from molgap.experiments.neural_properties import LABELS

    rows = [
        dict(
            model=m,
            target=t,
            unit=PROPERTY_UNITS[t],
            n_seeds=3,
            complete_three_seeds=True,
            rmse_mean=0.1234567,
            rmse_std=0.0123456,
        )
        for m in MODELS
        for t in NEW_TARGETS
    ]
    summary = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    source = tmp_path / "before.tex"
    lines = [
        " & ".join(
            ["Ridge Regression", "old", "old", "GAP_ML", "old", "old", "old", "old", "TIME_ML"]
        )
        + r" \\"
    ]
    for label in LABELS.values():
        lines.append(
            " & ".join([label, "--", "--", "GAP_ORIGINAL", "--", "--", "--", "--", "TIME_ORIGINAL"])
            + r" \\"
        )
    source.write_text("\n".join(lines))
    output = tmp_path / "after.tex"
    fill_latex(summary, source, output)
    text = output.read_text()
    assert text.count("0.123457") == 60 and text.count("0.012346") == 60
    assert text.count("GAP_ORIGINAL") == 10 and text.count("TIME_ORIGINAL") == 10
    assert text.splitlines()[0] == lines[0]


def test_parent_rejects_insufficient_shared_ram_before_launch(property_fixture, monkeypatch):
    import molgap.experiments.neural_properties as experiments
    import molgap.scheduling.hardware as hardware

    root, tasks, _ = property_fixture
    prepare(
        root,
        tasks,
        root / "ram",
        models=["chemprop_single_gap"],
        targets=["mu", "alpha"],
        seeds=[42],
        gpus=[0, 1],
    )
    monkeypatch.setattr(
        hardware, "snapshot", lambda *a: dict(gpu_free_mib=24576, ram_available_mib=100, cpus=15)
    )

    def forbidden(*a, **kw):
        pytest.fail("No subprocess may start before shared-memory admission succeeds")

    monkeypatch.setattr(experiments.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="host RAM"):
        experiments.run(root / "ram")
