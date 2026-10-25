"""Numerical contracts and backbone conformance for the fresh200 plug-ins."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from molgap.modular.config import PHYSICS, validate
from molgap.modular.heads import MLoREHead
from molgap.modular.physics import TASKS10, QM9PhysicsLoss
from scripts.prepare_qm9_epoch200_review import make_executable_config

ROOT = Path(__file__).resolve().parents[1]


def test_mlore_routing_balance_and_gradients():
    torch.manual_seed(17)
    head = MLoREHead(8, hidden_dim=16, rank_list=(2, 4), spe_rank=2, topk=2, num_modules=2)
    x = torch.randn(5, 8)
    out = head(x)
    assert out.shape == (5, 7)
    routes = [torch.stack(module_routes) for module_routes in head.last_route_probs]
    assert len(routes) == 2 and routes[0].shape == (7, 5, 2)
    torch.testing.assert_close(routes[0].sum(-1), torch.ones(7, 5))
    balance = head.balance_loss()
    assert torch.isfinite(balance) and float(balance) >= 0
    out.square().mean().add(balance).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def physical_labels():
    y = torch.zeros(3, 10, dtype=torch.float64)
    for t, v in dict(
        homo=-0.3, lumo=0.1, delta_e=0.4, u0=-100.0, u298=-99.9, h298=-99.9 + PHYSICS["kbt_hartree"]
    ).items():
        y[:, TASKS10.index(t)] = v
    return y


def test_physics_tolerance_direction_inverse_scaling_and_gradients():
    scalers = {t: dict(mean=10.0 * i, scale=0.1 + i) for i, t in enumerate(TASKS10)}
    loss = QM9PhysicsLoss(scalers, PHYSICS)
    raw = physical_labels()
    z = (raw - loss.means) / loss.scales
    total, parts, _ = loss(z, z, 20)
    assert float(total) == pytest.approx(0.0, abs=1e-20)
    disturbed = z.clone()
    disturbed[:, 2] += 0.02
    disturbed.requires_grad_()
    _, parts, _ = loss(disturbed, z, 20)
    parts["orbital_gap"].backward()
    assert (disturbed.grad[:, :3].abs().sum(0) > 0).all()
    assert disturbed.grad[:, 2].mean() > 0 and disturbed.grad[:, 1].mean() < 0
    for task, sign, part in [("h298", 1, "enthalpy_internal"), ("u0", 1, "thermal_nonnegative")]:
        zz = z.clone()
        zz[:, TASKS10.index(task)] += sign * 2
        _, pp, _ = loss(zz, z, 20)
        assert pp[part] > 0
    assert loss.weight(1) == 0 and loss.weight(10) == 0
    assert loss.weight(11) == pytest.approx(0.01) and loss.weight(20) == pytest.approx(0.1)
    # Double inverse scaling is essential when subtracting large total energies.
    assert loss.means.dtype == loss.scales.dtype == torch.float64


def test_config_refuses_loss_selection_seed_or_unit_changes():
    base = make_executable_config("chemprop", "physics_thermo10", 42)
    validate(base)
    for section, key, value in [
        ("training", "selection_metric", "val_loss"),
        ("training", "epochs", 100),
        ("training", "batch_size", 128),
        ("training", "seed", 2026),
    ]:
        cfg = copy.deepcopy(base)
        cfg[section][key] = value
        with pytest.raises(ValueError):
            validate(cfg)
    cfg = copy.deepcopy(base)
    cfg["data"]["units"]["h298"] = "eV"
    with pytest.raises(ValueError):
        validate(cfg)


@pytest.mark.parametrize(
    "name",
    [
        "chemprop",
        "schnet",
        "painn",
        "egnn",
        "dimenet_pp",
        "demol",
        "transformer_m",
        "esa",
        "torchmd_net",
        "3dinfomax",
        "smiles_transformer",
        "spe_transformer",
    ],
)
def test_feature_adapters_forward_backward_and_geometry_invariance(name):
    from test_demol_transformer_m import molecules

    from molgap.modular.backbones import MolecularEncoder
    from molgap.modular.inputs import collate_inputs
    from molgap.qm9_models.batch import collate_molecules

    cfg = make_executable_config(name, "mlore7", 42)
    p = cfg["source"]["model"].get("params", {})
    if name in {"schnet", "painn"}:
        p.update(n_atom_basis=16, n_interactions=2, n_rbf=8)
    if name == "egnn":
        p.update(hidden_nf=16, n_layers=2)
    if name == "dimenet_pp":
        p.update(
            hidden_channels=16,
            num_blocks=1,
            int_emb_size=8,
            basis_emb_size=8,
            out_emb_channels=16,
            num_spherical=3,
            num_radial=3,
        )
    if name in {"demol", "transformer_m"}:
        p.update(
            embed_dim=16,
            ffn_dim=32,
            num_layers=1,
            num_heads=4,
            num_kernel=8,
            activation_checkpointing=False,
        )
    if name == "torchmd_net":
        p.update(embedding_dimension=16, num_layers=1, num_heads=4, num_rbf=8)
    if name == "esa":
        p.update(
            hidden_dim=16,
            num_heads=4,
            max_edges=16,
            layer_types=["S", "P"],
            num_outputs=32,
            mlp_hidden_size=16,
        )
    if name == "3dinfomax":
        p.update(
            hidden_dim=16,
            propagation_depth=2,
            readout_hidden_dim=16,
            pretrain_checkpoint=None,
            require_pretrained=False,
        )
    inputs = collate_molecules(molecules())
    if name == "chemprop":
        from chemprop.data import MoleculeDatapoint, MoleculeDataset
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        ds = MoleculeDataset(
            [MoleculeDatapoint.from_smi(s, y=np.zeros(1)) for s in ["O", "C"]],
            SimpleMoleculeMolGraphFeaturizer(),
        )
        inputs, _, _ = collate_inputs(
            [(ds[i], np.zeros(7, dtype=np.float32), str(i)) for i in range(2)], "chemprop"
        )
    if name in {"smiles_transformer", "spe_transformer"}:
        cfg["source"]["model"].update(d_model=16, num_layers=1, num_heads=4, dim_feedforward=32)
        inputs = dict(
            input_ids=torch.tensor([[2, 3, 0], [2, 4, 3]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        )
    encoder = MolecularEncoder(cfg, ROOT, torch.device("cpu"), vocab_size=8)
    h = encoder(inputs)
    assert h.shape == (2, encoder.output_dim) and torch.isfinite(h).all()
    h.square().mean().backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if "positions" in inputs and name not in {"esa", "3dinfomax"}:
        encoder.eval()
        with torch.no_grad():
            ref = encoder(inputs)
            rotation, _ = torch.linalg.qr(torch.randn(3, 3))
            rotated = dict(inputs, positions=inputs["positions"] @ rotation + 4)
            torch.testing.assert_close(encoder(rotated), ref, rtol=2e-4, atol=2e-4)


def test_new_variants_are_distinct_in_existing_summary_and_units_are_preserved(tmp_path):
    import json

    import yaml

    from molgap.experiments.summary import load_records, write_summary

    paths = []
    for variant, target in [
        ("single_task", "delta_e"),
        ("single_task", "homo"),
        ("mlore7", None),
        ("physics_thermo10", None),
    ]:
        cfg = make_executable_config("chemprop", variant, 42, target)
        directory = tmp_path / cfg["experiment"]["name"]
        directory.mkdir()
        (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
        metric = dict(mae=0.1, mse=0.01, rmse=0.1, mape=1.0, r2=0.9)
        path = directory / "metrics.json"
        path.write_text(
            json.dumps(dict(diagnostic=False, targets={t: metric for t in cfg["data"]["targets"]}))
        )
        paths.append(path)
    gap = load_records(paths)
    assert len(gap) == 3 and len({r["model"] for r in gap}) == 3
    alpha = load_records(paths, target="alpha")
    assert len(alpha) == 2 and {r["unit"] for r in alpha} == {"bohr^3"}
    output = write_summary(alpha, tmp_path / "summary", metrics=["mae", "rmse"])
    assert "bohr^3" in output["summary_markdown"].read_text()
    assert "QM9 alpha model comparison" in output["comparison_chart_svg"].read_text()
    path = paths[-1]
    payload = json.loads(path.read_text())
    payload["diagnostic"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Diagnostic modular"):
        load_records([path])
