"""Scientific integration checks for the two new geometry/topology adapters."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from rdkit import Chem
from scipy.sparse.csgraph import shortest_path

from molgap.qm9_models.batch import Molecule, collate_molecules
from molgap.qm9_models.factory import build_model


def molecules():
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

    result = []
    for smi, positions in [
        ("O", [[0, 0, 0], [0.95, 0, 0], [-0.24, 0.92, 0]]),
        ("C", [[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]),
    ]:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
        bonds = torch.tensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
        adj = np.zeros((len(z), len(z)))
        adj[bonds[:, 0], bonds[:, 1]] = 1
        adj += adj.T.copy()
        result.append(
            Molecule(
                smi,
                smi,
                z,
                torch.tensor(positions, dtype=torch.float32),
                0.2,
                torch.tensor([atom_to_feature_vector(a) for a in mol.GetAtoms()]),
                bonds,
                torch.tensor([bond_to_feature_vector(b) for b in mol.GetBonds()]),
                torch.tensor(shortest_path(adj, directed=False, unweighted=True)).long(),
            )
        )
    return result


def parameters(model):
    p = dict(
        embed_dim=32,
        ffn_dim=32,
        num_layers=2,
        num_heads=4,
        num_kernel=8,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_checkpointing=False,
    )
    if model == "demol":
        p["bond_regularization"] = True
    return p


@pytest.mark.parametrize("name", ["demol", "transformer_m"])
def test_backward_symmetry_padding_sample_independence_and_reload(name):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    root = Path(__file__).parents[1]
    p = parameters(name)
    model = build_model(name, p, root, torch.device("cpu"))
    records = molecules()
    batch = collate_molecules(records)
    prediction = model(batch)
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()
    loss = prediction.square().mean()
    if name == "demol":
        loss += 0.1 * model.auxiliary_loss()
    loss.backward()
    grads = [x.grad for x in model.parameters() if x.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert sum(x.abs().sum().item() for x in grads) > 0
    model.eval()
    with torch.no_grad():
        reference = model(batch)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        transformed = dict(batch, positions=batch["positions"] @ rotation + 8)
        torch.testing.assert_close(model(transformed), reference, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            model(collate_molecules(records[:1])), reference[:1], atol=2e-5, rtol=2e-5
        )
        dirty = deepcopy(batch)
        dirty["positions"][~dirty["atom_mask"]] = 1234
        dirty["target"].fill_(float("nan"))
        torch.testing.assert_close(model(dirty), reference, atol=2e-5, rtol=2e-5)
        permuted = deepcopy(records)
        for m in permuted:
            order = torch.randperm(len(m.atomic_numbers))
            inverse = torch.argsort(order)
            m.atomic_numbers = m.atomic_numbers[order]
            m.positions = m.positions[order]
            m.node_features = m.node_features[order]
            m.spatial = m.spatial[order][:, order]
            m.bonds = inverse[m.bonds].flip(-1)
        torch.testing.assert_close(
            model(collate_molecules(permuted)), reference, atol=2e-5, rtol=2e-5
        )
        clone = build_model(name, p, root, torch.device("cpu")).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(batch), reference)


@pytest.mark.parametrize("name", ["demol", "transformer_m"])
def test_checkpointing_preserves_forward_and_gradients(name):
    torch.set_num_threads(2)
    torch.manual_seed(5)
    root = Path(__file__).parents[1]
    config = parameters(name)
    normal = build_model(name, config, root, torch.device("cpu"))
    checked = build_model(
        name, {**config, "activation_checkpointing": True}, root, torch.device("cpu")
    )
    checked.load_state_dict(normal.state_dict())
    batch = collate_molecules(molecules())
    outputs = []
    for model in (normal, checked):
        out = model(batch)
        loss = out.square().mean()
        if name == "demol":
            loss += 0.1 * model.auxiliary_loss()
        loss.backward()
        outputs.append(out.detach())
    torch.testing.assert_close(*outputs)
    for a, b in zip(normal.parameters(), checked.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_protocol_recipes_and_no_external_pretraining():
    root = Path(__file__).parents[1]
    for name in ("demol", "transformer_m"):
        cfg = yaml.safe_load((root / f"configs/qm9_models/{name}.yaml").read_text())
        assert cfg["training"]["batch_size"] == 256
        assert cfg["training"]["epochs"] == 100
        assert cfg["training"]["early_stopping"]["patience"] == 20
        assert cfg["training"]["early_stopping"]["monitor"] == "val_mae"
        assert cfg["data"]["label_unit"] == "hartree"
        assert cfg["training"]["target_scaling"] == "mad"
        assert cfg["training"]["loss"] == "mae"
        assert cfg["provenance"]["external_pretraining"] is False
        assert cfg["smoke"]["batch_size"] == 256
        assert cfg["smoke"]["train_rows"] == 107038


def test_transformer_m_single_heavy_atom_at_origin_keeps_3d_channel():
    from molgap.qm9_models.transformer_m import TransformerMAdapter

    torch.set_num_threads(2)
    torch.manual_seed(3)
    m = deepcopy(molecules()[0])
    m.atomic_numbers = m.atomic_numbers[:1]
    m.positions = torch.zeros(1, 3)
    m.node_features = m.node_features[:1]
    m.bonds = torch.empty(0, 2).long()
    m.bond_features = torch.empty(0, 3).long()
    m.spatial = torch.zeros(1, 1).long()
    other = deepcopy(m)
    other.positions = torch.ones(1, 3)
    model = TransformerMAdapter(parameters("transformer_m")).eval()
    with torch.no_grad():
        model.encoder.molecule_3d_bias.edge_proj.bias.copy_(torch.linspace(-10, 10, 32))
        alone = model(collate_molecules([m]))
        together = model(collate_molecules([m, other]))
        torch.testing.assert_close(alone, together[:1], atol=1e-6, rtol=1e-6)


def test_topology_cache_identity_mismatch_is_rejected(tmp_path):
    from molgap.qm9_models.topology import TopologyDataset

    records = molecules()
    import pandas as pd

    m = records[0]
    frame = pd.DataFrame([{"sample_id": m.sample_id, "smiles": m.smiles, "delta_e": 0.2}])
    arrays = {
        "offsets": np.array([0, 3]),
        "atomic_numbers": m.atomic_numbers.numpy(),
        "positions": m.positions.numpy(),
    }
    bad = {"offsets": np.array([0, 3]), "atomic_numbers": np.array([7, 1, 1])}
    data = TopologyDataset(
        frame, arrays, {m.sample_id: 0}, bad, {m.sample_id: 0}, remove_hydrogens=False
    )
    with pytest.raises(ValueError, match="atom order differs"):
        data[0]


def test_official_qm9_collator_feature_and_geometry_parity():
    import importlib.util
    from types import SimpleNamespace

    from molgap.qm9_models.transformer_m import TransformerMAdapter

    root = Path(__file__).parents[1]
    path = root / ".external/Transformer-M/Transformer-M/data/collator.py"
    if not path.exists():
        pytest.skip("Pinned official checkout required for upstream collator parity")
    spec = importlib.util.spec_from_file_location("official_transformer_m_collator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = molecules()
    items = []
    for i, m in enumerate(records):
        n = len(m.atomic_numbers)
        items.append(
            SimpleNamespace(
                idx=i,
                x=m.node_features + torch.arange(9) * 512 + 1,
                pos=m.positions,
                attn_bias=torch.zeros(n + 1, n + 1),
                spatial_pos=m.spatial,
                in_degree=torch.zeros(n).long(),
                out_degree=torch.zeros(n).long(),
                attn_edge_type=torch.zeros(n, n, 3).long(),
                edge_input=torch.zeros(n, n, 1, 3).long(),
                y=torch.tensor([m.target]),
                train_mean=0.0,
                train_std=1.0,
                type=torch.tensor([0]),
            )
        )
    original = mod.collator_3d_qm9(items)
    adapted = TransformerMAdapter.inputs(collate_molecules(records))
    for key in ("x", "pos", "attn_bias", "node_type_edge"):
        torch.testing.assert_close(adapted[key], original[key])


def test_dual_plan_seeds_budgets_and_formal_gate(tmp_path):
    import importlib.util

    root = Path(__file__).parents[1]
    path = root / "scripts/run_qm9_dual_reproduction.py"
    spec = importlib.util.spec_from_file_location("dual_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "queue"
    module.prepare(root, directory)
    import json

    for gpu, model in module.MODELS.items():
        plan = json.loads((directory / f"formal_gpu{gpu}/plan.json").read_text())
        assert [j["seed"] for j in plan["jobs"]] == [3407, 42, 2026]
        assert all(j["model"] == model and j["depends_on"] == [] for j in plan["jobs"])
        assert plan["policy"]["max_gpu_jobs"] == 1
        for job in plan["jobs"]:
            cfg = yaml.safe_load(Path(job["config"]).read_text())
            assert cfg["data"]["split"]["seed"] == cfg["training"]["seed"] == job["seed"]
            assert cfg["training"]["batch_size"] == 256
    with pytest.raises(RuntimeError, match="smoke evidence"):
        module.validate_smoke(directory)
    # Freeze full-epoch smoke evidence without running a GPU; verify admission
    # and that scheduling balances independent jobs instead of pinning a model.
    import pandas as pd

    for gpu in module.MODELS:
        run = directory / f"smoke_result_{gpu}"
        run.mkdir()
        (run / "model_summary.json").write_text(
            json.dumps(
                {
                    "budget": {"batch_size": 256},
                    "split_rows": {"train": 107038, "val": 13380, "test": 13380},
                }
            )
        )
        pd.DataFrame([dict(train_batches=419, epoch_seconds=100 if gpu == 0 else 10)]).to_csv(
            run / "history.csv", index=False
        )
        runtime = directory / f"smoke_gpu{gpu}/runtime"
        runtime.mkdir()
        (runtime / "state.json").write_text(
            json.dumps(
                {
                    "jobs": [
                        {
                            "status": "completed",
                            "run_dir": str(run),
                            "model": module.MODELS[gpu],
                            "seed": 3407,
                        }
                    ],
                }
            )
        )
    module.balance_formal(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["formal_balanced"] is True
    assignments = manifest["assignment"]
    assert len(assignments) == 6
    assert len({a["job"] for a in assignments}) == 6
    assert {a["gpu"] for a in assignments if a["job"].startswith("demol_")} == {0, 1}
    before = (directory / "manifest.json").read_bytes()
    module.balance_formal(directory)
    assert (directory / "manifest.json").read_bytes() == before
