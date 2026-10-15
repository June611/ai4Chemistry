"""SDF chemical topology joined to immutable geometry by original QM9 identity.

Preparation never reads property labels. Failed sanitizations are recorded, not
silently removed from this project's fixed splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

from molgap.qm9_models.batch import Molecule
from molgap.qm9_models.data import GeometryDataset
from molgap.utils.files import sha256


def prepare(sdf: Path, geometry: Path, output: Path, geometry_paths: list[Path]) -> dict:
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite topology: {output}")
    with np.load(geometry, allow_pickle=False) as f:
        base = {k: f[k] for k in f.files}
    ids = base["sample_ids"].tolist()
    lookup = {sid.split(":")[-1]: i for i, sid in enumerate(ids)}
    records, partial = {}, []
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
    for mol in tqdm(supplier, desc="SDF topology", mininterval=5):
        if mol is None:
            continue
        name = mol.GetProp("_Name")
        if name not in lookup:
            continue
        if name in records:
            raise ValueError(f"Duplicate SDF identity: {name}")
        i = lookup[name]
        start, end = base["offsets"][i : i + 2]
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        if not np.array_equal(z, base["atomic_numbers"][start:end]):
            raise ValueError(f"SDF/geometry atom order mismatch: {name}")
        if not np.allclose(pos, base["positions"][start:end], atol=1e-4, rtol=0):
            raise ValueError(f"SDF/geometry coordinates differ: {name}")
        status = Chem.SanitizeMol(mol, catchErrors=True)
        if status != Chem.SanitizeFlags.SANITIZE_NONE:
            partial.append({"sample_id": ids[i], "sanitize_flag": int(status)})
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
        for a in mol.GetAtoms():
            a.SetIntProp("geometry_index", a.GetIdx())
        features = np.array([atom_to_feature_vector(a) for a in mol.GetAtoms()], dtype=np.int16)
        # Match the official Transformer-M QM9 heavy-atom features, including
        # attached-H counts. Keep explicit-H features for DeMol's full-atom input.
        heavy = Chem.RemoveHs(mol, sanitize=False)
        heavy.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(heavy)
        for a in heavy.GetAtoms():
            features[a.GetIntProp("geometry_index")] = atom_to_feature_vector(a)
        bonds = np.array(
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=np.int16
        ).reshape(-1, 2)
        bf = np.array([bond_to_feature_vector(b) for b in mol.GetBonds()], dtype=np.int16).reshape(
            -1, 3
        )
        adj = np.zeros((len(z), len(z)), dtype=bool)
        if len(bonds):
            adj[bonds[:, 0], bonds[:, 1]] = True
            adj |= adj.T.copy()
        sp = shortest_path(adj, directed=False, unweighted=True)
        sp = np.where(np.isfinite(sp), sp, 510).astype(np.int16)
        records[name] = (features, bonds, bf, sp)
    if len(records) != len(ids):
        missing = sorted(set(lookup) - set(records))
        raise ValueError(f"Missing SDF molecules ({len(missing)}): {missing[:10]}")
    # Verify all split-specific geometry artifacts really differ only in order.
    geometry_hashes = {}
    for path in dict.fromkeys([geometry, *geometry_paths]):
        with np.load(path, allow_pickle=False) as f:
            other = {k: f[k] for k in f.files}
            for j, sid in enumerate(other["sample_ids"].tolist()):
                k = lookup[sid.split(":")[-1]]
                lo, hi = other["offsets"][j : j + 2]
                a, b = base["offsets"][k : k + 2]
                if not np.array_equal(other["atomic_numbers"][lo:hi], base["atomic_numbers"][a:b]):
                    raise ValueError(f"Cross-split atom order differs: {sid}")
                if not np.array_equal(other["positions"][lo:hi], base["positions"][a:b]):
                    raise ValueError(f"Cross-split coordinates differ: {sid}")
        geometry_hashes[str(path.name)] = sha256(path)
    ordered = [records[sid.split(":")[-1]] for sid in ids]
    edge_offsets = np.r_[0, np.cumsum([len(r[1]) for r in ordered])]
    spatial_offsets = np.r_[0, np.cumsum([r[3].size for r in ordered])]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=base["sample_ids"],
        offsets=base["offsets"],
        atomic_numbers=base["atomic_numbers"],
        node_features=np.concatenate([r[0] for r in ordered]),
        edge_offsets=edge_offsets,
        bonds=np.concatenate([r[1] for r in ordered]),
        bond_features=np.concatenate([r[2] for r in ordered]),
        spatial_offsets=spatial_offsets,
        spatial=np.concatenate([r[3].ravel() for r in ordered]),
    )
    report = {
        "rows": len(ids),
        "sdf_sha256": sha256(sdf),
        "topology_sha256": sha256(output),
        "geometry_hashes": geometry_hashes,
        "coordinate_comparison_atol_angstrom": 1e-4,
        "partially_sanitized": partial,
        "dropped_split_samples": 0,
        "features": "OGB 9 atom / 3 bond; heavy-atom features after RemoveHs",
        "property_labels_used": False,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class TopologyDataset(GeometryDataset):
    def __init__(
        self, frame, arrays, index, topology, topology_index, *, remove_hydrogens, target="delta_e"
    ):
        super().__init__(frame, arrays, index, target)
        self.topology = topology
        self.topology_index = topology_index
        self.remove_hydrogens = remove_hydrogens

    def __getitem__(self, index):
        m = super().__getitem__(index)
        t = self.topology
        k = self.topology_index[m.sample_id]
        lo, hi = t["offsets"][k : k + 2]
        if not np.array_equal(m.atomic_numbers.numpy(), t["atomic_numbers"][lo:hi]):
            raise ValueError(f"Topology atom order differs: {m.sample_id}")
        a, b = t["edge_offsets"][k : k + 2]
        c, d = t["spatial_offsets"][k : k + 2]
        keep = (m.atomic_numbers != 1).numpy() if self.remove_hydrogens else np.ones(hi - lo, bool)
        mapping = np.cumsum(keep) - 1
        bonds = t["bonds"][a:b]
        valid = keep[bonds].all(axis=1)
        return Molecule(
            m.sample_id,
            m.smiles,
            m.atomic_numbers[keep],
            m.positions[keep],
            m.target,
            torch.from_numpy(t["node_features"][lo:hi][keep].astype(np.int64)),
            torch.from_numpy(mapping[bonds[valid]].copy()).long(),
            torch.from_numpy(t["bond_features"][a:b][valid].astype(np.int64)),
            torch.from_numpy(
                t["spatial"][c:d].reshape(hi - lo, hi - lo)[keep][:, keep].copy()
            ).long(),
        )


def load_topology(config, root, geometry, frames, arrays, index):
    path = root / config["data"]["topology"]
    meta = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != meta["topology_sha256"]:
        raise ValueError("Topology hash mismatch")
    if geometry["geometry_sha256"] not in meta["geometry_hashes"].values():
        raise ValueError("Geometry was not checked against this SDF topology")
    with np.load(path, allow_pickle=False) as f:
        topology = {k: f[k] for k in f.files}
    ids = topology["sample_ids"].tolist()
    if len(set(ids)) != len(ids) or set(ids) != set(index):
        raise ValueError("Topology identities differ from fixed split cohort")
    ti = {sid: i for i, sid in enumerate(ids)}
    geometry["topology_sha256"] = meta["topology_sha256"]
    return {
        k: TopologyDataset(
            v,
            arrays,
            index,
            topology,
            ti,
            remove_hydrogens=config["data"].get("remove_hydrogens", False),
            target=config["data"].get("target_column", "delta_e"),
        )
        for k, v in frames.items()
    }


def collate_topology(molecules):
    b = len(molecules)
    n = max(len(m.atomic_numbers) for m in molecules)
    e = max(1, max(len(m.bonds) for m in molecules))
    nodes = torch.zeros(b, n, 9, dtype=torch.long)
    spatial = torch.full((b, n, n), 511, dtype=torch.long)
    bonds = torch.zeros(b, e, 2, dtype=torch.long)
    features = torch.zeros(b, e, 3, dtype=torch.long)
    mask = torch.zeros(b, e, dtype=torch.bool)
    for i, m in enumerate(molecules):
        nn, ee = len(m.atomic_numbers), len(m.bonds)
        nodes[i, :nn] = m.node_features
        spatial[i, :nn, :nn] = m.spatial
        bonds[i, :ee] = m.bonds
        features[i, :ee] = m.bond_features
        mask[i, :ee] = True
    return {
        "node_features": nodes,
        "spatial": spatial,
        "bonds": bonds,
        "bond_features": features,
        "bond_mask": mask,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdf", type=Path, default=Path("data/raw/qm9_geometry/gdb9.sdf"))
    p.add_argument("--geometry", type=Path, default=Path("data/geometry/qm9_full_seed3407.npz"))
    p.add_argument("--output", type=Path, default=Path("data/topology/qm9_full.npz"))
    args = p.parse_args()
    print(
        json.dumps(
            prepare(
                args.sdf,
                args.geometry,
                args.output,
                list(Path("data/geometry").glob("qm9_full_seed*.npz")),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
