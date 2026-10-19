"""Immutable seven/ten-target data, lineage-preserving thermochemical extension."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from molgap.data.neural_properties import scaler_record
from molgap.data.qm9_properties import PROPERTY_UNITS, SEEDS, digest
from molgap.modular.physics import TASKS10, UNITS10


def prepare_thermo_data(root, destination, seeds=SEEDS):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite thermochemical data: {destination}")
    raw_path = root / "data/raw/qm9_chn.csv"
    source_root = root / "data/processed/qm9_single_properties_v1"
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    required = ["mol_id", "u0", "u298", "h298"]
    if raw.mol_id.duplicated().any() or not np.isfinite(raw[required[1:]].to_numpy()).all():
        raise ValueError("Complete unique raw QM9 energy labels required")
    canonical = pd.read_csv(
        root / f"data/processed/qm9_full/random/seed{seeds[0]}/dataset.csv",
        float_precision="round_trip",
    )
    members = canonical[["sample_id", "source_id"]].copy()
    members["mol_id"] = members.source_id.str.split("|")
    members = members.explode("mol_id")
    if members.mol_id.duplicated().any() or set(members.mol_id) != set(raw.mol_id):
        raise ValueError("Raw labels do not cover the exact frozen source groups")
    joined = members.merge(raw[required], on="mol_id", how="left", validate="many_to_one")
    means = joined.groupby("sample_id", sort=False)[required[1:]].mean()
    # Validate every source before creating output; no repartition or row dropping.
    identity = json.loads((source_root / "data_identity.json").read_text())
    source_files, outputs = {}, {}
    prepared = []
    for seed in seeds:
        src = source_root / f"random/seed{seed}"
        frozen_canonical = pd.read_csv(
            root / f"data/processed/qm9_full/random/seed{seed}/dataset.csv",
            float_precision="round_trip",
        )
        pd.testing.assert_frame_equal(canonical, frozen_canonical)
        for split in ("train", "val", "test"):
            path = src / f"{split}.csv"
            relative = f"random/seed{seed}/{split}.csv"
            if digest(path) != identity["extended_files"][relative]:
                raise ValueError(f"Seven-label source changed: {path}")
            frame = pd.read_csv(path, float_precision="round_trip")
            for task in TASKS10[7:]:
                frame[task] = means.loc[frame.sample_id, task].to_numpy()
            prepared.append((relative, frame))
            source_files[str(path.relative_to(root))] = digest(path)
        manifest = src / "split_manifest.csv"
        original = root / f"data/processed/qm9_full/random/seed{seed}/split_manifest.csv"
        if digest(manifest) != digest(original):
            raise ValueError("Thermo source manifest differs from original split")
        source_files[str(manifest.relative_to(root))] = digest(manifest)
    destination.mkdir(parents=True)
    for relative, frame in prepared:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.17g")
        saved = pd.read_csv(path, float_precision="round_trip")
        pd.testing.assert_frame_equal(frame, saved)
        outputs[relative] = digest(path)
    for seed in seeds:
        shutil.copy2(
            source_root / f"random/seed{seed}/split_manifest.csv",
            destination / f"random/seed{seed}/split_manifest.csv",
        )
    audit = dict(
        protocol_id="qm9_thermo10_lineage_v1",
        targets=UNITS10,
        raw_sha256=digest(raw_path),
        split_seeds=list(seeds),
        source_files=source_files,
        extended_files=outputs,
        duplicate_policy="frozen source_id groups, equal-weight means",
        preserved_original_membership_order_and_energy_labels=True,
    )
    (destination / "data_identity.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def load_labels(config, root):
    tasks = tuple(config["data"]["targets"])
    units = {t: UNITS10[t] for t in tasks}
    data_dir = root / config["data"]["processed_dir"]
    identity_path = data_dir.parent.parent / "data_identity.json"
    identity = json.loads(identity_path.read_text())
    expected_units = UNITS10 if len(tasks) == 10 else PROPERTY_UNITS
    if (
        identity["targets"] != expected_units
        or not identity["preserved_original_membership_order_and_energy_labels"]
    ):
        raise ValueError("Label units or preserved lineage contract differ")
    frames, membership, hashes = {}, [], {}
    seed = config["training"]["seed"]
    original_dir = root / f"data/processed/qm9_full/random/seed{seed}"
    manifest_path = data_dir / "split_manifest.csv"
    if digest(manifest_path) != digest(original_dir / "split_manifest.csv"):
        raise ValueError("Manifest differs from frozen original partition")
    manifest = pd.read_csv(manifest_path)
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        key = f"random/seed{seed}/{split}.csv"
        if digest(path) != identity["extended_files"].get(key):
            raise ValueError(f"Label file content changed: {path}")
        f = pd.read_csv(path, float_precision="round_trip")
        if len(tasks) == 10:
            seven_path = (
                root / f"data/processed/qm9_single_properties_v1/random/seed{seed}/{split}.csv"
            )
            if digest(seven_path) != identity["source_files"].get(
                str(seven_path.relative_to(root))
            ):
                raise ValueError(
                    "Original seven-property source labels changed after thermo extension"
                )
            seven = pd.read_csv(seven_path, float_precision="round_trip")
            pd.testing.assert_frame_equal(f[list(seven.columns)], seven)
        source = pd.read_csv(original_dir / f"{split}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(f[list(source.columns)], source)
        expected = (
            manifest.loc[manifest.split.eq(split)].sort_values("split_position").sample_id.tolist()
        )
        if f.sample_id.tolist() != expected or f.sample_id.duplicated().any():
            raise ValueError("Label membership/order changed")
        if not np.isfinite(f[list(tasks)].to_numpy()).all():
            raise ValueError("All requested training/evaluation labels must be finite")
        frames[split] = f
        membership.append(set(f.sample_id))
        hashes[split] = digest(path)
    if any(a & b for i, a in enumerate(membership) for b in membership[i + 1 :]):
        raise ValueError("Split leakage detected")
    scalers = {t: scaler_record(frames["train"][t], t, units[t]) for t in tasks}
    return (
        frames,
        scalers,
        dict(
            labels=hashes,
            manifest_sha256=digest(manifest_path),
            identity_sha256=digest(identity_path),
        ),
    )


class InputLabels(Dataset):
    """Pair backbone inputs with labels without exposing labels to the encoder."""

    def __init__(self, inputs, frame, tasks, scalers):
        self.inputs = inputs
        self.ids = frame.sample_id.astype(str).tolist()
        self.y = (
            (
                frame[list(tasks)].to_numpy(dtype=np.float64)
                - np.array([scalers[t]["mean"] for t in tasks])
            )
            / np.array([scalers[t]["scale"] for t in tasks])
        ).astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.inputs[index], self.y[index], self.ids[index]


if __name__ == "__main__":
    import argparse

    from molgap.config import find_project_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/qm9_thermo10_v1")
    args = parser.parse_args()
    root = find_project_root(Path.cwd())
    print(json.dumps(prepare_thermo_data(root, root / args.output), indent=2))
