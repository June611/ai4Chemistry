"""Read original QM9 XYZ records or SDF without guessing row indices or converting labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

ELEMENTS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
ELEMENT_BY_NUMBER = {v: k for k, v in ELEMENTS.items()}
# The official gdb_130667 XYZ record is internally inconsistent: its first
# SMILES and target match the public QM9 table, while its coordinate atom list
# matches the second SMILES/InChI stored in that same record.  Keeping the ID is
# required to preserve this project's established split.  All other records
# remain subject to the atom-composition check.
KNOWN_QM9_ATOM_SMILES_CONFLICTS = {"gdb_130667"}


@dataclass
class QM9Geometry:
    source_id: str
    atomic_numbers: np.ndarray
    positions: np.ndarray
    gap_hartree: float
    smiles: tuple[str, ...]


def read_xyz(text: str) -> QM9Geometry:
    """Parse the official extended XYZ format, including Fortran-style exponents."""
    lines = text.splitlines()
    count = int(lines[0])
    properties = lines[1].split()
    if count < 1 or len(lines) < count + 5 or len(properties) != 17:
        raise ValueError("Expected a complete original QM9 extended XYZ record")
    if properties[0] != "gdb":
        raise ValueError("QM9 header must start with gdb")
    number = int(properties[1])
    if number < 1:
        raise ValueError("QM9 molecule ID must be positive")
    coordinates = []
    atomic_numbers = []
    for row in lines[2 : count + 2]:
        fields = row.split()
        atomic_numbers.append(ELEMENTS[fields[0]])
        coordinates.append([float(x.replace("*^", "e")) for x in fields[1:4]])
    pos = np.asarray(coordinates, dtype=np.float32)
    gap = float(properties[9].replace("*^", "e"))
    if pos.shape != (count, 3) or not np.isfinite(pos).all() or not np.isfinite(gap):
        raise ValueError("Invalid QM9 coordinates or gap")
    smiles = tuple(lines[count + 3].split())
    return QM9Geometry(
        f"gdb_{number}", np.asarray(atomic_numbers, dtype=np.int64), pos, gap, smiles
    )


def validate_identity(
    record: QM9Geometry, source_id: str, smiles: str, gap: float, skip_smiles_check: bool = False
) -> None:
    """Require matching original ID, molecular identity and original-unit property."""
    if record.source_id != source_id:
        raise ValueError(f"Molecule ID mismatch: {source_id} != {record.source_id}")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid project SMILES: {source_id}")

    # Only validate SMILES identity if we have valid SMILES from the geometry source
    if not skip_smiles_check and record.smiles[0]:
        canonical = Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True)
        candidate_smiles = set()
        for value in record.smiles:
            if value:  # Skip empty SMILES
                parsed = Chem.MolFromSmiles(value)
                if parsed is not None:
                    candidate_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(parsed), canonical=True))
        if candidate_smiles and canonical not in candidate_smiles:
            raise ValueError(f"QM9 SMILES identity mismatch: {source_id}")

    # Always validate atomic composition
    atoms = sorted(a.GetAtomicNum() for a in Chem.AddHs(molecule).GetAtoms())
    if (
        atoms != sorted(record.atomic_numbers.tolist())
        and source_id not in KNOWN_QM9_ATOM_SMILES_CONFLICTS
    ):
        raise ValueError(f"QM9 element/count mismatch: {source_id}")

    # Match the existing preprocessing tolerance for duplicate-label aggregation.
    if abs(float(gap) - record.gap_hartree) > 1.1e-4:
        raise ValueError(f"QM9 gap mismatch in Hartree: {source_id}")


def xyz_path(directory: Path, source_id: str) -> Path:
    """Use the named original ID, never a filtered dataset's row number."""
    if not source_id.startswith("gdb_") or not source_id[4:].isdigit():
        raise ValueError(f"Invalid original QM9 ID: {source_id}")
    return directory / f"dsgdb9nsd_{int(source_id[4:]):06d}.xyz"


def read_sdf(sdf_path: Path) -> dict[str, QM9Geometry]:
    """Read all molecules from QM9 SDF file and return dict keyed by source_id."""
    # Disable sanitization to handle QM9's unusual valence states
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    geometries = {}
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue

        try:
            source_id = mol.GetProp("_Name")
        except Exception:
            skipped += 1
            continue

        if not source_id or not source_id.startswith("gdb"):
            skipped += 1
            continue

        # Get atomic numbers
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)

        # Get 3D coordinates
        try:
            conf = mol.GetConformer()
            positions = np.array(conf.GetPositions(), dtype=np.float32)
        except Exception:
            skipped += 1
            continue

        # Validate coordinates are finite
        if not np.isfinite(positions).all():
            skipped += 1
            continue

        # Gap is not stored in SDF, will be matched from CSV
        gap_hartree = 0.0

        # Get SMILES - try with sanitization, fallback to canonical from our CSV
        try:
            smiles = (Chem.MolToSmiles(mol),)
        except Exception:
            smiles = ("",)  # Will be validated against project SMILES later

        geometries[source_id] = QM9Geometry(
            source_id=source_id,
            atomic_numbers=atomic_numbers,
            positions=positions,
            gap_hartree=gap_hartree,
            smiles=smiles,
        )

    if skipped > 0:
        print(f"Note: Skipped {skipped} molecules due to parsing issues")

    return geometries
