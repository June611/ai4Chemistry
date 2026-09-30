"""Fixed descriptors from the reference gap_29 feature selection."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# Preserve notebook selected_columns order after dropping BCUT2D_LOGPHI.
FEATURES = (
    "SMR_VSA7",
    "FractionCSP3",
    "SMR_VSA10",
    "BertzCT",
    "SlogP_VSA6",
    "BCUT2D_MRHI",
    "HallKierAlpha",
    "MolLogP",
    "BCUT2D_MWHI",
    "SMR_VSA5",
    "BalabanJ",
    "PEOE_VSA11",
    "BCUT2D_CHGHI",
    "SMR_VSA9",
    "PEOE_VSA2",
    "SlogP_VSA2",
    "BCUT2D_LOGPLOW",
    "VSA_EState2",
    "BCUT2D_MWLOW",
    "VSA_EState4",
    "VSA_EState5",
    "BCUT2D_MRLOW",
    "Kappa3",
    "fr_piperzine",
    "fr_piperdine",
    "fr_Ar_NH",
    "fr_aniline",
    "fr_imidazole",
    "fr_pyridine",
)


def descriptor_matrix(smiles: list[str], sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return ordered raw descriptors and canonical identities; never drop rows."""
    if len(smiles) != len(sample_ids) or not smiles:
        raise ValueError("SMILES and sample IDs must be nonempty and have equal length.")
    functions = [getattr(Descriptors, name) for name in FEATURES]
    matrix = np.empty((len(smiles), len(FEATURES)), dtype=np.float64)
    canonical = []
    for index, (text, sample_id) in enumerate(zip(smiles, sample_ids, strict=True)):
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES for sample_id={sample_id}: {text!r}")
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        for column, (name, function) in enumerate(zip(FEATURES, functions, strict=True)):
            try:
                matrix[index, column] = function(mol)
            except Exception as exc:
                raise ValueError(f"Descriptor {name} failed for sample_id={sample_id}") from exc
        if not np.isfinite(matrix[index]).all():
            raise ValueError(f"Nonfinite descriptors for sample_id={sample_id}")
    return matrix, canonical
