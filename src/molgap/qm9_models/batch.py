"""Framework-neutral molecular records and PyTorch batching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SPECIES = (1, 6, 7, 8, 9)


@dataclass
class Molecule:
    sample_id: str
    smiles: str
    atomic_numbers: Tensor
    positions: Tensor
    target: float
    node_features: Tensor | None = None
    bonds: Tensor | None = None
    bond_features: Tensor | None = None
    spatial: Tensor | None = None

    def validate(self) -> None:
        n = self.atomic_numbers.numel()
        if n == 0 or self.atomic_numbers.ndim != 1 or self.positions.shape != (n, 3):
            raise ValueError(f"Invalid atom/coordinate shapes for {self.sample_id}")
        if not torch.isfinite(self.positions).all():
            raise ValueError(f"Nonfinite coordinates for {self.sample_id}")
        if not torch.isin(self.atomic_numbers, torch.tensor(SPECIES)).all():
            raise ValueError(f"Non-QM9 element in {self.sample_id}")
        if not torch.isfinite(torch.tensor(self.target)):
            raise ValueError(f"Nonfinite target for {self.sample_id}")


def collate_molecules(molecules: list[Molecule]) -> dict:
    """Keep physical labels separate from model inputs and mask padding explicitly."""
    if not molecules:
        raise ValueError("Cannot collate an empty molecular batch")
    for molecule in molecules:
        molecule.validate()
    size = len(molecules)
    width = max(m.atomic_numbers.numel() for m in molecules)
    charges = torch.zeros(size, width, dtype=torch.long)
    positions = torch.zeros(size, width, 3)
    for i, molecule in enumerate(molecules):
        n = molecule.atomic_numbers.numel()
        charges[i, :n] = molecule.atomic_numbers
        positions[i, :n] = molecule.positions
    mask = charges != 0
    edge_mask = mask[:, :, None] & mask[:, None, :]
    edge_mask &= ~torch.eye(width, dtype=torch.bool)[None]
    result = {
        "sample_id": [m.sample_id for m in molecules],
        "smiles": [m.smiles for m in molecules],
        "target": torch.tensor([m.target for m in molecules], dtype=torch.float32),
        "charges": charges,
        "positions": positions,
        "atom_mask": mask,
        "edge_mask": edge_mask,
        "one_hot": charges[:, :, None].eq(torch.tensor(SPECIES)).float(),
    }
    if any(m.node_features is not None for m in molecules):
        from molgap.qm9_models.topology import collate_topology

        if any(m.node_features is None for m in molecules):
            raise ValueError("Mixed topology/non-topology molecular batch")
        result.update(collate_topology(molecules))
    return result


def flat_atoms(batch: dict) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten only real atoms, preserving molecule membership."""
    mask = batch["atom_mask"]
    groups = torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(mask)
    return batch["charges"][mask], batch["positions"][mask], groups[mask]


def radius_edges(positions: Tensor, groups: Tensor, cutoff: float) -> Tensor:
    """Directed, nonperiodic neighbor lists without cross-molecule edges or truncation."""
    edges = []
    for group in torch.unique(groups):
        idx = torch.where(groups == group)[0]
        distance = torch.cdist(positions[idx], positions[idx])
        valid = (distance < cutoff) & ~torch.eye(len(idx), dtype=torch.bool, device=idx.device)
        src, dst = torch.where(valid)
        edges.append(torch.stack((idx[src], idx[dst])))
    return torch.cat(edges, dim=1)
