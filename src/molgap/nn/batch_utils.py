"""Batch and unbatch atom representations for Transformer processing."""

import torch
from torch import Tensor


def batch_atoms(h_flat: Tensor, batch_ids: Tensor) -> tuple[Tensor, Tensor]:
    """
    Convert flat atom representations to batched format with padding.

    Args:
        h_flat: [N_total, d] flat atom representations from MPNN
        batch_ids: [N_total] molecule ID for each atom (from bmg.batch)

    Returns:
        h_batched: [B, N_max, d] batched and padded representations
        mask: [B, N_max] padding mask (True = padding, False = real atom)

    Example:
        h_flat shape: [11, 300]  # 3 molecules: 3+6+2 atoms
        batch_ids: [0,0,0, 1,1,1,1,1,1, 2,2]
        →
        h_batched: [3, 6, 300]  # padded to max 6 atoms
        mask: [[F,F,F,T,T,T], [F,F,F,F,F,F], [F,F,T,T,T,T]]
    """
    device = h_flat.device
    d = h_flat.size(-1)

    # Get batch size and max atoms per molecule
    B = batch_ids.max().item() + 1

    # Count atoms per molecule
    atom_counts = torch.bincount(batch_ids, minlength=B)
    N_max = atom_counts.max().item()

    # Initialize output tensors
    h_batched = torch.zeros(B, N_max, d, device=device, dtype=h_flat.dtype)
    mask = torch.ones(B, N_max, device=device, dtype=torch.bool)

    # Fill batched tensor
    for i in range(B):
        indices = (batch_ids == i).nonzero(as_tuple=True)[0]
        n_atoms = len(indices)
        h_batched[i, :n_atoms] = h_flat[indices]
        mask[i, :n_atoms] = False  # False = real atom

    return h_batched, mask


def unbatch_atoms(h_batched: Tensor, batch_ids: Tensor) -> Tensor:
    """
    Convert batched atom representations back to flat format.

    Args:
        h_batched: [B, N_max, d] batched representations
        batch_ids: [N_total] original batch IDs

    Returns:
        h_flat: [N_total, d] flat representations
    """
    device = h_batched.device
    d = h_batched.size(-1)
    N_total = len(batch_ids)
    B = batch_ids.max().item() + 1

    h_flat = torch.zeros(N_total, d, device=device, dtype=h_batched.dtype)

    for i in range(B):
        indices = (batch_ids == i).nonzero(as_tuple=True)[0]
        n_atoms = len(indices)
        h_flat[indices] = h_batched[i, :n_atoms]

    return h_flat
