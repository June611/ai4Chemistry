"""Shared runtime policy for deterministic QM9 GPU training.

The helpers in this module improve input-pipeline overlap without changing model
math, sample order, formal batch size, or checkpoint selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader


def dataloader_kwargs(
    training: Mapping[str, Any],
    *,
    device: torch.device,
    smoke: bool = False,
) -> dict[str, Any]:
    """Return safe DataLoader performance options for the selected device."""
    workers = 0 if smoke else int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", device.type == "cuda"))
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers:
        options["persistent_workers"] = bool(training.get("persistent_workers", False))
        prefetch_factor = int(training.get("prefetch_factor", 2))
        if prefetch_factor < 1:
            raise ValueError("training.prefetch_factor must be positive")
        options["prefetch_factor"] = prefetch_factor
    return options


def make_loader(
    dataset,
    *,
    training: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    generator: torch.Generator | None = None,
    smoke: bool = False,
) -> DataLoader:
    """Build a DataLoader with consistent CPU/GPU overlap settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
        **dataloader_kwargs(training, device=device, smoke=smoke),
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors, asynchronously when the loader supplied pinned memory."""
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
