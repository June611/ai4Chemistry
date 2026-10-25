"""Lazy model registry without importing unused optional frameworks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelRegistration:
    module: str
    class_name: str
    arguments: Callable[[str, dict, Path, torch.device], tuple]
    uses_geometry: bool


def _config(_name, config, _root, _device):
    return (config,)


def _name_config(name, config, _root, _device):
    return name, config


def _root_config(_name, config, root, _device):
    return root, config


def _root_config_device(_name, config, root, device):
    return root, config, device


def _egnn(_name, config, root, device):
    return root, config, device


MODEL_REGISTRY = {
    "egnn": ModelRegistration("egnn", "EGNNAdapter", _egnn, True),
    "demol": ModelRegistration("demol", "DeMolAdapter", _config, True),
    "transformer_m": ModelRegistration(
        "transformer_m", "TransformerMAdapter", _config, True
    ),
    "schnet": ModelRegistration("schnetpack", "SchNetPackAdapter", _name_config, True),
    "painn": ModelRegistration("schnetpack", "SchNetPackAdapter", _name_config, True),
    "cormorant": ModelRegistration(
        "cormorant", "CormorantAdapter", _root_config_device, True
    ),
    "dimenet_pp": ModelRegistration("dimenet", "DimeNetPPAdapter", _config, True),
    "lie_transformer": ModelRegistration(
        "lie_transformer", "LieTransformerAdapter", _root_config, True
    ),
    "esa": ModelRegistration("esa", "ESAAdapter", _config, False),
    "alignn": ModelRegistration("alignn", "ALIGNNAdapter", _root_config, True),
    "torchmd_net": ModelRegistration(
        "torchmd_net", "TorchMDNetAdapter", _root_config, True
    ),
    "3dinfomax": ModelRegistration(
        "infomax", "ThreeDInfomaxAdapter", _root_config, False
    ),
}


def build_model(name: str, config: dict, root: Path, device: torch.device) -> nn.Module:
    """Build one registered adapter while preserving lazy optional imports."""
    registration = MODEL_REGISTRY.get(name)
    if registration is None:
        raise ValueError(f"Adapter is not implemented/registered: {name}")
    module = import_module(f"molgap.qm9_models.{registration.module}")
    adapter = getattr(module, registration.class_name)
    model = adapter(*registration.arguments(name, config, root, device))
    return model.to(device)
