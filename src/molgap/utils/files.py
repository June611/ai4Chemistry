"""Small helpers for auditable experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def write_command_record(
    path: Path,
    command: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Store both argv and a human-readable representation of a command."""
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "argv": command,
        "display": shlex.join(command),
    }
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    root: Path,
) -> None:
    """Record the resolved config, software environment, seeds, and input identity."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    processed_dir = Path(config["data"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    tracked = ["train.csv", "val.csv", "test.csv", "split_manifest.csv", "lineage.sha256"]
    inputs = {
        name: {"path": str(processed_dir / name), "sha256": sha256(processed_dir / name)}
        for name in tracked
        if (processed_dir / name).is_file()
    }
    environment = {
        "created_at": datetime.now(UTC).isoformat(),
        "config_source": str(config_path),
        "config_sha256": sha256(config_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "chemprop": importlib.metadata.version("chemprop"),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": {
            "data": int(config["data"]["split"]["seed"]),
            "training": int(config["training"]["seed"]),
        },
        "inputs": inputs,
    }
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
