"""Load fixed upstream sources explicitly, without silently changing environments."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def activate_source(root: Path, name: str, subdir: str = "") -> Path:
    lock_path = root / "configs/qm9_models/sources.lock.json"
    lock = json.loads(lock_path.read_text())
    record = lock[name]
    source = root / ".external/qm9_models" / name
    if not source.is_dir():
        raise FileNotFoundError(
            f"Missing pinned upstream source {source}. Run the QM9 source bootstrap first."
        )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != record["commit"]:
        raise ValueError(f"Upstream version mismatch for {name}: {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError(
            f"Upstream tracked files modified for {name}; record an explicit patch first"
        )
    # LieConv's pinned release predates Python 3.11 and contains one now-invalid
    # starred call expression.  Keep the verified checkout pristine and apply the
    # small compatibility patch to a disposable, ignored copy.
    if name == "lieconv":
        patched = root / ".external/qm9_models/.patched" / f"lieconv-{commit}"
        marker = patched / ".molgap-python311-patch-v2"
        if not marker.exists():
            if patched.exists():
                shutil.rmtree(patched)
            shutil.copytree(source, patched, ignore=shutil.ignore_patterns(".git"))
            target = patched / "lie_conv/lieGroups.py"
            original = target.read_text()
            old = (
                "rotations = torch.arange(nsamples, dtype=p.dtype, device=p.device).repeat(\n"
                "            (*p.shape[:2])\n        )"
            )
            new = (
                "rotations = torch.arange(nsamples, dtype=p.dtype, device=p.device).repeat(\n"
                "            tuple(p.shape[:2])\n        )"
            )
            if original.count(old) != 1:
                raise RuntimeError(
                    "Pinned LieConv source no longer matches the recorded compatibility patch"
                )
            target.write_text(original.replace(old, new))
            marker.write_text("Python 3.11 starred-call compatibility patch\n")
        source = patched
    if name == "torchmd-net":
        patched = root / ".external/qm9_models/.patched" / f"torchmd-net-{commit}"
        marker = patched / ".molgap-torch26-patch-v2"
        if not marker.exists():
            if patched.exists():
                shutil.rmtree(patched)
            shutil.copytree(source, patched, ignore=shutil.ignore_patterns(".git"))
            changed = 0
            for target in (patched / "torchmdnet").rglob("*.py"):
                original = target.read_text()
                updated = original.replace("list[Tensor]", "typing.List[Tensor]")
                if updated != original:
                    future = "from __future__ import annotations\n"
                    if future in updated:
                        updated = updated.replace(future, future + "\nimport typing\n", 1)
                    else:
                        updated = "import typing\n" + updated
                    target.write_text(updated)
                    changed += 1
            if changed == 0:
                raise RuntimeError(
                    "Pinned TorchMD-Net source no longer needs the recorded PyTorch patch"
                )
            marker.write_text(
                "Use typing.List in custom-op schemas for the project's pinned PyTorch 2.6.\n"
            )
        source = patched
    if name == "3dinfomax":
        patched = root / ".external/qm9_models/.patched" / f"3dinfomax-{commit}"
        marker = patched / ".molgap-python311-patch-v2"
        if not marker.exists():
            if patched.exists():
                shutil.rmtree(patched)
            shutil.copytree(source, patched, ignore=shutil.ignore_patterns(".git"))
            target = patched / "commons/utils.py"
            original = target.read_text()
            old = "from collections import MutableMapping"
            if original.count(old) != 1:
                raise RuntimeError("Pinned 3D Infomax source no longer matches compatibility patch")
            target.write_text(original.replace(old, "from collections.abc import MutableMapping"))
            # The upstream package initializer eagerly imports every experimental
            # model and therefore pulls unrelated optional extensions.  Our adapter
            # imports the official PNA module explicitly.
            (patched / "models/__init__.py").write_text(
                '"""Modules are imported explicitly by the MolGap adapter."""\n'
            )
            marker.write_text("Python 3.11 collections compatibility patch\n")
        source = patched
    path = source / subdir
    sys.path.insert(0, str(path))
    return source
