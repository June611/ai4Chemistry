"""Fetch exact official source revisions into ignored local checkouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from molgap.config import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="Source key; repeat as needed")
    args = parser.parse_args()
    root = find_project_root(Path.cwd())
    lock = json.loads((root / "configs/qm9_models/sources.lock.json").read_text())
    selected = args.source or list(lock)
    unknown = set(selected) - set(lock)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    for name in selected:
        target = root / ".external/qm9_models" / name
        record = lock[name]
        if target.exists():
            actual = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != record["commit"]:
                raise ValueError(f"Existing source revision differs: {name}; refusing to reset")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(
            ["git", "-C", str(target), "remote", "add", "origin", record["url"]], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", record["commit"]],
            check=True,
        )
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)


if __name__ == "__main__":
    main()
