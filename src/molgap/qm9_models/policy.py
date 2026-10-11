"""Central user-requested model pauses shared by training and batch entrypoints."""

from __future__ import annotations

from pathlib import Path

import yaml


def paused_models(root: Path) -> dict[str, str]:
    path = root / "configs/qm9_models/paused_models.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text())
    paused = payload.get("paused_models") if isinstance(payload, dict) else None
    if not isinstance(paused, dict) or not all(
        isinstance(name, str) and isinstance(reason, str) for name, reason in paused.items()
    ):
        raise ValueError(f"Invalid paused model mapping: {path}")
    return paused


def require_enabled(model: str, root: Path) -> None:
    reason = paused_models(root).get(model)
    if reason is not None:
        raise RuntimeError(
            f"{model} is paused by user request: {reason} "
            "See README.md and configs/qm9_models/paused_models.yaml; dry-run remains available."
        )


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    print(" ".join(sorted(paused_models(root))))


if __name__ == "__main__":
    main()
