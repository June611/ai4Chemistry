"""Resume any completed-epoch checkpoint from the new unified fresh200 pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import torch

    from molgap.modular.train import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="last.ckpt")
    parser.add_argument("--diagnostic-epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run / "checkpoints" / checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "molgap_modular_epoch_v1":
        raise ValueError("Use a new fresh200 full checkpoint, not a legacy or inference export")
    context = payload["run_context"]
    if context["diagnostic"] and args.diagnostic_epochs is None:
        raise ValueError("Diagnostic recovery requires an explicit --diagnostic-epochs cap")
    result = train(
        run / "config.yaml",
        resume=checkpoint,
        output_dir=run,
        diagnostic_epochs=args.diagnostic_epochs,
        limit_rows=context["limit_rows"],
        device_override=args.device,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
