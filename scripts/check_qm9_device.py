"""Probe Torch and DGL on the target device before server training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/qm9_device_preflight.json"))
    args = parser.parse_args()
    report = {"torch": torch.__version__, "dgl": dgl.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        report.update(status="pending", reason="No accessible CUDA device; GPU checks unverified")
    else:
        try:
            device = torch.device("cuda:0")
            report["gpu"] = torch.cuda.get_device_name(device)
            report["memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
            tensor = torch.ones((2, 2), device=device)
            assert torch.isfinite(tensor @ tensor).all().item()
            graph = dgl.graph(([0, 1], [1, 0])).to(device)
            graph.ndata["probe"] = torch.ones((2, 1), device=device)
            torch.cuda.synchronize()
            report.update(status="pass", scope="Torch CUDA and DGL graph device transfer only")
        except Exception as error:
            report.update(
                status="fail",
                reason=f"{type(error).__name__}: {error}",
                action="Check server Torch/CUDA/DGL compatibility before ALIGNN/Infomax training",
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] == "fail" or (args.require_cuda and report["status"] != "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
