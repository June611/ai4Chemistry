"""Measure prediction wall time, keeping the measurement scope explicit."""

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from molgap.tracking.records import write_json


@contextmanager
def measure_prediction(run_dir: Path, rows: int, scope: str, *, model=None):
    # Synchronize real CUDA work for GPU trainers; CPU/CLI measurements need no torch import.
    torch = None
    if model is not None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    started = perf_counter()
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = perf_counter() - started
    parameters = sum(p.numel() for p in model.parameters()) if model is not None else None
    write_json(
        run_dir / "inference_timing.json",
        {
            "rows": rows,
            "seconds": seconds,
            "scope": scope,
            "samples_per_second": rows / seconds if seconds > 0 else None,
            "amortized_ms_per_sample": seconds * 1000 / rows if rows else None,
            "parameters": parameters,
            "flops": None,
            "flops_status": "not_measured",
            "latency_kind": "batch_wall_time_divided_by_rows_not_single_query_latency",
        },
    )
