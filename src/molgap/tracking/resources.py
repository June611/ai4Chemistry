"""Linux/WSL process-tree memory sampling without importing a training framework."""

from __future__ import annotations

import subprocess
from pathlib import Path


def process_identity(pid: int, proc: Path = Path("/proc")) -> str | None:
    """PID plus kernel start ticks avoids mistaking a reused PID for the trainer."""
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else f"{pid}:{fields[19]}"
    except (OSError, ValueError, IndexError):
        return None


def process_memory(pid: int, proc: Path = Path("/proc")) -> tuple[set[int], dict]:
    parents = {}
    for path in proc.glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {pid}
    while True:
        children = {child for child, parent in parents.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    values: dict[str, list[float]] = {"rss_mib": [], "pss_mib": []}
    for child in selected:
        try:
            lines = (proc / str(child) / "smaps_rollup").read_text().splitlines()
        except OSError:
            try:
                lines = (proc / str(child) / "status").read_text().splitlines()
            except OSError:
                continue
        for line in lines:
            key = {"Rss:": "rss_mib", "VmRSS:": "rss_mib", "Pss:": "pss_mib"}.get(
                line.split()[0] if line.split() else ""
            )
            if key:
                try:
                    values[key].append(float(line.split()[1]) / 1024)
                except (ValueError, IndexError):
                    continue
    # Missing data stays null; zero would incorrectly imply no memory use.
    result = {key: sum(items) if items else None for key, items in values.items()}
    result["rss_processes_measured"] = len(values["rss_mib"])
    result["pss_processes_measured"] = len(values["pss_mib"])
    result["tree_processes"] = len(selected)
    return selected, result


def gpu_memory(pids: set[int]) -> dict:
    result = {"gpu_process_mib": None, "gpu_device_mib": None}
    try:
        device = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        result["gpu_device_mib"] = sum(float(line) for line in device.splitlines())
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        used = 0.0
        for line in processes.splitlines():
            pid, memory = line.split(",", 1)
            if int(pid.strip()) in pids:
                used += float(memory.strip())
        result["gpu_process_mib"] = used
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def sample_resources(pid: int) -> dict:
    pids, memory = process_memory(pid)
    return {**memory, **gpu_memory(pids)}
