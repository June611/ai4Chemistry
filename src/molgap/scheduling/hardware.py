"""Linux/cgroup and NVIDIA observations; unavailable values block GPU admission."""

import os
import shutil
import subprocess
from pathlib import Path


def read_number(path):
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def snapshot(root: Path, gpu: int = 0):
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = float(value.split()[0]) / 1024
    current = read_number("/sys/fs/cgroup/memory.current")
    limit = read_number("/sys/fs/cgroup/memory.max")
    available = mem["MemAvailable"]
    total = mem["MemTotal"]
    if limit is not None and current is not None:
        total = min(total, limit / 2**20)
        available = min(available, (limit - current) / 2**20)
    cpus = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, float(quota) / float(period))
    except (OSError, ValueError):
        pass
    result = {
        "ram_available_mib": available,
        "ram_total_mib": total,
        "cpus": cpus,
        "disk_free_gib": shutil.disk_usage(root).free / 2**30,
        "gpu_total_mib": None,
        "gpu_free_mib": None,
        "gpu_util_pct": None,
        "gpu_used_mib": None,
        "gpu_name": None,
    }
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        name, total, free, used, util = output.stdout.strip().split(",")
        result.update(
            gpu_name=name.strip(),
            gpu_total_mib=float(total),
            gpu_free_mib=float(free),
            gpu_used_mib=float(used),
            gpu_util_pct=float(util),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def admission(job, running, hw, policy):
    """Return an actionable wait reason or None; reserve peaks during startup too."""
    if hw["disk_free_gib"] < policy["disk_reserve_gib"]:
        return "disk reserve"
    gpu_jobs = [j for j in running if j["gpu_mib"] > 0]
    cpu_jobs = [j for j in running if j["gpu_mib"] == 0]
    if sum(j["cpu_slots"] for j in running) + job["cpu_slots"] > hw["cpus"] - policy["cpu_reserve"]:
        return "CPU budget"
    # Total reservations catch pre-allocation races; live available RAM protects other processes.
    if (
        sum(j["ram_mib"] for j in running) + job["ram_mib"]
        > hw["ram_total_mib"] - policy["ram_reserve_mib"]
    ):
        return "RAM reservation"
    if job["ram_mib"] > hw["ram_available_mib"] - policy["ram_reserve_mib"]:
        return "available RAM"
    if not job["gpu_mib"]:
        return "CPU job limit" if len(cpu_jobs) >= policy["max_cpu_jobs"] else None
    if hw["gpu_free_mib"] is None:
        return "GPU unavailable"
    if len(gpu_jobs) >= policy["max_gpu_jobs"]:
        return "GPU job limit"
    if gpu_jobs and (job["exclusive"] or any(j["exclusive"] for j in gpu_jobs)):
        return "exclusive GPU job"
    reserve = (
        policy.get("exclusive_gpu_reserve_mib", policy["gpu_reserve_mib"])
        if job["exclusive"]
        else policy["gpu_reserve_mib"]
    )
    if job["exclusive"] and not gpu_jobs and hw["gpu_used_mib"] > 512:
        return "external GPU allocation"
    if sum(j["gpu_mib"] for j in gpu_jobs) + job["gpu_mib"] > hw["gpu_total_mib"] - reserve:
        return "GPU reservation"
    if job["gpu_mib"] > hw["gpu_free_mib"] - reserve:
        return "available GPU memory"
    if hw["gpu_util_pct"] is None or hw["gpu_util_pct"] >= policy["gpu_admission_utilization"]:
        return "GPU busy"
    return None
