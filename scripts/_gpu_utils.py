"""GPU / EGL helpers shared by scripts/launch.py and scripts/evaluate_runs.py."""
from __future__ import annotations

import os
import socket
import subprocess
import warnings

# Make CUDA device indices agree with nvidia-smi (PCI bus order).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def parse_int_list(raw: str) -> list[int]:
    values = [int(x) for x in raw.split(",") if x.strip()]
    assert values, f"Expected a comma-separated list of integers, got {raw!r}"
    return values


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def free_memory_mb() -> dict[int, int]:
    """Free memory per GPU index, from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    free = {}
    for line in out.splitlines():
        if line.strip():
            idx, mem = [x.strip() for x in line.split(",")]
            free[int(idx)] = int(mem)
    assert free, "nvidia-smi returned no GPUs"
    return free


def cuda_to_egl_map(gpus: list[int], explicit: str | None = None) -> dict[int, int]:
    """Map CUDA device index -> EGL device index (MuJoCo offscreen rendering).

    Uses --egl-map "cuda:egl,..." if given, else `egl_probe` if installed, else assumes the
    indices coincide (true on most single-vendor machines).
    """
    if explicit:
        mapping = {}
        for pair in explicit.split(","):
            c, e = pair.split(":")
            mapping[int(c)] = int(e)
    else:
        try:
            import egl_probe  # optional: pip install egl_probe
        except ImportError:
            warnings.warn("egl_probe not installed; assuming EGL device index == CUDA index. "
                          "Pass --egl-map if rendering fails.")
            return {g: g for g in gpus}
        mapping = {}
        for egl_index, dev in enumerate(egl_probe.get_available_devices()):
            mapping[int(egl_probe.get_cuda_device(dev))] = egl_index
    missing = [g for g in gpus if g not in mapping]
    assert not missing, f"No EGL device for CUDA GPU(s) {missing}; mapping={mapping}"
    return mapping


def visible_devices(cuda_gpu: int, egl_device: int) -> str:
    """CUDA_VISIBLE_DEVICES value for one job.

    robosuite 1.4 asserts that MUJOCO_EGL_DEVICE_ID appears in CUDA_VISIBLE_DEVICES, so when the
    EGL index differs from the CUDA index it is appended (the job still computes on `cuda_gpu`,
    which is listed first and becomes cuda:0).
    """
    return str(cuda_gpu) if cuda_gpu == egl_device else f"{cuda_gpu},{egl_device}"


def pick_gpu(gpus: list[int], free_mb: dict[int, int], reserved_mb: dict[int, int], need_mb: int) -> int | None:
    """GPU with the most effective free memory that still fits `need_mb`, else None."""
    best, best_free = None, -1
    for g in gpus:
        assert g in free_mb, f"GPU {g} not visible to nvidia-smi"
        eff = free_mb[g] - reserved_mb.get(g, 0)
        if eff >= need_mb and eff > best_free:
            best, best_free = g, eff
    return best
