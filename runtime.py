"""Runtime configuration shared by GPU experiment scripts."""

from __future__ import annotations

import os
import random

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Return the explicitly requested device or a safe CUDA/CPU default.

    Set ``DEVICE=cuda`` or ``DEVICE=cpu`` to make a run's hardware choice
    explicit. ``auto`` (the default) uses CUDA when available, otherwise CPU.
    CPU is useful for smoke tests but is not practical for full generation runs.
    """
    choice = (requested or os.environ.get("DEVICE", "auto")).strip().lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError("DEVICE must be one of: auto, cuda, cpu")


def model_dtype(device: torch.device) -> torch.dtype:
    """Use bfloat16 on CUDA and float32 everywhere else."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def describe_runtime(device: torch.device) -> str:
    """Return a concise, reproducibility-friendly runtime summary."""
    if device.type == "cuda":
        return f"device=cuda ({torch.cuda.get_device_name(device)}) dtype=bfloat16"
    return "device=cpu dtype=float32"


def set_experiment_seed(seed: int) -> None:
    """Seed Python and PyTorch generators used by generation experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
