from __future__ import annotations

import importlib.metadata
import platform
import random
from typing import Any

import numpy as np
import torch


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("device must be one of auto, mps, cuda, cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is false")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false")
    return torch.device(requested)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_info(device: torch.device, seed: int) -> dict[str, Any]:
    packages = ["torch", "torchvision", "numpy", "pandas", "opencv-python", "scipy"]
    info = {
        "device": str(device),
        "seed": seed,
        "libraries": {name: _version(name) for name in packages},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mps_determinism_note": "PyTorch MPS determinism is best effort and may vary by operation.",
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(device)
    return info
