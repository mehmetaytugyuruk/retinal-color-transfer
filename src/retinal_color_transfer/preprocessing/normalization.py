from __future__ import annotations

from pathlib import Path

import numpy as np

from retinal_color_transfer.artifacts import write_json
from retinal_color_transfer.config import RepresentationConfig, stable_fingerprint
from retinal_color_transfer.representations.contracts import (
    cache_path_for,
    canonical_tensor,
    read_representation_array,
)


def compute_channel_stats(frame, *, cache_root: str | Path, cfg: RepresentationConfig) -> dict:
    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for row in frame.itertuples(index=False):
        path = cache_path_for(cache_root, str(row.image_id), cfg)
        image = read_representation_array(path, cfg)
        scaled = canonical_tensor(image, cfg).permute(1, 2, 0).numpy()
        pixels = scaled.reshape(-1, 3)
        sums += pixels.sum(axis=0)
        sums_sq += np.square(pixels).sum(axis=0)
        pixel_count += pixels.shape[0]
    if pixel_count == 0:
        raise ValueError("Cannot compute normalization statistics from an empty training split")
    mean = sums / pixel_count
    variance = np.maximum((sums_sq / pixel_count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std == 0):
        raise ValueError("At least one channel has zero standard deviation")
    data = {
        "normalization_policy": "representation_train_channel_stats",
        "representation": cfg.name,
        "representation_fingerprint": cfg.fingerprint,
        "tensor_scaling": cfg.tensor_scaling,
        "split_used": "train",
        "num_images": int(len(frame)),
        "num_pixels": int(pixel_count),
        "channel_mean": [float(v) for v in mean],
        "channel_std": [float(v) for v in std],
    }
    data["normalization_fingerprint"] = stable_fingerprint(data)
    return data


def save_channel_stats(stats: dict, path: str | Path) -> None:
    write_json(stats, path)


def validate_normalization_compatibility(stats: dict, cfg: RepresentationConfig) -> None:
    if stats.get("normalization_policy") != "representation_train_channel_stats":
        raise ValueError("Normalization statistics must use representation_train_channel_stats")
    if stats.get("representation") != cfg.name:
        raise ValueError("Normalization representation does not match experiment representation")
    if stats.get("representation_fingerprint") != cfg.fingerprint:
        raise ValueError("Normalization representation fingerprint mismatch")
    if stats.get("tensor_scaling") != cfg.tensor_scaling:
        raise ValueError("Normalization tensor scaling does not match representation config")
    if "normalization_fingerprint" not in stats:
        raise ValueError("Normalization statistics missing normalization_fingerprint")
    if len(stats.get("channel_mean", [])) != 3 or len(stats.get("channel_std", [])) != 3:
        raise ValueError("Normalization statistics must contain three channel means and stds")
    if any(float(value) <= 0 for value in stats["channel_std"]):
        raise ValueError("Normalization channel standard deviations must be positive")
