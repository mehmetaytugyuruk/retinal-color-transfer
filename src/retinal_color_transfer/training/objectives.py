from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn


@dataclass(frozen=True)
class TargetStatistics:
    mean: float
    sample_std: float
    min_age: int
    max_age: int
    n: int


def target_statistics(ages: pd.Series | np.ndarray) -> TargetStatistics:
    values = pd.Series(ages).astype(float)
    if values.empty:
        raise ValueError("Cannot compute target statistics from empty ages")
    if not (values == values.astype(int)).all():
        raise ValueError("Approved LDS protocol requires integer-valued ages")
    std = float(values.std(ddof=1))
    if std <= 0:
        raise ValueError("Training age sample standard deviation must be positive")
    return TargetStatistics(
        mean=float(values.mean()),
        sample_std=std,
        min_age=int(values.min()),
        max_age=int(values.max()),
        n=int(values.shape[0]),
    )


def denormalize_age(value: float, stats: TargetStatistics) -> float:
    return float(value) * stats.sample_std + stats.mean


def lds_weights(
    ages: pd.Series | np.ndarray,
    *,
    sigma: float = 2.0,
    mode: str = "reflect",
    truncate: float = 4.0,
    epsilon: float = 1.0e-5,
) -> dict[int, float]:
    stats = target_statistics(ages)
    values = pd.Series(ages).astype(int)
    bins = np.arange(stats.min_age, stats.max_age + 1)
    counts = np.zeros(len(bins), dtype=np.float64)
    for age, count in values.value_counts().items():
        counts[int(age) - stats.min_age] = float(count)
    smoothed = gaussian_filter1d(counts, sigma=sigma, mode=mode, truncate=truncate)
    weights = 1.0 / (smoothed + epsilon)
    weights = weights / weights.mean()
    return {int(age): float(weight) for age, weight in zip(bins, weights, strict=True)}


def weighted_smooth_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    weights = weights.reshape(-1)
    if predictions.shape != targets.shape or predictions.shape != weights.shape:
        raise ValueError(
            "predictions, targets, and weights must have the same flattened shape; "
            f"got {predictions.shape}, {targets.shape}, {weights.shape}"
        )
    criterion = nn.SmoothL1Loss(beta=beta, reduction="none")
    per_sample = criterion(predictions, targets)
    return (per_sample * weights).mean()


def validation_smooth_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have same flattened shape; "
            f"got {predictions.shape}, {targets.shape}"
        )
    return nn.SmoothL1Loss(beta=beta, reduction="mean")(predictions, targets)
