from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models


def regression_head(in_features: int = 2048) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )


def _torchvision_weight_cache_path(weight_enum) -> Path:
    filename = Path(urlparse(weight_enum.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def build_resnet50_regressor(
    *,
    weights: str | None = "imagenet",
    allow_weight_download: bool = False,
) -> nn.Module:
    if weights == "imagenet":
        weight_enum = models.ResNet50_Weights.IMAGENET1K_V1
        if not allow_weight_download and not _torchvision_weight_cache_path(weight_enum).is_file():
            raise RuntimeError(
                "ImageNet pretrained ResNet50 weights are required, but downloads are disabled "
                "and the torchvision checkpoint is not present in the local torch hub cache."
            )
    elif weights is None or weights == "none":
        weight_enum = None
    else:
        raise ValueError("weights must be 'imagenet', 'none', or None")
    try:
        model = models.resnet50(weights=weight_enum)
    except Exception as exc:
        if weights == "imagenet":
            raise RuntimeError(
                "ImageNet pretrained ResNet50 weights are required but could not "
                "be loaded. Provide cached torchvision weights or network access for a real run."
            ) from exc
        raise
    model.fc = regression_head(model.fc.in_features)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model
