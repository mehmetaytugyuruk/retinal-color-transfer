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


def build_resnet50_12ch_regressor(
    *,
    weights: str | None = "imagenet",
    allow_weight_download: bool = False,
    in_channels: int = 12,
) -> nn.Module:
    """Build a ResNet-50 regressor that accepts ``in_channels`` input channels.

    The standard pretrained conv1 kernel (shape [64, 3, 7, 7]) is expanded to
    [64, in_channels, 7, 7] by repeating the 3-channel weights
    ``in_channels // 3`` times and scaling each copy by ``3 / in_channels``
    so that the expected pre-activation magnitude is preserved.

    Args:
        weights: ``'imagenet'`` to load ImageNet-pretrained weights (default),
            ``'none'`` or ``None`` to start from random initialisation.
        allow_weight_download: When ``False`` (default) the function raises if
            the pretrained checkpoint is not already cached locally.
        in_channels: Total number of input channels. Must be a positive
            multiple of 3 (default 12).
    """
    if in_channels <= 0 or in_channels % 3 != 0:
        raise ValueError(f"in_channels must be a positive multiple of 3, got {in_channels}")

    # Build the standard 3-channel model first to get pretrained weights.
    base = build_resnet50_regressor(weights=weights, allow_weight_download=allow_weight_download)

    if in_channels == 3:
        return base

    # Adapt conv1: [64, 3, 7, 7] → [64, in_channels, 7, 7]
    old_conv = base.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    repeats = in_channels // 3
    scale = 3.0 / in_channels  # keeps expected pre-activation magnitude
    with torch.no_grad():
        new_conv.weight.copy_(
            old_conv.weight.repeat(1, repeats, 1, 1) * scale
        )
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    base.conv1 = new_conv
    return base
