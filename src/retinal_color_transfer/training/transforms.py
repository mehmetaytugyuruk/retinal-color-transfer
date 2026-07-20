from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from retinal_color_transfer.config import RepresentationConfig
from retinal_color_transfer.representations.contracts import (
    canonical_tensor,
    representation_contract,
)


@dataclass(frozen=True)
class TrainTransform:
    representation_config: RepresentationConfig
    flip_probability: float = 0.5
    rotation_degrees: float = 10
    fill: int = 0

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        contract = representation_contract(self.representation_config.name)
        interpolation = (
            InterpolationMode.NEAREST
            if contract.is_binary
            else InterpolationMode.BILINEAR
        )
        tensor = canonical_tensor(image, self.representation_config)
        if torch.rand(()) < self.flip_probability:
            tensor = F.hflip(tensor)
        angle = float(torch.empty(1).uniform_(-self.rotation_degrees, self.rotation_degrees).item())
        tensor = F.rotate(
            tensor,
            angle=angle,
            interpolation=interpolation,
            fill=[float(self.fill)] * 3,
        )
        if contract.is_binary:
            tensor = (tensor > 0.5).float()
        return tensor


@dataclass(frozen=True)
class EvalTransform:
    representation_config: RepresentationConfig

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return canonical_tensor(image, self.representation_config)


def train_transform(
    representation_config: RepresentationConfig,
    *,
    flip_probability: float = 0.5,
    rotation_degrees: float = 10,
    fill: int = 0,
):
    return TrainTransform(
        representation_config=representation_config,
        flip_probability=flip_probability,
        rotation_degrees=rotation_degrees,
        fill=fill,
    )


def eval_transform(representation_config: RepresentationConfig):
    return EvalTransform(representation_config=representation_config)
