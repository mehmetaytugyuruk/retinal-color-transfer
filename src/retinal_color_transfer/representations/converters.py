from __future__ import annotations

import cv2
import numpy as np

from retinal_color_transfer.config import RepresentationConfig


class RepresentationError(ValueError):
    """Raised when a representation cannot be created."""


_CHANNEL_REPRESENTATIONS: dict[str, tuple[int | None, int]] = {
    "rgb_r": (None, 0),
    "rgb_g": (None, 1),
    "rgb_b": (None, 2),
    "lab_l": (cv2.COLOR_RGB2LAB, 0),
    "lab_a": (cv2.COLOR_RGB2LAB, 1),
    "lab_b": (cv2.COLOR_RGB2LAB, 2),
    "hsv_h": (cv2.COLOR_RGB2HSV, 0),
    "hsv_s": (cv2.COLOR_RGB2HSV, 1),
    "hsv_v": (cv2.COLOR_RGB2HSV, 2),
    "ycrcb_y": (cv2.COLOR_RGB2YCrCb, 0),
    "ycrcb_cr": (cv2.COLOR_RGB2YCrCb, 1),
    "ycrcb_cb": (cv2.COLOR_RGB2YCrCb, 2),
}


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RepresentationError(
            f"Expected uint8 RGB HxWx3 image, got shape={rgb.shape} dtype={rgb.dtype}"
        )


def _repeat_channel(channel: np.ndarray) -> np.ndarray:
    return np.repeat(channel[:, :, None], 3, axis=2)


def convert_representation(rgb: np.ndarray, cfg: RepresentationConfig) -> np.ndarray:
    _check_rgb(rgb)
    if cfg.name == "rgb":
        return rgb.copy()
    if cfg.name == "grayscale":
        return _repeat_channel(_luminance(rgb))
    if cfg.name in _CHANNEL_REPRESENTATIONS:
        conversion_code, channel_index = _CHANNEL_REPRESENTATIONS[cfg.name]
        converted = rgb if conversion_code is None else cv2.cvtColor(rgb, conversion_code)
        return _repeat_channel(converted[:, :, channel_index])
    if cfg.name == "lab":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if cfg.name == "hsv":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if cfg.name == "ycrcb":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    raise RepresentationError(f"Unknown representation: {cfg.name}")


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
