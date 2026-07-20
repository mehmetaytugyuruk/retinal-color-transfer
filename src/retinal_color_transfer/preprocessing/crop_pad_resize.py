from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PREPROCESSING_VERSION = "crop_pad_resize_v1"


class PreprocessingError(ValueError):
    """Raised when an image cannot be prepared safely."""


@dataclass(frozen=True)
class PreparedImage:
    rgb: np.ndarray
    source_shape: tuple[int, ...]
    crop_box_xywh: tuple[int, int, int, int]
    resized_shape: tuple[int, int]
    offset_xy: tuple[int, int]


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(f"Unreadable image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError(f"Invalid image shape for {path}: {image.shape}")
    return image


def crop_pad_resize_bgr(
    bgr: np.ndarray,
    *,
    output_size: int = 224,
    threshold: int = 25,
) -> PreparedImage:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise PreprocessingError(f"Expected HxWx3 BGR image, got shape {bgr.shape}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise PreprocessingError("Image has an empty threshold mask; likely all black")
    x, y, w, h = cv2.boundingRect(coords)
    if w <= 0 or h <= 0:
        raise PreprocessingError("Computed invalid crop dimensions")
    crop = bgr[y : y + h, x : x + w]
    scale = output_size / float(max(w, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    x_off = (output_size - new_w) // 2
    y_off = (output_size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return PreparedImage(
        rgb=rgb,
        source_shape=tuple(int(v) for v in bgr.shape),
        crop_box_xywh=(int(x), int(y), int(w), int(h)),
        resized_shape=(int(new_h), int(new_w)),
        offset_xy=(int(x_off), int(y_off)),
    )


def prepare_rgb_from_path(
    path: str | Path,
    *,
    output_size: int = 224,
    threshold: int = 25,
) -> PreparedImage:
    return crop_pad_resize_bgr(read_bgr(path), output_size=output_size, threshold=threshold)


def write_rgb_png(rgb: np.ndarray, path: str | Path) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PreprocessingError(f"Expected RGB HxWx3 image, got {rgb.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise PreprocessingError(f"Failed to write image: {target}")
