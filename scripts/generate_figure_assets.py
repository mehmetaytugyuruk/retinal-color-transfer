"""
Generate figure assets for the color-space visualization figures.

Produces two groups of images from a single retinal fundus PNG:

  Group 1 — pseudo-color composites (assets/group1_false_color/)
    The three encoded channels of each color space are mapped to the red, green,
    and blue display channels, respectively, to construct a pseudo-color image.
    Channel assignment:

        HSV   : H → R,  S  → G,  V  → B
        CIELAB: L → R,  a  → G,  b  → B
        YCrCb : Y → R,  Cr → G,  Cb → B

    The displayed colors do not represent natural retinal colors.  They are
    intended only to visualize differences in channel encoding.

  Group 2 — single-channel grayscale (assets/group2_channels/)
    Each individual channel is saved as a grayscale image so that its
    information content can be assessed independently.

Preprocessing applied to every output image
-------------------------------------------
1. Background mask  — pixels where max(B,G,R) < 10 in the *original* image are
   forced to black after conversion.  This prevents encoding artefacts (e.g.
   Lab's a/b offset of 128 turning black pixels green/grey) from polluting the
   background.
2. Crop             — the bounding rectangle of non-background pixels is
   computed once from the original BGR image (threshold = 25, matching the
   training pipeline in crop_pad_resize.py) and applied to every output.
   No resize is performed; the crop preserves the original pixel density.

Channel-range notes
-------------------
All channels are stored in [0, 255] by OpenCV *except* HSV-H, which uses
[0, 179].  Only HSV-H is linearly mapped from its fixed OpenCV range [0, 179]
to the display range [0, 255]:

    H_display = H × 255 / 179

Per-image auto-contrast (min-max normalization) is intentionally avoided: it
would misrepresent low-variance channels (e.g. H, which carries little retinal
discriminative information) as artificially detail-rich — contradicting the
quantitative MAE results.

Usage
-----
    python scripts/generate_figure_assets.py --source img00007.png

The output is written to assets/ which is listed in .gitignore.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CROP_THRESHOLD = 25   # matches training pipeline (crop_pad_resize.py)
BG_THRESHOLD   = 10   # max(B,G,R) below this → background


def _compute_crop_box(bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, CROP_THRESHOLD, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("Image appears entirely black after thresholding.")
    x, y, w, h = cv2.boundingRect(coords)
    return int(x), int(y), int(w), int(h)


def _bg_mask(bgr: np.ndarray) -> np.ndarray:
    """True where a pixel is background (max channel < BG_THRESHOLD)."""
    return bgr.max(axis=2) < BG_THRESHOLD


def _save_false_color(
    channels_rgb: np.ndarray,
    mask_bg: np.ndarray,
    crop: tuple[int, int, int, int],
    path: Path,
) -> None:
    """Save a pseudo-color composite.

    `channels_rgb` is treated as an RGB array: channel 0 → Red display,
    channel 1 → Green display, channel 2 → Blue display.
    cv2.imwrite expects BGR, so a COLOR_RGB2BGR conversion is applied before
    writing.
    """
    x, y, w, h = crop
    out = channels_rgb.copy()
    out[mask_bg] = 0
    cropped_bgr = cv2.cvtColor(out[y:y + h, x:x + w], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), cropped_bgr):
        raise OSError(f"Could not write image: {path}")


def _save_1ch(
    img: np.ndarray,
    mask_bg: np.ndarray,
    crop: tuple[int, int, int, int],
    path: Path,
) -> None:
    """Save a single-channel grayscale image."""
    x, y, w, h = crop
    out = img.copy()
    out[mask_bg] = 0
    if not cv2.imwrite(str(path), out[y:y + h, x:x + w]):
        raise OSError(f"Could not write image: {path}")


def generate(source: Path, output_root: Path) -> None:
    bgr = cv2.imread(str(source))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {source}")

    print(f"Source : {source}  shape={bgr.shape}")

    crop    = _compute_crop_box(bgr)
    mask_bg = _bg_mask(bgr)
    x, y, w, h = crop
    print(f"Crop   : x={x} y={y} w={w} h={h}  →  output size {h}×{w}")

    # Colour-space conversions
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # HSV-H: fixed theoretical-range mapping [0, 179] → [0, 255]
    h_stretched = (
        hsv[:, :, 0].astype(np.float32) * 255.0 / 179.0
    ).clip(0, 255).astype(np.uint8)

    # Pseudo-color composites with explicit channel assignment:
    #   HSV:   H → R, S  → G, V  → B
    #   Lab:   L → R, a  → G, b  → B
    #   YCrCb: Y → R, Cr → G, Cb → B
    hsv_fc   = cv2.merge((h_stretched, hsv[:, :, 1],   hsv[:, :, 2]))
    lab_fc   = cv2.merge((lab[:, :, 0],   lab[:, :, 1],   lab[:, :, 2]))
    ycrcb_fc = cv2.merge((ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]))

    # ── Group 1: pseudo-color composites ──────────────────────────────────
    g1 = output_root / "group1_false_color"
    g1.mkdir(parents=True, exist_ok=True)

    _save_false_color(rgb,      mask_bg, crop, g1 / "rgb.png")
    _save_1ch        (gray,     mask_bg, crop, g1 / "gray.png")
    _save_false_color(lab_fc,   mask_bg, crop, g1 / "lab.png")
    _save_false_color(hsv_fc,   mask_bg, crop, g1 / "hsv.png")
    _save_false_color(ycrcb_fc, mask_bg, crop, g1 / "ycrcb.png")

    print("Group 1 written ✓")

    # ── Group 2: single-channel grayscale ─────────────────────────────────
    g2 = output_root / "group2_channels"
    g2.mkdir(parents=True, exist_ok=True)

    _save_1ch(rgb[:, :, 0],    mask_bg, crop, g2 / "rgb_r.png")
    _save_1ch(rgb[:, :, 1],    mask_bg, crop, g2 / "rgb_g.png")
    _save_1ch(rgb[:, :, 2],    mask_bg, crop, g2 / "rgb_b.png")
    _save_1ch(lab[:, :, 0],    mask_bg, crop, g2 / "lab_l.png")
    _save_1ch(lab[:, :, 1],    mask_bg, crop, g2 / "lab_a.png")
    _save_1ch(lab[:, :, 2],    mask_bg, crop, g2 / "lab_b.png")
    _save_1ch(h_stretched,     mask_bg, crop, g2 / "hsv_h.png")
    _save_1ch(hsv[:, :, 1],    mask_bg, crop, g2 / "hsv_s.png")
    _save_1ch(hsv[:, :, 2],    mask_bg, crop, g2 / "hsv_v.png")
    _save_1ch(ycrcb[:, :, 0],  mask_bg, crop, g2 / "ycrcb_y.png")
    _save_1ch(ycrcb[:, :, 1],  mask_bg, crop, g2 / "ycrcb_cr.png")
    _save_1ch(ycrcb[:, :, 2],  mask_bg, crop, g2 / "ycrcb_cb.png")

    print("Group 2 written ✓")
    print(f"All assets in: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate color-space figure assets from a retinal fundus image."
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Path to the source retinal fundus PNG/JPG.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("assets"),
        help="Root directory for output assets (default: assets/).",
    )
    args = parser.parse_args()
    generate(args.source, args.output_dir)


if __name__ == "__main__":
    main()
