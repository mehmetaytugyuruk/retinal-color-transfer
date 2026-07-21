"""
Generate figure assets for the color-space visualization figures.

Produces two groups of images from a single retinal fundus PNG:

  Group 1 — false-color composites (assets/group1_false_color/)
    Each color space's 3 channels are placed directly into R,G,B slots and
    saved as a colour image.  The result is a "false-colour" render that looks
    unnatural but faithfully represents the raw channel values as stored by
    OpenCV.  This is the standard display convention for non-RGB spaces.

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
[0, 179].  Only HSV-H is stretched to [0, 255] using the fixed theoretical
range (H_display = H × 255/179).  Per-image auto-contrast is intentionally
avoided: it would misrepresent low-variance channels (e.g. H, which carries
little retinal information) as artificially rich in detail.

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


def _save_3ch(img: np.ndarray, mask_bg: np.ndarray,
              crop: tuple[int, int, int, int], path: Path) -> None:
    x, y, w, h = crop
    out = img.copy()
    out[mask_bg] = 0
    cv2.imwrite(str(path), out[y:y + h, x:x + w])


def _save_1ch(img: np.ndarray, mask_bg: np.ndarray,
              crop: tuple[int, int, int, int], path: Path) -> None:
    x, y, w, h = crop
    out = img.copy()
    out[mask_bg] = 0
    cv2.imwrite(str(path), out[y:y + h, x:x + w])


def generate(source: Path, output_root: Path) -> None:
    bgr = cv2.imread(str(source))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {source}")

    print(f"Source : {source}  shape={bgr.shape}")

    crop   = _compute_crop_box(bgr)
    mask_bg = _bg_mask(bgr)
    x, y, w, h = crop
    print(f"Crop   : x={x} y={y} w={w} h={h}  →  output size {h}×{w}")

    # Colour-space conversions
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # HSV-H: fixed theoretical-range stretch [0,179] → [0,255]
    h_stretched = (hsv[:, :, 0].astype(np.float32) * 255.0 / 179.0
                   ).clip(0, 255).astype(np.uint8)

    # HSV false-colour composite with stretched H
    hsv_fc = hsv.copy()
    hsv_fc[:, :, 0] = h_stretched

    # ── Group 1: false-colour composites ──────────────────────────────────
    g1 = output_root / "group1_false_color"
    g1.mkdir(parents=True, exist_ok=True)

    _save_3ch(bgr,     mask_bg, crop, g1 / "rgb.png")
    _save_3ch(lab,     mask_bg, crop, g1 / "lab.png")
    _save_3ch(hsv_fc,  mask_bg, crop, g1 / "hsv.png")
    _save_3ch(ycrcb,   mask_bg, crop, g1 / "ycrcb.png")
    _save_1ch(gray,    mask_bg, crop, g1 / "gray.png")

    print("Group 1 written ✓")

    # ── Group 2: single-channel grayscale ─────────────────────────────────
    g2 = output_root / "group2_channels"
    g2.mkdir(parents=True, exist_ok=True)

    _save_1ch(bgr[:, :, 2],   mask_bg, crop, g2 / "rgb_r.png")
    _save_1ch(bgr[:, :, 1],   mask_bg, crop, g2 / "rgb_g.png")
    _save_1ch(bgr[:, :, 0],   mask_bg, crop, g2 / "rgb_b.png")
    _save_1ch(lab[:, :, 0],   mask_bg, crop, g2 / "lab_l.png")
    _save_1ch(lab[:, :, 1],   mask_bg, crop, g2 / "lab_a.png")
    _save_1ch(lab[:, :, 2],   mask_bg, crop, g2 / "lab_b.png")
    _save_1ch(h_stretched,    mask_bg, crop, g2 / "hsv_h.png")
    _save_1ch(hsv[:, :, 1],   mask_bg, crop, g2 / "hsv_s.png")
    _save_1ch(hsv[:, :, 2],   mask_bg, crop, g2 / "hsv_v.png")
    _save_1ch(ycrcb[:, :, 0], mask_bg, crop, g2 / "ycrcb_y.png")
    _save_1ch(ycrcb[:, :, 1], mask_bg, crop, g2 / "ycrcb_cr.png")
    _save_1ch(ycrcb[:, :, 2], mask_bg, crop, g2 / "ycrcb_cb.png")

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
