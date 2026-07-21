from __future__ import annotations

REPRESENTATIONS = (
    "rgb",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "grayscale",
    "lab",
    "lab_l",
    "lab_a",
    "lab_b",
    "hsv",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "ycrcb",
    "ycrcb_y",
    "ycrcb_cr",
    "ycrcb_cb",
)

EXTRA_RGB_SEEDS = (43, 44, 45, 46)

CANONICAL_MODELS = (
    *((representation, 42) for representation in REPRESENTATIONS),
    *(("rgb", seed) for seed in EXTRA_RGB_SEEDS),
)

CANONICAL_MODEL_IDS = tuple(
    f"{representation}_seed{seed}" for representation, seed in CANONICAL_MODELS
)
