from __future__ import annotations

import numpy as np

from retinal_color_transfer.config import RepresentationConfig, stable_fingerprint


def validate_checkpoint_compatibility(
    checkpoint: dict, cfg: dict, norm: dict, rep_cfg: RepresentationConfig
) -> None:
    required = {
        "model_state_dict",
        "target_statistics",
        "model_selection",
        "resolved_config",
        "model",
        "preprocessing_fingerprint",
        "representation",
        "normalization",
        "config_fingerprint",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint missing required metadata: {', '.join(missing)}")
    if checkpoint["config_fingerprint"] != stable_fingerprint(checkpoint["resolved_config"]):
        raise ValueError("Checkpoint config fingerprint does not match its resolved config")
    if checkpoint["resolved_config"].get("model", {}).get("architecture") != cfg.get(
        "model", {}
    ).get("architecture"):
        raise ValueError("Checkpoint architecture is incompatible with resolved config")
    if checkpoint["model"].get("head") != "previous_resnet50_regression_head_v1":
        raise ValueError("Checkpoint head identity is incompatible")
    if checkpoint["representation"].get("name") != rep_cfg.name:
        raise ValueError("Checkpoint representation name mismatch")
    if checkpoint["representation"].get("fingerprint") != rep_cfg.fingerprint:
        raise ValueError("Checkpoint representation fingerprint mismatch")
    if checkpoint["normalization"].get("fingerprint") != norm.get("normalization_fingerprint"):
        raise ValueError("Checkpoint normalization fingerprint mismatch")
    for key in ("mean", "sample_std", "min_age", "max_age", "n"):
        if key not in checkpoint["target_statistics"]:
            raise ValueError(f"Checkpoint target statistics missing {key}")
    expected_weights = cfg.get("model", {}).get("weights")
    if checkpoint["model"].get("weights") != expected_weights:
        raise ValueError("Checkpoint model weights are incompatible with resolved config")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Regression metrics require finite targets and predictions")
    errors = pred - true
    mae = float(np.mean(np.abs(errors)))
    return {"mae": mae}


def prediction_rows(frame, predictions) -> list[dict]:
    rows = []
    for row, pred in zip(frame.itertuples(index=False), predictions, strict=True):
        age_true = float(row.age)
        age_pred = float(pred)
        rows.append(
            {
                "image_id": row.image_id,
                "patient_id": row.patient_id,
                "image_path": row.image_path,
                "age_true": age_true,
                "age_pred": age_pred,
                "absolute_error": abs(age_pred - age_true),
                "split": row.split,
            }
        )
    return rows
