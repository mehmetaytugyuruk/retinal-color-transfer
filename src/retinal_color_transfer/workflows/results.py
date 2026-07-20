from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from retinal_color_transfer.artifacts import resolve_model_dir, write_csv
from retinal_color_transfer.config import RepresentationConfig
from retinal_color_transfer.data import CachedRegressionDataset, load_manifest
from retinal_color_transfer.evaluation import (
    prediction_rows,
    regression_metrics,
    validate_checkpoint_compatibility,
)
from retinal_color_transfer.model import build_resnet50_regressor
from retinal_color_transfer.preprocessing.cache import (
    resolve_representation_cache_root,
    validate_cache_for_manifest,
)
from retinal_color_transfer.preprocessing.normalization import (
    validate_normalization_compatibility,
)
from retinal_color_transfer.reproducibility import select_device
from retinal_color_transfer.study import CANONICAL_MODEL_IDS
from retinal_color_transfer.training.objectives import TargetStatistics
from retinal_color_transfer.training.transforms import eval_transform

COLOR4_COLUMNS = ["pred_rgb", "pred_lab", "pred_hsv", "pred_ycrcb"]
RGB4_COLUMNS = [
    "pred_rgb_seed43",
    "pred_rgb_seed44",
    "pred_rgb_seed45",
    "pred_rgb_seed46",
]
MODEL_IDS = CANONICAL_MODEL_IDS
CUSTOM3_COLUMNS = (
    "pred_custom_lab_b_rgb_g_rgb_b",
    "pred_custom_lab_b_rgb_g_hsv_s",
    "pred_custom_lab_a_rgb_g_lab_b",
)
METRIC_COLUMNS = (
    "pred_dummy_train_mean",
    "pred_rgb",
    *RGB4_COLUMNS,
    "pred_rgb4_equal",
    "pred_grayscale",
    "pred_lab",
    "pred_hsv",
    "pred_ycrcb",
    "pred_color4_equal",
    "pred_rgb_r",
    "pred_rgb_g",
    "pred_rgb_b",
    "pred_lab_l",
    "pred_lab_a",
    "pred_lab_b",
    "pred_hsv_h",
    "pred_hsv_s",
    "pred_hsv_v",
    "pred_ycrcb_y",
    "pred_ycrcb_cr",
    "pred_ycrcb_cb",
    *CUSTOM3_COLUMNS,
)
PREDICTION_ROLES = ("validation", "test")
PREDICTION_COLUMNS = {
    "image_id",
    "patient_id",
    "image_path",
    "age_true",
    "age_pred",
    "absolute_error",
    "split",
}
PREDICTION_KEY_COLUMNS = [
    "image_id",
    "patient_id",
    "image_path",
    "age_true",
    "split",
]


def _prediction_name(model_id: str) -> str:
    return model_id.removesuffix("_seed42")


def load_model_predictions(
    model_root: Path,
    model_ids: list[str] | tuple[str, ...],
    split: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model_id in model_ids:
        path = resolve_model_dir(model_root, model_id) / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing model predictions: {path}")
        frame = pd.read_csv(path)
        required = PREDICTION_KEY_COLUMNS + ["age_pred"]
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame = frame.loc[frame["split"] == split, required].copy()
        if frame.empty:
            raise ValueError(f"{path} has no rows for split={split}")
        prediction_column = f"pred_{_prediction_name(model_id)}"
        frame = frame.rename(columns={"age_pred": prediction_column})
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on=PREDICTION_KEY_COLUMNS, how="inner")
        )
    if merged is None:
        raise ValueError("At least one model ID is required")
    return merged


def _validate_predictions(
    path: Path,
    expected_split_counts: dict[str, int],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {', '.join(missing)}")
    counts = frame["split"].value_counts().to_dict()
    if counts != expected_split_counts:
        raise ValueError(
            f"{path} has split counts {counts}; expected {expected_split_counts}"
        )
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate image IDs")
    return frame


def _evaluate_split(
    *,
    model_dir: Path,
    role: str,
    manifest_dir: Path,
    data_root: Path,
    device: str,
    batch_size: int,
) -> Path:
    output = model_dir / f".{role}_predictions.tmp.csv"
    checkpoint = torch.load(
        model_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = checkpoint["resolved_config"]
    rep_cfg = RepresentationConfig.from_file(
        cfg["representation_config"],
        require_approved=True,
    )
    norm = checkpoint["normalization"]["statistics"]
    validate_normalization_compatibility(norm, rep_cfg)
    validate_checkpoint_compatibility(checkpoint, cfg, norm, rep_cfg)
    target = checkpoint["target_statistics"]
    stats = TargetStatistics(
        mean=float(target["mean"]),
        sample_std=float(target["sample_std"]),
        min_age=int(target["min_age"]),
        max_age=int(target["max_age"]),
        n=int(target["n"]),
    )
    manifest = load_manifest(
        manifest_dir / f"{role}.csv",
        role=role,
        data_root=data_root,
    )
    representation_cache_root = resolve_representation_cache_root(
        cfg["cache"]["representation_root"],
        rep_cfg,
    )
    validate_cache_for_manifest(
        manifest.frame,
        cache_root=representation_cache_root,
        cfg=rep_cfg,
    )
    dataset = CachedRegressionDataset(
        manifest.frame,
        cache_root=representation_cache_root,
        representation_config=rep_cfg,
        age_mean=stats.mean,
        age_std=stats.sample_std,
        channel_mean=norm["channel_mean"],
        channel_std=norm["channel_std"],
        transform=eval_transform(rep_cfg),
        normalization_fingerprint=norm["normalization_fingerprint"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_resnet50_regressor(weights=None, allow_weight_download=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_device = select_device(device)
    model.to(selected_device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            pred_norm = (
                model(batch["image"].to(selected_device))
                .squeeze(1)
                .cpu()
                .tolist()
            )
            predictions.extend(
                [value * stats.sample_std + stats.mean for value in pred_norm]
            )
    rows = prediction_rows(manifest.frame, predictions)
    write_csv(
        rows,
        output,
        [
            "image_id",
            "patient_id",
            "image_path",
            "age_true",
            "age_pred",
            "absolute_error",
            "split",
        ],
    )
    metrics = regression_metrics(
        manifest.frame["age"].astype(float).to_numpy(),
        predictions,
    )
    print(
        f"Evaluated {model_dir.name} on {role}: "
        f"MAE={metrics['mae']:.6f}",
        flush=True,
    )
    return output


def ensure_predictions(
    *,
    model_root: Path,
    manifest_dir: Path,
    data_root: Path,
    device: str,
    batch_size: int,
    model_ids: tuple[str, ...] = MODEL_IDS,
) -> dict[str, str]:
    expected_split_counts = {
        role: len(pd.read_csv(manifest_dir / f"{role}.csv"))
        for role in PREDICTION_ROLES
    }
    actions = {}
    for model_id in model_ids:
        model_dir = resolve_model_dir(model_root, model_id)
        prediction_path = model_dir / "predictions.csv"
        if prediction_path.is_file():
            _validate_predictions(prediction_path, expected_split_counts)
            actions[model_id] = "reused"
            print(f"Reusing valid predictions: {prediction_path}")
            continue
        if not (model_dir / "best_checkpoint.pt").is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {model_dir}")
        if not (model_dir / "training_history.csv").is_file():
            raise FileNotFoundError(f"Missing training history: {model_dir}")
        temporary_paths = []
        combined_temporary_path = model_dir / ".predictions.tmp.csv"
        try:
            for role in PREDICTION_ROLES:
                temporary_paths.append(
                    _evaluate_split(
                        model_dir=model_dir,
                        role=role,
                        manifest_dir=manifest_dir,
                        data_root=data_root,
                        device=device,
                        batch_size=batch_size,
                    )
                )
            predictions = pd.concat(
                [pd.read_csv(path) for path in temporary_paths],
                ignore_index=True,
            )
            predictions.to_csv(combined_temporary_path, index=False)
            _validate_predictions(combined_temporary_path, expected_split_counts)
            combined_temporary_path.replace(prediction_path)
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
            combined_temporary_path.unlink(missing_ok=True)
        actions[model_id] = "created"
        print(f"Wrote {prediction_path} ({len(predictions)} rows)")
    return actions


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(pred - y_true))),
    }


def _name(column: str) -> str:
    return column.removeprefix("pred_")


def _add_derived_predictions(frame: pd.DataFrame, train_mean_age: float) -> pd.DataFrame:
    out = frame.copy()
    out["pred_dummy_train_mean"] = float(train_mean_age)
    out["pred_color4_equal"] = out[COLOR4_COLUMNS].mean(axis=1)
    out["pred_rgb4_equal"] = out[RGB4_COLUMNS].mean(axis=1)
    return out


def _metric_rows(frame: pd.DataFrame, split: str) -> list[dict[str, object]]:
    y_true = frame["age_true"].to_numpy(dtype=np.float64)
    rows = []
    for column in METRIC_COLUMNS:
        if column not in frame.columns:
            continue
        rows.append(
            {
                "split": split,
                "model": _name(column),
                **_metrics(y_true, frame[column].to_numpy(dtype=np.float64)),
            }
        )
    return rows


def _select_custom3(validation_frame: pd.DataFrame) -> tuple[str, float]:
    y_true = validation_frame["age_true"].to_numpy(dtype=np.float64)
    candidates = {
        column: float(
            np.mean(
                np.abs(
                    validation_frame[column].to_numpy(dtype=np.float64) - y_true
                )
            )
        )
        for column in CUSTOM3_COLUMNS
    }
    return min(candidates.items(), key=lambda item: (item[1], item[0]))


def _paired_bootstrap(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    y_true = frame["age_true"].to_numpy(dtype=np.float64)
    baseline_abs = np.abs(frame[baseline_column].to_numpy(dtype=np.float64) - y_true)
    candidate_abs = np.abs(frame[candidate_column].to_numpy(dtype=np.float64) - y_true)
    observed = float(np.mean(baseline_abs) - np.mean(candidate_abs))
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = rng.integers(0, len(frame), size=len(frame))
        boot[index] = float(np.mean(baseline_abs[sample]) - np.mean(candidate_abs[sample]))
    return {
        "bootstrap_unit": "image",
        "baseline": _name(baseline_column),
        "candidate": _name(candidate_column),
        "baseline_mae": float(np.mean(baseline_abs)),
        "candidate_mae": float(np.mean(candidate_abs)),
        "observed_mae_improvement_years": observed,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_improvement": float(np.mean(boot > 0.0)),
        "bootstrap_samples": int(samples),
    }


def ensemble_mae(members: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(members.mean(axis=1) - y_true)))


def diversity(members: np.ndarray) -> float:
    return float(np.mean(np.var(members, axis=1)))


def ambiguity_decomposition(
    members: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    ensemble_mse = float(np.mean((members.mean(axis=1) - y_true) ** 2))
    mean_member_mse = float(
        np.mean(
            [
                np.mean((members[:, index] - y_true) ** 2)
                for index in range(members.shape[1])
            ]
        )
    )
    member_diversity = diversity(members)
    return {
        "mean_member_mse": mean_member_mse,
        "diversity": member_diversity,
        "ensemble_mse": ensemble_mse,
        "ensemble_mae": ensemble_mae(members, y_true),
        "identity_residual": abs(
            ensemble_mse - (mean_member_mse - member_diversity)
        ),
    }


def paired_diversity_bootstrap(
    color_abs: np.ndarray,
    rgb_abs: np.ndarray,
    color_diversity_per_image: np.ndarray,
    rgb_diversity_per_image: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    observed_mae = float(rgb_abs.mean() - color_abs.mean())
    observed_diversity = float(
        color_diversity_per_image.mean() - rgb_diversity_per_image.mean()
    )
    bootstrap_mae = np.empty(samples)
    bootstrap_diversity = np.empty(samples)
    for index in range(samples):
        sampled = rng.integers(0, len(color_abs), size=len(color_abs))
        bootstrap_mae[index] = (
            rgb_abs[sampled].mean() - color_abs[sampled].mean()
        )
        bootstrap_diversity[index] = (
            color_diversity_per_image[sampled].mean()
            - rgb_diversity_per_image[sampled].mean()
        )
    return {
        "bootstrap_unit": "image",
        "mae_improvement_years": observed_mae,
        "mae_ci95_low": float(np.quantile(bootstrap_mae, 0.025)),
        "mae_ci95_high": float(np.quantile(bootstrap_mae, 0.975)),
        "mae_prob_improvement": float(np.mean(bootstrap_mae > 0.0)),
        "diversity_gain": observed_diversity,
        "diversity_ci95_low": float(np.quantile(bootstrap_diversity, 0.025)),
        "diversity_ci95_high": float(np.quantile(bootstrap_diversity, 0.975)),
        "diversity_prob_gain": float(np.mean(bootstrap_diversity > 0.0)),
        "bootstrap_samples": int(samples),
    }


def canonical_diversity_analysis(
    frame: pd.DataFrame,
    *,
    split: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    y_true = frame["age_true"].to_numpy(dtype=np.float64)
    color_members = frame[COLOR4_COLUMNS].to_numpy(dtype=np.float64)
    rgb4_members = frame[RGB4_COLUMNS].to_numpy(dtype=np.float64)

    color_absolute_error = np.abs(color_members.mean(axis=1) - y_true)
    rgb4_absolute_error = np.abs(rgb4_members.mean(axis=1) - y_true)
    color_diversity_per_image = np.var(color_members, axis=1)
    rgb4_diversity_per_image = np.var(rgb4_members, axis=1)
    bootstrap = paired_diversity_bootstrap(
        color_absolute_error,
        rgb4_absolute_error,
        color_diversity_per_image,
        rgb4_diversity_per_image,
        samples=samples,
        seed=seed,
    )
    return {
        "split": split,
        "n_images": int(len(frame)),
        "color4_canonical": ambiguity_decomposition(color_members, y_true),
        "rgb4_independent": ambiguity_decomposition(rgb4_members, y_true),
        "color4_members": ["rgb_seed42", "lab_seed42", "hsv_seed42", "ycrcb_seed42"],
        "rgb4_members": ["rgb_seed43", "rgb_seed44", "rgb_seed45", "rgb_seed46"],
        "diversity_ratio_color4_vs_rgb4": float(
            diversity(color_members) / diversity(rgb4_members)
        ),
        "bootstrap_color4_vs_rgb4": bootstrap,
    }


def _comparison_rows(
    frame: pd.DataFrame,
    split: str,
    *,
    selected_custom3_column: str,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    pairs = [
        ("pred_rgb4_equal", "pred_color4_equal"),
        ("pred_rgb", selected_custom3_column),
    ]
    return [
        {
            "split": split,
            **_paired_bootstrap(
                frame,
                baseline,
                candidate,
                samples=samples,
                seed=seed + index,
            ),
        }
        for index, (baseline, candidate) in enumerate(pairs)
    ]


def _split_diagnostics(frame: pd.DataFrame, split: str) -> dict[str, object]:
    counts = frame["patient_id"].value_counts()
    return {
        "split": split,
        "images": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "max_images_per_patient": int(counts.max()),
        "patients_with_multiple_images": int((counts > 1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create missing model predictions, then produce the canonical "
            "metrics, bootstrap, and ensemble-diversity analyses."
        )
    )
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/final_results"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    prediction_actions = ensure_predictions(
        model_root=args.model_root,
        manifest_dir=args.manifest_dir,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_mean_age = float(
        pd.read_csv(args.manifest_dir / "train.csv")["age"].mean()
    )
    frames = {}
    diagnostics = []
    metrics = []
    for split in ("validation", "test"):
        frame = load_model_predictions(args.model_root, MODEL_IDS, split)
        frame = _add_derived_predictions(frame, train_mean_age)
        frames[split] = frame
        diagnostics.append(_split_diagnostics(frame, split))
        metrics.extend(_metric_rows(frame, split))

    selected_custom3_column, selected_custom3_validation_mae = _select_custom3(
        frames["validation"]
    )
    comparisons = []
    for split, frame in frames.items():
        comparisons.extend(
            _comparison_rows(
                frame,
                split,
                selected_custom3_column=selected_custom3_column,
                samples=args.bootstrap_samples,
                seed=args.seed + (100000 if split == "validation" else 0),
            )
        )
    diversity_results = {
        split: canonical_diversity_analysis(
            frame,
            split=split,
            samples=args.bootstrap_samples,
            seed=args.seed + (100000 if split == "validation" else 0),
        )
        for split, frame in frames.items()
    }

    metrics_frame = pd.DataFrame(metrics)
    comparisons_frame = pd.DataFrame(comparisons)
    diagnostics_frame = pd.DataFrame(diagnostics)
    metrics_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)
    comparisons_frame.to_csv(
        args.output_dir / "paired_bootstrap_image.csv",
        index=False,
    )
    diagnostics_frame.to_csv(args.output_dir / "split_patient_diagnostics.csv", index=False)
    with (args.output_dir / "canonical_ensemble_diversity.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(diversity_results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary = {
        "prediction_actions": prediction_actions,
        "train_mean_age_baseline": train_mean_age,
        "split_patient_diagnostics": diagnostics,
        "custom3_selection": {
            "criterion": "lowest validation MAE",
            "selected_model": _name(selected_custom3_column),
            "validation_mae": selected_custom3_validation_mae,
        },
        "validation_metrics": metrics_frame.query("split == 'validation'")
        .sort_values("mae")
        .to_dict(orient="records"),
        "test_metrics_selected": metrics_frame.query("split == 'test'")
        .sort_values("mae")
        .to_dict(orient="records"),
        "test_comparisons": comparisons_frame.query("split == 'test'").to_dict(orient="records"),
        "canonical_ensemble_diversity": diversity_results,
    }
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Patient/image diagnostics:")
    print(diagnostics_frame.to_string(index=False))
    print("\nSelected test metrics:")
    selected = [
        "dummy_train_mean",
        "rgb4_equal",
        "rgb",
        "grayscale",
        "color4_equal",
        _name(selected_custom3_column),
    ]
    selected_metrics = metrics_frame.loc[
        (metrics_frame["split"] == "test") & metrics_frame["model"].isin(selected)
    ]
    print(selected_metrics.sort_values("mae").to_string(index=False))
    print("\nTest comparisons:")
    print(comparisons_frame.query("split == 'test'").to_string(index=False))
    test_diversity = diversity_results["test"]
    bootstrap = test_diversity["bootstrap_color4_vs_rgb4"]
    print("\nCanonical Color4 vs RGB4 diversity:")
    print(
        f"  diversity ratio={test_diversity['diversity_ratio_color4_vs_rgb4']:.2f}x "
        f"gain={bootstrap['diversity_gain']:.4f} "
        f"CI[{bootstrap['diversity_ci95_low']:.4f},"
        f"{bootstrap['diversity_ci95_high']:.4f}] "
        f"P={bootstrap['diversity_prob_gain']:.3f}"
    )
    print(f"\nWrote final analysis artifacts to {args.output_dir}")
