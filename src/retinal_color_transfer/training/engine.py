from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from retinal_color_transfer.artifacts import ensure_new_dir, write_csv
from retinal_color_transfer.config import (
    RepresentationConfig,
    load_yaml,
    stable_fingerprint,
    validate_experiment_config,
)
from retinal_color_transfer.data import (
    CachedRegressionDataset,
    load_manifest,
    validate_split_collection,
)
from retinal_color_transfer.evaluation import regression_metrics
from retinal_color_transfer.model import build_resnet50_regressor
from retinal_color_transfer.preprocessing.cache import (
    resolve_representation_cache_root,
    validate_cache_for_manifest,
)
from retinal_color_transfer.preprocessing.crop_pad_resize import PREPROCESSING_VERSION
from retinal_color_transfer.preprocessing.normalization import validate_normalization_compatibility
from retinal_color_transfer.representations.contracts import REPRESENTATION_IMPLEMENTATION_VERSION
from retinal_color_transfer.reproducibility import (
    runtime_info,
    seed_everything,
    seed_worker,
    select_device,
)
from retinal_color_transfer.training.objectives import (
    TargetStatistics,
    denormalize_age,
    lds_weights,
    target_statistics,
    validation_smooth_l1,
    weighted_smooth_l1,
)
from retinal_color_transfer.training.transforms import eval_transform, train_transform


def _require(value, name: str):
    if value in {None, ""}:
        raise ValueError(f"Experiment config must set {name}")
    return value


def accumulate_sample_mean(
    total: float,
    count: int,
    batch_mean: float,
    batch_size: int,
) -> tuple[float, int]:
    return total + batch_mean * batch_size, count + batch_size


def is_strict_improvement(value: float, best: float, *, strict: bool = True) -> bool:
    return value < best if strict else value <= best


def step_scheduler_on_validation_loss(scheduler, validation_loss: float) -> None:
    scheduler.step(validation_loss)


def _clear_training_outputs(out_dir: Path) -> None:
    for name in [
        "best_checkpoint.pt",
        "data_summary.json",
        "environment.json",
        "latest_checkpoint.pt",
        "normalization_statistics.json",
        "predictions.csv",
        "resolved_config.yaml",
        "summary.json",
        "target_statistics.json",
        "training_history.csv",
    ]:
        (out_dir / name).unlink(missing_ok=True)


def _clear_completed_training_transients(out_dir: Path) -> None:
    for name in [
        "data_summary.json",
        "environment.json",
        "latest_checkpoint.pt",
        "normalization_statistics.json",
        "predictions.csv",
        "resolved_config.yaml",
        "summary.json",
        "target_statistics.json",
    ]:
        (out_dir / name).unlink(missing_ok=True)


def _autocast_config(device: torch.device, runtime: dict) -> tuple[bool, torch.dtype | None, str]:
    requested = str(runtime.get("mixed_precision", "none")).lower()
    if requested not in {"none", "fp16", "bf16"}:
        raise ValueError("runtime.mixed_precision must be one of: none, fp16, bf16")
    if requested == "none":
        return False, None, requested
    if device.type != "cuda":
        print(
            f"runtime.mixed_precision={requested} requested, but device is {device}; "
            "mixed precision is enabled only for CUDA training.",
            flush=True,
        )
        return False, None, requested
    dtype = torch.float16 if requested == "fp16" else torch.bfloat16
    return True, dtype, requested


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.amp.GradScaler(enabled=enabled)


def _image_to_device(
    image: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool,
    channels_last: bool,
) -> torch.Tensor:
    image = image.to(device, non_blocking=non_blocking)
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
    return image


def run_training(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
    resume_if_available: bool = False,
) -> dict:
    if overwrite and (resume or resume_if_available):
        raise ValueError("--overwrite cannot be combined with resume options")
    cfg = load_yaml(config_path)
    validate_experiment_config(cfg)
    rep_cfg = RepresentationConfig.from_file(cfg["representation_config"], require_approved=True)
    data_cfg = cfg.get("data", {})
    data_root = _require(data_cfg.get("data_root"), "data.data_root")
    train_manifest = load_manifest(
        _require(data_cfg.get("train_manifest"), "data.train_manifest"),
        role="train",
        data_root=data_root,
    )
    val_manifest = load_manifest(
        _require(data_cfg.get("val_manifest"), "data.val_manifest"),
        role="validation",
        data_root=data_root,
    )
    test_path = data_cfg.get("test_manifest")
    manifests = [train_manifest, val_manifest]
    if test_path:
        manifests.append(load_manifest(test_path, role="test", data_root=data_root))
    if len(manifests) == 3:
        validate_split_collection(manifests)

    norm_path = _require(
        cfg.get("normalization", {}).get("statistics_path"),
        "normalization.statistics_path",
    )
    if str(norm_path).endswith((".yaml", ".yml")):
        norm = load_yaml(norm_path)
    else:
        with Path(norm_path).open("r", encoding="utf-8") as handle:
            norm = json.load(handle)
    validate_normalization_compatibility(norm, rep_cfg)
    representation_cache_root = resolve_representation_cache_root(
        cfg["cache"]["representation_root"],
        rep_cfg,
    )
    validate_cache_for_manifest(
        train_manifest.frame,
        cache_root=representation_cache_root,
        cfg=rep_cfg,
    )
    validate_cache_for_manifest(
        val_manifest.frame,
        cache_root=representation_cache_root,
        cfg=rep_cfg,
    )
    stats = target_statistics(train_manifest.frame["age"])
    lds_cfg = cfg.get("lds", {})
    weights = lds_weights(
        train_manifest.frame["age"],
        sigma=float(lds_cfg.get("sigma", 2.0)),
        mode=lds_cfg.get("mode", "reflect"),
        truncate=float(lds_cfg.get("truncate", 4.0)),
        epsilon=float(lds_cfg.get("epsilon", 1.0e-5)),
    )
    runtime = cfg.get("runtime", {})
    seed = int(runtime.get("seed", 42))
    generator = seed_everything(seed)
    device = select_device(runtime.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", False))
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(runtime.get("allow_tf32", True))

    out_dir_path = Path(cfg["experiment"]["output_dir"])
    latest_checkpoint_path = out_dir_path / "latest_checkpoint.pt"
    should_resume = resume or (resume_if_available and latest_checkpoint_path.is_file())
    if should_resume:
        if not latest_checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume; missing {latest_checkpoint_path}")
        out_dir = out_dir_path
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ensure_new_dir(out_dir_path, overwrite=overwrite)
        if overwrite:
            _clear_training_outputs(out_dir)
    opt_cfg = cfg.get("optimization", {})
    num_workers = int(runtime.get("num_workers", 2))
    loader_worker_args = (
        {"persistent_workers": True, "prefetch_factor": int(runtime.get("prefetch_factor", 2))}
        if num_workers > 0
        else {}
    )
    train_ds = CachedRegressionDataset(
        train_manifest.frame,
        cache_root=representation_cache_root,
        representation_config=rep_cfg,
        age_mean=stats.mean,
        age_std=stats.sample_std,
        channel_mean=norm["channel_mean"],
        channel_std=norm["channel_std"],
        transform=train_transform(rep_cfg),
        sample_weights=weights,
        normalization_fingerprint=norm["normalization_fingerprint"],
    )
    val_ds = CachedRegressionDataset(
        val_manifest.frame,
        cache_root=representation_cache_root,
        representation_config=rep_cfg,
        age_mean=stats.mean,
        age_std=stats.sample_std,
        channel_mean=norm["channel_mean"],
        channel_std=norm["channel_std"],
        transform=eval_transform(rep_cfg),
        normalization_fingerprint=norm["normalization_fingerprint"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(opt_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(runtime.get("pin_memory", False)),
        generator=generator,
        worker_init_fn=seed_worker,
        **loader_worker_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(opt_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(runtime.get("pin_memory", False)),
        worker_init_fn=seed_worker,
        **loader_worker_args,
    )

    model_cfg = cfg.get("model", {})
    model = build_resnet50_regressor(
        weights=model_cfg.get("weights", "imagenet"),
        allow_weight_download=bool(model_cfg.get("allow_weight_download", False)),
    )
    memory_format = str(runtime.get("memory_format", "contiguous")).lower()
    if memory_format not in {"contiguous", "channels_last"}:
        raise ValueError("runtime.memory_format must be 'contiguous' or 'channels_last'")
    channels_last = memory_format == "channels_last"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
    sched_cfg = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 5)),
        min_lr=float(sched_cfg.get("min_lr", 1e-7)),
    )
    autocast_enabled, autocast_dtype, mixed_precision = _autocast_config(device, runtime)
    scaler = _make_grad_scaler(autocast_enabled and mixed_precision == "fp16")
    non_blocking = bool(runtime.get("non_blocking_transfer", device.type == "cuda"))
    best_mae = float("inf")
    history = []
    epochs = int(opt_cfg.get("epochs", 80))
    start_epoch = 0
    if should_resume:
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        expected_fingerprint = stable_fingerprint(cfg)
        if checkpoint.get("config_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Refusing to resume because the current config fingerprint does not match "
                f"{latest_checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_mae = float(checkpoint.get("best_mae", float("inf")))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"])
    experiment_name = cfg["experiment"]["name"]
    print(
        f"Training {experiment_name} on {device}: "
        f"{len(train_ds)} train / {len(val_ds)} validation examples, "
        f"{len(train_loader)} train batches per epoch, {epochs} epochs, "
        f"starting at epoch {start_epoch + 1}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_examples = 0
        train_progress = tqdm(
            train_loader,
            desc=f"{experiment_name} epoch {epoch + 1}/{epochs} train",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in train_progress:
            optimizer.zero_grad(set_to_none=True)
            image = _image_to_device(
                batch["image"],
                device,
                non_blocking=non_blocking,
                channels_last=channels_last,
            )
            target = batch["target"].to(device, non_blocking=non_blocking)
            weight = batch["weight"].to(device, non_blocking=non_blocking)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                pred = model(image).squeeze(1)
                loss = weighted_smooth_l1(pred, target, weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(opt_cfg.get("gradient_clip_norm", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["target"].numel())
            train_loss, train_examples = accumulate_sample_mean(
                train_loss,
                train_examples,
                float(loss.detach().cpu()),
                batch_size,
            )
            train_progress.set_postfix(
                loss=f"{train_loss / max(train_examples, 1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        val_loss, y_true, y_pred = _evaluate_loader(
            model,
            val_loader,
            device,
            stats,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            non_blocking=non_blocking,
            channels_last=channels_last,
            desc=f"{experiment_name} epoch {epoch + 1}/{epochs} val",
            show_progress=True,
        )
        metrics = regression_metrics(y_true, y_pred)
        step_scheduler_on_validation_loss(scheduler, val_loss)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_examples, 1),
            "val_loss": val_loss,
            **metrics,
        }
        history.append(row)
        write_csv(
            history,
            out_dir / "training_history.csv",
            fieldnames=["epoch", "train_loss", "val_loss", "mae"],
        )
        tqdm.write(
            "Epoch "
            f"{epoch + 1}/{epochs} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"mae={row['mae']:.6f}"
        )
        if is_strict_improvement(metrics["mae"], best_mae):
            best_mae = metrics["mae"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_statistics": asdict(stats),
                    "model_selection": {"metric": "validation_mae", "value": best_mae},
                    "epoch": epoch + 1,
                    "resolved_config": cfg,
                    "model": {
                        "architecture": "resnet50",
                        "head": "previous_resnet50_regression_head_v1",
                        "weights": model_cfg.get("weights", "imagenet"),
                    },
                    "preprocessing_fingerprint": PREPROCESSING_VERSION,
                    "representation": {
                        "name": rep_cfg.name,
                        "fingerprint": rep_cfg.fingerprint,
                        "implementation_version": REPRESENTATION_IMPLEMENTATION_VERSION,
                    },
                    "normalization": {
                        "fingerprint": norm["normalization_fingerprint"],
                        "statistics": norm,
                    },
                    "runtime": runtime_info(device, seed),
                    "config_fingerprint": stable_fingerprint(cfg),
                },
                out_dir / "best_checkpoint.pt",
            )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "target_statistics": asdict(stats),
                "best_mae": best_mae,
                "epoch": epoch + 1,
                "history": history,
                "resolved_config": cfg,
                "runtime": runtime_info(device, seed),
                "config_fingerprint": stable_fingerprint(cfg),
            },
            latest_checkpoint_path,
        )
    best_checkpoint_path = out_dir / "best_checkpoint.pt"
    history_path = out_dir / "training_history.csv"
    if not best_checkpoint_path.is_file() or not history_path.is_file():
        raise RuntimeError(
            f"Training ended without required outputs in {out_dir}: "
            "best_checkpoint.pt and training_history.csv must both exist"
        )
    if len(history) != epochs:
        raise RuntimeError(
            f"Training history is incomplete in {out_dir}: "
            f"expected {epochs} epochs, found {len(history)}"
        )
    _clear_completed_training_transients(out_dir)
    return {
        "best_validation_mae": best_mae,
        "model_selection_metric": "validation_mae",
    }


def _evaluate_loader(
    model,
    loader,
    device,
    stats: TargetStatistics,
    *,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype | None = None,
    non_blocking: bool = False,
    channels_last: bool = False,
    desc: str | None = None,
    show_progress: bool = False,
) -> tuple[float, list[float], list[float]]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    total_loss = 0.0
    total_examples = 0
    batches = (
        tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
        if show_progress
        else loader
    )
    with torch.no_grad():
        for batch in batches:
            image = _image_to_device(
                batch["image"],
                device,
                non_blocking=non_blocking,
                channels_last=channels_last,
            )
            target = batch["target"].to(device, non_blocking=non_blocking)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                pred_norm = model(image).squeeze(1)
                batch_loss = validation_smooth_l1(pred_norm, target)
            batch_size = int(target.numel())
            total_loss, total_examples = accumulate_sample_mean(
                total_loss,
                total_examples,
                float(batch_loss.detach().cpu()),
                batch_size,
            )
            if show_progress:
                batches.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4f}")
            for value in batch["age"].tolist():
                y_true.append(float(value))
            for value in pred_norm.detach().cpu().tolist():
                y_pred.append(denormalize_age(float(value), stats))
    return total_loss / max(total_examples, 1), y_true, y_pred
