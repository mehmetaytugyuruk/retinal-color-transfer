"""
Train the 12-channel early-fusion model (Ensemble²).

Input: RGB (3ch) + Lab (3ch) + HSV (3ch) + YCrCb (3ch) → 12-ch tensor.
Model: ResNet-50 with conv1 adapted 3→12 channels via pretrained-weight
       repetition and 1/4 scaling to preserve activation magnitude.

Reuses the project's existing LDS, loss, and optimisation utilities;
only the multi-cache Dataset and the 12-ch model builder are custom.

Usage (from repo root):
    python scripts/train_fusion12ch.py \\
        --cache-root  /path/to/caches \\
        --model-dir   models/fusion/rgb_lab_hsv_ycrcb_seed42 \\
        [--device cuda] [--epochs 80] [--batch-size 32]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── project utilities (already installed via `pip install -e .`) ───────────
from retinal_color_transfer.model import build_resnet50_12ch_regressor
from retinal_color_transfer.training.objectives import (
    denormalize_age,
    lds_weights as compute_lds_weights,
    target_statistics,
    weighted_smooth_l1,
    validation_smooth_l1,
)

ROOT         = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
NORM_PATH    = (
    ROOT / "data" / "normalization" / "fusion"
    / "rgb_lab_hsv_ycrcb_train_stats.json"
)

# cache sub-paths relative to --cache-root
CACHE_SUBDIRS = {
    "rgb":   "rgb/rgb",
    "lab":   "lab/lab",
    "hsv":   "hsv/hsv",
    "ycrcb": "ycrcb/ycrcb",
}


# ---------------------------------------------------------------------------
# Image loaders (one per colour space)
# ---------------------------------------------------------------------------

def _load_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def load_rgb(path: Path) -> np.ndarray:
    """BGR PNG → HWC float32 [0, 1], RGB order."""
    img = cv2.cvtColor(_load_png(path), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_lab(path: Path) -> np.ndarray:
    """Lab uint8 PNG → HWC float32 [0, 1]."""
    return _load_png(path).astype(np.float32) / 255.0


def load_hsv(path: Path) -> np.ndarray:
    """HSV uint8 PNG → HWC float32 with H/179, S/255, V/255 scaling."""
    img = _load_png(path).astype(np.float32)
    img[:, :, 0] /= 179.0
    img[:, :, 1] /= 255.0
    img[:, :, 2] /= 255.0
    return img


def load_ycrcb(path: Path) -> np.ndarray:
    """YCrCb uint8 PNG → HWC float32 [0, 1]."""
    return _load_png(path).astype(np.float32) / 255.0


LOADERS = {"rgb": load_rgb, "lab": load_lab, "hsv": load_hsv, "ycrcb": load_ycrcb}


# ---------------------------------------------------------------------------
# 12-channel Dataset  (only custom part — 4 caches concatenated)
# ---------------------------------------------------------------------------

class Fusion12ChDataset(Dataset):
    """Loads RGB + Lab + HSV + YCrCb from separate caches → 12-ch tensor."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_root: Path,
        channel_mean: list[float],
        channel_std: list[float],
        age_mean: float,
        age_std: float,
        augment: bool = False,
        lds_weights: dict[int, float] | None = None,
    ) -> None:
        self.frame       = manifest.reset_index(drop=True)
        self.cache_root  = Path(cache_root)
        self.mean        = torch.tensor(channel_mean, dtype=torch.float32).view(12, 1, 1)
        self.std         = torch.tensor(channel_std,  dtype=torch.float32).view(12, 1, 1)
        self.age_mean    = float(age_mean)
        self.age_std     = float(age_std)
        self.augment     = augment
        self.lds_weights = lds_weights  # None → weight=1 for val/test

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row      = self.frame.iloc[idx]
        image_id = str(row["image_id"])

        # Load each colour space and stack → [12, H, W]
        arrays = []
        for rep, subdir in CACHE_SUBDIRS.items():
            path = self.cache_root / subdir / f"{image_id}.png"
            arr  = LOADERS[rep](path)                   # HWC float32
            arrays.append(arr.transpose(2, 0, 1))       # CHW
        tensor = torch.from_numpy(np.concatenate(arrays, axis=0))  # [12, H, W]

        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])   # horizontal flip
            angle = random.uniform(-10.0, 10.0)
            if abs(angle) > 0.5:
                tensor = _rotate_tensor(tensor, angle)

        tensor = (tensor - self.mean) / self.std

        age    = float(row["age"])
        weight = float(self.lds_weights.get(int(age), 1.0)) if self.lds_weights else 1.0
        return {
            "image":    tensor,
            "target":   torch.tensor((age - self.age_mean) / self.age_std, dtype=torch.float32),
            "age":      torch.tensor(age, dtype=torch.float32),
            "weight":   torch.tensor(weight, dtype=torch.float32),
            "image_id": image_id,
        }


def _rotate_tensor(tensor: torch.Tensor, angle_deg: float) -> torch.Tensor:
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    zero = torch.zeros(1).squeeze()
    theta = torch.stack([
        torch.stack([cos_a, -sin_a, zero]),
        torch.stack([sin_a,  cos_a, zero]),
    ]).unsqueeze(0)
    grid = F.affine_grid(theta, tensor.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(
        tensor.unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Training / evaluation loops  (mirror engine.py, using project objectives)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, scaler, device, grad_clip,
    autocast_enabled, autocast_dtype,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            dtype=autocast_dtype, enabled=autocast_enabled):
            preds = model(images).squeeze(1)
            loss  = weighted_smooth_l1(preds, targets, weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        n = targets.numel()
        total_loss += float(loss.detach().cpu()) * n
        total_n    += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model, loader, device, stats,
    autocast_enabled=False, autocast_dtype=None,
) -> tuple[float, list[float], list[float], list[str]]:
    """Returns (smooth_l1_loss, y_true, y_pred, image_ids)."""
    model.eval()
    y_true, y_pred, ids = [], [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type,
                            dtype=autocast_dtype, enabled=autocast_enabled):
            preds      = model(images).squeeze(1)
            batch_loss = validation_smooth_l1(preds, targets)

        n = targets.numel()
        total_loss += float(batch_loss.detach().cpu()) * n
        total_n    += n

        for v in preds.detach().cpu().float().tolist():
            y_pred.append(denormalize_age(v, stats))
        y_true.extend(batch["age"].tolist())
        ids.extend(batch["image_id"])

    return total_loss / max(total_n, 1), y_true, y_pred, ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Ensemble² (12-ch fusion)")
    p.add_argument("--cache-root",           required=True, type=Path)
    p.add_argument("--model-dir",            default="models/fusion/rgb_lab_hsv_ycrcb_seed42", type=Path)
    p.add_argument("--device",               default="auto")
    p.add_argument("--epochs",               default=80,   type=int)
    p.add_argument("--batch-size",           default=32,   type=int)
    p.add_argument("--num-workers",          default=4,    type=int)
    p.add_argument("--lr",                   default=1e-4, type=float)
    p.add_argument("--weight-decay",         default=1e-4, type=float)
    p.add_argument("--grad-clip",            default=1.0,  type=float)
    p.add_argument("--seed",                 default=42,   type=int)
    p.add_argument("--scheduler",            default="reduce_lr_on_plateau",
                   choices=["reduce_lr_on_plateau", "cosine_warmup"],
                   help="'reduce_lr_on_plateau' (default, matches all study models) or "
                        "'cosine_warmup' (linear warmup + cosine annealing, better for 12-ch)")
    p.add_argument("--warmup-epochs",        default=5,    type=int,
                   help="Linear warmup duration (only used with --scheduler cosine_warmup)")
    p.add_argument("--mixed-precision",      default="fp16",
                   choices=["fp16", "bf16", "none"])
    p.add_argument("--allow-weight-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"Device: {device}")

    # Output dir
    model_dir = ROOT / args.model_dir if not args.model_dir.is_absolute() else args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt    = model_dir / "best_checkpoint.pt"
    latest_ckpt  = model_dir / "latest_checkpoint.pt"
    history_path = model_dir / "training_history.csv"

    if best_ckpt.is_file() and history_path.is_file():
        if len(pd.read_csv(history_path)) == args.epochs:
            print("Training already complete, skipping.")
            return

    # Normalization
    with NORM_PATH.open() as fh:
        norm = json.load(fh)
    ch_mean, ch_std = norm["channel_mean"], norm["channel_std"]

    # Manifests & age statistics
    train_df = pd.read_csv(MANIFEST_DIR / "train.csv")
    val_df   = pd.read_csv(MANIFEST_DIR / "validation.csv")
    test_df  = pd.read_csv(MANIFEST_DIR / "test.csv")
    stats    = target_statistics(train_df["age"])   # uses project's utility
    print(f"Age stats — mean: {stats.mean:.4f}, std: {stats.sample_std:.4f}")

    # LDS weights (project utility — identical to all other models)
    ldsw = compute_lds_weights(train_df["age"])

    # Datasets
    train_ds = Fusion12ChDataset(
        train_df, args.cache_root, ch_mean, ch_std,
        stats.mean, stats.sample_std, augment=True, lds_weights=ldsw,
    )
    val_ds = Fusion12ChDataset(
        val_df, args.cache_root, ch_mean, ch_std,
        stats.mean, stats.sample_std, augment=False,
    )
    test_ds = Fusion12ChDataset(
        test_df, args.cache_root, ch_mean, ch_std,
        stats.mean, stats.sample_std, augment=False,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = build_resnet50_12ch_regressor(
        weights="imagenet",
        allow_weight_download=args.allow_weight_download,
        in_channels=12,
    ).to(device)

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    if args.scheduler == "cosine_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-7 / args.lr,  # near-zero start
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup_epochs, 1),
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
        print(f"Scheduler: LinearWarmup({args.warmup_epochs} ep) + CosineAnnealingLR({args.epochs - args.warmup_epochs} ep)")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
        )
        print("Scheduler: ReduceLROnPlateau(factor=0.5, patience=5)")

    # Mixed precision
    use_amp   = args.mixed_precision != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume
    start_epoch  = 1
    best_val_mae = float("inf")
    history_rows: list[dict] = []

    if latest_ckpt.is_file():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_mae = ckpt.get("best_val_mae", float("inf"))
        if history_path.is_file():
            history_rows = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed at epoch {start_epoch - 1}, best_val_mae={best_val_mae:.4f}")

    print(f"Starting training: epochs {start_epoch}→{args.epochs}, device={device}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            args.grad_clip, use_amp, amp_dtype,
        )
        val_loss, vt, vp, _ = evaluate(
            model, val_loader, device, stats, use_amp, amp_dtype,
        )
        val_mae = float(np.mean(np.abs(np.array(vp) - np.array(vt))))

        # Scheduler step — cosine_warmup is epoch-based, reduce_lr_on_plateau needs val_loss
        if args.scheduler == "cosine_warmup":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        row = {
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "val_smooth_l1": round(val_loss, 6),
            "mae":           round(val_mae, 4),
            "lr":            optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_mae={val_mae:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "best_val_mae":       best_val_mae,
                "representation":     "fusion_rgb_lab_hsv_ycrcb",
                "in_channels":        12,
                "seed":               args.seed,
                "norm_path":          str(NORM_PATH),
            }, best_ckpt)
            print(f"  ✓ New best val MAE: {best_val_mae:.4f}")

        torch.save({
            "epoch":                  epoch,
            "model_state_dict":       model.state_dict(),
            "optimizer_state_dict":   optimizer.state_dict(),
            "scheduler_state_dict":   scheduler.state_dict(),
            "scaler_state_dict":      scaler.state_dict(),
            "best_val_mae":           best_val_mae,
        }, latest_ckpt)

    latest_ckpt.unlink(missing_ok=True)
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f}")

    # Inference on val + test with best checkpoint
    print("Running inference on val + test with best checkpoint...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    all_rows: list[dict] = []
    for split, loader, df in [
        ("validation", val_loader, val_df),
        ("test",       test_loader, test_df),
    ]:
        _, y_true, y_pred, image_ids = evaluate(
            model, loader, device, stats, use_amp, amp_dtype,
        )
        mae = float(np.mean(np.abs(np.array(y_pred) - np.array(y_true))))
        print(f"  {split} MAE: {mae:.4f}")
        id_to_row = {str(r["image_id"]): r for _, r in df.iterrows()}
        for iid, age_t, age_p in zip(image_ids, y_true, y_pred):
            r = id_to_row[iid]
            all_rows.append({
                "image_id":       iid,
                "patient_id":     r["patient_id"],
                "image_path":     r["image_path"],
                "age_true":       age_t,
                "age_pred":       age_p,
                "absolute_error": abs(age_p - age_t),
                "split":          split,
            })

    pd.DataFrame(all_rows).to_csv(model_dir / "predictions.csv", index=False)
    print(f"Predictions saved to {model_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
