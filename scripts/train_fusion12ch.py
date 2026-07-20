"""
Train the 12-channel early-fusion model (Ensemble²).

Input: RGB (3ch) + Lab (3ch) + HSV (3ch) + YCrCb (3ch) concatenated → 12ch tensor.
Model: ResNet-50 with conv1 adapted from 3→12 channels (pretrained weights repeated
       and scaled by 1/4 to preserve expected activation magnitude).

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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Minimal inline imports — avoids touching the locked 3-ch contract system
# ---------------------------------------------------------------------------
import cv2
import pandas as pd
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
NORM_PATH = ROOT / "data" / "normalization" / "fusion" / "rgb_lab_hsv_ycrcb_train_stats.json"

# Channel order: [RGB(3), Lab(3), HSV(3), YCrCb(3)]
REPRESENTATIONS = ["rgb", "lab", "hsv", "ycrcb"]
CACHE_SUBDIRS = {
    "rgb":   "rgb/rgb",
    "lab":   "lab/lab",
    "hsv":   "hsv/hsv",
    "ycrcb": "ycrcb/ycrcb",
}


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    """Load PNG cache → HWC float32 [0,1], RGB channel order."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_lab(path: Path) -> np.ndarray:
    """Load Lab PNG cache → HWC float32 [0,1] (OpenCV uint8 encoding)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


def load_hsv(path: Path) -> np.ndarray:
    """Load HSV PNG cache → HWC float32 with H/179, S/255, V/255 scaling."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    out = img.astype(np.float32)
    out[:, :, 0] /= 179.0
    out[:, :, 1] /= 255.0
    out[:, :, 2] /= 255.0
    return out


def load_ycrcb(path: Path) -> np.ndarray:
    """Load YCrCb PNG cache → HWC float32 [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


LOADERS = {
    "rgb":   load_rgb,
    "lab":   load_lab,
    "hsv":   load_hsv,
    "ycrcb": load_ycrcb,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Fusion12ChDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_root: Path,
        channel_mean: list[float],
        channel_std: list[float],
        age_mean: float,
        age_std: float,
        augment: bool = False,
    ) -> None:
        self.frame = manifest.reset_index(drop=True)
        self.cache_root = cache_root
        self.mean = torch.tensor(channel_mean, dtype=torch.float32).view(12, 1, 1)
        self.std  = torch.tensor(channel_std,  dtype=torch.float32).view(12, 1, 1)
        self.age_mean = age_mean
        self.age_std  = age_std
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        image_id = str(row["image_id"])

        channels = []
        for rep in REPRESENTATIONS:
            subdir = CACHE_SUBDIRS[rep]
            path = self.cache_root / subdir / f"{image_id}.png"
            arr = LOADERS[rep](path)          # HWC float32
            channels.append(arr)

        # HWC → CHW, concatenate along channel dim
        tensor = torch.from_numpy(
            np.concatenate([c.transpose(2, 0, 1) for c in channels], axis=0)
        )  # shape [12, H, W]

        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])  # horizontal flip
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                tensor = _rotate_tensor(tensor, angle)

        tensor = (tensor - self.mean) / self.std

        age = float(row["age"])
        return {
            "image":  tensor,
            "target": torch.tensor((age - self.age_mean) / self.age_std, dtype=torch.float32),
            "age":    torch.tensor(age, dtype=torch.float32),
            "image_id": image_id,
        }


def _rotate_tensor(tensor: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a CHW tensor by angle_deg degrees (bilinear, fill=0)."""
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    theta = torch.stack([
        torch.stack([cos_a, -sin_a, torch.zeros(1).squeeze()]),
        torch.stack([sin_a,  cos_a, torch.zeros(1).squeeze()]),
    ]).unsqueeze(0)
    grid = F.affine_grid(theta, tensor.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(
        tensor.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(allow_weight_download: bool = False) -> nn.Module:
    """ResNet-50 with conv1 adapted to 12 input channels."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from retinal_color_transfer.model import build_resnet50_12ch_regressor
    return build_resnet50_12ch_regressor(
        weights="imagenet",
        allow_weight_download=allow_weight_download,
        in_channels=12,
    )


# ---------------------------------------------------------------------------
# LDS (label-distribution smoothing) sample weights
# ---------------------------------------------------------------------------

def compute_lds_weights(ages: pd.Series, sigma: float = 2.0) -> dict[int, float]:
    from scipy.ndimage import gaussian_filter1d
    min_age, max_age = int(ages.min()), int(ages.max())
    bins = np.zeros(max_age - min_age + 1)
    for a in ages:
        bins[int(a) - min_age] += 1
    smoothed = gaussian_filter1d(bins, sigma=sigma)
    smoothed = np.maximum(smoothed, 1e-5)
    weights = {}
    for i, age in enumerate(range(min_age, max_age + 1)):
        weights[age] = float(1.0 / smoothed[i])
    return weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, grad_clip,
                    autocast_enabled, autocast_dtype):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        weights = batch.get("weight")

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_enabled):
            preds = model(images).squeeze(1)
            loss  = F.smooth_l1_loss(preds, targets, reduction="none")
            if weights is not None:
                w = weights.to(device, non_blocking=True)
                w = w / w.sum() * len(w)
                loss = (loss * w).mean()
            else:
                loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.numel()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(model, loader, device, age_mean, age_std,
             autocast_enabled=False, autocast_dtype=None):
    model.eval()
    all_preds, all_true, all_ids = [], [], []
    total_loss    = 0.0
    total_examples = 0
    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type,
                            dtype=autocast_dtype, enabled=autocast_enabled):
            preds = model(images).squeeze(1)
            loss  = F.smooth_l1_loss(preds, targets, reduction="mean")

        batch_size = targets.numel()
        total_loss    += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

        pred_ages = preds.detach().cpu().float() * age_std + age_mean
        true_ages = batch["age"]
        all_preds.extend(pred_ages.tolist())
        all_true.extend(true_ages.tolist())
        all_ids.extend(batch["image_id"])

    mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_true))))
    return {
        "mae": mae,
        "smooth_l1_loss": total_loss / max(total_examples, 1),
        "predictions": list(zip(all_ids, all_true, all_preds)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train 12-channel fusion model (Ensemble²)")
    p.add_argument("--cache-root",  required=True, type=Path)
    p.add_argument("--model-dir",   default="models/fusion/rgb_lab_hsv_ycrcb_seed42", type=Path)
    p.add_argument("--device",      default="auto")
    p.add_argument("--epochs",      default=80,  type=int)
    p.add_argument("--batch-size",  default=32,  type=int)
    p.add_argument("--num-workers", default=4,   type=int)
    p.add_argument("--lr",          default=1e-4, type=float)
    p.add_argument("--weight-decay",default=1e-4, type=float)
    p.add_argument("--grad-clip",   default=1.0,  type=float)
    p.add_argument("--seed",        default=42,  type=int)
    p.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "none"])
    p.add_argument("--allow-weight-download", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Model output dir
    model_dir = ROOT / args.model_dir if not args.model_dir.is_absolute() else args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Resume logic
    best_ckpt  = model_dir / "best_checkpoint.pt"
    latest_ckpt = model_dir / "latest_checkpoint.pt"
    history_path = model_dir / "training_history.csv"

    if best_ckpt.is_file() and history_path.is_file():
        history = pd.read_csv(history_path)
        if len(history) == args.epochs:
            print("Training already complete, skipping.")
            return

    # Normalization stats
    with NORM_PATH.open() as f:
        norm = json.load(f)
    ch_mean = norm["channel_mean"]
    ch_std  = norm["channel_std"]

    # Manifests
    train_df = pd.read_csv(MANIFEST_DIR / "train.csv")
    val_df   = pd.read_csv(MANIFEST_DIR / "validation.csv")
    test_df  = pd.read_csv(MANIFEST_DIR / "test.csv")

    # Age statistics from train split (consistent with other models)
    age_mean = float(train_df["age"].mean())
    age_std  = float(train_df["age"].std())
    print(f"Age stats — mean: {age_mean:.4f}, std: {age_std:.4f}")

    # LDS weights
    lds_weights = compute_lds_weights(train_df["age"])

    # Datasets & loaders
    train_ds = Fusion12ChDataset(train_df, args.cache_root, ch_mean, ch_std,
                                  age_mean, age_std, augment=True)
    val_ds   = Fusion12ChDataset(val_df,   args.cache_root, ch_mean, ch_std,
                                  age_mean, age_std, augment=False)
    test_ds  = Fusion12ChDataset(test_df,  args.cache_root, ch_mean, ch_std,
                                  age_mean, age_std, augment=False)

    # Add LDS weights to train dataset items
    train_ds.lds_weights = lds_weights

    # Monkey-patch __getitem__ to include sample weight
    _orig_getitem = train_ds.__getitem__.__func__

    def _weighted_getitem(self, idx):
        item = _orig_getitem(self, idx)
        age = int(self.frame.iloc[idx]["age"])
        item["weight"] = torch.tensor(self.lds_weights.get(age, 1.0), dtype=torch.float32)
        return item

    import types
    train_ds.__getitem__ = types.MethodType(_weighted_getitem, train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(allow_weight_download=args.allow_weight_download)
    model = model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
    )

    # Mixed precision
    use_amp   = args.mixed_precision != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume from latest checkpoint if available
    start_epoch = 1
    best_val_mae = float("inf")
    history_rows = []

    if latest_ckpt.is_file():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
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

        val_metrics = evaluate(model, val_loader, device, age_mean, age_std,
                               autocast_enabled=use_amp, autocast_dtype=amp_dtype)
        val_mae  = val_metrics["mae"]
        val_loss = val_metrics["smooth_l1_loss"]

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

        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_mae": best_val_mae,
                "representation": "fusion_rgb_lab_hsv_ycrcb",
                "in_channels": 12,
                "seed": args.seed,
                "norm_path": str(NORM_PATH),
            }, best_ckpt)
            print(f"  ✓ New best val MAE: {best_val_mae:.4f}")

        # Save latest checkpoint (for resumption)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else {},
            "best_val_mae": best_val_mae,
        }, latest_ckpt)

    # Remove transient latest checkpoint after successful completion
    if latest_ckpt.is_file():
        latest_ckpt.unlink()
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f}")

    # Inference with best checkpoint
    print("Running inference on val + test with best checkpoint...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    all_preds = []
    for split, loader, df in [
        ("validation", val_loader, val_df),
        ("test",       test_loader, test_df),
    ]:
        metrics = evaluate(model, loader, device, age_mean, age_std)
        print(f"  {split} MAE: {metrics['mae']:.4f}")
        for image_id, age_true, age_pred in metrics["predictions"]:
            row_df = df[df["image_id"] == image_id].iloc[0]
            all_preds.append({
                "image_id":       image_id,
                "patient_id":     row_df["patient_id"],
                "image_path":     row_df["image_path"],
                "age_true":       age_true,
                "age_pred":       age_pred,
                "absolute_error": abs(age_pred - age_true),
                "split":          split,
            })

    pd.DataFrame(all_preds).to_csv(model_dir / "predictions.csv", index=False)
    print(f"Predictions saved to {model_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
