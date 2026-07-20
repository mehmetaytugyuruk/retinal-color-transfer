from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from retinal_color_transfer.config import RepresentationConfig
from retinal_color_transfer.preprocessing.cache import validate_cache_entry
from retinal_color_transfer.representations.contracts import (
    cache_path_for,
    canonical_tensor,
    read_representation_array,
)

REQUIRED_COLUMNS = ["image_id", "patient_id", "image_path", "age", "split"]
VALID_SPLITS = {"train", "validation", "test"}


class ManifestError(ValueError):
    """Raised when a manifest violates the project data contract."""


@dataclass(frozen=True)
class Manifest:
    frame: pd.DataFrame
    path: Path
    role: str
    data_root: Path


def resolve_image_path(image_path: str, *, data_root: Path | None, manifest_path: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    base = data_root if data_root is not None else manifest_path.parent
    return (base / path).resolve()


def load_manifest(
    path: str | Path,
    *,
    role: str,
    data_root: str | Path | None = None,
    strict_paths: bool = False,
    require_integer_age: bool = True,
) -> Manifest:
    manifest_path = Path(path)
    if role not in VALID_SPLITS:
        raise ManifestError(f"role must be one of {sorted(VALID_SPLITS)}")
    frame = pd.read_csv(manifest_path)
    validate_manifest_frame(
        frame,
        role=role,
        strict_paths=strict_paths,
        data_root=Path(data_root).resolve() if data_root is not None else None,
        manifest_path=manifest_path,
        require_integer_age=require_integer_age,
    )
    frame = frame.copy()
    root = Path(data_root).resolve() if data_root is not None else manifest_path.parent.resolve()
    frame["resolved_path"] = [
        str(resolve_image_path(path_value, data_root=root, manifest_path=manifest_path))
        for path_value in frame["image_path"]
    ]
    return Manifest(frame=frame, path=manifest_path, role=role, data_root=root)


def validate_manifest_frame(
    frame: pd.DataFrame,
    *,
    role: str,
    strict_paths: bool = False,
    data_root: Path | None = None,
    manifest_path: Path | None = None,
    require_integer_age: bool = True,
    min_age: float = 0.0,
    max_age: float = 120.0,
) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ManifestError(f"Manifest missing required column(s): {', '.join(missing)}")
    if frame.duplicated().any():
        raise ManifestError("Manifest contains duplicate rows")
    for col in ("image_id", "patient_id", "image_path"):
        if frame[col].isna().any() or (frame[col].astype(str).str.strip().str.len() == 0).any():
            raise ManifestError(f"Manifest column '{col}' must be present for every row")
    for col in ("image_id", "image_path"):
        duplicated = frame[col].duplicated(keep=False)
        if duplicated.any():
            values = sorted(frame.loc[duplicated, col].astype(str).unique())
            raise ManifestError(f"Manifest contains duplicate {col}: {values[:5]}")
    split_values = set(frame["split"].astype(str))
    if split_values != {role}:
        raise ManifestError(
            f"Manifest split labels {sorted(split_values)} do not match role '{role}'"
        )
    ages = pd.to_numeric(frame["age"], errors="coerce")
    if ages.isna().any() or not np.isfinite(ages.to_numpy(dtype=float)).all():
        raise ManifestError("Manifest contains non-finite or non-numeric ages")
    if ((ages < min_age) | (ages > max_age)).any():
        raise ManifestError(
            f"Manifest ages must be within the plausible range [{min_age}, {max_age}]"
        )
    if require_integer_age and not (ages == ages.astype(int)).all():
        raise ManifestError(
            "Approved LDS protocol requires integer-valued ages; fractional ages cannot be "
            "silently truncated."
        )
    if strict_paths:
        if manifest_path is None:
            manifest_path = Path(".")
        for value in frame["image_path"]:
            resolved = resolve_image_path(
                str(value),
                data_root=data_root,
                manifest_path=manifest_path,
            )
            if not resolved.is_file():
                raise ManifestError(f"Image path is not readable: {resolved}")


def validate_split_collection(manifests: list[Manifest]) -> None:
    by_role = {manifest.role: manifest.frame for manifest in manifests}
    missing_roles = sorted(VALID_SPLITS - set(by_role))
    if missing_roles:
        raise ManifestError(f"Missing manifest role(s): {', '.join(missing_roles)}")
    for field in ("patient_id", "image_id", "resolved_path"):
        seen: dict[str, set[str]] = {}
        for role, frame in by_role.items():
            for value in frame[field].astype(str):
                seen.setdefault(value, set()).add(role)
        overlaps = {value: roles for value, roles in seen.items() if len(roles) > 1}
        if overlaps:
            example = next(iter(overlaps))
            raise ManifestError(
                f"{field} overlaps across splits, for example {example}: "
                f"{sorted(overlaps[example])}"
            )


class CachedRegressionDataset(Dataset):
    def __init__(
        self,
        frame,
        *,
        cache_root: str | Path,
        representation_config: RepresentationConfig,
        age_mean: float,
        age_std: float,
        channel_mean: list[float],
        channel_std: list[float],
        transform: Callable | None = None,
        sample_weights: dict[int, float] | None = None,
        normalization_fingerprint: str | None = None,
        validate_each_item: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.representation_config = representation_config
        self.age_mean = float(age_mean)
        self.age_std = float(age_std)
        self.channel_mean = torch.tensor(channel_mean, dtype=torch.float32).view(3, 1, 1)
        self.channel_std = torch.tensor(channel_std, dtype=torch.float32).view(3, 1, 1)
        self.transform = transform
        self.sample_weights = sample_weights or {}
        self.normalization_fingerprint = normalization_fingerprint
        self.validate_each_item = validate_each_item

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = cache_path_for(
            self.cache_root,
            str(row.image_id),
            self.representation_config,
        )
        if self.validate_each_item:
            validate_cache_entry(
                self.cache_root,
                str(row.image_id),
                self.representation_config,
            )
        image = read_representation_array(image_path, self.representation_config)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = canonical_tensor(image, self.representation_config)
        image = (image - self.channel_mean) / self.channel_std
        age = float(row.age)
        norm_age = (age - self.age_mean) / self.age_std
        weight = float(self.sample_weights.get(int(age), 1.0))
        return {
            "image": image,
            "target": torch.tensor(norm_age, dtype=torch.float32),
            "age": torch.tensor(age, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "image_id": row.image_id,
            "patient_id": row.patient_id,
            "image_path": row.image_path,
        }
