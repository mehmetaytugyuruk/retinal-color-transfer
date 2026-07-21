from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

MODEL_GROUPS = ("grayscale", "rgb", "lab", "hsv", "ycrcb")


def representation_family(name: str) -> str:
    for group in MODEL_GROUPS:
        if name == group or name.startswith(f"{group}_"):
            return group
    raise ValueError(f"Unknown representation family for '{name}'")

def model_group(model_id: str) -> str:
    return representation_family(model_id)


def resolve_model_dir(model_root: str | Path, model_id: str) -> Path:
    root = Path(model_root)
    return root / model_group(model_id) / model_id


def resolve_cache_dir(cache_root: str | Path, representation: str) -> Path:
    root = Path(cache_root)
    return root / representation_family(representation) / representation


def resolve_normalization_path(
    normalization_root: str | Path,
    representation: str,
) -> Path:
    root = Path(normalization_root)
    return (
        root
        / representation_family(representation)
        / f"{representation}_train_stats.json"
    )


def resolve_representation_config_path(
    config_root: str | Path,
    representation: str,
) -> Path:
    root = Path(config_root)
    return root / representation_family(representation) / f"{representation}.yaml"


def iter_model_dirs(model_root: str | Path) -> list[Path]:
    root = Path(model_root)
    return sorted(
        model_dir
        for group in MODEL_GROUPS
        for model_dir in (root / group).iterdir()
        if model_dir.is_dir()
    )


def ensure_new_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(data: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: Iterable[dict], path: str | Path, fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
