from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from retinal_color_transfer.artifacts import read_json, write_json
from retinal_color_transfer.config import RepresentationConfig, stable_fingerprint
from retinal_color_transfer.preprocessing.crop_pad_resize import (
    PREPROCESSING_VERSION,
    PreprocessingError,
    prepare_rgb_from_path,
    write_rgb_png,
)
from retinal_color_transfer.representations.contracts import (
    REPRESENTATION_IMPLEMENTATION_VERSION,
    cache_path_for,
    representation_contract,
    write_representation_array,
)
from retinal_color_transfer.representations.converters import convert_representation
from retinal_color_transfer.reproducibility import runtime_info, select_device


@dataclass(frozen=True)
class CacheResult:
    image_id: str
    status: str
    path: Path
    metadata_path: Path


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": getattr(stat, "st_ino", None),
    }


def metadata_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _valid_existing(metadata_path: Path, expected_fingerprint: str) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("deterministic_fingerprint") == expected_fingerprint
        and metadata.get("status") == "ok"
    )


def build_base_cache_entry(row, *, data_root: Path, cache_root: str | Path) -> CacheResult:
    image_id = str(row.image_id)
    source = (
        Path(row.resolved_path)
        if hasattr(row, "resolved_path")
        else data_root / str(row.image_path)
    )
    cache_root = Path(cache_root)
    out_path = cache_root / f"{image_id}.png"
    meta_path = metadata_path_for(out_path)
    try:
        src_fp = source_fingerprint(source)
        expected = stable_fingerprint(
            {
                "image_id": image_id,
                "source_path": str(source),
                "source_fingerprint": src_fp,
                "preprocessing_version": PREPROCESSING_VERSION,
                "crop_threshold": 25,
                "output_size": 224,
                "interpolation": "cv2.INTER_LANCZOS4",
                "output_shape": [224, 224, 3],
                "contract": "RGB uint8 after non-black retinal-field crop/pad/resize",
            }
        )
        if out_path.is_file() and _valid_existing(meta_path, expected):
            return CacheResult(image_id, "reused", out_path, meta_path)
        prepared = prepare_rgb_from_path(source)
        write_rgb_png(prepared.rgb, out_path)
        metadata = {
            "status": "ok",
            "image_id": image_id,
            "source_path": str(source),
            "source_fingerprint": src_fp,
            "source_shape": list(prepared.source_shape),
            "crop_box_xywh": list(prepared.crop_box_xywh),
            "resized_shape": list(prepared.resized_shape),
            "offset_xy": list(prepared.offset_xy),
            "preprocessing_version": PREPROCESSING_VERSION,
            "crop_threshold": 25,
            "output_size": 224,
            "interpolation": "cv2.INTER_LANCZOS4",
            "output_contract": "RGB",
            "dtype": "uint8",
            "shape": [224, 224, 3],
            "numeric_range": [0, 255],
            "deterministic_fingerprint": expected,
        }
        write_json(metadata, meta_path)
        return CacheResult(image_id, "created", out_path, meta_path)
    except (OSError, PreprocessingError, ValueError) as exc:
        write_json({"status": "error", "image_id": image_id, "error": str(exc)}, meta_path)
        return CacheResult(image_id, "error", out_path, meta_path)


def build_representation_cache_entry(
    row,
    *,
    base_cache_root: str | Path,
    cache_root: str | Path,
    cfg: RepresentationConfig,
) -> CacheResult:
    image_id = str(row.image_id)
    base_path = Path(base_cache_root) / f"{image_id}.png"
    out_path = cache_path_for(cache_root, image_id, cfg)
    meta_path = metadata_path_for(out_path)
    try:
        if cfg.name == "rgb" and Path(cache_root) == Path(base_cache_root):
            if not base_path.is_file():
                raise PreprocessingError(f"Missing base RGB cache image: {base_path}")
            base_meta_path = metadata_path_for(base_path)
            if not base_meta_path.is_file():
                raise PreprocessingError(f"Missing base RGB cache metadata: {base_meta_path}")
            base_meta = read_json(base_meta_path)
            if base_meta.get("status") != "ok" or base_meta.get("output_contract") != "RGB":
                raise PreprocessingError(f"Invalid base RGB cache metadata: {base_meta_path}")
            return CacheResult(image_id, "reused", base_path, base_meta_path)
        base_meta = read_json(metadata_path_for(base_path))
        contract = representation_contract(cfg.name)
        library_versions = runtime_info(select_device("cpu"), seed=0)["libraries"]
        expected = stable_fingerprint(
            {
                "image_id": image_id,
                "base_fingerprint": base_meta["deterministic_fingerprint"],
                "representation": cfg.name,
                "representation_implementation_version": REPRESENTATION_IMPLEMENTATION_VERSION,
                "representation_params": cfg.params,
                "representation_fingerprint": cfg.fingerprint,
                "stored_dtype": contract.stored_dtype,
                "stored_range": contract.stored_range,
                "channel_order": contract.channel_order,
                "channel_names": contract.channel_names,
                "canonical_scaling": contract.canonical_scaling,
                "canonical_range": contract.canonical_range,
                "library_versions": library_versions,
            }
        )
        if out_path.is_file() and _valid_existing(meta_path, expected):
            return CacheResult(image_id, "reused", out_path, meta_path)
        base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        if base is None:
            raise PreprocessingError(f"Cannot read base cache image: {base_path}")
        rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        rep = convert_representation(rgb, cfg)
        write_representation_array(rep, out_path, cfg)
        metadata = {
            "status": "ok",
            "image_id": image_id,
            "base_cache_path": str(base_path),
            "base_fingerprint": base_meta["deterministic_fingerprint"],
            "representation_name": cfg.name,
            "representation_implementation_version": REPRESENTATION_IMPLEMENTATION_VERSION,
            "representation_params": cfg.params,
            "representation_fingerprint": cfg.fingerprint,
            "channel_order": contract.channel_order,
            "channel_names": list(contract.channel_names),
            "dtype": contract.stored_dtype,
            "shape": list(rep.shape),
            "numeric_range": contract.stored_range,
            "tensor_scaling": cfg.tensor_scaling,
            "library_versions": library_versions,
            "is_binary": contract.is_binary,
            "repeated_from_single_channel": contract.repeated_from_single_channel,
            "deterministic_fingerprint": expected,
        }
        write_json(metadata, meta_path)
        return CacheResult(image_id, "created", out_path, meta_path)
    except (OSError, KeyError, PreprocessingError, ValueError) as exc:
        write_json({"status": "error", "image_id": image_id, "error": str(exc)}, meta_path)
        return CacheResult(image_id, "error", out_path, meta_path)


def cache_status(cache_root: str | Path) -> dict[str, int]:
    counts = {"ok": 0, "error": 0, "missing_metadata": 0, "corrupt_metadata": 0}
    for image_path in [*Path(cache_root).glob("*.png"), *Path(cache_root).glob("*.npy")]:
        meta_path = metadata_path_for(image_path)
        if not meta_path.is_file():
            counts["missing_metadata"] += 1
            continue
        try:
            status = read_json(meta_path).get("status", "error")
        except (OSError, json.JSONDecodeError):
            counts["corrupt_metadata"] += 1
            continue
        counts["ok" if status == "ok" else "error"] += 1
    return counts


def validate_cache_entry(
    cache_root: str | Path,
    image_id: str,
    cfg: RepresentationConfig,
) -> dict[str, Any]:
    path = cache_path_for(cache_root, image_id, cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache artifact: {path}")
    meta_path = metadata_path_for(path)
    metadata = read_json(meta_path)
    if metadata.get("status") != "ok":
        raise ValueError(f"Cache entry is not ok for {image_id}: {metadata.get('status')}")
    if cfg.name == "rgb" and metadata.get("output_contract") == "RGB":
        if metadata.get("dtype") != "uint8":
            raise ValueError(f"Base RGB cache dtype mismatch for {image_id}")
        if metadata.get("shape") != [224, 224, 3]:
            raise ValueError(f"Base RGB cache shape mismatch for {image_id}")
        if not metadata.get("deterministic_fingerprint"):
            raise ValueError(
                f"Base RGB cache metadata missing deterministic fingerprint for {image_id}"
            )
        return metadata
    if metadata.get("representation_name") != cfg.name:
        raise ValueError(f"Cache representation name mismatch for {image_id}")
    if metadata.get("representation_fingerprint") != cfg.fingerprint:
        raise ValueError(f"Cache representation fingerprint mismatch for {image_id}")
    if metadata.get("tensor_scaling") != cfg.tensor_scaling:
        raise ValueError(f"Cache tensor scaling mismatch for {image_id}")
    if not metadata.get("deterministic_fingerprint"):
        raise ValueError(f"Cache metadata missing deterministic fingerprint for {image_id}")
    return metadata


def validate_cache_for_manifest(
    frame,
    *,
    cache_root: str | Path,
    cfg: RepresentationConfig,
) -> None:
    for row in frame.itertuples(index=False):
        validate_cache_entry(cache_root, str(row.image_id), cfg)


def resolve_representation_cache_root(
    cache_root: str | Path,
    cfg: RepresentationConfig,
) -> Path:
    """Resolve the renamed RGB cache while preserving legacy checkpoint configs."""
    configured = Path(cache_root)
    if configured.exists():
        return configured
    if cfg.name == "rgb" and configured.name == "base_rgb":
        renamed = configured.with_name("rgb")
        if renamed.exists():
            return renamed
    return configured
