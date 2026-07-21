from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping at top level in {path}")
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def stable_fingerprint(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def reject_unknown_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {', '.join(unknown)}")


@dataclass(frozen=True)
class RepresentationConfig:
    name: str
    approved_for_cache_build: bool
    params: dict[str, Any]
    tensor_scaling: dict[str, Any]
    channel_order: str
    numeric_range: Any
    notes: str | None = None
    fingerprint: str = ""

    @classmethod
    def from_file(cls, path: str | Path, *, require_approved: bool = False) -> RepresentationConfig:
        path = Path(path)
        raw = load_yaml(path)
        reject_unknown_keys(raw, {"representation"}, str(path))
        rep = raw.get("representation")
        if not isinstance(rep, dict):
            raise ConfigError(f"{path} must contain a representation mapping")
        reject_unknown_keys(
            rep,
            {
                "name",
                "approved_for_cache_build",
                "params",
                "tensor_scaling",
                "channel_order",
                "numeric_range",
                "notes",
            },
            f"{path}:representation",
        )
        required = {
            "name",
            "approved_for_cache_build",
            "params",
            "tensor_scaling",
            "channel_order",
            "numeric_range",
        }
        missing = sorted(required - set(rep))
        if missing:
            raise ConfigError(f"Missing representation key(s) in {path}: {', '.join(missing)}")
        if require_approved and not rep["approved_for_cache_build"]:
            raise ConfigError(
                f"{path} is a template or unapproved configuration; real cache builds require "
                "explicit approved parameters."
            )
        cfg = cls(
            name=str(rep["name"]).lower(),
            approved_for_cache_build=bool(rep["approved_for_cache_build"]),
            params=rep["params"] or {},
            tensor_scaling=rep["tensor_scaling"] or {},
            channel_order=str(rep["channel_order"]),
            numeric_range=rep["numeric_range"],
            notes=rep.get("notes"),
            fingerprint=stable_fingerprint(rep),
        )
        validate_representation_config(cfg)
        from retinal_color_transfer.representations.contracts import (
            validate_config_matches_contract,
        )

        validate_config_matches_contract(cfg)
        return cfg


def validate_representation_config(cfg: RepresentationConfig) -> None:
    channel_ablation = {
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "lab_l",
        "lab_a",
        "lab_b",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "ycrcb_y",
        "ycrcb_cr",
        "ycrcb_cb",
    }
    known = {
        "rgb",
        "grayscale",
        "lab",
        "hsv",
        "ycrcb",
        *channel_ablation,
    }
    if cfg.name not in known:
        raise ConfigError(f"Unknown representation '{cfg.name}'")
    if cfg.name in {
        "rgb",
        "grayscale",
        "lab",
        "hsv",
        "ycrcb",
        *channel_ablation,
    } and cfg.params:
        raise ConfigError(f"{cfg.name} does not accept representation parameters")


def validate_experiment_config(data: dict[str, Any]) -> None:
    allowed_top = {
        "experiment",
        "data",
        "cache",
        "representation_config",
        "normalization",
        "target_scaling",
        "lds",
        "model",
        "optimization",
        "scheduler",
        "checkpoint_selection",
        "early_stopping",
        "augmentation",
        "runtime",
    }
    reject_unknown_keys(data, allowed_top, "experiment config")
    model = data.get("model", {})
    reject_unknown_keys(
        model,
        {"architecture", "weights", "fully_finetune", "allow_weight_download"},
        "model",
    )
    if model.get("architecture") != "resnet50":
        raise ConfigError("Experiment requires model.architecture=resnet50")
    if model.get("weights") != "imagenet":
        raise ConfigError("Experiment requires model.weights=imagenet")
    if not isinstance(model.get("allow_weight_download", False), bool):
        raise ConfigError("model.allow_weight_download must be a boolean")
    if data.get("checkpoint_selection", {}).get("metric") != "validation_mae":
        raise ConfigError("Checkpoint selection must use validation_mae")
    if data.get("scheduler", {}).get("monitor") != "validation_smooth_l1_loss":
        raise ConfigError("Scheduler must monitor validation_smooth_l1_loss")
