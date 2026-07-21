from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from retinal_color_transfer.config import RepresentationConfig

REPRESENTATION_IMPLEMENTATION_VERSION = "representation_contract_v3"


@dataclass(frozen=True)
class RepresentationContract:
    name: str
    stored_dtype: str
    stored_range: Any
    channel_order: str
    channel_names: tuple[str, str, str]
    cache_extension: str
    canonical_scaling: str
    canonical_range: tuple[float, float]
    is_binary: bool = False
    repeated_from_single_channel: bool = False


def _single_uint8_channel_contract(
    name: str,
    channel_order: str,
    channel_name: str,
    *,
    stored_range: Any | None = None,
    canonical_range: tuple[float, float] = (0.0, 1.0),
) -> RepresentationContract:
    return RepresentationContract(
        name,
        "uint8",
        [0, 255] if stored_range is None else stored_range,
        channel_order,
        (channel_name, channel_name, channel_name),
        ".png",
        "uint8_divide_by_255",
        canonical_range,
        repeated_from_single_channel=True,
    )


_CONTRACTS: dict[str, RepresentationContract] = {
    "rgb": RepresentationContract(
        "rgb", "uint8", [0, 255], "RGB", ("R", "G", "B"), ".png", "uint8_divide_by_255", (0.0, 1.0)
    ),
    "rgb_r": _single_uint8_channel_contract("rgb_r", "repeated_R", "R"),
    "rgb_g": _single_uint8_channel_contract("rgb_g", "repeated_G", "G"),
    "rgb_b": _single_uint8_channel_contract("rgb_b", "repeated_B", "B"),
    "grayscale": RepresentationContract(
        "grayscale",
        "uint8",
        [0, 255],
        "repeated_luminance",
        ("gray", "gray", "gray"),
        ".png",
        "uint8_divide_by_255",
        (0.0, 1.0),
        repeated_from_single_channel=True,
    ),
    "lab": RepresentationContract(
        "lab",
        "uint8",
        [0, 255],
        "OpenCV_L_a_b_encoded",
        ("L_encoded", "a_encoded", "b_encoded"),
        ".png",
        "uint8_divide_by_255",
        (0.0, 1.0),
    ),
    "lab_l": _single_uint8_channel_contract(
        "lab_l", "repeated_OpenCV_L_encoded", "L_encoded"
    ),
    "lab_a": _single_uint8_channel_contract(
        "lab_a", "repeated_OpenCV_a_encoded", "a_encoded"
    ),
    "lab_b": _single_uint8_channel_contract(
        "lab_b", "repeated_OpenCV_b_encoded", "b_encoded"
    ),
    "hsv": RepresentationContract(
        "hsv",
        "uint8",
        {"h": [0, 179], "s": [0, 255], "v": [0, 255]},
        "OpenCV_HSV",
        ("H", "S", "V"),
        ".png",
        "hsv_uint8_h_over_179_sv_over_255",
        (0.0, 1.0),
    ),
    "hsv_h": _single_uint8_channel_contract(
        "hsv_h",
        "repeated_OpenCV_H",
        "H",
        stored_range=[0, 179],
        canonical_range=(0.0, 179.0 / 255.0),
    ),
    "hsv_s": _single_uint8_channel_contract("hsv_s", "repeated_OpenCV_S", "S"),
    "hsv_v": _single_uint8_channel_contract("hsv_v", "repeated_OpenCV_V", "V"),
    "ycrcb": RepresentationContract(
        "ycrcb",
        "uint8",
        [0, 255],
        "OpenCV_Y_Cr_Cb",
        ("Y", "Cr", "Cb"),
        ".png",
        "uint8_divide_by_255",
        (0.0, 1.0),
    ),
    "ycrcb_y": _single_uint8_channel_contract("ycrcb_y", "repeated_OpenCV_Y", "Y"),
    "ycrcb_cr": _single_uint8_channel_contract("ycrcb_cr", "repeated_OpenCV_Cr", "Cr"),
    "ycrcb_cb": _single_uint8_channel_contract("ycrcb_cb", "repeated_OpenCV_Cb", "Cb"),
}


def representation_contract(name: str) -> RepresentationContract:
    try:
        return _CONTRACTS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown representation contract: {name}") from exc


def validate_config_matches_contract(cfg: RepresentationConfig) -> None:
    contract = representation_contract(cfg.name)
    if cfg.channel_order != contract.channel_order:
        raise ValueError(f"{cfg.name} channel_order must be {contract.channel_order}")
    expected = {
        "stored_dtype": contract.stored_dtype,
        "stored_range": contract.stored_range,
        "canonical_scaling": contract.canonical_scaling,
        "canonical_range": list(contract.canonical_range),
        "channel_names": list(contract.channel_names),
        "cache_extension": contract.cache_extension,
        "is_binary": contract.is_binary,
        "repeated_from_single_channel": contract.repeated_from_single_channel,
    }
    for key, value in expected.items():
        if cfg.tensor_scaling.get(key) != value:
            raise ValueError(f"{cfg.name} tensor_scaling.{key} must be {value!r}")


def cache_path_for(cache_root: str | Path, image_id: str, cfg: RepresentationConfig) -> Path:
    return Path(cache_root) / f"{image_id}{representation_contract(cfg.name).cache_extension}"


def write_representation_array(
    array: np.ndarray,
    path: str | Path,
    cfg: RepresentationConfig,
) -> None:
    contract = representation_contract(cfg.name)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if contract.stored_dtype == "uint8":
        if array.dtype != np.uint8:
            raise ValueError(f"{cfg.name} cache expects uint8 array, got {array.dtype}")
        image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if cfg.name == "rgb" else array
        ok = cv2.imwrite(str(target), image)
        if not ok:
            raise OSError(f"Failed to write cache image: {target}")
        return
    if contract.stored_dtype == "float32":
        if array.dtype != np.float32:
            raise ValueError(f"{cfg.name} cache expects float32 array, got {array.dtype}")
        np.save(target, array, allow_pickle=False)
        return
    raise ValueError(f"Unsupported stored dtype: {contract.stored_dtype}")


def read_representation_array(path: str | Path, cfg: RepresentationConfig) -> np.ndarray:
    contract = representation_contract(cfg.name)
    source = Path(path)
    if contract.stored_dtype == "uint8":
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Cannot read cached representation: {source}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected three-channel cached representation, got {image.shape}")
        if cfg.name == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    if contract.stored_dtype == "float32":
        array = np.load(source, allow_pickle=False)
        if array.dtype != np.float32:
            raise ValueError(f"Expected float32 cached representation, got {array.dtype}")
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected three-channel cached representation, got {array.shape}")
        return array
    raise ValueError(f"Unsupported stored dtype: {contract.stored_dtype}")


def canonical_tensor(array: np.ndarray, cfg: RepresentationConfig) -> torch.Tensor:
    contract = representation_contract(cfg.name)
    if contract.canonical_scaling == "uint8_divide_by_255":
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float() / 255.0
    elif contract.canonical_scaling == "hsv_uint8_h_over_179_sv_over_255":
        data = array.astype(np.float32)
        data[:, :, 0] = data[:, :, 0] / 179.0
        data[:, :, 1:] = data[:, :, 1:] / 255.0
        tensor = torch.from_numpy(data.transpose(2, 0, 1).copy()).float()
    elif contract.canonical_scaling == "float32_identity_0_1":
        if array.dtype != np.float32:
            raise ValueError(f"{cfg.name} canonical float scaling requires float32 input")
        if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"{cfg.name} cache values must be finite within [0, 1]")
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float()
    else:
        raise ValueError(f"Unsupported canonical scaling: {contract.canonical_scaling}")
    return tensor
