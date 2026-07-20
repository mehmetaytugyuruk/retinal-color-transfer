from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from retinal_color_transfer.artifacts import (
    representation_family,
    resolve_cache_dir,
    resolve_model_dir,
    resolve_normalization_path,
    resolve_representation_config_path,
)
from retinal_color_transfer.config import (
    RepresentationConfig,
    dump_yaml,
    load_yaml,
    validate_experiment_config,
)
from retinal_color_transfer.data import load_manifest
from retinal_color_transfer.preprocessing.cache import (
    build_base_cache_entry,
    build_representation_cache_entry,
    cache_status,
    validate_cache_for_manifest,
)
from retinal_color_transfer.preprocessing.normalization import (
    compute_channel_stats,
    save_channel_stats,
    validate_normalization_compatibility,
)
from retinal_color_transfer.study import CANONICAL_MODELS, REPRESENTATIONS

SPLITS = ("train", "validation", "test")
TEMPLATE_CONFIG = Path("configs/resnet50_imagenet_local.yaml")


def validate_prepared_representation(
    rep: str,
    *,
    data_root: Path,
    cache_root: Path,
    manifest_dir: Path,
    normalization_dir: Path,
) -> None:
    representation_config = resolve_representation_config_path(
        Path("configs"),
        rep,
    )
    rep_cfg = RepresentationConfig.from_file(representation_config, require_approved=True)
    rep_cache_root = resolve_cache_dir(cache_root, rep)
    norm_output = resolve_normalization_path(normalization_dir, rep)
    expected_rows = 0
    for split in SPLITS:
        manifest = load_manifest(
            manifest_dir / f"{split}.csv",
            role=split,
            data_root=data_root,
        )
        expected_rows += len(manifest.frame)
        validate_cache_for_manifest(
            manifest.frame,
            cache_root=rep_cache_root,
            cfg=rep_cfg,
        )
    status = cache_status(rep_cache_root)
    expected_status = {
        "ok": expected_rows,
        "error": 0,
        "missing_metadata": 0,
        "corrupt_metadata": 0,
    }
    if status != expected_status:
        raise RuntimeError(
            f"{rep} cache status mismatch: expected {expected_status}, got {status}"
        )
    with norm_output.open("r", encoding="utf-8") as handle:
        normalization = json.load(handle)
    validate_normalization_compatibility(normalization, rep_cfg)
    print(
        f"Validated {rep}: {expected_rows} cache entries and compatible normalization",
        flush=True,
    )


def prepare_representation(
    rep: str,
    *,
    data_root: Path,
    cache_root: Path,
    manifest_dir: Path,
    normalization_dir: Path,
) -> None:
    print(f"=== Preparing representation artifacts: {rep} ===", flush=True)
    representation_config = resolve_representation_config_path(
        Path("configs"),
        rep,
    )
    if not representation_config.is_file():
        raise FileNotFoundError(f"Missing representation config: {representation_config}")

    rep_cache_root = resolve_cache_dir(cache_root, rep)
    base_rgb_root = resolve_cache_dir(cache_root, "rgb")
    rep_cfg = RepresentationConfig.from_file(
        representation_config,
        require_approved=True,
    )
    if rep == "rgb":
        for split in SPLITS:
            manifest = load_manifest(
                manifest_dir / f"{split}.csv",
                role=split,
                data_root=data_root,
            )
            counts = {"created": 0, "reused": 0, "error": 0}
            for row in manifest.frame.itertuples(index=False):
                result = build_base_cache_entry(
                    row,
                    data_root=manifest.data_root,
                    cache_root=base_rgb_root,
                )
                counts[result.status] = counts.get(result.status, 0) + 1
            print(f"{rep} {split}: {json.dumps(counts, sort_keys=True)}", flush=True)
    else:
        for split in SPLITS:
            manifest = load_manifest(
                manifest_dir / f"{split}.csv",
                role=split,
                data_root=data_root,
            )
            counts = {"created": 0, "reused": 0, "error": 0}
            for row in manifest.frame.itertuples(index=False):
                result = build_representation_cache_entry(
                    row,
                    base_cache_root=base_rgb_root,
                    cache_root=rep_cache_root,
                    cfg=rep_cfg,
                )
                counts[result.status] = counts.get(result.status, 0) + 1
            print(f"{rep} {split}: {json.dumps(counts, sort_keys=True)}", flush=True)

    norm_output = resolve_normalization_path(normalization_dir, rep)
    norm_output.parent.mkdir(parents=True, exist_ok=True)
    if norm_output.is_file():
        print(f"Reusing train-only normalization statistics at {norm_output}", flush=True)
    else:
        train_manifest = load_manifest(
            manifest_dir / "train.csv",
            role="train",
            data_root=data_root,
        )
        statistics = compute_channel_stats(
            train_manifest.frame,
            cache_root=rep_cache_root,
            cfg=rep_cfg,
        )
        save_channel_stats(statistics, norm_output)
        print(f"Wrote train-only normalization statistics to {norm_output}", flush=True)

    validate_prepared_representation(
        rep,
        data_root=data_root,
        cache_root=cache_root,
        manifest_dir=manifest_dir,
        normalization_dir=normalization_dir,
    )


def build_experiment_config(
    *,
    representation: str,
    seed: int,
    data_root: Path,
    cache_root: Path,
    normalization_dir: Path,
    model_root: Path,
    batch_size: int,
    epochs: int,
    num_workers: int,
    prefetch_factor: int,
    mixed_precision: str,
    memory_format: str,
    device: str,
    allow_weight_download: bool,
) -> dict:
    config = deepcopy(load_yaml(TEMPLATE_CONFIG))
    model_name = f"{representation}_seed{seed}"
    family = representation_family(representation)
    config["experiment"]["name"] = model_name
    config["experiment"]["output_dir"] = str(resolve_model_dir(model_root, model_name))
    config["data"]["data_root"] = str(data_root)
    config["cache"]["base_rgb_root"] = str(resolve_cache_dir(cache_root, "rgb"))
    config["cache"]["representation_root"] = str(
        resolve_cache_dir(cache_root, representation)
    )
    config["representation_config"] = str(
        resolve_representation_config_path("configs", representation)
    )
    config["normalization"]["statistics_path"] = str(
        resolve_normalization_path(normalization_dir, representation)
    )
    config["model"]["allow_weight_download"] = bool(allow_weight_download)
    config["optimization"]["batch_size"] = int(batch_size)
    config["optimization"]["epochs"] = int(epochs)
    uses_cuda = device == "cuda"
    config["runtime"].update(
        {
            "device": device,
            "num_workers": int(num_workers),
            "pin_memory": uses_cuda,
            "prefetch_factor": int(prefetch_factor),
            "mixed_precision": mixed_precision,
            "memory_format": memory_format,
            "non_blocking_transfer": uses_cuda,
            "cudnn_benchmark": uses_cuda,
            "allow_tf32": uses_cuda,
            "seed": int(seed),
        }
    )
    validate_experiment_config(config)
    RepresentationConfig.from_file(
        config["representation_config"],
        require_approved=True,
    )
    return config


def write_experiment_configs(
    *,
    output_dir: Path,
    data_root: Path,
    cache_root: Path,
    normalization_dir: Path,
    model_root: Path,
    batch_size: int,
    epochs: int,
    num_workers: int,
    prefetch_factor: int,
    mixed_precision: str,
    memory_format: str,
    device: str,
    allow_weight_download: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for representation, seed in CANONICAL_MODELS:
        config = build_experiment_config(
            representation=representation,
            seed=seed,
            data_root=data_root,
            cache_root=cache_root,
            normalization_dir=normalization_dir,
            model_root=model_root,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            mixed_precision=mixed_precision,
            memory_format=memory_format,
            device=device,
            allow_weight_download=allow_weight_download,
        )
        output = output_dir / f"{config['experiment']['name']}.yaml"
        dump_yaml(config, output)
        paths.append(output)
        print(f"Wrote {output}")
    if len(paths) != 24:
        raise RuntimeError(f"Expected 24 experiment configs, wrote {len(paths)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all caches, train-only normalization statistics, and "
            "canonical experiment configs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--cache-root", type=Path, default=Path("caches"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--normalization-dir",
        type=Path,
        default=Path("data/normalization"),
    )
    parser.add_argument(
        "--config-output-dir",
        type=Path,
        default=Path("configs/generated"),
    )
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument(
        "--memory-format",
        choices=["contiguous", "channels_last"],
        default="channels_last",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--allow-weight-download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    for rep in REPRESENTATIONS:
        prepare_representation(
            rep,
            data_root=args.data_root,
            cache_root=args.cache_root,
            manifest_dir=args.manifest_dir,
            normalization_dir=args.normalization_dir,
        )
    paths = write_experiment_configs(
        output_dir=args.config_output_dir,
        data_root=args.data_root,
        cache_root=args.cache_root,
        normalization_dir=args.normalization_dir,
        model_root=args.model_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        mixed_precision=args.mixed_precision,
        memory_format=args.memory_format,
        device=args.device,
        allow_weight_download=args.allow_weight_download,
    )
    print(
        f"Preparation complete: {len(REPRESENTATIONS)} representations and "
        f"{len(paths)} experiment configs."
    )


if __name__ == "__main__":
    main()
