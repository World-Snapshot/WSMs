#!/usr/bin/env python3
"""Encode paired instance-cropped MP-100 RGB and Universal Pose images.

The instance crop manifest is the source of truth. Each RGB crop and its
single-instance pose rendering receive the identical letterbox transform,
preserving geometry without stretching or discarding annotated content.

Output follows the standard WSM latent-cache contract::

    output_dir/shards/0_2/shard_00000_of_00008.pt
    output_dir/shards/Universal_Pose/shard_00000_of_00008.pt

Batch parts are atomic and resumable. A completed shard removes its parts to
avoid duplicate storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
RGB_MODALITY = "0-2"
POSE_MODALITY = "Universal_Pose"
MODALITIES = (RGB_MODALITY, POSE_MODALITY)
KNOWN_INVALID_SOURCE_FILES = {
    # Downloaded source is a woman's face rather than the annotated Amur tiger.
    "amur_tiger_body/flickr/2/image58124.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../../wsg/output/PoseAnything_MP100"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mp100_universal_pose_flux2_instance_crop_latents"),
    )
    parser.add_argument(
        "--flux-model-path",
        type=Path,
        default=Path("models/black-forest-labs/FLUX.2-klein-4B"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--cache-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--patch-hw", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--keep-parts-after-merge", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def safe_modality_name(modality: str) -> str:
    return str(modality).replace("/", "_").replace(",", "_").replace("-", "_")


def split_for(sample_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha1(sample_id.encode("utf-8")).digest()[:8], "big") / 2**64
    if bucket < 0.05:
        return "test"
    if bucket < 0.10:
        return "val"
    return "train"


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def shard_path(output_dir: Path, modality: str, shard_id: int, num_shards: int) -> Path:
    return (
        output_dir
        / "shards"
        / safe_modality_name(modality)
        / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
    )


def part_root(output_dir: Path, shard_id: int, num_shards: int) -> Path:
    return output_dir / "parts" / f"shard_{shard_id:05d}_of_{num_shards:05d}"


def part_path(output_dir: Path, modality: str, shard_id: int, num_shards: int, batch_id: int) -> Path:
    return part_root(output_dir, shard_id, num_shards) / safe_modality_name(modality) / f"batch_{batch_id:08d}.pt"


def batch_manifest_path(output_dir: Path, shard_id: int, num_shards: int, batch_id: int) -> Path:
    return part_root(output_dir, shard_id, num_shards) / "_manifests" / f"batch_{batch_id:08d}.json"


def build_records(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "instance_crop_pose_visualization_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run render_mp100_instance_crops.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rgb_root = dataset_root / str(manifest.get("rgb_crops", "crop_images"))
    pose_root = dataset_root / str(manifest.get("pose_black_crops", "crop_images_pose_black"))
    records: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        source_relative = Path(str(item["source_file_name"]))
        if source_relative.as_posix() in KNOWN_INVALID_SOURCE_FILES:
            continue
        rgb_relative = Path(str(item["rgb"]))
        pose_relative = Path(str(item["pose_black"]))
        rgb_path = rgb_root / rgb_relative
        pose_path = pose_root / pose_relative
        if not rgb_path.is_file() or not pose_path.is_file():
            continue
        sample_id = str(item["sample_id"])
        source_sample_id = f"mp100::{source_relative.with_suffix('').as_posix()}"
        category_name = str(source_relative.parts[0])
        records.append(
            {
                "global_index": len(records),
                "sample_id": sample_id,
                "source_sample_id": source_sample_id,
                "annotation_id": int(item["annotation_id"]),
                "relative_path": rgb_relative.as_posix(),
                "source_relative_path": source_relative.as_posix(),
                "rgb_path": str(rgb_path),
                "pose_path": str(pose_path),
                "category_id": int(item["category_id"]),
                "category_name": category_name,
                "official_category_name": str(item.get("category_name") or category_name),
                "prompt": f"a photo of {category_name.replace('_', ' ')}",
                # Prevent source-image leakage across train/val/test.
                "split": split_for(source_sample_id),
                "num_instances": 1,
                "source_size": item.get("source_size"),
                "instance_crop_size": item.get("crop_size"),
                "instance_crop_box": item.get("crop_box_source_ltrb"),
            }
        )
    category_count = len({record["category_name"] for record in records})
    if category_count != 100:
        raise RuntimeError(f"Expected 100 MP-100 categories, found {category_count}")
    return records

class MP100PairDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], patch_hw: int):
        self.records = records
        self.patch_hw = int(patch_hw)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _image_tensor(
        path: str,
        patch_hw: int,
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int, int, int]]:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_size = image.size
            width, height = original_size
            scale = min(patch_hw / width, patch_hw / height)
            resize_width = max(1, min(patch_hw, round(width * scale)))
            resize_height = max(1, min(patch_hw, round(height * scale)))
            resized = image.resize((resize_width, resize_height), Image.Resampling.BICUBIC)
            left = (patch_hw - resize_width) // 2
            top = (patch_hw - resize_height) // 2
            right = patch_hw - resize_width - left
            bottom = patch_hw - resize_height - top
            canvas = Image.new("RGB", (patch_hw, patch_hw), "black")
            canvas.paste(resized, (left, top))
            array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
        return (
            torch.from_numpy(array).permute(2, 0, 1).contiguous(),
            original_size,
            (left, top, right, bottom),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rgb, rgb_size, rgb_padding = self._image_tensor(record["rgb_path"], self.patch_hw)
        pose, pose_size, pose_padding = self._image_tensor(record["pose_path"], self.patch_hw)
        if rgb_size != pose_size or rgb_padding != pose_padding:
            raise RuntimeError(
                f"Pair geometry mismatch for {record['sample_id']}: "
                f"RGB={rgb_size}/{rgb_padding}, pose={pose_size}/{pose_padding}"
            )
        width, height = rgb_size
        left, top, right, bottom = rgb_padding
        return {
            "rgb": rgb,
            "pose": pose,
            "record": record,
            "crop_info": {
                "patch_hw": self.patch_hw,
                "original_width": int(width),
                "original_height": int(height),
                "resize_width": self.patch_hw - left - right,
                "resize_height": self.patch_hw - top - bottom,
                "pad_left": left,
                "pad_top": top,
                "pad_right": right,
                "pad_bottom": bottom,
                "crop_mode": "letterbox_square",
                "decode_restore": {
                    "remove_padding": [left, top, right, bottom],
                    "width": int(width),
                    "height": int(height),
                    "resampling": "bicubic",
                },
            },
        }


def collate_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rgb": torch.stack([item["rgb"] for item in items]),
        "pose": torch.stack([item["pose"] for item in items]),
        "records": [item["record"] for item in items],
        "crop_infos": [item["crop_info"] for item in items],
    }


def load_vae(model_path: Path, device: str, dtype: torch.dtype) -> AutoencoderKLFlux2:
    if (model_path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), torch_dtype=dtype)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    return vae.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def encode_images(
    vae: AutoencoderKLFlux2,
    images: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    latent = vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample()
    return latent.detach().cpu().to(cache_dtype).unsqueeze(2)


def complete_batches(output_dir: Path, shard_id: int, num_shards: int) -> set[int]:
    manifest_dir = part_root(output_dir, shard_id, num_shards) / "_manifests"
    completed = set()
    for path in sorted(manifest_dir.glob("batch_*.json")):
        try:
            batch_id = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if all(part_path(output_dir, modality, shard_id, num_shards, batch_id).is_file() for modality in MODALITIES):
            completed.add(batch_id)
    return completed


def merge_shard(
    output_dir: Path,
    shard_id: int,
    num_shards: int,
    common_meta: dict[str, Any],
    cache_dtype: torch.dtype,
    keep_parts: bool,
) -> None:
    root = part_root(output_dir, shard_id, num_shards)
    for modality in MODALITIES:
        paths = sorted((root / safe_modality_name(modality)).glob("batch_*.pt"))
        if not paths:
            raise RuntimeError(f"No batch parts found for {modality}, shard {shard_id}/{num_shards}")
        parts = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
        payload = {
            **common_meta,
            "modality": modality,
            "num_samples": sum(len(part["sample_ids"]) for part in parts),
            "sample_ids": sum((list(part["sample_ids"]) for part in parts), []),
            "sample_sources": sum((list(part["sample_sources"]) for part in parts), []),
            "crop_infos": sum((list(part["crop_infos"]) for part in parts), []),
            "prompts": sum((list(part["prompts"]) for part in parts), []),
            "splits": sum((list(part["splits"]) for part in parts), []),
            "source_shard_indices": sum((list(part["shard_indices"]) for part in parts), []),
            "shard_indices": sum((list(part["shard_indices"]) for part in parts), []),
            "latents": torch.cat([part["latents"] for part in parts]),
            "filtered_sample_ids": True,
            "filter_invalid_modalities": True,
            "part_cache_format": "mp100_universal_pose_instance_crop_batch_parts_v2",
            "partial_shard": False,
        }
        atomic_torch_save(payload, shard_path(output_dir, modality, shard_id, num_shards))
        print(
            f"saved {modality}: {tuple(payload['latents'].shape)} -> "
            f"{shard_path(output_dir, modality, shard_id, num_shards)}",
            flush=True,
        )
        del parts, payload
    if not keep_parts:
        shutil.rmtree(root)


def write_index(output_dir: Path, dataset_root: Path, records: list[dict[str, Any]]) -> None:
    split_counts = Counter(record["split"] for record in records)
    category_counts = Counter(record["category_name"] for record in records)
    index_payload = {
        "format": "mp100_universal_pose_instance_crop_index_v2",
        "dataset": "PoseAnything_MP100",
        "dataset_root": str(dataset_root),
        "modalities": list(MODALITIES),
        "source_representation": "annotation_bbox_instance_crop",
        "transform": "instance_crop_then_letterbox_square",
        "num_samples": len(records),
        "num_categories": len(category_counts),
        "split_counts": dict(split_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "routes": [
            {"condition": RGB_MODALITY, "target": POSE_MODALITY, "task": "universal_pose_estimation"},
            {"condition": POSE_MODALITY, "target": RGB_MODALITY, "task": "universal_pose_control"},
        ],
        "records": records,
    }
    atomic_json_save(index_payload, output_dir / "dataset_index.json")
    prompt_overrides: dict[str, str] = {}
    existing_prompt_path = output_dir / "prompts.pt"
    if existing_prompt_path.is_file():
        existing_prompts = torch.load(existing_prompt_path, map_location="cpu", weights_only=False)
        prompt_overrides = {
            str(sample_id): str(prompt)
            for sample_id, prompt in zip(
                existing_prompts.get("sample_ids", []),
                existing_prompts.get("prompts", []),
            )
        }
    atomic_torch_save(
        {
            "dataset": "PoseAnything_MP100",
            "num_samples": len(records),
            "sample_ids": [record["sample_id"] for record in records],
            "prompts": [
                prompt_overrides.get(
                    record["sample_id"],
                    prompt_overrides.get(record["source_sample_id"], record["prompt"]),
                )
                for record in records
            ],
            "splits": [record["split"] for record in records],
        },
        output_dir / "prompts.pt",
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_id < num_shards")
    dataset_root = resolve_project_path(args.dataset_root)
    output_dir = resolve_project_path(args.output_dir)
    model_path = resolve_project_path(args.flux_model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(dataset_root)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError("No complete MP-100 RGB/Pose pairs found")
    if args.index_only or not (output_dir / "dataset_index.json").is_file():
        write_index(output_dir, dataset_root, records)
    print(
        f"[mp100-index] pairs={len(records)} categories={len(set(r['category_name'] for r in records))} "
        f"transform=instance_crop_then_letterbox_square output={output_dir}",
        flush=True,
    )
    if args.index_only:
        return

    final_paths = [shard_path(output_dir, modality, args.shard_id, args.num_shards) for modality in MODALITIES]
    if args.skip_existing and all(path.is_file() for path in final_paths):
        print(f"skip complete shard {args.shard_id}/{args.num_shards}", flush=True)
        return

    shard_records = records[args.shard_id :: args.num_shards]
    dataset = MP100PairDataset(shard_records, args.patch_hw)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_pairs,
    )
    completed = complete_batches(output_dir, args.shard_id, args.num_shards)
    torch_dtype = dtype_from_name(args.dtype)
    cache_dtype = dtype_from_name(args.cache_dtype)
    vae = load_vae(model_path, args.device, torch_dtype)

    common_meta = {
        "dataset": "PoseAnything_MP100",
        "dataset_paths": [str(dataset_root)],
        "vae_path": str(model_path),
        "vae_family": "flux2_klein_frame",
        "latent_backend": "flux2_frame",
        "global_num_samples": len(records),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_frames": 1,
        "latent_num_frames": 1,
        "patch_hw": args.patch_hw,
        "height": args.patch_hw,
        "width": args.patch_hw,
        "crop_mode": "letterbox_square",
        "decode_restore": "remove symmetric padding, then resize to original_width/original_height",
        "normalization_mode": "linear",
        "normalization_modes": {RGB_MODALITY: "linear", POSE_MODALITY: "linear"},
        "data_type": "paired_images",
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype,
        "modalities": list(MODALITIES),
        "batch_size": args.batch_size,
    }

    progress = tqdm(enumerate(loader), total=len(loader), desc=f"MP100 shard {args.shard_id}/{args.num_shards}")
    for batch_id, batch in progress:
        if batch_id in completed:
            continue
        pair_images = torch.cat([batch["rgb"], batch["pose"]], dim=0)
        pair_latents = encode_images(vae, pair_images, args.device, torch_dtype, cache_dtype)
        rgb_latents, pose_latents = pair_latents.split(batch["rgb"].shape[0])
        records_batch = batch["records"]
        sample_ids = [record["sample_id"] for record in records_batch]
        sources = [
            {
                "dataset": "PoseAnything_MP100",
                "relative_path": record["relative_path"],
                "source_relative_path": record["source_relative_path"],
                "annotation_id": record["annotation_id"],
                "rgb_path": record["rgb_path"],
                "pose_path": record["pose_path"],
                "category_id": record["category_id"],
                "category_name": record["category_name"],
                "official_category_name": record["official_category_name"],
                "num_instances": record["num_instances"],
                "instance_crop_box": record["instance_crop_box"],
                "split": record["split"],
            }
            for record in records_batch
        ]
        common_part = {
            "sample_ids": sample_ids,
            "sample_sources": sources,
            "crop_infos": batch["crop_infos"],
            "prompts": [record["prompt"] for record in records_batch],
            "splits": [record["split"] for record in records_batch],
            "shard_indices": [record["global_index"] for record in records_batch],
            "part_batch_id": batch_id,
        }
        for modality, latents in ((RGB_MODALITY, rgb_latents), (POSE_MODALITY, pose_latents)):
            atomic_torch_save(
                {**common_part, "modality": modality, "latents": latents},
                part_path(output_dir, modality, args.shard_id, args.num_shards, batch_id),
            )
        atomic_json_save(
            {"batch_id": batch_id, "sample_ids": sample_ids, "modalities": list(MODALITIES)},
            batch_manifest_path(output_dir, args.shard_id, args.num_shards, batch_id),
        )
        completed.add(batch_id)

    if len(completed) != len(loader):
        raise RuntimeError(f"Incomplete shard: {len(completed)}/{len(loader)} batches")
    merge_shard(
        output_dir,
        args.shard_id,
        args.num_shards,
        common_meta,
        cache_dtype,
        args.keep_parts_after_merge,
    )


if __name__ == "__main__":
    main()
