#!/usr/bin/env python3
"""Build a derived FLUX.2 latent target for dense RGB-to-person-box-mask training."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image


PALETTE = np.asarray(
    [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 224, 64),
        (224, 64, 255),
        (64, 255, 224),
        (255, 144, 64),
        (160, 255, 64),
        (144, 64, 255),
        (255, 64, 160),
    ],
    dtype=np.uint8,
)


def safe_modality_name(value: str) -> str:
    return str(value).replace("-", "_")


def shard_path(root: Path, modality: str, shard_id: int, num_shards: int) -> Path:
    return root / "shards" / safe_modality_name(modality) / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"


def parse_shard_ids(value: str, num_shards: int) -> list[int]:
    if not value.strip():
        return list(range(num_shards))
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(token))
    result = list(dict.fromkeys(result))
    if any(item < 0 or item >= num_shards for item in result):
        raise ValueError(f"shard ids must be within [0, {num_shards})")
    return result


def sample_id(wsv_path: str, frame_key: str) -> str:
    return f"{wsv_path}::{frame_key}-{frame_key}"


def load_bbox_rows(paths: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for wsv_path, frame_key, boxes, has_box in zip(
                data["wsv_path"].astype(str),
                data["frame_key"].astype(str),
                data["boxes"].astype(np.float32),
                data["has_box"].astype(np.float32),
            ):
                key = sample_id(wsv_path, frame_key)
                if key in rows:
                    raise RuntimeError(f"duplicate bbox sample id: {key}")
                rows[key] = (boxes, has_box)
    return rows


def render_pseudocolor_box_mask(
    boxes: np.ndarray,
    has_box: np.ndarray,
    image_size: int,
) -> np.ndarray:
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    valid = [box for box, flag in zip(boxes, has_box) if flag > 0.5]
    valid.sort(key=lambda box: float(box[0]))
    for instance_index, box in enumerate(valid):
        cx, cy, width, height = [float(item) for item in box]
        x0 = int(np.floor((cx - width / 2.0) * image_size))
        y0 = int(np.floor((cy - height / 2.0) * image_size))
        x1 = int(np.ceil((cx + width / 2.0) * image_size))
        y1 = int(np.ceil((cy + height / 2.0) * image_size))
        x0 = int(np.clip(x0, 0, image_size - 1))
        y0 = int(np.clip(y0, 0, image_size - 1))
        x1 = int(np.clip(x1, x0 + 1, image_size))
        y1 = int(np.clip(y1, y0 + 1, image_size))
        canvas[y0:y1, x0:x1] = PALETTE[instance_index % len(PALETTE)]
    return canvas


def subset_payload(payload: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    total = len(payload["sample_ids"])
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "latents":
            continue
        if isinstance(value, list) and len(value) == total:
            result[key] = [value[index] for index in indices]
        elif isinstance(value, tuple) and len(value) == total:
            result[key] = [value[index] for index in indices]
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total:
            result[key] = value[indices].clone()
        else:
            result[key] = value
    return result


def load_vae(model_path: Path, device: str, dtype: torch.dtype) -> AutoencoderKLFlux2:
    if (model_path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), torch_dtype=dtype)
    return vae.to(device).eval().requires_grad_(False)


def save_previews(
    source_root: Path,
    rgb_modality: str,
    bbox_rows: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    num_shards: int,
    image_size: int,
    limit: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for shard_id in range(num_shards):
        payload = torch.load(
            shard_path(source_root, rgb_modality, shard_id, num_shards),
            map_location="cpu", weights_only=False, mmap=True,
        )
        for key in map(str, payload["sample_ids"]):
            row = bbox_rows.get(key)
            if row is None:
                continue
            image = render_pseudocolor_box_mask(*row, image_size)
            Image.fromarray(image).save(out_dir / f"sample_{written:03d}.png")
            written += 1
            if written >= limit:
                print(f"[bbox-mask-cache] previews={written} dir={out_dir}", flush=True)
                return
    print(f"[bbox-mask-cache] previews={written} dir={out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--output-modality", default="88-89_bbox_pseudocolor_mask")
    parser.add_argument("--rgb-modality", default="0-2")
    parser.add_argument("--bbox-npz", type=Path, action="append", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-ids", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=Path("debug/bbox_pseudocolor_mask/previews"))
    parser.add_argument("--preview-count", type=int, default=8)
    args = parser.parse_args()

    bbox_rows = load_bbox_rows(args.bbox_npz)
    print(f"[bbox-mask-cache] bbox_samples={len(bbox_rows)} modality={args.output_modality}", flush=True)
    if args.preview_only:
        save_previews(
            args.source_cache_dir, args.rgb_modality, bbox_rows, args.preview_dir,
            args.num_shards, args.image_size, args.preview_count,
        )
        return

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(20260811)
    vae = load_vae(args.model_path, args.device, torch_dtype)
    output_dir = args.output_cache_dir / "shards" / safe_modality_name(args.output_modality)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for shard_id in parse_shard_ids(args.shard_ids, args.num_shards):
        source_path = shard_path(args.source_cache_dir, args.rgb_modality, shard_id, args.num_shards)
        output_path = shard_path(args.output_cache_dir, args.output_modality, shard_id, args.num_shards)
        if args.skip_existing and output_path.exists():
            print(f"[bbox-mask-cache] skip existing {output_path}", flush=True)
            continue
        payload = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        indices = [index for index, key in enumerate(map(str, payload["sample_ids"])) if key in bbox_rows]
        print(f"[bbox-mask-cache] shard={shard_id} selected={len(indices)}", flush=True)
        latent_batches: list[torch.Tensor] = []
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start:start + args.batch_size]
            images = [
                render_pseudocolor_box_mask(*bbox_rows[str(payload["sample_ids"][index])], args.image_size)
                for index in chunk
            ]
            tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            with torch.no_grad():
                latents = vae.encode(tensor.to(args.device, dtype=torch_dtype)).latent_dist.sample()
            latent_batches.append(latents.detach().cpu().to(torch.bfloat16).unsqueeze(2))
            if start == 0 or (start // args.batch_size + 1) % 25 == 0:
                print(f"[bbox-mask-cache] shard={shard_id} rows={min(start + len(chunk), len(indices))}/{len(indices)}", flush=True)
        result = subset_payload(payload, indices)
        result["latents"] = torch.cat(latent_batches) if latent_batches else torch.empty((0, 32, 1, 60, 60), dtype=torch.bfloat16)
        result["sample_ids"] = [str(payload["sample_ids"][index]) for index in indices]
        result["num_samples"] = len(indices)
        result["global_num_samples"] = len(bbox_rows)
        result["modality"] = args.output_modality
        result["derived_from"] = {
            "source_modality": "88-89",
            "representation": "left-to-right fixed-palette filled person bounding-box instance mask",
            "background_rgb": [0, 0, 0],
            "palette_rgb": PALETTE.tolist(),
            "bbox_npz": [str(item) for item in args.bbox_npz],
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(result, temporary)
        os.replace(temporary, output_path)
        total_written += len(indices)
        print(f"[bbox-mask-cache] saved={output_path} samples={len(indices)}", flush=True)
    print(f"[bbox-mask-cache] done samples={total_written}", flush=True)


if __name__ == "__main__":
    main()
