#!/usr/bin/env python3
"""Rebuild RGB pose-visualization Flux2 VAE latents from source WSV/WSG.

This is intended for rendered visualization modalities such as:
  * 90-92: SMPLer-X RGB render
  * 99-101: MediaPipe RGB render

It preserves shard row order and metadata alignment. Source values are treated as
RGB bytes on [0,255], with -1 as black background, and encoded linearly as
VAE input in [-1,1].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERSION = "pose_rgb_linear_0_255_v1"


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(name)]


def safe_modality(value: str) -> str:
    return str(value).replace("-", "_")


def parse_channels(modality: str) -> tuple[list[int], str]:
    parts = str(modality).split("-")
    if len(parts) == 1:
        chans = [int(parts[0])]
    elif len(parts) == 2:
        start, end = map(int, parts)
        chans = list(range(start, end + 1))
    else:
        raise ValueError(f"Bad modality: {modality}")
    if len(chans) != 3:
        raise ValueError(f"Expected exactly 3 RGB channels, got {modality} -> {chans}")
    return chans, str(modality)


def parse_shard_ids(value: str, shard_paths: list[Path]) -> list[Path]:
    if not value:
        return shard_paths
    by_index = {}
    for path in shard_paths:
        idx = int(path.stem.split("_of_", 1)[0].split("shard_", 1)[1])
        by_index[idx] = path
    wanted = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.extend(range(int(a), int(b) + 1))
        else:
            wanted.append(int(part))
    out, seen = [], set()
    for idx in wanted:
        if idx not in by_index:
            raise ValueError(f"Shard id {idx} not found among {sorted(by_index)}")
        if idx not in seen:
            seen.add(idx)
            out.append(by_index[idx])
    return out


def load_vae(model_path: Path, device: str, dtype: torch.dtype) -> AutoencoderKLFlux2:
    if (model_path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), torch_dtype=dtype)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    vae.to(device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    return vae


def crop_frame(frame: np.ndarray, crop_info: dict[str, Any]) -> np.ndarray:
    patch_hw = int(crop_info.get("patch_hw", frame.shape[0]))
    crop_y = int(crop_info.get("crop_y", 0))
    crop_x = int(crop_info.get("crop_x", 0))
    h, w = frame.shape[:2]
    if h < patch_hw or w < patch_hw:
        pad_width = ((0, max(0, patch_hw - h)), (0, max(0, patch_hw - w)), (0, 0))
        frame = np.pad(frame, pad_width, mode="constant", constant_values=-1.0)
    return frame[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :]


def source_path_for(source: dict[str, Any], sample_id: str) -> str:
    path = str(source.get("path") or "") if isinstance(source, dict) else ""
    if not path and "::" in sample_id:
        path = sample_id.split("::", 1)[0]
    if not path:
        raise ValueError(f"Missing source path for sample_id={sample_id}")
    return path


def frame_key_for(source: dict[str, Any], fallback_frame_key: str) -> str:
    keys = source.get("frame_keys") if isinstance(source, dict) else None
    if isinstance(keys, (list, tuple)) and keys:
        return str(keys[0])
    return str(fallback_frame_key)


def load_frame_from_npz(npz: np.lib.npyio.NpzFile, path: str, source: dict[str, Any], sample_id: str, frame_key: str) -> np.ndarray:
    storage = str(source.get("storage") or "") if isinstance(source, dict) else ""
    if frame_key and frame_key in npz.files:
        return np.asarray(npz[frame_key]).astype(np.float32, copy=False)
    if storage == "wsv":
        raise KeyError(f"{path} missing frame_key={frame_key} for sample_id={sample_id}")
    if "data" in npz.files:
        return np.asarray(npz["data"]).astype(np.float32, copy=False)
    if "wsg_data" in npz.files:
        return np.asarray(npz["wsg_data"]).astype(np.float32, copy=False)
    raise KeyError(f"{path} has no frame {frame_key}, data, or wsg_data")


def frame_to_rgb_tensor(frame: np.ndarray, crop_info: dict[str, Any], channels: list[int]) -> torch.Tensor:
    cropped = crop_frame(frame, crop_info)
    h, w = cropped.shape[:2]
    image = np.zeros((h, w, 3), dtype=np.float32)
    if cropped.ndim == 3 and max(channels) < cropped.shape[-1]:
        image = cropped[..., channels].astype(np.float32, copy=True)
    image[~np.isfinite(image)] = 0.0
    image = np.where(image == -1, 0.0, image)
    image = np.clip(image, 0.0, 255.0) / 127.5 - 1.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False))


@torch.no_grad()
def encode_batch(vae, images: torch.Tensor, device: str, dtype: torch.dtype, cache_dtype: torch.dtype) -> torch.Tensor:
    latents = vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample()
    return latents.detach().cpu().to(cache_dtype).unsqueeze(2)


def patch_payload_metadata(payload: dict[str, Any], modality: str) -> None:
    payload["normalization_mode"] = "linear"
    modes = dict(payload.get("normalization_modes") or {})
    modes[str(modality)] = "linear"
    payload["normalization_modes"] = modes
    payload["pose_rgb_linear_version"] = VERSION
    payload["pose_rgb_linear_note"] = "RGB visualization channels are read as [0,255] with -1 as black and encoded linearly into [-1,1]."


def rebuild_shard(shard_path: Path, modality: str, channels: list[int], vae, device: str, dtype: torch.dtype, cache_dtype: torch.dtype, batch_size: int, dry_run: bool) -> dict[str, Any]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if str(payload.get("modality")) != str(modality):
        raise ValueError(f"Refusing wrong shard for {modality}: {shard_path} modality={payload.get('modality')}")
    sample_ids = list(payload.get("sample_ids") or [])
    sample_sources = list(payload.get("sample_sources") or [{} for _ in sample_ids])
    frame_keys = list(payload.get("frame_keys") or [""] * len(sample_ids))
    crop_infos = list(payload.get("crop_infos") or [{} for _ in sample_ids])
    n = int(payload["latents"].shape[0])
    if not (n == len(sample_ids) == len(sample_sources) == len(frame_keys) == len(crop_infos)):
        raise RuntimeError(f"Metadata length mismatch in {shard_path}")
    if dry_run:
        return {"shard": str(shard_path), "samples": n, "dry_run": True}
    out_latents = payload["latents"].clone()
    groups: dict[str, list[int]] = {}
    for i, sample_id in enumerate(sample_ids):
        groups.setdefault(source_path_for(sample_sources[i], str(sample_id)), []).append(i)
    pending_images: list[torch.Tensor] = []
    pending_positions: list[int] = []
    rebuilt = 0
    missing = []

    def flush() -> None:
        nonlocal pending_images, pending_positions, rebuilt
        if not pending_images:
            return
        images = torch.stack(pending_images, dim=0)
        encoded = encode_batch(vae, images, device, dtype, cache_dtype)
        for out_idx, pos in enumerate(pending_positions):
            out_latents[pos].copy_(encoded[out_idx])
        rebuilt += int(images.shape[0])
        pending_images = []
        pending_positions = []

    for path, indices in tqdm(groups.items(), desc=f"rebuild {modality} {shard_path.name}", dynamic_ncols=True):
        with np.load(path, allow_pickle=False) as npz:
            for i in indices:
                frame_key = frame_key_for(sample_sources[i], str(frame_keys[i]))
                try:
                    frame = load_frame_from_npz(npz, path, sample_sources[i], str(sample_ids[i]), frame_key)
                except KeyError:
                    missing.append({"row": int(i), "sample_id": str(sample_ids[i]), "path": path, "frame_key": frame_key})
                    continue
                pending_images.append(frame_to_rgb_tensor(frame, crop_infos[i], channels))
                pending_positions.append(i)
                if len(pending_images) >= batch_size:
                    flush()
        flush()
    patch_payload_metadata(payload, modality)
    payload["latents"] = out_latents
    payload["missing_source_frames"] = missing
    tmp = shard_path.with_suffix(shard_path.suffix + f".{safe_modality(modality)}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, shard_path)
    return {"shard": str(shard_path), "samples": n, "rebuilt": rebuilt, "missing": len(missing), "dry_run": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("/research/cbim/vast/sf895/code/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset"))
    ap.add_argument("--model-path", type=Path, default=Path("/research/cbim/vast/sf895/code/WSMs/WSM_v1/pretrained/FLUX.2-dev"))
    ap.add_argument("--modality", required=True, help="RGB modality range, e.g. 90-92 or 99-101")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--cache-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shards", default="", help="Comma/range shard ids, e.g. 0,2-4. Empty means all.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", type=Path, default=Path("/research/cbim/vast/sf895/code/WSMs/WSM_v1/train_log_sparse/debug_pose_rgb_rebuild/rebuild_pose_rgb_summary.json"))
    args = ap.parse_args()
    channels, modality = parse_channels(args.modality)
    shard_dir = resolve_path(args.cache_root) / "shards" / safe_modality(modality)
    shard_paths = parse_shard_ids(args.shards, sorted(shard_dir.glob("shard_*_of_*.pt")))
    if not shard_paths:
        raise RuntimeError(f"No shards found under {shard_dir}")
    dtype = dtype_from_name(args.dtype)
    cache_dtype = dtype_from_name(args.cache_dtype)
    vae = None if args.dry_run else load_vae(resolve_path(args.model_path), args.device, dtype)
    results = []
    for shard in shard_paths:
        results.append(rebuild_shard(shard, modality, channels, vae, args.device, dtype, cache_dtype, args.batch_size, args.dry_run))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"version": VERSION, "modality": modality, "channels": channels, "results": results}, indent=2), encoding="utf-8")
    print(args.summary)


if __name__ == "__main__":
    main()
