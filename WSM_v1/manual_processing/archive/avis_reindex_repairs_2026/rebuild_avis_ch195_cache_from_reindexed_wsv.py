#!/usr/bin/env python3
"""Rebuild channel-195 FLUX VAE latents after AVIS WSV reindexing.

The intended repair order is logically:
  old lexicographic index -> fixed semantic channel -> VAE encode -> metadata rekey.

If WSV files have already been reindexed to natural AVIS frame order, this tool
reconstructs the same result by using the old channel-195 shard sample_ids as the
source of truth and reading the corresponding natural frame from the reindexed
WSV. It keeps the old sample_ids in the rebuilt shard; run
rekey_avis_flux_cache_after_reindex.py afterwards for all caches.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLFlux2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.wsg_dataset import (  # noqa: E402
    MASK_VALUE,
    WSGNormalizationStore,
    apply_resize_crop_info,
    load_wsg_normalization_ranges,
)

CHANNEL = 195
SAMPLE_RE = re.compile(
    r"(?P<prefix>.*?/AVIS_(?P<split>train|val|test)_results/(?P<video>.*?)/wsv/merge\.wsv)::index_(?P<a>\d+)-index_(?P<b>\d+)$"
)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(name)]


def load_split_mapping(json_path: Path) -> Dict[str, Dict[int, int]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    result: Dict[str, Dict[int, int]] = {}
    for video in payload.get("videos", []):
        file_names = video.get("file_names", [])
        if not file_names:
            continue
        video_name = file_names[0].split("/", 1)[0]
        original_index = {name: idx + 1 for idx, name in enumerate(file_names)}
        lexicographic = sorted(file_names)
        result[video_name] = {old_idx + 1: original_index[name] for old_idx, name in enumerate(lexicographic)}
    return result


def load_mappings(base_output_dir: Path) -> Dict[str, Dict[str, Dict[int, int]]]:
    return {
        "train": load_split_mapping(base_output_dir / "AVIS_train_results" / "train.json"),
        "val": load_split_mapping(base_output_dir / "AVIS_val_results" / "val.json"),
        "test": load_split_mapping(base_output_dir / "AVIS_test_results" / "test.json"),
    }


def natural_key_for_old_sample(sample_id: str, source: dict, mappings: Dict[str, Dict[str, Dict[int, int]]]) -> str:
    match = SAMPLE_RE.fullmatch(str(sample_id))
    if match:
        split = match.group("split")
        video = match.group("video")
        old_index = int(match.group("a"))
        natural_index = int(mappings.get(split, {}).get(video, {}).get(old_index, old_index))
        return f"index_{natural_index:08d}"
    keys = list(source.get("frame_keys") or []) if isinstance(source, dict) else []
    if not keys:
        raise ValueError(f"Cannot infer frame key for non-AVIS sample_id={sample_id}")
    return str(keys[0])


def load_config_from_payload(payload: dict) -> dict:
    config_path = payload.get("config")
    if not config_path:
        return {}
    path = resolve_project_path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_vae(model_or_vae_path: str, device: str, torch_dtype: torch.dtype) -> AutoencoderKLFlux2:
    path = resolve_project_path(model_or_vae_path)
    if (path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(path), subfolder="vae", torch_dtype=torch_dtype)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(path), torch_dtype=torch_dtype)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    vae.to(device=device, dtype=torch_dtype)
    vae.eval()
    return vae


def make_normalizer(payload: dict) -> WSGNormalizationStore:
    cfg = load_config_from_payload(payload)
    params = ((cfg.get("dataset") or {}).get("params") or {}) if isinstance(cfg, dict) else {}
    normalization_cache = params.get("normalization_cache_path")
    modality_config_path = resolve_project_path(payload.get("modality_config_path") or params.get("modality_config_path") or "")
    portable_ranges = load_wsg_normalization_ranges(modality_config_path) if modality_config_path.exists() else {}
    return WSGNormalizationStore(
        resolve_project_path(normalization_cache) if normalization_cache else None,
        normalization_mode=str(payload.get("normalization_mode") or "log"),
        show_progress=False,
        portable_channel_ranges=portable_ranges,
    )


def build_image_tensor(frame: np.ndarray, crop_info: dict, normalizer: WSGNormalizationStore, dataset_paths: list[str]) -> torch.Tensor:
    cropped = apply_resize_crop_info(frame, crop_info)
    if cropped.ndim != 3:
        raise ValueError(f"Expected cropped frame [H,W,C], got {cropped.shape}")
    if CHANNEL >= cropped.shape[-1]:
        plane = np.full(cropped.shape[:2], MASK_VALUE, dtype=np.float32)
    else:
        plane = cropped[..., CHANNEL].astype(np.float32, copy=False)
    plane = normalizer.normalize_channel(plane, CHANNEL, dataset_paths)
    stacked = np.stack([plane, plane, plane], axis=0).astype(np.float32, copy=False)
    valid = stacked != MASK_VALUE
    stacked[valid] = stacked[valid] * 2.0 - 1.0
    return torch.from_numpy(stacked[:, None])  # [3,1,H,W]


def encode_batch(vae, videos: torch.Tensor, device: str, torch_dtype: torch.dtype, cache_dtype: torch.dtype) -> torch.Tensor:
    # videos: [B,3,1,H,W]
    images = videos[:, :, 0].to(device=device, dtype=torch_dtype)
    latents = vae.encode(images).latent_dist.sample()
    return latents.detach().cpu().to(dtype=cache_dtype).unsqueeze(2)


def iter_shards(cache_root: Path, shard_glob: str) -> list[Path]:
    shard_dir = cache_root / "shards" / "195"
    return sorted(shard_dir.glob(shard_glob))


def rebuild_shard(
    shard_path: Path,
    mappings: Dict[str, Dict[str, Dict[int, int]]],
    vae,
    device: str,
    torch_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    batch_size: int,
    dry_run: bool,
    limit_samples: Optional[int],
) -> dict[str, Any]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if str(payload.get("modality")) != "195":
        raise ValueError(f"{shard_path} is not a channel-195 shard")
    sample_ids = list(payload.get("sample_ids") or [])
    sample_sources = list(payload.get("sample_sources") or [{} for _ in sample_ids])
    crop_infos = list(payload.get("crop_infos") or [{} for _ in sample_ids])
    if not (len(sample_ids) == len(sample_sources) == len(crop_infos)):
        raise RuntimeError(f"metadata length mismatch in {shard_path}")
    if limit_samples is not None:
        sample_ids = sample_ids[:limit_samples]
        sample_sources = sample_sources[:limit_samples]
        crop_infos = crop_infos[:limit_samples]
    normalizer = make_normalizer(payload)
    dataset_paths = list(payload.get("normalization_dataset_paths") or payload.get("dataset_paths") or [])

    rebuilt_chunks = []
    rebuilt_count = 0
    missing = 0
    for start in tqdm(range(0, len(sample_ids), batch_size), desc=f"rebuild {shard_path.name}", dynamic_ncols=True):
        batch_tensors = []
        for sample_id, source, crop_info in zip(sample_ids[start:start + batch_size], sample_sources[start:start + batch_size], crop_infos[start:start + batch_size]):
            path = Path(str(source.get("path") if isinstance(source, dict) else ""))
            frame_key = natural_key_for_old_sample(str(sample_id), source if isinstance(source, dict) else {}, mappings)
            try:
                with np.load(path) as npz:
                    frame = np.asarray(npz[frame_key])
            except Exception as exc:
                missing += 1
                raise RuntimeError(f"Failed reading {path} {frame_key} for old sample_id={sample_id}: {exc}") from exc
            batch_tensors.append(build_image_tensor(frame, crop_info, normalizer, dataset_paths))
        if not batch_tensors:
            continue
        videos = torch.stack(batch_tensors, dim=0)
        rebuilt_chunks.append(encode_batch(vae, videos, device, torch_dtype, cache_dtype))
        rebuilt_count += int(videos.shape[0])
    if dry_run:
        return {"path": str(shard_path), "samples": len(sample_ids), "rebuilt": rebuilt_count, "dry_run": True}
    new_payload = dict(payload)
    if limit_samples is not None:
        # Debug mode only: keep metadata consistent with truncated tensor.
        for key in ("sample_ids", "sample_sources", "video_ids", "frame_keys", "frame_indices", "crop_infos", "source_shard_indices", "shard_indices"):
            if isinstance(new_payload.get(key), list):
                new_payload[key] = new_payload[key][:limit_samples]
        new_payload["num_samples"] = len(sample_ids)
    new_payload["latents"] = torch.cat(rebuilt_chunks, dim=0) if rebuilt_chunks else torch.empty((0, 32, 1, 60, 60), dtype=cache_dtype)
    tmp = shard_path.with_name(shard_path.name + ".rebuild195.tmp")
    torch.save(new_payload, tmp)
    os.replace(tmp, shard_path)
    return {"path": str(shard_path), "samples": len(sample_ids), "rebuilt": rebuilt_count, "missing": missing, "dry_run": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=Path("/research/cbim/vast/sf895/code/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset"))
    ap.add_argument("--base-output-dir", type=Path, default=Path("/common/users/sf895/wave2pixel"))
    ap.add_argument("--flux-model-path", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--cache-dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-glob", default="shard_*.pt")
    ap.add_argument("--limit-samples", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shard_paths = iter_shards(resolve_project_path(args.cache_root), args.shard_glob)
    if not shard_paths:
        raise SystemExit(f"No channel-195 shards found under {args.cache_root}")
    first_payload = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    flux_model_path = args.flux_model_path or first_payload.get("vae_path")
    if not flux_model_path:
        raise SystemExit("Could not infer FLUX VAE path; pass --flux-model-path")
    torch_dtype = dtype_from_name(args.dtype)
    cache_dtype = dtype_from_name(args.cache_dtype or args.dtype)
    mappings = load_mappings(args.base_output_dir)
    print(json.dumps({
        "shards": len(shard_paths),
        "cache_root": str(resolve_project_path(args.cache_root)),
        "flux_model_path": str(flux_model_path),
        "device": args.device,
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype or args.dtype,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
    }, indent=2), flush=True)
    if args.dry_run:
        vae = None
    else:
        vae = load_vae(str(flux_model_path), args.device, torch_dtype)
    totals = {"samples": 0, "rebuilt": 0, "missing": 0}
    for shard_path in shard_paths:
        if args.dry_run:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            count = len(payload.get("sample_ids") or [])
            result = {"path": str(shard_path), "samples": min(count, args.limit_samples) if args.limit_samples else count, "rebuilt": 0, "dry_run": True}
        else:
            result = rebuild_shard(shard_path, mappings, vae, args.device, torch_dtype, cache_dtype, args.batch_size, False, args.limit_samples)
        totals["samples"] += int(result.get("samples", 0))
        totals["rebuilt"] += int(result.get("rebuilt", 0))
        totals["missing"] += int(result.get("missing", 0))
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps(totals, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
