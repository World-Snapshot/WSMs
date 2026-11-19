#!/usr/bin/env python3
"""Rebuild MatSynth rows in dense FLUX2 0-2 RGB latent shards.

MatSynth WSG files were originally written with RGB in [0,1] while WSG RGB
normalization expects [0,255]. After fixing the source WSG files, this tool
re-encodes only rows whose sample source belongs to MatSynth_results and
atomically replaces the selected 0-2 shard files. All non-MatSynth rows and row
ordering metadata are preserved.
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

MODALITY = "0-2"
MODALITY_DIR = "0_2"
VERSION = "matsynth_rgb_0_255_rebuild_v1"


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(name)]


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
    out = []
    seen = set()
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


def load_frame_from_open_npz(npz: np.lib.npyio.NpzFile, path: str, source: dict[str, Any], sample_id: str, frame_key: str) -> np.ndarray:
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


def frame_to_rgb_image(frame: np.ndarray, crop_info: dict[str, Any]) -> torch.Tensor:
    cropped = crop_frame(frame, crop_info)
    if cropped.ndim != 3 or cropped.shape[-1] < 3:
        rgb = np.zeros((*cropped.shape[:2], 3), dtype=np.float32)
    else:
        rgb = cropped[..., :3].astype(np.float32, copy=True)
    rgb[~np.isfinite(rgb)] = 0.0
    # Source WSG should already be repaired to [0,255]. Keep a guard for any
    # stale row accidentally left in [0,1].
    finite = rgb[np.isfinite(rgb)]
    if finite.size and float(finite.max()) <= 2.0:
        rgb *= 255.0
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    image = np.moveaxis(rgb, -1, 0).astype(np.float32, copy=False)
    image = image * 2.0 - 1.0
    return torch.from_numpy(image)


@torch.no_grad()
def encode_batch(vae, images: torch.Tensor, device: str, dtype: torch.dtype, cache_dtype: torch.dtype) -> torch.Tensor:
    latents = vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample()
    return latents.detach().cpu().to(cache_dtype).unsqueeze(2)


def is_matsynth(source: dict[str, Any], sample_id: str) -> bool:
    dataset_name = str(source.get("dataset_name") or "") if isinstance(source, dict) else ""
    path = source_path_for(source, sample_id)
    return dataset_name == "MatSynth_results" or "/MatSynth_results/" in path


def patch_payload_metadata(payload: dict[str, Any], rebuilt: int, preserved: int) -> None:
    payload["matsynth_rgb_rebuild_version"] = VERSION
    payload["matsynth_rgb_rebuilt_rows"] = int(rebuilt)
    payload["matsynth_rgb_preserved_rows"] = int(preserved)
    payload["matsynth_rgb_rebuild_note"] = "Only MatSynth_results rows in modality 0-2 were re-encoded after repairing source WSG RGB from [0,1] to [0,255]."


def rebuild_shard(shard_path: Path, vae, device: str, dtype: torch.dtype, cache_dtype: torch.dtype, batch_size: int, dry_run: bool) -> dict[str, Any]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if str(payload.get("modality")) != MODALITY:
        raise ValueError(f"Refusing non-{MODALITY} shard: {shard_path}")
    sample_ids = list(payload.get("sample_ids") or [])
    sample_sources = list(payload.get("sample_sources") or [{} for _ in sample_ids])
    frame_keys = list(payload.get("frame_keys") or [""] * len(sample_ids))
    crop_infos = list(payload.get("crop_infos") or [{} for _ in sample_ids])
    n = int(payload["latents"].shape[0])
    if not (n == len(sample_ids) == len(sample_sources) == len(frame_keys) == len(crop_infos)):
        raise RuntimeError(f"Metadata length mismatch in {shard_path}")

    target_indices = [i for i, sid in enumerate(sample_ids) if is_matsynth(sample_sources[i], str(sid))]
    if dry_run:
        return {"shard": str(shard_path), "samples": n, "matsynth_rows": len(target_indices), "dry_run": True}
    if not target_indices:
        return {"shard": str(shard_path), "samples": n, "rebuilt": 0, "preserved": n, "source_files": 0, "version": VERSION}

    groups: dict[str, list[int]] = {}
    for i in target_indices:
        groups.setdefault(source_path_for(sample_sources[i], str(sample_ids[i])), []).append(i)

    out_latents = payload["latents"].clone()
    pending_images: list[torch.Tensor] = []
    pending_positions: list[int] = []
    rebuilt = 0
    missing_frames: list[dict[str, Any]] = []

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

    for path, indices in tqdm(groups.items(), desc=f"matsynth rgb {shard_path.name}", dynamic_ncols=True):
        with np.load(path, allow_pickle=False) as npz:
            for i in indices:
                frame_key = frame_key_for(sample_sources[i], str(frame_keys[i]))
                try:
                    frame = load_frame_from_open_npz(npz, path, sample_sources[i], str(sample_ids[i]), frame_key)
                except KeyError:
                    missing_frames.append({"row": int(i), "sample_id": str(sample_ids[i]), "source": path, "frame_key": frame_key})
                    continue
                pending_images.append(frame_to_rgb_image(frame, crop_infos[i]))
                pending_positions.append(i)
                if len(pending_images) >= batch_size:
                    flush()
    flush()

    new_payload = dict(payload)
    new_payload["latents"] = out_latents
    patch_payload_metadata(new_payload, rebuilt, n - rebuilt)
    if missing_frames:
        new_payload["matsynth_rgb_missing_frame_keys"] = missing_frames
    tmp = shard_path.with_name(shard_path.name + ".matsynth_rgb.tmp")
    torch.save(new_payload, tmp)
    os.replace(tmp, shard_path)
    return {"shard": str(shard_path), "samples": n, "rebuilt": rebuilt, "preserved": n - rebuilt, "missing_frame_keys": len(missing_frames), "source_files": len(groups), "version": VERSION}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=Path("data/wsg_flux2_frame_latents_dense_multidataset"))
    ap.add_argument("--model-path", type=Path, default=Path("models/black-forest-labs/FLUX.2-klein-4B"))
    ap.add_argument("--shard-ids", default="", help="Comma/range shard ids. Empty means all shards.")
    ap.add_argument("--shard-glob", default="shard_*.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--cache-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cache_root = resolve_path(args.cache_root)
    shard_dir = cache_root / "shards" / MODALITY_DIR
    shard_paths = parse_shard_ids(args.shard_ids, sorted(shard_dir.glob(args.shard_glob)))
    if not shard_paths:
        raise SystemExit(f"No shards selected under {shard_dir}")
    print(json.dumps({"cache_root": str(cache_root), "model_path": str(resolve_path(args.model_path)), "selected_shards": [p.name for p in shard_paths], "device": args.device, "dtype": args.dtype, "cache_dtype": args.cache_dtype, "batch_size": args.batch_size, "version": VERSION, "dry_run": args.dry_run}, indent=2), flush=True)
    vae = None if args.dry_run else load_vae(resolve_path(args.model_path), args.device, dtype_from_name(args.dtype))
    results = []
    for shard_path in shard_paths:
        result = rebuild_shard(shard_path, vae, args.device, dtype_from_name(args.dtype), dtype_from_name(args.cache_dtype), args.batch_size, args.dry_run)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"done": True, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
