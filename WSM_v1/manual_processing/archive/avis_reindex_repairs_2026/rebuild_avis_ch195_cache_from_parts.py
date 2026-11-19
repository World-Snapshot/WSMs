#!/usr/bin/env python3
"""Rebuild AVIS channel-195 FLUX VAE shards from one-pass repair parts.

pipeline30c already did the expensive WSV open -> inject -> reindex -> extract
pass. This script intentionally never opens WSV files again. It reads the
extracted raw channel-195 planes, VAE-encodes them, and writes them back into
the original channel-195 shard positions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    load_wsg_normalization_ranges,
)

CHANNEL = 195


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(name)]


def load_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
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


def make_normalizer(payload: dict[str, Any]) -> WSGNormalizationStore:
    cfg = load_config_from_payload(payload)
    params = ((cfg.get("dataset") or {}).get("params") or {}) if isinstance(cfg, dict) else {}
    normalization_cache = params.get("normalization_cache_path")
    modality_config_path = resolve_project_path(
        payload.get("modality_config_path") or params.get("modality_config_path") or ""
    )
    portable_ranges = load_wsg_normalization_ranges(modality_config_path) if modality_config_path.exists() else {}
    return WSGNormalizationStore(
        resolve_project_path(normalization_cache) if normalization_cache else None,
        normalization_mode=str(payload.get("normalization_mode") or "log"),
        show_progress=False,
        portable_channel_ranges=portable_ranges,
    )


def scan_parts(parts_dir: Path) -> dict[str, dict[str, int]]:
    """Return per-shard coverage without keeping raw planes in memory."""
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"records": 0, "parts": 0})
    for part_path in tqdm(sorted(parts_dir.glob("*.pt")), desc="scan parts", dynamic_ncols=True):
        payload = torch.load(part_path, map_location="cpu", weights_only=False)
        seen = set()
        for rec in payload.get("records") or []:
            shard = str(rec.get("shard_name") or "")
            if not shard:
                continue
            coverage[shard]["records"] += 1
            seen.add(shard)
        for shard in seen:
            coverage[shard]["parts"] += 1
    return dict(coverage)


def normalized_images_from_planes(
    planes: torch.Tensor,
    normalizer: WSGNormalizationStore,
    dataset_paths: list[str],
) -> torch.Tensor:
    # planes: [B,H,W] int/float raw channel values after crop.
    arr = planes.cpu().numpy().astype(np.float32, copy=False)
    normalized = []
    for plane in arr:
        norm = normalizer.normalize_channel(plane, CHANNEL, dataset_paths)
        stacked = np.stack([norm, norm, norm], axis=0).astype(np.float32, copy=False)
        valid = stacked != MASK_VALUE
        stacked[valid] = stacked[valid] * 2.0 - 1.0
        normalized.append(stacked)
    return torch.from_numpy(np.stack(normalized, axis=0))


@torch.no_grad()
def encode_planes(
    vae,
    planes: torch.Tensor,
    normalizer: WSGNormalizationStore,
    dataset_paths: list[str],
    device: str,
    torch_dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    images = normalized_images_from_planes(planes, normalizer, dataset_paths).to(device=device, dtype=torch_dtype)
    latents = vae.encode(images).latent_dist.sample()
    return latents.detach().cpu().to(dtype=cache_dtype).unsqueeze(2)


def rebuild_one_shard(
    shard_path: Path,
    parts_dir: Path,
    vae,
    device: str,
    torch_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    batch_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if str(payload.get("modality")) != "195":
        raise ValueError(f"{shard_path} is not a channel-195 shard")

    sample_count = len(payload.get("sample_ids") or [])
    normalizer = make_normalizer(payload)
    dataset_paths = list(payload.get("normalization_dataset_paths") or payload.get("dataset_paths") or [])

    replaced = 0
    seen_positions: set[int] = set()
    batches_planes: list[torch.Tensor] = []
    batches_positions: list[int] = []

    def flush() -> None:
        nonlocal replaced, batches_planes, batches_positions
        if not batches_planes:
            return
        planes = torch.stack(batches_planes, dim=0)
        if not dry_run:
            encoded = encode_planes(vae, planes, normalizer, dataset_paths, device, torch_dtype, cache_dtype)
            for out_idx, pos in enumerate(batches_positions):
                payload["latents"][pos].copy_(encoded[out_idx])
        replaced += len(batches_positions)
        batches_planes = []
        batches_positions = []

    for part_path in tqdm(sorted(parts_dir.glob("*.pt")), desc=f"parts -> {shard_path.name}", dynamic_ncols=True):
        part = torch.load(part_path, map_location="cpu", weights_only=False)
        raw = part.get("raw_ch195")
        records = part.get("records") or []
        if raw is None or len(records) != int(raw.shape[0]):
            raise RuntimeError(f"Bad repair part: {part_path}")
        for idx, rec in enumerate(records):
            if rec.get("shard_name") != shard_path.name:
                continue
            pos = int(rec["position"])
            if pos < 0 or pos >= sample_count:
                raise RuntimeError(f"{part_path}: position {pos} outside {shard_path.name} sample_count={sample_count}")
            if pos in seen_positions:
                raise RuntimeError(f"{part_path}: duplicate position {pos} for {shard_path.name}")
            seen_positions.add(pos)
            batches_positions.append(pos)
            batches_planes.append(raw[idx].to(torch.int16))
            if len(batches_planes) >= batch_size:
                flush()
    flush()

    if replaced != sample_count:
        raise RuntimeError(f"{shard_path.name}: replaced {replaced}/{sample_count}; refusing partial shard write")
    if dry_run:
        return {"path": str(shard_path), "samples": sample_count, "replaced": replaced, "dry_run": True}

    tmp = shard_path.with_name(shard_path.name + ".rebuild195parts.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, shard_path)
    return {"path": str(shard_path), "samples": sample_count, "replaced": replaced, "dry_run": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=Path("data/wsg_flux2_frame_latents_dense_multidataset"))
    ap.add_argument("--parts-dir", type=Path, default=Path("data/avis_ch195_repair_parts"))
    ap.add_argument("--flux-model-path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--cache-dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-glob", default="shard_*.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache_root = resolve_project_path(args.cache_root)
    parts_dir = resolve_project_path(args.parts_dir)
    shard_dir = cache_root / "shards" / "195"
    shard_paths = sorted(shard_dir.glob(args.shard_glob))
    if not shard_paths:
        raise SystemExit(f"No shards matched {shard_dir / args.shard_glob}")
    first_payload = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    flux_model_path = args.flux_model_path or first_payload.get("vae_path")
    if not flux_model_path:
        raise SystemExit("Could not infer FLUX VAE path; pass --flux-model-path")

    coverage = scan_parts(parts_dir)
    expected = {path.name: len(torch.load(path, map_location="cpu", weights_only=False).get("sample_ids") or []) for path in shard_paths}
    print(json.dumps({
        "cache_root": str(cache_root),
        "parts_dir": str(parts_dir),
        "shards": len(shard_paths),
        "device": args.device,
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype or args.dtype,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
        "coverage": coverage,
        "expected": expected,
    }, indent=2, sort_keys=True), flush=True)
    for shard_name, count in expected.items():
        got = int((coverage.get(shard_name) or {}).get("records", 0))
        if got != count:
            raise SystemExit(f"Coverage mismatch for {shard_name}: parts={got}, shard_samples={count}")
    if args.dry_run:
        return

    torch_dtype = dtype_from_name(args.dtype)
    cache_dtype = dtype_from_name(args.cache_dtype or args.dtype)
    vae = load_vae(str(flux_model_path), args.device, torch_dtype)
    totals = {"samples": 0, "replaced": 0}
    for shard_path in shard_paths:
        result = rebuild_one_shard(
            shard_path,
            parts_dir,
            vae,
            args.device,
            torch_dtype,
            cache_dtype,
            args.batch_size,
            False,
        )
        totals["samples"] += int(result["samples"])
        totals["replaced"] += int(result["replaced"])
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps(totals, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
