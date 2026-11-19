#!/usr/bin/env python3
"""Visual spot-check for the AVIS reindex repair.

For each sampled AVIS WSV, this writes two 4-row contact sheets:
  1. current WSV natural frame order;
  2. decoded FLUX latent cache rows addressed by the repaired natural frame ids.

Rows are RGB (0-2), normal (36-38), edge (40), and semantic/audio mask (195).
This is meant as a repair-audit helper, not as a training-time visualizer.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image, ImageDraw

WSM_ROOT = Path(__file__).resolve().parents[3]
WSG_MAIN = Path("/research/cbim/vast/sf895/code/wsg/main")
if str(WSM_ROOT) not in sys.path:
    sys.path.insert(0, str(WSM_ROOT))
if str(WSG_MAIN) not in sys.path:
    sys.path.insert(0, str(WSG_MAIN))

import pipeline30b_fix_avis_semantic_lexicographic_order as p30b  # noqa: E402
from src.dataset.wsg_dataset import WSGNormalizationStore, load_wsg_normalization_ranges  # noqa: E402

MODALITIES = {
    "rgb": ("0_2", "0-2", [0, 1, 2]),
    "normal": ("36_38", "36-38", [36, 37, 38]),
    "edge": ("40", "40", [40]),
    "semantic": ("195", "195", [195]),
}
SAMPLE_RE = re.compile(r"(?P<path>.*?/AVIS_(?:train|val|test)_results/(?P<video>.*?)/wsv/merge\.wsv)::index_(?P<a>\d+)-index_(?P<b>\d+)$")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return WSM_ROOT / path


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


def tile(arr: np.ndarray, label: str, width: int) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    im = Image.fromarray(arr, "RGB")
    h = max(1, int(round(width * im.height / max(im.width, 1))))
    im = im.resize((width, h), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.width, 20), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    return im


def row(frames: list[np.ndarray], name: str, width: int) -> Image.Image:
    tiles = [tile(frame, f"{name} {idx + 1:03d}", width) for idx, frame in enumerate(frames)]
    h = max((t.height for t in tiles), default=width)
    out = Image.new("RGB", (width * len(tiles), h), (0, 0, 0))
    for idx, im in enumerate(tiles):
        out.paste(im, (idx * width, 0))
    return out


def sheet(rows: list[Image.Image], out: Path) -> None:
    pad = 8
    image = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + pad * (len(rows) - 1)), (20, 20, 20))
    y = 0
    for r in rows:
        image.paste(r, (0, y))
        y += r.height + pad
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(out)


def render_wsv(wsv: Path, out: Path, tile_width: int, max_frames: int | None) -> list[int]:
    with np.load(wsv, allow_pickle=True) as npz:
        keys = [k for k in npz.files if p30b.frame_index_from_key(k) is not None]
        keys = sorted(keys, key=lambda k: p30b.frame_index_from_key(k) or 0)
        if max_frames is not None:
            keys = keys[:max_frames]
        frames = [np.asarray(npz[k]) for k in keys]
    rows = [
        row([p30b.rgb_from_frame(f) for f in frames], "wsv_rgb", tile_width),
        row([p30b.normal_from_frame(f) for f in frames], "wsv_normal", tile_width),
        row([p30b.edge_from_frame(f) for f in frames], "wsv_edge40", tile_width),
        row([p30b.semantic_from_frame(f) for f in frames], "wsv_sem195", tile_width),
    ]
    sheet(rows, out)
    # WSV keys are stored as index_00000001, but p30b.frame_index_from_key()
    # returns zero-based integers. Cache sample_ids are one-based, so return
    # one-based natural frame ids for cache lookup and manifest readability.
    return [(p30b.frame_index_from_key(k) or 0) + 1 for k in keys]


def sample_wsvs(roots: list[Path], count: int, seed: int) -> list[Path]:
    all_wsvs: list[Path] = []
    for root in roots:
        all_wsvs.extend(sorted(root.glob("*/wsv/merge.wsv")))
    rng = random.Random(seed)
    rng.shuffle(all_wsvs)
    selected = []
    for wsv in all_wsvs:
        try:
            with np.load(wsv, allow_pickle=True) as npz:
                frame_count = sum(1 for k in npz.files if p30b.frame_index_from_key(k) is not None)
            if frame_count >= 8:
                selected.append(wsv)
        except Exception:
            continue
        if len(selected) >= count:
            break
    return selected


def collect_cache_rows(cache_root: Path, wsv: Path, frame_indices: list[int]) -> dict[str, dict[int, torch.Tensor]]:
    wanted_path = str(wsv.resolve())
    wanted = set(frame_indices)
    found: dict[str, dict[int, torch.Tensor]] = {name: {} for name in MODALITIES}
    for name, (shard_dir_name, _modality, _channels) in MODALITIES.items():
        shard_dir = cache_root / "shards" / shard_dir_name
        for shard in sorted(shard_dir.glob("shard_*.pt")):
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            sample_ids = payload.get("sample_ids") or []
            latents = payload.get("latents")
            if latents is None:
                continue
            for i, sid in enumerate(sample_ids):
                m = SAMPLE_RE.fullmatch(str(sid))
                if not m:
                    continue
                if str(Path(m.group("path")).resolve()) != wanted_path:
                    continue
                idx = int(m.group("a"))
                if idx in wanted and idx not in found[name]:
                    found[name][idx] = latents[i].detach().cpu()
            if wanted.issubset(found[name].keys()):
                break
    return found


def make_normalizer(payload: dict[str, Any]) -> WSGNormalizationStore:
    modality_config = resolve_project_path(payload.get("modality_config_path") or "")
    ranges = load_wsg_normalization_ranges(modality_config) if modality_config.exists() else {}
    return WSGNormalizationStore(
        normalization_mode=str(payload.get("normalization_mode") or "log"),
        portable_channel_ranges=ranges,
        show_progress=False,
    )


def first_payload(cache_root: Path, shard_dir_name: str) -> dict[str, Any]:
    shard = next((cache_root / "shards" / shard_dir_name).glob("shard_*.pt"))
    return torch.load(shard, map_location="cpu", weights_only=False)


def semantic_palette(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64, copy=False)
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels >= 0
    vals = labels[valid]
    out[..., 0][valid] = (vals * 37 + 23) % 255
    out[..., 1][valid] = (vals * 67 + 91) % 255
    out[..., 2][valid] = (vals * 97 + 131) % 255
    return out


def visualize_decoded(decoded: torch.Tensor, normalizer: WSGNormalizationStore, channels: list[int], dataset_paths: list[str]) -> np.ndarray:
    image01 = ((decoded.detach().float().cpu() + 1.0) / 2.0).clamp(0, 1).numpy()
    if channels == [0, 1, 2]:
        return np.transpose(image01[:3], (1, 2, 0)) * 255.0
    if channels == [36, 37, 38]:
        vals = []
        for i, ch in enumerate(channels):
            vals.append(normalizer.denormalize_channel(image01[i], ch, dataset_paths))
        arr = np.stack(vals, axis=-1)
        lo, hi = np.nanpercentile(arr, [1, 99])
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        return arr
    ch = channels[0]
    vals = normalizer.denormalize_channel(image01[0], ch, dataset_paths)
    if ch == 195:
        return semantic_palette(np.rint(vals))
    lo, hi = np.nanpercentile(vals[np.isfinite(vals)], [1, 99]) if np.isfinite(vals).any() else (0, 1)
    if hi <= lo:
        hi = lo + 1
    gray = np.clip((vals - lo) / (hi - lo) * 255.0, 0, 255)
    return np.repeat(gray[..., None], 3, axis=-1)


def render_cache(cache_root: Path, wsv: Path, frame_indices: list[int], out: Path, tile_width: int, device: str, dtype: torch.dtype) -> None:
    sample_payload = first_payload(cache_root, "0_2")
    vae = load_vae(sample_payload["vae_path"], device, dtype)
    rows = []
    cache_rows = collect_cache_rows(cache_root, wsv, frame_indices)
    for name, (shard_dir_name, _modality, channels) in MODALITIES.items():
        payload = first_payload(cache_root, shard_dir_name)
        normalizer = make_normalizer(payload)
        dataset_paths = list(payload.get("normalization_dataset_paths") or payload.get("dataset_paths") or [])
        frames = []
        for idx in frame_indices:
            latent = cache_rows[name].get(idx)
            if latent is None:
                frames.append(np.zeros((480, 480, 3), dtype=np.uint8))
                continue
            if latent.ndim == 4:
                latent = latent[:, 0]
            with torch.no_grad():
                decoded = vae.decode(latent.unsqueeze(0).to(device=device, dtype=dtype)).sample[0]
            frames.append(visualize_decoded(decoded, normalizer, channels, dataset_paths))
        rows.append(row(frames, f"cache_{name}", tile_width))
    sheet(rows, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("data/wsg_flux2_frame_latents_dense_multidataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("train_log_sparse/debug_semantic_alignment/reindex_archive_spotcheck_20260713"))
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--wsv", type=Path, action="append", default=None, help="Explicit WSV path to visualize; can be repeated.")
    ap.add_argument("--seed", type=int, default=713)
    ap.add_argument("--tile-width", type=int, default=96)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    roots = [
        Path("/common/users/sf895/wave2pixel/AVIS_train_results"),
        Path("/common/users/sf895/wave2pixel/AVIS_val_results"),
        Path("/common/users/sf895/wave2pixel/AVIS_test_results"),
    ]
    wsvs = [Path(p) for p in args.wsv] if args.wsv else sample_wsvs(roots, args.count, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for n, wsv in enumerate(wsvs):
        prefix = f"sample_{n:02d}_{wsv.parent.parent.name}"
        wsv_out = args.out_dir / f"{prefix}_wsv_current_order.png"
        cache_out = args.out_dir / f"{prefix}_cache_decoded_rekeyed.png"
        frame_indices = render_wsv(wsv, wsv_out, args.tile_width, args.max_frames)
        render_cache(resolve_project_path(args.cache_root), wsv, frame_indices, cache_out, args.tile_width, args.device, dtype)
        manifest.append({"wsv": str(wsv), "frames": frame_indices, "wsv_image": str(wsv_out), "cache_image": str(cache_out)})
    (args.out_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2) + "\n")
    print(args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
