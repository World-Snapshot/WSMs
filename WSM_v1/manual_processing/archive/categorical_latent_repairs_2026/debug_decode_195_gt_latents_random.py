#!/usr/bin/env python3
"""Decode random channel-195 GT latents and compare with original WSV masks."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusers import AutoencoderKLFlux2  # noqa: E402
from src.dataset.wsg_dataset import (  # noqa: E402
    WSGNormalizationStore,
    load_wsg_normalization_ranges,
)


COLORS = {
    1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
    5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 128, 0), 8: (128, 255, 0),
    9: (128, 0, 255), 10: (255, 0, 128), 11: (255, 192, 128), 12: (128, 128, 255),
    13: (255, 255, 128), 14: (128, 255, 128), 15: (128, 0, 0), 16: (170, 255, 195),
    17: (128, 128, 0), 18: (255, 215, 180), 19: (0, 0, 128), 20: (128, 128, 128),
    21: (255, 255, 255), 22: (20, 120, 20), 23: (120, 20, 20), 24: (20, 20, 120),
    25: (120, 80, 20), 26: (20, 120, 120),
}


def render_semantic(values: np.ndarray) -> Image.Image:
    semantic = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(semantic) & (semantic > 0.5) & (semantic != -1.0)
    labels = np.rint(semantic).astype(np.int32, copy=False)
    rgb = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for label, color in COLORS.items():
        rgb[valid & (labels == label)] = color
    unknown = valid & ~np.isin(labels, list(COLORS))
    rgb[unknown] = (255, 128, 0)
    return Image.fromarray(rgb, mode="RGB")


def crop_frame(frame: np.ndarray, crop_info: dict[str, Any]) -> np.ndarray:
    patch_hw = int(crop_info["patch_hw"])
    crop_y = int(crop_info["crop_y"])
    crop_x = int(crop_info["crop_x"])
    h, w = frame.shape[:2]
    if h < patch_hw or w < patch_hw:
        pad_width = ((0, max(0, patch_hw - h)), (0, max(0, patch_hw - w)), (0, 0))
        frame = np.pad(frame, pad_width, mode="constant", constant_values=-1.0)
    return frame[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :]


def original_mask(path: str, frame_key: str, crop_info: dict[str, Any]) -> tuple[Image.Image, float, list[int]]:
    with np.load(path, allow_pickle=False) as data:
        frame = data[frame_key].astype(np.float32, copy=False)
    frame = crop_frame(frame, crop_info)
    values = frame[..., 195]
    valid = np.isfinite(values) & (values > 0.5) & (values != -1.0)
    labels = sorted(int(v) for v in np.unique(np.rint(values[valid]).astype(np.int32))) if valid.any() else []
    return render_semantic(values), float(valid.mean()), labels


def decoded_to_images(
    decoded: torch.Tensor,
    normalizer: WSGNormalizationStore,
) -> tuple[Image.Image, Image.Image, float, list[int]]:
    decoded = decoded.detach().float().cpu()
    raw_rgb = ((decoded[:3].clamp(-1, 1) + 1.0) / 2.0 * 255.0).byte().permute(1, 2, 0).numpy()
    image01 = ((decoded + 1.0) / 2.0).clamp(0, 1).numpy()
    values = normalizer.denormalize_channel(image01[0], 195, [])
    valid = np.isfinite(values) & (values > 0.5) & (values != -1.0)
    labels = sorted(int(v) for v in np.unique(np.rint(values[valid]).astype(np.int32))) if valid.any() else []
    return Image.fromarray(raw_rgb, mode="RGB"), render_semantic(values), float(valid.mean()), labels


def add_label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 24), (255, 255, 255))
    out.paste(img.convert("RGB"), (0, 24))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), text[:90], fill=(0, 0, 0))
    return out


def make_sheet(rows: list[list[Image.Image]], out_path: Path) -> None:
    cell_w, cell_h = 240, 264
    sheet = Image.new("RGB", (cell_w * 3, cell_h * len(rows)), (255, 255, 255))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            img = img.resize((cell_w, cell_h), Image.Resampling.NEAREST)
            sheet.paste(img, (c * cell_w, r * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    random.seed(7)
    cache_root = ROOT / "data/wsg_flux2_frame_latents_dense_multidataset/shards/195"
    out_dir = ROOT / "train_log_sparse/debug_gt_latent_decode_195_random"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(cache_root.glob("shard_*.pt"))

    selected: list[tuple[Path, int, dict[str, Any], Image.Image, float, list[int]]] = []
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        indices = list(range(len(payload["sample_ids"])))
        random.shuffle(indices)
        for idx in indices:
            src = payload["sample_sources"][idx]
            path = src["path"]
            if "AVIS" not in path:
                continue
            frame_key = payload["frame_keys"][idx]
            crop_info = payload["crop_infos"][idx]
            try:
                orig_img, orig_ratio, orig_labels = original_mask(path, frame_key, crop_info)
            except Exception:
                continue
            if orig_ratio < 0.001:
                continue
            selected.append((shard_path, idx, payload, orig_img, orig_ratio, orig_labels))
            if len(selected) >= 12:
                break
        if len(selected) >= 12:
            break

    if not selected:
        raise RuntimeError("No foreground AVIS channel-195 samples found")

    first_payload = selected[0][2]
    modality_config = ROOT / first_payload["modality_config_path"]
    ranges = load_wsg_normalization_ranges(modality_config)
    normalizer = WSGNormalizationStore(normalization_mode="log", portable_channel_ranges=ranges)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    vae_path = ROOT / "models/black-forest-labs/FLUX.2-klein-4B"
    vae = AutoencoderKLFlux2.from_pretrained(vae_path, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()

    rows: list[list[Image.Image]] = []
    stats = []
    with torch.no_grad():
        for shard_path, idx, payload, orig_img, orig_ratio, orig_labels in selected:
            latent = payload["latents"][idx]
            if latent.ndim == 4:
                latent = latent[:, 0]
            decoded = vae.decode(latent.unsqueeze(0).to(device=device, dtype=dtype)).sample[0]
            raw_img, decoded_mask, decoded_ratio, decoded_labels = decoded_to_images(decoded, normalizer)
            sample_id = payload["sample_ids"][idx]
            label = f"{Path(shard_path).name}:{idx} {payload['video_ids'][idx]} {payload['frame_keys'][idx]}"
            rows.append([
                add_label(raw_img, f"latent raw RGB | dec fg={decoded_ratio:.3f}"),
                add_label(decoded_mask, f"latent->195 | labels={decoded_labels[:8]}"),
                add_label(orig_img, f"original WSV | fg={orig_ratio:.3f} labels={orig_labels[:8]}"),
            ])
            stats.append({
                "sample_id": sample_id,
                "shard": str(shard_path),
                "index": int(idx),
                "video_id": payload["video_ids"][idx],
                "frame_key": payload["frame_keys"][idx],
                "decoded_nonblack_ratio": decoded_ratio,
                "decoded_labels": decoded_labels,
                "original_nonblack_ratio": orig_ratio,
                "original_labels": orig_labels,
            })

    make_sheet(rows, out_dir / "gt_latent_random_decode_contact.png")
    with (out_dir / "gt_latent_random_decode_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(out_dir / "gt_latent_random_decode_contact.png")
    print(out_dir / "gt_latent_random_decode_stats.json")


if __name__ == "__main__":
    main()
