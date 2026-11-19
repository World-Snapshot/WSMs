#!/usr/bin/env python3
"""Render aligned RGB/GT examples for categorical channels 111 and 195."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manual_processing.archive.categorical_latent_repairs_2026.reencode_channel_111_latents import Channel111Dataset
from manual_processing.archive.categorical_latent_repairs_2026.reencode_channel_195_latents import (
    frame_from_open_wsv,
    frame_key,
    source_path,
)
from src.dataset.wsg_dataset import apply_resize_crop_info
from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    CHANNEL_111_ID_TO_CATEGORY,
    categorical_ids_to_rgb,
    channel195_semantic_map,
    dense_channel111_category_map,
)

CACHE_ROOT = Path(
    "/common/users/sf895/WSMs/WSM_v1/data/"
    "wsg_flux2_frame_latents_dense_multidataset/shards"
)


def torch_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def rgb_image(frame: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(np.asarray(frame[..., :3], dtype=np.float32), nan=0.0)
    if rgb.size and float(rgb.max()) <= 1.5:
        if float(rgb.min()) < 0.0:
            rgb = (rgb + 1.0) * 127.5
        else:
            rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def load_111_examples() -> list[tuple[np.ndarray, np.ndarray, str]]:
    records: dict[str, tuple[dict, dict]] = {}
    for shard_path in sorted((CACHE_ROOT / "111").glob("*.pt")):
        payload = torch_load(shard_path)
        for source, crop in zip(payload["sample_sources"], payload["crop_infos"]):
            dataset = str(source.get("dataset_name", ""))
            if "ShapeNetSem" in dataset and "ShapeNetSem" not in records:
                records["ShapeNetSem"] = (source, crop)
            elif "Articulation" in dataset and "Articulation" not in records:
                records["Articulation"] = (source, crop)
            if len(records) == 2:
                break
        del payload
        if len(records) == 2:
            break

    examples = []
    for dataset_name in ("ShapeNetSem", "Articulation"):
        source, crop = records[dataset_name]
        path = str(source["path"])
        with np.load(path) as payload:
            raw = np.asarray(payload["data"] if "data" in payload else payload["wsg_data"])
        cropped = apply_resize_crop_info(raw, crop)
        resolver = Channel111Dataset([source], [crop], (-1.0, 417.0), "linear")
        category_id = resolver._category_id(cropped[..., 111], path)
        labels = dense_channel111_category_map(
            cropped,
            require_alpha=False,
            category_id=category_id,
            allow_category_mask_fallback=True,
        )
        gt = categorical_ids_to_rgb(labels)
        foreground = int(np.count_nonzero(labels != -1))
        title = (
            f"111 GT | {dataset_name} | global ID {category_id}: "
            f"{CHANNEL_111_ID_TO_CATEGORY[category_id]} | foreground {foreground:,} px"
        )
        examples.append((rgb_image(cropped), gt, title))
    return examples


def dominant_valid(labels: np.ndarray) -> int:
    valid = np.rint(labels[labels != -1]).astype(np.int32)
    if valid.size == 0:
        return -1
    values, counts = np.unique(valid, return_counts=True)
    return int(values[np.argmax(counts)])


def load_195_examples() -> list[tuple[np.ndarray, np.ndarray, str]]:
    candidates: list[tuple[dict, str, dict, str]] = []
    for shard_path in sorted((CACHE_ROOT / "195").glob("*.pt")):
        payload = torch_load(shard_path)
        for source, fallback, crop, sample_id in zip(
            payload["sample_sources"],
            payload["frame_keys"],
            payload["crop_infos"],
            payload["sample_ids"],
        ):
            candidates.append((source, str(fallback), crop, str(sample_id)))
            if len(candidates) >= 40:
                break
        del payload
        if len(candidates) >= 40:
            break

    examples = []
    seen_classes = set()
    for source, fallback, crop, sample_id in candidates:
        path = source_path(source, sample_id)
        key = frame_key(source, fallback)
        with np.load(path) as payload:
            frame = frame_from_open_wsv(payload, path, key)
        cropped = apply_resize_crop_info(frame, crop)
        labels = channel195_semantic_map(cropped)
        class_id = dominant_valid(labels)
        if class_id < 0 or class_id in seen_classes:
            continue
        seen_classes.add(class_id)
        gt = categorical_ids_to_rgb(labels, AVIS_CATEGORY_COLORS)
        ids = sorted(int(value) for value in np.unique(labels) if value != -1)
        foreground = int(np.count_nonzero(labels != -1))
        title = (
            f"195 GT | AVIS | frame {key} | IDs {ids} "
            f"| foreground {foreground:,} px"
        )
        examples.append((rgb_image(cropped), gt, title))
        if len(examples) == 2:
            break
    if len(examples) != 2:
        raise RuntimeError("Could not find two channel-195 samples with different classes")
    return examples


def render(examples: list[tuple[np.ndarray, np.ndarray, str]], output: Path) -> None:
    image_size = 480
    margin = 20
    title_height = 48
    row_height = title_height + image_size + margin
    canvas = Image.new("RGB", (margin * 3 + image_size * 2, margin + row_height * 4), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    for row, (rgb, gt, title) in enumerate(examples):
        y = margin + row * row_height
        draw.text((margin, y), title, fill="#111827", font=font)
        image_y = y + title_height
        canvas.paste(Image.fromarray(rgb).resize((image_size, image_size)), (margin, image_y))
        canvas.paste(
            Image.fromarray(gt).resize((image_size, image_size), Image.Resampling.NEAREST),
            (margin * 2 + image_size, image_y),
        )
        draw.rectangle((margin, image_y, margin + 66, image_y + 23), fill="white")
        draw.text((margin + 6, image_y + 3), "RGB", fill="#111827", font=small)
        gt_x = margin * 2 + image_size
        draw.rectangle((gt_x, image_y, gt_x + 56, image_y + 23), fill="white")
        draw.text((gt_x + 6, image_y + 3), "GT", fill="#111827", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "asset/categorical_gt_preview_111_195.png",
    )
    args = parser.parse_args()
    examples = load_111_examples() + load_195_examples()
    render(examples, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
