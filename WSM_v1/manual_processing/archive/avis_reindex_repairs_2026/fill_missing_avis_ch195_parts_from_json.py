#!/usr/bin/env python3
"""Fill missing channel-195 repair parts directly from AVIS JSON.

This is a narrow repair for samples whose WSV was already damaged by an earlier
bad reindex run. It does not read WSV files. It renders channel 195 from AVIS
annotations, applies the original latent-cache crop, and writes an extra part
compatible with rebuild_avis_ch195_cache_from_parts.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WSG_MAIN = Path("/research/cbim/vast/sf895/code/wsg/main")
if str(WSG_MAIN) not in sys.path:
    sys.path.insert(0, str(WSG_MAIN))

import pipeline30b_fix_avis_semantic_lexicographic_order as p30b  # noqa: E402
from src.dataset.wsg_dataset import apply_resize_crop_info  # noqa: E402

CHANNEL = 195
SAMPLE_RE = re.compile(
    r"(?P<prefix>.*?/AVIS_(?P<split>train|val|test)_results/(?P<video>.*?)/wsv/merge\.wsv)::index_(?P<a>\d+)-index_(?P<b>\d+)$"
)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_existing_positions(parts_dir: Path) -> set[tuple[str, int]]:
    existing: set[tuple[str, int]] = set()
    for part_path in sorted(parts_dir.glob("*.pt")):
        payload = torch.load(part_path, map_location="cpu", weights_only=False)
        for rec in payload.get("records") or []:
            shard = rec.get("shard_name")
            if shard is not None:
                existing.add((str(shard), int(rec["position"])))
    return existing


def avis_json_for_split(base_output_dir: Path, split: str) -> Path:
    return base_output_dir / f"AVIS_{split}_results" / f"{split}.json"


def render_raw_plane(avis: dict[str, Any], video_name: str, old_index: int, crop_info: dict[str, Any]) -> np.ndarray:
    video = p30b.find_video_record(avis, video_name)
    file_names = video.get("file_names", [])
    crop_to_json = p30b.build_legacy_crop_to_json_index(file_names)
    old_zero = old_index - 1
    if old_zero < 0 or old_zero >= len(crop_to_json):
        raise ValueError(f"{video_name}: old_index {old_index} outside mapping length {len(crop_to_json)}")
    natural_zero = crop_to_json[old_zero]
    frames = p30b.build_frame_instances(avis, int(video["id"]))
    instances = frames[natural_zero] if natural_zero < len(frames) else []
    h = int(crop_info.get("original_height") or crop_info.get("padded_height") or 480)
    w = int(crop_info.get("original_width") or crop_info.get("padded_width") or 480)
    arr = np.full((h, w, CHANNEL + 1), p30b.EMPTY_VALUE, dtype=np.float32)
    p30b.p29.inject_frame_channel_195(arr, instances, verify_unchanged=False, clear_207_255=False)
    cropped = apply_resize_crop_info(arr, crop_info)
    return np.rint(cropped[..., CHANNEL]).astype(np.int16, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, default=Path("data/wsg_flux2_frame_latents_dense_multidataset"))
    ap.add_argument("--parts-dir", type=Path, default=Path("data/avis_ch195_repair_parts"))
    ap.add_argument("--base-output-dir", type=Path, default=Path("/common/users/sf895/wave2pixel"))
    ap.add_argument("--output-name", default="missing_avis_ch195_from_json.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache_root = resolve_project_path(args.cache_root)
    parts_dir = resolve_project_path(args.parts_dir)
    existing = load_existing_positions(parts_dir)
    avis_by_split: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    planes: list[np.ndarray] = []
    missing_report: list[dict[str, Any]] = []

    for shard_path in sorted((cache_root / "shards" / "195").glob("shard_*.pt")):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        sample_ids = payload.get("sample_ids") or []
        crop_infos = payload.get("crop_infos") or [{} for _ in sample_ids]
        for pos, sample_id in enumerate(sample_ids):
            key = (shard_path.name, int(pos))
            if key in existing:
                continue
            match = SAMPLE_RE.fullmatch(str(sample_id))
            if not match:
                raise RuntimeError(f"Missing non-AVIS sample in {shard_path.name}:{pos}: {sample_id}")
            split = match.group("split")
            video = match.group("video")
            old_index = int(match.group("a"))
            if split not in avis_by_split:
                with avis_json_for_split(args.base_output_dir, split).open("r", encoding="utf-8") as f:
                    avis_by_split[split] = json.load(f)
            plane = render_raw_plane(avis_by_split[split], video, old_index, crop_infos[pos])
            records.append({
                "sample_id": str(sample_id),
                "old_index": old_index,
                "shard_name": shard_path.name,
                "position": int(pos),
                "source": "avis_json_missing_repair",
            })
            planes.append(plane)
            missing_report.append({"shard": shard_path.name, "position": int(pos), "sample_id": str(sample_id)})

    print(json.dumps({"missing": len(records), "items": missing_report[:50]}, indent=2), flush=True)
    if args.dry_run or not records:
        return
    out = parts_dir / args.output_name
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save({
        "source": "avis_json_missing_repair",
        "records": records,
        "raw_ch195": torch.from_numpy(np.stack(planes, axis=0).astype(np.int16, copy=False)),
    }, tmp)
    tmp.replace(out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
