#!/usr/bin/env python
"""Audit dense Channel Head pair availability from cached PT metadata.

The modality JSON records whether a pair is conceptually trainable. This tool
checks whether the cached modalities actually intersect on the same sample.
Target-dataset filtering and runtime reference routing are reported separately;
neither changes the conceptual ``pair_exists`` result.
Latent tensors remain mmap-backed and are never materialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


CONTROL_TOKENS = {
    "caption",
    "Reference_IMG",
    "FirstFrame_RGB",
    "CurrentFrame_RGB_Reference",
    "CurrentFrame_RGB_Condition",
}
V17_CHANGED_MODALITIES = {"3", "90-92", "195"}
V17_CHANGED_DATASETS = {"MatSynth_results", "ShapeNetSem_results"}
MODALITY_DATASET_ALLOWLISTS = {
    "3": {"ShapeNetSem_results"},
    "36-38": {"AVIS_train_results", "AVIS_val_results", "AVIS_test_results"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wsg_v1_modalities_dense.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/wsg_flux2_frame_latents_dense_multidataset"),
    )
    parser.add_argument(
        "--xpose-cache-dir",
        type=Path,
        default=Path("data/xpose_flux2_frame_latents_sample10pct_2frames"),
    )
    parser.add_argument("--historical-reference-policy", choices=("strict", "tolerant"), default="tolerant")
    parser.add_argument("--output", type=Path, default=Path("dense_pair_audit.json"))
    parser.add_argument(
        "--head-indices-file",
        type=Path,
        default=None,
        help="Optional handoff/gap text file; audit only channel_head_index entries found in it.",
    )
    parser.add_argument(
        "--completed-head-end",
        type=int,
        default=180,
        help="Inclusive last index in the earlier training batch.",
    )
    return parser.parse_args()


def as_conditions(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("condition") or []
    return [str(item) for item in (raw if isinstance(raw, list) else [raw])]


def modality_names(entries: dict[str, dict[str, Any]], shard_root: Path) -> list[str]:
    known = set()
    for entry in entries.values():
        known.add(str(entry.get("target") or ""))
        known.update(as_conditions(entry))
    names = []
    for directory in sorted(path for path in shard_root.iterdir() if path.is_dir()):
        matches = sorted(item for item in known if item.replace("-", "_") == directory.name)
        names.append(matches[0] if matches else directory.name.replace("_", "-"))
    return names


def split_for(sample_id: str, dataset_name: str, val_ratio: float = 0.05) -> str:
    if dataset_name.startswith("AVIS_train"):
        return "train"
    if dataset_name.startswith("AVIS_val"):
        return "val"
    if dataset_name.startswith("AVIS_test"):
        return "test"
    key = f"{dataset_name}:{sample_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / float(2**64)
    if bucket < val_ratio:
        return "val"
    if bucket < val_ratio * 2.0:
        return "test"
    return "train"


def resolve_route(entry: dict[str, Any]) -> tuple[str | None, str, bool]:
    conditions = as_conditions(entry)
    visual = [item for item in conditions if item not in CONTROL_TOKENS]
    task_type = str(entry.get("task_type") or "")
    condition = visual[0] if visual else None
    route = "direct"
    requires_avis_first_frame = False

    if "CurrentFrame_RGB_Condition" in conditions:
        route = "current_rgb_condition"
    elif "CurrentFrame_RGB_Reference" in conditions:
        route = "current_rgb_reference"
    elif "Reference_IMG" in conditions:
        route = "avis_native_first_frame"
        requires_avis_first_frame = True
    elif task_type == "inverse_rendering" and "0-2" in visual and len(visual) > 1:
        condition = next((item for item in visual if item != "0-2"), None)
        route = "avis_first_frame_condition"
        requires_avis_first_frame = True
    return condition, route, requires_avis_first_frame


def load_availability(
    shard_root: Path, modalities: list[str], xpose_shard_root: Path | None = None
) -> tuple[dict[str, int], dict[str, int], dict[str, str], dict[str, str], dict[str, int | None]]:
    modality_bits = {name: 1 << index for index, name in enumerate(modalities)}
    sample_masks: dict[str, int] = {}
    sample_datasets: dict[str, str] = {}
    sample_videos: dict[str, str] = {}
    sample_frames: dict[str, int | None] = {}
    modality_counts: dict[str, int] = {}

    for modality in modalities:
        directories = [shard_root / modality.replace("-", "_")]
        if xpose_shard_root is not None and modality in {"0-2", "122-124"}:
            directories.append(xpose_shard_root / modality.replace("-", "_"))
        count = 0
        for directory in directories:
            for shard in sorted(directory.glob("shard_*_of_*.pt")):
                payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
                sample_ids = payload.get("sample_ids", [])
                sources = payload.get("sample_sources", []) or [{} for _ in sample_ids]
                videos = payload.get("video_ids", []) or [
                    source.get("video_id") or source.get("video_name") or "" for source in sources
                ]
                frames = payload.get("frame_indices", []) or [
                    source.get("selected_frame_index", source.get("frame_index")) for source in sources
                ]
                count += len(sample_ids)
                for sample_id, source, video, frame in zip(sample_ids, sources, videos, frames):
                    sample_masks[sample_id] = sample_masks.get(sample_id, 0) | modality_bits[modality]
                    sample_datasets.setdefault(sample_id, str(source.get("dataset_name") or source.get("dataset") or ""))
                    sample_videos.setdefault(sample_id, str(video))
                    sample_frames.setdefault(sample_id, None if frame is None else int(frame))
                del payload
        modality_counts[modality] = count
    return modality_bits, sample_masks, sample_datasets, sample_videos, sample_frames, modality_counts


def load_reference_frames(cache_dir: Path) -> dict[str, int]:
    path = cache_dir / "reference_frames" / "0_2_avis_first_frame.pt"
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {
        str(video): int(frame)
        for video, frame in zip(payload.get("video_ids", []), payload.get("frame_indices", []))
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    entries = config.get("wsg_v1", config)
    if args.head_indices_file is not None:
        text = args.head_indices_file.read_text(encoding="utf-8")
        selected_indices = {int(value) for value in re.findall(r"channel_head_index=(\d+)", text)}
        entries = {
            key: entry for key, entry in entries.items()
            if key.startswith("Channel_Head_") and int(key.rsplit("_", 1)[1]) in selected_indices
        }
    shard_root = args.cache_dir / "shards"
    modalities = modality_names(entries, shard_root)
    (
        modality_bits,
        sample_masks,
        sample_datasets,
        sample_videos,
        sample_frames,
        modality_counts,
    ) = load_availability(
        shard_root, modalities, args.xpose_cache_dir / "shards"
    )
    rgb_bit = modality_bits.get("0-2", 0)
    first_frames: dict[str, int] = {}
    for sample_id, mask in sample_masks.items():
        frame = sample_frames.get(sample_id)
        video = sample_videos.get(sample_id, "")
        if not (mask & rgb_bit) or frame is None or not video:
            continue
        group_key = f"{sample_datasets.get(sample_id, '')}\x1f{video}"
        first_frames[group_key] = min(first_frames.get(group_key, frame), frame)
    whitelist_map = config.get("target_dataset_whitelist_by_modality") or {}

    rows = []
    for key, entry in entries.items():
        if not key.startswith("Channel_Head_"):
            continue
        index = int(key.rsplit("_", 1)[1])
        target = str(entry.get("target") or "")
        conditions = as_conditions(entry)
        condition, route, requires_first_frame = resolve_route(entry)
        required = modality_bits.get(target, 0)
        if condition:
            required |= modality_bits.get(condition, 0)
        allowed_datasets = set(whitelist_map.get(target, []))
        modalities_known = target in modality_bits and (condition is None or condition in modality_bits)

        required_modalities = [item for item in (target, condition) if item]
        base_ids = []
        modality_dataset_eligible_ids = []
        whitelist_eligible_ids = []
        if modalities_known:
            for sample_id, mask in sample_masks.items():
                if mask & required != required:
                    continue
                base_ids.append(sample_id)
                dataset_name = sample_datasets.get(sample_id, "")
                if any(
                    modality in MODALITY_DATASET_ALLOWLISTS
                    and dataset_name not in MODALITY_DATASET_ALLOWLISTS[modality]
                    for modality in required_modalities
                ):
                    continue
                modality_dataset_eligible_ids.append(sample_id)
                if not allowed_datasets or dataset_name in allowed_datasets:
                    whitelist_eligible_ids.append(sample_id)

        historical_ids = []
        fallback_candidates = []
        if requires_first_frame:
            for sample_id in whitelist_eligible_ids:
                video = sample_videos.get(sample_id, "")
                frame = sample_frames.get(sample_id)
                group_key = f"{sample_datasets.get(sample_id, '')}\x1f{video}"
                first_frame = first_frames.get(group_key)
                if frame is not None and first_frame is not None and frame > first_frame:
                    historical_ids.append(sample_id)
                else:
                    fallback_candidates.append(sample_id)

            historical_set = set(historical_ids)
            runtime_ids = []
            fallback_ids = []
            for split_name in ("train", "val", "test"):
                eligible_split = [
                    sample_id for sample_id in whitelist_eligible_ids
                    if split_for(sample_id, sample_datasets.get(sample_id, "")) == split_name
                ]
                historical_split = [sample_id for sample_id in eligible_split if sample_id in historical_set]
                if args.historical_reference_policy == "strict" or historical_split:
                    runtime_ids.extend(historical_split)
                else:
                    runtime_ids.extend(eligible_split)
                    fallback_ids.extend(eligible_split)
        else:
            historical_ids = []
            fallback_candidates = []
            fallback_ids = []
            runtime_ids = whitelist_eligible_ids

        datasets: dict[str, int] = {}
        runtime_splits = {"train": 0, "val": 0, "test": 0}
        fallback_splits = {"train": 0, "val": 0, "test": 0}
        for sample_id in fallback_ids:
            fallback_split = split_for(sample_id, sample_datasets.get(sample_id, ""))
            fallback_splits[fallback_split] += 1
        for sample_id in base_ids:
            dataset = sample_datasets.get(sample_id, "")
            datasets[dataset] = datasets.get(dataset, 0) + 1
        for sample_id in runtime_ids:
            split = split_for(sample_id, sample_datasets.get(sample_id, ""))
            runtime_splits[split] += 1

        affected_by_v17 = (
            target in V17_CHANGED_MODALITIES
            or condition in V17_CHANGED_MODALITIES
            or bool(set(datasets).intersection(V17_CHANGED_DATASETS))
        )
        rows.append(
            {
                "index": index,
                "key": key,
                "pair_exists": bool(entry.get("pair_exists")),
                "task_type": str(entry.get("task_type") or ""),
                "conditions": conditions,
                "effective_condition": condition,
                "target": target,
                "route": route,
                "base_pairs": len(base_ids),
                "modality_dataset_eligible_pairs": len(modality_dataset_eligible_ids),
                "whitelist_eligible_pairs": len(whitelist_eligible_ids),
                "runtime_pairs": len(runtime_ids),
                "historical_pairs": len(historical_ids),
                "fallback_candidates": len(fallback_candidates),
                "fallback_pairs": len(fallback_ids) if args.historical_reference_policy == "tolerant" else 0,
                "historical_reference_policy": args.historical_reference_policy,
                "runtime_splits": runtime_splits,
                "fallback_splits": fallback_splits,
                "datasets": datasets,
                "affected_by_v17": affected_by_v17,
            }
        )

    true_rows = [row for row in rows if row["pair_exists"]]
    base_gaps = [row for row in true_rows if row["base_pairs"] == 0]
    modality_dataset_gaps = [
        row for row in true_rows
        if row["base_pairs"] > 0 and row["modality_dataset_eligible_pairs"] == 0
    ]
    whitelist_gaps = [
        row for row in true_rows
        if row["modality_dataset_eligible_pairs"] > 0 and row["whitelist_eligible_pairs"] == 0
    ]
    runtime_gaps = [row for row in true_rows if row["runtime_splits"]["train"] == 0]
    route_only_gaps = [
        row for row in runtime_gaps
        if row["whitelist_eligible_pairs"] > 0
    ]
    old_batch_affected = [
        row for row in rows
        if row["index"] <= args.completed_head_end and row["pair_exists"] and row["affected_by_v17"]
    ]
    report = {
        "summary": {
            "config_entries": len(rows),
            "pair_exists_true": len(true_rows),
            "base_gap_count": len(base_gaps),
            "modality_dataset_gap_count": len(modality_dataset_gaps),
            "whitelist_gap_count": len(whitelist_gaps),
            "runtime_train_gap_count": len(runtime_gaps),
            "route_only_gap_count": len(route_only_gaps),
            "old_batch_affected_count": len(old_batch_affected),
            "base_gap_indices": [row["index"] for row in base_gaps],
            "modality_dataset_gap_indices": [row["index"] for row in modality_dataset_gaps],
            "whitelist_gap_indices": [row["index"] for row in whitelist_gaps],
            "route_only_gap_indices": [row["index"] for row in route_only_gaps],
            "old_batch_affected_indices": [row["index"] for row in old_batch_affected],
        },
        "modality_counts": modality_counts,
        "base_gaps": base_gaps,
        "modality_dataset_gaps": modality_dataset_gaps,
        "whitelist_gaps": whitelist_gaps,
        "route_only_gaps": route_only_gaps,
        "old_batch_affected": old_batch_affected,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
