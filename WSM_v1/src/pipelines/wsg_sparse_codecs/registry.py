from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .dwpose_heatmap import (
    DWPoseBodyOnlyHeatmapCodec,
    DWPoseBodyPartitionHeatmapCodec,
    DWPoseHeatmapCodec,
    DWPosePersonKeypointPack3HeatmapCodec,
)


LORA_GROUP_META_KEY = "__meta__"


def sparse_modalities_from_config(config_path: str | Path) -> set[str]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        graph = json.load(f).get("wsg_v1", {})
    sparse: set[str] = set()
    if not isinstance(graph, dict):
        return sparse
    for _group_name, group in graph.items():
        if not isinstance(group, dict):
            continue
        meta = group.get(LORA_GROUP_META_KEY)
        if not isinstance(meta, dict) or str(meta.get("target_density", "")).lower() != "sparse":
            continue
        for modality in group:
            if not str(modality).startswith("__"):
                sparse.add(str(modality))
    return sparse


def build_sparse_codec(
    modality: str,
    mode: str = "disabled",
    sigma: float = 3.0,
    radius: int = 8,
    peak_threshold: float = 0.10,
    nms_radius: int = 6,
    max_persons: int | None = None,
):
    mode = str(mode).lower()
    if mode not in {
        "dwpose_semantic_heatmap",
        "semantic_heatmap",
        "heatmap",
        "big_token_heatmap",
        "dwpose_body_partition_heatmap",
        "body_partition_heatmap",
        "dwpose_body_only_heatmap",
        "body_only_heatmap",
        "dwpose_person_keypoint_pack3_heatmap",
        "person_keypoint_pack3_heatmap",
    }:
        return None
    max_persons_value = None
    if max_persons is not None and int(max_persons) > 0:
        # Current DWPose sparse caches are encoded with up to 7 persons.  A
        # larger training cap cannot create missing cached lanes; clamp here so
        # model-side lane counts stay compatible with existing caches.
        max_persons_value = min(int(max_persons), 7)
    if str(modality) == "64-66":
        if mode in {"dwpose_person_keypoint_pack3_heatmap", "person_keypoint_pack3_heatmap"}:
            return DWPosePersonKeypointPack3HeatmapCodec(
                sigma=float(sigma),
                radius=int(radius),
                peak_threshold=float(peak_threshold),
                nms_radius=int(nms_radius),
                max_persons=max_persons_value or 7,
            )
        if mode in {"dwpose_body_partition_heatmap", "body_partition_heatmap"}:
            return DWPoseBodyPartitionHeatmapCodec(
                sigma=float(sigma),
                radius=int(radius),
                peak_threshold=float(peak_threshold),
                nms_radius=int(nms_radius),
                max_persons=max_persons_value or 7,
            )
        if mode in {"dwpose_body_only_heatmap", "body_only_heatmap"}:
            return DWPoseBodyOnlyHeatmapCodec(
                sigma=float(sigma),
                radius=int(radius),
                peak_threshold=float(peak_threshold),
                nms_radius=int(nms_radius),
                max_persons=max_persons_value or 7,
            )
        return DWPoseHeatmapCodec(
            sigma=float(sigma),
            radius=int(radius),
            peak_threshold=float(peak_threshold),
            nms_radius=int(nms_radius),
        )
    return None


def codec_metadata(
    mode: str,
    modalities: Iterable[str],
    sigma: float,
    radius: int,
    peak_threshold: float,
    nms_radius: int,
) -> dict:
    return {
        "sparse_codec_mode": str(mode).lower(),
        "sparse_codec_modalities": sorted(str(item) for item in modalities),
        "sparse_heatmap_sigma": float(sigma),
        "sparse_heatmap_radius": int(radius),
        "sparse_heatmap_peak_threshold": float(peak_threshold),
        "sparse_heatmap_nms_radius": int(nms_radius),
    }
