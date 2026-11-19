#!/usr/bin/env python
"""Stratified, resumable evaluation for trained WSM Channel Heads.

The CLI discovers trained runs, keeps one deployable checkpoint per Head, draws
Heads reproducibly within each task type, and delegates exact model inference to
the canonical training entry.  The metric helpers in this module are imported by
that entry so training and evaluation use the same latent/decoder semantics.

Default pilot: 10 random Heads from each of the four trained task types, one
random held-out sample per Head (approximately 40 generated samples total).
The default report retains decoded image-quality, latent-space, and categorical
metrics so later paper analyses can select columns without rerunning inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    CHANNEL_111_CATEGORY_TO_ID,
    CHANNEL_111_COLORS,
    COARSE_MATERIAL_CATEGORY_COLORS,
    COARSE_OBJECT_CATEGORY_COLORS,
    MASK_VALUE,
    OBJAVERSE_LOCAL_ID_TO_CATEGORY,
    SHAPENET_LOCAL_ID_TO_CATEGORY,
    categorical_rgb_to_ids,
    channel195_semantic_map,
    coarse_material_category_map,
    coarse_object_category_id,
    coarse_thing_category_map,
    dense_channel111_category_map,
    dominant_category_id,
)

from src.pipelines.categorical_refiner import (
    CATEGORICAL_TARGET_MODALITIES,
    DEFAULT_CATEGORICAL_ROOT,
    audio_region_selector_revision,
    categorical_refiner_revision,
)

DEFAULT_TASK_TYPES = (
    "estimation_or_reconstruction",
    "conditional_control",
    "conditional_control+narrow_generation",
    "inverse_rendering",
)
DEFAULT_DENSE_CACHE = PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset"
DEFAULT_XPOSE_CACHE = PROJECT_ROOT / "data/xpose_flux2_frame_latents_sample10pct_2frames"
DEFAULT_TEXT_CACHE = PROJECT_ROOT / "data/wsg_flux2_caption_embeddings_all_datasets"
DEFAULT_MODEL = PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B"
DEFAULT_DENSE_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json"
DEFAULT_SPARSE_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_sparse.json"
CHECKPOINT_NAME = "pytorch_lora_weights.safetensors"
METRIC_VERSION = 13
OUTPUT_SCHEMA_VERSION = 4
DEFAULT_EXPERIMENT_NAME = "channel_head_benchmark"
DEFAULT_RECOMMENDED_HEADS_CONFIG = (
    PROJECT_ROOT / "configs/wsm_v1_recommended_dense_channel_heads.json"
)
DEFAULT_MINIMUM_TRAINED_STEPS = {
    "estimation_or_reconstruction": 400,
    "conditional_control": 240,
    "conditional_control+narrow_generation": 400,
    "inverse_rendering": 240,
}
_INTEGER_CATEGORICAL_PATTERN = re.compile(
    r"(?:category|class value|material[_ ]id|mapped .*id|model id)", re.IGNORECASE
)
# Region-category targets whose values represent stable semantic identities.
# MediaPipe Seg (103) is a directly generated dense human mask and is therefore
# explicitly excluded from classification/refiner metrics. Semantic binary labels such
# as isContainerLike are included. Edge (40) is a structural map, while the
# material-region IDs in 143 are assigned independently inside every object;
# neither is a semantic category target even though both contain discrete
# numeric values.
DISCRETE_SCALAR_MODALITIES = frozenset({"131", "135", "136"})
NON_CATEGORICAL_DENSE_MODALITIES = frozenset({"103"})
DISCRETE_BINARY_VALUES: dict[str, tuple[float, ...]] = {
    "135": (0.0, 1.0),
}
DISCRETE_MODALITIES = frozenset(
    set(DISCRETE_SCALAR_MODALITIES) | {"111", "195"}
)
NON_SEMANTIC_DISCRETE_MODALITIES: dict[str, str] = {
    "40": "binary edge structure, not a semantic class label",
    "143": "per-object local material-region ID without cross-sample label identity",
}
_UUID_PATTERN = re.compile(
    r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})"
)


def _finite_mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def palette_for_modality(modality: str) -> dict[int, tuple[int, int, int]] | None:
    if str(modality) in {"111", "136"}:
        return COARSE_OBJECT_CATEGORY_COLORS
    if str(modality) == "131":
        return COARSE_MATERIAL_CATEGORY_COLORS
    if str(modality) == "195":
        return AVIS_CATEGORY_COLORS
    return None


def is_integer_categorical_target(modality: str, target_description: str) -> bool:
    if str(modality) in NON_CATEGORICAL_DENSE_MODALITIES:
        return False
    if str(modality) in NON_SEMANTIC_DISCRETE_MODALITIES:
        return False
    if palette_for_modality(modality) is not None:
        return False
    if "-" in str(modality):
        return False
    return (
        str(modality) in DISCRETE_SCALAR_MODALITIES
        or bool(_INTEGER_CATEGORICAL_PATTERN.search(str(target_description or "")))
    )


@lru_cache(maxsize=1)
def _articulation_uuid_to_category() -> dict[str, str]:
    """Load the same Articulation metadata fallback used by cache re-encoding."""
    metadata_path = (
        PROJECT_ROOT
        / "wsg_reader/tools/Objaverse/meta_Articulation_XL_2.0.csv"
    )
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row["uuid"]).lower(): str(row["category_label"])
            for row in csv.DictReader(handle)
        }


def global_channel111_category_id(source: np.ndarray, source_path: str) -> int:
    """Resolve channel 111 into the Object/Asset coarse-10 ontology."""
    plane = np.asarray(source[..., 111], dtype=np.float32)
    local_id = dominant_category_id(plane, max_class_id=999999)
    if "ShapeNetSem_results" in str(source_path):
        category_label = SHAPENET_LOCAL_ID_TO_CATEGORY.get(local_id, "")
    else:
        category_label = OBJAVERSE_LOCAL_ID_TO_CATEGORY.get(local_id, "")
        if not category_label:
            match = _UUID_PATTERN.search(Path(source_path).name)
            if match is not None:
                category_label = _articulation_uuid_to_category().get(
                    match.group(0).lower(), ""
                )
        if not category_label:
            category_label = "unknown"
    for category in (part.strip().lower() for part in category_label.split(",")):
        if category in CHANNEL_111_CATEGORY_TO_ID:
            return int(coarse_object_category_id(category))
    raise ValueError(
        f"No global category mapping for channel 111 sample {source_path}: "
        f"local_id={local_id}, label={category_label!r}"
    )


def is_discrete_target(modality: str, target_description: str = "") -> bool:
    return (
        palette_for_modality(modality) is not None
        or is_integer_categorical_target(modality, target_description)
    )


def requires_categorical_refiner(modality: str) -> bool:
    """Return whether deployment requires the region-category sidecar."""
    return str(modality) in CATEGORICAL_TARGET_MODALITIES


def _align_metric_tensor_shapes(
    generated: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove cache-only singleton frame axes; never broadcast metric tensors."""
    while generated.ndim > target.ndim:
        squeeze_dims = [index for index, size in enumerate(generated.shape) if index and size == 1]
        if not squeeze_dims:
            break
        generated = generated.squeeze(squeeze_dims[0])
    while target.ndim > generated.ndim:
        squeeze_dims = [index for index, size in enumerate(target.shape) if index and size == 1]
        if not squeeze_dims:
            break
        target = target.squeeze(squeeze_dims[0])
    if generated.shape != target.shape:
        raise ValueError(
            f"Metric tensor shape mismatch after singleton alignment: "
            f"generated={tuple(generated.shape)} target={tuple(target.shape)}"
        )
    return generated, target


def continuous_prediction_metrics(
    generated_normalized_latents: torch.Tensor,
    target_normalized_latents: torch.Tensor,
    generated_raw_latents: torch.Tensor,
    target_raw_latents: torch.Tensor,
    generated_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
) -> dict[str, float]:
    """Comparable latent and VAE-decoded errors for a true generated sample."""
    generated_normalized_latents, target_normalized_latents = _align_metric_tensor_shapes(
        generated_normalized_latents, target_normalized_latents
    )
    generated_raw_latents, target_raw_latents = _align_metric_tensor_shapes(
        generated_raw_latents, target_raw_latents
    )
    normalized_mse = F.mse_loss(
        generated_normalized_latents.float(), target_normalized_latents.float()
    ).item()
    raw_mse = F.mse_loss(
        generated_raw_latents.float(), target_raw_latents.float()
    ).item()
    return {
        "generated_normalized_latent_mse": float(normalized_mse),
        "generated_raw_latent_mse": float(raw_mse),
        **decoded_prediction_metrics(generated_decoded, target_decoded),
    }


def _decoded_ssim(generated: torch.Tensor, target: torch.Tensor) -> float:
    """Dependency-free local-window SSIM in the decoder's [-1, 1] space."""
    generated = generated.detach().float()
    target = target.detach().float()
    if generated.ndim == 3:
        generated = generated.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if generated.shape != target.shape or generated.ndim != 4:
        raise ValueError(
            f"SSIM expects matching BCHW tensors, got {generated.shape} and {target.shape}"
        )
    channels = int(generated.shape[1])
    minimum_side = int(min(generated.shape[-2:]))
    if minimum_side < 3:
        mean_x = generated.mean()
        mean_y = target.mean()
        var_x = generated.var(unbiased=False)
        var_y = target.var(unbiased=False)
        covariance = ((generated - mean_x) * (target - mean_y)).mean()
        c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
        return float(
            (((2 * mean_x * mean_y + c1) * (2 * covariance + c2))
             / ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)))
            .item()
        )
    kernel_size = min(11, minimum_side)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    sigma = 1.5 * kernel_size / 11.0
    coordinates = torch.arange(
        kernel_size, dtype=generated.dtype, device=generated.device
    ) - (kernel_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    window = window.expand(channels, 1, kernel_size, kernel_size).contiguous()
    padding = kernel_size // 2
    generated = F.pad(generated, (padding,) * 4, mode="reflect")
    target = F.pad(target, (padding,) * 4, mode="reflect")
    mean_x = F.conv2d(generated, window, groups=channels)
    mean_y = F.conv2d(target, window, groups=channels)
    mean_x2, mean_y2 = mean_x.square(), mean_y.square()
    mean_xy = mean_x * mean_y
    variance_x = F.conv2d(generated.square(), window, groups=channels) - mean_x2
    variance_y = F.conv2d(target.square(), window, groups=channels) - mean_y2
    covariance = F.conv2d(generated * target, window, groups=channels) - mean_xy
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    ssim = ((2 * mean_xy + c1) * (2 * covariance + c2)) / (
        (mean_x2 + mean_y2 + c1) * (variance_x + variance_y + c2)
    )
    return float(ssim.mean().item())


def decoded_prediction_metrics(
    generated_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
) -> dict[str, float]:
    """Image-quality metrics for final inference decode versus GT decode."""
    generated = generated_decoded.float()
    target = target_decoded.float()
    decoded_mse = float(F.mse_loss(generated, target).item())
    decoded_mae = float(F.l1_loss(generated, target).item())
    return {
        "decoded_mse": decoded_mse,
        "decoded_mae": decoded_mae,
        "decoded_rmse": float(math.sqrt(max(decoded_mse, 0.0))),
        "decoded_psnr_db": float(10.0 * math.log10(4.0 / max(decoded_mse, 1e-12))),
        "decoded_ssim": _decoded_ssim(generated, target),
    }


def _categorical_overlap_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    class_values: Sequence[int | float],
    *,
    background_value: int | float | None = None,
) -> dict[str, Any]:
    """Per-class precision/recall/F1/IoU plus optional binary foreground scores."""
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Categorical overlap expects matching arrays, got {ground_truth.shape} and {prediction.shape}"
        )
    candidates = sorted(set(class_values), key=float)
    evaluated = [
        value for value in candidates
        if np.any(ground_truth == value) or np.any(prediction == value)
    ]
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    per_class_iou: dict[str, float] = {}
    per_class_dice: dict[str, float] = {}
    for class_value in evaluated:
        gt_class = ground_truth == class_value
        pred_class = prediction == class_value
        true_positive = int(np.logical_and(gt_class, pred_class).sum())
        false_positive = int(np.logical_and(~gt_class, pred_class).sum())
        false_negative = int(np.logical_and(gt_class, ~pred_class).sum())
        key = str(class_value)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        overlap_denominator = true_positive + false_positive + false_negative
        dice_denominator = 2 * true_positive + false_positive + false_negative
        per_class_precision[key] = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        per_class_recall[key] = (
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        per_class_f1[key] = (
            2 * true_positive / dice_denominator if dice_denominator else 0.0
        )
        per_class_iou[key] = (
            true_positive / overlap_denominator if overlap_denominator else 0.0
        )
        per_class_dice[key] = per_class_f1[key]
    result: dict[str, Any] = {
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "macro_precision": _finite_mean(per_class_precision.values()),
        "macro_recall": _finite_mean(per_class_recall.values()),
        "macro_f1": _finite_mean(per_class_f1.values()),
        "mean_iou": _finite_mean(per_class_iou.values()),
        "mean_dice": _finite_mean(per_class_dice.values()),
    }
    if background_value is not None:
        gt_foreground = ground_truth != background_value
        pred_foreground = prediction != background_value
        true_positive = int(np.logical_and(gt_foreground, pred_foreground).sum())
        false_positive = int(np.logical_and(~gt_foreground, pred_foreground).sum())
        false_negative = int(np.logical_and(gt_foreground, ~pred_foreground).sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        union = true_positive + false_positive + false_negative
        dice_denominator = 2 * true_positive + false_positive + false_negative
        result.update({
            "all_pixel_accuracy": float(np.mean(ground_truth == prediction)),
            "foreground_precision": (
                true_positive / precision_denominator if precision_denominator else 0.0
            ),
            "foreground_recall": (
                true_positive / recall_denominator if recall_denominator else 0.0
            ),
            "foreground_iou": true_positive / union if union else 0.0,
            "foreground_dice": (
                2 * true_positive / dice_denominator if dice_denominator else 0.0
            ),
        })
    return result


def _resize_rgb(rgb: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    if tuple(rgb.shape[:2]) == tuple(output_hw):
        return np.asarray(rgb, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32)).permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=output_hw, mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).numpy()


def _resize_scalar(values: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    if tuple(values.shape) == tuple(output_hw):
        return np.asarray(values, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=output_hw, mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def palette_categorical_metrics(
    decoded_prediction: torch.Tensor,
    ground_truth_ids: np.ndarray,
    palette: Mapping[int, tuple[int, int, int]],
    tolerance_fraction: float = 0.20,
) -> dict[str, Any]:
    """Strict nearest-palette and tolerant raw-color categorical accuracy.

    Strict accuracy is not RGB equality: every generated pixel is assigned to
    its nearest fixed class color (black is background).  Tolerant accuracy
    additionally accepts raw decoded RGB within ``tolerance_fraction`` of the
    full RGB-cube diagonal from the correct class anchor.
    """
    prediction = decoded_prediction.detach().float().cpu()
    if prediction.ndim == 4:
        prediction = prediction[0]
    prediction_rgb = ((prediction[:3] + 1.0) * 127.5).clamp(0, 255)
    prediction_rgb = prediction_rgb.permute(1, 2, 0).numpy()
    gt_ids = np.asarray(ground_truth_ids, dtype=np.float32)
    prediction_rgb = _resize_rgb(prediction_rgb, gt_ids.shape)
    pred_ids = categorical_rgb_to_ids(prediction_rgb, dict(palette)).astype(np.int64)
    class_ids = np.asarray(sorted(int(item) for item in palette), dtype=np.int64)
    minimum, maximum = int(class_ids.min()), int(class_ids.max())
    valid = np.isfinite(gt_ids) & (gt_ids != MASK_VALUE)
    gt_rounded = np.rint(gt_ids).astype(np.int64, copy=False)
    valid &= (gt_rounded >= minimum) & (gt_rounded <= maximum)
    if not valid.any():
        return {"kind": "palette", "valid_pixels": 0}

    gt_full = np.full(gt_rounded.shape, int(MASK_VALUE), dtype=np.int64)
    gt_full[valid] = gt_rounded[valid]
    overlap_metrics = _categorical_overlap_metrics(
        gt_full, pred_ids, class_ids.tolist(), background_value=int(MASK_VALUE)
    )
    segmentation_confusion_counts = Counter(
        (int(gt_value), int(pred_value))
        for gt_value, pred_value in zip(gt_full.ravel().tolist(), pred_ids.ravel().tolist())
    )
    gt_valid = gt_rounded[valid]
    pred_valid = pred_ids[valid]
    strict = pred_valid == gt_valid
    palette_array = np.zeros((maximum + 1, 3), dtype=np.float32)
    for class_id, color in palette.items():
        palette_array[int(class_id)] = np.asarray(color, dtype=np.float32)
    correct_colors = palette_array[gt_valid]
    raw_distances = np.linalg.norm(prediction_rgb[valid] - correct_colors, axis=1)
    tolerance_rgb_distance = float(tolerance_fraction) * 255.0 * math.sqrt(3.0)
    # A tolerant score is an expansion of strict accuracy, never a competing
    # criterion: nearest-class matches stay correct even when VAE colors are
    # desaturated, while boundary-crossing colors can additionally pass by raw
    # distance to the correct anchor.
    tolerant = strict | (raw_distances <= tolerance_rgb_distance)
    present_classes = np.unique(gt_valid)
    support_counts = Counter(int(value) for value in gt_valid.tolist())
    confusion_counts = Counter(
        (int(gt_value), int(pred_value))
        for gt_value, pred_value in zip(gt_valid.tolist(), pred_valid.tolist())
    )
    per_class_strict = {
        str(int(class_id)): float(strict[gt_valid == class_id].mean())
        for class_id in present_classes
    }
    per_class_tolerant = {
        str(int(class_id)): float(tolerant[gt_valid == class_id].mean())
        for class_id in present_classes
    }

    gt_category = int(np.bincount(gt_valid, minlength=maximum + 1).argmax())
    pred_foreground = pred_valid[(pred_valid >= minimum) & (pred_valid <= maximum)]
    pred_category = (
        int(np.bincount(pred_foreground, minlength=maximum + 1).argmax())
        if pred_foreground.size else -1
    )
    dominant_distance = (
        float(np.linalg.norm(palette_array[pred_category] - palette_array[gt_category]))
        if pred_category >= minimum else float("inf")
    )
    return {
        "kind": "palette",
        "valid_pixels": int(valid.sum()),
        "strict_correct_pixels": int(strict.sum()),
        "tolerant_correct_pixels": int(tolerant.sum()),
        "per_class_support": {
            str(class_id): int(count) for class_id, count in sorted(support_counts.items())
        },
        "confusion_counts": {
            f"{gt_id}:{pred_id}": int(count)
            for (gt_id, pred_id), count in sorted(confusion_counts.items())
        },
        "strict_pixel_accuracy": float(strict.mean()),
        "tolerant_pixel_accuracy": float(tolerant.mean()),
        "tolerance_fraction": float(tolerance_fraction),
        "tolerance_rgb_distance": tolerance_rgb_distance,
        "mean_correct_anchor_rgb_distance": float(raw_distances.mean()),
        "median_correct_anchor_rgb_distance": float(np.median(raw_distances)),
        "present_gt_classes": [int(value) for value in present_classes],
        "per_class_strict_accuracy": per_class_strict,
        "per_class_tolerant_accuracy": per_class_tolerant,
        "balanced_strict_accuracy": float(np.mean(list(per_class_strict.values()))),
        "balanced_tolerant_accuracy": float(np.mean(list(per_class_tolerant.values()))),
        "dominant_gt_category": gt_category,
        "dominant_pred_category": pred_category,
        "dominant_strict_accuracy": float(pred_category == gt_category),
        "dominant_tolerant_accuracy": float(
            pred_category == gt_category or dominant_distance <= tolerance_rgb_distance
        ),
        "dominant_palette_distance": dominant_distance,
        "segmentation_confusion_counts": {
            f"{gt_id}:{pred_id}": int(count)
            for (gt_id, pred_id), count in sorted(segmentation_confusion_counts.items())
        },
        **overlap_metrics,
    }


def scalar_categorical_metrics(
    decoded_prediction: torch.Tensor,
    ground_truth_values: np.ndarray,
    value_range: tuple[float, float],
    tolerance_fraction: float = 0.20,
    class_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Accuracy for integer scalar targets, with tolerance over the configured span."""
    prediction = decoded_prediction.detach().float().cpu()
    if prediction.ndim == 4:
        prediction = prediction[0]
    normalized = ((prediction[0] + 1.0) * 0.5).numpy()
    gt = np.asarray(ground_truth_values, dtype=np.float32)
    normalized = _resize_scalar(normalized, gt.shape)
    minimum, maximum = map(float, value_range)
    predicted_values = normalized * (maximum - minimum) + minimum
    valid = np.isfinite(gt) & (gt != MASK_VALUE) & (gt >= minimum) & (gt <= maximum)
    if not valid.any():
        return {"kind": "integer_scalar", "valid_pixels": 0}
    pred_raw = predicted_values[valid]
    gt_raw = gt[valid]
    if class_values:
        classes = np.asarray(sorted(set(float(value) for value in class_values)), dtype=np.float32)
        gt_ids = classes[np.abs(gt_raw[:, None] - classes[None, :]).argmin(axis=1)]
        pred_ids = classes[np.abs(pred_raw[:, None] - classes[None, :]).argmin(axis=1)]
    else:
        gt_ids = np.rint(gt_raw).astype(np.int64)
        pred_ids = np.rint(pred_raw).astype(np.int64)
    # Scalar categorical colors span the configured normalized channel range;
    # tolerance is therefore expressed as a fraction of that full span.
    tolerance_value = max(0.0, float(tolerance_fraction)) * (maximum - minimum)
    strict = pred_ids == gt_ids
    tolerant = strict | (np.abs(pred_raw - gt_ids.astype(np.float32)) <= tolerance_value)

    def mode(values: np.ndarray) -> float:
        labels, counts = np.unique(values, return_counts=True)
        return float(labels[int(np.argmax(counts))])

    scalar_classes = sorted(
        set(gt_ids.tolist()) | set(pred_ids.tolist()), key=float
    )
    overlap_metrics = _categorical_overlap_metrics(
        gt_ids, pred_ids, scalar_classes
    )
    gt_category = mode(gt_ids)
    pred_category = mode(pred_ids)
    present_classes = np.unique(gt_ids)
    support_counts = Counter(float(value) for value in gt_ids.tolist())
    confusion_counts = Counter(
        (float(gt_value), float(pred_value))
        for gt_value, pred_value in zip(gt_ids.tolist(), pred_ids.tolist())
    )
    per_class_strict = {
        str(float(class_id)): float(strict[gt_ids == class_id].mean())
        for class_id in present_classes
    }
    per_class_tolerant = {
        str(float(class_id)): float(tolerant[gt_ids == class_id].mean())
        for class_id in present_classes
    }
    return {
        "kind": "integer_scalar",
        "valid_pixels": int(valid.sum()),
        "strict_correct_pixels": int(strict.sum()),
        "tolerant_correct_pixels": int(tolerant.sum()),
        "per_class_support": {
            str(class_id): int(count) for class_id, count in sorted(support_counts.items())
        },
        "confusion_counts": {
            f"{gt_id}:{pred_id}": int(count)
            for (gt_id, pred_id), count in sorted(confusion_counts.items())
        },
        "strict_pixel_accuracy": float(strict.mean()),
        "tolerant_pixel_accuracy": float(tolerant.mean()),
        "tolerance_fraction": float(tolerance_fraction),
        "tolerance_value_units": tolerance_value,
        "mean_absolute_class_error": float(np.abs(pred_raw - gt_ids).mean()),
        "present_gt_classes": [float(value) for value in present_classes],
        "per_class_strict_accuracy": per_class_strict,
        "per_class_tolerant_accuracy": per_class_tolerant,
        "balanced_strict_accuracy": float(np.mean(list(per_class_strict.values()))),
        "balanced_tolerant_accuracy": float(np.mean(list(per_class_tolerant.values()))),
        "dominant_gt_category": gt_category,
        "dominant_pred_category": pred_category,
        "dominant_strict_accuracy": float(pred_category == gt_category),
        **overlap_metrics,
        "dominant_tolerant_accuracy": float(
            pred_category == gt_category
            or abs(pred_category - gt_category) <= tolerance_value
        ),
    }


def _first_item(value: Any, index: int = 0) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        flattened = value.flatten()
        return flattened[index].item() if index < flattened.numel() else None
    return value if index == 0 else None


def batch_crop_info(batch: Mapping[str, Any], index: int = 0) -> dict[str, Any] | None:
    value = batch.get("crop_info")
    if not isinstance(value, Mapping):
        return None
    crop = {key: _first_item(raw, index) for key, raw in value.items()}
    crop = {key: raw for key, raw in crop.items() if raw is not None}
    return crop if {"patch_hw", "crop_y", "crop_x"}.issubset(crop) else None


def _apply_crop(data: np.ndarray, crop: Mapping[str, Any] | None) -> np.ndarray:
    if not crop:
        return data
    patch_hw = int(crop["patch_hw"])
    crop_y, crop_x = int(crop["crop_y"]), int(crop["crop_x"])
    if data.ndim == 3:
        height, width = data.shape[:2]
        padding = ((0, max(0, patch_hw - height)), (0, max(0, patch_hw - width)), (0, 0))
    elif data.ndim == 4:
        height, width = data.shape[1:3]
        padding = ((0, 0), (0, max(0, patch_hw - height)), (0, max(0, patch_hw - width)), (0, 0))
    else:
        raise ValueError(f"Unsupported source target shape: {data.shape}")
    if any(item[1] for item in padding):
        data = np.pad(data, padding, mode="constant", constant_values=MASK_VALUE)
    if data.ndim == 3:
        return data[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw]
    return data[:, crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw]


def load_original_target(
    batch: Mapping[str, Any], target_modality: str, index: int = 0
) -> np.ndarray | None:
    """Load and crop the original WSG/WSV target used before VAE encoding."""
    sample_id = str(_first_item(batch.get("sample_id"), index) or "")
    if not sample_id:
        return None
    if "::" in sample_id:
        source_path, frame_range = sample_id.split("::", 1)
        frame_key = frame_range.split("-", 1)[0]
    else:
        source_path, frame_key = sample_id, ""
    try:
        with np.load(source_path, allow_pickle=False) as payload:
            if frame_key and frame_key in payload:
                source = payload[frame_key]
            elif "data" in payload:
                source = payload["data"]
            elif "wsg_data" in payload:
                source = payload["wsg_data"]
            else:
                return None

            source = np.asarray(source, dtype=np.float32)
        if source.ndim == 4:
            source = source[0]
        asset_category_id = None
        if str(target_modality) == "111":
            # Resolve the sparse asset label before cropping. The crop may omit
            # the mesh record that carries the category while retaining RGB.
            asset_category_id = global_channel111_category_id(source, source_path)
        source = _apply_crop(source, batch_crop_info(batch, index))
        if str(target_modality) == "111":
            return dense_channel111_category_map(
                source,
                require_alpha=True,
                category_id=asset_category_id,
                allow_region_mask_fallback=True,
                allow_category_mask_fallback=True,
                allow_rgb_mask_fallback=True,
            )
        if str(target_modality) == "131":
            return coarse_material_category_map(source)
        if str(target_modality) == "136":
            return coarse_thing_category_map(source, source_path)
        if str(target_modality) == "195":
            return channel195_semantic_map(source)
        if "-" not in str(target_modality):
            channel = int(target_modality)
            if source.ndim == 3 and channel < source.shape[-1]:
                return source[..., channel]
    except Exception:
        return None
    return None


@lru_cache(maxsize=16)
def _categorical_sidecar_store(
    target_modality: str,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, tuple[int, int]]]:
    """Memory-map categorical sidecars and index their exact latent sample IDs."""
    sidecar_dir = DEFAULT_CATEGORICAL_ROOT / str(target_modality)
    paths = sorted(sidecar_dir.glob("shard_*.pt"))
    payloads = tuple(
        torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for path in paths
    )
    sample_index: dict[str, tuple[int, int]] = {}
    for shard_index, payload in enumerate(payloads):
        sample_index.update(
            {
                str(sample_id): (shard_index, row_index)
                for row_index, sample_id in enumerate(payload.get("sample_ids") or [])
            }
        )
    return payloads, sample_index


def _categorical_target_tensors(
    batch: Mapping[str, Any],
    target_modality: str,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    spatial = batch.get("categorical_target_spatial_index_map")
    values = batch.get("categorical_target_values")
    if spatial is not None and values is not None:
        spatial_tensor = spatial.detach().cpu() if torch.is_tensor(spatial) else torch.as_tensor(spatial)
        value_tensor = values.detach().cpu() if torch.is_tensor(values) else torch.as_tensor(values)
        if spatial_tensor.ndim >= 3:
            spatial_tensor = spatial_tensor[batch_index]
        if value_tensor.ndim >= 2:
            value_tensor = value_tensor[batch_index]
        return spatial_tensor, value_tensor, "latent_cache_embedded_categorical_sidecar"

    sample_id = str(_first_item(batch.get("sample_id"), batch_index) or "")
    if not sample_id:
        return None
    payloads, sample_index = _categorical_sidecar_store(str(target_modality))
    location = sample_index.get(sample_id)
    if location is None:
        return None
    shard_index, row_index = location
    payload = payloads[shard_index]
    return (
        torch.as_tensor(payload["category_spatial_index_map"][row_index]),
        torch.as_tensor(payload["category_values"]),
        "categorical_targets_sidecar_by_sample_id",
    )


def cached_categorical_target(
    decoded_prediction: torch.Tensor,
    batch: Mapping[str, Any],
    target_modality: str,
    batch_index: int = 0,
) -> tuple[np.ndarray, str] | None:
    """Recover the exact category-value map paired with the encoded target."""
    tensors = _categorical_target_tensors(batch, target_modality, batch_index)
    if tensors is None:
        return None
    spatial_tensor, value_tensor, source = tensors
    value_tensor = value_tensor.reshape(-1).to(torch.float32)
    spatial_tensor = spatial_tensor.to(torch.long)
    lookup = torch.cat((torch.tensor([MASK_VALUE], dtype=torch.float32), value_tensor))
    valid = (spatial_tensor >= 0) & (spatial_tensor < lookup.numel())
    target = torch.full(spatial_tensor.shape, float(MASK_VALUE), dtype=torch.float32)
    target[valid] = lookup[spatial_tensor[valid]]
    output_hw = tuple(int(item) for item in decoded_prediction.shape[-2:])
    if tuple(target.shape[-2:]) != output_hw:
        target = F.interpolate(target[None, None], size=output_hw, mode="nearest")[0, 0]
    return target.numpy(), source


def discrete_prediction_metrics(
    decoded_prediction: torch.Tensor,
    batch: Mapping[str, Any],
    target_modality: str,
    target_description: str,
    channel_ranges: Mapping[int, tuple[float, float]],
    tolerance_fraction: float = 0.20,
    batch_index: int = 0,
) -> dict[str, Any] | None:
    cached = cached_categorical_target(
        decoded_prediction, batch, target_modality, batch_index
    )
    if cached is None and (DEFAULT_CATEGORICAL_ROOT / str(target_modality)).is_dir():
        return None
    if cached is None:
        original = None
        ground_truth_source = "raw_source_annotation_fallback"
    else:
        original, ground_truth_source = cached
    if original is None:
        original = load_original_target(batch, target_modality, batch_index)
        ground_truth_source = "raw_source_annotation_fallback"
    if original is None:
        return None
    palette = palette_for_modality(target_modality)
    if palette is not None:
        result = palette_categorical_metrics(
            decoded_prediction, original, palette, tolerance_fraction
        )
        result["ground_truth_source"] = ground_truth_source
        return result
    if is_integer_categorical_target(target_modality, target_description):
        channel = int(target_modality)
        if channel in channel_ranges:
            result = scalar_categorical_metrics(
                decoded_prediction,
                original,
                channel_ranges[channel],
                tolerance_fraction,
                class_values=DISCRETE_BINARY_VALUES.get(str(target_modality)),
            )
            result["ground_truth_source"] = ground_truth_source
            return result
    return None


@dataclass(frozen=True)
class HeadCandidate:
    head_index: int
    task_type: str
    condition: tuple[str, ...]
    condition_description: str
    target: str
    target_description: str
    checkpoint: str
    checkpoint_step: int
    checkpoint_validation_loss: float | None
    trained_steps: int
    declared_max_steps: int
    training_completion_ratio: float
    run_dir: str
    runtime_config: str
    log_root: str
    rank: int
    alpha: float
    trainable_target: bool
    trainable_condition: bool
    separate_condition_adapters: bool
    attention_mode: str
    historical_reference_policy: str
    target_dataset_whitelist: tuple[str, ...]
    uses_xpose_cache: bool
    default_compatible: bool
    default_mismatches: tuple[str, ...]
    head_source: str = "dense"


def load_recommended_candidates(
    registry_path: Path,
) -> tuple[list[HeadCandidate], dict[str, Any]]:
    """Load the frozen, reviewed Channel-Head deployment registry.

    Paths in the registry are relative to PROJECT_ROOT so the same model table
    can be used on another checkout without rediscovering runs. Missing files
    fail loudly: silently replacing an official model would make evaluations
    from different machines incomparable.
    """
    registry_path = Path(registry_path).expanduser().resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    default_source = str(payload.get("head_source_default") or "dense")
    candidates: list[HeadCandidate] = []
    missing: list[str] = []
    invalid: list[str] = []
    for entry in entries:
        raw = dict((entry.get("recommended") or {}).get("candidate") or {})
        if not raw:
            invalid.append(str(entry.get("head_index", "unknown")))
            continue
        raw["head_source"] = str(
            raw.get("head_source") or entry.get("head_source") or default_source
        )
        for key in ("checkpoint", "run_dir", "runtime_config", "log_root"):
            value = Path(str(raw[key]))
            resolved = value if value.is_absolute() else PROJECT_ROOT / value
            raw[key] = str(resolved.resolve())
        raw["condition"] = tuple(str(item) for item in raw.get("condition") or [])
        raw["target_dataset_whitelist"] = tuple(
            str(item) for item in raw.get("target_dataset_whitelist") or []
        )
        raw["default_mismatches"] = tuple(
            str(item) for item in raw.get("default_mismatches") or []
        )
        for key in ("checkpoint", "run_dir", "runtime_config"):
            if not Path(raw[key]).exists():
                missing.append(
                    f"Head {int(raw['head_index']):06d}: {key}={raw[key]}"
                )
        candidates.append(HeadCandidate(**raw))
    if invalid:
        raise ValueError(
            f"Recommended-Head registry contains {len(invalid)} invalid entries: "
            + ", ".join(invalid[:10])
        )
    if missing:
        details = "\n".join(missing[:30])
        raise FileNotFoundError(
            f"Recommended-Head registry has {len(missing)} missing paths. "
            "Update the registry explicitly instead of silently rediscovering models.\n"
            f"{details}"
        )
    duplicate_keys = [
        key
        for key, count in Counter(
            (item.head_source, item.task_type, item.head_index) for item in candidates
        ).items()
        if count > 1
    ]
    if duplicate_keys:
        raise ValueError(f"Duplicate recommended Heads: {duplicate_keys[:10]}")
    audit = {
        "selection_source": "recommended_registry",
        "registry_path": str(registry_path),
        "registry_schema_version": payload.get("schema_version"),
        "registry_name": payload.get("name"),
        "registry_generated_at": payload.get("generated_at"),
        "registry_source_evaluation": payload.get("source_evaluation"),
        "registry_evaluation_contract_id": payload.get("evaluation_contract_id"),
        "registry_entries": len(entries),
        "registry_head_source_counts": dict(
            Counter(item.head_source for item in candidates)
        ),
        "eligible_unique_heads": len(candidates),
        "eligible_heads_by_task": dict(
            Counter(item.task_type for item in candidates)
        ),
        "registry_status_counts": dict(
            Counter(
                str(entry.get("recommendation_status") or "unknown")
                for entry in entries
            )
        ),
    }
    return sorted(
        candidates, key=lambda item: (item.head_source, item.task_type, item.head_index)
    ), audit


def _stable_digest(payload: Mapping[str, Any], length: int = 12) -> str:
    encoded = json.dumps(
        _json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def _path_slug(value: str, maximum: int = 72) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (slug or "unnamed")[: int(maximum)]


def checkpoint_revision(candidate: HeadCandidate) -> dict[str, Any]:
    checkpoint = Path(candidate.checkpoint)
    stat = checkpoint.stat()
    return {
        "path": str(checkpoint.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "checkpoint_step": int(candidate.checkpoint_step),
        "checkpoint_validation_loss": candidate.checkpoint_validation_loss,
        "run": Path(candidate.run_dir).name,
        "audio_region_selector": audio_region_selector_revision(checkpoint),
        "categorical_refiner": (
            categorical_refiner_revision(checkpoint)
            if requires_categorical_refiner(candidate.target)
            else None
        ),
    }


def candidate_cache_key(candidate: HeadCandidate) -> str:
    revision = checkpoint_revision(candidate)
    identity = {
        "checkpoint": revision,
        "head_index": int(candidate.head_index),
        "task_type": candidate.task_type,
        "condition": list(candidate.condition),
        "target": candidate.target,
        "rank": int(candidate.rank),
        "alpha": float(candidate.alpha),
        "attention_mode": candidate.attention_mode,
    }
    return (
        f"{_path_slug(Path(candidate.run_dir).name, 56)}"
        f"_step{int(candidate.checkpoint_step):07d}_{_stable_digest(identity)}"
    )


def build_evaluation_contract(
    *,
    samples_per_head: int,
    inference_steps: int,
    tolerance: float,
    seed: int,
    latent_diagnostics: bool,
    sample_selection_digest: str = "",
) -> dict[str, Any]:
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "split": "test",
        "samples_per_head": int(samples_per_head),
        "inference_steps": int(inference_steps),
        "categorical_tolerance": float(tolerance),
        "base_seed": int(seed),
        "head_seed_policy": "base_seed_plus_head_index",
        "sample_selection": (
            "exact_per_head_sample_id"
            if sample_selection_digest else
            "deterministic_random_without_replacement"
        ),
        "sample_selection_digest": str(sample_selection_digest),
        "inference_method": "Euler_from_pure_noise",
        "latent_diagnostics": bool(latent_diagnostics),
        "categorical_refiner": "required_only_for_region_category_targets",
    }


def evaluation_contract_id(contract: Mapping[str, Any]) -> str:
    tolerance = int(round(float(contract["categorical_tolerance"]) * 1000.0))
    return (
        f"m{int(contract['metric_version']):02d}"
        f"_n{int(contract['samples_per_head']):03d}"
        f"_s{int(contract['inference_steps']):03d}"
        f"_seed{int(contract['base_seed'])}"
        f"_tol{tolerance:03d}_{_stable_digest(contract)}"
    )


def result_directory(
    output_root: Path,
    candidate: HeadCandidate,
    contract_id: str,
) -> Path:
    return (
        output_root
        / "results"
        / f"target_{_path_slug(candidate.target)}"
        / _path_slug(candidate.task_type.replace("+", "_plus_"))
        / f"Channel_Head_{candidate.head_index:06d}"
        / candidate_cache_key(candidate)
        / contract_id
    )


def _condition_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _checkpoint_for_run(run_dir: Path) -> tuple[Path, int, float | None] | None:
    """Select validation-best checkpoint inside one already selected run."""
    checkpoints = list((run_dir / "ckpt").glob(f"*/{CHECKPOINT_NAME}"))
    if not checkpoints:
        return None
    by_step: dict[int, Path] = {}
    for checkpoint in checkpoints:
        match = re.search(r"(\d+)$", checkpoint.parent.name)
        if match:
            by_step[int(match.group(1))] = checkpoint
    val_pairs: list[tuple[float, int, Path]] = []
    stats_path = run_dir / "stats.npz"
    if stats_path.is_file():
        try:
            with np.load(stats_path, allow_pickle=False) as stats:
                val_steps = np.asarray(stats.get("val_step", []), dtype=np.float64)
                val_losses = np.asarray(stats.get("val_loss", []), dtype=np.float64)
                for raw_step, raw_loss in zip(val_steps.tolist(), val_losses.tolist()):
                    step = int(round(raw_step))
                    if step in by_step and np.isfinite(raw_loss):
                        val_pairs.append((float(raw_loss), step, by_step[step]))
        except (OSError, TypeError, ValueError):
            val_pairs = []
    if val_pairs:
        loss, step, checkpoint = min(val_pairs, key=lambda item: (item[0], -item[1]))
        return checkpoint, int(step), float(loss)
    ranked = []
    for step, checkpoint in by_step.items():
        final = checkpoint.parent.name.startswith("final_step_")
        ranked.append((step, int(final), checkpoint.parent.name, checkpoint))
    step, _final, _parent, checkpoint = max(ranked, key=lambda item: item[:3])
    return checkpoint, int(step), None


def _trained_steps_for_run(run_dir: Path) -> int:
    """Return the furthest persisted step, including a final checkpoint."""
    steps = []
    for checkpoint in (run_dir / "ckpt").glob(f"*/{CHECKPOINT_NAME}"):
        match = re.search(r"(\d+)$", checkpoint.parent.name)
        if match:
            steps.append(int(match.group(1)))
    return max(steps, default=-1)


def _launcher_flags(run_dir: Path) -> dict[str, Any]:
    """Read the small scalar ``resolved_*`` contract from launcher YAML.

    PyYAML is intentionally not required by the evaluator.  These fields are
    emitted as one-line scalar values by the launcher, so parsing them directly
    also works for historical runs created by older environments.
    """
    path = run_dir / "resolved_config.yaml"
    if not path.exists():
        return {}
    output: dict[str, Any] = {}
    wanted = {
        "resolved_condition_attention_mode",
        "resolved_trainable_target",
        "resolved_trainable_condition",
        "resolved_dense_enhance",
        "resolved_de_timestep",
        "resolved_de_minsnr",
        "resolved_de_huber",
        "resolved_single_step_x0",
        "resolved_high_noise_anchor",
        "resolved_multires_noise",
        "direction_loss_weight",
        "hard_foreground_loss_weight",
        "low_noise_anchor_prob",
        "multi_scale_gradient_loss_weight",
        "soft_distance_loss_weight",
        "topology_loss_weight",
        "endpoint_loss_weight",
    }
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in lines:
        key, separator, raw = line.partition(":")
        key = key.strip()
        if not separator or key not in wanted:
            continue
        value = raw.strip().strip('"').strip("'")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            output[key] = lowered == "true"
        else:
            try:
                output[key] = float(value)
            except ValueError:
                output[key] = value
    return output


def _default_mismatches(
    *,
    task_type: str,
    target: str,
    attention_mode: str,
    rank: int,
    alpha: float,
    trainable_target: bool,
    trainable_condition: bool,
    train: Mapping[str, Any],
    model: Mapping[str, Any],
    launcher: Mapping[str, Any],
) -> tuple[str, ...]:
    """Describe deviations from the stable dense task recipe.

    This deliberately represents the plain/default model rather than a later
    ablation.  Experimental auxiliary losses may be valuable, but must not
    silently replace the baseline when producing a general benchmark.
    """
    mismatches = []
    expected_attention = (
        "native_joint"
        if task_type == "conditional_control+narrow_generation"
        else "modality_tunneling"
    )
    resolved_attention = str(
        launcher.get("resolved_condition_attention_mode") or attention_mode or "auto"
    )
    if resolved_attention not in {"auto", expected_attention}:
        mismatches.append("attention_mode")
    if int(rank) != 8 or not math.isclose(float(alpha), 274.0):
        mismatches.append("rank_or_alpha")
    if not trainable_target:
        mismatches.append("target_adapter_frozen")
    if trainable_condition:
        mismatches.append("condition_adapter_trainable")
    if bool(model.get("separate_condition_adapters", False)):
        mismatches.append("separate_condition_adapters")
    if bool(model.get("audio_dense_condition", False)):
        mismatches.append("audio_dense_condition")

    pose_map_targets = {"67-69", "90-92", "99-101"}
    expected_gradient_weight = 0.1 if str(target) in pose_map_targets else 0.0
    expected_numeric = {
        "edit_loss_weight": 0.0,
        "gradient_domain_loss_weight": expected_gradient_weight,
        "focal_regression_gamma": 0.0,
        "foreground_iou_loss_weight": 0.0,
    }
    for key, expected in expected_numeric.items():
        try:
            differs = not math.isclose(float(train.get(key, 0.0) or 0.0), expected)
        except (TypeError, ValueError):
            differs = True
        if differs:
            mismatches.append(key)
    categorical_mode = str(train.get("categorical_loss_mode", "off") or "off").lower()
    expected_categorical_mode = "prototype_ce" if str(target) == "195" else "off"
    normalized_categorical_mode = "off" if categorical_mode in {"", "none", "off"} else categorical_mode
    if normalized_categorical_mode != expected_categorical_mode:
        mismatches.append("categorical_loss_mode")
    if expected_categorical_mode == "prototype_ce":
        try:
            categorical_weight = float(train.get("categorical_loss_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            categorical_weight = float("nan")
        if not math.isclose(categorical_weight, 0.5):
            mismatches.append("categorical_loss_weight")

    for key in (
        "direction_loss_weight",
        "hard_foreground_loss_weight",
        "low_noise_anchor_prob",
        "multi_scale_gradient_loss_weight",
        "soft_distance_loss_weight",
        "topology_loss_weight",
        "endpoint_loss_weight",
    ):
        try:
            enabled = not math.isclose(float(launcher.get(key, 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            enabled = True
        if enabled:
            mismatches.append(key)

    expected_launcher_flags = {
        "resolved_trainable_target": True,
        "resolved_trainable_condition": False,
        "resolved_dense_enhance": False,
        "resolved_de_timestep": False,
        "resolved_de_minsnr": False,
        "resolved_de_huber": False,
        "resolved_single_step_x0": False,
        "resolved_high_noise_anchor": True,
        "resolved_multires_noise": False,
    }
    for key, expected in expected_launcher_flags.items():
        if key in launcher and launcher[key] != expected:
            mismatches.append(key)
    return tuple(dict.fromkeys(mismatches))


def _attention_mode_from_log(run_dir: Path) -> str:
    launch_log = run_dir / "launch.log"
    if launch_log.exists():
        try:
            with launch_log.open("r", encoding="utf-8", errors="replace") as handle:
                for _ in range(80):
                    line = handle.readline()
                    if not line:
                        break
                    match = re.search(r"\[WSM_v1\] attention=([^ ]+)", line)
                    if match:
                        return match.group(1)
        except OSError:
            pass
    return "auto"


def _candidate_from_runtime(path: Path, log_root: Path) -> HeadCandidate | None:
    try:
        runtime = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    wsg = runtime.get("wsg") or {}
    entry = wsg.get("entry") or {}
    train = runtime.get("train") or {}
    model = runtime.get("model") or {}
    head_index, task_type = wsg.get("channel_head_index"), entry.get("task_type")
    if head_index is None or not task_type or not entry.get("pair_exists", True):
        return None
    run_dir = path.parent
    selected = _checkpoint_for_run(run_dir)
    if selected is None:
        return None
    checkpoint, step, checkpoint_validation_loss = selected
    trained_steps = _trained_steps_for_run(run_dir)
    try:
        declared_max_steps = int(train.get("max_steps") or trained_steps)
    except (TypeError, ValueError):
        declared_max_steps = trained_steps
    completion_ratio = (
        float(trained_steps) / float(declared_max_steps)
        if trained_steps >= 0 and declared_max_steps > 0 else 0.0
    )
    routing = list(model.get("adapter_routing") or [])
    rank_alpha = train.get("channel_head_config") or {}
    condition = _condition_tuple(
        entry.get("generation_condition") or entry.get("condition")
    )
    condition_lane_routing = routing[2:2 + len(condition)]
    trainable_condition = any(item is not None for item in condition_lane_routing)
    trainable_target = len(routing) > 1 and routing[1] is not None
    rank = int(rank_alpha.get("r", 8))
    alpha = float(rank_alpha.get("lora_alpha", rank_alpha.get("r", 8)))
    attention_mode = _attention_mode_from_log(run_dir)
    launcher = _launcher_flags(run_dir)
    default_mismatches = _default_mismatches(
        task_type=str(task_type),
        target=str(entry.get("target") or ""),
        attention_mode=attention_mode,
        rank=rank,
        alpha=alpha,
        trainable_target=trainable_target,
        trainable_condition=trainable_condition,
        train=train,
        model=model,
        launcher=launcher,
    )
    return HeadCandidate(
        head_index=int(head_index),
        task_type=str(task_type),
        condition=condition,
        condition_description=str(entry.get("condition_description") or ""),
        target=str(entry.get("target") or ""),
        target_description=str(entry.get("target_description") or ""),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_step=step,
        checkpoint_validation_loss=checkpoint_validation_loss,
        trained_steps=trained_steps,
        declared_max_steps=declared_max_steps,
        training_completion_ratio=completion_ratio,
        run_dir=str(run_dir.resolve()),
        runtime_config=str(path.resolve()),
        log_root=str(log_root.resolve()),
        rank=rank,
        alpha=alpha,
        trainable_target=trainable_target,
        trainable_condition=trainable_condition,
        separate_condition_adapters=bool(model.get("separate_condition_adapters", False)),
        attention_mode=attention_mode,
        historical_reference_policy=str(train.get("historical_reference_policy") or "tolerant"),
        target_dataset_whitelist=tuple(str(item) for item in train.get("target_dataset_whitelist") or []),
        uses_xpose_cache=bool(wsg.get("uses_xpose_cache", False)),
        default_compatible=not default_mismatches,
        default_mismatches=default_mismatches,
    )


def discover_candidates(
    log_roots: Sequence[Path],
    *,
    selection_policy: str = "default",
    minimum_trained_steps: Mapping[str, int] | None = None,
    minimum_completion_ratio: float = 0.80,
) -> tuple[list[HeadCandidate], dict[str, Any]]:
    """Return one sufficiently trained run for every unique Channel Head.

    ``default`` excludes experimental runtime deviations and selects the most
    trained compatible run. ``latest`` preserves the historical chronological
    behavior while enforcing the requested training thresholds. ``coverage``
    keeps every Head with a checkpoint, preferring a sufficiently trained
    default run and falling back to the strongest available historical run.
    """
    minimums = dict(DEFAULT_MINIMUM_TRAINED_STEPS)
    minimums.update({str(key): int(value) for key, value in (minimum_trained_steps or {}).items()})
    runs: list[HeadCandidate] = []
    for log_root in log_roots:
        if not log_root.exists():
            continue
        for path in log_root.glob("Channel_Head_*/*/resolved_runtime_config.json"):
            candidate = _candidate_from_runtime(path, log_root)
            if candidate is not None:
                runs.append(candidate)

    rejection_reasons: Counter[str] = Counter()
    sufficient = []
    for candidate in runs:
        rejected = False
        if selection_policy == "default" and not candidate.default_compatible:
            rejection_reasons.update(candidate.default_mismatches)
            rejected = True
        minimum_steps = int(minimums.get(candidate.task_type, 0))
        if (
            selection_policy not in {"coverage", "canonical-coverage"}
            and candidate.trained_steps < minimum_steps
        ):
            rejection_reasons["below_task_minimum_steps"] += 1
            rejected = True
        if (
            selection_policy not in {"coverage", "canonical-coverage"}
            and candidate.training_completion_ratio < float(minimum_completion_ratio)
        ):
            rejection_reasons["incomplete_declared_training"] += 1
            rejected = True
        if not rejected:
            sufficient.append(candidate)

    by_head: dict[tuple[str, int], HeadCandidate] = {}
    for candidate in sufficient:
        key = (candidate.task_type, candidate.head_index)
        old = by_head.get(key)
        if old is None:
            by_head[key] = candidate
            continue
        if selection_policy == "latest":
            new_key = (candidate.trained_steps, Path(candidate.run_dir).name)
            old_key = (old.trained_steps, Path(old.run_dir).name)
        elif selection_policy in {"coverage", "canonical-coverage"}:
            def coverage_key(item: HeadCandidate) -> tuple[Any, ...]:
                training_sufficient = (
                    item.trained_steps >= int(minimums.get(item.task_type, 0))
                    and item.training_completion_ratio >= float(minimum_completion_ratio)
                )
                if selection_policy == "canonical-coverage":
                    return (
                        item.default_compatible,
                        item.trained_steps,
                        training_sufficient,
                        Path(item.run_dir).name,
                    )
                return (
                    training_sufficient and item.default_compatible,
                    training_sufficient,
                    item.default_compatible,
                    item.trained_steps,
                    Path(item.run_dir).name,
                )

            new_key = coverage_key(candidate)
            old_key = coverage_key(old)
        else:
            new_key = (candidate.trained_steps, Path(candidate.run_dir).name)
            old_key = (old.trained_steps, Path(old.run_dir).name)
        if new_key > old_key:
            by_head[key] = candidate
    selected = sorted(by_head.values(), key=lambda item: (item.task_type, item.head_index))
    discovered_by_head: dict[tuple[str, int], HeadCandidate] = {}
    for candidate in runs:
        discovered_by_head.setdefault(
            (candidate.task_type, candidate.head_index), candidate
        )
    discovered_discrete = [
        candidate for candidate in discovered_by_head.values()
        if is_discrete_target(candidate.target, candidate.target_description)
    ]
    eligible_discrete = [
        candidate for candidate in selected
        if is_discrete_target(candidate.target, candidate.target_description)
    ]
    selected_sufficient = [
        candidate for candidate in selected
        if (
            candidate.trained_steps >= int(minimums.get(candidate.task_type, 0))
            and candidate.training_completion_ratio >= float(minimum_completion_ratio)
        )
    ]
    audit = {
        "selection_policy": selection_policy,
        "minimum_completion_ratio": float(minimum_completion_ratio),
        "minimum_trained_steps": minimums,
        "discovered_runs": len(runs),
        "discovered_unique_heads": len({(item.task_type, item.head_index) for item in runs}),
        "default_compatible_runs": sum(item.default_compatible for item in runs),
        "sufficient_candidate_runs": len(sufficient),
        "eligible_unique_heads": len(selected),
        "eligible_heads_by_task": dict(Counter(item.task_type for item in selected)),
        "eligible_default_compatible_heads": sum(item.default_compatible for item in selected),
        "eligible_training_sufficient_heads": len(selected_sufficient),
        "eligible_default_and_training_sufficient_heads": sum(
            item.default_compatible for item in selected_sufficient
        ),
        "discrete_discovered_unique_heads": len(discovered_discrete),
        "discrete_discovered_by_target": dict(Counter(item.target for item in discovered_discrete)),
        "discrete_discovered_by_task": dict(Counter(item.task_type for item in discovered_discrete)),
        "discrete_eligible_unique_heads": len(eligible_discrete),
        "discrete_eligible_by_target": dict(Counter(item.target for item in eligible_discrete)),
        "discrete_eligible_by_task": dict(Counter(item.task_type for item in eligible_discrete)),
        "semantic_categorical_modalities": sorted(DISCRETE_MODALITIES, key=int),
        "non_categorical_dense_modalities": sorted(NON_CATEGORICAL_DENSE_MODALITIES, key=int),
        "excluded_discrete_numeric_modalities": NON_SEMANTIC_DISCRETE_MODALITIES,
        "excluded_run_reasons": dict(rejection_reasons),
    }
    return selected, audit


def stratified_selection(
    candidates: Sequence[HeadCandidate],
    task_types: Sequence[str],
    heads_per_task: int,
    seed: int,
    all_heads: bool = False,
) -> list[HeadCandidate]:
    grouped: dict[str, list[HeadCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.task_type in task_types:
            grouped[candidate.task_type].append(candidate)
    selected = []
    for task_offset, task_type in enumerate(task_types):
        population = sorted(grouped.get(task_type, []), key=lambda item: item.head_index)
        if all_heads:
            selected.extend(population)
            continue
        rng = random.Random(int(seed) + 10_007 * task_offset)
        selected.extend(rng.sample(population, k=min(int(heads_per_task), len(population))))
    return selected


def _config_for(candidate: HeadCandidate) -> Path:
    log_root_name = Path(candidate.log_root).name
    if log_root_name == "train_log_sparse":
        return DEFAULT_SPARSE_CONFIG
    if log_root_name == "train_log_custom":
        return PROJECT_ROOT / "configs/wsg_v1_modalities_custom.json"
    return DEFAULT_DENSE_CONFIG


def build_worker_command(
    candidate: HeadCandidate,
    head_output: Path,
    inference_steps: int,
    samples_per_head: int,
    tolerance: float,
    seed: int,
    latent_diagnostics: bool = False,
    preview_sample_id: str = "",
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_TunnelingNet_Flux2_4B_wsg.py"),
        "--model-path", str(DEFAULT_MODEL),
        "--modality-config", str(_config_for(candidate)),
        "--dense-latent-cache-dir", str(DEFAULT_DENSE_CACHE),
        "--text-embedding-cache-dir", str(DEFAULT_TEXT_CACHE),
        "--channel-head-index", str(candidate.head_index),
        "--target-modality", candidate.target,
        "--condition-modalities", ",".join(candidate.condition),
        "--channel-head-rank", str(candidate.rank),
        "--channel-head-alpha", str(candidate.alpha),
        "--max-steps", "1",
        "--batch-size", "1",
        "--num-workers", "0",
        "--val-num-workers", "0",
        "--val-loss-batches", "0",
        "--seed", str(seed),
        "--disable-preflight",
        "--output-dir", str(head_output),
        "--inference-only",
        "--inference-checkpoint", candidate.checkpoint,
        "--categorical-refiner", (
            "auto" if requires_categorical_refiner(candidate.target) else "off"
        ),
        "--inference-label", "benchmark",
        "--inference-steps", str(inference_steps),
        "--preview-split", "test",
        "--preview-require-split",
        "--preview-num-samples", str(samples_per_head),
        "--benchmark-metrics",
        "--benchmark-categorical-tolerance", str(tolerance),
        "--trainable-target", "on" if candidate.trainable_target else "off",
        "--trainable-condition", "on" if candidate.trainable_condition else "off",
        "--historical-reference-policy", candidate.historical_reference_policy,
    ]
    if preview_sample_id:
        command.extend(("--preview-sample-id", str(preview_sample_id)))
    else:
        command.append("--preview-random")
    if latent_diagnostics:
        command.append("--benchmark-latent-diagnostics")
    if candidate.attention_mode != "auto":
        command.extend(("--condition-attention-mode", candidate.attention_mode))
    if candidate.separate_condition_adapters:
        command.append("--separate-condition-adapters")
    if candidate.target_dataset_whitelist:
        command.extend(("--target-dataset-whitelist", ",".join(candidate.target_dataset_whitelist)))
    if candidate.uses_xpose_cache:
        command.extend(("--xpose-latent-cache-dir", str(DEFAULT_XPOSE_CACHE)))
    if candidate.head_index == 10 and Path(candidate.log_root).name == "train_log_sparse":
        command.extend((
            "--categorical-refiner-audio-dataset",
            str(
                PROJECT_ROOT
                / "data/sparse_structure/datasets/"
                / "wsv_structured_scorehmr_scene_audio_175_184_sem195_frame_videoall_ast_test.npz"
            ),
        ))
    return command


def _metadata_files(head_output: Path) -> list[Path]:
    root = head_output / "pseudo-visualization-inference" / "benchmark"
    return sorted(root.glob("**/metadata.json")) if root.exists() else []


def _artifact_record(path: Path, output_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_root.resolve())
    except ValueError:
        relative = resolved
    return {"absolute": str(resolved), "relative": str(relative)}


def load_sample_records(
    head_output: Path,
    output_root: Path,
    candidate: HeadCandidate,
) -> list[dict[str, Any]]:
    """Load worker metadata and attach stable, paper-friendly artifact paths."""
    samples: list[dict[str, Any]] = []
    artifact_names = (
        "metadata.json", "prompt.txt", "condition.png", "reference.png",
        "first_condition.png", "inference.png", "inference_unrefined.png", "gt.png", "summary.png",
    )
    for sample_index, metadata_path in enumerate(_metadata_files(head_output)):
        try:
            sample = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sample_dir = metadata_path.parent
        target_uses_refiner = requires_categorical_refiner(
            str(candidate.target or ""),
        )
        artifacts = {"sample_dir": _artifact_record(sample_dir, output_root)}
        for name in artifact_names:
            artifact_path = sample_dir / name
            if artifact_path.exists():
                artifacts[Path(name).stem] = _artifact_record(artifact_path, output_root)
        reader_dir = sample_dir / "wsg_reader"
        reader_artifacts = {}
        for name in ("condition", "reference", "first_condition", "inference", "generated", "gt"):
            reader_path = reader_dir / f"{name}.png"
            if reader_path.exists():
                record = _artifact_record(reader_path, output_root)
                reader_artifacts[name] = record
                # Formal artifacts prefer the Reader rendering. Pseudo images
                # remain on disk for debugging but are not paper inputs.
                # Categorical predictions are the categorical-refiner result;
                # the Reader copy is retained for audit, not substituted over it.
                if not (
                    target_uses_refiner
                    and name in {"inference", "generated", "gt"}
                ):
                    artifacts[name] = record
        reader_manifest = reader_dir / "manifest.json"
        if reader_manifest.exists():
            artifacts["wsg_reader_manifest"] = _artifact_record(reader_manifest, output_root)
        reader_tensors = {}
        for name in ("condition", "reference", "first_condition", "inference", "generated", "gt"):
            tensor_path = reader_dir / "tensors" / f"{name}.wsg.npz"
            if tensor_path.exists():
                reader_tensors[name] = _artifact_record(tensor_path, output_root)
        if reader_tensors:
            artifacts["wsg_reader_tensors"] = reader_tensors
        identity = {
            "head_index": int(candidate.head_index),
            "target": candidate.target,
            "sample_id": sample.get("sample_id"),
            "sample_index": int(sample_index),
        }
        sample["sample_index"] = int(sample_index)
        required_reader_keys = [] if target_uses_refiner else ["inference", "gt"]
        for routed_input in ("condition", "reference", "first_condition"):
            if routed_input in artifacts and routed_input not in required_reader_keys:
                required_reader_keys.append(routed_input)
        missing_reader_keys = [
            name for name in required_reader_keys if name not in reader_artifacts
        ]
        missing_reader_tensors = [
            name for name in required_reader_keys if name not in reader_tensors
        ]
        sample["wsg_reader_audit"] = {
            "status": (
                "passed"
                if not missing_reader_keys and not missing_reader_tensors
                else "failed"
            ),
            "required": list(required_reader_keys),
            "missing": missing_reader_keys,
            "missing_denormalized_wsg": missing_reader_tensors,
            "available": sorted(reader_artifacts),
            "available_denormalized_wsg": sorted(reader_tensors),
            "formal_artifact_source": (
                "categorical_refiner" if target_uses_refiner else "wsg_reader"
            ),
        }
        sample["inference_route_audit"] = inference_route_audit(candidate, sample, artifacts)
        sample["figure_id"] = (
            f"h{candidate.head_index:06d}_t{_path_slug(candidate.target)}_"
            f"s{sample_index:03d}_{_stable_digest(identity, 8)}"
        )
        sample["artifacts"] = artifacts
        samples.append(sample)
    return samples


def inference_route_audit(
    candidate: HeadCandidate,
    sample: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that saved inference inputs satisfy the task routing contract.

    Older imported evaluations predate temporal provenance in metadata. They
    remain usable for metrics, but are marked legacy_unverified rather than
    being mistaken for verified same- or cross-frame inference.
    """
    conditions = set(candidate.condition)
    task_type = str(candidate.task_type)
    video_id = str(sample.get("video_id") or "")
    target_frame = sample.get("target_frame")
    condition_video_id = str(sample.get("condition_video_id") or "")
    condition_frame = sample.get("condition_frame")
    reference_video_id = str(sample.get("reference_video_id") or "")
    reference_frame = sample.get("reference_frame")
    extra_video_id = str(sample.get("extra_condition_video_id") or "")
    extra_frame = sample.get("extra_condition_frame")
    has_condition = bool((artifacts.get("condition") or {}).get("absolute"))
    has_reference = bool((artifacts.get("reference") or {}).get("absolute"))
    has_extra = bool((artifacts.get("first_condition") or {}).get("absolute"))
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    dataset_split = str(sample.get("dataset_split") or "")
    add(
        "held_out_test_split",
        None if not dataset_split else dataset_split == "test",
        f"dataset_split={dataset_split or 'legacy-unrecorded'}",
    )

    if has_condition:
        add(
            "condition_matches_target_video",
            None if not condition_video_id or not video_id
            else condition_video_id == video_id,
            f"condition_video={condition_video_id!r}, target_video={video_id!r}",
        )
        add(
            "condition_matches_target_frame",
            None if condition_frame is None or target_frame is None
            else int(condition_frame) == int(target_frame),
            f"condition_frame={condition_frame}, target_frame={target_frame}",
        )

    if "Reference_IMG" in conditions:
        if not has_reference:
            add(
                "historical_native_reference_present",
                None,
                "reference omitted by tolerant fallback because no valid earlier frame was available",
            )
        else:
            add(
                "historical_native_reference_same_video",
                None if not video_id or not reference_video_id else video_id == reference_video_id,
                f"reference_video={reference_video_id!r}, target_video={video_id!r}",
            )
            add(
                "historical_native_reference_precedes_target",
                None if reference_frame is None or target_frame is None
                else int(reference_frame) < int(target_frame),
                f"reference_frame={reference_frame}, target_frame={target_frame}",
            )
    elif "CurrentFrame_RGB_Reference" in conditions and has_reference:
        add(
            "current_native_reference_matches_target",
            None if reference_frame is None or target_frame is None
            else int(reference_frame) == int(target_frame),
            f"reference_frame={reference_frame}, target_frame={target_frame}",
        )

    visual_conditions = {
        token for token in conditions
        if token not in {
            "caption", "Reference_IMG", "FirstFrame_RGB",
            "CurrentFrame_RGB_Reference", "CurrentFrame_RGB_Condition",
        }
    }
    inverse_rgb_pair = (
        task_type == "inverse_rendering"
        and "0-2" in visual_conditions
        and len(visual_conditions) > 1
    )
    if "FirstFrame_RGB" in conditions or inverse_rgb_pair:
        if not has_extra:
            add(
                "auxiliary_rgb_present", False,
                "expected first/current RGB condition artifact is missing",
            )
        else:
            kind = str(sample.get("extra_condition_reference_kind") or "")
            expected_historical = kind != "current_rgb_fallback"
            if expected_historical:
                add(
                    "historical_auxiliary_rgb_same_video",
                    None if not video_id or not extra_video_id else video_id == extra_video_id,
                    f"extra_video={extra_video_id!r}, target_video={video_id!r}",
                )
                add(
                    "historical_auxiliary_rgb_precedes_target",
                    None if extra_frame is None or target_frame is None
                    else int(extra_frame) < int(target_frame),
                    f"extra_frame={extra_frame}, target_frame={target_frame}, kind={kind or 'legacy'}",
                )
            else:
                add(
                    "current_auxiliary_rgb_matches_target",
                    None if extra_frame is None or target_frame is None
                    else int(extra_frame) == int(target_frame),
                    f"extra_frame={extra_frame}, target_frame={target_frame}, kind={kind}",
                )

    known = [check for check in checks if check["passed"] is not None]
    failures = [check for check in known if not check["passed"]]
    if failures:
        status = "failed"
    elif checks and len(known) == len(checks):
        status = "verified"
    elif checks:
        status = "legacy_unverified"
    else:
        status = "not_applicable"
    return {
        "status": status,
        "checks": checks,
        "task_type": task_type,
        "condition_tokens": list(candidate.condition),
    }


def _cached_result(
    result_path: Path,
    candidate: HeadCandidate,
    contract: Mapping[str, Any],
    contract_id: str,
) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    samples = result.get("samples") or []
    valid = (
        result.get("status") == "complete"
        and int(result.get("output_schema_version") or 0) == OUTPUT_SCHEMA_VERSION
        and int(result.get("metric_version") or 0) == METRIC_VERSION
        and result.get("candidate_cache_key") == candidate_cache_key(candidate)
        and result.get("evaluation_contract_id") == contract_id
        and result.get("evaluation_contract") == dict(contract)
        and 0 < len(samples) <= int(contract["samples_per_head"])
    )
    if not valid:
        return None
    # A complete cache entry must still point to usable inference artifacts.
    for sample in samples:
        artifacts = sample.get("artifacts") or {}
        for key in ("metadata", "inference", "gt", "summary"):
            artifact = artifacts.get(key) or {}
            if not Path(str(artifact.get("absolute") or "")).is_file():
                return None
    cached = dict(result)
    cached["cache_hit"] = True
    return cached


def evaluate_one(
    candidate: HeadCandidate,
    device: str,
    output_root: Path,
    contract: Mapping[str, Any],
    contract_id: str,
    force: bool = False,
    preview_sample_id: str = "",
) -> dict[str, Any]:
    result_dir = result_directory(output_root, candidate, contract_id)
    result_path = result_dir / "evaluation_result.json"
    revision = checkpoint_revision(candidate)
    if (
        requires_categorical_refiner(candidate.target)
        and revision.get("categorical_refiner") is None
    ):
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "failed",
            "failure_reason": "missing_categorical_refiner",
            "message": (
                "Categorical outputs require an embedded refiner or "
                f"{Path(candidate.run_dir) / 'categorical_refiner' / 'best.pt'}"
            ),
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "metric_version": METRIC_VERSION,
            "candidate_cache_key": candidate_cache_key(candidate),
            "checkpoint_revision": revision,
            "evaluation_contract_id": contract_id,
            "evaluation_contract": dict(contract),
            "elapsed_seconds": 0.0,
            "cache_hit": False,
            "candidate": asdict(candidate),
            "samples": [],
        }
        atomic_json_dump(result, result_path)
        atomic_json_dump(result, result_dir / "latest_failure.json")
        return result
    if not force:
        cached = _cached_result(result_path, candidate, contract, contract_id)
        if cached is not None:
            return cached

    result_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pid{os.getpid()}_gpu{_path_slug(device)}"
    )
    attempt_output = result_dir / "attempts" / attempt_id
    attempt_output.mkdir(parents=True, exist_ok=False)
    head_seed = int(contract["base_seed"]) + int(candidate.head_index)
    command = build_worker_command(
        candidate,
        attempt_output,
        int(contract["inference_steps"]),
        int(contract["samples_per_head"]),
        float(contract["categorical_tolerance"]),
        head_seed,
        bool(contract["latent_diagnostics"]),
        preview_sample_id,
    )
    (attempt_output / "command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    # Formal evaluation artifacts must use the same WSG Reader visualization
    # contract as the released data. Never substitute the debug pseudo renderer.
    environment["WSG_READER_VISUALIZATION"] = "1"
    environment["WSG_READER_REQUIRED"] = "1"
    environment["WSG_READER_STRICT_GLOBAL"] = "1"
    started = time.time()
    worker_log = attempt_output / "worker.log"
    with worker_log.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    samples = load_sample_records(attempt_output, output_root, candidate)
    requested_samples = int(contract["samples_per_head"])
    skipped_samples = []
    for skipped_path in sorted(attempt_output.rglob("skipped.json")):
        try:
            skipped_samples.append(json.loads(skipped_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            skipped_samples.append(
                {
                    "status": "skipped",
                    "reason": "unreadable_skip_record",
                    "path": str(skipped_path.resolve()),
                }
            )
    route_audit_counts = Counter(
        str((sample.get("inference_route_audit") or {}).get("status") or "missing")
        for sample in samples
    )
    route_audit_failures = [
        {
            "sample_id": sample.get("sample_id"),
            "sample_index": sample.get("sample_index"),
            "audit": sample.get("inference_route_audit"),
        }
        for sample in samples
        if (sample.get("inference_route_audit") or {}).get("status") == "failed"
    ]
    wsg_reader_failures = [
        {
            "sample_id": sample.get("sample_id"),
            "sample_index": sample.get("sample_index"),
            "audit": sample.get("wsg_reader_audit"),
        }
        for sample in samples
        if (sample.get("wsg_reader_audit") or {}).get("status") == "failed"
    ]
    # A held-out split may contain fewer examples than the requested cap.
    # A zero return code means the canonical worker exhausted every available
    # sample; require a non-empty bounded result rather than exactly the cap.
    # Temporal/reference contract violations invalidate the evaluation rather
    # than silently turning control into same-frame reconstruction.
    complete = (
        process.returncode == 0
        and 0 < len(samples) <= requested_samples
        and not route_audit_failures
        and not wsg_reader_failures
    )
    result = {
        "status": "complete" if complete else "failed",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "metric_profile": (
            "decoded+categorical+latent_diagnostics"
            if contract["latent_diagnostics"] else "decoded+categorical"
        ),
        "candidate_cache_key": candidate_cache_key(candidate),
        "checkpoint_revision": checkpoint_revision(candidate),
        "evaluation_contract_id": contract_id,
        "evaluation_contract": dict(contract),
        "returncode": int(process.returncode),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "cache_hit": False,
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_output.resolve()),
        "result_dir": str(result_dir.resolve()),
        "candidate": asdict(candidate),
        "samples": samples,
        "requested_samples": requested_samples,
        "successful_samples": len(samples),
        "skipped_samples": skipped_samples,
        "skipped_sample_count": len(skipped_samples),
        "inference_route_audit_counts": dict(route_audit_counts),
        "inference_route_audit_failures": route_audit_failures,
        "wsg_reader_audit_failures": wsg_reader_failures,
        "worker_log": str(worker_log.resolve()),
    }
    atomic_json_dump(result, attempt_output / "attempt_result.json")
    if complete:
        atomic_json_dump(result, result_path)
    else:
        # Preserve every failed attempt and expose the newest one without ever
        # replacing an earlier valid canonical result.
        atomic_json_dump(result, result_dir / "latest_failure.json")
    return result

def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_quality: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    samples = []
    for result in results:
        candidate = result.get("candidate") or {}
        task_type = str(candidate.get("task_type") or "unknown")
        by_task[task_type].append(result)
        by_target[str(candidate.get("target") or "unknown")].append(result)
        by_quality[candidate_quality_tier(candidate)].append(result)
        if result.get("status") == "complete":
            for sample in result.get("samples") or []:
                samples.append((task_type, candidate, sample))

    def summarize_samples(
        items: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]]
    ) -> dict[str, Any]:
        continuous = [sample.get("benchmark_metrics") or {} for _, _, sample in items]
        baseline_continuous = [
            sample.get("benchmark_metrics_unrefined") or {} for _, _, sample in items
        ]
        refined_samples = sum(
            bool((sample.get("categorical_refiner") or {}).get("enabled", False))
            for _, _, sample in items
        )
        baseline_categorical = [
            metric.get("categorical")
            for _, candidate, sample in items
            for metric in [sample.get("benchmark_metrics_unrefined") or {}]
            if (
                is_discrete_target(
                    str(candidate.get("target") or ""),
                    str(candidate.get("target_description") or ""),
                )
                and metric.get("categorical")
            )
        ]
        categorical_expected_samples = sum(
            is_discrete_target(
                str(candidate.get("target") or ""),
                str(candidate.get("target_description") or ""),
            )
            for _, candidate, _ in items
        )
        categorical = [
            metric.get("categorical")
            for _, candidate, sample in items
            for metric in [sample.get("benchmark_metrics") or {}]
            if (
                is_discrete_target(
                    str(candidate.get("target") or ""),
                    str(candidate.get("target_description") or ""),
                )
                and metric.get("categorical")
            )
        ]
        categorical_valid = [
            metric for metric in categorical
            if metric.get("strict_pixel_accuracy") is not None
            and math.isfinite(float(metric["strict_pixel_accuracy"]))
        ]

        def mean(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
            return _finite_mean(metric.get(key, float("nan")) for metric in metrics)

        def median(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
            return _finite_median(metric.get(key, float("nan")) for metric in metrics)

        def mean_gain(
            refined: Sequence[Mapping[str, Any]],
            baseline: Sequence[Mapping[str, Any]],
            key: str,
        ) -> float | None:
            values = []
            for refined_item, baseline_item in zip(refined, baseline):
                if key not in refined_item or key not in baseline_item:
                    continue
                values.append(float(refined_item[key]) - float(baseline_item[key]))
            return _finite_mean(values)

        return {
            "samples": len(items),
            "decoded_mse": mean(continuous, "decoded_mse"),
            "decoded_mse_median": median(continuous, "decoded_mse"),
            "decoded_mae": mean(continuous, "decoded_mae"),
            "decoded_mae_median": median(continuous, "decoded_mae"),
            "decoded_rmse": mean(continuous, "decoded_rmse"),
            "decoded_psnr_db": mean(continuous, "decoded_psnr_db"),
            "decoded_psnr_db_median": median(continuous, "decoded_psnr_db"),
            "decoded_ssim": mean(continuous, "decoded_ssim"),
            "decoded_ssim_median": median(continuous, "decoded_ssim"),
            "generated_normalized_latent_mse": mean(
                continuous, "generated_normalized_latent_mse"
            ),
            "generated_raw_latent_mse": mean(continuous, "generated_raw_latent_mse"),
            "categorical_refiner_samples": refined_samples,
            "decoded_mse_unrefined": mean(baseline_continuous, "decoded_mse"),
            "decoded_mae_unrefined": mean(baseline_continuous, "decoded_mae"),
            "decoded_rmse_unrefined": mean(baseline_continuous, "decoded_rmse"),
            "decoded_psnr_db_unrefined": mean(baseline_continuous, "decoded_psnr_db"),
            "decoded_ssim_unrefined": mean(baseline_continuous, "decoded_ssim"),
            "decoded_mse_gain": (
                None if mean_gain(continuous, baseline_continuous, "decoded_mse") is None
                else -float(mean_gain(continuous, baseline_continuous, "decoded_mse"))
            ),
            "decoded_psnr_db_gain": mean_gain(
                continuous, baseline_continuous, "decoded_psnr_db"
            ),
            "decoded_ssim_gain": mean_gain(
                continuous, baseline_continuous, "decoded_ssim"
            ),
            "categorical_strict_pixel_accuracy_unrefined": mean(
                baseline_categorical, "strict_pixel_accuracy"
            ),
            "categorical_tolerant_pixel_accuracy_unrefined": mean(
                baseline_categorical, "tolerant_pixel_accuracy"
            ),
            "categorical_mean_iou_unrefined": mean(baseline_categorical, "mean_iou"),
            "categorical_mean_dice_unrefined": mean(baseline_categorical, "mean_dice"),
            "categorical_macro_f1_unrefined": mean(baseline_categorical, "macro_f1"),
            "categorical_samples": len(categorical),
            "categorical_expected_samples": categorical_expected_samples,
            "categorical_valid_samples": len(categorical_valid),
            "categorical_unavailable_samples": categorical_expected_samples - len(categorical_valid),
            "categorical_strict_pixel_accuracy": mean(categorical, "strict_pixel_accuracy"),
            "categorical_tolerant_pixel_accuracy": mean(categorical, "tolerant_pixel_accuracy"),
            "categorical_balanced_strict_accuracy": mean(categorical, "balanced_strict_accuracy"),
            "categorical_balanced_tolerant_accuracy": mean(categorical, "balanced_tolerant_accuracy"),
            "categorical_dominant_strict_accuracy": mean(categorical, "dominant_strict_accuracy"),
            "categorical_dominant_tolerant_accuracy": mean(categorical, "dominant_tolerant_accuracy"),
            "categorical_all_pixel_accuracy": mean(categorical, "all_pixel_accuracy"),
            "categorical_foreground_precision": mean(categorical, "foreground_precision"),
            "categorical_foreground_recall": mean(categorical, "foreground_recall"),
            "categorical_foreground_iou": mean(categorical, "foreground_iou"),
            "categorical_foreground_dice": mean(categorical, "foreground_dice"),
            "categorical_macro_precision": mean(categorical, "macro_precision"),
            "categorical_macro_recall": mean(categorical, "macro_recall"),
            "categorical_macro_f1": mean(categorical, "macro_f1"),
            "categorical_mean_iou": mean(categorical, "mean_iou"),
            "categorical_mean_dice": mean(categorical, "mean_dice"),
            "categorical_strict_pixel_accuracy_gain": mean_gain(
                categorical, baseline_categorical, "strict_pixel_accuracy"
            ),
            "categorical_mean_iou_gain": mean_gain(
                categorical, baseline_categorical, "mean_iou"
            ),
            "categorical_macro_f1_gain": mean_gain(
                categorical, baseline_categorical, "macro_f1"
            ),
        }

    def summarize_groups(
        groups: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]:
        return {
            group: {
                "heads": len(group_results),
                "successful_heads": sum(
                    result.get("status") == "complete" for result in group_results
                ),
                **summarize_samples([
                    (
                        str((result.get("candidate") or {}).get("task_type") or "unknown"),
                        result.get("candidate") or {},
                        sample,
                    )
                    for result in group_results if result.get("status") == "complete"
                    for sample in result.get("samples") or []
                ]),
            }
            for group, group_results in sorted(groups.items())
        }

    return {
        "status": "complete" if all(item.get("status") == "complete" for item in results) else "partial",
        "heads": len(results),
        "successful_heads": sum(item.get("status") == "complete" for item in results),
        "failed_heads": sum(item.get("status") != "complete" for item in results),
        "overall": summarize_samples(samples),
        "by_task": summarize_groups(by_task),
        "by_target": summarize_groups(by_target),
        "by_checkpoint_quality": summarize_groups(by_quality),
    }


def candidate_quality_tier(candidate: Mapping[str, Any]) -> str:
    """Separate robust baselines from incomplete or experimental fallbacks."""
    task_type = str(candidate.get("task_type") or "")
    sufficient = (
        int(candidate.get("trained_steps") or -1)
        >= int(DEFAULT_MINIMUM_TRAINED_STEPS.get(task_type, 0))
        and float(candidate.get("training_completion_ratio") or 0.0) >= 0.80
    )
    canonical = bool(candidate.get("default_compatible", False))
    if canonical and sufficient:
        return "canonical_sufficient"
    if sufficient:
        return "experimental_sufficient"
    if canonical:
        return "canonical_incomplete"
    return "experimental_incomplete"


def _categorical_payload(
    candidate: Mapping[str, Any], sample: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not is_discrete_target(
        str(candidate.get("target") or ""),
        str(candidate.get("target_description") or ""),
    ):
        return {}
    return (sample.get("benchmark_metrics") or {}).get("categorical") or {}


def per_head_metric_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One stable row per evaluated sample for pandas, sheets, and paper plots."""
    rows = []
    for result in results:
        candidate = result.get("candidate") or {}
        samples = result.get("samples") or [None]
        for sample_index, sample in enumerate(samples):
            sample = sample or {}
            metrics = sample.get("benchmark_metrics") or {}
            categorical = _categorical_payload(candidate, sample)
            baseline_metrics = sample.get("benchmark_metrics_unrefined") or {}
            baseline_categorical = baseline_metrics.get("categorical") or {}
            refiner = sample.get("categorical_refiner") or {}
            artifacts = sample.get("artifacts") or {}

            def relative_artifact(key: str) -> str:
                return str((artifacts.get(key) or {}).get("relative") or "")

            def metric_gain(key: str, *, lower_is_better: bool = False) -> float | None:
                if key not in metrics or key not in baseline_metrics:
                    return None
                difference = float(metrics[key]) - float(baseline_metrics[key])
                return -difference if lower_is_better else difference

            rows.append({
                "status": result.get("status"),
                "evaluation_contract_id": result.get("evaluation_contract_id"),
                "candidate_cache_key": result.get("candidate_cache_key"),
                "task_type": candidate.get("task_type"),
                "head_index": candidate.get("head_index"),
                "condition": ",".join(candidate.get("condition") or []),
                "target": candidate.get("target"),
                "sample_index": sample.get("sample_index", sample_index),
                "sample_id": sample.get("sample_id"),
                "figure_id": sample.get("figure_id"),
                "dataset_split": sample.get("dataset_split"),
                "video_id": sample.get("video_id"),
                "condition_video_id": sample.get("condition_video_id"),
                "condition_frame": sample.get("condition_frame"),
                "target_frame": sample.get("target_frame"),
                "reference_video_id": sample.get("reference_video_id"),
                "reference_frame": sample.get("reference_frame"),
                "reference_frame_delta": sample.get("reference_frame_delta"),
                "reference_is_historical": sample.get("reference_is_historical"),
                "extra_condition_video_id": sample.get("extra_condition_video_id"),
                "extra_condition_frame": sample.get("extra_condition_frame"),
                "extra_condition_frame_delta": sample.get("extra_condition_frame_delta"),
                "extra_condition_reference_kind": sample.get("extra_condition_reference_kind"),
                "inference_route_audit_status": (
                    sample.get("inference_route_audit") or {}
                ).get("status"),
                "inference_route_audit": sample.get("inference_route_audit"),
                "run": Path(str(candidate.get("run_dir") or "")).name,
                "trained_steps": candidate.get("trained_steps"),
                "declared_max_steps": candidate.get("declared_max_steps"),
                "checkpoint_step": candidate.get("checkpoint_step"),
                "checkpoint": candidate.get("checkpoint"),
                "checkpoint_quality": candidate_quality_tier(candidate),
                "default_compatible": candidate.get("default_compatible"),
                "default_mismatches": ",".join(candidate.get("default_mismatches") or []),
                "training_completion_ratio": candidate.get("training_completion_ratio"),
                "generated_normalized_latent_mse": metrics.get("generated_normalized_latent_mse"),
                "generated_raw_latent_mse": metrics.get("generated_raw_latent_mse"),
                "decoded_mse": metrics.get("decoded_mse"),
                "decoded_mae": metrics.get("decoded_mae"),
                "decoded_rmse": metrics.get("decoded_rmse"),
                "decoded_psnr_db": metrics.get("decoded_psnr_db"),
                "decoded_ssim": metrics.get("decoded_ssim"),
                "decoded_mse_unrefined": baseline_metrics.get("decoded_mse"),
                "decoded_mae_unrefined": baseline_metrics.get("decoded_mae"),
                "decoded_rmse_unrefined": baseline_metrics.get("decoded_rmse"),
                "decoded_psnr_db_unrefined": baseline_metrics.get("decoded_psnr_db"),
                "decoded_ssim_unrefined": baseline_metrics.get("decoded_ssim"),
                "decoded_mse_gain": metric_gain("decoded_mse", lower_is_better=True),
                "decoded_psnr_db_gain": metric_gain("decoded_psnr_db"),
                "decoded_ssim_gain": metric_gain("decoded_ssim"),
                "categorical_refiner_enabled": bool(refiner.get("enabled", False)),
                "categorical_refiner_source_kind": refiner.get("source_kind"),
                "categorical_refiner_source_path": refiner.get("source_path"),
                "categorical_refiner_step": refiner.get("step"),
                "categorical_refiner_val_accuracy": refiner.get("validation_accuracy"),
                "categorical_refiner_val_balanced_accuracy": refiner.get("validation_balanced_accuracy"),
                "categorical_strict_pixel_accuracy_unrefined": baseline_categorical.get("strict_pixel_accuracy"),
                "categorical_tolerant_pixel_accuracy_unrefined": baseline_categorical.get("tolerant_pixel_accuracy"),
                "categorical_mean_iou_unrefined": baseline_categorical.get("mean_iou"),
                "categorical_mean_dice_unrefined": baseline_categorical.get("mean_dice"),
                "categorical_macro_f1_unrefined": baseline_categorical.get("macro_f1"),
                "categorical_kind": categorical.get("kind"),
                "categorical_valid_pixels": categorical.get("valid_pixels"),
                "categorical_strict_correct_pixels": categorical.get("strict_correct_pixels"),
                "categorical_tolerant_correct_pixels": categorical.get("tolerant_correct_pixels"),
                "categorical_strict_pixel_accuracy": categorical.get("strict_pixel_accuracy"),
                "categorical_tolerant_pixel_accuracy": categorical.get("tolerant_pixel_accuracy"),
                "categorical_balanced_strict_accuracy": categorical.get("balanced_strict_accuracy"),
                "categorical_balanced_tolerant_accuracy": categorical.get("balanced_tolerant_accuracy"),
                "categorical_dominant_gt": categorical.get("dominant_gt_category"),
                "categorical_dominant_pred": categorical.get("dominant_pred_category"),
                "categorical_dominant_strict_accuracy": categorical.get("dominant_strict_accuracy"),
                "categorical_dominant_tolerant_accuracy": categorical.get("dominant_tolerant_accuracy"),
                "categorical_all_pixel_accuracy": categorical.get("all_pixel_accuracy"),
                "categorical_foreground_precision": categorical.get("foreground_precision"),
                "categorical_foreground_recall": categorical.get("foreground_recall"),
                "categorical_foreground_iou": categorical.get("foreground_iou"),
                "categorical_foreground_dice": categorical.get("foreground_dice"),
                "categorical_macro_precision": categorical.get("macro_precision"),
                "categorical_macro_recall": categorical.get("macro_recall"),
                "categorical_macro_f1": categorical.get("macro_f1"),
                "categorical_mean_iou": categorical.get("mean_iou"),
                "categorical_mean_dice": categorical.get("mean_dice"),
                "categorical_strict_pixel_accuracy_gain": (
                    None if categorical.get("strict_pixel_accuracy") is None
                    or baseline_categorical.get("strict_pixel_accuracy") is None
                    else float(categorical["strict_pixel_accuracy"])
                    - float(baseline_categorical["strict_pixel_accuracy"])
                ),
                "categorical_mean_iou_gain": (
                    None if categorical.get("mean_iou") is None
                    or baseline_categorical.get("mean_iou") is None
                    else float(categorical["mean_iou"])
                    - float(baseline_categorical["mean_iou"])
                ),
                "categorical_macro_f1_gain": (
                    None if categorical.get("macro_f1") is None
                    or baseline_categorical.get("macro_f1") is None
                    else float(categorical["macro_f1"])
                    - float(baseline_categorical["macro_f1"])
                ),
                "sample_dir": relative_artifact("sample_dir"),
                "condition_image": relative_artifact("condition"),
                "reference_image": relative_artifact("reference"),
                "first_condition_image": relative_artifact("first_condition"),
                "inference_image": relative_artifact("inference"),
                "inference_unrefined_image": relative_artifact("inference_unrefined"),
                "gt_image": relative_artifact("gt"),
                "summary_image": relative_artifact("summary"),
                "metadata_file": relative_artifact("metadata"),
            })
    return sorted(rows, key=lambda row: (
        str(row.get("task_type")), int(row.get("head_index") or -1),
        int(row.get("sample_index") or 0),
    ))


def _sample_metric_summary(
    samples: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    metrics = [sample.get("benchmark_metrics") or {} for _, sample in samples]
    baseline_metrics = [
        sample.get("benchmark_metrics_unrefined") or {} for _, sample in samples
    ]
    categorical = [_categorical_payload(candidate, sample) for candidate, sample in samples]
    categorical = [item for item in categorical if item]
    baseline_categorical = [
        (sample.get("benchmark_metrics_unrefined") or {}).get("categorical") or {}
        for candidate, sample in samples
        if is_discrete_target(
            str(candidate.get("target") or ""),
            str(candidate.get("target_description") or ""),
        )
        and (sample.get("benchmark_metrics_unrefined") or {}).get("categorical")
    ]

    def mean(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
        return _finite_mean(item.get(key, float("nan")) for item in items)

    def median(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
        return _finite_median(item.get(key, float("nan")) for item in items)

    def gain(
        refined: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], key: str
    ) -> float | None:
        values = [
            float(refined_item[key]) - float(baseline_item[key])
            for refined_item, baseline_item in zip(refined, baseline)
            if key in refined_item and key in baseline_item
        ]
        return _finite_mean(values)

    valid_pixels = sum(int(item.get("valid_pixels") or 0) for item in categorical)
    strict_correct = sum(int(item.get("strict_correct_pixels") or 0) for item in categorical)
    tolerant_correct = sum(int(item.get("tolerant_correct_pixels") or 0) for item in categorical)
    baseline_valid_pixels = sum(
        int(item.get("valid_pixels") or 0) for item in baseline_categorical
    )
    baseline_strict_correct = sum(
        int(item.get("strict_correct_pixels") or 0) for item in baseline_categorical
    )
    return {
        "samples": len(samples),
        "generated_normalized_latent_mse_mean": mean(
            metrics, "generated_normalized_latent_mse"
        ),
        "generated_raw_latent_mse_mean": mean(metrics, "generated_raw_latent_mse"),
        "decoded_mse_mean": mean(metrics, "decoded_mse"),
        "decoded_mse_median": median(metrics, "decoded_mse"),
        "decoded_mae_mean": mean(metrics, "decoded_mae"),
        "decoded_mae_median": median(metrics, "decoded_mae"),
        "decoded_rmse_mean": mean(metrics, "decoded_rmse"),
        "decoded_psnr_db_mean": mean(metrics, "decoded_psnr_db"),
        "decoded_psnr_db_median": median(metrics, "decoded_psnr_db"),
        "decoded_ssim_mean": mean(metrics, "decoded_ssim"),
        "decoded_ssim_median": median(metrics, "decoded_ssim"),
        "decoded_mse_unrefined_mean": mean(baseline_metrics, "decoded_mse"),
        "decoded_mae_unrefined_mean": mean(baseline_metrics, "decoded_mae"),
        "decoded_rmse_unrefined_mean": mean(baseline_metrics, "decoded_rmse"),
        "decoded_psnr_db_unrefined_mean": mean(baseline_metrics, "decoded_psnr_db"),
        "decoded_ssim_unrefined_mean": mean(baseline_metrics, "decoded_ssim"),
        "decoded_mse_gain": (
            None if gain(metrics, baseline_metrics, "decoded_mse") is None
            else -float(gain(metrics, baseline_metrics, "decoded_mse"))
        ),
        "decoded_psnr_db_gain": gain(metrics, baseline_metrics, "decoded_psnr_db"),
        "decoded_ssim_gain": gain(metrics, baseline_metrics, "decoded_ssim"),
        "categorical_samples": len(categorical),
        "categorical_valid_pixels": valid_pixels,
        "categorical_strict_correct_pixels": strict_correct,
        "categorical_tolerant_correct_pixels": tolerant_correct,
        "categorical_strict_pixel_accuracy_micro": (
            strict_correct / valid_pixels if valid_pixels else None
        ),
        "categorical_tolerant_pixel_accuracy_micro": (
            tolerant_correct / valid_pixels if valid_pixels else None
        ),
        "categorical_strict_pixel_accuracy_unrefined_micro": (
            baseline_strict_correct / baseline_valid_pixels if baseline_valid_pixels else None
        ),
        "categorical_strict_pixel_accuracy_macro": mean(categorical, "strict_pixel_accuracy"),
        "categorical_tolerant_pixel_accuracy_macro": mean(categorical, "tolerant_pixel_accuracy"),
        "categorical_balanced_strict_accuracy_macro": mean(categorical, "balanced_strict_accuracy"),
        "categorical_balanced_tolerant_accuracy_macro": mean(categorical, "balanced_tolerant_accuracy"),
        "categorical_dominant_strict_accuracy": mean(categorical, "dominant_strict_accuracy"),
        "categorical_dominant_tolerant_accuracy": mean(categorical, "dominant_tolerant_accuracy"),
        "categorical_all_pixel_accuracy": mean(categorical, "all_pixel_accuracy"),
        "categorical_foreground_precision": mean(categorical, "foreground_precision"),
        "categorical_foreground_recall": mean(categorical, "foreground_recall"),
        "categorical_foreground_iou": mean(categorical, "foreground_iou"),
        "categorical_foreground_dice": mean(categorical, "foreground_dice"),
        "categorical_macro_precision": mean(categorical, "macro_precision"),
        "categorical_macro_recall": mean(categorical, "macro_recall"),
        "categorical_macro_f1": mean(categorical, "macro_f1"),
        "categorical_mean_iou": mean(categorical, "mean_iou"),
        "categorical_mean_dice": mean(categorical, "mean_dice"),
        "categorical_mean_iou_unrefined": mean(baseline_categorical, "mean_iou"),
        "categorical_macro_f1_unrefined": mean(baseline_categorical, "macro_f1"),
        "categorical_strict_pixel_accuracy_gain": gain(
            categorical, baseline_categorical, "strict_pixel_accuracy"
        ),
        "categorical_mean_iou_gain": gain(categorical, baseline_categorical, "mean_iou"),
        "categorical_macro_f1_gain": gain(categorical, baseline_categorical, "macro_f1"),
    }


def head_metric_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        candidate = result.get("candidate") or {}
        pairs = [(candidate, sample) for sample in result.get("samples") or []]
        rows.append({
            "status": result.get("status"),
            "evaluation_contract_id": result.get("evaluation_contract_id"),
            "candidate_cache_key": result.get("candidate_cache_key"),
            "task_type": candidate.get("task_type"),
            "head_index": candidate.get("head_index"),
            "condition": ",".join(candidate.get("condition") or []),
            "target": candidate.get("target"),
            "run": Path(str(candidate.get("run_dir") or "")).name,
            "checkpoint": candidate.get("checkpoint"),
            "checkpoint_step": candidate.get("checkpoint_step"),
            "trained_steps": candidate.get("trained_steps"),
            "checkpoint_quality": candidate_quality_tier(candidate),
            **_sample_metric_summary(pairs),
        })
    return sorted(rows, key=lambda row: (
        str(row.get("target")), str(row.get("task_type")), int(row.get("head_index") or -1)
    ))


def target_metric_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str((result.get("candidate") or {}).get("target") or "unknown")].append(result)
    rows = []
    for target, group in sorted(grouped.items(), key=lambda item: item[0]):
        pairs = [
            (result.get("candidate") or {}, sample)
            for result in group if result.get("status") == "complete"
            for sample in result.get("samples") or []
        ]
        rows.append({
            "target": target,
            "heads": len(group),
            "successful_heads": sum(item.get("status") == "complete" for item in group),
            **_sample_metric_summary(pairs),
        })
    return rows


def categorical_detail_rows(
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_totals: dict[tuple[str, str], dict[str, Any]] = {}
    confusion_totals: Counter[tuple[str, str, str]] = Counter()
    for result in results:
        candidate = result.get("candidate") or {}
        target = str(candidate.get("target") or "unknown")
        head_index = int(candidate.get("head_index") or -1)
        for sample in result.get("samples") or []:
            categorical = _categorical_payload(candidate, sample)
            support = categorical.get("per_class_support") or {}
            strict = categorical.get("per_class_strict_accuracy") or {}
            tolerant = categorical.get("per_class_tolerant_accuracy") or {}
            for class_id, count_value in support.items():
                count = int(count_value)
                key = (target, str(class_id))
                total = class_totals.setdefault(key, {
                    "target": target, "class_id": str(class_id), "support_pixels": 0,
                    "strict_correct_pixels": 0.0, "tolerant_correct_pixels": 0.0,
                    "heads": set(), "samples": 0,
                })
                total["support_pixels"] += count
                total["strict_correct_pixels"] += float(strict.get(str(class_id), 0.0)) * count
                total["tolerant_correct_pixels"] += float(tolerant.get(str(class_id), 0.0)) * count
                total["heads"].add(head_index)
                total["samples"] += 1
            confusion = (
                categorical.get("segmentation_confusion_counts")
                or categorical.get("confusion_counts")
                or {}
            )
            for pair, count in confusion.items():
                gt_class, separator, pred_class = str(pair).partition(":")
                if separator:
                    confusion_totals[(target, gt_class, pred_class)] += int(count)
    class_rows = []
    for total in class_totals.values():
        support = int(total["support_pixels"])
        target = str(total["target"])
        class_id = str(total["class_id"])
        true_positive = int(confusion_totals.get((target, class_id, class_id), 0))
        false_positive = sum(
            int(count) for (row_target, gt_class, pred_class), count in confusion_totals.items()
            if row_target == target and pred_class == class_id and gt_class != class_id
        )
        false_negative = sum(
            int(count) for (row_target, gt_class, pred_class), count in confusion_totals.items()
            if row_target == target and gt_class == class_id and pred_class != class_id
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        dice_denominator = 2 * true_positive + false_positive + false_negative
        class_rows.append({
            "target": target,
            "class_id": class_id,
            "support_pixels": support,
            "heads": len(total["heads"]),
            "samples": total["samples"],
            "true_positive_pixels": true_positive,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": false_negative,
            "precision": true_positive / precision_denominator if precision_denominator else 0.0,
            "recall": true_positive / recall_denominator if recall_denominator else 0.0,
            "f1": 2 * true_positive / dice_denominator if dice_denominator else 0.0,
            "iou": true_positive / iou_denominator if iou_denominator else 0.0,
            "dice": 2 * true_positive / dice_denominator if dice_denominator else 0.0,
            "strict_pixel_accuracy": total["strict_correct_pixels"] / support if support else None,
            "tolerant_pixel_accuracy": total["tolerant_correct_pixels"] / support if support else None,
        })
    confusion_rows = [
        {"target": target, "gt_class": gt, "pred_class": pred, "pixels": count}
        for (target, gt, pred), count in sorted(confusion_totals.items())
    ]
    return sorted(class_rows, key=lambda row: (row["target"], row["class_id"])), confusion_rows


def image_index_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        candidate = result.get("candidate") or {}
        for sample in result.get("samples") or []:
            artifacts = sample.get("artifacts") or {}
            row = {
                "figure_id": sample.get("figure_id"),
                "head_index": candidate.get("head_index"),
                "task_type": candidate.get("task_type"),
                "condition": ",".join(candidate.get("condition") or []),
                "target": candidate.get("target"),
                "sample_id": sample.get("sample_id"),
                "sample_index": sample.get("sample_index"),
                "dataset_split": sample.get("dataset_split"),
                "video_id": sample.get("video_id"),
                "condition_video_id": sample.get("condition_video_id"),
                "condition_frame": sample.get("condition_frame"),
                "target_frame": sample.get("target_frame"),
                "reference_video_id": sample.get("reference_video_id"),
                "reference_frame": sample.get("reference_frame"),
                "reference_frame_delta": sample.get("reference_frame_delta"),
                "reference_is_historical": sample.get("reference_is_historical"),
                "extra_condition_video_id": sample.get("extra_condition_video_id"),
                "extra_condition_frame": sample.get("extra_condition_frame"),
                "extra_condition_frame_delta": sample.get("extra_condition_frame_delta"),
                "extra_condition_reference_kind": sample.get("extra_condition_reference_kind"),
                "inference_route_audit_status": (
                    sample.get("inference_route_audit") or {}
                ).get("status"),
                "wsg_reader_audit_status": (
                    sample.get("wsg_reader_audit") or {}
                ).get("status"),
                "formal_artifact_source": (
                    sample.get("wsg_reader_audit") or {}
                ).get("formal_artifact_source"),
            }
            for key in ("sample_dir", "condition", "reference", "first_condition", "inference", "inference_unrefined", "gt", "summary", "metadata", "prompt"):
                record = artifacts.get(key) or {}
                row[f"{key}_relative"] = record.get("relative", "")
                row[f"{key}_absolute"] = record.get("absolute", "")
            reader_tensors = artifacts.get("wsg_reader_tensors") or {}
            for key in ("condition", "reference", "first_condition", "inference", "generated", "gt"):
                record = reader_tensors.get(key) or {}
                row[f"{key}_wsg_relative"] = record.get("relative", "")
                row[f"{key}_wsg_absolute"] = record.get("absolute", "")
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row.get("head_index") or -1), int(row.get("sample_index") or 0)))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    return _json_ready(value)


def _write_rows(stem: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_json_dump({"metric_version": METRIC_VERSION, "rows": list(rows)}, stem.with_suffix(".json"))
    csv_path = stem.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(value) for key, value in row.items()})
    temporary.replace(csv_path)


def _latex_escape(value: Any) -> str:
    text = str(value if value is not None else "--")
    return (text.replace("\\", r"\textbackslash{}")
                .replace("_", r"\_").replace("%", r"\%")
                .replace("&", r"\&").replace("#", r"\#"))


def _latex_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _latex_escape(value)


def write_latex_tables(
    output_root: Path,
    head_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> None:
    latex_dir = output_root / "tables/latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    target_lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Target & Heads & Samples & MSE $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & Strict $\uparrow$ & mIoU $\uparrow$ & F1 $\uparrow$ \\",
        r"\midrule",
    ]
    for row in target_rows:
        target_lines.append(
            f"{_latex_escape(row['target'])} & {row['successful_heads']} & {row['samples']} & "
            f"{_latex_number(row.get('decoded_mse_mean'))} & "
            f"{_latex_number(row.get('decoded_psnr_db_mean'))} & "
            f"{_latex_number(row.get('decoded_ssim_mean'))} & "
            f"{_latex_number(row.get('categorical_strict_pixel_accuracy_micro'))} & "
            f"{_latex_number(row.get('categorical_mean_iou'))} & "
            f"{_latex_number(row.get('categorical_macro_f1'))} \\\\"
        )
    target_lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    (latex_dir / "categorical_by_target.tex").write_text(
        "\n".join(target_lines), encoding="utf-8"
    )

    head_lines = [
        r"\begin{tabular}{rlrrrrrrrr}",
        r"\toprule",
        r"Head & Target & Samples & MSE $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & Strict $\uparrow$ & Balanced $\uparrow$ & mIoU $\uparrow$ & F1 $\uparrow$ \\",
        r"\midrule",
    ]
    for row in head_rows:
        head_lines.append(
            f"{int(row['head_index']):06d} & {_latex_escape(row['target'])} & {row['samples']} & "
            f"{_latex_number(row.get('decoded_mse_mean'))} & "
            f"{_latex_number(row.get('decoded_psnr_db_mean'))} & "
            f"{_latex_number(row.get('decoded_ssim_mean'))} & "
            f"{_latex_number(row.get('categorical_strict_pixel_accuracy_micro'))} & "
            f"{_latex_number(row.get('categorical_balanced_strict_accuracy_macro'))} & "
            f"{_latex_number(row.get('categorical_mean_iou'))} & "
            f"{_latex_number(row.get('categorical_macro_f1'))} \\\\"
        )
    head_lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    (latex_dir / "categorical_heads.tex").write_text(
        "\n".join(head_lines), encoding="utf-8"
    )


def write_metric_tables(output_root: Path, results: Sequence[Mapping[str, Any]]) -> None:
    sample_rows = per_head_metric_rows(results)
    head_rows = head_metric_rows(results)
    target_rows = target_metric_rows(results)
    class_rows, confusion_rows = categorical_detail_rows(results)
    images = image_index_rows(results)
    tables = output_root / "tables"
    for name, rows in (
        ("sample_metrics", sample_rows), ("head_metrics", head_rows),
        ("target_metrics", target_rows), ("class_metrics", class_rows),
        ("confusion_counts", confusion_rows), ("image_index", images),
    ):
        _write_rows(tables / name, rows)
    # Retain the original top-level filenames for existing notebooks.
    _write_rows(output_root / "per_head_metrics", sample_rows)
    write_latex_tables(output_root, head_rows, target_rows)


def build_catalog_entries(
    candidates: Sequence[HeadCandidate],
    output_root: Path,
    contract: Mapping[str, Any],
    contract_id: str,
) -> list[dict[str, Any]]:
    entries = []
    for candidate in candidates:
        result_dir = result_directory(output_root, candidate, contract_id)
        cached = _cached_result(
            result_dir / "evaluation_result.json", candidate, contract, contract_id
        )
        entries.append({
            "head_index": int(candidate.head_index),
            "task_type": candidate.task_type,
            "condition": list(candidate.condition),
            "target": candidate.target,
            "candidate_cache_key": candidate_cache_key(candidate),
            "evaluation_contract_id": contract_id,
            "result_dir": str(result_dir.resolve()),
            "result_file": str((result_dir / "evaluation_result.json").resolve()),
            "status": (
                "complete" if cached is not None
                else "failed" if (result_dir / "latest_failure.json").exists()
                else "pending"
            ),
            "candidate": asdict(candidate),
        })
    return entries


def collect_catalog_results(
    candidates: Sequence[HeadCandidate],
    output_root: Path,
    contract: Mapping[str, Any],
    contract_id: str,
) -> list[dict[str, Any]]:
    results = []
    for candidate in candidates:
        result_path = result_directory(output_root, candidate, contract_id) / "evaluation_result.json"
        cached = _cached_result(result_path, candidate, contract, contract_id)
        if cached is not None:
            results.append(cached)
    return results


def collect_frozen_manifest_results(
    candidates: Sequence[HeadCandidate],
    output_root: Path,
    contract_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match cached results without stat'ing cluster-only checkpoint paths."""
    exact: dict[tuple[Any, ...], tuple[dict[str, Any], Path]] = {}
    fallback: dict[tuple[Any, ...], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for result_path in output_root.glob("results/**/evaluation_result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            result.get("status") != "complete"
            or result.get("evaluation_contract_id") != contract_id
        ):
            continue
        payload = result.get("candidate") or {}
        identity = (
            str(payload.get("task_type") or ""),
            int(payload.get("head_index", -1)),
            tuple(str(item) for item in payload.get("condition") or []),
            str(payload.get("target") or ""),
        )
        checkpoint = str(payload.get("checkpoint") or "")
        exact[identity + (checkpoint,)] = (result, result_path)
        fallback[identity].append((result, result_path))

    catalog = []
    results = []
    for candidate in candidates:
        identity = (
            candidate.task_type,
            int(candidate.head_index),
            tuple(candidate.condition),
            candidate.target,
        )
        matched = exact.get(identity + (candidate.checkpoint,))
        alternatives = fallback.get(identity, [])
        if matched is None:
            same_run_step = [
                item for item in alternatives
                if Path(str((item[0].get("candidate") or {}).get("run_dir") or "")).name
                == Path(candidate.run_dir).name
                and int((item[0].get("candidate") or {}).get("checkpoint_step") or -1)
                == int(candidate.checkpoint_step)
            ]
            if same_run_step:
                matched = max(
                    same_run_step,
                    key=lambda item: item[1].stat().st_mtime_ns,
                )
            elif len(alternatives) == 1:
                matched = alternatives[0]
        result, result_path = matched if matched is not None else (None, None)
        candidate_key = (
            str((result or {}).get("candidate_cache_key") or "")
            or hashlib.sha256(
                json.dumps(asdict(candidate), sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
        )
        catalog.append({
            "head_index": int(candidate.head_index),
            "task_type": candidate.task_type,
            "condition": list(candidate.condition),
            "target": candidate.target,
            "candidate_cache_key": candidate_key,
            "evaluation_contract_id": contract_id,
            "result_dir": str(result_path.parent.resolve()) if result_path else "",
            "result_file": str(result_path.resolve()) if result_path else "",
            "status": "complete" if result is not None else "pending",
            "candidate": asdict(candidate),
        })
        if result is not None:
            results.append(result)
    return catalog, results


def write_experiment_readme(
    output_root: Path,
    contract_id: str,
    progress: Mapping[str, Any],
) -> None:
    text = f"""# WSM Channel-Head Evaluation Output

This is an incremental, resumable evaluation workspace. Exact completed results
are reused only when both the checkpoint revision and evaluation contract match.
Failed attempts remain under each result's `attempts/` directory and never replace
a valid canonical `evaluation_result.json`.

- Contract: `{contract_id}`
- Catalogued heads: {progress.get('catalogued_heads', 0)}
- Complete: {progress.get('complete_heads', 0)}
- Pending: {progress.get('pending_heads', 0)}
- Last update: {progress.get('updated_at', '')}

## Metrics retained by default

- Latent: normalized and raw generated-latent MSE.
- Decoded image: MSE, MAE, RMSE, PSNR, and local-window SSIM.
- Categorical: strict and tolerant accuracy, balanced and dominant accuracy, foreground precision/recall/IoU/Dice, macro precision/recall/F1, mIoU, per-class scores, and full spatial confusion.
- Refiner experiments retain unrefined baselines and refined-minus-baseline gains.

## Paper-oriented indexes

- `tables/head_metrics.csv`: one row per Head.
- `tables/sample_metrics.csv`: one row per generated test sample.
- `tables/target_metrics.csv`: aggregate metrics per categorical target.
- `tables/class_metrics.csv`: per-class support, precision, recall, F1, IoU, Dice, and accuracy.
- `tables/confusion_counts.csv`: sparse full-image decoded class-confusion counts, including background.
- `tables/image_index.csv`: stable figure IDs plus relative/absolute artifact paths.
- `tables/latex/`: ready-to-include table fragments.

`catalog.json` is the current eligible checkpoint inventory, `progress.json` is
its completion state, and `runs/` records every invocation/subset separately.
"""
    temporary = output_root / "README.md.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output_root / "README.md")

def parse_minimum_trained_steps(raw: str) -> dict[str, int]:
    text = str(raw or "").strip()
    if not text:
        return dict(DEFAULT_MINIMUM_TRAINED_STEPS)
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("--minimum-trained-steps JSON must be an object")
        return {str(key): int(value) for key, value in payload.items()}
    output = {}
    for item in text.split(","):
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(
                "--minimum-trained-steps expects task=steps pairs or a JSON object"
            )
        output[key.strip()] = int(value.strip())
    return output


def load_exact_sample_selection(path: Path) -> dict[int, str]:
    """Load one exact sample ID per Head from a generic or paper manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("heads"), Mapping):
        return {
            int(head): str(sample_id)
            for head, sample_id in payload["heads"].items()
            if str(sample_id)
        }
    if isinstance(payload, Mapping) and isinstance(payload.get("parts"), Mapping):
        for part in payload["parts"].values():
            if isinstance(part, Mapping):
                records.extend(part.get("records") or [])
    elif isinstance(payload, list):
        records.extend(payload)
    selection: dict[int, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or not record.get("sample_id"):
            continue
        head = int(record["head_index"])
        sample_id = str(record["sample_id"])
        previous = selection.get(head)
        if previous is not None and previous != sample_id:
            raise ValueError(
                f"Exact sample manifest assigns multiple samples to Head {head}: "
                f"{previous!r} and {sample_id!r}"
            )
        selection[head] = sample_id
    if not selection:
        raise ValueError(f"No head_index/sample_id records in {path}")
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-roots", nargs="+", default=["train_log"],
                        help="Training log roots to scan. Defaults to dense train_log; add train_log_sparse when desired.")
    parser.add_argument("--task-types", nargs="+", default=list(DEFAULT_TASK_TYPES))
    parser.add_argument("--heads-per-task", type=int, default=10)
    parser.add_argument("--all-heads", action="store_true", help="Evaluate every discovered Head in the requested task types.")
    parser.add_argument(
        "--categorical-only", action="store_true",
        help="Restrict selection to semantic category/binary-label targets (excludes Edge and local region IDs).",
    )
    parser.add_argument(
        "--head-indices", default="",
        help="Optional comma-separated Head indices used for a resumable pilot/subset.",
    )
    parser.add_argument(
        "--target-modalities", default="",
        help=(
            "Optional comma-separated target modalities used to disambiguate "
            "Head indices across dense and sparse registries."
        ),
    )
    parser.add_argument(
        "--sample-selection-manifest", default="",
        help=(
            "Optional JSON manifest containing one exact sample_id per head_index. "
            "Paper expanded-gallery manifests are accepted directly; this forces "
            "one requested sample per selected Head."
        ),
    )
    parser.add_argument("--samples-per-head", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--categorical-tolerance", type=float, default=0.20)
    latent_group = parser.add_mutually_exclusive_group()
    latent_group.add_argument(
        "--latent-diagnostics", dest="latent_diagnostics", action="store_true",
        default=True,
        help="Retain normalized/raw latent MSE diagnostics (enabled by default).",
    )
    latent_group.add_argument(
        "--no-latent-diagnostics", dest="latent_diagnostics", action="store_false",
        help="Skip latent diagnostics for a decoded-image-only benchmark.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--head-source",
        choices=("recommended", "discovery"),
        default="recommended",
        help=(
            "Use the frozen official model registry by default. Choose discovery "
            "only when intentionally rebuilding that registry."
        ),
    )
    parser.add_argument(
        "--recommended-heads-config",
        default=str(DEFAULT_RECOMMENDED_HEADS_CONFIG),
        help="Versioned official Channel-Head registry used by --head-source recommended.",
    )
    parser.add_argument(
        "--report-contract-id", default="",
        help=(
            "With --plan-only, rebuild aggregate reports for an existing stored "
            "contract ID without changing or rerunning its results."
        ),
    )
    parser.add_argument(
        "--catalog-manifest", default="",
        help=(
            "Reuse the exact catalogued HeadCandidate list from an earlier run "
            "manifest. This is intended for report-only reconstruction when "
            "checkpoint paths differ from the current discovery environment."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=("default", "latest", "coverage", "canonical-coverage"),
        default="canonical-coverage",
        help=("canonical-coverage prefers the canonical recipe while retaining every "
              "Head with a checkpoint; default requires sufficient canonical training; "
              "latest permits variants; coverage ranks by training coverage."),
    )
    parser.add_argument("--minimum-completion-ratio", type=float, default=0.80)
    parser.add_argument(
        "--minimum-trained-steps",
        default=",".join(f"{key}={value}" for key, value in DEFAULT_MINIMUM_TRAINED_STEPS.items()),
        help="Per-task persisted-step floors as task=steps pairs or a JSON object.",
    )
    parser.add_argument("--devices", default="5,6", help="Comma-separated physical CUDA devices used by independent workers.")
    parser.add_argument(
        "--output-dir", default="",
        help="Stable experiment directory override. Empty uses src/evaluation_output/<experiment-name>.",
    )
    parser.add_argument(
        "--experiment-name", default=DEFAULT_EXPERIMENT_NAME,
        help="Stable resumable workspace name below src/evaluation_output.",
    )
    parser.add_argument(
        "--run-label", default="",
        help="Optional human-readable label stored on this invocation record.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run selected candidates even when an exact complete cached result exists.",
    )
    parser.add_argument(
        "--report-every", type=int, default=8,
        help="Refresh aggregate reports after this many newly finished workers.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Discover/select Heads and write the manifest without running models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_sample_selection: dict[int, str] = {}
    exact_sample_manifest = None
    if args.sample_selection_manifest:
        exact_sample_manifest = Path(args.sample_selection_manifest).expanduser()
        if not exact_sample_manifest.is_absolute():
            exact_sample_manifest = PROJECT_ROOT / exact_sample_manifest
        exact_sample_manifest = exact_sample_manifest.resolve()
        exact_sample_selection = load_exact_sample_selection(exact_sample_manifest)
        args.samples_per_head = 1
    if args.catalog_manifest and not args.plan_only:
        raise ValueError("--catalog-manifest is report-only and requires --plan-only")
    log_roots = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in map(Path, args.log_roots)
    ]
    minimum_trained_steps = parse_minimum_trained_steps(args.minimum_trained_steps)
    recommended_config = Path(args.recommended_heads_config)
    if not recommended_config.is_absolute():
        recommended_config = PROJECT_ROOT / recommended_config
    if args.catalog_manifest:
        catalog_manifest = Path(args.catalog_manifest).expanduser().resolve()
        frozen_payload = json.loads(catalog_manifest.read_text(encoding="utf-8"))
        candidates = [
            HeadCandidate(**item) for item in frozen_payload.get("catalogued", [])
        ]
        if not candidates:
            raise ValueError(
                f"No catalogued Head candidates in {catalog_manifest}"
            )
        selection_audit = {
            "selection_source": "frozen_run_manifest",
            "catalog_manifest": str(catalog_manifest),
            "catalogued": len(candidates),
        }
    elif args.head_source == "recommended":
        candidates, selection_audit = load_recommended_candidates(
            recommended_config
        )
    else:
        candidates, selection_audit = discover_candidates(
            log_roots,
            selection_policy=args.selection_policy,
            minimum_trained_steps=minimum_trained_steps,
            minimum_completion_ratio=args.minimum_completion_ratio,
        )
        selection_audit["selection_source"] = "train_log_discovery"
    requested_targets = {
        item.strip() for item in str(args.target_modalities).split(",") if item.strip()
    }
    if requested_targets:
        candidates = [
            candidate for candidate in candidates
            if str(candidate.target) in requested_targets
        ]
    if args.categorical_only:
        candidates = [
            candidate for candidate in candidates
            if requires_categorical_refiner(candidate.target)
        ]
    # The catalog always records all currently eligible Heads in the requested
    # task families. --head-indices and stratified sampling affect only this run.
    catalog_candidates = stratified_selection(
        candidates, args.task_types, args.heads_per_task, args.seed, all_heads=True
    )
    requested_head_indices = {
        int(item.strip()) for item in str(args.head_indices).split(",") if item.strip()
    }
    if exact_sample_selection and not requested_head_indices:
        requested_head_indices = set(exact_sample_selection)
    if requested_head_indices:
        selected = [
            candidate for candidate in catalog_candidates
            if candidate.head_index in requested_head_indices
        ]
    else:
        selected = stratified_selection(
            catalog_candidates, args.task_types, args.heads_per_task,
            args.seed, args.all_heads,
        )

    missing_exact_samples = [
        candidate.head_index for candidate in selected
        if exact_sample_selection and candidate.head_index not in exact_sample_selection
    ]
    if missing_exact_samples:
        raise ValueError(
            f"Exact sample manifest is missing selected Heads: {missing_exact_samples[:20]}"
        )
    sample_selection_digest = (
        hashlib.sha256(
            json.dumps(exact_sample_selection, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        if exact_sample_selection else ""
    )

    contract = build_evaluation_contract(
        samples_per_head=args.samples_per_head,
        inference_steps=args.inference_steps,
        tolerance=args.categorical_tolerance,
        seed=args.seed,
        latent_diagnostics=args.latent_diagnostics,
        sample_selection_digest=sample_selection_digest,
    )
    contract_id = evaluation_contract_id(contract)
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        PROJECT_ROOT / "src/evaluation_output" / _path_slug(args.experiment_name)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.report_contract_id:
        if not args.plan_only:
            raise ValueError("--report-contract-id is report-only and requires --plan-only")
        stored_experiment_path = output_root / "experiment.json"
        stored_experiment = json.loads(stored_experiment_path.read_text(encoding="utf-8"))
        stored_contracts = stored_experiment.get("contracts") or {}
        if args.report_contract_id not in stored_contracts:
            raise ValueError(
                f"Unknown stored contract {args.report_contract_id!r}; "
                f"available={sorted(stored_contracts)}"
            )
        contract_id = str(args.report_contract_id)
        contract = dict(stored_contracts[contract_id])
    run_suffix = f"_{_path_slug(args.run_label, 32)}" if args.run_label else ""
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pid{os.getpid()}{run_suffix}"
    )
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    inventory_counts = Counter(item.task_type for item in catalog_candidates)
    selection_counts = Counter(item.task_type for item in selected)
    manifest = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_label": args.run_label,
        "created_at": datetime.now().isoformat(),
        "status": "planned" if args.plan_only else "running",
        "experiment_name": args.experiment_name,
        "output_root": str(output_root.resolve()),
        "evaluation_contract_id": contract_id,
        "evaluation_contract": contract,
        "categorical_only": bool(args.categorical_only),
        "requested_head_indices": sorted(requested_head_indices),
        "sample_selection_manifest": (
            str(exact_sample_manifest) if exact_sample_manifest else None
        ),
        "sample_selection_heads": len(exact_sample_selection),
        "head_source": args.head_source,
        "recommended_heads_config": (
            str(recommended_config.resolve())
            if args.head_source == "recommended" else None
        ),
        "selection_policy": args.selection_policy,
        "force": bool(args.force),
        "devices": [item.strip() for item in str(args.devices).split(",") if item.strip()],
        "log_roots": [str(path.resolve()) for path in log_roots],
        "selection_audit": selection_audit,
        "inventory_counts": dict(inventory_counts),
        "selection_counts": dict(selection_counts),
        "selection_target_counts": dict(Counter(item.target for item in selected)),
        "selection_default_compatible": sum(item.default_compatible for item in selected),
        "selection_training_sufficient": sum(
            item.trained_steps >= int(minimum_trained_steps.get(item.task_type, 0))
            and item.training_completion_ratio >= float(args.minimum_completion_ratio)
            for item in selected
        ),
        "catalogued": [asdict(item) for item in catalog_candidates],
        "selected": [asdict(item) for item in selected],
    }
    atomic_json_dump(manifest, run_dir / "manifest.json")

    experiment_path = output_root / "experiment.json"
    try:
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        experiment = {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "experiment_name": args.experiment_name,
            "created_at": datetime.now().isoformat(),
            "contracts": {},
        }
    experiment["updated_at"] = datetime.now().isoformat()
    experiment.setdefault("contracts", {})[contract_id] = contract
    experiment["latest_run_id"] = run_id
    atomic_json_dump(experiment, experiment_path)

    invocation_status: dict[str, dict[str, Any]] = {
        (
            hashlib.sha256(
                json.dumps(asdict(candidate), sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            if args.catalog_manifest else candidate_cache_key(candidate)
        ): {
            "head_index": int(candidate.head_index),
            "target": candidate.target,
            "status": "queued",
        }
        for candidate in selected
    }
    reports_lock = threading.Lock()

    def refresh_reports(final: bool = False) -> dict[str, Any]:
        if args.catalog_manifest:
            catalog, aggregate_results = collect_frozen_manifest_results(
                catalog_candidates, output_root, contract_id
            )
        else:
            catalog = build_catalog_entries(
                catalog_candidates, output_root, contract, contract_id
            )
            aggregate_results = collect_catalog_results(
                catalog_candidates, output_root, contract, contract_id
            )
        counts = Counter(entry["status"] for entry in catalog)
        progress = {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "experiment_name": args.experiment_name,
            "evaluation_contract_id": contract_id,
            "updated_at": datetime.now().isoformat(),
            "catalogued_heads": len(catalog),
            "complete_heads": int(counts.get("complete", 0)),
            "failed_heads": int(counts.get("failed", 0)),
            "pending_heads": int(counts.get("pending", 0)),
            "completion_fraction": (
                counts.get("complete", 0) / len(catalog) if catalog else 1.0
            ),
            "latest_run_id": run_id,
            "latest_run_final": bool(final),
            "latest_invocation": list(invocation_status.values()),
        }
        summary = summarize_results(aggregate_results)
        summary.update({
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "status": (
                "complete" if counts.get("complete", 0) == len(catalog)
                else "partial"
            ),
            "evaluation_contract_id": contract_id,
            "catalogued_heads": len(catalog),
            "complete_heads": progress["complete_heads"],
            "failed_heads": progress["failed_heads"],
            "pending_heads": progress["pending_heads"],
            "catalog": str((output_root / "catalog.json").resolve()),
            "progress": str((output_root / "progress.json").resolve()),
            "updated_at": progress["updated_at"],
        })
        atomic_json_dump({
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "evaluation_contract_id": contract_id,
            "entries": catalog,
        }, output_root / "catalog.json")
        atomic_json_dump(progress, output_root / "progress.json")
        atomic_json_dump(summary, output_root / "summary.json")
        atomic_json_dump({
            "run_id": run_id,
            "updated_at": progress["updated_at"],
            "final": bool(final),
            "items": list(invocation_status.values()),
        }, run_dir / "status.json")
        write_metric_tables(output_root, aggregate_results)
        write_experiment_readme(output_root, contract_id, progress)
        return summary

    print(
        f"[WSM eval] catalogued={len(catalog_candidates)} selected={len(selected)} "
        f"contract={contract_id} output={output_root}", flush=True,
    )
    print(f"[WSM eval] selection_by_task={dict(selection_counts)}", flush=True)
    with reports_lock:
        initial_summary = refresh_reports(final=bool(args.plan_only))
    if args.plan_only:
        manifest["status"] = "planned"
        manifest["completed_at"] = datetime.now().isoformat()
        atomic_json_dump(manifest, run_dir / "manifest.json")
        print(json.dumps(_json_ready(initial_summary), indent=2, ensure_ascii=False), flush=True)
        return

    devices = manifest["devices"]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device")
    work_queue: queue.Queue[HeadCandidate] = queue.Queue()
    for candidate in selected:
        work_queue.put(candidate)
    finished_results: list[dict[str, Any]] = []
    completed_since_report = 0

    def worker(device: str) -> None:
        nonlocal completed_since_report
        while True:
            try:
                candidate = work_queue.get_nowait()
            except queue.Empty:
                return
            key = candidate_cache_key(candidate)
            invocation_status[key]["status"] = "running"
            invocation_status[key]["device"] = str(device)
            invocation_status[key]["started_at"] = datetime.now().isoformat()
            print(
                f"[WSM eval] gpu={device} start head={candidate.head_index:06d} "
                f"task={candidate.task_type} {','.join(candidate.condition)}->{candidate.target}",
                flush=True,
            )
            try:
                result = evaluate_one(
                    candidate, device, output_root, contract, contract_id,
                    force=args.force,
                    preview_sample_id=exact_sample_selection.get(candidate.head_index, ""),
                )
            except Exception as error:
                result = {
                    "status": "failed", "cache_hit": False,
                    "elapsed_seconds": 0.0, "candidate": asdict(candidate),
                    "candidate_cache_key": key,
                    "evaluation_contract_id": contract_id,
                    "exception": repr(error), "samples": [],
                }
            with reports_lock:
                finished_results.append(result)
                invocation_status[key].update({
                    "status": result.get("status", "failed"),
                    "cache_hit": bool(result.get("cache_hit", False)),
                    "elapsed_seconds": float(result.get("elapsed_seconds") or 0.0),
                    "completed_at": datetime.now().isoformat(),
                    "worker_log": result.get("worker_log"),
                    "exception": result.get("exception"),
                })
                completed_since_report += 1
                should_report = (
                    completed_since_report >= max(1, int(args.report_every))
                    or len(finished_results) == len(selected)
                )
                if should_report:
                    refresh_reports(final=False)
                    completed_since_report = 0
                atomic_json_dump({
                    "run_id": run_id,
                    "updated_at": datetime.now().isoformat(),
                    "final": False,
                    "items": list(invocation_status.values()),
                }, run_dir / "status.json")
                progress_value = len(finished_results)
            cache_label = "cached" if result.get("cache_hit") else result.get("status")
            print(
                f"[WSM eval] gpu={device} {cache_label} head={candidate.head_index:06d} "
                f"elapsed={float(result.get('elapsed_seconds') or 0.0):.1f}s "
                f"progress={progress_value}/{len(selected)}",
                flush=True,
            )
            work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(device,), daemon=False)
        for device in devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with reports_lock:
        summary = refresh_reports(final=True)
    manifest["status"] = (
        "complete" if all(
            item.get("status") == "complete" for item in invocation_status.values()
        ) else "partial"
    )
    manifest["completed_at"] = datetime.now().isoformat()
    manifest["invocation_status"] = list(invocation_status.values())
    atomic_json_dump(manifest, run_dir / "manifest.json")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
