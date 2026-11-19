#!/usr/bin/env python3
"""Evaluate WSM and external baselines under one held-out-data contract.

This entry point is deliberately inference-framework agnostic.  WSM and every
official baseline run in their own environment and export a small JSONL index of
canonical predictions.  This script then enforces identical sample IDs and
computes metrics with one implementation.  Keeping inference separate avoids
changing a baseline's VAE, tokenizer, backbone, or dependency versions.

Config schema (paths are relative to the config file)::

  {
    "schema_version": 1,
    "benchmark_name": "wsm_external_comparison",
    "sample_manifest": "samples.jsonl",
    "workspace": "../../evaluation_output/external/wsm_external_comparison",
    "task_protocols": {
      "conditional_rgb": {"edge_threshold": 0.5, "compute_fid": false,
                          "compute_clip": false},
      "semantic_estimation": {"shared_class_ids": [0, 1, 2]},
      "any_to_any": {"depth_alignment": "scale_shift"},
      "depth_estimation": {"depth_alignment": "scale_shift",
                           "min_depth": 0.001, "max_depth": 80.0}
    },
    "methods": [
      {"id": "wsm", "display_name": "WSM", "enabled": true,
       "inference_regime": "zero_shot_official",
       "prediction_manifest": "predictions/wsm.jsonl",
       "checkpoint": "/absolute/checkpoint/path"}
    ]
  }

Sample JSONL records require ``sample_id``, ``task``, and ``target_path``.
Depending on the task they also carry ``condition_path``, ``mask_path``,
``prompt``, ``target_type``, ``dataset``, and ``split``.  Prediction JSONL
records require ``sample_id`` and ``prediction_path``.  Canonical prediction
formats are RGB images, 2-D label PNG/NPY, scalar depth PNG/NPY/NPZ, or RGB/NPY
surface normals.  NPZ records may specify ``array_key``.

Official paper numbers use ``inference_regime=reported_only`` plus a
``reported_metrics`` object.  They are written to a separate table and are
never ranked together with locally rerun predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
TASKS = frozenset({
    "conditional_rgb",
    "semantic_estimation",
    "any_to_any",
    "depth_estimation",
})
INFERENCE_REGIMES = frozenset({
    "zero_shot_official",
    "held_out_test",
    "trained_on_ours",
    "reported_only",
})

# Table 6 and Table 7 intentionally report only the metrics that make the
# cross-method comparison legible. Keep that exact public reproduction surface
# as the default; the diagnostic superset remains available through
# --extended-evaluation.
PAPER_METRICS: Mapping[tuple[str, str], frozenset[str]] = {
    ("conditional_rgb", "rgb"): frozenset({
        "edge_density_error", "edge_f0_5", "edge_precision",
        "mae", "mse", "psnr_db", "ssim",
    }),
    ("semantic_estimation", "semantic"): frozenset({
        "shared_pixel_accuracy", "shared_miou", "foreground_miou",
        "mean_class_accuracy", "shared_macro_f1",
    }),
    ("depth_estimation", "depth"): frozenset({
        "unit_mae", "unit_rmse", "unit_acc_005", "unit_acc_010",
        "unit_acc_020", "unit_psnr_db", "unit_pearson", "unit_spearman",
    }),
    ("any_to_any", "edge"): frozenset({
        "edge_accuracy", "edge_false_positive_rate", "edge_specificity",
        "edge_balanced_accuracy", "edge_f1", "edge_f2",
        "edge_false_negative_rate", "edge_iou", "edge_npv", "edge_recall",
    }),
    ("any_to_any", "normal"): frozenset({
        "normal_mean_deg", "normal_median_deg", "normal_within_11_25",
        "normal_within_22_5", "normal_within_30",
    }),
    ("any_to_any", "semantic"): frozenset({
        "shared_pixel_accuracy", "shared_miou", "foreground_miou",
        "mean_class_accuracy", "shared_macro_f1",
    }),
    ("any_to_any", "depth"): frozenset({
        "unit_mae", "unit_rmse", "unit_acc_005", "unit_acc_010",
        "unit_acc_020", "unit_psnr_db", "unit_pearson", "unit_spearman",
    }),
    ("any_to_any", "rgb"): frozenset({"mae", "mse", "psnr_db", "ssim"}),
}
PAPER_ROW_METADATA = frozenset({
    "method_id", "display_name", "inference_regime", "sample_id", "dataset",
    "split", "task", "target_type", "samples", "source",
})


class ContractError(RuntimeError):
    """Raised when an evaluation contract would permit an unfair comparison."""


@dataclass(frozen=True)
class Method:
    method_id: str
    display_name: str
    inference_regime: str
    tasks: frozenset[str]
    prediction_manifest: Path | None
    checkpoint: str
    metadata: Mapping[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text_dump(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ContractError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def finite_mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(numbers)) if numbers else None


def paper_metric_names(task: str, target_type: str = "") -> frozenset[str]:
    """Return the Table 6/7 metric contract for one directed task."""
    normalized_target = target_type.strip().lower()
    if task == "conditional_rgb":
        normalized_target = "rgb"
    elif task == "semantic_estimation":
        normalized_target = "semantic"
    elif task == "depth_estimation":
        normalized_target = "depth"
    metrics = PAPER_METRICS.get((task, normalized_target))
    if metrics is None:
        raise ContractError(
            f"No paper metric profile for task={task!r}, "
            f"target_type={normalized_target!r}"
        )
    return metrics


def select_metrics(
    metrics: Mapping[str, float], task: str, target_type: str,
    extended_evaluation: bool,
) -> dict[str, float]:
    """Select the paper-facing metrics or retain the diagnostic superset."""
    if extended_evaluation:
        return dict(metrics)
    allowed = paper_metric_names(task, target_type)
    return {key: value for key, value in metrics.items() if key in allowed}


def select_paper_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove diagnostic columns from an existing or newly computed row."""
    task = str(row.get("task") or "")
    target_type = str(row.get("target_type") or "")
    allowed = paper_metric_names(task, target_type)
    return {
        key: value for key, value in row.items()
        if key in PAPER_ROW_METADATA or key in allowed
    }


def bootstrap_mean_ci(
    values: Iterable[Any], seed: int, repeats: int = 10000
) -> tuple[float | None, float | None]:
    numbers = np.asarray([
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ], dtype=np.float64)
    if numbers.size < 2 or repeats < 1:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    chunk = 1000
    for start in range(0, repeats, chunk):
        stop = min(start + chunk, repeats)
        indices = rng.integers(0, numbers.size, size=(stop - start, numbers.size))
        means[start:stop] = numbers[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ContractError(f"Missing {label}: {path}")


def load_array(path: Path, array_key: str = "") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            keys = list(payload.files)
            key = array_key or (keys[0] if len(keys) == 1 else "")
            if not key or key not in payload:
                raise ContractError(
                    f"{path}: NPZ needs array_key; available keys={keys}"
                )
            return np.asarray(payload[key])
    with Image.open(path) as image:
        return np.asarray(image)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def _inverse_matplotlib_colormap(values: np.ndarray, name: str) -> np.ndarray:
    """Recover a scalar [0,1] field from a quantized Matplotlib colormap."""
    try:
        from matplotlib import colormaps
    except ImportError as exc:
        raise ContractError(
            f"Decoding {name} depth requires matplotlib"
        ) from exc
    rgb = np.asarray(values[..., :3], dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    lut = np.asarray(colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3], dtype=np.float32)
    flat = rgb.reshape(-1, 3)
    decoded = np.empty(flat.shape[0], dtype=np.float32)
    lut_norm = np.sum(lut * lut, axis=1)[None, :]
    for start in range(0, flat.shape[0], 65536):
        chunk = flat[start:start + 65536]
        distances = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + lut_norm
            - 2.0 * chunk @ lut.T
        )
        decoded[start:start + len(chunk)] = np.argmin(distances, axis=1) / 255.0
    return decoded.reshape(rgb.shape[:2]).astype(np.float64)


def load_scalar(
    path: Path, array_key: str = "", scale: float = 1.0, encoding: str = "auto"
) -> np.ndarray:
    raw = np.asarray(load_array(path, array_key)).squeeze()
    mode = str(encoding or "auto").lower()
    if raw.ndim == 3 and raw.shape[-1] in (3, 4):
        if mode in {"spectral", "matplotlib_spectral"}:
            values = _inverse_matplotlib_colormap(raw, "Spectral")
        elif mode in {"spectral_r", "matplotlib_spectral_r"}:
            values = _inverse_matplotlib_colormap(raw, "Spectral_r")
        else:
            values = np.asarray(raw[..., :3], dtype=np.float64).mean(axis=-1)
            if np.issubdtype(raw.dtype, np.integer):
                values /= float(np.iinfo(raw.dtype).max)
    else:
        values = np.asarray(raw, dtype=np.float64)
        if path.suffix.lower() not in {".npy", ".npz"} and np.issubdtype(raw.dtype, np.integer):
            values /= float(np.iinfo(raw.dtype).max)
    if values.ndim != 2:
        raise ContractError(f"Expected scalar HxW array at {path}, got {values.shape}")
    return values * float(scale)


def load_labels(
    path: Path, array_key: str = "", palette_rgb: Sequence[Sequence[int]] | None = None
) -> np.ndarray:
    values = np.asarray(load_array(path, array_key)).squeeze()
    if values.ndim == 3:
        if values.shape[-1] == 4:
            values = values[..., :3]
        if values.shape[-1] != 3 or not palette_rgb:
            raise ContractError(
                f"Expected canonical class IDs, got RGB data at {path}; provide "
                "class_palette_rgb or convert the palette in the adapter"
            )
        palette = np.asarray(palette_rgb, dtype=np.float32)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ContractError("class_palette_rgb must be a list of RGB triplets")
        pixels = values.astype(np.float32)[..., None, :]
        return np.argmin(
            np.sum(np.square(pixels - palette), axis=-1), axis=-1
        ).astype(np.int64)
    if values.ndim != 2:
        raise ContractError(f"Expected label HxW array at {path}, got {values.shape}")
    return values.astype(np.int64, copy=False)


def resize_array(values: np.ndarray, output_hw: tuple[int, int], nearest: bool) -> np.ndarray:
    height, width = output_hw
    if values.shape[:2] == (height, width):
        return values
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if values.ndim == 2:
        image = Image.fromarray(values.astype(np.float32))
        return np.asarray(image.resize((width, height), resampling), dtype=values.dtype)
    image = Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(image.resize((width, height), resampling), dtype=np.float32) / 255.0


def resize_vector_field(values: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    """Resize an HxWxC floating-point field without quantizing its components."""
    height, width = output_hw
    values = np.asarray(values, dtype=np.float64)
    if values.shape[:2] == (height, width):
        return values
    channels = [
        np.asarray(
            Image.fromarray(values[..., channel].astype(np.float32)).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        for channel in range(values.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def valid_mask_for(sample: Mapping[str, Any], base: Path, shape: tuple[int, int]) -> np.ndarray:
    mask_path = sample.get("mask_path")
    if not mask_path:
        return np.ones(shape, dtype=bool)
    mask = load_scalar(resolve_path(mask_path, base), str(sample.get("mask_array_key") or ""))
    mask = resize_array(mask, shape, nearest=True)
    return np.isfinite(mask) & (mask > float(sample.get("mask_threshold", 0.5)))


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = resize_array(prediction, target.shape[:2], nearest=False)
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    mu_x, mu_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    covariance = float(((x - mu_x) * (y - mu_y)).mean())
    c1, c2 = 0.01**2, 0.03**2
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    ssim = 1.0 if denominator == 0 else (
        (2 * mu_x * mu_y + c1) * (2 * covariance + c2) / denominator
    )
    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": float(ssim)}


def _rgb_to_edge(rgb: np.ndarray, protocol: Mapping[str, Any]) -> np.ndarray:
    detector = str(protocol.get("edge_detector", "canny"))
    gray = np.clip(
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 1
    )
    if detector == "canny":
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ContractError(
                "edge_detector=canny requires opencv-python in the evaluation environment"
            ) from exc
        low = int(protocol.get("canny_low", 100))
        high = int(protocol.get("canny_high", 200))
        return cv2.Canny((gray * 255).astype(np.uint8), low, high) > 0
    if detector == "sobel":
        gy, gx = np.gradient(gray.astype(np.float64))
        magnitude = np.sqrt(gx * gx + gy * gy)
        threshold = float(protocol.get("sobel_threshold", 0.12))
        return magnitude >= threshold
    raise ContractError(f"Unsupported edge_detector={detector!r}")


def binary_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)[mask]
    target = target.astype(bool)[mask]
    tp = int(np.logical_and(prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    npv = tn / (tn + fn) if tn + fn else 1.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    def f_beta(beta: float) -> float:
        beta_squared = beta * beta
        denominator = beta_squared * precision + recall
        return (
            (1.0 + beta_squared) * precision * recall / denominator
            if denominator else 0.0
        )

    return {
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "edge_f0_5": float(f_beta(0.5)),
        "edge_f2": float(f_beta(2.0)),
        "edge_iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "edge_accuracy": float(accuracy),
        "edge_specificity": float(specificity),
        "edge_npv": float(npv),
        "edge_balanced_accuracy": float(0.5 * (recall + specificity)),
        "edge_mcc": float((tp * tn - fp * fn) / mcc_denominator) if mcc_denominator else 0.0,
        "edge_false_positive_rate": float(1.0 - specificity),
        "edge_false_negative_rate": float(1.0 - recall),
        "edge_prediction_density": float(prediction.mean()),
        "edge_target_density": float(target.mean()),
        "edge_density_error": float(abs(prediction.mean() - target.mean())),
    }


def conditional_rgb_metrics(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], base: Path,
    protocol: Mapping[str, Any]
) -> dict[str, float]:
    pred_rgb = load_rgb(resolve_path(prediction["prediction_path"], base))
    target_rgb = load_rgb(resolve_path(sample["target_path"], base))
    result = image_metrics(pred_rgb, target_rgb)
    condition_path = sample.get("condition_path")
    if condition_path and str(sample.get("condition_type", "edge")) == "edge":
        condition = load_scalar(
            resolve_path(condition_path, base), str(sample.get("condition_array_key") or "")
        )
        condition = resize_array(condition, target_rgb.shape[:2], nearest=True)
        condition_edge = condition > float(protocol.get("edge_threshold", 0.5))
        pred_edge = _rgb_to_edge(
            resize_array(pred_rgb, target_rgb.shape[:2], nearest=False), protocol
        )
        mask = valid_mask_for(sample, base, target_rgb.shape[:2])
        result.update(binary_metrics(pred_edge, condition_edge, mask))
    return result


def semantic_boundary_metrics(
    target: np.ndarray, prediction: np.ndarray, valid: np.ndarray, tolerance: int
) -> dict[str, float]:
    def boundaries(labels: np.ndarray) -> np.ndarray:
        result = np.zeros(labels.shape, dtype=bool)
        horizontal = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
        vertical = valid[1:, :] & valid[:-1, :] & (labels[1:, :] != labels[:-1, :])
        result[:, 1:] |= horizontal
        result[:, :-1] |= horizontal
        result[1:, :] |= vertical
        result[:-1, :] |= vertical
        return result & valid

    target_boundary = boundaries(target)
    prediction_boundary = boundaries(prediction)
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ContractError("semantic boundary metrics require opencv-python") from exc
    radius = max(0, int(tolerance))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    target_dilated = cv2.dilate(target_boundary.astype(np.uint8), kernel) > 0
    prediction_dilated = cv2.dilate(prediction_boundary.astype(np.uint8), kernel) > 0
    precision = (
        float(target_dilated[prediction_boundary].mean())
        if prediction_boundary.any() else float(not target_boundary.any())
    )
    recall = (
        float(prediction_dilated[target_boundary].mean())
        if target_boundary.any() else float(not prediction_boundary.any())
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": float(f1),
    }


def semantic_metrics(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], base: Path,
    protocol: Mapping[str, Any], extended_evaluation: bool = True,
) -> dict[str, float]:
    expected_channel = str(protocol.get("wsm_target_channel") or "").strip()
    sample_channel = str(sample.get("wsm_target_channel") or "").strip()
    if expected_channel and sample_channel != expected_channel:
        raise ContractError(
            f"{sample['sample_id']}: semantic GT channel {sample_channel!r} does "
            f"not match the frozen evaluation channel {expected_channel!r}; map the "
            "dataset ontology explicitly before evaluation"
        )
    class_ids = [int(item) for item in protocol.get("shared_class_ids", [])]
    if not class_ids:
        raise ContractError(
            "semantic_estimation requires non-empty shared_class_ids; do not report "
            "mIoU when the WSM/baseline/dataset ontology intersection is undefined"
        )
    palette = protocol.get("class_palette_rgb")
    target = load_labels(
        resolve_path(sample["target_path"], base),
        str(sample.get("target_array_key") or ""),
        palette,
    )
    pred = load_labels(
        resolve_path(prediction["prediction_path"], base),
        str(prediction.get("array_key") or ""),
        palette,
    )
    pred = resize_array(pred, target.shape, nearest=True).astype(np.int64)
    mask = valid_mask_for(sample, base, target.shape)
    ignore_id = protocol.get("ignore_id")
    if ignore_id is not None:
        mask &= target != int(ignore_id)
    mask &= np.isin(target, class_ids)
    if not mask.any():
        raise ContractError(f"{sample['sample_id']}: no valid shared-class pixels")
    boundary_result = (
        semantic_boundary_metrics(
            target, pred, mask, int(protocol.get("boundary_tolerance_px", 2))
        ) if extended_evaluation else {}
    )
    target, pred = target[mask], pred[mask]
    ious, foreground_ious, f1s, class_accuracies = [], [], [], []
    per_class: dict[str, float] = {}
    correct = int((target == pred).sum())
    for class_id in class_ids:
        gt_class = target == class_id
        pred_class = pred == class_id
        tp = int(np.logical_and(gt_class, pred_class).sum())
        fp = int(np.logical_and(~gt_class, pred_class).sum())
        fn = int(np.logical_and(gt_class, ~pred_class).sum())
        if tp + fp + fn:
            iou = tp / (tp + fp + fn)
            ious.append(iou)
            per_class[f"iou_class_{class_id}"] = float(iou)
            if class_id != 0:
                foreground_ious.append(iou)
        if 2 * tp + fp + fn:
            f1 = 2 * tp / (2 * tp + fp + fn)
            f1s.append(f1)
            per_class[f"f1_class_{class_id}"] = float(f1)
        if tp + fn:
            accuracy = tp / (tp + fn)
            class_accuracies.append(accuracy)
            per_class[f"accuracy_class_{class_id}"] = float(accuracy)
    result = {
        "shared_pixel_accuracy": float(correct / target.size),
        "shared_miou": float(np.mean(ious)) if ious else float("nan"),
        "foreground_miou": (
            float(np.mean(foreground_ious)) if foreground_ious else float("nan")
        ),
        "mean_class_accuracy": (
            float(np.mean(class_accuracies)) if class_accuracies else float("nan")
        ),
        "shared_macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "shared_valid_pixels": float(target.size),
        "shared_class_count": float(len(class_ids)),
    }
    result.update(boundary_result)
    if extended_evaluation:
        result.update(per_class)
    return result


def align_depth(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, mode: str
) -> tuple[np.ndarray, dict[str, float]]:
    pred_valid = prediction[valid].astype(np.float64)
    target_valid = target[valid].astype(np.float64)
    if mode == "none":
        return prediction.astype(np.float64), {"depth_scale": 1.0, "depth_shift": 0.0}
    if mode == "median_scale":
        median = float(np.median(pred_valid))
        if abs(median) < 1e-12:
            raise ContractError("Cannot median-align a zero-median depth prediction")
        scale = float(np.median(target_valid) / median)
        shift = 0.0
    elif mode == "scale_shift":
        design = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
        scale, shift = np.linalg.lstsq(design, target_valid, rcond=None)[0]
        scale, shift = float(scale), float(shift)
    else:
        raise ContractError(f"Unsupported depth_alignment={mode!r}")
    return prediction.astype(np.float64) * scale + shift, {
        "depth_scale": scale, "depth_shift": shift
    }


def _depth_core_metrics(
    target_valid: np.ndarray,
    pred_valid: np.ndarray,
) -> dict[str, float]:
    """Metrics shared by the primary and supplementary depth protocols."""
    difference = pred_valid - target_valid
    log_difference = np.log(pred_valid) - np.log(target_valid)
    ratio = np.maximum(target_valid / pred_valid, pred_valid / target_valid)
    return {
        "abs_rel": float(np.mean(np.abs(difference) / target_valid)),
        "sq_rel": float(np.mean(np.square(difference) / target_valid)),
        "mae_depth": float(np.mean(np.abs(difference))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "log_rmse": float(np.sqrt(np.mean(np.square(log_difference)))),
        "silog": float(
            100.0 * np.sqrt(max(
                0.0,
                float(np.mean(np.square(log_difference)) - np.mean(log_difference) ** 2),
            ))
        ),
        "log10": float(np.mean(np.abs(np.log10(pred_valid) - np.log10(target_valid)))),
    }


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    """Return scipy-compatible average ranks without adding a scipy dependency."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def _robust_unit_depth(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map one depth field to [0, 1] using frozen robust percentiles."""
    lower, upper = np.percentile(values, [low, high])
    span = float(upper - lower)
    if not span > 1e-12:
        raise ContractError("Cannot normalize a constant depth prediction")
    return np.clip((values.astype(np.float64) - float(lower)) / span, 0.0, 1.0)


def depth_metrics(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], base: Path,
    protocol: Mapping[str, Any], extended_evaluation: bool = True,
) -> dict[str, float]:
    target = load_scalar(
        resolve_path(sample["target_path"], base),
        str(sample.get("target_array_key") or ""),
        float(sample.get("target_scale", 1.0)),
        str(sample.get("target_scalar_encoding") or "auto"),
    )
    pred = load_scalar(
        resolve_path(prediction["prediction_path"], base),
        str(prediction.get("array_key") or ""),
        float(prediction.get("prediction_scale", 1.0)),
        str(prediction.get("scalar_encoding") or "auto"),
    )
    pred = resize_array(pred, target.shape, nearest=False).astype(np.float64)
    minimum = float(protocol.get("min_depth", 1e-3))
    maximum = float(protocol.get("max_depth", float("inf")))
    valid = valid_mask_for(sample, base, target.shape)
    valid &= np.isfinite(target) & np.isfinite(pred) & (target > minimum) & (target < maximum)
    if not valid.any():
        raise ContractError(f"{sample['sample_id']}: no valid depth pixels")
    target_valid = target[valid]
    result: dict[str, float] = {}
    if extended_evaluation:
        aligned, alignment = align_depth(
            pred, target, valid, str(protocol.get("depth_alignment", "scale_shift"))
        )
        aligned = np.clip(aligned, minimum, maximum)
        pred_valid = aligned[valid]
        result = _depth_core_metrics(target_valid, pred_valid)
        result["valid_depth_pixels"] = float(target_valid.size)

        # Supplementary correlations use the affine-aligned metric-depth field.
        stride = max(1, target_valid.size // 100_000)
        target_rank_values = target_valid[::stride]
        pred_rank_values = pred_valid[::stride]
        result["pearson_depth"] = _correlation(target_rank_values, pred_rank_values)
        result["spearman_depth"] = _correlation(
            _average_tie_ranks(target_rank_values),
            _average_tie_ranks(pred_rank_values),
        )

    # Cap deterministic rank calculations for megapixel predictions.
    stride = max(1, target_valid.size // 100_000)

    # This unit-range protocol compares relative scene structure without
    # assigning metric units to a relative-depth model. Its names stay distinct
    # from AbsRel/delta metrics, which are defined on positive metric depths.
    norm_low = float(protocol.get("normalization_low_percentile", 2.0))
    norm_high = float(protocol.get("normalization_high_percentile", 98.0))
    if not 0.0 <= norm_low < norm_high <= 100.0:
        raise ContractError(
            "Depth normalization percentiles must satisfy "
            "0 <= low < high <= 100"
        )
    try:
        target_unit = _robust_unit_depth(target_valid, norm_low, norm_high)
        pred_unit = _robust_unit_depth(pred[valid], norm_low, norm_high)
        # Some official relative-depth decoders use the opposite near/far
        # intensity convention. The adapter/evaluation recipe must declare
        # this per method; never infer it from each sample's ground truth.
        if bool(prediction.get("invert_unit_depth", False)):
            pred_unit = 1.0 - pred_unit
        unit_error = np.abs(pred_unit - target_unit)
        unit_mse = float(np.mean(np.square(unit_error)))
        result.update({
            "unit_mae": float(np.mean(unit_error)),
            "unit_rmse": float(np.sqrt(unit_mse)),
            "unit_psnr_db": (
                float("inf") if unit_mse == 0
                else float(10.0 * math.log10(1.0 / unit_mse))
            ),
            "unit_acc_005": float(np.mean(unit_error < 0.05)),
            "unit_acc_010": float(np.mean(unit_error < 0.10)),
            "unit_acc_020": float(np.mean(unit_error < 0.20)),
            "unit_pearson": _correlation(target_unit[::stride], pred_unit[::stride]),
            "unit_spearman": _correlation(
                _average_tie_ranks(target_unit[::stride]),
                _average_tie_ranks(pred_unit[::stride]),
            ),
        })
    except ContractError:
        result.update({
            key: float("nan") for key in (
                "unit_mae", "unit_rmse", "unit_acc_005", "unit_acc_010",
                "unit_acc_020", "unit_psnr_db", "unit_pearson", "unit_spearman",
            )
        })

    if not extended_evaluation:
        return result
    # Preserve two supplementary protocols in every row. The primary unprefixed
    # metrics remain the configured affine-invariant protocol used for ranking.
    for prefix, mode in (("median", "median_scale"), ("raw_metric", "none")):
        try:
            alternate, _ = align_depth(pred, target, valid, mode)
            alternate = np.clip(alternate, minimum, maximum)[valid]
            alternate_metrics = _depth_core_metrics(target_valid, alternate)
        except ContractError:
            # A constant-zero relative prediction has no defined median scale.
            # Keep the valid primary affine result and mark only this optional
            # supplementary protocol as unavailable.
            alternate_metrics = {
                key: float("nan")
                for key in _depth_core_metrics(target_valid, target_valid)
            }
        result.update({
            f"{prefix}_{key}": value for key, value in alternate_metrics.items()
        })
    result.update(alignment)
    return result


def normal_metrics(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], base: Path,
    protocol: Mapping[str, Any]
) -> dict[str, float]:
    target_path = resolve_path(sample["target_path"], base)
    pred_path = resolve_path(prediction["prediction_path"], base)
    target = load_array(target_path, str(sample.get("target_array_key") or ""))
    pred = load_array(pred_path, str(prediction.get("array_key") or ""))
    target, pred = np.asarray(target, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if target.ndim != 3 or target.shape[-1] < 3 or pred.ndim != 3 or pred.shape[-1] < 3:
        raise ContractError("Surface normals must be HxWx3 RGB/NPY arrays")
    target, pred = target[..., :3], pred[..., :3]

    def decode(
        values: np.ndarray, encoding: str, ranges: Any, source: str
    ) -> np.ndarray:
        mode = str(encoding or "rgb_01").lower()
        values = np.asarray(values, dtype=np.float64)
        if mode in {"xyz", "physical_xyz"}:
            return values
        if values.max() > 1.5:
            values = values / 255.0
        values = np.clip(values, 0.0, 1.0)
        if mode == "rgb_01":
            return values * 2.0 - 1.0
        if mode in {"wsg_linear_01", "wsg_log_01"}:
            channel_ranges = np.asarray(ranges, dtype=np.float64)
            if channel_ranges.shape != (3, 2):
                raise ContractError(
                    f"{source}: {mode} requires three [minimum, maximum] "
                    "normal_channel_ranges"
                )
            low = channel_ranges[:, 0]
            span = channel_ranges[:, 1] - channel_ranges[:, 0]
            if np.any(span <= 0):
                raise ContractError(f"{source}: normal_channel_ranges must be increasing")
            if mode == "wsg_log_01":
                return np.expm1(values * np.log1p(span)) + low
            return values * span + low
        raise ContractError(f"{source}: unsupported normal encoding {mode!r}")

    default_encoding = str(protocol.get("normal_encoding", "rgb_01"))
    default_ranges = protocol.get("normal_channel_ranges")
    target = decode(
        target,
        str(sample.get("target_normal_encoding") or default_encoding),
        sample.get("target_normal_channel_ranges", default_ranges),
        f"{sample['sample_id']} target",
    )
    pred = decode(
        pred,
        str(prediction.get("normal_encoding") or default_encoding),
        prediction.get("normal_channel_ranges", default_ranges),
        f"{sample['sample_id']} prediction",
    )
    pred = resize_vector_field(pred, target.shape[:2])
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    target /= np.maximum(target_norm, 1e-8)
    pred /= np.maximum(pred_norm, 1e-8)
    valid = valid_mask_for(sample, base, target.shape[:2])
    valid &= (
        np.isfinite(target).all(axis=-1)
        & np.isfinite(pred).all(axis=-1)
        & (target_norm[..., 0] > 1e-8)
        & (pred_norm[..., 0] > 1e-8)
    )
    if not valid.any():
        raise ContractError(f"{sample['sample_id']}: no valid normal pixels")
    cosine = np.clip(np.sum(target * pred, axis=-1)[valid], -1, 1)
    angles = np.degrees(np.arccos(cosine))
    return {
        "normal_cosine_similarity": float(np.mean(cosine)),
        "normal_rmse_xyz": float(np.sqrt(np.mean(np.square(target[valid] - pred[valid])))),
        "normal_mean_deg": float(np.mean(angles)),
        "normal_rmse_deg": float(np.sqrt(np.mean(np.square(angles)))),
        "normal_median_deg": float(np.median(angles)),
        "normal_p90_deg": float(np.percentile(angles, 90.0)),
        "normal_within_11_25": float(np.mean(angles < 11.25)),
        "normal_within_22_5": float(np.mean(angles < 22.5)),
        "normal_within_30": float(np.mean(angles < 30.0)),
    }


def any_to_any_metrics(
    sample: Mapping[str, Any], prediction: Mapping[str, Any], base: Path,
    protocol: Mapping[str, Any], extended_evaluation: bool = True,
) -> dict[str, float]:
    target_type = str(sample.get("target_type") or "")
    if target_type == "edge":
        target = load_scalar(resolve_path(sample["target_path"], base))
        pred = load_scalar(resolve_path(prediction["prediction_path"], base))
        pred = resize_array(pred, target.shape, nearest=True)
        valid = valid_mask_for(sample, base, target.shape)
        threshold = float(protocol.get("edge_threshold", 0.5))
        return binary_metrics(pred > threshold, target > threshold, valid)
    if target_type == "depth":
        return depth_metrics(sample, prediction, base, protocol, extended_evaluation)
    if target_type == "normal":
        return normal_metrics(sample, prediction, base, protocol)
    if target_type == "semantic":
        return semantic_metrics(sample, prediction, base, protocol, extended_evaluation)
    if target_type == "rgb":
        return image_metrics(
            load_rgb(resolve_path(prediction["prediction_path"], base)),
            load_rgb(resolve_path(sample["target_path"], base)),
        )
    raise ContractError(
        f"{sample['sample_id']}: any_to_any needs target_type in "
        "edge/depth/normal/semantic/rgb"
    )


def optional_distribution_metrics(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], base: Path,
    protocol: Mapping[str, Any]
) -> dict[str, float]:
    result: dict[str, float] = {}
    if protocol.get("compute_fid"):
        try:
            import torch
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError as exc:
            raise ContractError(
                "compute_fid=true requires torchmetrics[image] and torch-fidelity"
            ) from exc
        metric = FrechetInceptionDistance(feature=2048, normalize=True)
        for samples in (pairs[index:index + 16] for index in range(0, len(pairs), 16)):
            real = np.stack([load_rgb(resolve_path(s["target_path"], base)) for s, _ in samples])
            fake = np.stack([load_rgb(resolve_path(p["prediction_path"], base)) for _, p in samples])
            if real.shape[1:3] != fake.shape[1:3]:
                fake = np.stack([resize_array(item, real.shape[1:3], nearest=False) for item in fake])
            real_tensor = torch.from_numpy(real).permute(0, 3, 1, 2)
            fake_tensor = torch.from_numpy(fake).permute(0, 3, 1, 2)
            metric.update(real_tensor, real=True)
            metric.update(fake_tensor, real=False)
        result["fid"] = float(metric.compute().item())
    if protocol.get("compute_clip"):
        try:
            import torch
            from torchmetrics.multimodal.clip_score import CLIPScore
        except ImportError as exc:
            raise ContractError("compute_clip=true requires torchmetrics[multimodal]") from exc
        model_name = str(protocol.get("clip_model", "openai/clip-vit-base-patch32"))
        metric = CLIPScore(model_name_or_path=model_name)
        scores = []
        for sample, prediction in pairs:
            prompt = str(sample.get("prompt") or "").strip()
            if not prompt:
                raise ContractError(f"{sample['sample_id']}: CLIP score requires prompt")
            rgb = load_rgb(resolve_path(prediction["prediction_path"], base))
            tensor = torch.from_numpy((rgb * 255).astype(np.uint8)).permute(2, 0, 1)
            scores.append(float(metric(tensor, prompt).item()))
        result["clip_score"] = float(np.mean(scores))
    return result


def _method_from_config(raw: Mapping[str, Any], base: Path) -> Method | None:
    if not bool(raw.get("enabled", True)):
        return None
    method_id = str(raw.get("id") or "").strip()
    regime = str(raw.get("inference_regime") or "").strip()
    if not method_id:
        raise ContractError("Every enabled method needs a non-empty id")
    if regime not in INFERENCE_REGIMES:
        raise ContractError(f"{method_id}: invalid inference_regime={regime!r}")
    manifest = raw.get("prediction_manifest")
    prediction_manifest = resolve_path(manifest, base) if manifest else None
    if regime != "reported_only" and prediction_manifest is None:
        raise ContractError(f"{method_id}: prediction_manifest is required")
    if regime == "reported_only" and not raw.get("reported_metrics"):
        raise ContractError(f"{method_id}: reported_only requires reported_metrics")
    raw_tasks = raw.get("tasks")
    tasks = frozenset(str(item) for item in raw_tasks) if raw_tasks else TASKS
    unknown_tasks = tasks - TASKS
    if unknown_tasks:
        raise ContractError(f"{method_id}: unsupported tasks={sorted(unknown_tasks)}")
    return Method(
        method_id=method_id,
        display_name=str(raw.get("display_name") or method_id),
        inference_regime=regime,
        tasks=tasks,
        prediction_manifest=prediction_manifest,
        checkpoint=str(raw.get("checkpoint") or ""),
        metadata=raw,
    )


def validate_contract(
    config_path: Path, config: Mapping[str, Any], selected_methods: set[str],
    selected_tasks: set[str], allow_missing: bool
) -> tuple[Path, Path, list[dict[str, Any]], list[Method], dict[str, Any]]:
    base = config_path.parent.resolve()
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ContractError(
            f"schema_version must be {SCHEMA_VERSION}, got {config.get('schema_version')}"
        )
    sample_manifest = resolve_path(config.get("sample_manifest", ""), base)
    _require_file(sample_manifest, "sample manifest")
    workspace = resolve_path(config.get("workspace", "external_baseline_evaluation"), base)
    samples = read_jsonl(sample_manifest)
    seen: set[str] = set()
    filtered_samples = []
    for record in samples:
        sample_id = str(record.get("sample_id") or "")
        task = str(record.get("task") or "")
        if not sample_id or sample_id in seen:
            raise ContractError(f"Duplicate or empty sample_id={sample_id!r}")
        seen.add(sample_id)
        if task not in TASKS:
            raise ContractError(f"{sample_id}: unsupported task={task!r}")
        if selected_tasks and task not in selected_tasks:
            continue
        if not record.get("target_path"):
            raise ContractError(f"{sample_id}: target_path is required")
        _require_file(resolve_path(record["target_path"], base), f"target for {sample_id}")
        filtered_samples.append(record)
    protocols = dict(config.get("task_protocols") or {})
    for task in {str(item["task"]) for item in filtered_samples}:
        if task not in protocols:
            raise ContractError(f"Missing task_protocols.{task}")
    methods = []
    for raw in config.get("methods") or []:
        method = _method_from_config(raw, base)
        if method and (not selected_methods or method.method_id in selected_methods):
            methods.append(method)
    if not methods:
        raise ContractError("No enabled/selected methods")
    coverage: dict[str, Any] = {}
    for method in methods:
        if method.inference_regime == "reported_only":
            coverage[method.method_id] = {"reported_only": True}
            continue
        assert method.prediction_manifest is not None
        _require_file(method.prediction_manifest, f"prediction manifest for {method.method_id}")
        predictions = read_jsonl(method.prediction_manifest)
        prediction_ids = [str(item.get("sample_id") or "") for item in predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ContractError(f"{method.method_id}: duplicate prediction sample IDs")
        required_ids = {
            str(item["sample_id"]) for item in filtered_samples
            if str(item["task"]) in method.tasks
        }
        missing = sorted(required_ids - set(prediction_ids))
        extra = sorted(set(prediction_ids) - required_ids)
        coverage[method.method_id] = {
            "tasks": sorted(method.tasks),
            "required": len(required_ids),
            "present": len(required_ids) - len(missing),
            "missing": missing, "extra": extra,
        }
        if missing and not allow_missing:
            raise ContractError(
                f"{method.method_id}: missing {len(missing)} predictions; use "
                "--allow-missing only for debugging, never for a paper table"
            )
    return base, workspace, filtered_samples, methods, {
        "sample_manifest": str(sample_manifest),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "coverage": coverage,
    }


def evaluate_method(
    method: Method, samples: Sequence[Mapping[str, Any]], base: Path,
    protocols: Mapping[str, Any], allow_missing: bool,
    extended_evaluation: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if method.inference_regime == "reported_only":
        reported = []
        for task, metrics in (method.metadata.get("reported_metrics") or {}).items():
            metric_values = dict(metrics)
            target_type = str(metric_values.pop("target_type", ""))
            metric_values = select_metrics(
                metric_values, str(task), target_type, extended_evaluation
            )
            reported.append({
                "method_id": method.method_id,
                "display_name": method.display_name,
                "inference_regime": method.inference_regime,
                "task": task,
                "target_type": target_type,
                "source": str(method.metadata.get("reported_source") or ""),
                **metric_values,
            })
        return [], reported
    assert method.prediction_manifest is not None
    samples = [sample for sample in samples if str(sample["task"]) in method.tasks]
    predictions = {
        str(item["sample_id"]): item for item in read_jsonl(method.prediction_manifest)
    }
    rows: list[dict[str, Any]] = []
    pairs_by_task: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for sample in samples:
        sample_id, task = str(sample["sample_id"]), str(sample["task"])
        target_type = str(sample.get("target_type") or "")
        prediction = predictions.get(sample_id)
        if prediction is None:
            if allow_missing:
                continue
            raise ContractError(f"{method.method_id}: missing prediction {sample_id}")
        prediction_path = resolve_path(prediction["prediction_path"], base)
        _require_file(prediction_path, f"prediction for {method.method_id}/{sample_id}")
        protocol = dict(protocols[task])
        if task == "conditional_rgb":
            metrics = conditional_rgb_metrics(sample, prediction, base, protocol)
        elif task == "semantic_estimation":
            metrics = semantic_metrics(sample, prediction, base, protocol, extended_evaluation)
        elif task == "depth_estimation":
            metrics = depth_metrics(sample, prediction, base, protocol, extended_evaluation)
        else:
            metrics = any_to_any_metrics(sample, prediction, base, protocol, extended_evaluation)
        metrics = select_metrics(
            metrics, task, target_type, extended_evaluation
        )
        rows.append({
            "method_id": method.method_id,
            "display_name": method.display_name,
            "inference_regime": method.inference_regime,
            "sample_id": sample_id,
            "dataset": sample.get("dataset_id", sample.get("dataset", "")),
            "split": sample.get("dataset_split", sample.get("split", "")),
            "task": task,
            "target_type": target_type,
            **metrics,
        })
        pairs_by_task[task].append((sample, prediction))
    aggregates = []
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row["task"]), str(row.get("target_type") or ""))].append(row)
    for (task, target_type), task_rows in grouped_rows.items():
        metric_keys = sorted({key for row in task_rows for key in row} - {
            "method_id", "display_name", "inference_regime", "sample_id",
            "dataset", "split", "task", "target_type",
        })
        aggregate = {
            "method_id": method.method_id,
            "display_name": method.display_name,
            "inference_regime": method.inference_regime,
            "task": task,
            "target_type": target_type,
            "samples": len(task_rows),
        }
        aggregate.update({
            key: finite_mean(row.get(key) for row in task_rows)
            for key in metric_keys
        })
        if extended_evaluation:
            repeats = int(protocols[task].get("bootstrap_samples", 10000))
            for metric_key in metric_keys:
                seed_bytes = hashlib.sha256(
                    f"{method.method_id}:{task}:{target_type}:{metric_key}".encode("utf-8")
                ).digest()
                lower, upper = bootstrap_mean_ci(
                    (row.get(metric_key) for row in task_rows),
                    int.from_bytes(seed_bytes[:8], "big"), repeats,
                )
                aggregate[f"{metric_key}_ci95_low"] = lower
                aggregate[f"{metric_key}_ci95_high"] = upper
        if extended_evaluation and task == "conditional_rgb":
            aggregate.update(optional_distribution_metrics(pairs_by_task[task], base, protocols[task]))
        aggregates.append(aggregate)
    return rows, aggregates


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "")].append(row)
    return dict(grouped)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                     ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_latex(aggregates: Sequence[Mapping[str, Any]], path: Path) -> None:
    preferred = {
        "conditional_rgb": ("edge_density_error", "mae", "psnr_db"),
        "semantic_estimation": ("shared_miou", "shared_macro_f1", "shared_pixel_accuracy"),
        "depth_estimation": ("unit_mae", "unit_rmse", "unit_acc_020"),
    }
    any_to_any_preferred = {
        "edge": ("edge_accuracy", "edge_false_positive_rate", "edge_specificity"),
        "normal": ("normal_mean_deg", "normal_median_deg", "normal_within_30"),
        "semantic": ("shared_pixel_accuracy", "shared_miou", "shared_macro_f1"),
        "depth": ("unit_mae", "unit_rmse", "unit_acc_020"),
        "rgb": ("mae", "mse", "psnr_db"),
    }
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Method & Regime & Task & Metric 1 & Metric 2 & Metric 3 & Samples \\",
        r"\midrule",
    ]
    for row in aggregates:
        task = str(row["task"])
        metric_order = preferred.get(task, ())
        if task == "any_to_any":
            metric_order = any_to_any_preferred.get(str(row.get("target_type") or ""), ())
        values = [row.get(key) for key in metric_order[:3]]
        formatted = ["--" if value is None else f"{float(value):.4f}" for value in values]
        while len(formatted) < 3:
            formatted.append("--")
        lines.append(
            f"{latex_escape(row['display_name'])} & {latex_escape(row['inference_regime'])} & "
            f"{latex_escape(task)} & {' & '.join(formatted)} & {int(row.get('samples', 0))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    atomic_text_dump("\n".join(lines), path)


def template_payload() -> dict[str, Any]:
    methods = [
        ("wsm", "WSM", "held_out_test"),
        ("controlnet", "ControlNet", "zero_shot_official"),
        ("unicontrol", "UniControl", "zero_shot_official"),
        ("uni_controlnet", "Uni-ControlNet", "zero_shot_official"),
        ("t2i_adapter", "T2I-Adapter", "zero_shot_official"),
        ("dmp", "DMP", "zero_shot_official"),
        ("semflow", "SemFlow", "zero_shot_official"),
        ("modus", "MODUS", "zero_shot_official"),
        ("marigold", "Marigold", "zero_shot_official"),
        ("geowizard", "GeoWizard", "zero_shot_official"),
        ("vggt", "VGGT", "zero_shot_official"),
        ("sensenova_vision", "SenseNova-Vision", "zero_shot_official"),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_name": "wsm_external_comparison",
        "sample_manifest": "samples.jsonl",
        "workspace": "../../evaluation_output/external/wsm_external_comparison",
        "fairness_notes": {
            "vae_policy": "Each method retains its official VAE/tokenizer/backbone.",
            "pretraining_overlap": "Disclose possible public-benchmark pretraining overlap.",
            "paper_rule": "Never mix reported_only rows with locally rerun rankings.",
        },
        "task_protocols": {
            "conditional_rgb": {
                "edge_detector": "canny",
                "canny_low": 100,
                "canny_high": 200,
                "edge_threshold": 0.5,
                "compute_fid": False,
                "compute_clip": False,
            },
            "semantic_estimation": {
                "shared_class_ids": [],
                "ignore_id": 255,
                "note": (
                    "Populate only after freezing the WSM/baseline/dataset "
                    "ontology intersection."
                ),
            },
            "any_to_any": {
                "edge_threshold": 0.5,
                "depth_alignment": "scale_shift",
                "normal_encoding": "rgb_01",
                "shared_class_ids": [],
            },
            "depth_estimation": {
                "depth_alignment": "scale_shift",
                "min_depth": 0.001,
                "max_depth": 80.0,
            },
        },
        "methods": [
            {
                "id": method_id,
                "display_name": display_name,
                "enabled": method_id == "wsm",
                "inference_regime": regime,
                "prediction_manifest": f"predictions/{method_id}.jsonl",
                "checkpoint": "FILL_ME",
            }
            for method_id, display_name, regime in methods
        ],
    }


def write_template(path: Path) -> None:
    atomic_json_dump(template_payload(), path)
    sample_path = path.parent / "samples.example.jsonl"
    prediction_path = path.parent / "predictions.example.jsonl"
    atomic_text_dump(
        json.dumps(
            {
                "sample_id": "nyu_test_000001",
                "task": "depth_estimation",
                "dataset": "NYUv2",
                "split": "test",
                "input_path": "images/000001.png",
                "target_path": "depth/000001.npy",
                "mask_path": "masks/000001.png",
            },
            ensure_ascii=False,
        )
        + "\n",
        sample_path,
    )
    atomic_text_dump(
        json.dumps(
            {
                "sample_id": "nyu_test_000001",
                "prediction_path": "outputs/000001.npy",
                "array_key": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        prediction_path,
    )
    print(f"Wrote config template: {path}")
    print(f"Wrote sample example: {sample_path}")
    print(f"Wrote prediction example: {prediction_path}")


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="External benchmark JSON contract.")
    parser.add_argument(
        "--write-template",
        default="",
        help="Write a config and JSONL examples, then exit.",
    )
    parser.add_argument("--methods", default="", help="Optional comma-separated method IDs.")
    parser.add_argument("--tasks", default="", help="Optional comma-separated task names.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--fresh-tables",
        action="store_true",
        help="Discard prior method/task rows instead of incrementally replacing selected rows.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Permit incomplete prediction coverage for debugging; "
            "never use for paper tables."
        ),
    )
    parser.add_argument(
        "--extended-evaluation",
        action="store_true",
        help=(
            "Compute the complete diagnostic metric set, bootstrap confidence "
            "intervals, and enabled distribution metrics. By default only the "
            "quality metrics reported in paper Tables 6 and 7 are written."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_template(Path(args.write_template).expanduser().resolve())
        return
    if not args.config:
        raise SystemExit("Provide --config or --write-template")
    config_path = Path(args.config).expanduser().resolve()
    config = read_json(config_path)
    selected_methods = parse_csv_set(args.methods)
    selected_tasks = parse_csv_set(args.tasks)
    unknown_tasks = selected_tasks - TASKS
    if unknown_tasks:
        raise ContractError(f"Unknown --tasks values: {sorted(unknown_tasks)}")
    base, workspace, samples, methods, audit = validate_contract(
        config_path,
        config,
        selected_methods,
        selected_tasks,
        args.allow_missing,
    )
    audit.update(
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark_name": config.get("benchmark_name", ""),
            "config": str(config_path),
            "metric_profile": (
                "extended" if args.extended_evaluation else "paper_tables_6_7"
            ),
            "config_sha256": sha256_file(config_path),
            "allow_missing": bool(args.allow_missing),
            "selected_tasks": sorted({str(item["task"]) for item in samples}),
            "methods": [method.method_id for method in methods],
        }
    )
    if args.validate_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    protocols = dict(config.get("task_protocols") or {})
    sample_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    reported_rows: list[dict[str, Any]] = []
    for method in methods:
        rows, aggregates = evaluate_method(
            method, samples, base, protocols, args.allow_missing,
            args.extended_evaluation,
        )
        sample_rows.extend(rows)
        if method.inference_regime == "reported_only":
            reported_rows.extend(aggregates)
        else:
            aggregate_rows.extend(aggregates)
    tables = workspace / "tables"

    def merge_prior(
        current: list[dict[str, Any]], filename: str
    ) -> list[dict[str, Any]]:
        if args.fresh_tables:
            return current
        path = tables / filename
        if not path.is_file():
            return current
        payload = read_json(path)
        prior = list(payload.get("rows") or [])
        method_ids = {method.method_id for method in methods}
        task_ids = {str(sample["task"]) for sample in samples}
        retained = [
            row for row in prior
            if not (
                str(row.get("method_id")) in method_ids
                and str(row.get("task")) in task_ids
            )
        ]
        merged = retained + current
        return sorted(
            merged,
            key=lambda row: (
                str(row.get("task") or ""),
                str(row.get("method_id") or ""),
                str(row.get("sample_id") or ""),
            ),
        )

    sample_rows = merge_prior(sample_rows, "sample_metrics.json")
    aggregate_rows = merge_prior(aggregate_rows, "method_metrics.json")
    reported_rows = merge_prior(reported_rows, "reported_metrics_separate.json")
    if not args.extended_evaluation:
        sample_rows = [select_paper_row(row) for row in sample_rows]
        aggregate_rows = [select_paper_row(row) for row in aggregate_rows]
        reported_rows = [select_paper_row(row) for row in reported_rows]
    write_csv(sample_rows, tables / "sample_metrics.csv")
    write_csv(aggregate_rows, tables / "method_metrics.csv")
    write_csv(reported_rows, tables / "reported_metrics_separate.csv")
    atomic_json_dump({"rows": sample_rows}, tables / "sample_metrics.json")
    atomic_json_dump({"rows": aggregate_rows}, tables / "method_metrics.json")
    atomic_json_dump(
        {"rows": reported_rows}, tables / "reported_metrics_separate.json"
    )
    write_latex(aggregate_rows, tables / "method_metrics.tex")
    atomic_json_dump(audit, workspace / "audit.json")
    print(f"Validated samples: {len(samples)}")
    print(
        "Locally evaluated methods: "
        f"{len({row['method_id'] for row in aggregate_rows})}"
    )
    print(
        "Reported-only methods: "
        f"{len({row['method_id'] for row in reported_rows})}"
    )
    print(f"Workspace: {workspace}")


if __name__ == "__main__":
    main()
