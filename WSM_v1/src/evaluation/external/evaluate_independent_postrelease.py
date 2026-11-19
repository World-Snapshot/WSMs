#!/usr/bin/env python3
"""Reproduce the paper-facing independent-annotation benchmark.

The default profile evaluates only the frozen, paper-facing task/metric
profile materialized by this runner. Pass --extended-evaluation to emit
additional diagnostic tasks and metrics into a separate output directory.
The two profiles are explicit so exploratory runs are never mixed into the
paper reproduction outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
SUITE = PROJECT / "src/evaluation_output/external/wsm_postrelease_video_20260905/frozen_suite_v2_independent"
EXTERNAL = PROJECT / "src/evaluation/external"
RECIPE = EXTERNAL / "recipes/paper_heldout24"
OLD_SUITE = PROJECT / "src/evaluation_output/external/wsm_postrelease_video_20260905/frozen_suite_v1"
sys.path[:0] = [str(EXTERNAL), str(RECIPE)]

from evaluate_crop_aligned_depth import metrics_for as depth_metrics_for
from evaluate_predictions import (
    ContractError,
    _robust_unit_depth,
    any_to_any_metrics,
    conditional_rgb_metrics,
    load_array,
    load_labels,
    load_scalar,
    resize_array,
    valid_mask_for,
)


REPORTED_DEPTH_METRICS = (
    "depth_boundary_precision", "depth_boundary_precision_3px",
    "depth_boundary_fpr", "depth_gradient_mae", "depth_gradient_rmse",
)
REPORTED_RGB_METRICS = (
    "edge_density_error", "edge_f0_5", "edge_precision",
    "mae", "mse", "psnr_db", "ssim",
)
REPORTED_NORMAL_METRICS = (
    "normal_cosine_similarity", "normal_rmse_xyz", "normal_mean_deg",
    "normal_rmse_deg", "normal_p90_deg",
)
REPORTED_INTERFACE_C_TASKS = {
    "Edge-to-Normal": (
        SUITE / "samples/edge_to_normal.jsonl",
        SUITE / "predictions/wsm_edge_to_normal.jsonl",
    ),
    "Relative-Depth-to-Normal": (
        SUITE / "samples/relative_depth_to_normal.jsonl",
        SUITE / "predictions/wsm_relative_depth_to_normal_independent.jsonl",
    ),
}
EXTENDED_INTERFACE_C_TASKS = {
    "Normal-to-Edge": (
        SUITE / "samples/normal_to_edge.jsonl",
        SUITE / "predictions/wsm_normal_to_edge_independent.jsonl",
    ),
    "Edge-to-Human-Segmentation": (
        SUITE / "samples/edge_to_human_seg.jsonl",
        PROJECT / "src/evaluation_output/external/wsm_postrelease_video_20260905/frozen_suite_v1_human_revision1/predictions/wsm_merged24.jsonl",
    ),
    "Relative-Depth-to-Edge": (
        SUITE / "samples/relative_depth_to_edge.jsonl",
        SUITE / "predictions/wsm_relative_depth_to_edge.jsonl",
    ),
    "Edge-to-Relative-Depth": (
        SUITE / "samples/edge_to_relative_depth.jsonl",
        SUITE / "predictions/wsm_edge_to_relative_depth.jsonl",
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize_dmp(samples: list[dict]) -> list[dict]:
    base = OLD_SUITE / "full/external/dmp/depth_estimation"
    rows = []
    for sample in samples:
        path = base / f"{sample['sample_id']}.npz"
        if path.is_file():
            rows.append({
                "sample_id": sample["sample_id"],
                "prediction_path": str(path),
                "array_key": "x",
                "adapter": "dmp_official_infer",
                "checkpoint": "depth-hypersim-notext",
                "steps": 5,
                "target_scale": 8.0,
            })
    write_jsonl(SUITE / "predictions/dmp.jsonl", rows)
    return rows


def finalize_geowizard(samples: list[dict]) -> list[dict]:
    base = OLD_SUITE / "full/external/geowizard/official_outputs/depth_npy"
    rows = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        exact = base / f"{sample_id}_pred.npy"
        matches = [exact] if exact.is_file() else sorted(base.glob(f"{sample_id}*_pred.npy"))
        if len(matches) == 1:
            rows.append({
                "sample_id": sample_id,
                "prediction_path": str(matches[0]),
                "adapter": "geowizard_official_run_infer_v2",
                "ensemble_size": 10,
                "denoise_steps": 50,
                "domain": "indoor",
            })
    write_jsonl(SUITE / "predictions/geowizard.jsonl", rows)
    return rows


def merge_modus_shards(extended_evaluation: bool) -> list[dict]:
    rows = [
        row for row in read_jsonl(OLD_SUITE / "predictions/modus.jsonl")
        if str(row.get("sample_id", "")).endswith("_rgb_depth")
    ]
    for shard in range(8):
        rows.extend(read_jsonl(
            SUITE / f"predictions/modus_independent_shard{shard}.jsonl"
        ))
        rows.extend(read_jsonl(
            SUITE / f"predictions/modus_edge_normal_full_shard{shard}.jsonl"
        ))
        if extended_evaluation:
            rows.extend(read_jsonl(
                SUITE / f"predictions/modus_edge_depth_full_shard{shard}.jsonl"
            ))
    if extended_evaluation:
        rows.extend(read_jsonl(
            PROJECT / "src/evaluation_output/external/wsm_postrelease_video_20260905/frozen_suite_v1_human_revision1/predictions/modus_merged24.jsonl"
        ))
    if not extended_evaluation:
        reported_ids = {
            str(sample["sample_id"])
            for sample_path, _ in REPORTED_INTERFACE_C_TASKS.values()
            for sample in read_jsonl(sample_path)
        }
        reported_ids.update(
            str(sample["sample_id"])
            for sample in read_jsonl(SUITE / "samples/rgb_to_relative_depth.jsonl")
        )
        rows = [
            row for row in rows
            if str(row.get("sample_id", "")) in reported_ids
        ]
    rows.sort(key=lambda row: str(row.get("sample_id", "")))
    if rows:
        suffix = "_extended" if extended_evaluation else ""
        write_jsonl(SUITE / f"predictions/modus_independent{suffix}.jsonl", rows)
    return rows


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_id"]), str(row["task_name"]))].append(row)
    summary = []
    for (method_id, task_name), group in sorted(grouped.items()):
        item = {"method_id": method_id, "task_name": task_name, "samples": len(group)}
        for key in keys:
            values = [float(row[key]) for row in group if key in row and np.isfinite(float(row[key]))]
            if values:
                item[key] = float(np.mean(values))
        summary.append(item)
    return summary


def score_depth(
    samples: list[dict], extended_evaluation: bool,
) -> tuple[list[dict], list[dict]]:
    modus_suffix = "_extended" if extended_evaluation else ""
    modus_manifest = SUITE / f"predictions/modus_independent{modus_suffix}.jsonl"
    manifests = {
        "marigold": OLD_SUITE / "predictions/marigold.jsonl",
        "dmp": SUITE / "predictions/dmp.jsonl",
        "geowizard": SUITE / "predictions/geowizard.jsonl",
        "scaling_diffusion_perception": OLD_SUITE / "predictions/scaling_diffusion_perception.jsonl",
        "vggt": OLD_SUITE / "predictions/vggt.jsonl",
        "modus": modus_manifest,
        "sensenova_vision": OLD_SUITE / "predictions/sensenova_vision.jsonl",
        "wsm_relative_depth_crop_aligned": OLD_SUITE / "predictions/wsm_rgb_to_relative_depth.jsonl",
    }
    if extended_evaluation:
        manifests["wsm_absolute_depth_normalized"] = (
            SUITE / "predictions/wsm_rgb_to_absolute_depth_independent.jsonl"
        )
    per_sample = []
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    for method_id, manifest in manifests.items():
        predictions = read_jsonl(manifest)
        if not predictions:
            continue
        predictions_by_id = {str(row["sample_id"]): row for row in predictions}
        if extended_evaluation:
            rows = depth_metrics_for(SUITE, samples, method_id, predictions)
        else:
            rows = [
                {"method_id": method_id, "sample_id": sample_id}
                for sample_id in samples_by_id if sample_id in predictions_by_id
            ]
        for row in rows:
            sample_id = str(row["sample_id"])
            row["task_name"] = "RGB-to-Relative-Depth"
            row.update(depth_boundary_metrics(
                samples_by_id[sample_id], predictions_by_id[sample_id], method_id
            ))
        per_sample.extend(rows)
    keys = REPORTED_DEPTH_METRICS
    if extended_evaluation:
        keys = (
            "mae", "rmse", "psnr", "acc_005", "acc_010", "acc_020",
            "pearson", "spearman", "depth_boundary_precision",
            "depth_boundary_precision_3px", "depth_boundary_recall",
            "depth_boundary_f1", "depth_boundary_fpr", "depth_gradient_mae",
            "depth_gradient_rmse",
        )
    per_sample = retain_metrics(per_sample, keys)
    return per_sample, summarize(per_sample, keys)


def binary_seg_metrics(sample: dict, prediction: dict) -> dict[str, float]:
    target = load_labels(Path(sample["target_path"]))
    pred = np.asarray(load_array(
        Path(prediction["prediction_path"]),
        str(prediction.get("array_key") or ""),
    )).squeeze()
    if pred.ndim != 2:
        raise ContractError(
            f"{sample['sample_id']}: expected a scalar human-part field, "
            f"got {pred.shape}"
        )
    pred = resize_array(pred.astype(np.float32), target.shape, nearest=True)
    # Channel 103 stores the six MediaPipe part IDs 0..5. VAE predictions
    # remain continuous, so recover the nearest categorical ID before merging
    # the five foreground parts into the binary person mask used by MODUS.
    pred = np.rint(np.clip(pred, 0.0, 5.0)).astype(np.int64)
    valid = valid_mask_for(sample, SUITE, target.shape)
    target = target[valid] > 0
    pred = pred[valid] > 0
    tp = float(np.sum(pred & target))
    tn = float(np.sum(~pred & ~target))
    fp = float(np.sum(pred & ~target))
    fn = float(np.sum(~pred & target))
    total = tp + tn + fp + fn
    fg_iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    bg_iou = tn / (tn + fp + fn) if tn + fp + fn else 1.0
    fg_acc = tp / (tp + fn) if tp + fn else 1.0
    bg_acc = tn / (tn + fp) if tn + fp else 1.0
    fg_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    bg_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 1.0
    return {
        "pixel_accuracy": (tp + tn) / total,
        "miou": 0.5 * (fg_iou + bg_iou),
        "human_foreground_iou": fg_iou,
        "mean_class_accuracy": 0.5 * (fg_acc + bg_acc),
        "macro_f1": 0.5 * (fg_f1 + bg_f1),
    }


def score_conditional_rgb(
    samples: list[dict], extended_evaluation: bool,
) -> tuple[list[dict], list[dict]]:
    manifests = {
        "controlnet": OLD_SUITE / "predictions/controlnet.jsonl",
        "controlnet_ip_adapter": OLD_SUITE / "predictions/controlnet_ip_adapter.jsonl",
        "t2i_adapter": OLD_SUITE / "predictions/t2i_adapter.jsonl",
        "unicontrol": OLD_SUITE / "predictions/unicontrol.jsonl",
        "uni_controlnet": OLD_SUITE / "predictions/uni_controlnet.jsonl",
        "wsm": OLD_SUITE / "predictions/wsm_edge_to_rgb.jsonl",
    }
    protocol = {"edge_threshold": 0.5}
    per_sample = []
    for method_id, manifest in manifests.items():
        predictions = {str(row["sample_id"]): row for row in read_jsonl(manifest)}
        for sample in samples:
            prediction = predictions.get(str(sample["sample_id"]))
            if prediction is None:
                continue
            metrics = conditional_rgb_metrics(sample, prediction, SUITE, protocol)
            per_sample.append({
                "method_id": method_id, "task_name": "Edge-to-RGB",
                "sample_id": sample["sample_id"], **metrics,
            })
    metric_keys = REPORTED_RGB_METRICS
    if extended_evaluation:
        metric_keys = tuple(sorted(
            {key for row in per_sample for key in row}
            - {"method_id", "task_name", "sample_id"}
        ))
    per_sample = retain_metrics(per_sample, metric_keys)
    return per_sample, summarize(per_sample, metric_keys)


def score_any_to_any(
    modus: list[dict], extended_evaluation: bool,
) -> tuple[list[dict], list[dict]]:
    tasks = dict(REPORTED_INTERFACE_C_TASKS)
    if extended_evaluation:
        tasks.update(EXTENDED_INTERFACE_C_TASKS)
    modus_by_id = {str(row["sample_id"]): row for row in modus}
    per_sample = []
    protocol = {
        "edge_threshold": 0.5, "normal_encoding": "rgb_01",
        "shared_class_ids": [0, 1],
    }
    for task_name, (sample_path, wsm_path) in tasks.items():
        samples = read_jsonl(sample_path)
        methods = {
            "modus": modus_by_id,
            "wsm": {str(row["sample_id"]): row for row in read_jsonl(wsm_path)},
        }
        for method_id, predictions in methods.items():
            for sample in samples:
                prediction = predictions.get(str(sample["sample_id"]))
                if prediction is None:
                    continue
                if method_id == "modus" and sample["target_type"] == "depth":
                    prediction = {
                        **prediction, "invert_unit_depth": True,
                        "depth_polarity_conversion": "fixed_modus_near_far_flip",
                    }
                if sample["target_type"] == "semantic":
                    metrics = binary_seg_metrics(sample, prediction)
                else:
                    metrics = any_to_any_metrics(
                        sample, prediction, SUITE, protocol,
                        extended_evaluation=extended_evaluation,
                    )
                per_sample.append({
                    "method_id": method_id, "task_name": task_name,
                    "sample_id": sample["sample_id"], **metrics,
                })
    metric_keys = REPORTED_NORMAL_METRICS
    if extended_evaluation:
        metric_keys = tuple(sorted(
            {key for row in per_sample for key in row}
            - {"method_id", "task_name", "sample_id"}
        ))
    per_sample = retain_metrics(per_sample, metric_keys)
    return per_sample, summarize(per_sample, metric_keys)


def retain_metrics(rows: list[dict], metric_keys: tuple[str, ...]) -> list[dict]:
    """Keep row identity plus the metrics exposed by the active profile."""
    identity = ("method_id", "task_name", "sample_id")
    allowed = set(identity + metric_keys)
    return [
        {key: value for key, value in row.items() if key in allowed}
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper-facing independent-annotation benchmark."
    )
    parser.add_argument(
        "--extended-evaluation", action="store_true",
        help=(
            "Also evaluate diagnostic tasks and metrics not reported in Table 8; "
            "results are written to an independent extended directory."
        ),
    )
    args = parser.parse_args()
    depth_samples = read_jsonl(SUITE / "samples/rgb_to_relative_depth.jsonl")
    finalize_dmp(depth_samples)
    finalize_geowizard(depth_samples)
    modus = merge_modus_shards(args.extended_evaluation)
    depth_rows, depth_summary = score_depth(
        depth_samples, args.extended_evaluation
    )
    rgb_samples = read_jsonl(SUITE / "samples/edge_to_rgb.jsonl")
    rgb_rows, rgb_summary = score_conditional_rgb(
        rgb_samples, args.extended_evaluation
    )
    any_rows, any_summary = score_any_to_any(
        modus, args.extended_evaluation
    )
    profile = "extended" if args.extended_evaluation else "table8"
    dirname = (
        "independent_annotations_extended"
        if args.extended_evaluation else "independent_annotations"
    )
    tables = SUITE / "metrics" / dirname
    write_csv(tables / "depth_sample_metrics.csv", depth_rows)
    write_csv(tables / "depth_method_metrics.csv", depth_summary)
    write_csv(tables / "conditional_rgb_sample_metrics.csv", rgb_rows)
    write_csv(tables / "conditional_rgb_method_metrics.csv", rgb_summary)
    write_csv(tables / "interface_c_sample_metrics.csv", any_rows)
    write_csv(tables / "interface_c_method_metrics.csv", any_summary)
    coverage = {
        "metric_profile": profile,
        "depth": {row["method_id"]: row["samples"] for row in depth_summary},
        "conditional_rgb": {
            row["method_id"]: row["samples"] for row in rgb_summary
        },
        "interface_c": {
            f"{row['method_id']}:{row['task_name']}": row["samples"]
            for row in any_summary
        },
    }
    (tables / "coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(coverage, indent=2))


def depth_boundary_metrics(
    sample: dict, prediction: dict, method_id: str
) -> dict[str, float]:
    """Measure depth discontinuities in the common normalized depth space."""
    target = load_scalar(Path(sample["target_path"])).astype(np.float64)
    pred = load_scalar(
        Path(prediction["prediction_path"]),
        str(prediction.get("array_key") or ""),
        float(prediction.get("prediction_scale", 1.0)),
        str(prediction.get("scalar_encoding") or "auto"),
    )
    pred = resize_array(pred, target.shape, nearest=False).astype(np.float64)
    valid = valid_mask_for(sample, SUITE, target.shape)
    valid &= np.isfinite(target) & np.isfinite(pred)
    if not valid.any():
        raise ContractError(f"{sample['sample_id']}: no valid depth pixels")

    pred_unit = np.zeros_like(pred, dtype=np.float64)
    if method_id == "wsm_relative_depth_crop_aligned":
        pred_unit[valid] = np.clip(pred[valid], 0.0, 1.0)
    else:
        pred_unit[valid] = _robust_unit_depth(pred[valid], 2.0, 98.0)
        if method_id in {"modus", "sensenova_vision"}:
            pred_unit[valid] = 1.0 - pred_unit[valid]
    target = np.clip(target, 0.0, 1.0)

    target_x = cv2.Sobel(target, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    target_y = cv2.Sobel(target, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    pred_x = cv2.Sobel(pred_unit, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    pred_y = cv2.Sobel(pred_unit, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    target_grad = np.hypot(target_x, target_y)
    pred_grad = np.hypot(pred_x, pred_y)
    interior = valid.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    threshold = max(0.03, float(np.quantile(target_grad[interior], 0.90)))
    target_edge = (target_grad >= threshold) & interior
    pred_edge = (pred_grad >= threshold) & interior
    tolerance = np.ones((5, 5), dtype=np.uint8)
    target_dilated = cv2.dilate(
        target_edge.astype(np.uint8), tolerance
    ).astype(bool)
    pred_dilated = cv2.dilate(
        pred_edge.astype(np.uint8), tolerance
    ).astype(bool)
    true_positive = float(np.sum(pred_edge & target_dilated))
    predicted_positive = float(np.sum(pred_edge))
    boundary_precision = true_positive / max(predicted_positive, 1.0)
    target_dilated_3px = cv2.dilate(
        target_edge.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
    ).astype(bool)
    boundary_precision_3px = float(
        np.sum(pred_edge & target_dilated_3px)
    ) / max(predicted_positive, 1.0)
    recalled_positive = float(np.sum(target_edge & pred_dilated))
    target_positive = float(np.sum(target_edge))
    boundary_recall = recalled_positive / max(target_positive, 1.0)
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-12)
    )
    raw_false_positive = float(np.sum(pred_edge & ~target_edge & interior))
    raw_true_negative = float(np.sum(~pred_edge & ~target_edge & interior))
    boundary_fpr = raw_false_positive / max(
        raw_false_positive + raw_true_negative, 1.0
    )
    gradient_mae = float(np.mean(
        np.abs(target_grad - pred_grad)[interior]
    ))
    gradient_rmse = float(np.sqrt(np.mean(
        0.5 * (
            (target_x - pred_x) ** 2 + (target_y - pred_y) ** 2
        )[interior]
    )))
    return {
        "depth_boundary_precision": boundary_precision,
        "depth_boundary_precision_3px": boundary_precision_3px,
        "depth_boundary_recall": boundary_recall,
        "depth_boundary_f1": boundary_f1,
        "depth_boundary_fpr": boundary_fpr,
        "depth_gradient_mae": gradient_mae,
        "depth_gradient_rmse": gradient_rmse,
    }


if __name__ == "__main__":
    main()
