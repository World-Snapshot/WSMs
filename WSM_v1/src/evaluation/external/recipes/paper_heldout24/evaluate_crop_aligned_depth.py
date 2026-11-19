#!/usr/bin/env python3
"""Finalize and evaluate the WSM-Reader crop-aligned depth rerun."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EXTERNAL_ROOT = Path(__file__).resolve().parents[2]
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from evaluate_predictions import (
    ContractError,
    _average_tie_ranks,
    _correlation,
    _robust_unit_depth,
    load_scalar,
    resize_array,
    valid_mask_for,
)


# These official decoders use the opposite near/far intensity convention from
# the native WSM relative-depth target. The transform is fixed per method and
# applied after the same 2nd--98th percentile map used for every baseline.
INVERTED_DEPTH_METHODS = {"modus", "sensenova_vision"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def finalize_dmp(root: Path, samples: list[dict]) -> list[dict]:
    rows = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        path = root / "outputs/dmp" / f"{sample_id}_rgb_crop.npz"
        if not path.is_file():
            continue
        rows.append({
            "sample_id": sample_id,
            "prediction_path": str(path.resolve()),
            "array_key": "x",
            "adapter": "dmp_official_infer_crop_aligned",
            "checkpoint": "depth-hypersim-notext",
            "steps": 5,
            "target_scale": 8.0,
        })
    write_jsonl(root / "predictions/dmp.jsonl", rows)
    return rows


def finalize_geowizard(root: Path, samples: list[dict]) -> list[dict]:
    rows = []
    output = root / "outputs/geowizard/official_outputs/depth_npy"
    for sample in samples:
        sample_id = str(sample["sample_id"])
        path = output / f"{sample_id}_rgb_crop_pred.npy"
        if not path.is_file():
            continue
        rows.append({
            "sample_id": sample_id,
            "prediction_path": str(path.resolve()),
            "adapter": "geowizard_official_run_infer_v2_crop_aligned",
            "checkpoint": "fuxiao0719/GeoWizard unet_v2",
            "ensemble_size": 10,
            "denoise_steps": 50,
            "domain": "indoor",
        })
    write_jsonl(root / "predictions/geowizard.jsonl", rows)
    return rows


def merge_shards(root: Path, method_id: str, shard_count: int = 3) -> list[dict]:
    rows = []
    runtimes = []
    for shard in range(shard_count):
        manifest = root / f"predictions/{method_id}_shard{shard}.jsonl"
        rows.extend(read_jsonl(manifest))
        runtime = manifest.with_suffix(".runtime.json")
        if runtime.is_file():
            runtimes.append(json.loads(runtime.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: str(row["sample_id"]))
    if rows:
        write_jsonl(root / f"predictions/{method_id}.jsonl", rows)
    if runtimes:
        (root / f"predictions/{method_id}.runtime.json").write_text(
            json.dumps({"shards": runtimes}, indent=2) + "\n", encoding="utf-8"
        )
    return rows


def metrics_for(
    root: Path, samples: list[dict], method_id: str, predictions: list[dict]
) -> list[dict]:
    prediction_by_id = {str(row["sample_id"]): row for row in predictions}
    per_sample = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        prediction = prediction_by_id.get(sample_id)
        if prediction is None:
            continue
        target = load_scalar(Path(sample["target_path"]))
        pred = load_scalar(
            Path(prediction["prediction_path"]),
            str(prediction.get("array_key") or ""),
            float(prediction.get("prediction_scale", 1.0)),
            str(prediction.get("scalar_encoding") or "auto"),
        )
        pred = resize_array(pred, target.shape, nearest=False).astype(np.float64)
        target = np.asarray(target, dtype=np.float64)
        valid = valid_mask_for(sample, root, target.shape)
        valid &= np.isfinite(target) & np.isfinite(pred)
        if not valid.any():
            continue

        if method_id == "wsm_relative_depth_crop_aligned":
            pred_unit = np.clip(pred, 0.0, 1.0)
        else:
            pred_unit = np.zeros_like(pred, dtype=np.float64)
            try:
                pred_unit[valid] = _robust_unit_depth(pred[valid], 2.0, 98.0)
            except ContractError:
                continue
            if method_id in INVERTED_DEPTH_METHODS:
                pred_unit[valid] = 1.0 - pred_unit[valid]
        target_values = np.clip(target[valid], 0.0, 1.0)
        pred_values = np.clip(pred_unit[valid], 0.0, 1.0)
        error = np.abs(pred_values - target_values)
        mse = float(np.mean(np.square(error)))
        stride = max(1, target_values.size // 100_000)
        target_thin = target_values[::stride]
        pred_thin = pred_values[::stride]
        per_sample.append({
            "method_id": method_id,
            "sample_id": sample_id,
            "mae": float(np.mean(error)),
            "rmse": float(np.sqrt(mse)),
            "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
            "acc_005": float(np.mean(error < 0.05)),
            "acc_010": float(np.mean(error < 0.10)),
            "acc_020": float(np.mean(error < 0.20)),
            "pearson": _correlation(target_thin, pred_thin),
            "spearman": _correlation(
                _average_tie_ranks(target_thin),
                _average_tie_ranks(pred_thin),
            ),
            "valid_pixels": int(valid.sum()),
        })
    return per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    samples = read_jsonl(root / "samples.jsonl")
    expected = {str(row["sample_id"]) for row in samples}

    finalize_dmp(root, samples)
    finalize_geowizard(root, samples)
    merge_shards(root, "scaling_diffusion_perception")
    merge_shards(root, "modus")
    methods = {
        "marigold": root / "predictions/marigold.jsonl",
        "dmp": root / "predictions/dmp.jsonl",
        "geowizard": root / "predictions/geowizard.jsonl",
        "scaling_diffusion_perception": root / "predictions/scaling_diffusion_perception.jsonl",
        "vggt": root / "predictions/vggt.jsonl",
        "modus": root / "predictions/modus.jsonl",
        "sensenova_vision": root / "predictions/sensenova_vision.jsonl",
        "wsm_relative_depth_crop_aligned": root / "predictions/wsm_relative_depth_crop_aligned.jsonl",
    }
    all_rows = []
    coverage = {}
    for method_id, manifest in methods.items():
        predictions = read_jsonl(manifest)
        present = {str(row.get("sample_id") or "") for row in predictions}
        coverage[method_id] = {
            "predictions": len(present & expected),
            "expected": len(expected),
            "complete": present >= expected,
            "missing": sorted(expected - present),
            "manifest": str(manifest),
        }
        all_rows.extend(metrics_for(root, samples, method_id, predictions))

    table_root = root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "method_id", "sample_id", "mae", "rmse", "psnr", "acc_005",
        "acc_010", "acc_020", "pearson", "spearman", "valid_pixels",
    ]
    with (table_root / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_fields = ["method_id", "samples"] + metric_fields[2:-1]
    summaries = []
    for method_id in methods:
        rows = [row for row in all_rows if row["method_id"] == method_id]
        if not rows:
            continue
        summaries.append({
            "method_id": method_id,
            "samples": len(rows),
            **{
                name: float(np.nanmean([float(row[name]) for row in rows]))
                for name in summary_fields[2:]
            },
        })
    with (table_root / "method_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    (root / "coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "coverage": coverage,
        "metrics": str((table_root / "method_metrics.csv").resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
