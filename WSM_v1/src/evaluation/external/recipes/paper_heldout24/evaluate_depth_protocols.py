#!/usr/bin/env python3
"""One-stop WSM absolute/relative depth value-space audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
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
from finalize_predictions import (
    _normalize_depth_lane,
    finalize_wsm_depth_audit,
    read_jsonl,
)


WSM_VARIANTS = (
    ("wsm_depth_head2_denormalized", "WSM absolute, Reader value"),
    ("wsm_depth_head2_direct_normalized", "WSM absolute, direct normalized"),
    ("wsm_depth_head1_denormalized", "WSM relative, Reader value"),
    ("wsm_depth_head1_direct_normalized", "WSM relative, direct normalized"),
)
BASELINES = (
    ("marigold", "Marigold"),
    ("dmp", "DMP"),
    ("geowizard", "GeoWizard"),
    ("vggt", "VGGT"),
    ("modus", "MODUS"),
    ("sensenova_vision", "SenseNova-Vision"),
    ("scaling_diffusion_perception", "Scaling Diffusion Perception"),
)


def native_relative_metrics(
    workspace: Path, methods: list[dict], samples: list[dict]
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Compare predictions with the VAE-decoded native channel-32 [0,1] GT."""
    gt_by_source = {}
    for root in (workspace / "outputs").glob(
        "wsm_relative_depth_head1_exact24_tensors*"
    ):
        for metadata_path in root.glob(
            "pseudo-visualization-inference/benchmark/sample_*/metadata.json"
        ):
            source_id = str(
                json.loads(metadata_path.read_text(encoding="utf-8")).get("sample_id")
                or ""
            )
            gt_path = metadata_path.parent / "wsg_reader/tensors/gt.normalized.wsg.npz"
            if source_id and gt_path.is_file():
                gt_by_source[source_id] = gt_path

    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    per_sample = []
    for method in methods:
        method_id = str(method["id"])
        predictions = {
            str(row["sample_id"]): row
            for row in read_jsonl(Path(method["prediction_manifest"]))
        }
        for sample_id, sample in sample_by_id.items():
            source_id = str(sample.get("source_sample_id") or "")
            prediction = predictions.get(sample_id)
            gt_path = gt_by_source.get(source_id)
            if prediction is None or gt_path is None:
                continue
            with np.load(gt_path, allow_pickle=False) as payload:
                target = np.asarray(payload["wsg_data"][..., 32], dtype=np.float64)
            pred = load_scalar(Path(prediction["prediction_path"]).resolve())
            pred = resize_array(pred, target.shape, nearest=False).astype(np.float64)
            valid = valid_mask_for(sample, workspace, target.shape)
            valid &= np.isfinite(target) & np.isfinite(pred)
            if not valid.any():
                continue

            if method_id == "wsm_depth_head1_denormalized":
                pred_unit = _normalize_depth_lane(pred, 32).astype(np.float64)
            elif method_id == "wsm_depth_head2_denormalized":
                pred_unit = _normalize_depth_lane(pred, 34).astype(np.float64)
            elif method_id.startswith("wsm_"):
                pred_unit = np.clip(pred, 0.0, 1.0)
            else:
                pred_unit = np.zeros_like(pred, dtype=np.float64)
                try:
                    pred_unit[valid] = _robust_unit_depth(pred[valid], 2.0, 98.0)
                except ContractError:
                    # A constant output has no defined relative-depth scale.
                    continue
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
                "native_mae": float(np.mean(error)),
                "native_rmse": float(np.sqrt(mse)),
                "native_psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
                "native_acc_005": float(np.mean(error < 0.05)),
                "native_acc_010": float(np.mean(error < 0.10)),
                "native_acc_020": float(np.mean(error < 0.20)),
                "native_pearson": _correlation(target_thin, pred_thin),
                "native_spearman": _correlation(
                    _average_tie_ranks(target_thin),
                    _average_tie_ranks(pred_thin),
                ),
            })

    metric_names = [
        "native_mae", "native_rmse", "native_psnr", "native_acc_005",
        "native_acc_010", "native_acc_020", "native_pearson",
        "native_spearman",
    ]
    summaries = {}
    for method in methods:
        method_id = str(method["id"])
        method_rows = [row for row in per_sample if row["method_id"] == method_id]
        if not method_rows:
            continue
        summaries[method_id] = {
            name: float(np.nanmean([row[name] for row in method_rows]))
            for name in metric_names
        }
    return summaries, per_sample


def complete_depth_manifest(path: Path, expected: set[str]) -> bool:
    if not path.is_file():
        return False
    present = {
        str(row.get("sample_id") or "")
        for row in read_jsonl(path)
        if str(row.get("sample_id") or "") in expected
    }
    return present == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--reuse-primary", action="store_true",
        help="Reuse existing affine/unit tables and only refresh native metrics.",
    )
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    if not args.reuse_primary and not finalize_wsm_depth_audit(workspace):
        raise SystemExit("WSM relative-depth inference is not complete yet")

    depth_ids = {
        str(row["sample_id"])
        for row in read_jsonl(
            workspace / "predictions/wsm_depth_head1_direct_normalized.jsonl"
        )
    }
    audit_root = workspace / "depth_protocol_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_samples = [
        row for row in read_jsonl(workspace / "samples.jsonl")
        if row.get("task") == "depth_estimation"
        and str(row.get("sample_id") or "") in depth_ids
    ]
    audit_sample_manifest = audit_root / "samples_common_depth.jsonl"
    with audit_sample_manifest.open("w", encoding="utf-8") as handle:
        for row in audit_samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    methods = []
    for method_id, display_name in WSM_VARIANTS + BASELINES:
        manifest = workspace / f"predictions/{method_id}.jsonl"
        if not complete_depth_manifest(manifest, depth_ids):
            continue
        methods.append({
            "id": method_id,
            "display_name": display_name,
            "enabled": True,
            "tasks": ["depth_estimation"],
            "inference_regime": (
                "held_out_test" if method_id.startswith("wsm_")
                else "zero_shot_official"
            ),
            "prediction_manifest": str(manifest),
        })

    contract = {
        "schema_version": 1,
        "benchmark_name": "wsm_depth_protocol_audit",
        "sample_manifest": str(audit_sample_manifest),
        "workspace": str(audit_root),
        "task_protocols": {
            "depth_estimation": {
                "depth_alignment": "scale_shift",
                "min_depth": 0.001,
                "max_depth": 80.0,
                "normalization_low_percentile": 2.0,
                "normalization_high_percentile": 98.0,
                "bootstrap_samples": max(0, int(args.bootstrap_samples)),
            }
        },
        "methods": methods,
    }
    contract_path = audit_root / "evaluation_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    evaluator = Path(__file__).with_name("evaluate_predictions.py")
    if not args.reuse_primary:
        subprocess.run([
            sys.executable, str(evaluator), "--config", str(contract_path),
            "--tasks", "depth_estimation", "--fresh-tables",
        ], check=True)

    source = audit_root / "tables/method_metrics.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    native_summaries, native_rows = native_relative_metrics(
        workspace, methods, audit_samples
    )
    native_sample_path = audit_root / "tables/native_relative_sample_metrics.csv"
    native_fields = [
        "method_id", "sample_id", "native_mae", "native_rmse", "native_psnr",
        "native_acc_005", "native_acc_010", "native_acc_020",
        "native_pearson", "native_spearman",
    ]
    with native_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=native_fields)
        writer.writeheader()
        writer.writerows(native_rows)
    for row in rows:
        row.update(native_summaries.get(str(row.get("method_id") or ""), {}))
    fields = [
        "method_id", "display_name", "samples", "abs_rel", "sq_rel", "rmse",
        "mae_depth", "delta1", "delta2", "delta3", "pearson_depth",
        "spearman_depth", "unit_mae", "unit_rmse", "unit_acc_005",
        "unit_acc_010", "unit_acc_020", "unit_pearson", "unit_spearman",
        "raw_metric_abs_rel", "raw_metric_rmse", "median_abs_rel", "median_rmse",
        "native_mae", "native_rmse", "native_psnr", "native_acc_005",
        "native_acc_010", "native_acc_020", "native_pearson", "native_spearman",
    ]
    destination = audit_root / "tables/depth_protocol_comparison.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    print(f"Depth protocol comparison: {destination}")


if __name__ == "__main__":
    main()
