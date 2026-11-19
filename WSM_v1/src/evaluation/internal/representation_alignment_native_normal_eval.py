#!/usr/bin/env python
"""Evaluate representation-alignment trajectories in native normal space.

Each checkpoint is run on the same deterministic held-out samples.  The
evaluation wrapper first inverts the experimental latent permutation, decodes
with the native VAE, and writes normals with one fixed XYZ-to-RGB mapping.
Metrics therefore compare physical normal directions, not stretched preview
colors.  All outputs live below ``debug/representation_alignment``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_ROOT = (
    PROJECT_ROOT / "debug/representation_alignment/head003_controlled_training"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "debug/representation_alignment/head003_native_normal_trajectory"
)
DEFAULT_STEPS = (20, 40, 100, 140, 320, 480, 740, 1000)
REPRESENTATIONS = (
    "native",
    "vae-channel-cycle-15",
    "vae-channel-cycle-01",
    "vae-channel-cycle-03",
    "vae-channel-cycle-07",
)


def checkpoint_for(root: Path, representation: str, step: int) -> Path:
    trajectory = root / representation / "trajectory_ckpts"
    if step == 1000:
        candidate = trajectory / "final_step_0001000/pytorch_lora_weights.safetensors"
        if candidate.is_file():
            return candidate
    candidate = trajectory / f"{step:07d}/pytorch_lora_weights.safetensors"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"No shared trajectory checkpoint for {representation} at step {step}: "
            f"{candidate}"
        )
    return candidate


def normal_angles(prediction_path: Path, target_path: Path) -> np.ndarray:
    pred = np.asarray(Image.open(prediction_path).convert("RGB"), dtype=np.float64)
    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.float64)
    if pred.shape[:2] != target.shape[:2]:
        pred_image = Image.fromarray(pred.astype(np.uint8), mode="RGB")
        pred_image = pred_image.resize(
            (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
        )
        pred = np.asarray(pred_image, dtype=np.float64)
    pred = pred / 127.5 - 1.0
    target = target / 127.5 - 1.0
    pred_norm = np.linalg.norm(pred, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    valid = (
        np.isfinite(pred).all(axis=-1)
        & np.isfinite(target).all(axis=-1)
        & (pred_norm > 1e-8)
        & (target_norm > 1e-8)
    )
    pred = pred[valid] / pred_norm[valid, None]
    target = target[valid] / target_norm[valid, None]
    cosine = np.clip(np.sum(pred * target, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def sample_metrics(angles: np.ndarray) -> dict[str, float]:
    return {
        "normal_mean_deg": float(np.mean(angles)),
        "normal_rmse_deg": float(np.sqrt(np.mean(np.square(angles)))),
        "normal_median_deg": float(np.median(angles)),
        "normal_p90_deg": float(np.percentile(angles, 90.0)),
        "normal_within_11_25": float(np.mean(angles < 11.25)),
        "normal_within_22_5": float(np.mean(angles < 22.5)),
        "normal_within_30": float(np.mean(angles < 30.0)),
        "valid_pixels": int(angles.size),
    }


def metadata_files(step_dir: Path) -> list[Path]:
    root = step_dir / "pseudo-visualization-inference/native_normal"
    return sorted(root.glob("**/metadata.json"))


def evaluate_step(step_dir: Path, representation: str, step: int) -> dict:
    rows = []
    all_angles = []
    for metadata_path in metadata_files(step_dir):
        sample_dir = metadata_path.parent
        prediction_path = sample_dir / "inference.png"
        target_path = sample_dir / "gt.png"
        if not prediction_path.is_file() or not target_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        angles = normal_angles(prediction_path, target_path)
        metrics = sample_metrics(angles)
        rows.append({
            "representation": representation,
            "step": step,
            "sample_id": metadata.get("sample_id", ""),
            **metrics,
        })
        all_angles.append(angles)
    if not rows:
        raise RuntimeError(f"No complete inference samples in {step_dir}")
    pooled = sample_metrics(np.concatenate(all_angles))
    sample_means = np.asarray([row["normal_mean_deg"] for row in rows])
    summary = {
        "representation": representation,
        "step": step,
        "samples": len(rows),
        **pooled,
        "sample_mean_deg_mean": float(np.mean(sample_means)),
        "sample_mean_deg_std": float(np.std(sample_means)),
    }
    with (step_dir / "native_normal_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (step_dir / "native_normal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def inference_command(
    representation: str,
    checkpoint: Path,
    step_dir: Path,
    samples: int,
    inference_steps: int,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "src/evaluation/internal/representation_alignment_train.py"),
        "--representation-alignment", representation,
        "--representation-alignment-native-normal-render",
        "--representation-alignment-skip-reader-visuals",
        "--model-path", str(PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B"),
        "--modality-config", str(PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json"),
        "--dense-latent-cache-dir", str(PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset"),
        "--text-embedding-cache-dir", str(PROJECT_ROOT / "data/wsg_flux2_caption_embeddings_all_datasets"),
        "--channel-head-index", "3",
        "--target-modality", "36-38",
        "--condition-modalities", "0-2",
        "--channel-head-rank", "8",
        "--channel-head-alpha", "274",
        "--max-steps", "1",
        "--batch-size", "1",
        "--num-workers", "0",
        "--val-num-workers", "0",
        "--val-loss-batches", "0",
        "--seed", "20260622",
        "--disable-preflight",
        "--output-dir", str(step_dir),
        "--inference-only",
        "--inference-checkpoint", str(checkpoint),
        "--categorical-refiner", "off",
        "--inference-label", "native_normal",
        "--inference-steps", str(inference_steps),
        "--preview-split", "test",
        "--preview-require-split",
        "--preview-num-samples", str(samples),
        "--preview-random",
        "--trainable-target", "on",
        "--trainable-condition", "off",
        "--historical-reference-policy", "tolerant",
    ]


def run_representation(args: argparse.Namespace) -> None:
    train_root = Path(args.train_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    representation = args.representation
    steps = tuple(int(item) for item in args.steps.split(",") if item.strip())
    summaries = []
    for step in steps:
        checkpoint = checkpoint_for(train_root, representation, step)
        step_dir = output_root / representation / f"step_{step:07d}"
        summary_path = step_dir / "native_normal_summary.json"
        if summary_path.is_file() and not args.force:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[native-normal] cached {representation} step={step}", flush=True)
            continue
        step_dir.mkdir(parents=True, exist_ok=True)
        # Production inference locates optional run-level sidecars relative to
        # a directory literally named ``ckpt``.  Archived trajectory files are
        # read-only views, so expose each one through that conventional layout.
        checkpoint_view = (
            step_dir / "checkpoint_view/ckpt" / checkpoint.parent.name
            / "pytorch_lora_weights.safetensors"
        )
        checkpoint_view.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_view.exists():
            checkpoint_view.symlink_to(checkpoint)
        command = inference_command(
            representation, checkpoint_view, step_dir, args.samples, args.inference_steps
        )
        (step_dir / "command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[native-normal] run {representation} step={step}", flush=True)
        with (step_dir / "inference.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=os.environ.copy(),
            )
        summary = evaluate_step(step_dir, representation, step)
        summaries.append(summary)
        print(
            f"[native-normal] done {representation} step={step} "
            f"mean={summary['normal_mean_deg']:.3f}deg "
            f"within22.5={summary['normal_within_22_5']:.3f}",
            flush=True,
        )
    representation_dir = output_root / representation
    representation_dir.mkdir(parents=True, exist_ok=True)
    with (representation_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def aggregate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    rows = []
    for representation in REPRESENTATIONS:
        path = output_root / representation / "trajectory.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No trajectory.csv files below {output_root}")
    fields = list(rows[0])
    with (output_root / "native_normal_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    frozen_path = (
        PROJECT_ROOT
        / "debug/representation_alignment/head003_channel_cycles_trainmix_16/representation_alignment.json"
    )
    frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen = {
        item["representation"]: {
            "mean": item["frozen_flow_mse_mean"],
            "sample_std": item["frozen_flow_mse_sample_std"],
        }
        for item in frozen_payload["results"]
    }
    final = {row["representation"]: row for row in rows if int(row["step"]) == 1000}
    common = [item for item in REPRESENTATIONS if item in final and item in frozen]
    frozen_values = np.asarray([float(frozen[item]["mean"]) for item in common])
    quality_values = np.asarray([-float(final[item]["normal_mean_deg"]) for item in common])
    frozen_rank = np.argsort(np.argsort(frozen_values))
    quality_rank = np.argsort(np.argsort(-quality_values))
    spearman = float(np.corrcoef(frozen_rank, quality_rank)[0, 1]) if len(common) > 1 else math.nan
    result = {
        "representations": common,
        "frozen_residual": {item: float(frozen[item]["mean"]) for item in common},
        "final_normal_mean_deg": {item: float(final[item]["normal_mean_deg"]) for item in common},
        "spearman_frozen_residual_vs_final_error": spearman,
        "note": "Lower frozen residual and lower native angular error are better.",
    }
    (output_root / "mechanism_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--train-root", default=str(DEFAULT_TRAIN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.aggregate and not args.representation:
        parser.error("choose --representation or --aggregate")
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_representation(args)


if __name__ == "__main__":
    main()
