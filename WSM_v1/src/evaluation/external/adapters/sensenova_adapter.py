#!/usr/bin/env python3
"""Run SenseNova-Vision's official dense-perception depth interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


DEPTH_PROMPT = (
    "<image> Estimate relative depth for each pixel in the image, with closer "
    "objects appearing brighter and distant objects appearing darker. Output is "
    "a grayscale image with pixel values ranging from 0-255."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max-mem-per-gpu", default="42GiB")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    sample_manifest = Path(args.samples).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve() / "depth_estimation"
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    if not (repo / "inference" / "sensenova_vision.py").is_file():
        raise SystemExit(f"Invalid SenseNova-Vision repository: {repo}")
    if not (model_path / "ema.safetensors").is_file():
        raise SystemExit(f"Incomplete SenseNova-Vision model: {model_path}")
    sys.path.insert(0, str(repo))

    from inference.sensenova_vision import SenseNovaVisionModel

    samples = [
        row for row in read_jsonl(sample_manifest)
        if str(row.get("task")) == "depth_estimation"
    ]
    if not samples:
        raise SystemExit(f"No depth_estimation records in {sample_manifest}")
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    output_root.mkdir(parents=True, exist_ok=True)
    model = SenseNovaVisionModel(
        model_path=str(model_path),
        dtype=args.dtype,
        device="cuda",
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_folder=str(output_root.parent / "offload"),
        seed=args.seed,
        download_local_files_only=True,
    )

    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        sample_id = str(sample.get("sample_id") or "")
        input_value = str(sample.get("input_path") or sample.get("condition_path") or "")
        if not sample_id or not input_value:
            raise ValueError(f"Depth record {index} needs sample_id and input_path")
        input_path = resolve(input_value, sample_manifest.parent)
        output_path = output_root / f"{sample_id}.npy"
        preview_path = output_root / f"{sample_id}.png"
        if args.overwrite or not output_path.is_file():
            image = model.generate(
                question=DEPTH_PROMPT,
                images=[str(input_path)],
                mode="dense_perception",
                noise_seed=args.seed + index - 1,
            )
            image.convert("L").save(preview_path)
            values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            np.save(output_path, values, allow_pickle=False)
        predictions.append({
            "sample_id": sample_id,
            "prediction_path": str(output_path),
            "preview_path": str(preview_path),
            "adapter": "sensenova_official_dense_perception_depth",
            "checkpoint": "ema.safetensors",
            "dtype": args.dtype,
            "hardware_compatibility": "Turing FP16 SDPA fallback",
            "prompt": DEPTH_PROMPT,
        })
        print(f"[{index}/{len(samples)}] {sample_id} -> {output_path}", flush=True)

    atomic_jsonl(prediction_manifest, predictions)
    print(f"Wrote {len(predictions)} predictions: {prediction_manifest}", flush=True)


if __name__ == "__main__":
    main()
