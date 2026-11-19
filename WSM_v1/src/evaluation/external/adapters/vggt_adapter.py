#!/usr/bin/env python3
"""Run the official VGGT aggregator and depth head on frozen RGB samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
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
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if not (repo / "vggt" / "models" / "vggt.py").is_file():
        raise SystemExit(f"Invalid official VGGT repository: {repo}")
    sys.path.insert(0, str(repo))

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    sample_manifest = Path(args.samples).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve() / "depth_estimation"
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    samples = [
        row for row in read_jsonl(sample_manifest)
        if str(row.get("task")) == "depth_estimation"
    ]
    if not samples:
        raise SystemExit(f"No depth_estimation records in {sample_manifest}")

    device = torch.device(args.device)
    major, _ = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    model = VGGT.from_pretrained(args.model_id).eval().to(device)
    predictions = []
    for index, sample in enumerate(samples, 1):
        sample_id = str(sample.get("sample_id") or "")
        input_value = str(sample.get("input_path") or sample.get("condition_path") or "")
        if not sample_id or not input_value:
            raise ValueError(f"Depth record {index} needs sample_id and input_path")
        input_path = resolve(input_value, sample_manifest.parent)
        output_path = output_root / f"{sample_id}.npy"
        if args.overwrite or not output_path.is_file():
            images = load_and_preprocess_images([str(input_path)]).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=dtype
            ):
                batched = images[None]
                tokens, patch_start_idx = model.aggregator(batched)
                depth, _ = model.depth_head(
                    tokens, images=batched, patch_start_idx=patch_start_idx
                )
            values = depth.detach().float().cpu().numpy().squeeze()
            if values.ndim != 2:
                raise RuntimeError(f"{sample_id}: VGGT depth shape={values.shape}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, values.astype(np.float32), allow_pickle=False)
            del images, batched, tokens, depth
        predictions.append({
            "sample_id": sample_id,
            "prediction_path": str(output_path),
            "adapter": "vggt_official_depth_head",
            "model_id": args.model_id,
        })
        print(f"[{index}/{len(samples)}] {sample_id} -> {output_path}", flush=True)
    atomic_jsonl(prediction_manifest, predictions)
    print(f"Wrote {len(predictions)} predictions: {prediction_manifest}")


if __name__ == "__main__":
    main()
