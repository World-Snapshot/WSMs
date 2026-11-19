#!/usr/bin/env python3
"""Run SCHP-LIP and map its 20 human-parsing classes to channel 103."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download


# Frozen LIP-20 -> channel-103 mapping:
# background, hair, body skin, face skin, clothes, accessories.
LIP_TO_CHANNEL103 = np.asarray([
    0, 5, 1, 5, 5, 4, 4, 4, 4, 4,
    4, 5, 4, 3, 2, 2, 2, 2, 4, 4,
], dtype=np.uint8)

CHANNEL103_PALETTE = np.asarray([
    [0, 0, 0], [0, 0, 255], [255, 255, 0],
    [0, 255, 255], [0, 255, 0], [255, 0, 255],
], dtype=np.uint8)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
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
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--model", default="pirocheto/schp-lip-20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_manifest = Path(args.samples).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve() / "semantic_estimation"
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = [
        row for row in read_jsonl(sample_manifest)
        if str(row.get("task")) == "semantic_estimation"
    ]
    if not samples:
        raise SystemExit(f"No semantic_estimation records in {sample_manifest}")

    # The converted checkpoint's remote-code metadata uses a top-level sibling
    # import that Transformers cannot resolve in its dynamic-module cache. Load
    # the exact downloaded snapshot as a regular local Python package instead.
    snapshot = Path(snapshot_download(repo_id=args.model)).resolve()
    sys.path.insert(0, str(snapshot))
    from image_processing_schp import SCHPImageProcessor
    from modeling_schp import SCHPForSemanticSegmentation

    processor = SCHPImageProcessor.from_pretrained(snapshot)
    model = SCHPForSemanticSegmentation.from_pretrained(snapshot).eval().to(args.device)

    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, 1):
            sample_id = str(sample["sample_id"])
            input_path = resolve(str(sample["input_path"]), sample_manifest.parent)
            output_path = output_root / f"{sample_id}.png"
            elapsed = 0.0
            if args.overwrite or not output_path.is_file():
                image = Image.open(input_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                outputs = model(**inputs)
                logits = getattr(outputs, "parsing_logits", outputs.logits)
                logits = F.interpolate(
                    logits, size=(image.height, image.width), mode="bilinear",
                    align_corners=False,
                )
                lip_labels = logits.argmax(dim=1)[0].cpu().numpy()
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - started
                mapped = LIP_TO_CHANNEL103[lip_labels]
                Image.fromarray(CHANNEL103_PALETTE[mapped], mode="RGB").save(output_path)
            predictions.append({
                "sample_id": sample_id,
                "task": "semantic_estimation",
                "method_id": "schp_lip",
                "prediction_path": str(output_path),
                "adapter": "schp_lip20_to_channel103_frozen_mapping",
                "checkpoint": args.model,
                "inference_seconds": elapsed,
                "lip_to_channel103": LIP_TO_CHANNEL103.tolist(),
            })
            print(f"[{index}/{len(samples)}] {sample_id}: {elapsed:.4f}s", flush=True)

    atomic_jsonl(prediction_manifest, predictions)
    print(f"Wrote {len(predictions)} predictions: {prediction_manifest}", flush=True)


if __name__ == "__main__":
    main()
