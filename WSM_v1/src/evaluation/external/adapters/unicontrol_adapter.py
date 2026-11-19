#!/usr/bin/env python3
"""Batch the official Salesforce UniControl v1.1 Canny inference."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from PIL import Image


POSITIVE = "best quality, extremely detailed"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
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


def square_resize(image: np.ndarray, resolution: int) -> np.ndarray:
    height, width = image.shape[:2]
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    image = image[top:top + crop, left:left + crop]
    interpolation = cv2.INTER_LANCZOS4 if crop < resolution else cv2.INTER_AREA
    return cv2.resize(image, (resolution, resolution), interpolation=interpolation)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--control-strength", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    # Do not resolve symlinks here: the official release uses a `.st` suffix,
    # while its loader recognizes the same file as safetensors by extension.
    checkpoint = Path(args.checkpoint).expanduser().absolute()
    sample_manifest = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from cldm.ddim_unicontrol_hacked import DDIMSampler
    from cldm.model import create_model, load_state_dict

    model = create_model(str(repo / "models/cldm_v15_unicontrol.yaml")).cpu()
    missing, unexpected = model.load_state_dict(
        load_state_dict(str(checkpoint), location="cpu"), strict=False
    )
    print(
        f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )
    model = model.cuda().eval()
    sampler = DDIMSampler(model)
    task_instruction = model.get_learned_conditioning(["canny edge to image"])[:, :1, :]

    rows = [
        row for row in read_jsonl(sample_manifest)
        if row.get("task") == "conditional_rgb"
    ]
    if len(rows) != 24:
        raise ValueError(f"Expected 24 conditional_rgb samples, found {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict] = []

    for index, row in enumerate(rows, 1):
        sample_id = str(row["sample_id"])
        condition_path = resolve(str(row["condition_path"]), sample_manifest.parent)
        image = cv2.imread(str(condition_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(condition_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = square_resize(image, args.resolution).astype(np.float32) / 255.0
        control = torch.from_numpy(image).float().cuda().unsqueeze(0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        seed = args.seed + index - 1
        seed_all(seed)
        prompt = str(row.get("prompt") or "a high quality photo of a person")
        task = {"name": "control_canny", "feature": task_instruction}
        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + POSITIVE])
            ],
            "task": task,
        }
        uncond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([""])],
        }
        model.control_scales = [args.control_strength] * 13
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            samples, _ = sampler.sample(
                args.steps,
                1,
                (4, args.resolution // 8, args.resolution // 8),
                cond,
                verbose=False,
                eta=0.0,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=uncond,
            )
            decoded = model.decode_first_stage(samples)
        array = (
            torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)[0]
            .permute(1, 2, 0)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        prediction_path = output_dir / f"{sample_id}.png"
        Image.fromarray((array * 255).round().astype(np.uint8)).save(prediction_path)
        predictions.append({
            "sample_id": sample_id,
            "prediction_path": str(prediction_path),
            "adapter": "salesforce_unicontrol_v1_1_official",
            "seed": seed,
            "steps": args.steps,
            "resolution": args.resolution,
            "guidance_scale": args.guidance_scale,
            "control_strength": args.control_strength,
        })
        print(f"[{index}/24] {sample_id} -> {prediction_path}", flush=True)

    atomic_jsonl(prediction_manifest, predictions)
    print(f"Wrote 24 predictions: {prediction_manifest}", flush=True)


if __name__ == "__main__":
    main()
