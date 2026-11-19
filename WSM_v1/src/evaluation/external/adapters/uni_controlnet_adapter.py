#!/usr/bin/env python3
"""Batch the official Uni-ControlNet Canny inference on the frozen manifest."""

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
NEGATIVE = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality"
)


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


def resize_64(image: np.ndarray, resolution: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = resolution / min(height, width)
    height = int(round(height * scale / 64.0)) * 64
    width = int(round(width * scale / 64.0)) * 64
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


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
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--low-threshold", type=int, default=100)
    parser.add_argument("--high-threshold", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    sample_manifest = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from models.ddim_hacked import DDIMSampler
    from models.util import create_model, load_state_dict

    model = create_model(str(repo / "configs/uni_v15.yaml")).cpu()
    missing, unexpected = model.load_state_dict(
        load_state_dict(str(checkpoint), location="cpu"), strict=False
    )
    print(
        f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )
    model = model.cuda().eval()
    sampler = DDIMSampler(model)

    rows = [
        row for row in read_jsonl(sample_manifest)
        if row.get("task") == "conditional_rgb"
    ]
    if len(rows) != 24:
        raise ValueError(f"Expected 24 conditional_rgb samples, found {len(rows)}")
    predictions: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, 1):
        sample_id = str(row["sample_id"])
        condition_path = resolve(str(row["condition_path"]), sample_manifest.parent)
        image = cv2.imread(str(condition_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(condition_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_64(image, args.resolution)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, args.low_threshold, args.high_threshold)
        edge = np.repeat(edge[:, :, None], 3, axis=2)

        local = np.concatenate(
            [edge] + [np.zeros_like(edge) for _ in range(6)], axis=2
        )
        local_control = torch.from_numpy(local).float().cuda() / 255.0
        local_control = einops.rearrange(
            local_control.unsqueeze(0), "b h w c -> b c h w"
        ).clone()
        global_control = torch.zeros((1, 768), device="cuda")
        prompt = str(row.get("prompt") or "a high quality photo of a person")
        seed = args.seed + index - 1
        seed_all(seed)
        cond = {
            "local_control": [local_control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + POSITIVE])
            ],
            "global_control": [global_control],
        }
        uncond = {
            "local_control": [local_control],
            "c_crossattn": [model.get_learned_conditioning([NEGATIVE])],
            "global_control": [global_control],
        }
        model.control_scales = [args.control_strength] * 13
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            samples, _ = sampler.sample(
                args.steps,
                1,
                (4, height // 8, width // 8),
                cond,
                verbose=False,
                eta=0.0,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=uncond,
                global_strength=1.0,
            )
            decoded = model.decode_first_stage(samples)
        array = (
            einops.rearrange(decoded, "b c h w -> b h w c")[0] * 127.5 + 127.5
        ).detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        prediction_path = output_dir / f"{sample_id}.png"
        Image.fromarray(array).save(prediction_path)
        predictions.append({
            "sample_id": sample_id,
            "prediction_path": str(prediction_path),
            "adapter": "uni_controlnet_official",
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
