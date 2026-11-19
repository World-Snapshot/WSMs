#!/usr/bin/env python3
"""Run the official Scaling Diffusion Perception generalist MoE on JSONL samples."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY_COMMIT = "cb9f2c72ec8828d437293e73126938362a7e4fba"
MODEL_REPOSITORY_REVISION = "1d5f70710298aa9bd5406cfd2bc7e78e36066965"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def deterministic_seed_sequence(initial_seed: int, length: int) -> list[int]:
    generator = random.Random(initial_seed)
    return [
        generator.randint(-0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF)
        for _ in range(length)
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sd-root", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.ensemble_size < 1:
        raise ValueError("--steps and --ensemble-size must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    repo = Path(args.repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    sd_root = Path(args.sd_root).expanduser().resolve()
    sample_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    for required in (repo, checkpoint, sd_root, sample_path):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from accelerate import Accelerator
    from PIL import Image
    from torchvision.transforms import InterpolationMode, Resize
    from torchvision.transforms.functional import pil_to_tensor

    from models import find_model
    from pipeline.moe_pipeline import MoEPipeline
    from pipeline.util.ensemble import ensemble_depth
    from utils.depth_utils import colorize_depth_maps
    from utils.model_utils import update_model

    rows = [row for row in read_jsonl(sample_path) if row.get("task") == "depth_estimation"]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    indexed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    if not indexed_rows:
        raise ValueError("No depth samples selected for this shard")

    accelerator = Accelerator()
    device = accelerator.device
    if not str(device).startswith(args.device.split(":", 1)[0]):
        raise RuntimeError(f"Requested {args.device}, accelerator selected {device}")

    model = MoEPipeline.from_pretrained(
        str(sd_root),
        True,
        model_path=str(checkpoint),
        model_type="DiTMultiTaskMoE_uc",
        num_exps=args.num_experts,
        top_k=args.top_k,
        scale_invariant=True,
        shift_invariant=True,
        causal_attn=False,
        mean_pool_gen=False,
        beta_schedule="squaredcos_cap_v2",
        extra_routing_util=None,
    )
    # The released checkpoint was trained with the original Stable Diffusion 2
    # v-prediction scheduler. Some public SD2 mirrors expose an epsilon scheduler
    # despite otherwise compatible components; that silently produces severe
    # high-frequency noise throughout the DDIM trajectory.
    expected_scheduler = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
    }
    for key, expected in expected_scheduler.items():
        actual = getattr(model.scheduler.config, key)
        if actual != expected:
            raise RuntimeError(
                f"Incompatible SD2 scheduler {key}: expected {expected!r}, got {actual!r}"
            )
    model.scheduler.register_to_config(prediction_type="v_prediction")
    if model.scheduler.config.prediction_type != "v_prediction":
        raise RuntimeError("Failed to restore the official SD2 v-prediction scheduler")

    update_model(model, accelerator)
    state_dict = find_model(str(checkpoint))
    missing_keys, unexpected_keys = model.unet.load_state_dict(state_dict)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    del state_dict
    gc.collect()
    model.unet.eval()
    model.vae.eval()

    resize = Resize((512, 512), interpolation=InterpolationMode.NEAREST_EXACT)
    seed_sequence = deterministic_seed_sequence(
        args.seed, len(rows) * args.ensemble_size
    )
    predictions: list[dict[str, Any]] = []
    runtimes: list[float] = []

    for local_index, (global_index, sample) in enumerate(indexed_rows, 1):
        sample_id = str(sample["sample_id"])
        input_path = resolve(str(sample["input_path"]), sample_path.parent)
        target_dir = output_dir / "depth_estimation"
        output_path = target_dir / f"{sample_id}.npy"
        visual_path = target_dir / f"{sample_id}.png"
        elapsed = 0.0
        used_seeds = [
            seed_sequence[-1 - (global_index * args.ensemble_size + member)]
            for member in range(args.ensemble_size)
        ]

        if args.overwrite or not output_path.is_file():
            image = Image.open(input_path).convert("RGB")
            rgb = resize(pil_to_tensor(image).unsqueeze(0)).to(
                device=device, dtype=torch.float32
            )
            rgb = rgb / 255.0 * 2.0 - 1.0
            # The official loader passes a GT-validity mask as this auxiliary
            # image. Use an all-valid mask so evaluation GT is never exposed.
            prompt = torch.ones_like(rgb, dtype=torch.float32, device=device)
            label = torch.tensor([1000], dtype=torch.long, device=device)
            ensemble_members = torch.zeros(
                (args.ensemble_size, 1, 512, 512),
                dtype=torch.float32,
                device=device,
            )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for member, seed in enumerate(used_seeds):
                generator = torch.Generator(device=device).manual_seed(seed)
                prediction = model.single_infer(
                    rgb,
                    prompt,
                    label,
                    num_inference_steps=args.steps,
                    generator=generator,
                    show_pbar=False,
                    device=device,
                )
                ensemble_members[member] = prediction.mean(dim=1)
            depth, _ = ensemble_depth(
                ensemble_members,
                scale_invariant=True,
                shift_invariant=True,
                max_res=50,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            scalar = depth.squeeze().detach().cpu().numpy().astype(np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, scalar)
            colored = colorize_depth_maps(scalar, 0.0, 1.0)
            colored = np.moveaxis(colored.squeeze(), 0, -1)
            Image.fromarray(np.clip(colored * 255.0, 0, 255).astype(np.uint8)).save(
                visual_path
            )
        runtimes.append(elapsed)
        predictions.append(
            {
                "sample_id": sample_id,
                "prediction_path": str(output_path),
                "adapter": "scaling_diffusion_perception_official_moe",
                "prediction_representation": "affine_invariant_relative_depth_0_1",
                "scalar_encoding": "npy_float32",
                "steps": args.steps,
                "ensemble_size": args.ensemble_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "prompt_mask": "all_valid_no_gt_mask",
                "seeds": used_seeds,
                "inference_seconds": elapsed,
                "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
                "model_repository_revision": MODEL_REPOSITORY_REVISION,
            }
        )
        print(
            f"[{local_index}/{len(indexed_rows)}] {sample_id}: {elapsed:.3f}s -> {output_path}",
            flush=True,
        )

    atomic_jsonl(prediction_manifest, predictions)
    nonzero = [value for value in runtimes if value > 0]
    total = float(sum(nonzero))
    atomic_json(
        prediction_manifest.with_suffix(".runtime.json"),
        {
            "method": "Scaling Diffusion Perception",
            "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
            "model_repository_revision": MODEL_REPOSITORY_REVISION,
            "checkpoint_sha256": sha256(checkpoint),
            "samples": len(predictions),
            "steps": args.steps,
            "ensemble_size": args.ensemble_size,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "prompt_mask": "all_valid_no_gt_mask",
            "total_inference_seconds": total,
            "inference_fps": len(nonzero) / total if total else None,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
    )


if __name__ == "__main__":
    main()
