#!/usr/bin/env python3
"""Run official released Diffusers-compatible baselines on a frozen JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_TASK = {
    "controlnet": "conditional_rgb",
    "controlnet_ip_adapter": "conditional_rgb",
    "t2i_adapter": "conditional_rgb",
    "marigold": "depth_estimation",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def deterministic_seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def load_common():
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "This adapter must run in the prepared method environment: " + str(exc)
        ) from exc
    return torch, Image


def resize_generation_input(image, Image):
    width = max(64, image.width - image.width % 8)
    height = max(64, image.height - image.height % 8)
    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def model_source(model_root: Path | None, repo_id: str) -> str:
    if model_root is not None:
        candidate = model_root / repo_id.replace("/", "--")
        if candidate.is_dir():
            return str(candidate)
    return repo_id


def build_controlnet(device: str, dtype_name: str, model_root: Path | None):
    torch, _ = load_common()
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    from diffusers import UniPCMultistepScheduler

    dtype = getattr(torch, dtype_name)
    control = ControlNetModel.from_pretrained(
        model_source(model_root, "lllyasviel/control_v11p_sd15_canny"),
        torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_source(model_root, "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        controlnet=control,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def build_controlnet_ip_adapter(
    device: str,
    dtype_name: str,
    model_root: Path | None,
    scale: float,
):
    """Combine released SD1.5 Canny ControlNet and released IP-Adapter."""
    pipe = build_controlnet(device, dtype_name, model_root)
    pipe.load_ip_adapter(
        model_source(model_root, "h94/IP-Adapter"),
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )
    pipe.set_ip_adapter_scale(float(scale))
    return pipe


def build_t2i_adapter(device: str, dtype_name: str, model_root: Path | None):
    torch, _ = load_common()
    from diffusers import (
        AutoencoderKL,
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLAdapterPipeline,
        T2IAdapter,
    )

    dtype = getattr(torch, dtype_name)
    adapter = T2IAdapter.from_pretrained(
        model_source(model_root, "TencentARC/t2i-adapter-canny-sdxl-1.0"),
        torch_dtype=dtype,
        variant="fp16" if dtype_name == "float16" else None,
    )
    vae = AutoencoderKL.from_pretrained(
        model_source(model_root, "madebyollin/sdxl-vae-fp16-fix"),
        torch_dtype=dtype
    )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        model_source(model_root, "stabilityai/stable-diffusion-xl-base-1.0"),
        subfolder="scheduler",
    )
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_source(model_root, "stabilityai/stable-diffusion-xl-base-1.0"),
        vae=vae,
        adapter=adapter,
        scheduler=scheduler,
        torch_dtype=dtype,
        variant="fp16" if dtype_name == "float16" else None,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def build_marigold(device: str, dtype_name: str, model_root: Path | None):
    torch, _ = load_common()
    from diffusers import MarigoldDepthPipeline

    dtype = getattr(torch, dtype_name)
    kwargs = {"torch_dtype": dtype}
    if dtype_name == "float16":
        kwargs["variant"] = "fp16"
    pipe = MarigoldDepthPipeline.from_pretrained(
        model_source(model_root, "prs-eth/marigold-depth-v1-1"), **kwargs
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_generation(
    method: str,
    pipe,
    sample: dict[str, Any],
    base: Path,
    output_path: Path,
    device: str,
    seed: int,
    steps: int,
    guidance: float,
    ip_adapter_scale: float,
) -> None:
    torch, Image = load_common()
    condition_value = str(sample.get("condition_path") or sample.get("input_path") or "")
    if not condition_value:
        raise ValueError(f"{sample['sample_id']}: condition_path is required")
    condition = Image.open(resolve(condition_value, base)).convert("RGB")
    if method == "t2i_adapter":
        # The released SDXL Canny adapter is trained and demonstrated at 1024px.
        # Our frozen conditions are already edge maps, so keep them binary and
        # use nearest-neighbour resizing instead of blurring the contours.
        condition = condition.convert("L").point(lambda value: 255 if value >= 128 else 0)
        condition = condition.resize((1024, 1024), Image.Resampling.NEAREST).convert("RGB")
    else:
        condition = resize_generation_input(condition, Image)
    prompt = str(sample.get("prompt") or "a high quality photo")
    generator = torch.Generator(device=device).manual_seed(
        deterministic_seed(str(sample["sample_id"]), seed)
    )
    common = {
        "prompt": prompt,
        "image": condition,
        "height": condition.height,
        "width": condition.width,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    if method in {"controlnet", "controlnet_ip_adapter"}:
        common["controlnet_conditioning_scale"] = 1.0
    else:
        common["negative_prompt"] = (
            "anime, cartoon, graphic, text, painting, crayon, graphite, "
            "abstract, glitch, deformed, mutated, ugly, disfigured"
        )
        common["adapter_conditioning_scale"] = 0.8
        common["adapter_conditioning_factor"] = 1.0
    if method == "controlnet_ip_adapter":
        reference_value = str(sample.get("reference_path") or "")
        if not reference_value:
            raise ValueError(
                f"{sample['sample_id']}: reference_path is required for "
                "ControlNet + IP-Adapter"
            )
        reference = Image.open(resolve(reference_value, base)).convert("RGB")
        common["ip_adapter_image"] = reference
    image = pipe(**common).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run_marigold(
    pipe,
    sample: dict[str, Any],
    base: Path,
    output_path: Path,
    steps: int,
    ensemble_size: int,
) -> None:
    import numpy as np

    _, Image = load_common()
    input_value = str(sample.get("input_path") or sample.get("condition_path") or "")
    if not input_value:
        raise ValueError(f"{sample['sample_id']}: input_path is required")
    image = Image.open(resolve(input_value, base)).convert("RGB")
    result = pipe(
        image,
        num_inference_steps=steps,
        ensemble_size=ensemble_size,
        output_type="np",
    )
    prediction = getattr(result, "prediction", None)
    if prediction is None:
        prediction = result[0]
    prediction = np.asarray(prediction, dtype=np.float32).squeeze()
    if prediction.ndim != 2:
        raise RuntimeError(
            f"{sample['sample_id']}: expected HxW depth, got {prediction.shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prediction, allow_pickle=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(SUPPORTED_TASK))
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--model-root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    model_root = Path(args.model_root).expanduser().resolve() if args.model_root else None
    all_samples = read_jsonl(sample_path)
    expected_task = SUPPORTED_TASK[args.method]
    samples = [row for row in all_samples if str(row.get("task")) == expected_task]
    if not samples:
        raise SystemExit(f"No {expected_task} records in {sample_path}")
    if args.method == "controlnet":
        pipe = build_controlnet(args.device, args.dtype, model_root)
        default_steps, guidance = 30, 7.5
    elif args.method == "controlnet_ip_adapter":
        pipe = build_controlnet_ip_adapter(
            args.device, args.dtype, model_root, args.ip_adapter_scale
        )
        default_steps, guidance = 30, 7.5
    elif args.method == "t2i_adapter":
        pipe = build_t2i_adapter(args.device, args.dtype, model_root)
        default_steps, guidance = 30, 7.5
    else:
        pipe = build_marigold(args.device, args.dtype, model_root)
        default_steps, guidance = 1, 0.0
    steps = args.steps or default_steps
    rows: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    for index, sample in enumerate(samples, 1):
        sample_id = str(sample.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"Sample {index} has no sample_id")
        suffix = ".npy" if args.method == "marigold" else ".png"
        output_path = output_dir / expected_task / f"{sample_id}{suffix}"
        if args.overwrite or not output_path.is_file():
            if args.device.startswith("cuda"):
                torch, _ = load_common()
                torch.cuda.synchronize()
            started = time.perf_counter()
            if args.method == "marigold":
                run_marigold(
                    pipe, sample, sample_path.parent, output_path,
                    steps, args.ensemble_size
                )
            else:
                run_generation(
                    args.method, pipe, sample, sample_path.parent, output_path,
                    args.device, args.seed, steps, guidance, args.ip_adapter_scale
                )
            if args.device.startswith("cuda"):
                torch, _ = load_common()
                torch.cuda.synchronize()
            inference_seconds.append(time.perf_counter() - started)
        else:
            inference_seconds.append(0.0)
        rows.append({
            "sample_id": sample_id,
            "prediction_path": str(output_path),
            "adapter": args.method,
            "seed": deterministic_seed(sample_id, args.seed),
            "inference_seconds": inference_seconds[-1],
        })
        print(f"[{index}/{len(samples)}] {sample_id} -> {output_path}", flush=True)
    atomic_jsonl(prediction_manifest, rows)
    atomic_json(
        output_dir / "adapter_runtime.json",
        {
            "adapter": args.method,
            "sample_manifest": str(sample_path),
            "prediction_manifest": str(prediction_manifest),
            "samples": len(rows),
            "device": args.device,
            "dtype": args.dtype,
            "steps": steps,
            "ensemble_size": args.ensemble_size if args.method == "marigold" else 1,
            "seed": args.seed,
            "ip_adapter_scale": (
                args.ip_adapter_scale
                if args.method == "controlnet_ip_adapter" else None
            ),
            "total_inference_seconds": float(sum(inference_seconds)),
            "inference_fps": (
                float(len(rows) / sum(inference_seconds))
                if sum(inference_seconds) > 0 else None
            ),
        },
    )
    print(f"Wrote {prediction_manifest}")


if __name__ == "__main__":
    main()
