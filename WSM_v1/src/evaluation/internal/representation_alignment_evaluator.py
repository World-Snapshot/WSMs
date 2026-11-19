#!/usr/bin/env python
"""Controlled evaluation of target-representation alignment.

The experiment keeps a task, its data, and the frozen FLUX.2 backbone fixed.
It applies exactly invertible, norm-preserving changes of coordinates to the
packed target latent, then measures the frozen backbone's native flow-matching
residual.  Because the target and noise receive the same transform, every
variant contains exactly the same information and follows the same diffusion
path in a different coordinate system.

This file implements the pre-training stage of the experiment.  Its output is
an ordinary CSV/JSON table containing round-trip error and mean frozen-backbone
residual for each representation.  A later training comparison can use the
same named transforms without changing their definition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUBPOSITIONS_PER_VAE_CHANNEL = 4
CHANNEL_CYCLE_SHIFTS = {
    "vae-channel-cycle-01": 1,
    "vae-channel-cycle-03": 3,
    "vae-channel-cycle-07": 7,
    "vae-channel-cycle-15": 15,
}
TRANSFORM_NAMES = ("native", *CHANNEL_CYCLE_SHIFTS)


def _roll_vae_channels(tensor: torch.Tensor, shift: int) -> torch.Tensor:
    """Permute VAE channels while preserving each packed 2x2 subposition.

    FLUX.2 packs the final feature dimension as ``vae_channel * 4 +
    subposition``.  Reshaping to ``[..., C, 4]`` and rolling only ``C`` keeps
    every spatial subposition and every token in place.  For the 32-channel
    FLUX.2 VAE, each odd shift is one complete 32-cycle, so all non-native
    candidates have the same cycle structure and rank(P-I)=31.
    """
    feature_count = int(tensor.shape[-1])
    if feature_count % SUBPOSITIONS_PER_VAE_CHANNEL:
        raise ValueError(
            f"Packed feature count {feature_count} is not divisible by "
            f"{SUBPOSITIONS_PER_VAE_CHANNEL}"
        )
    channel_count = feature_count // SUBPOSITIONS_PER_VAE_CHANNEL
    unpacked = tensor.reshape(*tensor.shape[:-1], channel_count, SUBPOSITIONS_PER_VAE_CHANNEL)
    return torch.roll(unpacked, shifts=int(shift), dims=-2).reshape_as(tensor)


def apply_representation_transform(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Map a packed target/noise tensor into a named reversible representation."""
    if name not in TRANSFORM_NAMES:
        raise ValueError(f"Unknown representation {name!r}; choose from {TRANSFORM_NAMES}")
    if name == "native":
        return tensor.clone()
    return _roll_vae_channels(tensor, CHANNEL_CYCLE_SHIFTS[name])


def invert_representation_transform(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Return a named representation to the native packed coordinate system."""
    if name not in TRANSFORM_NAMES:
        raise ValueError(f"Unknown representation {name!r}; choose from {TRANSFORM_NAMES}")
    if name == "native":
        return tensor.clone()
    return _roll_vae_channels(tensor, -CHANNEL_CYCLE_SHIFTS[name])


@dataclass(frozen=True)
class RepresentationResult:
    representation: str
    samples: int
    evaluations: int
    roundtrip_max_abs: float
    relative_norm_change: float
    frozen_flow_mse_mean: float
    frozen_flow_mse_median: float
    frozen_flow_mse_sample_std: float
    frozen_clean_mse_mean: float


def training_timestep_quadrature(
    points: int,
    *,
    shift: float = 0.0,
    scale: float = 1.0,
    high_noise_anchor: bool = True,
    anchor_probability: float = 0.5,
    anchor_timestep: float = 0.999,
) -> list[tuple[float, float]]:
    """Deterministically integrate the unchanged training timestep mixture.

    The trainer draws ``sigmoid(N(0,1) * scale + shift)`` and, for the dense
    task recipes, replaces a fraction with the high-noise anchor.  Gauss-Hermite
    nodes measure that exact distribution without changing the trainer or
    introducing Monte-Carlo noise into the compatibility ranking.
    """
    if points < 2:
        raise ValueError("quadrature points must be at least 2")
    if not 0.0 <= anchor_probability < 1.0:
        raise ValueError("anchor probability must lie in [0, 1)")
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    base_mass = 1.0 - anchor_probability if high_noise_anchor else 1.0
    schedule = []
    for node, weight in zip(nodes, weights):
        normal_value = math.sqrt(2.0) * float(node)
        timestep = 1.0 / (1.0 + math.exp(-(normal_value * scale + shift)))
        schedule.append((timestep, base_mass * float(weight) / math.sqrt(math.pi)))
    if high_noise_anchor:
        schedule.append((float(anchor_timestep), float(anchor_probability)))
    total = sum(weight for _timestep, weight in schedule)
    return [(timestep, weight / total) for timestep, weight in schedule]


def fixed_timestep_schedule(value: str) -> list[tuple[float, float]]:
    timesteps = [float(item) for item in value.split(",") if item.strip()]
    if not timesteps or any(not 0.0 < item < 1.0 for item in timesteps):
        raise ValueError("--timesteps must contain values strictly between 0 and 1")
    weight = 1.0 / len(timesteps)
    return [(timestep, weight) for timestep in timesteps]


def audit_representation(tensor: torch.Tensor, name: str) -> tuple[float, float]:
    transformed = apply_representation_transform(tensor, name)
    restored = invert_representation_transform(transformed, name)
    restored_audit = restored.double()
    native_audit = tensor.double()
    transformed_audit = transformed.double()
    roundtrip = float((restored_audit - native_audit).abs().max().item())
    native_norm = native_audit.norm().clamp_min(1e-12)
    relative_norm = float(
        ((transformed_audit.norm() - native_norm).abs() / native_norm).item()
    )
    return roundtrip, relative_norm


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(results: Iterable[RepresentationResult], path: Path) -> None:
    import csv

    rows = [asdict(result) for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def _normalization_pack(pipe: Any, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if latents.ndim == 5:
        latents = latents[:, :, 0]
    latents = pipe._patchify_latents(latents)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    variance = pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    epsilon = float(pipe.vae.config.batch_norm_eps)
    latents = (latents - mean) / torch.sqrt(variance + epsilon)
    ids = pipe._prepare_latent_ids(latents).to(latents.device)
    return pipe._pack_latents(latents), ids


def _load_task(args: argparse.Namespace):
    from scripts.train_TunnelingNet_Flux2_4B_wsg import WSGOminiLatentDataset

    config = json.loads(args.modality_config.read_text(encoding="utf-8"))
    key = f"Channel_Head_{args.channel_head_index:06d}"
    entry = config["wsg_v1"][key]
    raw_conditions = entry.get("condition")
    if isinstance(raw_conditions, str):
        conditions = [raw_conditions]
    else:
        conditions = list(raw_conditions or [])
    conditions = [value for value in conditions if value not in {"caption", "Reference_IMG"}]
    if len(conditions) != 1:
        raise ValueError(
            "The first controlled experiment requires exactly one dense visual condition; "
            f"{key} declares {conditions}"
        )
    target = str(entry["target"])
    condition = str(conditions[0])
    whitelist_map = config.get("target_dataset_whitelist_by_modality") or {}
    whitelist = set(whitelist_map.get(target) or []) or None
    dataset = WSGOminiLatentDataset(
        args.dense_latent_cache,
        target,
        condition,
        None,
        None,
        None,
        None,
        None,
        "tolerant",
        False,
        0,
        1,
        args.samples,
        entry,
        args.text_embedding_cache,
        split=args.split,
        val_split_ratio=0.05,
        target_dataset_whitelist=whitelist,
        lazy_latents=True,
    )
    return entry, dataset


@torch.no_grad()
def run_frozen_compatibility(args: argparse.Namespace) -> list[RepresentationResult]:
    from diffusers import Flux2KleinPipeline
    from scripts.train_TunnelingNet_Flux2_4B_wsg import (
        flux2_tunneling_forward,
        set_condition_attention_mode,
        set_trainable_condition,
        set_trainable_target,
        specify_lora,
        task_recipe,
    )

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    entry, dataset = _load_task(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    pipe = Flux2KleinPipeline.from_pretrained(
        str(args.model), torch_dtype=dtype, low_cpu_mem_usage=True
    )
    pipe.transformer.to(device).requires_grad_(False).eval()
    pipe.text_encoder.to("cpu").requires_grad_(False).eval()
    pipe.vae.to("cpu").requires_grad_(False).eval()
    set_condition_attention_mode("modality_tunneling")
    set_trainable_target(False)
    set_trainable_condition(False)

    if args.timestep_mode == "training":
        recipe = task_recipe(str(entry.get("task_type") or ""))
        schedule = training_timestep_quadrature(
            args.quadrature_points,
            shift=0.0,
            scale=1.0,
            high_noise_anchor=bool(recipe.get("high_noise_anchor", False)),
            anchor_probability=args.anchor_probability,
            anchor_timestep=args.anchor_timestep,
        )
    else:
        schedule = fixed_timestep_schedule(args.timesteps)

    flow_losses: dict[str, list[float]] = {name: [] for name in TRANSFORM_NAMES}
    clean_losses: dict[str, list[float]] = {name: [] for name in TRANSFORM_NAMES}
    sample_flow_losses: dict[str, list[float]] = {name: [] for name in TRANSFORM_NAMES}
    timestep_flow_losses: dict[str, dict[float, list[float]]] = {
        name: {timestep: [] for timestep, _weight in schedule}
        for name in TRANSFORM_NAMES
    }
    audit_roundtrip: dict[str, list[float]] = {name: [] for name in TRANSFORM_NAMES}
    audit_norm: dict[str, list[float]] = {name: [] for name in TRANSFORM_NAMES}

    for sample_index, batch in enumerate(loader):
        target_raw = batch["target_latent"].to(device=device, dtype=dtype)
        condition_raw = batch["omini_condition_latent"].to(device=device, dtype=dtype)
        target_native, target_ids = _normalization_pack(pipe, target_raw)
        condition_clean, condition_ids = _normalization_pack(pipe, condition_raw)
        if "prompt_embeds" in batch and "text_ids" in batch:
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
            text_ids = batch["text_ids"].to(device=device)
        else:
            pipe.text_encoder.to(device)
            prompt_embeds, text_ids = pipe.encode_prompt(
                list(batch["description"]), device=device, max_sequence_length=128
            )
            prompt_embeds = prompt_embeds.to(dtype=dtype)
            pipe.text_encoder.to("cpu")

        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(104729 + sample_index)
        noise_native = torch.randn(
            target_native.shape, generator=noise_generator, device=device, dtype=dtype
        )

        for name in TRANSFORM_NAMES:
            target = apply_representation_transform(target_native, name)
            noise = apply_representation_transform(noise_native, name)
            roundtrip, norm_change = audit_representation(target_native, name)
            audit_roundtrip[name].append(roundtrip)
            audit_norm[name].append(norm_change)
            sample_weighted_flow = 0.0
            for timestep_value, timestep_weight in schedule:
                timestep = torch.full(
                    (target.shape[0],), timestep_value, device=device, dtype=dtype
                )
                noisy = (1.0 - timestep[:, None, None]) * target + timestep[:, None, None] * noise
                objective = noise - target
                prediction = flux2_tunneling_forward(
                    pipe.transformer,
                    noisy,
                    [condition_clean],
                    None,
                    prompt_embeds,
                    target_ids,
                    [condition_ids],
                    None,
                    text_ids,
                    timestep,
                    specify_lora,
                    False,
                )
                flow_mse = torch.mean((prediction.float() - objective.float()) ** 2)
                clean_prediction = noisy - timestep[:, None, None] * prediction
                clean_mse = torch.mean((clean_prediction.float() - target.float()) ** 2)
                flow_value = float(flow_mse.item())
                flow_losses[name].append(flow_value)
                timestep_flow_losses[name][timestep_value].append(flow_value)
                sample_weighted_flow += timestep_weight * flow_value
                clean_losses[name].append(float(clean_mse.item()))
            sample_flow_losses[name].append(sample_weighted_flow)
        print(f"evaluated {sample_index + 1}/{len(dataset)} samples", flush=True)

    results = []
    for name in TRANSFORM_NAMES:
        values = np.asarray(flow_losses[name], dtype=np.float64)
        sample_values = np.asarray(sample_flow_losses[name], dtype=np.float64)
        results.append(
            RepresentationResult(
                representation=name,
                samples=len(dataset),
                evaluations=len(values),
                roundtrip_max_abs=max(audit_roundtrip[name]),
                relative_norm_change=max(audit_norm[name]),
                frozen_flow_mse_mean=float(sample_values.mean()),
                frozen_flow_mse_median=float(np.median(values)),
                frozen_flow_mse_sample_std=float(sample_values.std(ddof=1))
                if len(sample_values) > 1 else 0.0,
                frozen_clean_mse_mean=float(np.mean(clean_losses[name])),
            )
        )
    results.sort(key=lambda result: result.frozen_flow_mse_mean)

    payload = {
        "task": {
            "channel_head": args.channel_head_index,
            "condition": entry.get("condition"),
            "target": entry.get("target"),
            "split": args.split,
        },
        "timestep_mode": args.timestep_mode,
        "timestep_schedule": [
            {"timestep": timestep, "weight": weight}
            for timestep, weight in schedule
        ],
        "per_timestep": {
            name: [
                {
                    "timestep": timestep,
                    "weight": weight,
                    "mean": float(np.mean(timestep_flow_losses[name][timestep])),
                    "std": float(np.std(timestep_flow_losses[name][timestep], ddof=1))
                    if len(timestep_flow_losses[name][timestep]) > 1 else 0.0,
                }
                for timestep, weight in schedule
            ]
            for name in TRANSFORM_NAMES
        },
        "pairwise_sample_order": [
            {
                "lower_residual": left,
                "higher_residual": right,
                "consistent_samples": int(sum(
                    left_value < right_value
                    for left_value, right_value in zip(
                        sample_flow_losses[left], sample_flow_losses[right]
                    )
                )),
                "samples": len(sample_flow_losses[left]),
            }
            for left_index, left in enumerate(TRANSFORM_NAMES)
            for right in TRANSFORM_NAMES[left_index + 1:]
            if np.mean(sample_flow_losses[left]) < np.mean(sample_flow_losses[right])
        ] + [
            {
                "lower_residual": right,
                "higher_residual": left,
                "consistent_samples": int(sum(
                    right_value < left_value
                    for left_value, right_value in zip(
                        sample_flow_losses[left], sample_flow_losses[right]
                    )
                )),
                "samples": len(sample_flow_losses[left]),
            }
            for left_index, left in enumerate(TRANSFORM_NAMES)
            for right in TRANSFORM_NAMES[left_index + 1:]
            if np.mean(sample_flow_losses[right]) < np.mean(sample_flow_losses[left])
        ],
        "interpretation": (
            "Lower frozen flow MSE indicates a target representation that is more "
            "compatible with the unchanged backbone before task-specific training."
        ),
        "results": [asdict(result) for result in results],
    }
    _atomic_json(payload, args.output / "representation_alignment.json")
    _write_csv(results, args.output / "representation_alignment.csv")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B",
    )
    parser.add_argument(
        "--modality-config",
        type=Path,
        default=PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json",
    )
    parser.add_argument(
        "--dense-latent-cache",
        type=Path,
        default=PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset",
    )
    parser.add_argument(
        "--text-embedding-cache",
        type=Path,
        default=PROJECT_ROOT / "data/wsg_flux2_caption_embeddings_all_datasets",
    )
    parser.add_argument("--channel-head-index", type=int, default=3)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--timestep-mode", choices=("training", "fixed"), default="training",
        help="Measure the unchanged training distribution or an explicit fixed grid.",
    )
    parser.add_argument("--quadrature-points", type=int, default=5)
    parser.add_argument("--anchor-probability", type=float, default=0.5)
    parser.add_argument("--anchor-timestep", type=float, default=0.999)
    parser.add_argument("--timesteps", default="0.25,0.5,0.75")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "debug/representation_alignment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_frozen_compatibility(args)
    print("\nFrozen-backbone compatibility order (lower is better):")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:2d}. {result.representation:36s} "
            f"flow_mse={result.frozen_flow_mse_mean:.6f} "
            f"roundtrip={result.roundtrip_max_abs:.3e}"
        )


if __name__ == "__main__":
    main()
