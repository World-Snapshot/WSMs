#!/usr/bin/env python3
"""Atomically replace Universal_Pose placeholders with Head-0 AVIS annotations."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/wsg_flux2_frame_latents_dense_multidataset"),
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=None,
        help="Defaults to annotations/Universal_Pose/head0_step0940_euler10.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "train_log_custom/Channel_Head_000000/"
            "20260828_024500_mp100_instance_crop_gradient_v1/ckpt/0000940/"
            "pytorch_lora_weights.safetensors"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    inference_root = (
        args.inference_root.expanduser().resolve()
        if args.inference_root is not None
        else cache_root / "annotations/Universal_Pose/head0_step0940_euler10"
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    pose_paths = sorted((cache_root / "shards/Universal_Pose").glob("*.pt"))
    latent_paths = sorted(inference_root.glob("shard_*/pseudo-visualization-inference/annotations/**/generated_raw_latent.pt"))
    if not pose_paths:
        raise FileNotFoundError(cache_root / "shards/Universal_Pose")
    if not latent_paths:
        raise FileNotFoundError(f"No generated latents below {inference_root}")

    expected: set[str] = set()
    shard_payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in pose_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ids = [str(item) for item in payload["sample_ids"]]
        overlap = expected.intersection(ids)
        if overlap:
            raise RuntimeError(f"Duplicate target sample IDs across shards: {sorted(overlap)[:3]}")
        expected.update(ids)
        shard_payloads.append((path, payload))

    generated: dict[str, Path] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for path in latent_paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = str(item.get("sample_id") or "")
        if not sample_id:
            raise RuntimeError(f"Missing sample_id: {path}")
        if sample_id in generated:
            raise RuntimeError(f"Duplicate generated sample_id: {sample_id}")
        generated[sample_id] = path
        metadata[sample_id] = item

    missing = sorted(expected.difference(generated))
    unexpected = sorted(set(generated).difference(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Annotation set mismatch: expected={len(expected)} generated={len(generated)} "
            f"missing={len(missing)} unexpected={len(unexpected)}; "
            f"missing_examples={missing[:3]} unexpected_examples={unexpected[:3]}"
        )

    checkpoint_digest = sha256(checkpoint)
    for path, payload in shard_payloads:
        rows = []
        for sample_id in map(str, payload["sample_ids"]):
            item = metadata[sample_id]
            latent = item["latent"].to(device="cpu", dtype=torch.bfloat16).contiguous()
            if latent.ndim == 4:
                latent = latent.unsqueeze(2)
            if latent.ndim != 5 or latent.shape[0] != 1:
                raise RuntimeError(f"Unexpected latent shape {tuple(latent.shape)} for {sample_id}")
            rows.append(latent[0])
        payload["latents"] = torch.stack(rows)
        payload["annotation_state"] = "complete_head0_true_inference"
        payload["annotation_provenance"] = {
            "method": "Head 0 RGB-to-Universal-Pose true inference",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": 940,
            "inference_steps": 10,
            "sampler": "Euler from pure noise",
            "seed": 20260622,
            "inference_root": str(inference_root),
        }
        atomic_torch_save(payload, path)
        print(f"[avis-pose-finalize] {path.name}: {len(rows)} samples", flush=True)

    print(
        f"[avis-pose-finalize] complete: {len(expected)} samples in {len(pose_paths)} standard modality shards",
        flush=True,
    )


if __name__ == "__main__":
    main()

