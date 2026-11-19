#!/usr/bin/env python3
"""Build the crop-aligned AVIS depth protocol used by WSM's Reader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_crop_info(values: np.ndarray, info: dict, fill_value: int | bool = 0) -> np.ndarray:
    patch = int(info["patch_hw"])
    crop_y = int(info["crop_y"])
    crop_x = int(info["crop_x"])
    height, width = values.shape[:2]
    pad_bottom = max(0, patch - height)
    pad_right = max(0, patch - width)
    pad = [(0, pad_bottom), (0, pad_right)] + [(0, 0)] * (values.ndim - 2)
    if pad_bottom or pad_right:
        values = np.pad(values, pad, mode="constant", constant_values=fill_value)
    return values[crop_y:crop_y + patch, crop_x:crop_x + patch, ...]


def load_crop_infos(cache_root: Path, wanted: set[str]) -> dict[str, dict]:
    import torch

    found: dict[str, dict] = {}
    for shard_path in sorted((cache_root / "shards/0_2").glob("shard_*.pt")):
        payload = torch.load(
            shard_path, map_location="cpu", weights_only=False, mmap=True
        )
        sources = payload.get("sample_sources") or []
        infos = payload.get("crop_infos") or []
        if len(sources) != len(infos):
            raise RuntimeError(f"crop-info length mismatch in {shard_path}")
        for source, info in zip(sources, infos):
            source_id = str(source.get("sample_id") or "") if isinstance(source, dict) else ""
            if source_id in wanted:
                found[source_id] = dict(info)
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"missing crop_info for {len(missing)} samples: {missing}")
    return found


def wsm_sample_dirs(workspace: Path) -> dict[str, Path]:
    roots = sorted(
        (workspace / "outputs").glob("wsm_relative_depth_head1_exact24_tensors*"),
        key=lambda path: ("shard" in path.name, path.name),
    )
    found: dict[str, Path] = {}
    for root in roots:
        for metadata_path in sorted(root.glob(
            "pseudo-visualization-inference/benchmark/sample_*/metadata.json"
        )):
            source_id = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("sample_id") or "")
            tensor_root = metadata_path.parent / "wsg_reader/tensors"
            required = (
                tensor_root / "condition.wsg.npz",
                tensor_root / "gt.normalized.wsg.npz",
                tensor_root / "inference.normalized.wsg.npz",
            )
            if source_id and all(path.is_file() for path in required):
                found.setdefault(source_id, metadata_path.parent)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    cache_root = Path(args.latent_cache).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    image_root = output_root / "inputs"
    target_root = output_root / "targets"
    mask_root = output_root / "masks"
    prediction_root = output_root / "wsm_predictions"
    for path in (image_root, target_root, mask_root, prediction_root):
        path.mkdir(parents=True, exist_ok=True)

    source_rows = [
        row for row in read_jsonl(workspace / "samples.jsonl")
        if row.get("task") == "depth_estimation"
    ]
    sample_dirs = wsm_sample_dirs(workspace)
    rows = [row for row in source_rows if str(row.get("source_sample_id") or "") in sample_dirs]
    crop_infos = load_crop_infos(
        cache_root, {str(row["source_sample_id"]) for row in rows}
    )

    from PIL import Image

    manifest_rows: list[dict] = []
    wsm_prediction_rows: list[dict] = []
    audit_rows: list[dict] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        source_id = str(row["source_sample_id"])
        sample_dir = sample_dirs[source_id]
        tensor_root = sample_dir / "wsg_reader/tensors"
        with np.load(tensor_root / "condition.wsg.npz", allow_pickle=False) as payload:
            rgb = np.asarray(payload["wsg_data"][..., :3], dtype=np.float32)
        with np.load(tensor_root / "gt.normalized.wsg.npz", allow_pickle=False) as payload:
            target = np.asarray(payload["wsg_data"][..., 32], dtype=np.float32)
        with np.load(tensor_root / "inference.normalized.wsg.npz", allow_pickle=False) as payload:
            prediction = np.asarray(payload["wsg_data"][..., 32], dtype=np.float32)
        info = crop_infos[source_id]
        original_mask = np.load(Path(row["mask_path"]), allow_pickle=False).astype(bool)
        mask = apply_crop_info(original_mask, info, fill_value=False).astype(bool)

        expected = (int(info["patch_hw"]), int(info["patch_hw"]))
        if rgb.shape[:2] != expected or target.shape != expected or prediction.shape != expected:
            raise RuntimeError(
                f"{sample_id}: Reader tensor shape mismatch: "
                f"rgb={rgb.shape}, target={target.shape}, pred={prediction.shape}, expected={expected}"
            )
        if mask.shape != expected:
            raise RuntimeError(f"{sample_id}: aligned mask shape {mask.shape} != {expected}")

        input_path = image_root / f"{sample_id}_rgb_crop.png"
        target_path = target_root / f"{sample_id}_native_relative_depth.npy"
        mask_path = mask_root / f"{sample_id}_valid_crop.npy"
        prediction_path = prediction_root / f"{sample_id}_wsm_relative_depth.npy"
        Image.fromarray(np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8), mode="RGB").save(input_path)
        np.save(target_path, target, allow_pickle=False)
        np.save(mask_path, mask, allow_pickle=False)
        np.save(prediction_path, prediction, allow_pickle=False)

        updated = dict(row)
        updated.update({
            "input_path": str(input_path.resolve()),
            "target_path": str(target_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "original_input_path": str(Path(row["input_path"]).resolve()),
            "original_target_path": str(Path(row["target_path"]).resolve()),
            "original_mask_path": str(Path(row["mask_path"]).resolve()),
            "crop_protocol": "wsm_reader_exact_center_short_side_square",
            "crop_info": info,
            "wsm_reader_sample_dir": str(sample_dir.resolve()),
            "target_depth_unit": "native_normalized_relative_depth_0_1",
        })
        manifest_rows.append(updated)
        wsm_prediction_rows.append({
            "sample_id": sample_id,
            "task": "depth_estimation",
            "method_id": "wsm_relative_depth_crop_aligned",
            "prediction_path": str(prediction_path.resolve()),
            "channel_head": 1,
            "target_channel": 32,
            "value_space": "direct_normalized_0_1",
            "inference_steps": 10,
        })
        audit_rows.append({
            "sample_id": sample_id,
            "source_sample_id": source_id,
            "crop_info": info,
            "input_sha256": sha256(input_path),
            "target_sha256": sha256(target_path),
            "mask_sha256": sha256(mask_path),
            "valid_fraction": float(mask.mean()),
        })

    write_jsonl(output_root / "samples.jsonl", manifest_rows)
    write_jsonl(output_root / "predictions/wsm_relative_depth_crop_aligned.jsonl", wsm_prediction_rows)
    audit = {
        "schema_version": 1,
        "protocol": "WSM Reader exact crop alignment",
        "source_workspace": str(workspace),
        "latent_cache": str(cache_root),
        "selected_depth_samples": len(source_rows),
        "common_head1_samples": len(rows),
        "excluded_sample_ids": [
            str(row["sample_id"]) for row in source_rows
            if str(row.get("source_sample_id") or "") not in sample_dirs
        ],
        "samples": audit_rows,
    }
    (output_root / "crop_alignment_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "samples": len(rows),
        "manifest": str((output_root / "samples.jsonl").resolve()),
        "audit": str((output_root / "crop_alignment_audit.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
