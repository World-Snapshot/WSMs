#!/usr/bin/env python3
"""Finalize official outputs into canonical external-evaluation manifests.

This keeps format conversion separate from inference and is safe to rerun. It
finalizes GeoWizard, WSM absolute-depth Head 2, sharded MODUS, and the validated
full-precision SenseNova-Vision compatibility run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
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


def depth_samples(workspace: Path) -> list[dict]:
    rows = [
        row for row in read_jsonl(workspace / "samples.jsonl")
        if row.get("task") == "depth_estimation"
    ]
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 frozen depth samples, found {len(rows)}")
    return rows


def finalize_geowizard(workspace: Path) -> bool:
    samples = depth_samples(workspace)
    source = workspace / "outputs/geowizard/official_outputs/depth_npy"
    rows = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        prediction = source / f"{sample_id}_rgb_pred.npy"
        if not prediction.is_file():
            return False
        rows.append({
            "sample_id": sample_id,
            "prediction_path": str(prediction.resolve()),
            "adapter": "geowizard_official_run_infer_v2",
            "checkpoint": "fuxiao0719/GeoWizard unet_v2",
            "ensemble_size": 10,
            "denoise_steps": 50,
            "domain": "indoor",
            "sd2_component_mirror": "LanguageMachines/stable-diffusion-2-1-base",
        })
    atomic_jsonl(workspace / "predictions/geowizard.jsonl", rows)
    print("Finalized GeoWizard: 24 predictions", flush=True)
    return True


def finalize_wsm_depth(workspace: Path) -> bool:
    samples = depth_samples(workspace)
    root = workspace / "outputs/wsm_depth_head2_final2000_exact24"
    destination = workspace / "outputs/wsm/depth_estimation"
    destination.mkdir(parents=True, exist_ok=True)
    depth_rows = []
    for index, sample in enumerate(samples):
        sample_dir = root / f"pseudo-visualization-inference/benchmark/sample_{index:02d}"
        archive = sample_dir / "wsg_reader/tensors/inference.wsg.npz"
        metadata = sample_dir / "metadata.json"
        if not archive.is_file() or not metadata.is_file():
            return False
        source_id = str(json.loads(metadata.read_text(encoding="utf-8")).get("sample_id") or "")
        if source_id != str(sample.get("source_sample_id") or ""):
            raise RuntimeError(
                f"WSM sample_{index:02d} provenance mismatch: {source_id!r} != "
                f"{sample.get('source_sample_id')!r}"
            )
        with np.load(archive, allow_pickle=False) as payload:
            if str(payload["modality"].item()) != "34-35":
                raise RuntimeError(f"Unexpected WSM modality in {archive}")
            depth = np.asarray(payload["wsg_data"][..., 34], dtype=np.float32)
        output = destination / f"{sample['sample_id']}.npy"
        np.save(output, depth, allow_pickle=False)
        depth_rows.append({
            "sample_id": str(sample["sample_id"]),
            "task": "depth_estimation",
            "method_id": "wsm",
            "prediction_path": str(output.resolve()),
            "adapter": "wsm_denormalized_wsg_channel34",
            "channel_head": 2,
            "target_modality": "34-35",
            "checkpoint_step": 2000,
            "inference_steps": 10,
        })
    manifest = workspace / "predictions/wsm.jsonl"
    retained = [row for row in read_jsonl(manifest) if row.get("task") != "depth_estimation"]
    atomic_jsonl(manifest, retained + depth_rows)
    print(f"Finalized WSM: {len(retained)} existing + 24 depth predictions", flush=True)
    return True


def _normalize_depth_lane(values: np.ndarray, channel: int) -> np.ndarray:
    config_path = Path(__file__).resolve().parents[3] / "configs/wsg_v1_modalities_dense.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalization = config["normalization"]
    minimum, maximum = normalization["channel_ranges"][str(channel)]
    shifted = np.clip(np.asarray(values, dtype=np.float32) - float(minimum), 0.0, None)
    span = max(float(maximum) - float(minimum), 1e-6)
    mode = "linear"
    for modality, candidate in (normalization.get("modalities") or {}).items():
        members = []
        for part in str(modality).split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                members.extend(range(int(start), int(end) + 1))
            else:
                members.append(int(part))
        if channel in members:
            mode = str(candidate)
            break
    if mode == "log":
        result = np.log1p(shifted) / max(float(np.log1p(span)), 1e-6)
    else:
        result = shifted / span
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def finalize_wsm_depth_audit(workspace: Path) -> bool:
    """Export both WSM depth Heads before and after Reader denormalization."""
    samples = depth_samples(workspace)
    relative_sample_dirs = {}
    for relative_root in (workspace / "outputs").glob(
        "wsm_relative_depth_head1_exact24_tensors*"
    ):
        for metadata_path in relative_root.glob(
            "pseudo-visualization-inference/benchmark/sample_*/metadata.json"
        ):
            source_id = str(
                json.loads(metadata_path.read_text(encoding="utf-8")).get("sample_id") or ""
            )
            if source_id:
                relative_sample_dirs[source_id] = metadata_path.parent
    indexed_common_samples = [
        (index, sample)
        for index, sample in enumerate(samples)
        if str(sample.get("source_sample_id") or "") in relative_sample_dirs
    ]
    if len(indexed_common_samples) < 20:
        return False
    excluded = [
        {
            "sample_id": str(sample.get("sample_id") or ""),
            "source_sample_id": str(sample.get("source_sample_id") or ""),
            "reason": "relative_depth_channel_32_unavailable",
        }
        for index, sample in enumerate(samples)
        if str(sample.get("source_sample_id") or "") not in relative_sample_dirs
    ]
    audit_root = workspace / "depth_protocol_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "excluded_samples.json").write_text(
        json.dumps(excluded, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    variants = {
        "wsm_depth_head2_denormalized": (
            workspace / "outputs/wsm_depth_head2_final2000_exact24", 34, 2, False
        ),
        "wsm_depth_head2_direct_normalized": (
            workspace / "outputs/wsm_depth_head2_final2000_exact24", 34, 2, True
        ),
        "wsm_depth_head1_denormalized": (
            workspace / "outputs/wsm_relative_depth_head1_exact24_tensors", 32, 1, False
        ),
        "wsm_depth_head1_direct_normalized": (
            workspace / "outputs/wsm_relative_depth_head1_exact24_tensors", 32, 1, True
        ),
    }
    for method_id, (root, channel, head, normalized) in variants.items():
        rows = []
        destination = workspace / "outputs/depth_protocol_audit" / method_id
        destination.mkdir(parents=True, exist_ok=True)
        for original_index, sample in indexed_common_samples:
            source_sample_id = str(sample.get("source_sample_id") or "")
            sample_dir = (
                relative_sample_dirs.get(source_sample_id)
                if head == 1 else
                root / f"pseudo-visualization-inference/benchmark/sample_{original_index:02d}"
            )
            if sample_dir is None:
                return False
            metadata_path = sample_dir / "metadata.json"
            denormalized_path = sample_dir / "wsg_reader/tensors/inference.wsg.npz"
            normalized_path = sample_dir / "wsg_reader/tensors/inference.normalized.wsg.npz"
            if not metadata_path.is_file() or not denormalized_path.is_file():
                return False
            source_id = str(
                json.loads(metadata_path.read_text(encoding="utf-8")).get("sample_id") or ""
            )
            if source_id != source_sample_id:
                raise RuntimeError(
                    f"{method_id} sample_{original_index:02d} provenance mismatch: "
                    f"{source_id!r} != {sample.get('source_sample_id')!r}"
                )
            archive_path = normalized_path if normalized and normalized_path.is_file() else denormalized_path
            with np.load(archive_path, allow_pickle=False) as payload:
                values = np.asarray(payload["wsg_data"][..., channel], dtype=np.float32)
            if normalized and archive_path == denormalized_path:
                values = _normalize_depth_lane(values, channel)
            output = destination / f"{sample['sample_id']}.npy"
            np.save(output, values, allow_pickle=False)
            rows.append({
                "sample_id": str(sample["sample_id"]),
                "task": "depth_estimation",
                "method_id": method_id,
                "prediction_path": str(output.resolve()),
                "channel_head": head,
                "target_channel": channel,
                "value_space": "direct_normalized_0_1" if normalized else "reader_denormalized",
                "inference_steps": 10,
            })
        atomic_jsonl(workspace / f"predictions/{method_id}.jsonl", rows)
        print(f"Finalized {method_id}: {len(rows)} predictions", flush=True)
    return True


def finalize_modus(workspace: Path) -> bool:
    manifests = [workspace / f"predictions/modus_shard{index}.jsonl" for index in range(3)]
    if all(path.is_file() for path in manifests):
        rows = [row for path in manifests for row in read_jsonl(path)]
    else:
        # The official CLI processes modality pairs sequentially and writes each
        # shard manifest only after every pair has finished. Recover a fully
        # completed pair directly from its canonical outputs so Normal->Edge can
        # be evaluated while the slower RGB->Depth pair is still running.
        rows = []
        samples = read_jsonl(workspace / "samples.jsonl")
        task_specs = {
            "any_to_any": ("normal2canny", "normal", "canny"),
            "depth_estimation": ("rgb2depth", "rgb", "depth"),
        }
        for task, (pair, source, target) in task_specs.items():
            task_samples = [row for row in samples if row.get("task") == task]
            recovered = []
            for global_index, sample in enumerate(task_samples):
                shard_index = global_index % 3
                local_index = global_index // 3
                sample_id = str(sample["sample_id"])
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_")
                sample_dir = (
                    workspace / f"outputs/modus_shards/shard{shard_index}" / pair /
                    "official_outputs" / f"inputs__rgb_{local_index:04d}_{safe_id}"
                )
                candidates = sorted(
                    path for path in sample_dir.glob(f"{pair}_*0.png")
                    if not any(marker in path.name for marker in (
                        "_compare", "_input", "_bbox", "_intermediate"
                    ))
                )
                if len(candidates) != 1:
                    recovered = []
                    break
                recovered.append({
                    "sample_id": sample_id,
                    "prediction_path": str(candidates[0].resolve()),
                    "adapter": "modus_official_cli_recovered_completed_pair",
                    "source_modality": source,
                    "target_modality": target,
                    "seed": 20260829,
                    "steps": 50,
                    "shard_index": shard_index,
                    "num_shards": 3,
                })
            if recovered:
                rows.extend(recovered)

    by_id = {str(row["sample_id"]): row for row in rows}
    samples = read_jsonl(workspace / "samples.jsonl")
    completed_tasks = []
    for task in ("any_to_any", "depth_estimation"):
        expected = {str(row["sample_id"]) for row in samples if row.get("task") == task}
        present = expected & set(by_id)
        if present == expected:
            completed_tasks.append(task)
        elif present:
            raise RuntimeError(
                f"MODUS has partial {task} coverage ({len(present)}/{len(expected)}); "
                "the task is not eligible for the paper table yet"
            )
    if not completed_tasks:
        return False
    finalized_ids = {
        str(row["sample_id"]) for row in samples if row.get("task") in completed_tasks
    }
    finalized = [by_id[key] for key in sorted(finalized_ids)]
    atomic_jsonl(workspace / "predictions/modus.jsonl", finalized)
    print(
        f"Finalized MODUS: {len(finalized)} predictions; "
        f"complete tasks={','.join(completed_tasks)}",
        flush=True,
    )
    return len(completed_tasks) == 2


def finalize_sensenova(workspace: Path) -> bool:
    source = workspace / "predictions/sensenova_vision_fp32.jsonl"
    if not source.is_file():
        return False
    rows = read_jsonl(source)
    expected = {str(row["sample_id"]) for row in depth_samples(workspace)}
    by_id = {str(row.get("sample_id") or ""): row for row in rows}
    if set(by_id) != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"SenseNova FP32 coverage mismatch: rows={len(rows)}, "
            f"unique={len(by_id)}, expected={len(expected)}"
        )
    finalized = []
    for sample_id in sorted(by_id):
        row = dict(by_id[sample_id])
        prediction = Path(str(row.get("prediction_path") or ""))
        if not prediction.is_file():
            raise RuntimeError(f"Missing SenseNova FP32 prediction: {prediction}")
        values = np.asarray(np.load(prediction, allow_pickle=False), dtype=np.float32)
        if not np.isfinite(values).all() or not float(values.max()) > float(values.min()):
            raise RuntimeError(
                f"Invalid constant/non-finite SenseNova FP32 prediction: {prediction}"
            )
        row.update({
            "task": "depth_estimation",
            "method_id": "sensenova_vision",
            "dtype": "fp32",
            "hardware_compatibility": (
                "RTX 8000 four-GPU FP32 with PyTorch SDPA fallback; "
                "validated non-constant dense output"
            ),
        })
        finalized.append(row)
    atomic_jsonl(workspace / "predictions/sensenova_vision.jsonl", finalized)
    print("Finalized SenseNova-Vision FP32: 24 valid predictions", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--method", action="append",
        choices=["geowizard", "wsm", "wsm_depth_audit", "modus", "sensenova"],
        help="Method(s) to finalize; default: all",
    )
    parser.add_argument("--strict", action="store_true", help="Fail when a run is incomplete")
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    functions = {
        "geowizard": finalize_geowizard,
        "wsm": finalize_wsm_depth,
        "wsm_depth_audit": finalize_wsm_depth_audit,
        "modus": finalize_modus,
        "sensenova": finalize_sensenova,
    }
    for method in args.method or list(functions):
        complete = functions[method](workspace)
        if not complete:
            message = f"{method}: inference is not complete yet"
            if args.strict:
                raise SystemExit(message)
            print(message, flush=True)


if __name__ == "__main__":
    main()
