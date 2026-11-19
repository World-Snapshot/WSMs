#!/usr/bin/env python3
"""Rebuild channel 195 as discrete AVIS RGB latents with -1 background.

Source WSV files are assigned to exactly one worker, even when their rows span
all cache shards. Workers emit position-addressed latent parts; the orchestrator
merges them into staged shards, validates alignment, and optionally atomically
promotes the complete channel directory.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.wsg_dataset import apply_resize_crop_info
from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    categorical_ids_to_rgb,
    channel195_semantic_map,
)

CHANNEL = 195
EXPECTED_RANGE = (-1.0, 26.0)
ENCODING = "avis_discrete_rgb_bg_minus1_v1"
RENAME_EXCHANGE = 2


def torch_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_range(config_path: Path) -> tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)["normalization"]["channel_ranges"][str(CHANNEL)]
    result = (float(values[0]), float(values[1]))
    if result != EXPECTED_RANGE:
        raise ValueError(f"Channel 195 range must be {EXPECTED_RANGE}, got {result}")
    return result


def load_vae(model_path: Path, dtype: torch.dtype) -> AutoencoderKLFlux2:
    kwargs = {"torch_dtype": dtype}
    if (model_path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), subfolder="vae", **kwargs)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), **kwargs)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    return vae.to(device="cuda:0", dtype=dtype).eval().requires_grad_(False)


def source_path(source: dict, sample_id: str) -> str:
    path = str(source.get("path") or "")
    if not path and "::" in sample_id:
        path = sample_id.split("::", 1)[0]
    if not path:
        raise ValueError(f"Missing source path for {sample_id}")
    return path


def frame_key(source: dict, fallback: str) -> str:
    keys = source.get("frame_keys")
    return str(keys[0]) if isinstance(keys, (list, tuple)) and keys else str(fallback)


def frame_from_open_wsv(npz: np.lib.npyio.NpzFile, path: str, key: str) -> np.ndarray:
    if not key or key not in npz.files:
        raise KeyError(f"{path} missing indexed frame {key!r}")
    frame = np.asarray(npz[key])
    if frame.ndim != 3 or frame.shape[-1] <= CHANNEL:
        raise ValueError(f"Invalid channel-195 frame {path}:{key}: {frame.shape}")
    return frame


def categorical_image(frame: np.ndarray, crop: dict) -> torch.Tensor:
    cropped = apply_resize_crop_info(frame, crop)
    labels = channel195_semantic_map(cropped)
    rgb = categorical_ids_to_rgb(labels, AVIS_CATEGORY_COLORS).astype(np.float32)
    return torch.from_numpy((rgb.transpose(2, 0, 1) / 127.5) - 1.0)


def build_manifests(shards: list[Path], manifest_dir: Path, workers: int) -> tuple[list[Path], int]:
    groups: dict[str, list[dict]] = {}
    total = 0
    for shard_index, shard_path in enumerate(shards):
        payload = torch_load(shard_path)
        if str(payload.get("modality")) != str(CHANNEL):
            raise ValueError(f"Unexpected modality in {shard_path}")
        sample_ids = list(payload.get("sample_ids") or [])
        sources = list(payload.get("sample_sources") or [])
        frame_keys = list(payload.get("frame_keys") or [])
        crops = list(payload.get("crop_infos") or [])
        rows = int(payload["latents"].shape[0])
        if not (rows == len(sample_ids) == len(sources) == len(frame_keys) == len(crops)):
            raise ValueError(f"Alignment metadata mismatch in {shard_path}")
        for row, (sample_id, source, fallback_key, crop) in enumerate(
            zip(sample_ids, sources, frame_keys, crops)
        ):
            path = source_path(source, str(sample_id))
            if str(source.get("storage")) != "wsv":
                raise ValueError(f"Channel 195 expected WSV source, got {source.get('storage')}")
            groups.setdefault(path, []).append({
                "shard": shard_path.name,
                "row": row,
                "sample_id": str(sample_id),
                "frame_key": frame_key(source, str(fallback_key)),
                "crop": crop,
            })
        total += rows
        del payload

    assignments = [[] for _ in range(workers)]
    loads = [0] * workers
    for path, records in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        worker = min(range(workers), key=loads.__getitem__)
        assignments[worker].append({"path": path, "records": records})
        loads[worker] += len(records)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    seen_sources = set()
    for worker, sources in enumerate(assignments):
        for source in sources:
            if source["path"] in seen_sources:
                raise ValueError(f"WSV assigned more than once: {source['path']}")
            seen_sources.add(source["path"])
        path = manifest_dir / f"worker_{worker:02d}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"worker": worker, "rows": loads[worker], "sources": sources}, handle)
        paths.append(path)
    print(f"indexed {total} rows from {len(seen_sources)} WSV files; worker loads={loads}", flush=True)
    return paths, total


@torch.no_grad()
def run_worker(args: argparse.Namespace) -> None:
    manifest_path = Path(args.worker_manifest)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    vae = load_vae(Path(args.model_path), dtype)
    output_dir = Path(args.parts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = int(manifest["worker"])
    checkpoint_dir = output_dir / f"worker_{worker:02d}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for source_index, source in enumerate(manifest["sources"]):
        checkpoint_path = checkpoint_dir / f"source_{source_index:05d}.pt"
        if checkpoint_path.exists():
            checkpoint = torch_load(checkpoint_path)
            if checkpoint.get("source") != source["path"]:
                raise ValueError(f"Checkpoint source mismatch: {checkpoint_path}")
            if int(checkpoint.get("record_count", -1)) != len(source["records"]):
                raise ValueError(f"Checkpoint row count mismatch: {checkpoint_path}")
            if source_index % 20 == 0:
                print(
                    f"worker {worker}: resumed {source_index + 1}/{len(manifest['sources'])} WSV",
                    flush=True,
                )
            continue

        path = source["path"]
        positions: dict[str, list[torch.Tensor]] = {}
        latents: dict[str, list[torch.Tensor]] = {}
        pending_images: list[torch.Tensor] = []
        pending_records: list[dict] = []
        missing_records: list[dict] = []

        def flush() -> None:
            if not pending_images:
                return
            images = torch.stack(pending_images).to("cuda:0", dtype=dtype, non_blocking=True)
            encoded = vae.encode(images).latent_dist.sample().detach().cpu().to(dtype).unsqueeze(2)
            for index, record in enumerate(pending_records):
                shard = record["shard"]
                positions.setdefault(shard, []).append(
                    torch.tensor([record["row"]], dtype=torch.long)
                )
                latents.setdefault(shard, []).append(encoded[index:index + 1])
            pending_images.clear()
            pending_records.clear()

        # Each source WSV is opened exactly once per run. Completed source
        # checkpoints are reused without reopening the source after a restart.
        with np.load(path, allow_pickle=False) as npz:
            for record in source["records"]:
                try:
                    frame = frame_from_open_wsv(npz, path, record["frame_key"])
                except KeyError:
                    missing_records.append(record)
                    continue
                pending_images.append(categorical_image(frame, record["crop"]))
                pending_records.append(record)
                if len(pending_images) >= args.batch_size:
                    flush()
        flush()
        entries = {
            shard: {
                "positions": torch.cat(positions[shard]),
                "latents": torch.cat(latents[shard]),
            }
            for shard in positions
        }
        atomic_torch_save(
            {
                "source": path,
                "source_index": source_index,
                "record_count": len(source["records"]),
                "encoded_count": sum(
                    int(entry["positions"].numel()) for entry in entries.values()
                ),
                "missing_records": missing_records,
                "entries": entries,
            },
            checkpoint_path,
        )
        if source_index % 20 == 0:
            print(f"worker {worker}: {source_index + 1}/{len(manifest['sources'])} WSV", flush=True)

    positions_by_shard: dict[str, list[torch.Tensor]] = {}
    latents_by_shard: dict[str, list[torch.Tensor]] = {}
    missing_by_shard: dict[str, list[dict]] = {}
    encoded_rows = 0
    missing_rows = 0
    for source_index, _source in enumerate(manifest["sources"]):
        checkpoint = torch_load(checkpoint_dir / f"source_{source_index:05d}.pt")
        encoded_rows += int(checkpoint["encoded_count"])
        for shard, entry in checkpoint["entries"].items():
            positions_by_shard.setdefault(shard, []).append(entry["positions"])
            latents_by_shard.setdefault(shard, []).append(entry["latents"])
        for record in checkpoint["missing_records"]:
            missing_by_shard.setdefault(record["shard"], []).append(record)
            missing_rows += 1
    if encoded_rows + missing_rows != int(manifest["rows"]):
        raise ValueError(
            f"Worker row mismatch: {encoded_rows} encoded + {missing_rows} missing "
            f"!= {manifest['rows']}"
        )
    for shard in sorted(set(positions_by_shard) | set(missing_by_shard)):
        atomic_torch_save(
            {
                "positions": torch.cat(positions_by_shard[shard])
                if shard in positions_by_shard else torch.empty(0, dtype=torch.long),
                "latents": torch.cat(latents_by_shard[shard])
                if shard in latents_by_shard else torch.empty(0),
                "missing_records": missing_by_shard.get(shard, []),
            },
            output_dir / f"worker_{worker:02d}__{shard}",
        )
    print(
        f"worker {worker} encoded {encoded_rows} rows; excluded {missing_rows} missing-source rows",
        flush=True,
    )

def merge_staged(shards: list[Path], parts_dir: Path, staged_dir: Path, workers: int) -> None:
    staged_dir.mkdir(parents=True, exist_ok=True)
    total_missing = 0
    for shard_path in shards:
        for worker in range(workers):
            part_path = parts_dir / f"worker_{worker:02d}__{shard_path.name}"
            if part_path.exists():
                total_missing += len(torch_load(part_path).get("missing_records", []))

    for shard_path in shards:
        payload = torch_load(shard_path)
        old_latents = payload["latents"]
        rebuilt = torch.empty_like(old_latents)
        written = torch.zeros(old_latents.shape[0], dtype=torch.bool)
        excluded = torch.zeros(old_latents.shape[0], dtype=torch.bool)
        missing_records: list[dict] = []
        for worker in range(workers):
            part_path = parts_dir / f"worker_{worker:02d}__{shard_path.name}"
            if not part_path.exists():
                continue
            part = torch_load(part_path)
            pos = part["positions"].long()
            if pos.numel() and bool((written | excluded).index_select(0, pos).any()):
                raise ValueError(f"Duplicate rows while merging {shard_path.name}")
            if pos.numel() and tuple(part["latents"].shape[1:]) != tuple(old_latents.shape[1:]):
                raise ValueError(f"Latent shape mismatch in {part_path}")
            if pos.numel():
                rebuilt.index_copy_(0, pos, part["latents"].to(old_latents.dtype))
                written.index_fill_(0, pos, True)
            for record in part.get("missing_records", []):
                row = int(record["row"])
                if bool(written[row]) or bool(excluded[row]):
                    raise ValueError(f"Duplicate missing row {row} in {shard_path.name}")
                excluded[row] = True
                missing_records.append(record)
        if not bool((written | excluded).all()):
            missing = torch.nonzero(~(written | excluded)).flatten().tolist()
            raise ValueError(f"Unaccounted rows in {shard_path.name}: {missing[:20]}")
        expected_missing = {
            (int(record["row"]), str(record["frame_key"]))
            for record in payload.get("channel_195_missing_frame_keys", [])
        }
        observed_missing = {
            (int(record["row"]), str(record["frame_key"])) for record in missing_records
        }
        if observed_missing != expected_missing:
            raise ValueError(
                f"Missing-source rows changed in {shard_path.name}: "
                f"expected={sorted(expected_missing)} observed={sorted(observed_missing)}"
            )

        keep = torch.nonzero(written).flatten()
        keep_list = keep.tolist()
        old_rows = int(old_latents.shape[0])
        for key, value in list(payload.items()):
            if key == "latents":
                continue
            if isinstance(value, list) and len(value) == old_rows:
                payload[key] = [value[index] for index in keep_list]
            elif torch.is_tensor(value) and value.ndim and int(value.shape[0]) == old_rows:
                payload[key] = value.index_select(0, keep)
        payload["latents"] = rebuilt.index_select(0, keep)
        payload["num_samples"] = int(keep.numel())
        payload["global_num_samples"] = int(payload.get("global_num_samples", 0)) - total_missing
        removed_ids = [str(record.get("sample_id", "")) for record in missing_records]
        previous_filtered = payload.get("filtered_sample_ids")
        previous_filtered_ids = previous_filtered if isinstance(previous_filtered, list) else []
        payload["filtered_sample_ids"] = list(previous_filtered_ids) + removed_ids
        payload["modality_config_path"] = str(Path(payload.get("modality_config_path", "")))
        payload["channel_195_background_value"] = -1.0
        payload["channel_195_class_range"] = [1.0, 26.0]
        payload["channel_195_normalization_range"] = list(EXPECTED_RANGE)
        payload["channel_195_encoding"] = ENCODING
        payload["channel_195_reencoded_at"] = datetime.now(timezone.utc).isoformat()
        payload["channel_195_rebuilt_rows"] = int(keep.numel())
        payload["channel_195_preserved_old_rows"] = 0
        payload["channel_195_excluded_missing_frames"] = missing_records
        for key in (
            "channel_195_surrogate_range",
            "channel_195_surrogate_version",
            "channel_195_surrogate_note",
            "channel_195_missing_frame_keys",
        ):
            payload.pop(key, None)
        atomic_torch_save(payload, staged_dir / shard_path.name)
        print(
            f"staged {shard_path.name}: {keep.numel()} encoded, {len(missing_records)} excluded",
            flush=True,
        )

def validate_staged(shards: list[Path], staged_dir: Path) -> None:
    for old_path in shards:
        old = torch_load(old_path)
        new = torch_load(staged_dir / old_path.name)
        excluded_rows = {
            int(record["row"])
            for record in new.get("channel_195_excluded_missing_frames", [])
        }
        keep = [index for index in range(len(old["sample_ids"])) if index not in excluded_rows]
        for key in (
            "sample_ids",
            "sample_sources",
            "frame_keys",
            "frame_indices",
            "crop_infos",
            "shard_indices",
        ):
            expected = [old[key][index] for index in keep]
            if expected != new.get(key):
                raise ValueError(f"Filtered alignment changed in {old_path.name}: {key}")
        expected_shape = (len(keep), *old["latents"].shape[1:])
        if tuple(new["latents"].shape) != expected_shape:
            raise ValueError(f"Filtered shape changed in {old_path.name}")
        if not bool(torch.isfinite(new["latents"].float()).all()):
            raise ValueError(f"Non-finite staged latent in {old_path.name}")
        if new.get("channel_195_encoding") != ENCODING:
            raise ValueError(f"Missing encoding marker in {old_path.name}")
        if int(new.get("channel_195_preserved_old_rows", -1)) != 0:
            raise ValueError(f"Old channel-195 rows remain in {old_path.name}")
        if not new.get("filtered_sample_ids"):
            raise ValueError(f"Filtered cache marker missing in {old_path.name}")

def atomic_exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(left), -100, os.fsencode(right), RENAME_EXCHANGE)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def promote(cache_root: Path, staged_dir: Path) -> Path:
    active = cache_root / "shards" / str(CHANNEL)
    backups = cache_root / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / f"195_before_discrete_rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        atomic_exchange(active, staged_dir)
        staged_dir.replace(backup)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        active.replace(backup)
        try:
            staged_dir.replace(active)
        except Exception:
            backup.replace(active)
            raise
    return backup


def orchestrate(args: argparse.Namespace) -> None:
    cache_root = Path(args.cache_root).resolve()
    read_range(Path(args.modality_config).resolve())
    shards = sorted((cache_root / "shards" / str(CHANNEL)).glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError("No channel 195 cache shards")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if args.resume_run:
        run_root = Path(args.resume_run).resolve()
        manifests = sorted((run_root / "manifests").glob("worker_*.json"))
        if len(manifests) != len(gpu_ids):
            raise ValueError(
                f"Resume manifest/GPU mismatch: {len(manifests)} manifests, {len(gpu_ids)} GPUs"
            )
        total = sum(int(json.load(path.open())["rows"]) for path in manifests)
        print(f"resuming staged run {run_root} with {total} rows", flush=True)
    else:
        run_root = Path(args.staging_root).resolve() / f"channel_195_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        manifests, total = build_manifests(shards, run_root / "manifests", len(gpu_ids))
    if args.dry_run:
        print(f"dry-run validated {len(shards)} shards and {total} rows")
        return
    parts_dir = run_root / "parts"
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu_id, manifest in zip(gpu_ids, manifests):
        log = (logs / f"{manifest.stem}.log").open(
            "a" if args.resume_run else "w", encoding="utf-8"
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-manifest", str(manifest),
                   "--parts-dir", str(parts_dir), "--model-path", str(Path(args.model_path).resolve()),
                   "--batch-size", str(args.batch_size), "--dtype", args.dtype]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log, manifest, gpu_id))
        print(f"started {manifest.name} on GPU {gpu_id}, pid={process.pid}", flush=True)
    failed = []
    for process, log, manifest, gpu_id in processes:
        status = process.wait()
        log.close()
        if status:
            failed.append((manifest.name, gpu_id, status))
    if failed:
        raise RuntimeError(f"Channel 195 workers failed: {failed}; staging retained at {run_root}")
    staged_dir = run_root / "shards" / str(CHANNEL)
    merge_staged(shards, parts_dir, staged_dir, len(gpu_ids))
    validate_staged(shards, staged_dir)
    print(f"validated staged channel 195 cache: {staged_dir}", flush=True)
    if args.promote:
        print(f"promoted channel 195; previous cache: {promote(cache_root, staged_dir)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset"))
    parser.add_argument("--modality-config", default=str(PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B"))
    parser.add_argument("--staging-root", default=str(PROJECT_ROOT / "data/channel_195_rebuild_staging"))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--worker-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parts-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_manifest:
        if not args.parts_dir:
            raise ValueError("--parts-dir is required in worker mode")
        run_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
