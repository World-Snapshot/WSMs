#!/usr/bin/env python3
"""Re-encode channel 131 latents in place without changing cache alignment.

The existing channel-131 cache is the index of record. Each worker preserves its
source shard's sample order, frame metadata, and crop metadata. Source WSG NPZ
paths must be globally unique so every NPZ is opened exactly once.
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.wsg_dataset import MASK_VALUE, apply_resize_crop_info

CHANNEL = 131
EXPECTED_RANGE = (-1.0, 19.0)
RENAME_EXCHANGE = 2


def _torch_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _read_range(config_path: Path) -> tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value_range = payload["normalization"]["channel_ranges"][str(CHANNEL)]
    result = (float(value_range[0]), float(value_range[1]))
    if result != EXPECTED_RANGE:
        raise ValueError(
            f"Channel {CHANNEL} range must be {EXPECTED_RANGE}, got {result} in {config_path}"
        )
    return result


def _source_path(source: dict) -> str:
    path = str(source.get("path") or "")
    if not path:
        raise ValueError(f"Missing source path: {source}")
    return path


def _validate_old_shards(shard_paths: list[Path]) -> tuple[int, str]:
    seen_paths: dict[str, tuple[Path, int]] = {}
    total = 0
    normalization_mode = ""
    for shard_path in shard_paths:
        payload = _torch_load(shard_path)
        if str(payload.get("modality")) != str(CHANNEL):
            raise ValueError(f"Unexpected modality in {shard_path}: {payload.get('modality')}")
        sample_ids = list(payload.get("sample_ids") or [])
        sources = list(payload.get("sample_sources") or [])
        crops = list(payload.get("crop_infos") or [])
        latents = payload.get("latents")
        if not torch.is_tensor(latents):
            raise TypeError(f"Missing latent tensor in {shard_path}")
        if not (len(sample_ids) == len(sources) == len(crops) == int(latents.shape[0])):
            raise ValueError(f"Metadata length mismatch in {shard_path}")
        mode = str((payload.get("normalization_modes") or {}).get(str(CHANNEL), payload.get("normalization_mode", "")))
        if mode not in {"linear", "log"}:
            raise ValueError(f"Unsupported normalization mode {mode!r} in {shard_path}")
        if normalization_mode and mode != normalization_mode:
            raise ValueError(f"Mixed normalization modes: {normalization_mode} and {mode}")
        normalization_mode = mode
        for position, (sample_id, source) in enumerate(zip(sample_ids, sources)):
            path = _source_path(source)
            if str(source.get("storage")) != "wsg":
                raise ValueError(
                    "Single-open aligned rebuild currently requires one WSG file per cache entry; "
                    f"found storage={source.get('storage')!r} at {shard_path}:{position}"
                )
            previous = seen_paths.get(path)
            if previous is not None:
                raise ValueError(
                    f"Source NPZ would be opened more than once: {path}; "
                    f"entries={previous} and {(shard_path, position)}"
                )
            if str(source.get("sample_id") or sample_id) != str(sample_id):
                raise ValueError(f"sample_id mismatch at {shard_path}:{position}")
            seen_paths[path] = (shard_path, position)
        total += len(sample_ids)
    if not normalization_mode:
        raise ValueError("No channel 131 samples found")
    return total, normalization_mode


class Channel131Dataset(Dataset):
    def __init__(
        self,
        sample_sources: list[dict],
        crop_infos: list[dict],
        value_range: tuple[float, float],
        normalization_mode: str,
    ) -> None:
        self.sample_sources = sample_sources
        self.crop_infos = crop_infos
        self.min_value, self.max_value = value_range
        self.normalization_mode = normalization_mode

    def __len__(self) -> int:
        return len(self.sample_sources)

    def __getitem__(self, position: int) -> tuple[int, torch.Tensor]:
        source = self.sample_sources[position]
        path = _source_path(source)
        # This is the only source payload open in the encoding path.
        with np.load(path) as payload:
            if "data" in payload:
                raw = payload["data"]
            elif "wsg_data" in payload:
                raw = payload["wsg_data"]
            else:
                raise ValueError(f"WSG file {path} has neither 'data' nor 'wsg_data'")
        cropped = apply_resize_crop_info(raw, self.crop_infos[position])
        if cropped.ndim != 3 or cropped.shape[-1] <= CHANNEL:
            raise ValueError(f"Channel {CHANNEL} missing from {path}: shape={cropped.shape}")
        plane = np.asarray(cropped[:, :, CHANNEL], dtype=np.float32)
        valid = plane != MASK_VALUE
        if not bool(np.any(valid)):
            raise ValueError(f"Channel {CHANNEL} has no valid pixels in indexed sample {path}")
        normalized = plane.copy()
        shifted = np.clip(normalized[valid] - self.min_value, 0.0, None)
        width = max(self.max_value - self.min_value, 1e-6)
        if self.normalization_mode == "log":
            values = np.log1p(shifted) / max(np.log1p(width), 1e-6)
        else:
            values = shifted / width
        normalized[valid] = np.clip(values, 0.0, 1.0)
        stacked = np.repeat(normalized[None, :, :], 3, axis=0)
        valid_stacked = stacked != MASK_VALUE
        stacked[valid_stacked] = stacked[valid_stacked] * 2.0 - 1.0
        tensor = torch.from_numpy(stacked).unsqueeze(1)
        return position, tensor


def _load_vae(model_path: Path, device: str, dtype: torch.dtype) -> AutoencoderKLFlux2:
    if (model_path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(
            str(model_path), subfolder="vae", torch_dtype=dtype
        )
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(model_path), torch_dtype=dtype)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    return vae.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def _run_worker(args: argparse.Namespace) -> None:
    old_path = Path(args.old_shard)
    output_path = Path(args.output_shard)
    config_path = Path(args.modality_config).resolve()
    value_range = _read_range(config_path)
    payload = _torch_load(old_path)
    old_latents = payload["latents"]
    normalization_mode = str(
        (payload.get("normalization_modes") or {}).get(
            str(CHANNEL), payload.get("normalization_mode", "")
        )
    )
    dataset = Channel131Dataset(
        list(payload["sample_sources"]),
        list(payload["crop_infos"]),
        value_range,
        normalization_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpu_workers,
        pin_memory=True,
        persistent_workers=args.cpu_workers > 0,
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    vae = _load_vae(Path(args.model_path), "cuda:0", dtype)
    rebuilt = torch.empty_like(old_latents)
    written = torch.zeros(len(dataset), dtype=torch.bool)
    for positions, videos in loader:
        images = videos[:, :, 0].to(device="cuda:0", dtype=dtype, non_blocking=True)
        encoded = vae.encode(images).latent_dist.sample().detach().cpu().to(old_latents.dtype)
        encoded = encoded.unsqueeze(2)
        if tuple(encoded.shape[1:]) != tuple(old_latents.shape[1:]):
            raise ValueError(
                f"Latent shape mismatch for {old_path}: new={tuple(encoded.shape)}, "
                f"old={tuple(old_latents.shape)}"
            )
        positions = positions.to(dtype=torch.long)
        if bool(torch.any(written.index_select(0, positions))):
            raise ValueError(f"Duplicate cache positions while rebuilding {old_path}")
        rebuilt.index_copy_(0, positions, encoded)
        written.index_fill_(0, positions, True)
        print(
            f"[{old_path.name}] encoded {int(written.sum())}/{len(dataset)}",
            flush=True,
        )
    if not bool(torch.all(written)):
        missing = torch.nonzero(~written, as_tuple=False).flatten().tolist()
        raise ValueError(f"Missing positions in {old_path}: {missing[:20]}")
    payload["latents"] = rebuilt
    payload["modality_config_path"] = str(config_path)
    payload["channel_131_normalization_range"] = list(value_range)
    payload["channel_131_reencoded_at"] = datetime.now(timezone.utc).isoformat()
    payload["channel_131_reencoded_from"] = str(old_path.resolve())
    _atomic_torch_save(payload, output_path)
    print(f"saved aligned shard: {output_path}", flush=True)


def _validate_staged(old_paths: list[Path], staged_dir: Path) -> None:
    for old_path in old_paths:
        new_path = staged_dir / old_path.name
        if not new_path.is_file():
            raise FileNotFoundError(f"Missing staged shard: {new_path}")
        old = _torch_load(old_path)
        new = _torch_load(new_path)
        for key in (
            "sample_ids",
            "sample_sources",
            "frame_keys",
            "frame_indices",
            "crop_infos",
            "source_shard_indices",
            "shard_indices",
        ):
            if old.get(key) != new.get(key):
                raise ValueError(f"Alignment metadata changed for {old_path.name}: {key}")
        if old["latents"].shape != new["latents"].shape:
            raise ValueError(f"Latent shape changed for {old_path.name}")
        if not bool(torch.isfinite(new["latents"].float()).all()):
            raise ValueError(f"Non-finite latent values in {new_path}")
        if tuple(new.get("channel_131_normalization_range") or ()) != EXPECTED_RANGE:
            raise ValueError(f"Missing corrected range marker in {new_path}")


def _atomic_exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing non-atomic cache replacement")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left} <-> {right}")


def _promote(cache_dir: Path, staged_dir: Path) -> Path:
    active_dir = cache_dir / "shards" / str(CHANNEL)
    backup_root = cache_dir / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / f"{CHANNEL}_before_range_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if backup_dir.exists():
        raise FileExistsError(backup_dir)
    try:
        _atomic_exchange(active_dir, staged_dir)
    except OSError as error:
        unsupported = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
        if error.errno not in unsupported:
            raise
        # NFS commonly rejects RENAME_EXCHANGE. Use two atomic directory renames
        # with rollback; the active path is absent only between the two renames.
        active_dir.replace(backup_dir)
        try:
            staged_dir.replace(active_dir)
        except Exception:
            backup_dir.replace(active_dir)
            raise
        return backup_dir
    staged_dir.replace(backup_dir)
    return backup_dir


def _run_orchestrator(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).resolve()
    config_path = Path(args.modality_config).resolve()
    _read_range(config_path)
    active_dir = cache_dir / "shards" / str(CHANNEL)
    old_paths = sorted(active_dir.glob("shard_*.pt"))
    if not old_paths:
        raise FileNotFoundError(f"No channel {CHANNEL} shards in {active_dir}")
    total, mode = _validate_old_shards(old_paths)
    print(
        f"validated {len(old_paths)} aligned shards, {total} unique WSG NPZ files, mode={mode}",
        flush=True,
    )
    if args.dry_run:
        return
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU")
    run_name = f"channel_131_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.staging_root).resolve() / run_name
    staged_dir = run_root / "shards" / str(CHANNEL)
    log_dir = run_root / "logs"
    staged_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    pending = list(old_paths)
    running: list[tuple[subprocess.Popen, Path, object, str]] = []
    failed = False
    while pending or running:
        while pending and len(running) < len(gpu_ids):
            busy_gpu_ids = {item[3] for item in running}
            gpu_id = next(item for item in gpu_ids if item not in busy_gpu_ids)
            old_path = pending.pop(0)
            output_path = staged_dir / old_path.name
            log_handle = (log_dir / f"{old_path.stem}.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-shard",
                str(old_path),
                "--output-shard",
                str(output_path),
                "--modality-config",
                str(config_path),
                "--model-path",
                str(Path(args.model_path).resolve()),
                "--batch-size",
                str(args.batch_size),
                "--cpu-workers",
                str(args.cpu_workers),
                "--dtype",
                args.dtype,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((process, old_path, log_handle, gpu_id))
            print(f"started {old_path.name} on GPU {gpu_id}, pid={process.pid}", flush=True)
        time.sleep(1)
        survivors = []
        for process, old_path, log_handle, gpu_id in running:
            status = process.poll()
            if status is None:
                survivors.append((process, old_path, log_handle, gpu_id))
                continue
            log_handle.close()
            if status != 0:
                failed = True
                print(f"worker failed rc={status}: {old_path.name}", file=sys.stderr)
            else:
                print(f"worker completed: {old_path.name}", flush=True)
        running = survivors
        if failed:
            for process, _old_path, log_handle, _gpu_id in running:
                process.terminate()
                process.wait()
                log_handle.close()
            raise RuntimeError(f"Channel {CHANNEL} rebuild failed; staging retained at {run_root}")
    _validate_staged(old_paths, staged_dir)
    print(f"validated staged cache: {staged_dir}", flush=True)
    if args.promote:
        backup = _promote(cache_dir, staged_dir)
        print(f"atomically promoted channel {CHANNEL}; previous cache: {backup}", flush=True)
    else:
        print("staging only; pass --promote to atomically replace the active channel 131 cache", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset"),
    )
    parser.add_argument(
        "--modality-config",
        default=str(PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json"),
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B"),
    )
    parser.add_argument("--staging-root", default=str(PROJECT_ROOT / "data/channel_131_rebuild_staging"))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--worker-shard", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-shard", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_shard:
        if not args.output_shard:
            raise ValueError("--output-shard is required with --worker-shard")
        args.old_shard = args.worker_shard
        _run_worker(args)
    else:
        _run_orchestrator(args)


if __name__ == "__main__":
    main()
