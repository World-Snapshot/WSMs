#!/usr/bin/env python
"""Precompute FLUX.2 frame latents for XPose RGB videos and pose frames.

XPose is not stored as WSG/WSV. Each batch has a CSV plus a zip. The zip
contains one MP4 and many pose PNGs for each sample. This script converts each
sample window into two WSG-style dense modalities:

  - 0-2: RGB video frames
  - 122-124: XPose pose condition frames

The output follows the shard payload format used by WSGLatentDataset:
```
output_dir/
  shards/0-2/shard_00000_of_00008.pt
  shards/122-124/shard_00000_of_00008.pt
  parts/...
```
Batch parts are written atomically. Partial merged shards record completed
batches, so jobs can resume after parts are removed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKLFlux2

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError("precompute_xpose_pose_vae_latents.py requires opencv-python/cv2") from exc


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.wsg_dataset import safe_modality_name


RGB_MODALITY = "0-2"
POSE_MODALITY = "122-124"


@dataclass(frozen=True)
class XPoseWindow:
    global_index: int
    batch_name: str
    csv_path: Path
    zip_path: Path
    file_name: str
    prompt: str
    condition_images: tuple[str, ...]
    condition_indices: tuple[int, ...]
    window_start: int
    window_indices: tuple[int, ...]
    padded_short: bool

    @property
    def sample_id(self) -> str:
        first = self.condition_indices[self.window_indices[0]]
        last = self.condition_indices[self.window_indices[-1]]
        return f"xpose::{self.batch_name}::{self.file_name}::frames_{first:06d}_{last:06d}::w{self.window_start:06d}"


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_vae(model_path: str, device: str, torch_dtype: torch.dtype):
    path = _resolve_project_path(model_path)
    if (path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(path), subfolder="vae", torch_dtype=torch_dtype)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(path), torch_dtype=torch_dtype)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    vae.to(device=device, dtype=torch_dtype)
    vae.eval()
    return vae


def _atomic_torch_save(payload: dict, path: Path, storage_dtype: torch.dtype | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    if storage_dtype is not None:
        payload = _convert_float_tensors(payload, storage_dtype)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _convert_float_tensors(value, dtype: torch.dtype):
    if torch.is_tensor(value):
        if value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        return {key: _convert_float_tensors(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_float_tensors(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert_float_tensors(item, dtype) for item in value)
    return value


def _safe_name(modality: str) -> str:
    return safe_modality_name(modality)


def _part_root(output_dir: Path, shard_id: int, num_shards: int) -> Path:
    return output_dir / "parts" / f"shard_{shard_id:05d}_of_{num_shards:05d}"


def _manifest_dir(output_dir: Path, shard_id: int, num_shards: int) -> Path:
    return _part_root(output_dir, shard_id, num_shards) / "_manifests"


def _part_path(output_dir: Path, modality: str, shard_id: int, num_shards: int, batch_id: int) -> Path:
    return _part_root(output_dir, shard_id, num_shards) / _safe_name(modality) / f"batch_{batch_id:08d}.pt"


def _manifest_path(output_dir: Path, shard_id: int, num_shards: int, batch_id: int) -> Path:
    return _manifest_dir(output_dir, shard_id, num_shards) / f"batch_{batch_id:08d}.json"


def _shard_path(output_dir: Path, modality: str, shard_id: int, num_shards: int) -> Path:
    return output_dir / "shards" / _safe_name(modality) / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"


def _completed_batches_from_shard(
    output_dir: Path,
    modalities: list[str],
    shard_id: int,
    num_shards: int,
    batch_size: int,
    total_batches: int,
    shard_num_samples: int,
) -> set[int]:
    completed_by_modality = []
    for modality in modalities:
        path = _shard_path(output_dir, modality, shard_id, num_shards)
        if not path.exists():
            return set()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded = set(int(item) for item in payload.get("completed_batches", []))
        if recorded:
            completed_by_modality.append(recorded)
            del payload
            continue
        local_indices = set()
        for global_index in payload.get("source_shard_indices", payload.get("shard_indices", [])):
            global_index = int(global_index)
            if global_index >= shard_id and (global_index - shard_id) % num_shards == 0:
                local_indices.add((global_index - shard_id) // num_shards)
        complete = set()
        for batch_id in range(total_batches):
            start = batch_id * batch_size
            end = min(start + batch_size, shard_num_samples)
            if end > start and all(index in local_indices for index in range(start, end)):
                complete.add(batch_id)
        completed_by_modality.append(complete)
        del payload
    return set.intersection(*completed_by_modality) if completed_by_modality else set()


def _completed_batches(
    output_dir: Path,
    modalities: list[str],
    shard_id: int,
    num_shards: int,
    batch_size: int,
    total_batches: int,
    shard_num_samples: int,
) -> set[int]:
    completed = _completed_batches_from_shard(
        output_dir,
        modalities,
        shard_id,
        num_shards,
        batch_size,
        total_batches,
        shard_num_samples,
    )
    root = _manifest_dir(output_dir, shard_id, num_shards)
    if not root.exists():
        return completed
    for path in sorted(root.glob("batch_*.json")):
        try:
            batch_id = int(path.stem.split("_", 1)[1])
            with path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            modalities = [str(item) for item in manifest.get("modalities", [])]
            if modalities and all(_part_path(output_dir, modality, shard_id, num_shards, batch_id).exists() for modality in modalities):
                completed.add(batch_id)
        except (OSError, ValueError, json.JSONDecodeError, IndexError):
            continue
    return completed


def _parse_list(value: str, field: str) -> list:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {field}: {value[:120]}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a list, got {type(parsed)}")
    return parsed


def _iter_csv_rows(dataset_dir: Path) -> Iterable[tuple[str, Path, Path, dict]]:
    for csv_path in sorted(dataset_dir.glob("batch_*.csv")):
        batch_name = csv_path.stem
        zip_path = dataset_dir / f"{batch_name}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip for {csv_path}: {zip_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield batch_name, csv_path, zip_path, row


def build_xpose_index(
    dataset_dir: Path,
    num_frames: int,
    stride: int,
    pad_short: bool,
    one_window_per_sample: bool,
    max_samples: int | None,
) -> list[XPoseWindow]:
    windows: list[XPoseWindow] = []
    for batch_name, csv_path, zip_path, row in _iter_csv_rows(dataset_dir):
        file_name = str(row["file_name"])
        images = tuple(str(item) for item in _parse_list(row["condition_images"], "condition_images"))
        condition_indices = tuple(int(item) for item in _parse_list(row["condition_index"], "condition_index"))
        if len(images) != len(condition_indices):
            raise ValueError(f"{file_name} condition_images/index length mismatch: {len(images)} vs {len(condition_indices)}")
        if not images:
            continue
        if one_window_per_sample:
            if len(images) < num_frames and not pad_short:
                continue
            if len(images) == 1:
                indices = [0] * num_frames
            else:
                indices = np.linspace(0, len(images) - 1, num=num_frames).round().astype(np.int64).tolist()
            starts = [(0, indices)]
        elif len(images) < num_frames:
            if not pad_short:
                continue
            starts = [(0, list(range(len(images))))]
        else:
            starts = [(start, list(range(start, min(start + num_frames, len(images))))) for start in range(0, len(images) - num_frames + 1, stride)]
            if not starts:
                starts = [(0, list(range(min(num_frames, len(images)))))]
        for start, indices in starts:
            padded_short = False
            while len(indices) < num_frames:
                indices.append(indices[-1])
                padded_short = True
            windows.append(
                XPoseWindow(
                    global_index=len(windows),
                    batch_name=batch_name,
                    csv_path=csv_path,
                    zip_path=zip_path,
                    file_name=file_name,
                    prompt=str(row.get("text") or ""),
                    condition_images=images,
                    condition_indices=condition_indices,
                    window_start=start,
                    window_indices=tuple(indices),
                    padded_short=padded_short,
                )
            )
            if max_samples is not None and len(windows) >= max_samples:
                return windows
    return windows


def _read_pose_image(zip_file: zipfile.ZipFile, member: str) -> np.ndarray:
    with zip_file.open(member, "r") as f:
        image = Image.open(f).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def _extract_video_to_temp(zip_file: zipfile.ZipFile, member: str, temp_dir: Path) -> Path:
    out_path = temp_dir / Path(member).name
    with zip_file.open(member, "r") as src, out_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return out_path


def _read_video_frames(video_path: Path, frame_indices: list[int]) -> tuple[list[np.ndarray], float | None, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps_raw = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps_raw if fps_raw > 0 else None
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    last_frame = None
    for frame_index in frame_indices:
        target = int(frame_index)
        if total_frames > 0:
            target = max(0, min(target, total_frames - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            if last_frame is None:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            frames.append(last_frame.copy())
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        last_frame = frame_rgb
        frames.append(frame_rgb)
    capture.release()
    return frames, fps, total_frames


def _center_square_resize(frames: list[np.ndarray], patch_hw: int) -> tuple[torch.Tensor, dict]:
    if not frames:
        raise ValueError("No frames to preprocess")
    first = frames[0]
    height, width = int(first.shape[0]), int(first.shape[1])
    crop_size = min(height, width)
    crop_y = max(0, (height - crop_size) // 2)
    crop_x = max(0, (width - crop_size) // 2)
    processed = []
    for frame in frames:
        if int(frame.shape[0]) != height or int(frame.shape[1]) != width:
            frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)
        cropped = frame[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
        resized = Image.fromarray(cropped).resize((patch_hw, patch_hw), Image.Resampling.BICUBIC)
        processed.append(np.asarray(resized, dtype=np.float32))
    array = np.stack(processed, axis=0) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    crop_info = {
        "patch_hw": int(patch_hw),
        "original_height": height,
        "original_width": width,
        "crop_y": int(crop_y),
        "crop_x": int(crop_x),
        "crop_size": int(crop_size),
        "resize_height": int(patch_hw),
        "resize_width": int(patch_hw),
        "crop_mode": "center_short_side_square",
    }
    return tensor, crop_info


def _members_for_window(window: XPoseWindow) -> tuple[str, list[str]]:
    video_member = f"{window.file_name}/{window.file_name}.mp4"
    pose_members = [f"{window.file_name}/pose/{window.condition_images[index]}" for index in window.window_indices]
    return video_member, pose_members


def _load_window_tensors(window: XPoseWindow, temp_dir: Path, patch_hw: int) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    with zipfile.ZipFile(window.zip_path, "r") as zf:
        video_member, pose_members = _members_for_window(window)
        names = set(zf.namelist())
        if video_member not in names:
            raise FileNotFoundError(f"{window.zip_path} does not contain {video_member}")
        missing_pose = [member for member in pose_members if member not in names]
        if missing_pose:
            raise FileNotFoundError(f"{window.zip_path} missing pose frames for {window.file_name}: {missing_pose[:5]}")
        video_path = _extract_video_to_temp(zf, video_member, temp_dir)
        frame_indices = [int(window.condition_indices[index]) for index in window.window_indices]
        rgb_frames, fps, total_frames = _read_video_frames(video_path, frame_indices)
        pose_frames = [_read_pose_image(zf, member) for member in pose_members]

    rgb_tensor, crop_info = _center_square_resize(rgb_frames, patch_hw)
    pose_tensor, pose_crop_info = _center_square_resize(pose_frames, patch_hw)
    source = {
        "dataset": "XPose",
        "batch_name": window.batch_name,
        "csv_path": str(window.csv_path),
        "zip_path": str(window.zip_path),
        "file_name": window.file_name,
        "video_member": video_member,
        "pose_members": pose_members,
        "condition_images": [window.condition_images[index] for index in window.window_indices],
        "condition_indices": frame_indices,
        "window_start": int(window.window_start),
        "source_fps": fps,
        "source_frame_count": int(total_frames),
        "padded_short": bool(window.padded_short),
        "rgb_shape": [int(rgb_frames[0].shape[0]), int(rgb_frames[0].shape[1]), 3],
        "pose_shape": [int(pose_frames[0].shape[0]), int(pose_frames[0].shape[1]), 3],
        "pose_crop_info": pose_crop_info,
    }
    return rgb_tensor, pose_tensor, crop_info, source


def _encode_batch(
    vae,
    videos: torch.Tensor,
    device: str,
    torch_dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    # [B,3,T,H,W] -> [B,T,32,1,H/8,W/8].
    batch, channels, frames, height, width = videos.shape
    images = videos.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    images = images.to(device=device, dtype=torch_dtype)
    latents = vae.encode(images).latent_dist.sample()
    latents = latents.detach().cpu().to(dtype=cache_dtype).unsqueeze(2)
    return latents.reshape(batch, frames, *latents.shape[1:])


def _write_prompts(output_dir: Path, windows: list[XPoseWindow]) -> None:
    path = output_dir / "prompts.pt"
    if path.exists():
        return
    sample_ids = []
    prompts = []
    video_ids = []
    frame_indices = []
    for window in windows:
        for selected_index in window.window_indices:
            frame_index = int(window.condition_indices[selected_index])
            sample_ids.append(f"{window.sample_id}::frame_{frame_index:06d}")
            prompts.append(window.prompt)
            video_ids.append(window.file_name)
            frame_indices.append(frame_index)
    _atomic_torch_save(
        {
            "dataset": "XPose",
            "num_samples": len(sample_ids),
            "global_num_samples": len(sample_ids),
            "sample_ids": sample_ids,
            "prompts": prompts,
            "video_ids": video_ids,
            "frame_indices": frame_indices,
        },
        path,
    )


def _merge_final_shards(
    output_dir: Path,
    modalities: list[str],
    shard_id: int,
    num_shards: int,
    common_meta: dict,
    cache_dtype: torch.dtype,
    completed_batches: set[int],
    total_batches: int,
    batch_size: int,
    cleanup_parts: bool,
) -> None:
    for modality in modalities:
        parts = []
        existing_path = _shard_path(output_dir, modality, shard_id, num_shards)
        if existing_path.exists():
            parts.append(torch.load(existing_path, map_location="cpu", weights_only=False))
        part_dir = _part_root(output_dir, shard_id, num_shards) / _safe_name(modality)
        for path in sorted(part_dir.glob("batch_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            parts.append(payload)
        if not parts:
            raise RuntimeError(f"No batch parts found for modality {modality}, shard {shard_id}/{num_shards}")

        latents = torch.cat([part["latents"] for part in parts], dim=0)
        sample_ids: list = []
        sample_sources: list = []
        crop_infos: list = []
        shard_indices: list = []
        prompts: list = []
        for part in parts:
            sample_ids.extend(list(part.get("sample_ids", [])))
            sample_sources.extend(list(part.get("sample_sources", [])))
            crop_infos.extend(list(part.get("crop_infos", [])))
            shard_indices.extend(list(part.get("shard_indices", [])))
            prompts.extend(list(part.get("prompts", [])))

        if not all(len(items) == len(shard_indices) for items in (sample_ids, sample_sources, crop_infos, prompts)):
            raise RuntimeError(f"Shard metadata length mismatch while merging {modality}, shard {shard_id}/{num_shards}")
        latest_position = {int(index): position for position, index in enumerate(shard_indices)}
        ordered_indices = sorted(latest_position)
        keep = [latest_position[index] for index in ordered_indices]
        latents = latents[keep]
        sample_ids = [sample_ids[position] for position in keep]
        sample_sources = [sample_sources[position] for position in keep]
        crop_infos = [crop_infos[position] for position in keep]
        prompts = [prompts[position] for position in keep]
        shard_indices = ordered_indices

        payload = {
            **common_meta,
            "num_samples": len(sample_ids),
            "global_num_samples": int(common_meta["global_num_samples"]),
            "modality": modality,
            "sample_ids": sample_ids,
            "sample_sources": sample_sources,
            "crop_infos": crop_infos,
            "prompts": prompts,
            "latents": latents,
            "source_shard_indices": shard_indices,
            "shard_indices": shard_indices,
            "filtered_sample_ids": True,
            "filter_invalid_modalities": True,
            "part_cache_format": "xpose_sample_major_batch_parts_v1",
            "completed_batches": sorted(int(item) for item in completed_batches),
            "total_batches": int(total_batches),
            "batch_size": int(batch_size),
            "partial_shard": len(completed_batches) < total_batches,
        }
        _atomic_torch_save(payload, existing_path, storage_dtype=cache_dtype)
        print(f"saved {modality}: {tuple(payload['latents'].shape)} -> {existing_path}", flush=True)
    if cleanup_parts:
        part_root = _part_root(output_dir, shard_id, num_shards)
        shutil.rmtree(part_root)
        print(f"removed merged parts -> {part_root}", flush=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/common/users/sf895/wsg/output/XPose_results")
    parser.add_argument("--config", default="manual_processing/flux2_preprocessing/configs/flux2_precompute_avis_train.yaml")
    parser.add_argument("--output-dir", default="data/xpose_flux2_frame_latents")
    parser.add_argument("--flux-model-path", default="models/black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cache-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--video-window-stride", type=int, default=16)
    parser.add_argument("--patch-hw", type=int, default=480)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--video-sample-fraction", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=20260621)
    parser.add_argument(
        "--one-window-per-sample",
        action="store_true",
        help="Encode one window per XPose sample, with frame indices uniformly sampled across the full clip.",
    )
    parser.add_argument("--pad-short", action="store_true", default=True)
    parser.add_argument("--no-pad-short", dest="pad_short", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Build and report the XPose window index without loading VAE.")
    parser.add_argument("--stop-file", default=None, help="If this file appears, stop after the current batch and merge completed parts.")
    parser.add_argument(
        "--merge-partial-on-stop",
        action="store_true",
        default=True,
        help="When --stop-file requests a stop, merge completed batch parts into final shard files instead of failing on missing batches.",
    )
    parser.add_argument(
        "--no-merge-partial-on-stop",
        dest="merge_partial_on_stop",
        action="store_false",
        help="Disable partial shard merging when --stop-file requests a stop.",
    )
    parser.add_argument(
        "--keep-parts-after-merge",
        action="store_true",
        help="Keep batch parts after a successful shard merge. By default they are removed to avoid duplicate storage.",
    )
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    cache_dtype = dtype_map[args.cache_dtype]
    if not 0.0 < args.video_sample_fraction <= 1.0:
        raise ValueError("--video-sample-fraction must be in (0, 1]")
    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = [RGB_MODALITY, POSE_MODALITY]
    config = _load_yaml(_resolve_project_path(args.config))
    vae_path = str(_resolve_project_path(args.flux_model_path))
    print(
        f"[xpose-cache] dataset_dir={dataset_dir} output_dir={output_dir} "
        f"num_frames={args.num_frames} video_sample_fraction={args.video_sample_fraction} "
        f"sample_seed={args.sample_seed} patch_hw={args.patch_hw}",
        flush=True,
    )
    # Build one record per source video first, then select a reproducible random
    # video subset and random frame positions. Stable selection is required so
    # all shards and resumed runs see the same sample IDs.
    windows = build_xpose_index(
        dataset_dir=dataset_dir,
        num_frames=1,
        stride=args.video_window_stride,
        pad_short=bool(args.pad_short),
        one_window_per_sample=True,
        max_samples=None,
    )
    rng = random.Random(args.sample_seed)
    selected_count = max(1, round(len(windows) * args.video_sample_fraction))
    selected_positions = sorted(rng.sample(range(len(windows)), selected_count))
    selected_windows = []
    for position in selected_positions:
        window = windows[position]
        frame_count = len(window.condition_images)
        take = min(args.num_frames, frame_count)
        frame_positions = sorted(rng.sample(range(frame_count), take))
        while len(frame_positions) < args.num_frames:
            frame_positions.append(frame_positions[-1])
        selected_windows.append(
            XPoseWindow(
                global_index=window.global_index,
                batch_name=window.batch_name,
                csv_path=window.csv_path,
                zip_path=window.zip_path,
                file_name=window.file_name,
                prompt=window.prompt,
                condition_images=window.condition_images,
                condition_indices=window.condition_indices,
                window_start=0,
                window_indices=tuple(frame_positions),
                padded_short=len(set(frame_positions)) < args.num_frames,
            )
        )
    windows = selected_windows
    if args.max_samples is not None:
        windows = windows[: args.max_samples]
    if not windows:
        raise RuntimeError(f"No XPose windows found under {dataset_dir}")
    if args.index_only:
        shard_count = len(windows[args.shard_id :: args.num_shards])
        print(
            f"[xpose-cache] index_only total_windows={len(windows)} "
            f"shard_windows={shard_count} shard={args.shard_id}/{args.num_shards}",
            flush=True,
        )
        return
    _write_prompts(output_dir, windows)

    shard_windows = windows[args.shard_id :: args.num_shards]
    if not shard_windows:
        raise RuntimeError(f"Shard {args.shard_id}/{args.num_shards} has no samples")
    total_batches = (len(shard_windows) + args.batch_size - 1) // args.batch_size
    completed = _completed_batches(
        output_dir,
        modalities,
        args.shard_id,
        args.num_shards,
        args.batch_size,
        total_batches,
        len(shard_windows),
    )
    if args.skip_existing and len(completed) == total_batches:
        print(f"skip complete XPose shard {args.shard_id}/{args.num_shards}: {modalities}", flush=True)
        return

    common_meta = {
        "config": str(_resolve_project_path(args.config)),
        "dataset": "XPose",
        "dataset_paths": [str(dataset_dir)],
        "modality_config_path": "",
        "normalization_dataset_paths": [],
        "vae_path": vae_path,
        "num_samples": len(shard_windows) * int(args.num_frames),
        "global_num_samples": len(windows) * int(args.num_frames),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "num_frames": 1,
        "source_frames_per_video": int(args.num_frames),
        "video_sample_fraction": float(args.video_sample_fraction),
        "sample_seed": int(args.sample_seed),
        "latent_num_frames": 1,
        "patch_hw": int(args.patch_hw),
        "height": int(args.patch_hw),
        "width": int(args.patch_hw),
        "crop_mode": "center_short_side_square",
        "normalization_mode": "linear",
        "normalization_modes": {RGB_MODALITY: "linear", POSE_MODALITY: "linear"},
        "vae_family": "flux2_klein_frame",
        "latent_backend": "flux2_frame",
        "data_type": "xpose_zip",
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype,
        "modalities": modalities,
        "batch_size": int(args.batch_size),
        "total_batches": int(total_batches),
    }

    vae = _load_vae(vae_path, args.device, torch_dtype)
    if completed:
        print(f"[xpose-cache] resume shard {args.shard_id}/{args.num_shards}: {len(completed)} completed batches", flush=True)

    start_time = time.time()
    stop_file = Path(args.stop_file) if args.stop_file else None
    stop_requested = False
    progress = tqdm(range(total_batches), desc=f"encode XPose shard {args.shard_id}/{args.num_shards}", dynamic_ncols=True)
    for batch_id in progress:
        if stop_file is not None and stop_file.exists():
            stop_requested = True
            print(
                f"[xpose-cache] shard={args.shard_id}/{args.num_shards} stop requested before batch={batch_id}",
                flush=True,
            )
            break
        if batch_id in completed:
            continue
        start = batch_id * args.batch_size
        end = min(start + args.batch_size, len(shard_windows))
        batch_windows = shard_windows[start:end]
        with tempfile.TemporaryDirectory(prefix=f"xpose_shard{args.shard_id}_batch{batch_id}_") as tmp:
            temp_dir = Path(tmp)
            rgb_videos = []
            pose_videos = []
            crop_infos = []
            sample_sources = []
            sample_ids = []
            prompts = []
            shard_indices = []
            for window in batch_windows:
                rgb, pose, crop_info, source = _load_window_tensors(window, temp_dir, args.patch_hw)
                rgb_videos.append(rgb)
                pose_videos.append(pose)
                crop_infos.append(crop_info)
                sample_sources.append(source)
                sample_ids.append(window.sample_id)
                prompts.append(window.prompt)
                shard_indices.append(int(window.global_index))

            rgb_batch = torch.stack(rgb_videos, dim=0)
            pose_batch = torch.stack(pose_videos, dim=0)
            print(
                f"[xpose-cache] shard={args.shard_id}/{args.num_shards} batch={batch_id}/{total_batches} "
                f"samples={len(batch_windows)} encode {RGB_MODALITY}+{POSE_MODALITY}",
                flush=True,
            )
            combined_latents = _encode_batch(
                vae,
                torch.cat([rgb_batch, pose_batch], dim=0),
                args.device,
                torch_dtype,
                cache_dtype,
            )
            rgb_latents, pose_latents = combined_latents.split([rgb_batch.shape[0], pose_batch.shape[0]], dim=0)

        expanded_ids = []
        expanded_sources = []
        expanded_crop_infos = []
        expanded_prompts = []
        expanded_indices = []
        for local_index, (window, source) in enumerate(zip(batch_windows, sample_sources)):
            selected = [int(window.condition_indices[index]) for index in window.window_indices]
            for frame_offset, source_frame_index in enumerate(selected):
                expanded_ids.append(f"{window.sample_id}::frame_{source_frame_index:06d}")
                frame_source = dict(source)
                frame_source["video_id"] = window.file_name
                frame_source["selected_frame_offset"] = int(frame_offset)
                frame_source["selected_frame_index"] = int(source_frame_index)
                expanded_sources.append(frame_source)
                expanded_crop_infos.append(crop_infos[local_index])
                expanded_prompts.append(prompts[local_index])
                expanded_indices.append(int(window.global_index * args.num_frames + frame_offset))

        for modality, latents in ((RGB_MODALITY, rgb_latents), (POSE_MODALITY, pose_latents)):
            latents = latents.reshape(-1, *latents.shape[2:])
            payload = {
                **common_meta,
                "num_samples": len(expanded_ids),
                "modality": modality,
                "sample_ids": expanded_ids,
                "sample_sources": expanded_sources,
                "video_ids": [source["video_id"] for source in expanded_sources],
                "frame_indices": [source["selected_frame_index"] for source in expanded_sources],
                "crop_infos": expanded_crop_infos,
                "prompts": expanded_prompts,
                "latents": latents,
                "source_shard_indices": expanded_indices,
                "shard_indices": expanded_indices,
                "filtered_sample_ids": True,
                "filter_invalid_modalities": True,
                "part_batch_id": int(batch_id),
                "part_batch_size": len(expanded_ids),
                "part_cache_format": "xpose_sample_major_batch_parts_v1",
            }
            _atomic_torch_save(payload, _part_path(output_dir, modality, args.shard_id, args.num_shards, batch_id), storage_dtype=cache_dtype)

        _atomic_json_save(
            {
                "batch_id": int(batch_id),
                "batch_size": len(expanded_ids),
                "sample_ids": expanded_ids,
                "modalities": modalities,
                "num_shards": int(args.num_shards),
                "shard_id": int(args.shard_id),
            },
            _manifest_path(output_dir, args.shard_id, args.num_shards, batch_id),
        )
        completed.add(batch_id)
        done = len(completed)
        elapsed = max(time.time() - start_time, 1e-6)
        eta_hours = ((total_batches - done) * (elapsed / max(done, 1))) / 3600.0
        print(
            f"[xpose-cache] shard={args.shard_id}/{args.num_shards} progress={done}/{total_batches} "
            f"eta_hours={eta_hours:.2f}",
            flush=True,
        )
        if stop_file is not None and stop_file.exists():
            stop_requested = True
            print(
                f"[xpose-cache] shard={args.shard_id}/{args.num_shards} stop requested after batch={batch_id}",
                flush=True,
            )
            break

    completed_after = completed
    missing = sorted(set(range(total_batches)) - completed_after)
    if missing:
        if stop_requested and args.merge_partial_on_stop and completed_after:
            print(
                f"[xpose-cache] shard={args.shard_id}/{args.num_shards} merging partial shard: "
                f"completed_batches={len(completed_after)}/{total_batches}",
                flush=True,
            )
            _merge_final_shards(
                output_dir,
                modalities,
                args.shard_id,
                args.num_shards,
                common_meta,
                cache_dtype,
                completed_after,
                total_batches,
                args.batch_size,
                not args.keep_parts_after_merge,
            )
            return
        raise RuntimeError(f"Shard {args.shard_id}/{args.num_shards} has incomplete batches; first missing={missing[:10]}")
    _merge_final_shards(
        output_dir,
        modalities,
        args.shard_id,
        args.num_shards,
        common_meta,
        cache_dtype,
        completed_after,
        total_batches,
        args.batch_size,
        not args.keep_parts_after_merge,
    )


if __name__ == "__main__":
    main()
