import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset.wsg_dataset import WSGModalityDataset, safe_modality_name
from src.pipelines.wsg_sparse_codecs.registry import build_sparse_codec, codec_metadata, sparse_modalities_from_config


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _safe_modality_name(modality: str) -> str:
    return safe_modality_name(modality)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_vae_path(model_params: dict) -> str:
    model_paths = model_params.get("model_paths")
    if model_paths is None:
        raise ValueError("model.params.model_paths is required to locate Wan VAE")
    paths = json.loads(model_paths) if isinstance(model_paths, str) else model_paths
    for path in paths:
        if "vae" in str(path).lower():
            return str(_resolve_project_path(path))
    raise ValueError(f"Could not find a VAE path in model_paths: {paths}")


def _load_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_dataset(config: dict):
    dataset_cfg = config["dataset"]
    if dataset_cfg["name"] != "WSGModalityDataset":
        raise ValueError("precompute_wsg_vae_latents.py currently expects dataset.name = WSGModalityDataset")
    params = dict(dataset_cfg["params"])
    for key in ("modality_config_path", "normalization_cache_path"):
        if key in params and params[key] is not None:
            params[key] = str(_resolve_project_path(params[key]))
    return WSGModalityDataset(**params)


def _load_vae(vae_path: str, device: str, torch_dtype: torch.dtype):
    models_dir = Path(project_root) / "src" / "models"
    vae_module = _load_module_from_file("wsm_direct_wan_video_vae", models_dir / "wan_video_vae.py")
    util_module = _load_module_from_file("wsm_direct_model_util", models_dir / "util.py")

    vae = vae_module.WanVideoVAE()
    state_dict = util_module.load_state_dict(vae_path, torch_dtype=torch_dtype, device="cpu")
    state_dict = vae_module.WanVideoVAE.state_dict_converter().from_civitai(state_dict)
    missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"Unexpected VAE load result: missing={missing_keys}, unexpected={unexpected_keys}")
    vae.to(device=device, dtype=torch_dtype)
    vae.eval()
    return vae


def _preprocess_video(video: torch.Tensor, device: str, torch_dtype: torch.dtype) -> torch.Tensor:
    video = video.to(dtype=torch_dtype, device=device)
    assert video.ndim == 5 and video.shape[1] == 3
    return video


def _batch_crop_infos_to_list(batch_crop_infos) -> list[dict]:
    if isinstance(batch_crop_infos, list):
        return [dict(item) for item in batch_crop_infos]
    if not isinstance(batch_crop_infos, dict):
        raise TypeError(f"Unsupported crop_info batch type: {type(batch_crop_infos)}")
    lengths = []
    for value in batch_crop_infos.values():
        if torch.is_tensor(value):
            lengths.append(int(value.shape[0]) if value.ndim > 0 else 1)
        elif isinstance(value, (list, tuple)):
            lengths.append(len(value))
        else:
            lengths.append(1)
    batch_size = max(lengths) if lengths else 0
    crop_infos = []
    for index in range(batch_size):
        item = {}
        for key, value in batch_crop_infos.items():
            if torch.is_tensor(value):
                selected = value[index] if value.ndim > 0 else value
                item[key] = int(selected.item()) if selected.numel() == 1 else selected.detach().cpu().tolist()
            elif isinstance(value, (list, tuple)):
                item[key] = value[index]
            else:
                item[key] = value
        crop_infos.append(item)
    return crop_infos


def _batch_mapping_to_list(batch_mapping) -> list[dict]:
    if isinstance(batch_mapping, list):
        return [dict(item) for item in batch_mapping]
    if not isinstance(batch_mapping, dict):
        raise TypeError(f"Unsupported metadata batch type: {type(batch_mapping)}")
    preferred_batch_keys = ("sample_id", "path", "dataset_root", "storage", "video_name")
    batch_size = 0
    for key in preferred_batch_keys:
        value = batch_mapping.get(key)
        if torch.is_tensor(value):
            batch_size = int(value.shape[0]) if value.ndim > 0 else 1
            break
        if isinstance(value, (list, tuple)):
            batch_size = len(value)
            break
    lengths = []
    for value in batch_mapping.values():
        if torch.is_tensor(value):
            lengths.append(int(value.shape[0]) if value.ndim > 0 else 1)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(item, (list, tuple)) for item in value):
                nested_lengths = [len(item) for item in value if isinstance(item, (list, tuple))]
                if nested_lengths:
                    lengths.append(max(nested_lengths))
                else:
                    lengths.append(len(value))
            else:
                lengths.append(len(value))
        else:
            lengths.append(1)
    if batch_size <= 0:
        batch_size = max(lengths) if lengths else 0
    items = []
    for index in range(batch_size):
        item = {}
        for key, value in batch_mapping.items():
            if torch.is_tensor(value):
                selected = value[index] if value.ndim > 0 else value
                item[key] = int(selected.item()) if selected.numel() == 1 else selected.detach().cpu().tolist()
            elif isinstance(value, (list, tuple)):
                if len(value) == batch_size and not (
                    value and all(isinstance(part, (list, tuple)) for part in value)
                ):
                    item[key] = value[index]
                elif value and all(isinstance(part, (list, tuple)) and len(part) == batch_size for part in value):
                    item[key] = [part[index] for part in value]
                elif len(value) > index:
                    item[key] = value[index]
                else:
                    item[key] = list(value)
            else:
                item[key] = value
        items.append(item)
    return items


def _batch_valid_flags(batch_valid, modalities: list[str], batch_size: int) -> dict[str, torch.Tensor]:
    out = {}
    if not isinstance(batch_valid, dict):
        return {modality: torch.ones(batch_size, dtype=torch.bool) for modality in modalities}
    for modality in modalities:
        value = batch_valid.get(modality)
        if value is None:
            out[modality] = torch.ones(batch_size, dtype=torch.bool)
        elif torch.is_tensor(value):
            out[modality] = value.to(dtype=torch.bool).reshape(-1)
        else:
            out[modality] = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    return out


def _collate_wsg_modality_batch(samples: list[dict]) -> dict:
    if not samples:
        return {}

    first = samples[0]
    batch: dict = {}

    modalities = first.get("modalities", {})
    if isinstance(modalities, dict):
        batch["modalities"] = {
            modality: torch.stack([sample["modalities"][modality] for sample in samples], dim=0)
            for modality in modalities.keys()
        }

    modality_valid = first.get("modality_valid", {})
    if isinstance(modality_valid, dict):
        batch["modality_valid"] = {
            modality: torch.as_tensor(
                [bool(sample.get("modality_valid", {}).get(modality, True)) for sample in samples],
                dtype=torch.bool,
            )
            for modality in modality_valid.keys()
        }

    list_metadata_keys = {"sample_id", "sample_source", "crop_info", "prompt"}
    for key in list_metadata_keys:
        if key in first:
            values = [sample.get(key) for sample in samples]
            batch[key] = [dict(value) if isinstance(value, dict) else value for value in values]

    for key in ("height", "width", "num_frames"):
        if key in first:
            values = [sample.get(key) for sample in samples]
            if all(isinstance(value, (int, bool)) for value in values):
                batch[key] = torch.as_tensor(values, dtype=torch.long)
            else:
                batch[key] = values

    for key in first.keys():
        if key in batch or key in {"modalities", "modality_valid"}:
            continue
        values = [sample.get(key) for sample in samples]
        if all(torch.is_tensor(value) and value.shape == values[0].shape for value in values):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch


def _convert_floating_tensors_inplace(value, dtype: torch.dtype):
    if torch.is_tensor(value):
        if value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _convert_floating_tensors_inplace(item, dtype)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _convert_floating_tensors_inplace(item, dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_convert_floating_tensors_inplace(item, dtype) for item in value)
    return value


def _atomic_torch_save(payload: dict, path: Path, storage_dtype: torch.dtype | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    modality = payload.get("modality")
    mode_map = payload.get("normalization_modes")
    if modality is not None and isinstance(mode_map, dict) and str(modality) in mode_map:
        payload["normalization_mode"] = str(mode_map[str(modality)])
    if storage_dtype is not None:
        _convert_floating_tensors_inplace(payload, storage_dtype)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _sample_major_part_root(output_dir: Path, shard_id: int, num_shards: int) -> Path:
    return output_dir / "parts" / f"shard_{shard_id:05d}_of_{num_shards:05d}"


def _sample_major_manifest_dir(output_dir: Path, shard_id: int, num_shards: int) -> Path:
    return _sample_major_part_root(output_dir, shard_id, num_shards) / "_manifests"


def _sample_major_modality_part_path(output_dir: Path, modality: str, shard_id: int, num_shards: int, batch_id: int) -> Path:
    return (
        _sample_major_part_root(output_dir, shard_id, num_shards)
        / _safe_modality_name(modality)
        / f"batch_{batch_id:08d}.pt"
    )


def _sample_major_batch_manifest_path(output_dir: Path, shard_id: int, num_shards: int, batch_id: int) -> Path:
    return _sample_major_manifest_dir(output_dir, shard_id, num_shards) / f"batch_{batch_id:08d}.json"


def _sample_major_chunk_path(
    output_dir: Path,
    modality: str,
    shard_id: int,
    num_shards: int,
    batch_start: int,
    batch_end: int,
) -> Path:
    return (
        _sample_major_part_root(output_dir, shard_id, num_shards)
        / "_chunks"
        / _safe_modality_name(modality)
        / f"batches_{batch_start:08d}_{batch_end:08d}.pt"
    )


def _sample_major_manifest_is_merged(manifest: dict, output_dir: Path, shard_id: int, num_shards: int) -> bool:
    if not manifest.get("merged_chunk"):
        return False
    batch_start = int(manifest.get("merged_chunk_batch_start", -1))
    batch_end = int(manifest.get("merged_chunk_batch_end", -1))
    if batch_start < 0 or batch_end < batch_start:
        return False
    modalities = [str(item) for item in (manifest.get("modalities") or [])]
    if not modalities:
        return True
    return any(
        _sample_major_chunk_path(output_dir, modality, shard_id, num_shards, batch_start, batch_end).exists()
        for modality in modalities
    )


def _load_completed_sample_major_batches(
    output_dir: Path,
    shard_id: int,
    num_shards: int,
    expected_batch_size: int | None = None,
) -> set[int]:
    manifest_dir = _sample_major_manifest_dir(output_dir, shard_id, num_shards)
    completed = set()
    if not manifest_dir.exists():
        return completed
    for path in manifest_dir.glob("batch_*.json"):
        try:
            batch_id = int(path.stem.split("_", 1)[1])
            with path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if expected_batch_size is not None:
                manifest_batch_size = int(manifest.get("batch_size", -1))
                if manifest_batch_size <= 0 or manifest_batch_size > int(expected_batch_size):
                    continue
            modalities = manifest.get("modalities") or []
            if _sample_major_manifest_is_merged(manifest, output_dir, shard_id, num_shards):
                completed.add(batch_id)
            elif all(_sample_major_modality_part_path(output_dir, str(modality), shard_id, num_shards, batch_id).exists() for modality in modalities):
                completed.add(batch_id)
        except (IndexError, ValueError, OSError, json.JSONDecodeError):
            continue
    return completed


def _load_completed_sample_major_modalities(
    output_dir: Path,
    modalities: list[str],
    shard_id: int,
    num_shards: int,
    batch_id: int,
    expected_batch_size: int | None = None,
) -> set[str]:
    completed = set()
    for modality in modalities:
        path = _sample_major_modality_part_path(output_dir, modality, shard_id, num_shards, batch_id)
        if not path.exists():
            continue
        if expected_batch_size is None:
            completed.add(modality)
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload.get("part_batch_size", -1)) == int(expected_batch_size):
                completed.add(modality)
        except Exception:
            continue
    return completed


def _load_sample_major_parts(output_dir: Path, modality: str, shard_id: int, num_shards: int) -> list[dict]:
    part_dir = _sample_major_part_root(output_dir, shard_id, num_shards) / _safe_modality_name(modality)
    parts = []
    for root, pattern in (
        (part_dir, "batch_*.pt"),
        (_sample_major_part_root(output_dir, shard_id, num_shards) / "_chunks" / _safe_modality_name(modality), "batches_*.pt"),
    ):
        if not root.exists():
            continue
        for path in sorted(root.glob(pattern)):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["_part_path"] = str(path)
            parts.append(payload)
    parts.sort(key=lambda item: (int(item.get("part_batch_id", item.get("chunk_batch_start", 0))), int(item.get("part_modality_index", 0))))
    return parts


def _merge_sample_major_batch_window(
    output_dir: Path,
    modalities: list[str],
    shard_id: int,
    num_shards: int,
    batch_start: int,
    batch_end: int,
    common_meta: dict,
    storage_dtype: torch.dtype,
    delete_batch_parts: bool = True,
) -> None:
    manifest_paths = [
        _sample_major_batch_manifest_path(output_dir, shard_id, num_shards, batch_id)
        for batch_id in range(batch_start, batch_end + 1)
    ]
    if not all(path.exists() for path in manifest_paths):
        return

    manifests = []
    for path in manifest_paths:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if _sample_major_manifest_is_merged(manifest, output_dir, shard_id, num_shards):
            return
        manifests.append(manifest)

    saved_chunks = []
    for modality in modalities:
        part_paths = [
            _sample_major_modality_part_path(output_dir, modality, shard_id, num_shards, batch_id)
            for batch_id in range(batch_start, batch_end + 1)
        ]
        existing_part_paths = [path for path in part_paths if path.exists()]
        if not existing_part_paths:
            continue
        parts = [torch.load(path, map_location="cpu", weights_only=False) for path in existing_part_paths]
        compact_parts = [part for part in parts if part.get("sparse_cache_format") == "compact_lanes_v1"]
        if compact_parts:
            empty_lane_latent = compact_parts[0]["empty_lane_latent"]
            latent_chunks = [part["latents"] for part in compact_parts if int(part["latents"].shape[0]) > 0]
            latents = torch.cat(latent_chunks, dim=0) if latent_chunks else torch.empty((0, *empty_lane_latent.shape), dtype=storage_dtype)
            lane_indices = []
            sample_lane_offsets = [0]
            sample_ids_for_modality = []
            sample_sources_for_modality = []
            video_ids_for_modality = []
            frame_keys_for_modality = []
            frame_indices_for_modality = []
            crop_infos_for_modality = []
            shard_indices_for_modality = []
            for part in compact_parts:
                part_offsets = torch.as_tensor(part["sample_lane_offsets"], dtype=torch.long).tolist()
                lane_indices.extend(int(item) for item in torch.as_tensor(part["lane_indices"], dtype=torch.long).tolist())
                for sample_idx in range(len(part.get("sample_ids", []))):
                    lane_count_for_sample = int(part_offsets[sample_idx + 1]) - int(part_offsets[sample_idx])
                    sample_lane_offsets.append(sample_lane_offsets[-1] + lane_count_for_sample)
                sample_ids_for_modality.extend(list(part.get("sample_ids", [])))
                sample_sources_for_modality.extend(list(part.get("sample_sources", [])))
                video_ids_for_modality.extend(list(part.get("video_ids", [])))
                frame_keys_for_modality.extend(list(part.get("frame_keys", [])))
                frame_indices_for_modality.extend(list(part.get("frame_indices", [])))
                crop_infos_for_modality.extend(list(part.get("crop_infos", [])))
                shard_indices_for_modality.extend(list(part.get("shard_indices", [])))
            payload = {
                **common_meta,
                "num_samples": len(sample_ids_for_modality),
                "modality": modality,
                "sample_ids": sample_ids_for_modality,
                "sample_sources": sample_sources_for_modality,
                "video_ids": video_ids_for_modality,
                "frame_keys": frame_keys_for_modality,
                "frame_indices": frame_indices_for_modality,
                "crop_infos": crop_infos_for_modality,
                "latents": latents,
                "sparse_cache_format": "compact_lanes_v1",
                "lane_indices": torch.as_tensor(lane_indices, dtype=torch.long),
                "sample_lane_offsets": torch.as_tensor(sample_lane_offsets, dtype=torch.long),
                "latent_lane_count": int(compact_parts[0]["latent_lane_count"]),
                "latent_channels_per_lane": int(compact_parts[0]["latent_channels_per_lane"]),
                "empty_lane_latent": empty_lane_latent,
                "dense_num_samples": len(sample_ids_for_modality),
                "source_shard_indices": shard_indices_for_modality,
                "shard_indices": shard_indices_for_modality,
                "filtered_sample_ids": bool(common_meta.get("filter_invalid_modalities")),
                "filter_invalid_modalities": bool(common_meta.get("filter_invalid_modalities")),
                "part_cache_format": "sample_major_chunk_parts_v1",
                "chunk_batch_start": int(batch_start),
                "chunk_batch_end": int(batch_end),
            }
        else:
            latent_chunks = [part["latents"] for part in parts if int(part["latents"].shape[0]) > 0]
            latents = torch.cat(latent_chunks, dim=0) if latent_chunks else torch.empty((0, 0), dtype=storage_dtype)
            sample_ids_for_modality = []
            sample_sources_for_modality = []
            video_ids_for_modality = []
            frame_keys_for_modality = []
            frame_indices_for_modality = []
            crop_infos_for_modality = []
            shard_indices_for_modality = []
            for part in parts:
                sample_ids_for_modality.extend(list(part.get("sample_ids", [])))
                sample_sources_for_modality.extend(list(part.get("sample_sources", [])))
                video_ids_for_modality.extend(list(part.get("video_ids", [])))
                frame_keys_for_modality.extend(list(part.get("frame_keys", [])))
                frame_indices_for_modality.extend(list(part.get("frame_indices", [])))
                crop_infos_for_modality.extend(list(part.get("crop_infos", [])))
                shard_indices_for_modality.extend(list(part.get("shard_indices", [])))
            payload = {
                **common_meta,
                "num_samples": len(sample_ids_for_modality),
                "modality": modality,
                "sample_ids": sample_ids_for_modality,
                "sample_sources": sample_sources_for_modality,
                "video_ids": video_ids_for_modality,
                "frame_keys": frame_keys_for_modality,
                "frame_indices": frame_indices_for_modality,
                "crop_infos": crop_infos_for_modality,
                "latents": latents,
                "source_shard_indices": shard_indices_for_modality,
                "shard_indices": shard_indices_for_modality,
                "filtered_sample_ids": bool(common_meta.get("filter_invalid_modalities")),
                "filter_invalid_modalities": bool(common_meta.get("filter_invalid_modalities")),
                "part_cache_format": "sample_major_chunk_parts_v1",
                "chunk_batch_start": int(batch_start),
                "chunk_batch_end": int(batch_end),
            }
        chunk_path = _sample_major_chunk_path(output_dir, modality, shard_id, num_shards, batch_start, batch_end)
        _atomic_torch_save(payload, chunk_path, storage_dtype=storage_dtype)
        saved_chunks.append(chunk_path)

    if not saved_chunks:
        return

    for manifest, manifest_path in zip(manifests, manifest_paths):
        manifest["merged_chunk"] = True
        manifest["merged_chunk_batch_start"] = int(batch_start)
        manifest["merged_chunk_batch_end"] = int(batch_end)
        _atomic_json_save(manifest, manifest_path)

    if delete_batch_parts:
        for modality in modalities:
            for batch_id in range(batch_start, batch_end + 1):
                path = _sample_major_modality_part_path(output_dir, modality, shard_id, num_shards, batch_id)
                if path.exists():
                    path.unlink()
    print(
        f"[sample-major] shard={shard_id}/{num_shards} merged batches "
        f"{batch_start}-{batch_end} into {len(saved_chunks)} modality chunks",
        flush=True,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wsm_v1_augustus_train.yaml")
    parser.add_argument("--output-dir", default="data/wsg_vae_latents_avis_train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cache-dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--sample-major", action="store_true")
    parser.add_argument("--video-window-stride", type=int, default=None)
    parser.add_argument(
        "--crop-mode",
        choices=["center_short_side_square", "random"],
        default=None,
        help=(
            "Override WSGModalityDataset crop_mode. Use center_short_side_square "
            "for deterministic offline latent caches; random only samples one "
            "fixed crop per cached sample, so it is not ongoing augmentation."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--filter-invalid-modalities",
        dest="filter_invalid_modalities",
        action="store_true",
        default=True,
        help="Only write each modality for samples whose raw WSG channels contain non-mask data.",
    )
    parser.add_argument(
        "--no-filter-invalid-modalities",
        dest="filter_invalid_modalities",
        action="store_false",
        help="Write every requested modality for every sample, including all-mask channels.",
    )
    parser.add_argument(
        "--compact-sparse-lanes",
        dest="compact_sparse_lanes",
        action="store_true",
        default=True,
        help=(
            "Store sparse heatmap VAE caches in compact lane format. Only lanes "
            "with non-empty heatmaps are written, and WSGLatentDataset pads them "
            "back to the codec's fixed lane count when training."
        ),
    )
    parser.add_argument(
        "--no-compact-sparse-lanes",
        dest="compact_sparse_lanes",
        action="store_false",
        help="Disable compact sparse-lane cache writing and store dense padded sparse latents.",
    )
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument(
        "--sparse-codec-mode",
        default="disabled",
        choices=[
            "disabled",
            "dwpose_semantic_heatmap",
            "semantic_heatmap",
            "heatmap",
            "big_token_heatmap",
            "dwpose_body_partition_heatmap",
            "body_partition_heatmap",
            "dwpose_body_only_heatmap",
            "body_only_heatmap",
            "dwpose_person_keypoint_pack3_heatmap",
            "person_keypoint_pack3_heatmap",
        ],
    )
    parser.add_argument("--sparse-heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--sparse-heatmap-radius", type=int, default=8)
    parser.add_argument("--sparse-heatmap-peak-threshold", type=float, default=0.10)
    parser.add_argument("--sparse-heatmap-nms-radius", type=int, default=6)
    parser.add_argument(
        "--sample-major-merge-every",
        type=int,
        default=0,
        help="Merge sample-major batch parts into chunk files every N completed batches and delete the merged batch parts.",
    )
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    cache_dtype = dtype_map[args.cache_dtype or args.dtype]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")

    config_path = _resolve_project_path(args.config)
    config = _load_config(str(config_path))
    if args.video_window_stride is not None:
        config["dataset"]["params"]["video_window_stride"] = int(args.video_window_stride)
    if args.crop_mode is not None:
        config["dataset"]["params"]["crop_mode"] = args.crop_mode
    config["dataset"]["params"]["sparse_codec_mode"] = args.sparse_codec_mode
    config["dataset"]["params"]["sparse_heatmap_sigma"] = float(args.sparse_heatmap_sigma)
    config["dataset"]["params"]["sparse_heatmap_radius"] = int(args.sparse_heatmap_radius)
    config["dataset"]["params"]["sparse_heatmap_peak_threshold"] = float(args.sparse_heatmap_peak_threshold)
    config["dataset"]["params"]["sparse_heatmap_nms_radius"] = int(args.sparse_heatmap_nms_radius)
    dataset = _build_dataset(config)
    if not hasattr(dataset, "samples"):
        raise ValueError("Expected WSGModalityDataset to expose samples for deterministic latent caching")
    modality_keys = list(dataset.modality_keys)
    if args.modalities is not None:
        unknown = sorted(set(args.modalities) - set(modality_keys))
        if unknown:
            raise ValueError(f"Unknown modalities requested: {unknown}; available={modality_keys}")
        modality_keys = list(args.modalities)
    sparse_modalities = sparse_modalities_from_config(config["dataset"]["params"]["modality_config_path"])
    active_sparse_modalities = [
        modality for modality in modality_keys
        if modality in sparse_modalities
        and build_sparse_codec(
            modality,
            mode=args.sparse_codec_mode,
            sigma=args.sparse_heatmap_sigma,
            radius=args.sparse_heatmap_radius,
            peak_threshold=args.sparse_heatmap_peak_threshold,
            nms_radius=args.sparse_heatmap_nms_radius,
        ) is not None
    ]
    sparse_lane_counts = {}
    for modality in active_sparse_modalities:
        codec = build_sparse_codec(
            modality,
            mode=args.sparse_codec_mode,
            sigma=args.sparse_heatmap_sigma,
            radius=args.sparse_heatmap_radius,
            peak_threshold=args.sparse_heatmap_peak_threshold,
            nms_radius=args.sparse_heatmap_nms_radius,
        )
        lane_count = getattr(codec, "latent_lane_count", None) if codec is not None else None
        if lane_count is not None:
            sparse_lane_counts[modality] = max(1, int(lane_count))
    if len(modality_keys) > 1 and not args.sample_major:
        raise ValueError(
            "Encoding multiple WSG modalities now requires --sample-major so every modality "
            "for a sample shares the same crop_info. Run separate single-modality jobs or add --sample-major."
        )
    vae_path = _select_vae_path(config["model"]["params"])

    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = _load_vae(vae_path, args.device, torch_dtype)

    def encode_video_tensor(video: torch.Tensor) -> torch.Tensor:
        """Encode [B,C,T,H,W], allowing sparse codecs to pack many RGB groups.

        Wan VAE itself is RGB-only. Sparse pack3 codecs return C=3*N channels;
        encode those N RGB packs as extra batch items and concatenate the
        resulting latent channels back per sample.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor [B,C,T,H,W], got {tuple(video.shape)}")
        batch_size, channels = int(video.shape[0]), int(video.shape[1])
        if channels == 3:
            encoded = vae.encode(
                _preprocess_video(video, args.device, torch_dtype),
                device=args.device,
                tiled=args.tiled,
                tile_size=tuple(args.tile_size),
                tile_stride=tuple(args.tile_stride),
            ).detach().cpu().to(dtype=cache_dtype)
            return encoded
        if channels % 3 != 0:
            raise ValueError(f"Wan VAE sparse pack input channels must be divisible by 3, got {channels}")
        packs = channels // 3
        packed = video.contiguous().reshape(batch_size, packs, 3, *video.shape[2:]).reshape(batch_size * packs, 3, *video.shape[2:])
        encoded = vae.encode(
            _preprocess_video(packed, args.device, torch_dtype),
            device=args.device,
            tiled=args.tiled,
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        ).detach().cpu().to(dtype=cache_dtype)
        encoded = encoded.reshape(batch_size, packs, *encoded.shape[1:])
        return encoded.reshape(batch_size, packs * encoded.shape[2], *encoded.shape[3:])

    empty_lane_latent_cache = {}

    def encode_empty_lane_like(video: torch.Tensor) -> torch.Tensor:
        """Encode one all-empty sparse RGB lane for padding compact caches.

        WSGModalityDataset maps sparse heatmaps from [0, 1] to [-1, 1], so an
        empty heatmap lane is an all -1 RGB video, not latent zeros.
        """
        cache_key = (tuple(int(dim) for dim in video.shape[2:]), str(video.dtype))
        cached = empty_lane_latent_cache.get(cache_key)
        if cached is not None:
            return cached
        empty_video = torch.full(
            (1, 3, *video.shape[2:]),
            -1.0,
            dtype=video.dtype,
            device=video.device,
        )
        empty_lane_latent = encode_video_tensor(empty_video)[0]
        empty_lane_latent_cache[cache_key] = empty_lane_latent
        return empty_lane_latent

    def _active_sparse_pack_indices(sample_video: torch.Tensor, lane_count: int) -> list[int]:
        packs = int(sample_video.shape[0]) // 3
        packs = min(packs, int(lane_count))
        active = []
        for pack_idx in range(packs):
            chunk = sample_video[pack_idx * 3 : (pack_idx + 1) * 3]
            # Empty sparse heatmaps are all -1 after WSGModalityDataset scaling.
            if bool(torch.any(chunk > -0.999)):
                active.append(pack_idx)
        return active

    def encode_sparse_video_tensor_compact(video: torch.Tensor, lane_count: int) -> dict:
        """Encode only active sparse RGB packs and keep lane metadata.

        Returned latents are [total_active_lanes, latent_C, T, H, W]. The loader
        reconstructs dense [B, lane_count * latent_C, T, H, W] tensors by filling
        missing lanes with empty_lane_latent.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected sparse video tensor [B,C,T,H,W], got {tuple(video.shape)}")
        batch_size, channels = int(video.shape[0]), int(video.shape[1])
        if channels != int(lane_count) * 3:
            raise ValueError(
                f"Compact sparse cache expected {lane_count} RGB lanes ({lane_count * 3} channels), "
                f"got {channels} channels"
            )
        per_sample_indices = []
        active_packs = []
        for sample_idx in range(batch_size):
            indices = _active_sparse_pack_indices(video[sample_idx], int(lane_count))
            per_sample_indices.append(indices)
            for lane_idx in indices:
                active_packs.append(video[sample_idx, lane_idx * 3 : (lane_idx + 1) * 3])

        empty_lane_latent = encode_empty_lane_like(video)
        if active_packs:
            packed = torch.stack(active_packs, dim=0)
            latents = encode_video_tensor(packed)
        else:
            latents = torch.empty(
                (0, *empty_lane_latent.shape),
                dtype=cache_dtype,
            )
        return {
            "latents": latents,
            "per_sample_lane_indices": per_sample_indices,
            "empty_lane_latent": empty_lane_latent.detach().cpu().to(dtype=cache_dtype),
            "latent_lane_count": int(lane_count),
            "latent_channels_per_lane": int(empty_lane_latent.shape[0]),
        }

    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    if total <= 0:
        raise ValueError("No samples to encode")

    shard_indices = list(range(args.shard_id, total, args.num_shards))
    if not shard_indices:
        raise ValueError(f"Shard {args.shard_id}/{args.num_shards} has no samples")
    sample_ids = [dataset.samples[index]["sample_id"] for index in shard_indices]
    prompts = [dataset.samples[index].get("prompt", "") for index in shard_indices]

    common_meta = {
        "config": str(config_path),
        "modality_config_path": str(config["dataset"]["params"].get("modality_config_path", "")),
        "dataset_paths": list(getattr(dataset, "dataset_paths", [])),
        "normalization_dataset_paths": list(getattr(dataset, "normalization_dataset_paths", [])),
        "vae_path": vae_path,
        "num_samples": len(shard_indices),
        "global_num_samples": total,
        "sample_ids": sample_ids,
        "shard_indices": shard_indices,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_frames": int(dataset.num_frames),
        "patch_hw": int(dataset.patch_hw),
        "height": int(dataset.patch_hw),
        "width": int(dataset.patch_hw),
        "crop_mode": getattr(dataset, "crop_mode", "random"),
        "normalization_mode": dataset.normalization_mode,
        "normalization_modes": dict(getattr(dataset, "modality_normalization_modes", {})),
        "data_type": dataset.data_type,
        "tiled": bool(args.tiled),
        "tile_size": tuple(args.tile_size),
        "tile_stride": tuple(args.tile_stride),
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype or args.dtype,
        "compact_sparse_lanes": bool(args.compact_sparse_lanes),
        "filter_invalid_modalities": bool(args.filter_invalid_modalities),
        **codec_metadata(
            args.sparse_codec_mode,
            active_sparse_modalities,
            args.sparse_heatmap_sigma,
            args.sparse_heatmap_radius,
            args.sparse_heatmap_peak_threshold,
            args.sparse_heatmap_nms_radius,
        ),
    }

    prompts_path = output_dir / "prompts.pt"
    if args.num_shards == 1 and (not args.skip_existing or not prompts_path.exists()):
        torch.save(
            {
                "config": str(config_path),
                "num_samples": len(shard_indices),
                "global_num_samples": total,
                "sample_ids": sample_ids,
                "prompts": prompts,
            },
            prompts_path,
        )

    def out_path_for(modality: str) -> Path:
        if args.num_shards == 1:
            return output_dir / f"{_safe_modality_name(modality)}.pt"
        return output_dir / "shards" / _safe_modality_name(modality) / f"shard_{args.shard_id:05d}_of_{args.num_shards:05d}.pt"

    if args.sample_major:
        existing = [out_path_for(modality).exists() for modality in modality_keys]
        if args.skip_existing and all(existing):
            print(f"skip existing shard {args.shard_id}/{args.num_shards}: {modality_keys}")
            return

        completed_batches = _load_completed_sample_major_batches(
            output_dir,
            args.shard_id,
            args.num_shards,
            expected_batch_size=args.batch_size,
        )
        if completed_batches:
            print(
                f"resume shard {args.shard_id}/{args.num_shards}: "
                f"found {len(completed_batches)} completed batch parts"
            )
        pending_batch_ids = []
        pending_indices = []
        for batch_id, start in enumerate(range(0, len(shard_indices), args.batch_size)):
            if batch_id in completed_batches:
                continue
            end = min(start + args.batch_size, len(shard_indices))
            pending_batch_ids.append(batch_id)
            pending_indices.extend(shard_indices[start:end])

        dataset.modality_keys = modality_keys
        subset = Subset(dataset, pending_indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
            collate_fn=_collate_wsg_modality_batch,
        )
        offset = 0
        for pending_index, batch in enumerate(tqdm(loader, desc=f"encode WSG shard {args.shard_id}/{args.num_shards}", dynamic_ncols=True)):
            batch_id = pending_batch_ids[pending_index]
            offset = batch_id * args.batch_size
            batch_crop_infos = batch.get("crop_info")
            if batch_crop_infos is None:
                raise ValueError("WSGModalityDataset did not provide crop_info; cannot write aligned latent cache")
            batch_crop_infos_list = _batch_crop_infos_to_list(batch_crop_infos)
            batch_sample_sources = _batch_mapping_to_list(batch.get("sample_source", {}))
            batch_sample_ids = list(batch.get("sample_id") or [])
            videos = [batch["modalities"][modality] for modality in modality_keys]
            batch_size = videos[0].shape[0]
            if not batch_sample_ids:
                batch_sample_ids = [dataset.samples[shard_indices[offset + item]]["sample_id"] for item in range(batch_size)]
            if not batch_sample_sources:
                batch_sample_sources = [
                    WSGModalityDataset._sample_source(dataset.samples[shard_indices[offset + item]])
                    for item in range(batch_size)
                ]
            valid_flags = _batch_valid_flags(batch.get("modality_valid"), modality_keys, batch_size)
            completed_modalities = _load_completed_sample_major_modalities(
                output_dir,
                modality_keys,
                args.shard_id,
                args.num_shards,
                batch_id,
                expected_batch_size=batch_size,
            )
            batch_part_modalities = set(completed_modalities)
            if completed_modalities:
                print(
                    f"resume shard {args.shard_id}/{args.num_shards} batch {batch_id}: "
                    f"skip {len(completed_modalities)} completed modality parts",
                    flush=True,
                )
            for idx, (modality, video) in enumerate(zip(modality_keys, videos)):
                if modality in completed_modalities:
                    continue
                lane_count = sparse_lane_counts.get(modality)
                valid_mask = valid_flags[modality]
                if not args.filter_invalid_modalities:
                    valid_mask = torch.ones(batch_size, dtype=torch.bool)
                if int(valid_mask.numel()) != batch_size:
                    raise ValueError(f"modality_valid length mismatch for {modality}: {valid_mask.numel()} vs {batch_size}")
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                selected_video = video.index_select(0, valid_indices)
                selected_indices = [int(item) for item in valid_indices.tolist()]
                print(
                    f"[sample-major] shard={args.shard_id}/{args.num_shards} "
                    f"batch={batch_id} modality={modality} start "
                    f"valid={len(selected_indices)}/{batch_size}",
                    flush=True,
                )
                if (
                    args.compact_sparse_lanes
                    and lane_count is not None
                    and lane_count > 1
                    and int(selected_video.shape[1]) == int(lane_count) * 3
                ):
                    compact_result = encode_sparse_video_tensor_compact(selected_video, int(lane_count))
                    sample_lane_offsets = [0]
                    lane_indices_flat = []
                    sample_ids_for_part = []
                    sample_sources_for_part = []
                    crop_infos_for_part = []
                    shard_indices_for_part = []
                    for local_idx, lane_indices in zip(selected_indices, compact_result["per_sample_lane_indices"]):
                        lane_indices_flat.extend(int(item) for item in lane_indices)
                        sample_lane_offsets.append(sample_lane_offsets[-1] + len(lane_indices))
                        sample_ids_for_part.append(batch_sample_ids[local_idx])
                        sample_sources_for_part.append(batch_sample_sources[local_idx])
                        crop_infos_for_part.append(batch_crop_infos_list[local_idx])
                        shard_indices_for_part.append(shard_indices[offset + local_idx])
                    payload = {
                        **common_meta,
                        "num_samples": len(sample_ids_for_part),
                        "modality": modality,
                        "sample_ids": sample_ids_for_part,
                        "sample_sources": sample_sources_for_part,
                        "crop_infos": crop_infos_for_part,
                        "latents": compact_result["latents"],
                        "sparse_cache_format": "compact_lanes_v1",
                        "lane_indices": torch.as_tensor(lane_indices_flat, dtype=torch.long),
                        "sample_lane_offsets": torch.as_tensor(sample_lane_offsets, dtype=torch.long),
                        "latent_lane_count": int(compact_result["latent_lane_count"]),
                        "latent_channels_per_lane": int(compact_result["latent_channels_per_lane"]),
                        "empty_lane_latent": compact_result["empty_lane_latent"],
                        "dense_num_samples": len(sample_ids_for_part),
                        "source_shard_indices": shard_indices_for_part,
                        "shard_indices": shard_indices_for_part,
                        "filtered_sample_ids": bool(args.filter_invalid_modalities),
                        "filter_invalid_modalities": bool(args.filter_invalid_modalities),
                        "part_batch_id": int(batch_id),
                        "part_batch_size": int(batch_size),
                        "part_modality_index": int(idx),
                    }
                    _atomic_torch_save(
                        payload,
                        _sample_major_modality_part_path(output_dir, modality, args.shard_id, args.num_shards, batch_id),
                        storage_dtype=cache_dtype,
                    )
                    batch_part_modalities.add(modality)
                    print(
                        f"[sample-major] shard={args.shard_id}/{args.num_shards} "
                        f"batch={batch_id} modality={modality} saved",
                        flush=True,
                    )
                    continue

                modality_latents = encode_video_tensor(selected_video)
                sample_ids_for_part = []
                sample_sources_for_part = []
                crop_infos_for_part = []
                shard_indices_for_part = []
                for local_idx in selected_indices:
                    sample_ids_for_part.append(batch_sample_ids[local_idx])
                    sample_sources_for_part.append(batch_sample_sources[local_idx])
                    crop_infos_for_part.append(batch_crop_infos_list[local_idx])
                    shard_indices_for_part.append(shard_indices[offset + local_idx])
                payload = {
                    **common_meta,
                    "num_samples": len(sample_ids_for_part),
                    "modality": modality,
                    "sample_ids": sample_ids_for_part,
                    "sample_sources": sample_sources_for_part,
                    "crop_infos": crop_infos_for_part,
                    "latents": modality_latents,
                    "source_shard_indices": shard_indices_for_part,
                    "shard_indices": shard_indices_for_part,
                    "filtered_sample_ids": bool(args.filter_invalid_modalities),
                    "filter_invalid_modalities": bool(args.filter_invalid_modalities),
                    "part_batch_id": int(batch_id),
                    "part_batch_size": int(batch_size),
                    "part_modality_index": int(idx),
                }
                _atomic_torch_save(
                    payload,
                    _sample_major_modality_part_path(output_dir, modality, args.shard_id, args.num_shards, batch_id),
                    storage_dtype=cache_dtype,
                )
                batch_part_modalities.add(modality)
                print(
                    f"[sample-major] shard={args.shard_id}/{args.num_shards} "
                    f"batch={batch_id} modality={modality} saved",
                    flush=True,
                )
            _atomic_json_save(
                {
                    "batch_id": int(batch_id),
                    "offset": int(offset),
                    "batch_size": int(batch_size),
                    "sample_ids": batch_sample_ids,
                    "modalities": sorted(batch_part_modalities),
                    "num_modalities": len(batch_part_modalities),
                    "num_shards": int(args.num_shards),
                    "shard_id": int(args.shard_id),
                },
                _sample_major_batch_manifest_path(output_dir, args.shard_id, args.num_shards, batch_id),
            )
            if args.sample_major_merge_every > 0 and (batch_id + 1) % args.sample_major_merge_every == 0:
                _merge_sample_major_batch_window(
                    output_dir,
                    modality_keys,
                    args.shard_id,
                    args.num_shards,
                    batch_id + 1 - args.sample_major_merge_every,
                    batch_id,
                    common_meta,
                    cache_dtype,
                    delete_batch_parts=True,
                )

        completed_after = _load_completed_sample_major_batches(
            output_dir,
            args.shard_id,
            args.num_shards,
            expected_batch_size=args.batch_size,
        )
        total_batches = (len(shard_indices) + args.batch_size - 1) // args.batch_size
        missing_batches = sorted(set(range(total_batches)) - completed_after)
        if missing_batches:
            raise RuntimeError(
                f"Shard {args.shard_id}/{args.num_shards} still has {len(missing_batches)} incomplete batches; "
                f"first missing={missing_batches[:10]}"
            )

        if args.sample_major_merge_every > 0:
            final_window_start = (len(shard_indices) // args.batch_size) // args.sample_major_merge_every * args.sample_major_merge_every
            final_batch_id = (len(shard_indices) + args.batch_size - 1) // args.batch_size - 1
            if final_window_start <= final_batch_id:
                _merge_sample_major_batch_window(
                    output_dir,
                    modality_keys,
                    args.shard_id,
                    args.num_shards,
                    final_window_start,
                    final_batch_id,
                    common_meta,
                    cache_dtype,
                    delete_batch_parts=True,
                )

        for modality in modality_keys:
            out_path = out_path_for(modality)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and out_path.exists():
                print(f"skip existing {modality}: {out_path}")
                continue
            parts = _load_sample_major_parts(output_dir, modality, args.shard_id, args.num_shards)
            compact_parts = [part for part in parts if part.get("sparse_cache_format") == "compact_lanes_v1"]
            if compact_parts:
                empty_lane_latent = compact_parts[0]["empty_lane_latent"]
                chunks = [part["latents"] for part in compact_parts if int(part["latents"].shape[0]) > 0]
                if chunks:
                    latents = torch.cat(chunks, dim=0)
                else:
                    latents = torch.empty((0, *empty_lane_latent.shape), dtype=cache_dtype)
                lane_indices = []
                sample_lane_offsets = [0]
                sample_ids_for_modality = []
                sample_sources_for_modality = []
                crop_infos_for_modality = []
                shard_indices_for_modality = []
                for part in compact_parts:
                    part_offsets = torch.as_tensor(part["sample_lane_offsets"], dtype=torch.long).tolist()
                    part_lane_indices = torch.as_tensor(part["lane_indices"], dtype=torch.long).tolist()
                    lane_indices.extend(int(item) for item in part_lane_indices)
                    for sample_idx in range(len(part.get("sample_ids", []))):
                        lane_count_for_sample = int(part_offsets[sample_idx + 1]) - int(part_offsets[sample_idx])
                        sample_lane_offsets.append(sample_lane_offsets[-1] + lane_count_for_sample)
                    sample_ids_for_modality.extend(list(part.get("sample_ids", [])))
                    sample_sources_for_modality.extend(list(part.get("sample_sources", [])))
                    crop_infos_for_modality.extend(list(part.get("crop_infos", [])))
                    shard_indices_for_modality.extend(list(part.get("shard_indices", [])))
                payload = {
                    **common_meta,
                    "num_samples": len(sample_ids_for_modality),
                    "modality": modality,
                    "sample_ids": sample_ids_for_modality,
                    "sample_sources": sample_sources_for_modality,
                    "crop_infos": crop_infos_for_modality,
                    "latents": latents,
                    "sparse_cache_format": "compact_lanes_v1",
                    "lane_indices": torch.as_tensor(lane_indices, dtype=torch.long),
                    "sample_lane_offsets": torch.as_tensor(sample_lane_offsets, dtype=torch.long),
                    "latent_lane_count": int(compact_parts[0]["latent_lane_count"]),
                    "latent_channels_per_lane": int(compact_parts[0]["latent_channels_per_lane"]),
                    "empty_lane_latent": empty_lane_latent,
                    "dense_num_samples": len(sample_ids_for_modality),
                    "source_shard_indices": shard_indices_for_modality,
                    "shard_indices": shard_indices_for_modality,
                    "filtered_sample_ids": bool(args.filter_invalid_modalities),
                    "filter_invalid_modalities": bool(args.filter_invalid_modalities),
                    "part_cache_format": "sample_major_batch_parts_v1",
                }
            else:
                chunks = [part["latents"] for part in parts if int(part["latents"].shape[0]) > 0]
                if chunks:
                    latents = torch.cat(chunks, dim=0)
                else:
                    latents = torch.empty((0, 0), dtype=cache_dtype)
                sample_ids_for_modality = []
                sample_sources_for_modality = []
                crop_infos_for_modality = []
                shard_indices_for_modality = []
                for part in parts:
                    sample_ids_for_modality.extend(list(part.get("sample_ids", [])))
                    sample_sources_for_modality.extend(list(part.get("sample_sources", [])))
                    crop_infos_for_modality.extend(list(part.get("crop_infos", [])))
                    shard_indices_for_modality.extend(list(part.get("shard_indices", [])))
                payload = {
                    **common_meta,
                    "num_samples": len(sample_ids_for_modality),
                    "modality": modality,
                    "sample_ids": sample_ids_for_modality,
                    "sample_sources": sample_sources_for_modality,
                    "crop_infos": crop_infos_for_modality,
                    "latents": latents,
                    "source_shard_indices": shard_indices_for_modality,
                    "shard_indices": shard_indices_for_modality,
                    "filtered_sample_ids": bool(args.filter_invalid_modalities),
                    "filter_invalid_modalities": bool(args.filter_invalid_modalities),
                    "part_cache_format": "sample_major_batch_parts_v1",
                }
            _atomic_torch_save(payload, out_path, storage_dtype=cache_dtype)
            if payload.get("sparse_cache_format") == "compact_lanes_v1":
                offsets = payload["sample_lane_offsets"]
                active = int(offsets[-1].item()) if torch.is_tensor(offsets) and offsets.numel() else 0
                dense_channels = int(payload["latent_lane_count"]) * int(payload["latent_channels_per_lane"])
                print(
                    f"saved {modality}: compact active_lanes={active} "
                    f"dense_shape=({payload['num_samples']}, {dense_channels}, *{tuple(payload['empty_lane_latent'].shape[1:])}) "
                    f"stored={tuple(payload['latents'].shape)} -> {out_path}"
                )
            else:
                print(f"saved {modality}: {tuple(payload['latents'].shape)} -> {out_path}")
        return

    for modality in modality_keys:
        out_path = out_path_for(modality)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and out_path.exists():
            print(f"skip existing {modality}: {out_path}")
            continue

        dataset.modality_keys = [modality]
        subset = Subset(dataset, shard_indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
            collate_fn=_collate_wsg_modality_batch,
        )

        latents = None
        compact_state = None
        crop_infos = []
        sample_ids_for_modality = []
        sample_sources_for_modality = []
        source_shard_indices = []
        offset = 0
        for batch in tqdm(loader, desc=f"encode WSG modality {modality}", dynamic_ncols=True):
            batch_crop_infos = batch.get("crop_info")
            if batch_crop_infos is None:
                raise ValueError("WSGModalityDataset did not provide crop_info; cannot write aligned latent cache")
            batch_crop_infos_list = _batch_crop_infos_to_list(batch_crop_infos)
            video = batch["modalities"][modality]
            batch_size = int(video.shape[0])
            valid_flags = _batch_valid_flags(batch.get("modality_valid"), [modality], batch_size)[modality]
            if not args.filter_invalid_modalities:
                valid_flags = torch.ones(batch_size, dtype=torch.bool)
            valid_indices = torch.nonzero(valid_flags, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                offset += batch_size
                continue
            selected_indices = [int(item) for item in valid_indices.tolist()]
            video = video.index_select(0, valid_indices)
            batch_sample_ids = list(batch.get("sample_id") or [])
            if not batch_sample_ids:
                batch_sample_ids = [dataset.samples[shard_indices[offset + item]]["sample_id"] for item in range(batch_size)]
            batch_sample_sources = _batch_mapping_to_list(batch.get("sample_source", {}))
            if not batch_sample_sources:
                batch_sample_sources = [
                    WSGModalityDataset._sample_source(dataset.samples[shard_indices[offset + item]])
                    for item in range(batch_size)
                ]
            for local_idx in selected_indices:
                sample_ids_for_modality.append(batch_sample_ids[local_idx])
                sample_sources_for_modality.append(batch_sample_sources[local_idx])
                crop_infos.append(batch_crop_infos_list[local_idx])
                source_shard_indices.append(shard_indices[offset + local_idx])
            lane_count = sparse_lane_counts.get(modality)
            if (
                args.compact_sparse_lanes
                and lane_count is not None
                and lane_count > 1
                and int(video.shape[1]) == int(lane_count) * 3
            ):
                compact_result = encode_sparse_video_tensor_compact(video, int(lane_count))
                if compact_state is None:
                    compact_state = {
                        "chunks": [],
                        "lane_indices": [],
                        "sample_lane_offsets": [0],
                        "empty_lane_latent": compact_result["empty_lane_latent"],
                        "latent_lane_count": compact_result["latent_lane_count"],
                        "latent_channels_per_lane": compact_result["latent_channels_per_lane"],
                    }
                if compact_result["latents"].shape[0] > 0:
                    compact_state["chunks"].append(compact_result["latents"])
                for lane_indices in compact_result["per_sample_lane_indices"]:
                    compact_state["lane_indices"].extend(int(item) for item in lane_indices)
                    compact_state["sample_lane_offsets"].append(compact_state["sample_lane_offsets"][-1] + len(lane_indices))
            else:
                latent_batch = encode_video_tensor(video)
                if latents is None:
                    latents = []
                latents.append(latent_batch)
            offset += batch_size

        if offset != len(shard_indices):
            raise RuntimeError(f"Encoded {offset}/{len(shard_indices)} source samples for modality {modality}")
        if compact_state is not None:
            empty_lane_latent = compact_state["empty_lane_latent"]
            if compact_state["chunks"]:
                compact_latents = torch.cat(compact_state["chunks"], dim=0)
            else:
                compact_latents = torch.empty((0, *empty_lane_latent.shape), dtype=cache_dtype)
            payload = {
                **common_meta,
                "num_samples": len(sample_ids_for_modality),
                "modality": modality,
                "sample_ids": sample_ids_for_modality,
                "sample_sources": sample_sources_for_modality,
                "crop_infos": crop_infos,
                "latents": compact_latents,
                "sparse_cache_format": "compact_lanes_v1",
                "lane_indices": torch.as_tensor(compact_state["lane_indices"], dtype=torch.long),
                "sample_lane_offsets": torch.as_tensor(compact_state["sample_lane_offsets"], dtype=torch.long),
                "latent_lane_count": int(compact_state["latent_lane_count"]),
                "latent_channels_per_lane": int(compact_state["latent_channels_per_lane"]),
                "empty_lane_latent": empty_lane_latent,
                "dense_num_samples": len(sample_ids_for_modality),
                "source_shard_indices": source_shard_indices,
                "shard_indices": source_shard_indices,
                "filtered_sample_ids": bool(args.filter_invalid_modalities),
            }
        else:
            if latents:
                latents = torch.cat(latents, dim=0)
            else:
                latents = torch.empty((0, 0), dtype=cache_dtype)
            payload = {
                **common_meta,
                "num_samples": len(sample_ids_for_modality),
                "modality": modality,
                "sample_ids": sample_ids_for_modality,
                "sample_sources": sample_sources_for_modality,
                "crop_infos": crop_infos,
                "latents": latents,
                "source_shard_indices": source_shard_indices,
                "shard_indices": source_shard_indices,
                "filtered_sample_ids": bool(args.filter_invalid_modalities),
            }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        _atomic_torch_save(payload, out_path, storage_dtype=cache_dtype)
        if payload.get("sparse_cache_format") == "compact_lanes_v1":
            active = int(payload["sample_lane_offsets"][-1].item())
            dense_channels = int(payload["latent_lane_count"]) * int(payload["latent_channels_per_lane"])
            print(
                f"saved {modality}: compact active_lanes={active} "
                f"dense_shape=({payload['num_samples']}, {dense_channels}, *{tuple(payload['empty_lane_latent'].shape[1:])}) "
                f"stored={tuple(payload['latents'].shape)} -> {out_path}"
            )
        else:
            print(f"saved {modality}: {tuple(payload['latents'].shape)} -> {out_path}")


if __name__ == "__main__":
    main()
