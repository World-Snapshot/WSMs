import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path

import torch
import yaml
from diffusers import AutoencoderKLFlux2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset.wsg_dataset import WSGModalityDataset, safe_modality_name
from src.pipelines.wsg_sparse_codecs.registry import build_sparse_codec, codec_metadata, sparse_modalities_from_config
from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    COARSE_MATERIAL_CATEGORY_COLORS,
    COARSE_OBJECT_CATEGORY_COLORS,
)


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _safe_modality_name(modality: str) -> str:
    return safe_modality_name(modality)


_CATEGORICAL_TARGET_PALETTES = {
    "111": COARSE_OBJECT_CATEGORY_COLORS,
    "131": COARSE_MATERIAL_CATEGORY_COLORS,
    "136": COARSE_OBJECT_CATEGORY_COLORS,
    "195": AVIS_CATEGORY_COLORS,
}

# Stable semantic integer targets. Edge (40) and per-object local region IDs
# (143) are deliberately absent: their values are not cross-sample identities.
_CATEGORICAL_TARGET_SCALAR_RANGES = {
    "135": (0, 1),
}

_CATEGORICAL_TARGET_DISPLAY_RANGES = {
    "135": (0.0, 1.0),
}

_CATEGORICAL_TARGET_ONTOLOGIES = {
    "111": {
        "semantic_name": "Object/Asset Category",
        "ontology_revision": "wsg_v1_object_asset_coarse10_v1",
        "class_count": 10,
    },
    "131": {
        "semantic_name": "Material Class",
        "ontology_revision": "wsg_v1_material_coarse6_v1",
        "class_count": 6,
    },
    "136": {
        "semantic_name": "Thing/Physical Object Category",
        "ontology_revision": "wsg_v1_thing_physical_object_coarse10_v1",
        "class_count": 10,
    },
}

_CATEGORICAL_TARGET_MODALITIES = tuple(
    sorted(
        set(_CATEGORICAL_TARGET_PALETTES) | set(_CATEGORICAL_TARGET_SCALAR_RANGES),
        key=int,
    )
)


def _merge_categorical_target_parts(parts: list[dict], payload: dict) -> None:
    annotated = [part for part in parts if "categorical_target_multihot" in part]
    if not annotated:
        return
    if len(annotated) != len(parts):
        raise RuntimeError("Categorical target metadata is missing from some cache parts")
    values = torch.as_tensor(annotated[0]["categorical_target_values"], dtype=torch.long)
    for part in annotated[1:]:
        if not torch.equal(
            values, torch.as_tensor(part["categorical_target_values"], dtype=torch.long)
        ):
            raise RuntimeError("Categorical target values changed across cache parts")
    multihot = torch.cat(
        [torch.as_tensor(part["categorical_target_multihot"], dtype=torch.uint8) for part in annotated],
        dim=0,
    )
    if int(multihot.shape[0]) != len(payload.get("sample_ids") or []):
        raise RuntimeError("Categorical target rows are not aligned with cache sample IDs")
    payload["categorical_target_values"] = values
    payload["categorical_target_multihot"] = multihot
    spatial_maps = torch.cat(
        [torch.as_tensor(part["categorical_target_spatial_index_map"]) for part in annotated],
        dim=0,
    )
    pixel_counts = torch.cat(
        [torch.as_tensor(part["categorical_target_pixel_counts"], dtype=torch.int32) for part in annotated],
        dim=0,
    )
    if int(spatial_maps.shape[0]) != len(payload.get("sample_ids") or []):
        raise RuntimeError("Categorical target spatial rows are not aligned with cache sample IDs")
    payload["categorical_target_spatial_index_map"] = spatial_maps
    payload["categorical_target_pixel_counts"] = pixel_counts
    prototype_rgb = torch.as_tensor(
        annotated[0]["categorical_target_prototypes_rgb"], dtype=torch.uint8
    )
    background_rgb = annotated[0].get("categorical_target_background_rgb")
    background_rgb = (
        torch.as_tensor(background_rgb, dtype=torch.uint8)
        if background_rgb is not None else None
    )
    for part in annotated[1:]:
        if not torch.equal(
            prototype_rgb,
            torch.as_tensor(part["categorical_target_prototypes_rgb"], dtype=torch.uint8),
        ):
            raise RuntimeError("Categorical target prototypes changed across cache parts")
        other_background = part.get("categorical_target_background_rgb")
        other_background = (
            torch.as_tensor(other_background, dtype=torch.uint8)
            if other_background is not None else None
        )
        if (background_rgb is None) != (other_background is None) or (
            background_rgb is not None and not torch.equal(background_rgb, other_background)
        ):
            raise RuntimeError("Categorical target background changed across cache parts")
    payload["categorical_target_prototypes_rgb"] = prototype_rgb
    payload["categorical_target_background_rgb"] = background_rgb


def _categorical_target_sidecar_path(
    output_dir: Path, modality: str, latent_path: Path
) -> Path:
    return (
        output_dir / "categorical_targets" / _safe_modality_name(modality)
        / latent_path.name
    )


def _load_source_frame(
    source: dict, sample_id: str
) -> tuple[np.ndarray, str]:
    source_path = str(source.get("path") or "")
    if not source_path:
        source_path = sample_id.split("::", 1)[0]
    frame_keys = list(source.get("frame_keys") or [])
    frame_key = str(frame_keys[0]) if frame_keys else ""
    with np.load(source_path, allow_pickle=False) as data:
        if frame_key and frame_key in data:
            frame = data[frame_key]
        elif "data" in data:
            frame = data["data"]
        elif "wsg_data" in data:
            frame = data["wsg_data"]
        else:
            raise KeyError(f"No frame {frame_key!r}, data, or wsg_data in {source_path}")
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected HxWxC source frame, got {frame.shape}: {source_path}")
    return frame, source_path


def _load_cached_source_frame(payload: dict, row_index: int) -> tuple[np.ndarray, str]:
    source = (payload.get("sample_sources") or [{} for _ in payload["sample_ids"]])[row_index]
    if not source.get("frame_keys"):
        frame_keys = payload.get("frame_keys") or [""] * len(payload["sample_ids"])
        source = {**source, "frame_keys": [frame_keys[row_index]]}
    return _load_source_frame(source, str(payload["sample_ids"][row_index]))


def _apply_cached_crop(frame: np.ndarray, crop: dict | None) -> np.ndarray:
    if not crop:
        return frame
    patch_hw = int(crop["patch_hw"])
    crop_y = int(crop["crop_y"])
    crop_x = int(crop["crop_x"])
    height, width = frame.shape[:2]
    if height < patch_hw or width < patch_hw:
        frame = np.pad(
            frame,
            ((0, max(0, patch_hw - height)), (0, max(0, patch_hw - width)), (0, 0)),
            mode="constant",
            constant_values=-1.0,
        )
    return frame[crop_y : crop_y + patch_hw, crop_x : crop_x + patch_hw]


def _raw_categorical_map(
    modality: str, frame: np.ndarray, source_path: str, crop: dict | None
) -> np.ndarray:
    from src.evaluation.internal.channel_head_evaluator import (
        global_channel111_category_id,
    )
    from wsg_reader.categorical_visualization import (
        channel195_semantic_map,
        coarse_material_category_map,
        coarse_thing_category_map,
        dense_channel111_category_map,
    )

    modality = str(modality)
    if modality == "111":
        category_id = global_channel111_category_id(frame, source_path)
        cropped = _apply_cached_crop(frame, crop)
        return dense_channel111_category_map(
            cropped,
            require_alpha=True,
            category_id=category_id,
            allow_region_mask_fallback=True,
            allow_category_mask_fallback=True,
            allow_rgb_mask_fallback=True,
        )
    cropped = _apply_cached_crop(frame, crop)
    if modality == "131":
        return coarse_material_category_map(cropped)
    if modality == "136":
        return coarse_thing_category_map(cropped, source_path)
    if modality == "195":
        return channel195_semantic_map(cropped)
    channel = int(modality)
    if channel >= cropped.shape[-1]:
        return np.full(cropped.shape[:2], -1.0, dtype=np.float32)
    values = np.asarray(cropped[..., channel], dtype=np.float32)
    minimum, maximum = _CATEGORICAL_TARGET_SCALAR_RANGES[modality]
    valid = (
        np.isfinite(values)
        & (values != -1.0)
        & (values >= float(minimum))
        & (values <= float(maximum))
    )
    output = np.full(values.shape, -1.0, dtype=np.float32)
    output[valid] = np.rint(values[valid])
    return output


def _categorical_contract_arrays(
    modality: str, semantic: np.ndarray, spatial_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    palette = _CATEGORICAL_TARGET_PALETTES.get(str(modality))
    if palette:
        values = tuple(sorted(int(value) for value in palette))
    else:
        minimum, maximum = _CATEGORICAL_TARGET_SCALAR_RANGES[str(modality)]
        values = tuple(range(int(minimum), int(maximum) + 1))
    index_map = np.zeros(semantic.shape, dtype=np.uint16)
    multihot = np.zeros(len(values), dtype=np.uint8)
    counts = np.zeros(len(values), dtype=np.int32)
    finite = np.isfinite(semantic) & (semantic != -1.0)
    rounded = np.rint(semantic).astype(np.int64, copy=False)
    value_to_index = {value: index for index, value in enumerate(values)}
    for value in np.unique(rounded[finite]).tolist() if finite.any() else []:
        category_index = value_to_index.get(int(value))
        if category_index is None:
            continue
        matched = finite & (rounded == int(value))
        multihot[category_index] = 1
        counts[category_index] = int(np.count_nonzero(matched))
        index_map[matched] = category_index + 1
    # Match PyTorch ``nearest-exact`` center sampling.  The VAE latent cell at
    # (i, j) represents the center of its source-image block, not an endpoint
    # from a linspace grid.
    y_indices = np.floor(
        (np.arange(int(spatial_size), dtype=np.float64) + 0.5)
        * index_map.shape[0]
        / int(spatial_size)
    ).astype(np.int64)
    x_indices = np.floor(
        (np.arange(int(spatial_size), dtype=np.float64) + 0.5)
        * index_map.shape[1]
        / int(spatial_size)
    ).astype(np.int64)
    y_indices = np.clip(y_indices, 0, index_map.shape[0] - 1)
    x_indices = np.clip(x_indices, 0, index_map.shape[1] - 1)
    spatial = index_map[np.ix_(y_indices, x_indices)]
    spatial_dtype = torch.uint16 if len(values) > 255 else torch.uint8
    return (
        torch.from_numpy(multihot),
        torch.from_numpy(counts),
        torch.from_numpy(spatial).to(spatial_dtype),
    )


def _categorical_prototypes(modality: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    palette = _CATEGORICAL_TARGET_PALETTES.get(str(modality))
    scalar_range = _CATEGORICAL_TARGET_SCALAR_RANGES.get(str(modality))
    if palette:
        values = tuple(sorted(int(value) for value in palette))
        prototypes = np.asarray([palette[value] for value in values], dtype=np.uint8)
        background = torch.zeros(3, dtype=torch.uint8)
    else:
        valid_minimum, valid_maximum = scalar_range
        values = tuple(range(int(valid_minimum), int(valid_maximum) + 1))
        display_minimum, display_maximum = _CATEGORICAL_TARGET_DISPLAY_RANGES[
            str(modality)
        ]
        fraction = (
            (np.asarray(values, dtype=np.float32) - float(display_minimum))
            / max(float(display_maximum - display_minimum), 1.0)
        )
        grayscale = np.rint(fraction * 255.0).astype(np.uint8)
        prototypes = np.repeat(grayscale[:, None], 3, axis=1)
        background = torch.zeros(3, dtype=torch.uint8)
    return (
        torch.as_tensor(values, dtype=torch.long),
        torch.from_numpy(prototypes),
        background,
    )


def _categorical_target_from_semantic_maps(
    modality: str, semantic_maps: list[np.ndarray], spatial_size: int = 60
) -> dict[str, torch.Tensor | None]:
    """Pack exact pre-render category maps into the compact sidecar contract."""
    rows = [
        _categorical_contract_arrays(modality, semantic, spatial_size)
        for semantic in semantic_maps
    ]
    values, prototypes, background = _categorical_prototypes(modality)
    return {
        "values": values,
        "multihot": torch.stack([row[0] for row in rows]),
        "pixel_counts": torch.stack([row[1] for row in rows]),
        "spatial_index_map": torch.stack([row[2] for row in rows]),
        "prototypes_rgb": prototypes,
        "background_rgb": background,
    }


def _backfill_categorical_sidecars(
    output_dir: Path,
    modalities: list[str],
    shard_id: int,
    num_shards: int,
    spatial_size: int,
) -> None:
    payloads: dict[str, dict] = {}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    sample_entries: dict[str, list[tuple[str, int]]] = {}
    for modality in modalities:
        latent_path = (
            output_dir / "shards" / _safe_modality_name(modality)
            / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
        )
        if not latent_path.exists():
            raise FileNotFoundError(latent_path)
        payload = torch.load(latent_path, map_location="cpu", mmap=True, weights_only=False)
        payloads[modality] = payload
        values, _prototypes, _background = _categorical_prototypes(modality)
        sample_count = len(payload["sample_ids"])
        spatial_dtype = torch.uint16 if len(values) > 255 else torch.uint8
        outputs[modality] = {
            "multihot": torch.zeros((sample_count, len(values)), dtype=torch.uint8),
            "counts": torch.zeros((sample_count, len(values)), dtype=torch.int32),
            "spatial": torch.zeros(
                (sample_count, int(spatial_size), int(spatial_size)), dtype=spatial_dtype
            ),
        }
        for row_index, sample_id in enumerate(payload["sample_ids"]):
            sample_entries.setdefault(str(sample_id), []).append((modality, row_index))

    failures: dict[str, list[dict[str, str]]] = {modality: [] for modality in modalities}
    for sample_number, (sample_id, entries) in enumerate(
        tqdm(sorted(sample_entries.items()), desc=f"categorical shard {shard_id}/{num_shards}", dynamic_ncols=True)
    ):
        reference_modality, reference_row = entries[0]
        try:
            frame, source_path = _load_cached_source_frame(
                payloads[reference_modality], reference_row
            )
        except Exception as exc:
            for modality, _row_index in entries:
                failures[modality].append(
                    {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
                )
            continue
        for modality, row_index in entries:
            try:
                payload = payloads[modality]
                crop = (payload.get("crop_infos") or [None] * len(payload["sample_ids"]))[row_index]
                semantic = _raw_categorical_map(modality, frame, source_path, crop)
                multihot, counts, spatial = _categorical_contract_arrays(
                    modality, semantic, spatial_size
                )
                outputs[modality]["multihot"][row_index] = multihot
                outputs[modality]["counts"][row_index] = counts
                outputs[modality]["spatial"][row_index] = spatial
            except Exception as exc:
                failures[modality].append(
                    {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
                )
        if sample_number and sample_number % 500 == 0:
            print(
                f"[categorical-sidecar] shard={shard_id}/{num_shards} "
                f"processed={sample_number}/{len(sample_entries)} "
                f"failures={sum(len(items) for items in failures.values())}",
                flush=True,
            )

    for modality in modalities:
        payload = payloads[modality]
        values, prototypes, background = _categorical_prototypes(modality)
        arrays = outputs[modality]
        sidecar_path = _categorical_target_sidecar_path(
            output_dir,
            modality,
            output_dir / "shards" / _safe_modality_name(modality)
            / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt",
        )
        sidecar = {
            "contract": {
                "version": 3,
                "kind": "semantic_categorical_target",
                "alignment": "latent_shard_sample_ids",
                "source": "pre_vae_target_variable_backfill",
                "runtime_requires_source_annotations": False,
                "spatial_encoding": "zero_background_one_based_category_index",
                "spatial_alignment": "latent_grid",
                "spatial_shape": [int(spatial_size), int(spatial_size)],
                "complete": len(failures[modality]) == 0,
                **_CATEGORICAL_TARGET_ONTOLOGIES.get(modality, {}),
            },
            "modality": modality,
            "sample_ids": list(payload["sample_ids"]),
            "category_values": values,
            "category_multihot": arrays["multihot"],
            "category_pixel_counts": arrays["counts"],
            "category_spatial_index_map": arrays["spatial"],
            "category_prototypes_rgb": prototypes.to(torch.uint8),
            "background_rgb": background,
            "failed_sample_ids": [item["sample_id"] for item in failures[modality]],
        }
        _atomic_torch_save(sidecar, sidecar_path)
        print(
            f"[categorical-sidecar] saved modality={modality} "
            f"multihot={tuple(arrays['multihot'].shape)} spatial={tuple(arrays['spatial'].shape)} "
            f"failures={len(failures[modality])} -> {sidecar_path}",
            flush=True,
        )
    if any(failures.values()):
        failure_path = (
            output_dir / "categorical_targets"
            / f"failures_shard_{shard_id:05d}_of_{num_shards:05d}.json"
        )
        _atomic_json_save({"failures_by_modality": failures}, failure_path)
        print(f"[categorical-sidecar] failures recorded -> {failure_path}", flush=True)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _builtin_config(preset: str) -> dict:
    """Return the reproducible pilot dataset contract without a YAML file."""
    presets = {
        "magick_transparency_v1": {
            "dataset_paths": ["data/custom_pilots/magick_transparency_v1/wsg_named"],
            "normalization_cache_path": "data/custom_pilots/magick_transparency_v1/wsg_named/normalization_cache.json",
            "modalities": ["0-2", "LayerDiffuse_Transparency"],
        },
        "iclight_relighting_v3": {
            "dataset_paths": ["data/custom_pilots/iclight_relighting_v3/named"],
            "normalization_cache_path": "data/custom_pilots/iclight_relighting_v3/named/normalization_cache.json",
            "modalities": ["0-2", "ICLight_Lighting", "ICLight_Relighted_RGB"],
        },
    }
    if preset not in presets:
        raise ValueError(f"Unknown preprocessing preset: {preset}")
    return {"dataset": {"name": "WSGModalityDataset", "params": {
        **presets[preset],
        "modality_config_path": "configs/wsg_v1_modalities_custom.json",
        "include_conditions": False, "data_type": "wsg",
        "normalization_mode": "linear",
        "crop_mode": "center_short_side_square", "patch_hw": 480,
        "num_frames": 1, "augment": False, "output_dtype": "float32",
        "split": None,
    }}}


def _build_dataset(config: dict):
    dataset_cfg = config["dataset"]
    if dataset_cfg["name"] != "WSGModalityDataset":
        raise ValueError("precompute_wsg_vae_latents.py currently expects dataset.name = WSGModalityDataset")
    params = dict(dataset_cfg["params"])
    for key in ("modality_config_path", "normalization_cache_path"):
        if key in params and params[key] is not None:
            params[key] = str(_resolve_project_path(params[key]))
    # Encoding must NEVER augment. WSGModalityDataset draws independent random
    # flip/rot90 PER modality, so with augment on the condition and target get
    # DIFFERENT transforms and become spatially misaligned in the latent cache
    # (the "rotated/flipped target" transpose bug). Force it off here regardless
    # of what the preset/config asks, so a stray "augment": true can never
    # silently corrupt an encode again. (Training-time datasets may still augment.)
    if params.get("augment"):
        print(
            "[precompute][WARN] dataset params requested augment=True; forcing "
            "augment=False for encoding (per-modality random aug misaligns "
            "condition vs target in the latent cache).",
            flush=True,
        )
    params["augment"] = False
    return WSGModalityDataset(**params)


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
        _merge_categorical_target_parts(parts, payload)
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



def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _select_flux_vae_path(model_params: dict, explicit_path: str | None) -> str:
    if explicit_path:
        return str(_resolve_project_path(explicit_path))
    flux_model_path = model_params.get("flux_model_path")
    if flux_model_path:
        return str(_resolve_project_path(flux_model_path))
    model_paths = model_params.get("model_paths")
    if model_paths is not None:
        paths = json.loads(model_paths) if isinstance(model_paths, str) else model_paths
        for path in paths:
            path_str = str(path)
            if "flux" in path_str.lower() or path_str.endswith("/vae"):
                return str(_resolve_project_path(path_str))
    return str(_resolve_project_path("models/black-forest-labs/FLUX.2-klein-4B"))


def _load_flux_vae(model_or_vae_path: str, device: str, torch_dtype: torch.dtype) -> AutoencoderKLFlux2:
    path = Path(model_or_vae_path)
    if (path / "vae").is_dir():
        vae = AutoencoderKLFlux2.from_pretrained(str(path), subfolder="vae", torch_dtype=torch_dtype)
    else:
        vae = AutoencoderKLFlux2.from_pretrained(str(path), torch_dtype=torch_dtype)
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    vae.to(device=device, dtype=torch_dtype)
    vae.eval()
    return vae


def _frame_group_metadata(sample_id: str, sample_source: dict) -> dict:
    """Keep frame samples regroupable by source video without changing intersection keys."""
    path = str(sample_source.get("path") or "")
    video_name = str(sample_source.get("video_name") or "")
    video_id = video_name or path or str(sample_id).split("::", 1)[0]
    frame_keys = list(sample_source.get("frame_keys") or [])
    frame_key = str(frame_keys[0]) if frame_keys else ""
    frame_index = None
    digits = "".join(ch for ch in frame_key if ch.isdigit())
    if digits:
        try:
            frame_index = int(digits)
        except ValueError:
            frame_index = None
    return {
        "video_id": video_id,
        "frame_key": frame_key,
        "frame_index": frame_index,
    }


def _prepare_flux_images(video: torch.Tensor, device: str, torch_dtype: torch.dtype) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected frame-level tensor [B,C,1,H,W], got {tuple(video.shape)}")
    if int(video.shape[2]) != 1:
        raise ValueError(
            "FLUX frame cache expects dataset num_frames=1. "
            f"Got temporal length {int(video.shape[2])}."
        )
    images = video[:, :, 0].to(device=device, dtype=torch_dtype)
    return images


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Explicit dataset YAML for standard pipelines")
    source.add_argument(
        "--preset",
        choices=["magick_transparency_v1", "iclight_relighting_v3"],
        help="Built-in pilot preprocessing contract; no YAML file is needed",
    )
    parser.add_argument("--output-dir", default="data/wsg_flux2_frame_latents_dense_multidataset")
    parser.add_argument("--dataset-paths", nargs="+", default=None,
                        help="Override the preset/config dataset_paths (discovery roots). Use to encode only a "
                             "subset, e.g. an incremental refill dir, without re-encoding the base data.")
    parser.add_argument("--flux-model-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cache-dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--filter-invalid-modalities", dest="filter_invalid_modalities", action="store_true", default=True)
    parser.add_argument("--no-filter-invalid-modalities", dest="filter_invalid_modalities", action="store_false")
    parser.add_argument("--sample-major-merge-every", type=int, default=100)
    parser.add_argument("--crop-mode", choices=["center_short_side_square", "random"], default=None)
    parser.add_argument("--sparse-codec-mode", default="disabled")
    parser.add_argument("--sparse-heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--sparse-heatmap-radius", type=int, default=8)
    parser.add_argument("--sparse-heatmap-peak-threshold", type=float, default=0.10)
    parser.add_argument("--sparse-heatmap-nms-radius", type=int, default=6)
    parser.add_argument(
        "--categorical-sidecars-only",
        action="store_true",
        help=(
            "Backfill exact category sets and latent-grid category/mask maps from "
            "the pre-VAE variables without loading or rewriting the VAE latent shards."
        ),
    )
    parser.add_argument("--categorical-spatial-size", type=int, default=60)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")

    if args.categorical_sidecars_only:
        requested = list(args.modalities or _CATEGORICAL_TARGET_MODALITIES)
        unsupported = sorted(set(requested) - set(_CATEGORICAL_TARGET_MODALITIES), key=int)
        if unsupported:
            raise ValueError(
                f"Non-categorical modalities requested for sidecar backfill: {unsupported}"
            )
        if int(args.categorical_spatial_size) <= 0:
            raise ValueError("--categorical-spatial-size must be positive")
        output_dir = _resolve_project_path(args.output_dir)
        _backfill_categorical_sidecars(
            output_dir=output_dir,
            modalities=requested,
            shard_id=int(args.shard_id),
            num_shards=int(args.num_shards),
            spatial_size=int(args.categorical_spatial_size),
        )
        return

    torch_dtype = _dtype_from_name(args.dtype)
    cache_dtype = _dtype_from_name(args.cache_dtype or args.dtype)
    if args.preset:
        config = _builtin_config(args.preset)
        config_source = f"preset:{args.preset}"
    else:
        config_source = str(_resolve_project_path(args.config))
        config = _load_config(config_source)
    params = config["dataset"]["params"]
    if args.dataset_paths is not None:
        params["dataset_paths"] = list(args.dataset_paths)
        print(f"[precompute] dataset_paths overridden -> {params['dataset_paths']}", flush=True)
    params["num_frames"] = 1
    params["video_window_stride"] = 1
    params["sparse_codec_mode"] = args.sparse_codec_mode
    params["sparse_heatmap_sigma"] = float(args.sparse_heatmap_sigma)
    params["sparse_heatmap_radius"] = int(args.sparse_heatmap_radius)
    params["sparse_heatmap_peak_threshold"] = float(args.sparse_heatmap_peak_threshold)
    params["sparse_heatmap_nms_radius"] = int(args.sparse_heatmap_nms_radius)
    if args.crop_mode is not None:
        params["crop_mode"] = args.crop_mode

    dataset = _build_dataset(config)
    if int(dataset.num_frames) != 1:
        raise ValueError(f"FLUX frame cache requires num_frames=1, got {dataset.num_frames}")
    modality_keys = list(dataset.modality_keys)
    if args.modalities is not None:
        unknown = sorted(set(args.modalities) - set(modality_keys))
        if unknown:
            raise ValueError(f"Unknown modalities requested: {unknown}; available={modality_keys}")
        modality_keys = list(args.modalities)
    if not modality_keys:
        raise ValueError("No modalities selected for FLUX frame cache")

    sparse_modalities = sparse_modalities_from_config(config["dataset"]["params"]["modality_config_path"])
    active_sparse_modalities = [
        modality
        for modality in modality_keys
        if modality in sparse_modalities
        and build_sparse_codec(
            modality,
            mode=args.sparse_codec_mode,
            sigma=args.sparse_heatmap_sigma,
            radius=args.sparse_heatmap_radius,
            peak_threshold=args.sparse_heatmap_peak_threshold,
            nms_radius=args.sparse_heatmap_nms_radius,
        )
        is not None
    ]

    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flux_vae_path = _select_flux_vae_path(config.get("model", {}).get("params", {}), args.flux_model_path)
    vae = _load_flux_vae(flux_vae_path, args.device, torch_dtype)

    def encode_frame_tensor(video: torch.Tensor) -> torch.Tensor:
        batch_size, channels = int(video.shape[0]), int(video.shape[1])
        if channels == 3:
            encoded = vae.encode(_prepare_flux_images(video, args.device, torch_dtype)).latent_dist.sample()
            # Keep the legacy WSG latent layout [B,C,T,H,W] with T=1.
            return encoded.detach().cpu().to(dtype=cache_dtype).unsqueeze(2)
        if channels % 3 != 0:
            raise ValueError(f"FLUX VAE pack input channels must be divisible by 3, got {channels}")
        packs = channels // 3
        packed = video.contiguous().reshape(batch_size, packs, 3, *video.shape[2:]).reshape(batch_size * packs, 3, *video.shape[2:])
        encoded = vae.encode(_prepare_flux_images(packed, args.device, torch_dtype)).latent_dist.sample()
        encoded = encoded.detach().cpu().to(dtype=cache_dtype).unsqueeze(2)
        encoded = encoded.reshape(batch_size, packs, *encoded.shape[1:])
        return encoded.reshape(batch_size, packs * encoded.shape[2], *encoded.shape[3:])

    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    if total <= 0:
        raise ValueError("No samples to encode")
    shard_indices = list(range(args.shard_id, total, args.num_shards))
    if not shard_indices:
        raise ValueError(f"Shard {args.shard_id}/{args.num_shards} has no samples")

    sample_ids = [dataset.samples[index]["sample_id"] for index in shard_indices]
    prompts = [dataset.samples[index].get("prompt", "") for index in shard_indices]
    common_meta = {
        "config": config_source,
        "modality_config_path": str(config["dataset"]["params"].get("modality_config_path", "")),
        "dataset_paths": list(getattr(dataset, "dataset_paths", [])),
        "normalization_dataset_paths": list(getattr(dataset, "normalization_dataset_paths", [])),
        "vae_path": flux_vae_path,
        "vae_family": "flux2_klein_frame",
        "latent_backend": "flux2_frame",
        "num_samples": len(shard_indices),
        "global_num_samples": total,
        "sample_ids": sample_ids,
        "shard_indices": shard_indices,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_frames": 1,
        "latent_num_frames": 1,
        "patch_hw": int(dataset.patch_hw),
        "height": int(dataset.patch_hw),
        "width": int(dataset.patch_hw),
        "crop_mode": getattr(dataset, "crop_mode", "random"),
        "normalization_mode": dataset.normalization_mode,
        "normalization_modes": dict(getattr(dataset, "modality_normalization_modes", {})),
        "data_type": dataset.data_type,
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype or args.dtype,
        "vae_force_upcast": bool(getattr(vae.config, "force_upcast", False)),
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
                "config": config_source,
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

    if args.skip_existing and all(out_path_for(modality).exists() for modality in modality_keys):
        print(f"skip existing shard {args.shard_id}/{args.num_shards}: {modality_keys}")
        return

    completed_batches = _load_completed_sample_major_batches(
        output_dir,
        args.shard_id,
        args.num_shards,
        expected_batch_size=args.batch_size,
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
    loader = DataLoader(
        Subset(dataset, pending_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=_collate_wsg_modality_batch,
    )

    for pending_index, batch in enumerate(tqdm(loader, desc=f"encode FLUX frame shard {args.shard_id}/{args.num_shards}", dynamic_ncols=True)):
        batch_id = pending_batch_ids[pending_index]
        offset = batch_id * args.batch_size
        batch_crop_infos = batch.get("crop_info")
        if batch_crop_infos is None:
            raise ValueError("WSGModalityDataset did not provide crop_info; cannot write aligned latent cache")
        batch_crop_infos_list = _batch_crop_infos_to_list(batch_crop_infos)
        batch_sample_sources = _batch_mapping_to_list(batch.get("sample_source", {}))
        batch_sample_ids = list(batch.get("sample_id") or [])
        videos = [batch["modalities"][modality] for modality in modality_keys]
        batch_size = int(videos[0].shape[0])
        if not batch_sample_ids:
            batch_sample_ids = [dataset.samples[shard_indices[offset + item]]["sample_id"] for item in range(batch_size)]
        if not batch_sample_sources:
            batch_sample_sources = [
                WSGModalityDataset._sample_source(dataset.samples[shard_indices[offset + item]])
                for item in range(batch_size)
            ]
        valid_flags = _batch_valid_flags(batch.get("modality_valid"), modality_keys, batch_size)
        categorical_semantics: dict[str, dict[int, np.ndarray]] = {
            modality: {}
            for modality in modality_keys
            if modality in _CATEGORICAL_TARGET_MODALITIES
        }
        if categorical_semantics:
            for local_idx in range(batch_size):
                requested = [
                    modality
                    for modality in categorical_semantics
                    if bool(valid_flags[modality][local_idx]) or not args.filter_invalid_modalities
                ]
                if not requested:
                    continue
                frame, source_path = _load_source_frame(
                    batch_sample_sources[local_idx], batch_sample_ids[local_idx]
                )
                for modality in requested:
                    categorical_semantics[modality][local_idx] = _raw_categorical_map(
                        modality,
                        frame,
                        source_path,
                        batch_crop_infos_list[local_idx],
                    )
        completed_modalities = _load_completed_sample_major_modalities(
            output_dir,
            modality_keys,
            args.shard_id,
            args.num_shards,
            batch_id,
            expected_batch_size=batch_size,
        )
        batch_part_modalities = set(completed_modalities)
        for idx, (modality, video) in enumerate(zip(modality_keys, videos)):
            if modality in completed_modalities:
                continue
            valid_mask = valid_flags[modality]
            if not args.filter_invalid_modalities:
                valid_mask = torch.ones(batch_size, dtype=torch.bool)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            selected_indices = [int(item) for item in valid_indices.tolist()]
            selected_video = video.index_select(0, valid_indices)
            print(
                f"[flux-frame] shard={args.shard_id}/{args.num_shards} "
                f"batch={batch_id} modality={modality} valid={len(selected_indices)}/{batch_size}",
                flush=True,
            )
            modality_latents = encode_frame_tensor(selected_video)
            frame_meta = [
                _frame_group_metadata(batch_sample_ids[local_idx], batch_sample_sources[local_idx])
                for local_idx in selected_indices
            ]
            payload = {
                **common_meta,
                "num_samples": len(selected_indices),
                "modality": modality,
                "sample_ids": [batch_sample_ids[local_idx] for local_idx in selected_indices],
                "sample_sources": [batch_sample_sources[local_idx] for local_idx in selected_indices],
                "video_ids": [item["video_id"] for item in frame_meta],
                "frame_keys": [item["frame_key"] for item in frame_meta],
                "frame_indices": [item["frame_index"] for item in frame_meta],
                "crop_infos": [batch_crop_infos_list[local_idx] for local_idx in selected_indices],
                "latents": modality_latents,
                "source_shard_indices": [shard_indices[offset + local_idx] for local_idx in selected_indices],
                "shard_indices": [shard_indices[offset + local_idx] for local_idx in selected_indices],
                "filtered_sample_ids": bool(args.filter_invalid_modalities),
                "filter_invalid_modalities": bool(args.filter_invalid_modalities),
                "part_batch_id": int(batch_id),
                "part_batch_size": int(batch_size),
                "part_modality_index": int(idx),
            }
            if modality in categorical_semantics:
                categorical_target = _categorical_target_from_semantic_maps(
                    modality,
                    [categorical_semantics[modality][local_idx] for local_idx in selected_indices],
                    spatial_size=int(args.categorical_spatial_size),
                )
                payload["categorical_target_values"] = categorical_target["values"]
                payload["categorical_target_multihot"] = categorical_target["multihot"]
                payload["categorical_target_pixel_counts"] = categorical_target["pixel_counts"]
                payload["categorical_target_spatial_index_map"] = categorical_target["spatial_index_map"]
                payload["categorical_target_prototypes_rgb"] = categorical_target["prototypes_rgb"]
                payload["categorical_target_background_rgb"] = categorical_target["background_rgb"]
            _atomic_torch_save(
                payload,
                _sample_major_modality_part_path(output_dir, modality, args.shard_id, args.num_shards, batch_id),
                storage_dtype=cache_dtype,
            )
            batch_part_modalities.add(modality)
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
        chunks = [part["latents"] for part in parts if int(part["latents"].shape[0]) > 0]
        latents = torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=cache_dtype)
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
            "filtered_sample_ids": bool(args.filter_invalid_modalities),
            "filter_invalid_modalities": bool(args.filter_invalid_modalities),
            "part_cache_format": "sample_major_batch_parts_v1",
        }
        _merge_categorical_target_parts(parts, payload)
        if "categorical_target_multihot" in payload:
            payload["categorical_target_ontology"] = dict(
                _CATEGORICAL_TARGET_ONTOLOGIES.get(modality, {})
            )
        _atomic_torch_save(payload, out_path, storage_dtype=cache_dtype)
        if "categorical_target_multihot" in payload:
            sidecar_path = _categorical_target_sidecar_path(output_dir, modality, out_path)
            sidecar = {
                "contract": {
                    "version": 3,
                    "kind": "semantic_categorical_target",
                    "alignment": "latent_shard_sample_ids",
                    "source": "pre_vae_target_variable",
                    "runtime_requires_source_annotations": False,
                    "spatial_encoding": "zero_background_one_based_category_index",
                    "spatial_alignment": "latent_grid",
                    **_CATEGORICAL_TARGET_ONTOLOGIES.get(modality, {}),
                },
                "modality": modality,
                "sample_ids": list(payload["sample_ids"]),
                "category_values": payload["categorical_target_values"].to(torch.long),
                "category_multihot": payload["categorical_target_multihot"].to(torch.uint8),
                "category_pixel_counts": payload["categorical_target_pixel_counts"].to(torch.int32),
                "category_spatial_index_map": payload["categorical_target_spatial_index_map"],
                "category_prototypes_rgb": payload["categorical_target_prototypes_rgb"].to(torch.uint8),
                "background_rgb": payload.get("categorical_target_background_rgb"),
            }
            _atomic_torch_save(sidecar, sidecar_path)
            print(
                f"saved categorical target metadata: {tuple(sidecar['category_multihot'].shape)} "
                f"-> {sidecar_path}",
                flush=True,
            )
        # The final shard is now durable; remove merged batch/chunk parts to avoid a second copy.
        removed_parts = 0
        for part in parts:
            part_path = part.get("_part_path")
            if not part_path:
                continue
            part_path = Path(part_path)
            if part_path.exists():
                part_path.unlink()
                removed_parts += 1
        print(
            f"saved {modality}: {tuple(payload['latents'].shape)} -> {out_path} "
            f"(removed_parts={removed_parts})",
            flush=True,
        )


if __name__ == "__main__":
    main()
