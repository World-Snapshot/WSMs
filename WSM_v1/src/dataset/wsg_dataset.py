import csv
import bisect
import json
from datetime import datetime
import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    categorical_ids_to_rgb,
    channel195_semantic_map,
    dense_channel111_category_map,
)
from tqdm import tqdm

from src.pipelines.wsg_sparse_codecs import build_sparse_codec, sparse_modalities_from_config


MASK_VALUE = -1.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_AND_MULTI_LANES_PREFIX = "lora_and_muti-lanes_index_"
LORA_SINGLE_CONDITION_PREFIX = "lora_"
LORA_GROUP_META_KEY = "__meta__"
KNOWN_WSG_DATASET_FAMILIES = (
    "AVIS",
    "Articulation_results",
    "ShapeNetSem_results",
    "MatSynth_results",
    "XPose_results",
    "MMFi_Dataset_results",
    "human_signer_480",
    "unknown",
)
MODALITY_DATASET_FAMILY_BLACKLISTS = {
    # MatSynth normal maps are [0, 1] texture-space maps and ShapeNet/Object
    # normals are not distribution-compatible with AVIS geometry normals.
    "36-38": tuple(family for family in KNOWN_WSG_DATASET_FAMILIES if family != "AVIS"),
    # Channel 3 is used as RGBA alpha in ShapeNet/Object. Other datasets may
    # carry alpha-like data, but should not train the alpha semantic.
    "3": tuple(family for family in KNOWN_WSG_DATASET_FAMILIES if family != "ShapeNetSem_results"),
}
NORMALIZATION_MODE_ALIASES = {
    # XPose RGB/pose images are encoded with the same [0, 255] -> [-1, 1]
    # transform represented by the regular linear WSG cache at runtime.
    "xpose_rgb_minus1_1": "linear",
}


def _canonical_normalization_mode(mode: Any) -> str:
    mode = str(mode or "linear").lower()
    return NORMALIZATION_MODE_ALIASES.get(mode, mode)


def load_wsg_normalization_contract(path: str | Path) -> dict[str, str]:
    """Load the inference-visible per-modality normalization contract."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"{path} must contain a top-level 'normalization' object")
    default_mode = _canonical_normalization_mode(normalization.get("default_mode", "linear"))
    if default_mode not in {"linear", "log"}:
        raise ValueError(f"{path} has invalid normalization.default_mode={default_mode!r}")
    overrides = normalization.get("modalities") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path} normalization.modalities must be an object")
    contract = {"__default__": default_mode}
    for modality, mode in overrides.items():
        canonical = _canonical_normalization_mode(mode)
        if canonical not in {"linear", "log"}:
            raise ValueError(f"{path} has invalid normalization mode for {modality}: {mode!r}")
        contract[str(modality)] = canonical
    return contract


def normalization_mode_for_modality(contract: Mapping[str, str], modality: str) -> str:
    return str(contract.get(str(modality), contract.get("__default__", "linear")))


def effective_cached_normalization_mode(modality: str, recorded_mode: Any) -> str:
    """Return the transform that produced a cached modality.

    Historical dense-cache payloads record the dataset-wide mode for RGB even
    though :meth:`WSGNormalizationStore.normalize_channel` has always encoded
    channels 0--2 with the fixed ``value / 255`` transform. Treating the old
    metadata label literally would reject otherwise valid RGB caches after the
    public normalization contract was corrected.
    """
    if str(modality) == "0-2":
        return "linear"
    return _canonical_normalization_mode(recorded_mode)


def load_wsg_normalization_ranges(path: str | Path) -> dict[int, tuple[float, float]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ranges = (payload.get("normalization") or {}).get("channel_ranges") or {}
    result = {}
    for channel, value_range in ranges.items():
        if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
            result[int(channel)] = (float(value_range[0]), float(value_range[1]))
    return result


def _is_lora_group_meta_key(key: str) -> bool:
    return str(key).startswith("__")


def _is_lora_group_key(key: str) -> bool:
    key = str(key)
    return key.startswith(LORA_AND_MULTI_LANES_PREFIX) or key.startswith(LORA_SINGLE_CONDITION_PREFIX)


def _condition_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


NON_LATENT_CONDITION_TOKENS = {"caption", "Reference_IMG"}


def _latent_condition_list(value) -> List[str]:
    return [item for item in _condition_list(value) if item not in NON_LATENT_CONDITION_TOKENS]


def _flat_lora_entry_to_group(entry: dict) -> dict:
    target = str(entry["target"])
    conditions = _condition_list(entry.get("condition"))
    spec = {
        "description": str(entry.get("target_description") or "").strip(),
        "conditions": {condition: {} for condition in _latent_condition_list(conditions)},
    }
    if "Reference_IMG" in conditions:
        spec["reference_image_condition"] = True
    return {target: spec}


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _torch_dtype_from_numpy(dtype: np.dtype) -> Optional[torch.dtype]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float16"):
        return torch.float16
    if dtype == np.dtype("float32"):
        return torch.float32
    if dtype == np.dtype("float64"):
        return torch.float64
    return None


def parse_channel_key(channel_key: str) -> List[int]:
    channels: List[int] = []
    for part in str(channel_key).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            channels.extend(range(int(start), int(end) + 1))
        else:
            channels.append(int(part))
    return channels


def safe_modality_name(modality: str) -> str:
    return str(modality).replace("/", "_").replace(",", "_").replace("-", "_")


def _sample_file_stem(path: Path) -> str:
    name = path.name
    for suffix in (".wsg.npz", ".wsv.npz", ".wsg", ".wsv", ".npz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _caption_lookup_keys(value: str) -> List[str]:
    value = str(value or "").strip()
    if not value:
        return []
    keys = [value]
    path = Path(value)
    keys.extend([path.name, path.stem])
    for suffix in (".wsg.npz", ".wsv.npz", ".wsg", ".wsv", ".npz"):
        if value.endswith(suffix):
            keys.append(value[: -len(suffix)])
            keys.append(path.name[: -len(suffix)])
    if "_frame" in value:
        keys.append(value.split("_frame", 1)[0])
    uuid_match = re.match(
        r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        value,
    )
    if uuid_match:
        keys.append(uuid_match.group(1))
    compact_id_match = re.match(r"^([0-9a-fA-F]{32})", value)
    if compact_id_match:
        keys.append(compact_id_match.group(1))
    return [key for key in dict.fromkeys(keys) if key]


def _clean_caption_text(text: Any, max_chars: int = 2048) -> str:
    if text is None:
        return ""
    text = " ".join(str(text).replace("\x00", " ").split())
    if not text or text.lower() in {"none", "null", "nan", "[]", "{}"}:
        return ""
    return text[:max_chars]


def _join_caption_parts(parts: Sequence[Any]) -> str:
    cleaned = []
    for part in parts:
        text = _clean_caption_text(part)
        if text and text not in cleaned:
            cleaned.append(text)
    return " ".join(cleaned)


def _caption_from_mapping(data: Any) -> str:
    primary_keys = (
        "caption",
        "captions",
        "prompt",
        "prompts",
        "text",
        "description",
        "descriptions",
        "parameter_caption",
    )
    fallback_keys = (
        "category_label",
        "category",
        "class_label",
        "class_name",
        "material_name",
        "object_name",
        "model_name",
        "name",
    )

    def collect(obj: Any, keys: Sequence[str]) -> List[str]:
        values: List[str] = []
        if isinstance(obj, dict):
            lowered = {str(key).lower(): value for key, value in obj.items()}
            for key in keys:
                if key in lowered:
                    value = lowered[key]
                    if isinstance(value, (list, tuple)):
                        values.extend(_clean_caption_text(item) for item in value)
                    elif not isinstance(value, (dict, list)):
                        values.append(_clean_caption_text(value))
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    values.extend(collect(value, keys))
        elif isinstance(obj, list):
            for value in obj:
                values.extend(collect(value, keys))
        return [value for value in values if value]

    primary = collect(data, primary_keys)
    if primary:
        return _join_caption_parts(primary)
    fallback = collect(data, fallback_keys)
    return _join_caption_parts(fallback[:4])


def flatten_wsg_modality_graph(graph: dict) -> dict:
    """Accept either the legacy target graph or the LoRA-grouped graph."""
    if not isinstance(graph, dict) or not graph:
        return graph
    if not all(_is_lora_group_key(key) for key in graph):
        return graph

    flattened = {}
    for lora_key, targets in graph.items():
        if not isinstance(targets, dict) or not targets:
            raise ValueError(f"{lora_key} must contain one or more target modality entries")
        if targets.get("target") and targets.get("condition") and str(targets["target"]) not in targets:
            targets = _flat_lora_entry_to_group(targets)
        for target, spec in targets.items():
            if _is_lora_group_meta_key(target):
                continue
            if target in flattened:
                raise ValueError(f"Duplicate target modality {target} in LoRA-grouped WSG graph")
            flattened[target] = spec
        if not any(not _is_lora_group_meta_key(target) for target in targets):
            raise ValueError(f"{lora_key} must contain one or more target modality entries")
    return flattened


def wsg_lora_groups_from_graph(graph: dict) -> list[dict]:
    """Return LoRA training groups without discarding grouped JSON structure."""
    if not isinstance(graph, dict) or not graph:
        return []
    if all(_is_lora_group_key(key) for key in graph):
        groups = []
        for lora_key, targets in graph.items():
            if not isinstance(targets, dict) or not targets:
                raise ValueError(f"{lora_key} must contain one or more target modality entries")
            if targets.get("target") and targets.get("condition") and str(targets["target"]) not in targets:
                targets = _flat_lora_entry_to_group(targets)
            target_names = [str(target) for target in targets if not _is_lora_group_meta_key(target)]
            if not target_names:
                raise ValueError(f"{lora_key} must contain one or more target modality entries")
            group = {"lora": str(lora_key), "targets": target_names}
            meta = targets.get(LORA_GROUP_META_KEY)
            if isinstance(meta, dict):
                group.update({f"meta_{key}": value for key, value in meta.items()})
            groups.append(group)
        return groups
    return [{"lora": str(target), "targets": [str(target)]} for target in graph]


def random_resize_crop(data: np.ndarray, patch_hw: int, crop_mode: str = "random") -> np.ndarray:
    cropped, _info = random_resize_crop_with_info(data, patch_hw, crop_mode=crop_mode)
    return cropped


def random_resize_crop_with_info(data: np.ndarray, patch_hw: int, crop_mode: str = "random") -> tuple[np.ndarray, dict]:
    if data.ndim == 3:
        h, w, c = data.shape
        pad_width = ((0, 0), (0, 0), (0, 0))
    elif data.ndim == 4:
        _, h, w, c = data.shape
        pad_width = ((0, 0), (0, 0), (0, 0), (0, 0))
    else:
        raise ValueError(f"Expected WSG/WSV data with 3 or 4 dimensions, got shape {data.shape}")
    original_h, original_w = int(h), int(w)
    pad_h = max(0, patch_hw - h)
    pad_w = max(0, patch_hw - w)
    if h < patch_hw or w < patch_hw:
        if data.ndim == 3:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        else:
            pad_width = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
        data = np.pad(data, pad_width, mode="constant", constant_values=MASK_VALUE)
        h, w = data.shape[-3:-1]
    crop_mode = str(crop_mode or "random").lower()
    if crop_mode in {"center", "center_crop", "center_short_side", "center_short_side_square"}:
        crop_y = max(0, (h - patch_hw) // 2)
        crop_x = max(0, (w - patch_hw) // 2)
        crop_mode = "center_short_side_square"
    elif crop_mode == "random":
        crop_y = 0 if h == patch_hw else np.random.randint(0, h - patch_hw + 1)
        crop_x = 0 if w == patch_hw else np.random.randint(0, w - patch_hw + 1)
    else:
        raise ValueError(
            "crop_mode must be 'center_short_side_square' or 'random', "
            f"got {crop_mode!r}"
        )
    info = {
        "patch_hw": int(patch_hw),
        "crop_mode": crop_mode,
        "original_height": original_h,
        "original_width": original_w,
        "padded_height": int(h),
        "padded_width": int(w),
        "pad_bottom": int(pad_h),
        "pad_right": int(pad_w),
        "crop_y": int(crop_y),
        "crop_x": int(crop_x),
    }
    if data.ndim == 3:
        return data[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :], info
    return data[:, crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :], info


def apply_resize_crop_info(data: np.ndarray, crop_info: dict) -> np.ndarray:
    patch_hw = int(crop_info["patch_hw"])
    crop_y = int(crop_info["crop_y"])
    crop_x = int(crop_info["crop_x"])
    if data.ndim == 3:
        h, w, _c = data.shape
        pad_width = ((0, max(0, patch_hw - h)), (0, max(0, patch_hw - w)), (0, 0))
    elif data.ndim == 4:
        _t, h, w, _c = data.shape
        pad_width = ((0, 0), (0, max(0, patch_hw - h)), (0, max(0, patch_hw - w)), (0, 0))
    else:
        raise ValueError(f"Expected WSG/WSV data with 3 or 4 dimensions, got shape {data.shape}")
    if any(width[1] > 0 for width in pad_width):
        data = np.pad(data, pad_width, mode="constant", constant_values=MASK_VALUE)
    if data.ndim == 3:
        return data[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :]
    return data[:, crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :]


class WSGNormalizationStore:
    def __init__(
        self,
        cache_file: Optional[str] = None,
        normalization_mode: str = "linear",
        show_progress: bool = False,
        portable_channel_ranges: Optional[Mapping[int, Sequence[float]]] = None,
    ):
        default_path = Path(__file__).resolve().parents[2] / "configs" / "wsg_dataset_info.json"
        self.cache_file = Path(cache_file).expanduser().resolve() if cache_file else default_path
        self.normalization_mode = str(normalization_mode).lower()
        if self.normalization_mode not in {"linear", "log"}:
            raise ValueError("normalization_mode must be either 'linear' or 'log'")
        self.show_progress = bool(show_progress)
        self.portable_channel_ranges = {
            int(channel): (float(value_range[0]), float(value_range[1]))
            for channel, value_range in (portable_channel_ranges or {}).items()
        }
        self.cache = self._load_cache()
        self._range_cache: Dict[tuple[int, tuple[str, ...]], Optional[tuple[float, float]]] = {}

    def _load_cache(self) -> dict:
        if not self.cache_file.exists():
            return {}
        with self.cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.cache_file)

    def get_channel_range(self, channel: int, dataset_paths: Sequence[str]) -> Optional[tuple[float, float]]:
        if int(channel) in self.portable_channel_ranges:
            return self.portable_channel_ranges[int(channel)]
        range_key = (int(channel), tuple(str(_resolve_project_path(path).resolve()) for path in dataset_paths))
        if range_key in self._range_cache:
            return self._range_cache[range_key]

        min_val = None
        max_val = None
        saw_cached_channel = False
        missing_cached_channel = False
        for dataset_path in dataset_paths:
            entry = self.cache.get(str(_resolve_project_path(dataset_path).resolve()))
            if not isinstance(entry, dict):
                entry = self.cache.get(str(dataset_path))
            if not isinstance(entry, dict):
                missing_cached_channel = True
                continue
            channel_info = (entry.get("channels") or {}).get(str(channel))
            if not isinstance(channel_info, dict):
                missing_cached_channel = True
                continue
            saw_cached_channel = True
            if channel_info.get("has_data") is False:
                continue
            if not channel_info.get("has_data", False):
                missing_cached_channel = True
                continue
            norm = channel_info.get("normalization")
            if not isinstance(norm, dict):
                missing_cached_channel = True
                continue
            cur_min = float(norm["min_val"])
            cur_max = float(norm["max_val"])
            min_val = cur_min if min_val is None else min(min_val, cur_min)
            max_val = cur_max if max_val is None else max(max_val, cur_max)
        if min_val is None or max_val is None:
            if saw_cached_channel and not missing_cached_channel:
                self._range_cache[range_key] = None
                return None
            computed = self._compute_and_cache_channel_range(channel, dataset_paths)
            self._range_cache[range_key] = computed
            return computed

        result = (min_val, max_val)
        self._range_cache[range_key] = result
        return result

    @staticmethod
    def _iter_wsg_arrays(path: Path, max_files: Optional[int] = 200):
        candidates = sorted(list(path.glob("**/*.wsg")) + list(path.glob("**/*.wsg.npz")))
        wsv_candidates = sorted(list(path.glob("**/*.wsv")) + list(path.glob("**/*.wsv.npz")))
        all_items = [("wsg", p, None) for p in candidates]
        for wsv_path in wsv_candidates:
            try:
                with np.load(wsv_path) as data:
                    for key in data.files:
                        all_items.append(("wsv", wsv_path, key))
            except Exception:
                continue
        if max_files is not None and len(all_items) > max_files:
            all_items = random.sample(all_items, max_files)
        for storage, file_path, frame_key in all_items:
            try:
                with np.load(file_path) as data:
                    if storage == "wsv":
                        yield str(file_path), data[frame_key]
                    elif "data" in data:
                        yield str(file_path), data["data"]
                    elif "wsg_data" in data:
                        yield str(file_path), data["wsg_data"]
            except Exception:
                continue

    @staticmethod
    def _channel_values(wsg_array: np.ndarray, channel: int) -> np.ndarray:
        if channel >= wsg_array.shape[-1]:
            return np.empty(0, dtype=np.float32)
        values = np.asarray(wsg_array[..., channel]).reshape(-1)
        values = values[values != MASK_VALUE]
        if values.size == 0:
            return values.astype(np.float32, copy=False)
        return values[np.isfinite(values)].astype(np.float32, copy=False)

    def _scan_channel_stats(
        self,
        root: Path,
        channel: int,
        max_files: Optional[int],
        max_values_per_array: Optional[int],
        collect_unique: bool,
        desc_prefix: str,
    ) -> dict:
        stats = {
            "values_seen": 0,
            "local_min": None,
            "local_max": None,
            "is_integer_like": True,
            "unique_values": set(),
            "unique_overflow": False,
        }
        arrays = self._iter_wsg_arrays(root, max_files=max_files)
        if self.show_progress:
            arrays = tqdm(arrays, desc=f"{desc_prefix} ch{channel} {root.name}", unit="array", dynamic_ncols=True)
        for _, wsg_array in arrays:
            values = self._channel_values(wsg_array, channel)
            if values.size == 0:
                continue
            if max_values_per_array is not None and values.size > max_values_per_array:
                indices = np.random.choice(values.size, max_values_per_array, replace=False)
                values = values[indices]
            stats["values_seen"] += int(values.size)
            cur_min = float(np.min(values))
            cur_max = float(np.max(values))
            stats["local_min"] = cur_min if stats["local_min"] is None else min(stats["local_min"], cur_min)
            stats["local_max"] = cur_max if stats["local_max"] is None else max(stats["local_max"], cur_max)
            if stats["is_integer_like"] and not np.allclose(values, np.round(values)):
                stats["is_integer_like"] = False
            if collect_unique and not stats["unique_overflow"]:
                unique_values = np.unique(values)
                if unique_values.size > 4096:
                    stats["unique_overflow"] = True
                    stats["unique_values"].clear()
                else:
                    stats["unique_values"].update(float(value) for value in unique_values.tolist())
                    if len(stats["unique_values"]) > 4096:
                        stats["unique_overflow"] = True
                        stats["unique_values"].clear()
        return stats

    def _compute_and_cache_channel_range(self, channel: int, dataset_paths: Sequence[str]) -> Optional[tuple[float, float]]:
        min_val = None
        max_val = None
        now = datetime.now().isoformat()
        for dataset_path in dataset_paths:
            root = Path(dataset_path).expanduser().resolve()
            sample_stats = self._scan_channel_stats(
                root,
                channel,
                max_files=400,
                max_values_per_array=1000,
                collect_unique=True,
                desc_prefix="sample",
            )
            unique_values = sorted(sample_stats["unique_values"])
            sample_unique_count = len(unique_values)
            is_sampled_discrete = (
                sample_stats["values_seen"] > 0
                and sample_stats["is_integer_like"]
                and not sample_stats["unique_overflow"]
                and sample_unique_count <= 2048
            )
            scan_stats = sample_stats
            normalization_source = (
                "WSGModalityDataset lazy sampled discrete unique scan"
                if is_sampled_discrete
                else "WSGModalityDataset lazy sampled continuous range scan"
            )

            key = str(root)
            entry = self.cache.setdefault(key, {
                "dataset_name": root.name,
                "dataset_path": str(root),
                "channels": {},
            })
            entry.setdefault("channels", {})
            entry["last_updated"] = now
            channel_entry = entry["channels"].setdefault(str(channel), {})
            local_min = scan_stats["local_min"]
            local_max = scan_stats["local_max"]
            if local_min is None or local_max is None:
                channel_entry["valid_ratio"] = 0.0
                channel_entry["has_data"] = False
                continue
            channel_entry["valid_ratio"] = 1.0
            channel_entry["has_data"] = True
            channel_entry["normalization"] = {
                "min_val": local_min,
                "max_val": local_max,
                "percentile_low": 0.0,
                "percentile_high": 100.0,
                "computed_at": now,
                "source": normalization_source,
                "sampled_values": int(scan_stats["values_seen"]),
                "is_integer_like": bool(scan_stats["is_integer_like"]),
                "unique_count": len(unique_values) if unique_values else None,
                "unique_values": unique_values if unique_values and len(unique_values) <= 2048 else None,
                "unique_full_scan": False,
                "sampled_arrays": 400,
                "sampled_values_per_array": 1000,
                "discrete_unique_protected": bool(is_sampled_discrete),
            }
            min_val = local_min if min_val is None else min(min_val, local_min)
            max_val = local_max if max_val is None else max(max_val, local_max)

        self._save_cache()
        if min_val is None or max_val is None:
            return None
        return min_val, max_val

    def normalize_channel(self, values: np.ndarray, channel: int, dataset_paths: Sequence[str]) -> np.ndarray:
        if channel in (0, 1, 2):
            valid = values != MASK_VALUE
            out = values.astype(np.float32, copy=True)
            out[valid] = np.clip(out[valid] / 255.0, 0.0, 1.0)
            return out

        channel_range = self.get_channel_range(channel, dataset_paths)
        if channel_range is None:
            # Keep unknown-range channels stable; callers can later populate wsg_dataset_info.json.
            return values.astype(np.float32, copy=True)

        min_val, max_val = channel_range
        valid = values != MASK_VALUE
        out = values.astype(np.float32, copy=True)
        shifted = np.clip(out[valid] - min_val, 0.0, None)
        value_range = max(max_val - min_val, 1e-6)
        if self.normalization_mode == "log":
            normalized = np.log1p(shifted) / max(np.log1p(value_range), 1e-6)
        else:
            normalized = shifted / value_range
        normalized = np.clip(normalized, 0.0, 1.0)
        out[valid] = normalized
        return out

    def denormalize_channel(self, values: np.ndarray, channel: int, dataset_paths: Sequence[str]) -> np.ndarray:
        valid = values != MASK_VALUE
        out = values.astype(np.float32, copy=True)
        if channel in (0, 1, 2):
            out[valid] = np.clip(out[valid], 0.0, 1.0) * 255.0
            return out

        channel_range = self.get_channel_range(channel, dataset_paths)
        if channel_range is None:
            return out

        min_val, max_val = channel_range
        value_range = max(max_val - min_val, 1e-6)
        clipped = np.clip(out[valid], 0.0, 1.0)
        if self.normalization_mode == "log":
            out[valid] = np.expm1(clipped * max(np.log1p(value_range), 1e-6)) + min_val
        else:
            out[valid] = clipped * value_range + min_val
        return out


class WSGModalityDataset(Dataset):
    def __init__(
        self,
        dataset_paths: Sequence[str],
        modality_config_path: str,
        patch_hw: int = 480,
        num_frames: int = 1,
        normalization_cache_path: Optional[str] = None,
        modalities: Optional[Sequence[str]] = None,
        include_conditions: bool = True,
        data_type: str = "wsv",
        normalization_mode: str = "linear",
        normalization_show_progress: bool = True,
        video_window_stride: int = 1,
        crop_mode: str = "random",
        augment: bool = False,
        output_dtype: str = "float32",
        split: Optional[str] = None,
        split_ratios: Sequence[float] = (0.9, 0.05, 0.05),
        split_seed: int = 42,
        split_dataset_paths: Optional[Sequence[str]] = None,
        normalization_dataset_paths: Optional[Sequence[str]] = None,
        caption_file_name: str = "video_captions.csv",
        sparse_codec_mode: str = "disabled",
        sparse_heatmap_sigma: float = 3.0,
        sparse_heatmap_radius: int = 8,
        sparse_heatmap_peak_threshold: float = 0.10,
        sparse_heatmap_nms_radius: int = 6,
    ):
        self.dataset_paths = [str(_resolve_project_path(path)) for path in dataset_paths]
        self.normalization_dataset_paths = [
            str(_resolve_project_path(path))
            for path in (normalization_dataset_paths or dataset_paths)
        ]
        self.split = None if split is None else str(split).lower()
        if self.split is not None and self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train, val, test, or None")
        if len(split_ratios) != 3:
            raise ValueError("split_ratios must contain exactly three values: train, val, test")
        ratio_sum = float(sum(split_ratios))
        if ratio_sum <= 0:
            raise ValueError("split_ratios must sum to a positive value")
        self.split_ratios = tuple(float(value) / ratio_sum for value in split_ratios)
        self.split_seed = int(split_seed)
        self.split_dataset_paths = {
            str(_resolve_project_path(path).resolve())
            for path in (split_dataset_paths or [])
        }
        self.modality_config_path = Path(modality_config_path)
        self.patch_hw = int(patch_hw)
        self.num_frames = int(num_frames)
        self.include_conditions = bool(include_conditions)
        self.data_type = str(data_type).lower()
        self.augment = bool(augment)
        self.output_dtype = str(output_dtype).lower()
        if self.output_dtype not in {"float32", "source"}:
            raise ValueError("output_dtype must be either 'float32' or 'source'")
        self.video_window_stride = max(1, int(video_window_stride))
        self.crop_mode = str(crop_mode or "random").lower()
        if self.crop_mode in {"center", "center_crop", "center_short_side"}:
            self.crop_mode = "center_short_side_square"
        if self.crop_mode not in {"center_short_side_square", "random"}:
            raise ValueError("crop_mode must be either 'center_short_side_square' or 'random'")
        self.normalization_mode = str(normalization_mode).lower()
        self.normalization_show_progress = bool(normalization_show_progress)
        self.normalization_contract = load_wsg_normalization_contract(self.modality_config_path)
        self.portable_normalization_ranges = load_wsg_normalization_ranges(self.modality_config_path)
        self.normalization = WSGNormalizationStore(
            normalization_cache_path,
            normalization_mode=self.normalization_mode,
            show_progress=self.normalization_show_progress,
            portable_channel_ranges=self.portable_normalization_ranges,
        )
        self._normalization_stores_by_mode = {self.normalization.normalization_mode: self.normalization}
        self.caption_file_name = caption_file_name
        self.sparse_codec_mode = str(sparse_codec_mode or "disabled").lower()
        self.sparse_heatmap_sigma = float(sparse_heatmap_sigma)
        self.sparse_heatmap_radius = int(sparse_heatmap_radius)
        self.sparse_heatmap_peak_threshold = float(sparse_heatmap_peak_threshold)
        self.sparse_heatmap_nms_radius = int(sparse_heatmap_nms_radius)
        self.caption_map = self._load_caption_map(self.dataset_paths)
        self.modality_graph = self._load_modality_graph(self.modality_config_path)
        self.sparse_modalities = sparse_modalities_from_config(self.modality_config_path)
        self.modality_keys = list(modalities) if modalities else list(self.modality_graph.keys())
        if self.include_conditions:
            for modality in list(self.modality_keys):
                self.modality_keys.extend((self.modality_graph.get(modality, {}).get("conditions") or {}).keys())
        self.modality_keys = list(dict.fromkeys(self.modality_keys))
        self.modality_normalization_modes = {
            modality: normalization_mode_for_modality(self.normalization_contract, modality)
            for modality in self.modality_keys
        }
        self.samples = self._discover_samples(self.dataset_paths)
        self.samples = self._apply_split(self.samples)
        if not self.samples:
            raise ValueError(f"No WSG/WSV samples found in {self.dataset_paths}")

    @staticmethod
    def _load_modality_graph(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        graph = data.get("wsg_v1")
        if not isinstance(graph, dict):
            raise ValueError(f"{path} must contain top-level key 'wsg_v1'")
        return flatten_wsg_modality_graph(graph)

    def _load_caption_map(self, dataset_paths: Sequence[str]) -> Dict[str, str]:
        captions: Dict[str, str] = {}
        for root in dataset_paths:
            root_path = Path(root)
            csv_names = [
                self.caption_file_name,
                "video_captions.csv",
                "articulation_captions.csv",
                "shapenet_captions.csv",
                "matsynth_captions.csv",
                "captions.csv",
                "metadata.csv",
            ]
            seen_paths = set()
            for csv_name in csv_names:
                if not csv_name:
                    continue
                csv_path = root_path / csv_name
                if csv_path in seen_paths or not csv_path.exists():
                    continue
                seen_paths.add(csv_path)
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key_parts = [
                            row.get("video_name"),
                            row.get("name"),
                            row.get("material_name"),
                            row.get("model_id"),
                            row.get("uuid"),
                            row.get("id"),
                        ]
                        caption = _join_caption_parts([
                            row.get("caption"),
                            row.get("prompt"),
                            row.get("text"),
                            row.get("description"),
                            row.get("parameter_caption"),
                        ])
                        if not caption:
                            caption = _caption_from_mapping(row)
                        if not caption:
                            continue
                        for key_part in key_parts:
                            for key in _caption_lookup_keys(str(key_part or "")):
                                captions.setdefault(key, caption)
        return captions

    def _video_name_from_path(self, path: Path) -> str:
        if path.parent.name.lower() in {"wsv", "wsg"}:
            return path.parent.parent.name
        return path.parent.name

    def _caption_for_video(self, video_name: str, path: Optional[Path] = None) -> str:
        keys = list(_caption_lookup_keys(video_name))
        if path is not None:
            sample_stem = _sample_file_stem(path)
            keys.extend(_caption_lookup_keys(sample_stem))
            keys.extend(_caption_lookup_keys(path.name))
            keys.extend(_caption_lookup_keys(path.parent.name))
        for key in dict.fromkeys(keys):
            caption = self.caption_map.get(key)
            if caption:
                return caption
        if path is not None:
            return self._caption_from_sidecar(path)
        return ""

    def _caption_from_sidecar(self, path: Path) -> str:
        sample_stem = _sample_file_stem(path)
        parent = path.parent
        candidates = [
            parent / f"{sample_stem}.txt",
            parent / f"{sample_stem}.json",
            parent / f"{sample_stem}_info.txt",
            parent / f"{sample_stem}_manifest.json",
            parent / f"{parent.name}.txt",
            parent / f"{parent.name}.json",
            parent / f"{parent.name}_info.txt",
            parent / f"{parent.name}_manifest.json",
            parent / "material_properties.txt",
            parent / "export_caption_audio_from_wsg" / "recognized_captions.txt",
        ]
        for candidate in dict.fromkeys(candidates):
            if not candidate.exists() or not candidate.is_file():
                continue
            caption = self._read_caption_sidecar(candidate)
            if caption:
                return caption
        return ""

    def _read_caption_sidecar(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        stripped = text.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            caption = _caption_from_mapping(parsed)
            if caption:
                return caption
        lines = []
        for line in stripped.splitlines():
            line = _clean_caption_text(line)
            if not line or set(line) <= {"-", "=", "_"}:
                continue
            lines.append(line)
            if len(lines) >= 4:
                break
        return _join_caption_parts(lines)

    def _discover_samples(self, dataset_paths: Sequence[str]) -> List[dict]:
        samples: List[dict] = []
        shadowed_wsg_dirs = set()
        for root in dataset_paths:
            root_path = Path(root)
            root_resolved = str(root_path.expanduser().resolve())
            root_wsv_count = 0
            load_wsv = self.data_type in {"wsv", "both", "auto"}
            load_wsg = self.data_type in {"wsg", "both", "auto"}
            if self.data_type == "rgb":
                load_wsg = True
            for wsv_path in sorted(list(root_path.glob("**/*.wsv")) + list(root_path.glob("**/*.wsv.npz"))) if load_wsv else []:
                try:
                    with np.load(wsv_path) as data:
                        keys = sorted(data.files)
                except Exception:
                    keys = []
                if not keys:
                    continue
                if wsv_path.parent.name.lower() == "wsv":
                    shadowed_wsg_dirs.add(str((wsv_path.parent.parent / "wsg").resolve()))
                if len(keys) >= self.num_frames:
                    starts = range(0, len(keys) - self.num_frames + 1, self.video_window_stride)
                else:
                    starts = range(1)
                for start in starts:
                    frame_keys = keys[start:start + self.num_frames]
                    if len(frame_keys) < self.num_frames:
                        frame_keys = frame_keys + [frame_keys[-1]] * (self.num_frames - len(frame_keys))
                    video_name = self._video_name_from_path(wsv_path)
                    samples.append({
                        "storage": "wsv",
                        "path": str(wsv_path),
                        "dataset_root": root_resolved,
                        "video_name": video_name,
                        "prompt": self._caption_for_video(video_name, wsv_path),
                        "frame_keys": frame_keys,
                        "sample_id": f"{wsv_path}::{frame_keys[0]}-{frame_keys[-1]}",
                    })
                    root_wsv_count += 1

            if self.data_type == "wsv" and root_wsv_count > 0:
                continue
            wsg_paths = sorted(list(root_path.glob("**/*.wsg")) + list(root_path.glob("**/*.wsg.npz"))) if (load_wsg or self.data_type == "wsv") else []
            for wsg_path in wsg_paths:
                if str(wsg_path.parent.resolve()) in shadowed_wsg_dirs:
                    continue
                video_name = self._video_name_from_path(wsg_path)
                samples.append({
                    "storage": "wsg",
                    "path": str(wsg_path),
                    "dataset_root": root_resolved,
                    "video_name": video_name,
                    "prompt": self._caption_for_video(video_name, wsg_path),
                    "sample_id": str(wsg_path),
                })
        return sorted(samples, key=lambda x: x["sample_id"])

    def _split_name_for_sample(self, sample: dict) -> str:
        split_key = sample.get("path", sample["sample_id"])
        digest = hashlib.sha1(f"{self.split_seed}:{split_key}".encode("utf-8")).hexdigest()
        value = int(digest[:16], 16) / float(16 ** 16)
        train_cut = self.split_ratios[0]
        val_cut = self.split_ratios[0] + self.split_ratios[1]
        if value < train_cut:
            return "train"
        if value < val_cut:
            return "val"
        return "test"

    def _apply_split(self, samples: List[dict]) -> List[dict]:
        if self.split is None or not self.split_dataset_paths:
            return samples
        filtered = []
        for sample in samples:
            dataset_root = str(Path(sample.get("dataset_root", "")).expanduser().resolve())
            if dataset_root not in self.split_dataset_paths:
                filtered.append(sample)
                continue
            if self._split_name_for_sample(sample) == self.split:
                filtered.append(sample)
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def _read_sample(self, sample: dict):
        with np.load(sample["path"]) as data:
            if sample["storage"] == "wsv":
                return np.stack([data[key] for key in sample["frame_keys"]], axis=0)
            if "data" in data:
                return data["data"]
            if "wsg_data" in data:
                return data["wsg_data"]
            # Self-describing "named" storage: each modality lives under its own
            # string key (e.g. "0-2", "LayerDiffuse_Transparency") instead of one
            # packed channel array. Return the whole mapping; crop/encode select
            # per modality by key. Custom-pilot modalities avoid colliding with
            # WSG's fixed numeric channel semantics this way.
            named = {key: data[key] for key in data.files}
            if named:
                return named
            raise ValueError(f"WSG file {sample['path']} missing 'data' or 'wsg_data'")

    def _cropped_sample(self, sample: dict, crop_info: Optional[dict] = None):
        raw = self._read_sample(sample)
        if isinstance(raw, dict):
            # Named storage: crop every modality array with one shared crop_info
            # so all lanes stay spatially aligned.
            if crop_info is None:
                first_array = next(iter(raw.values()))
                _cropped, crop_info = random_resize_crop_with_info(
                    first_array, self.patch_hw, crop_mode=self.crop_mode
                )
            cropped = {key: apply_resize_crop_info(arr, crop_info) for key, arr in raw.items()}
            return cropped, dict(crop_info)
        if crop_info is None:
            return random_resize_crop_with_info(raw, self.patch_hw, crop_mode=self.crop_mode)
        return apply_resize_crop_info(raw, crop_info), dict(crop_info)

    def _modality_to_video(self, wsg_data, modality_key: str) -> torch.Tensor:
        if isinstance(wsg_data, dict):
            # Named storage: the modality array is stored directly under its key
            # as an image-like HxWx3 lane in [0,255]. Fixed rgb_0_255_to_minus1_1
            # encoding (no per-channel normalization cache needed).
            frame = np.asarray(wsg_data[modality_key], dtype=np.float32)
            if frame.ndim == 2:
                frame = frame[..., None]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] != 3:
                frame = frame[..., :3]
            arr = frame / 127.5 - 1.0
            stacked = np.ascontiguousarray(arr.transpose(2, 0, 1))[:, None]  # (3,1,H,W)
            if self.num_frames > 1:
                stacked = np.repeat(stacked, self.num_frames, axis=1)
            tensor = torch.from_numpy(np.ascontiguousarray(stacked))
            if self.augment:
                if random.random() > 0.5:
                    tensor = torch.flip(tensor, dims=[3])
                if random.random() > 0.75:
                    tensor = torch.rot90(tensor, random.choice([1, 2, 3]), dims=[2, 3])
            return tensor
        if modality_key in {"111", "195"}:
            frames = list(wsg_data) if wsg_data.ndim == 4 else [wsg_data]
            dense_rgb = []
            for frame in frames:
                if modality_key == "111":
                    category_map = dense_channel111_category_map(frame, require_alpha=True)
                    colors = None
                else:
                    category_map = channel195_semantic_map(frame)
                    colors = AVIS_CATEGORY_COLORS
                if colors is None:
                    rgb = categorical_ids_to_rgb(category_map)
                else:
                    rgb = categorical_ids_to_rgb(category_map, colors)
                dense_rgb.append(rgb.astype(np.float32) / 127.5 - 1.0)
            stacked = np.stack(dense_rgb, axis=0).transpose(3, 0, 1, 2)
            if wsg_data.ndim != 4 and self.num_frames > 1:
                stacked = np.repeat(stacked, self.num_frames, axis=1)
            tensor = torch.from_numpy(stacked)
            if self.augment:
                if random.random() > 0.5:
                    tensor = torch.flip(tensor, dims=[3])
                if random.random() > 0.75:
                    tensor = torch.rot90(tensor, random.choice([1, 2, 3]), dims=[2, 3])
            return tensor

        channels = parse_channel_key(modality_key)
        normalization_mode = self.modality_normalization_modes.get(
            modality_key, self.normalization_contract["__default__"]
        )
        normalizer = self._normalization_stores_by_mode.get(normalization_mode)
        if normalizer is None:
            normalizer = WSGNormalizationStore(
                self.normalization.cache_file,
                normalization_mode=normalization_mode,
                show_progress=self.normalization_show_progress,
                portable_channel_ranges=self.portable_normalization_ranges,
            )
            self._normalization_stores_by_mode[normalization_mode] = normalizer
        planes = []
        raw_planes = []
        is_video = wsg_data.ndim == 4
        for channel in channels:
            channel_axis = 3 if is_video else 2
            if channel >= wsg_data.shape[channel_axis]:
                plane_shape = wsg_data.shape[:3] if is_video else wsg_data.shape[:2]
                plane = np.full(plane_shape, MASK_VALUE, dtype=np.float32)
            else:
                plane = wsg_data[:, :, :, channel] if is_video else wsg_data[:, :, channel]
            raw_planes.append(plane.astype(np.float32, copy=False))
            planes.append(normalizer.normalize_channel(plane, channel, self.normalization_dataset_paths))

        while len(planes) < 3:
            zero_shape = wsg_data.shape[:3] if is_video else wsg_data.shape[:2]
            planes.append(planes[-1] if planes else np.zeros(zero_shape, dtype=np.float32))
        if modality_key in self.sparse_modalities:
            codec = build_sparse_codec(
                modality_key,
                mode=self.sparse_codec_mode,
                sigma=self.sparse_heatmap_sigma,
                radius=self.sparse_heatmap_radius,
                peak_threshold=self.sparse_heatmap_peak_threshold,
                nms_radius=self.sparse_heatmap_nms_radius,
            )
            if codec is not None:
                while len(raw_planes) < 3:
                    zero_shape = wsg_data.shape[:3] if is_video else wsg_data.shape[:2]
                    raw_planes.append(raw_planes[-1] if raw_planes else np.zeros(zero_shape, dtype=np.float32))
                stacked = np.stack(raw_planes[:3], axis=0)
                if not is_video:
                    stacked = np.repeat(stacked[:, None], self.num_frames, axis=1)
                stacked = codec.to_heatmap(stacked)
            else:
                stacked = np.stack(planes[:3], axis=0)
                if not is_video:
                    stacked = np.repeat(stacked[:, None], self.num_frames, axis=1)
        else:
            stacked = np.stack(planes[:3], axis=0)
            if not is_video:
                stacked = np.repeat(stacked[:, None], self.num_frames, axis=1)

        # Wan VAE tensor inputs are expected in [-1, 1].
        valid = stacked != MASK_VALUE
        stacked = stacked.astype(np.float32, copy=False)
        stacked[valid] = stacked[valid] * 2.0 - 1.0

        if self.output_dtype == "source":
            source_dtype = _torch_dtype_from_numpy(wsg_data.dtype)
            if source_dtype is not None and source_dtype != torch.float32:
                stacked = stacked.astype(wsg_data.dtype, copy=False)

        tensor = torch.from_numpy(stacked)
        if self.augment:
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[3])
            if random.random() > 0.75:
                tensor = torch.rot90(tensor, random.choice([1, 2, 3]), dims=[2, 3])
        return tensor

    def _modality_has_valid_data(self, wsg_data, modality_key: str) -> bool:
        if isinstance(wsg_data, dict):
            arr = wsg_data.get(modality_key)
            if arr is None:
                return False
            values = np.asarray(arr)
            return bool(np.any(np.isfinite(values) & (values != MASK_VALUE)))
        channels = parse_channel_key(modality_key)
        if not channels:
            return False
        is_video = wsg_data.ndim == 4
        channel_axis = 3 if is_video else 2
        channel_count = int(wsg_data.shape[channel_axis])
        for channel in channels:
            if channel >= channel_count:
                continue
            plane = wsg_data[:, :, :, channel] if is_video else wsg_data[:, :, channel]
            values = np.asarray(plane)
            valid = np.isfinite(values) & (values != MASK_VALUE)
            if bool(np.any(valid)):
                return True
        return False

    @staticmethod
    def _sample_source(sample: dict) -> dict:
        return {
            "sample_id": sample.get("sample_id"),
            "dataset_root": sample.get("dataset_root"),
            "dataset_name": Path(str(sample.get("dataset_root", ""))).name if sample.get("dataset_root") else "",
            "path": sample.get("path"),
            "storage": sample.get("storage"),
            "frame_keys": list(sample.get("frame_keys") or []),
            "video_name": sample.get("video_name", ""),
        }

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        wsg_data, crop_info = self._cropped_sample(sample)
        modalities = {
            key: self._modality_to_video(wsg_data, key)
            for key in self.modality_keys
        }
        modality_valid = {
            key: self._modality_has_valid_data(wsg_data, key)
            for key in self.modality_keys
        }
        return {
            "modalities": modalities,
            "modality_valid": modality_valid,
            "sample_id": sample["sample_id"],
            "sample_source": self._sample_source(sample),
            "crop_info": crop_info,
            "height": self.patch_hw,
            "width": self.patch_hw,
            "num_frames": self.num_frames,
            "prompt": sample.get("prompt", ""),
        }

    def get_with_crop_info(self, index: int, crop_info: dict) -> dict:
        sample = self.samples[index]
        wsg_data, crop_info = self._cropped_sample(sample, crop_info)
        modalities = {
            key: self._modality_to_video(wsg_data, key)
            for key in self.modality_keys
        }
        modality_valid = {
            key: self._modality_has_valid_data(wsg_data, key)
            for key in self.modality_keys
        }
        return {
            "modalities": modalities,
            "modality_valid": modality_valid,
            "sample_id": sample["sample_id"],
            "sample_source": self._sample_source(sample),
            "crop_info": crop_info,
            "height": self.patch_hw,
            "width": self.patch_hw,
            "num_frames": self.num_frames,
            "prompt": sample.get("prompt", ""),
        }


class WSGLatentDataset(Dataset):
    def __init__(
        self,
        latent_cache_dir: str,
        modality_config_path: str,
        modalities: Optional[Sequence[str]] = None,
        include_conditions: bool = True,
        exclude_conditions: Optional[Sequence[str]] = None,
        sparse_codec_mode: str = "disabled",
        sparse_train_max_persons: Optional[int] = None,
        pair_dataset_policy: str = "prefer_avis",
        pair_dataset_owner_overrides: Optional[Mapping[str, Mapping[str, str]]] = None,
        modality_prompt_style: str = "semantic",
        latent_time_fraction: Optional[float] = None,
        latent_time_steps: Optional[int] = None,
        load_reference_image: bool = False,
    ):
        self.latent_cache_dir = Path(latent_cache_dir).expanduser()
        self.modality_config_path = Path(modality_config_path)
        self.include_conditions = bool(include_conditions)
        self.modality_prompt_style = str(modality_prompt_style or "semantic").lower()
        if self.modality_prompt_style not in {"numeric", "semantic"}:
            raise ValueError("modality_prompt_style must be 'numeric' or 'semantic'")
        self.pair_dataset_policy = str(pair_dataset_policy or "disabled").lower()
        if self.pair_dataset_policy not in {"disabled", "prefer_avis"}:
            raise ValueError("pair_dataset_policy must be 'disabled' or 'prefer_avis'")
        self.pair_dataset_owner_overrides = self._normalize_pair_dataset_owner_overrides(
            pair_dataset_owner_overrides
        )
        self.sparse_codec_mode = str(sparse_codec_mode or "disabled").lower()
        self.sparse_train_max_persons = (
            int(sparse_train_max_persons)
            if sparse_train_max_persons is not None and int(sparse_train_max_persons) > 0
            else None
        )
        self.latent_time_fraction = (
            float(latent_time_fraction)
            if latent_time_fraction is not None and float(latent_time_fraction) > 0
            else None
        )
        if self.latent_time_fraction is not None and self.latent_time_fraction > 1:
            raise ValueError("latent_time_fraction must be in (0, 1]")
        self.latent_time_steps = (
            int(latent_time_steps)
            if latent_time_steps is not None and int(latent_time_steps) > 0
            else None
        )
        self.load_reference_image = bool(load_reference_image)
        self.modality_graph = WSGModalityDataset._load_modality_graph(self.modality_config_path)
        self.normalization_contract = load_wsg_normalization_contract(self.modality_config_path)
        self.requested_modalities = list(modalities) if modalities else list(self.modality_graph.keys())
        self.modality_keys = list(self.requested_modalities)
        excluded_conditions = set(str(item) for item in (exclude_conditions or []))
        if self.include_conditions:
            for modality in list(self.modality_keys):
                self.modality_keys.extend(
                    condition
                    for condition in (self.modality_graph.get(modality, {}).get("conditions") or {}).keys()
                    if condition not in excluded_conditions
                )
        self.modality_keys = list(dict.fromkeys(self.modality_keys))

        self.payloads = {}
        self.latents = {}
        for modality in self.modality_keys:
            path = self.latent_cache_dir / f"{safe_modality_name(modality)}.pt"
            payload = self._load_modality_payload(modality, path)
            expected_mode = normalization_mode_for_modality(self.normalization_contract, modality)
            actual_mode = effective_cached_normalization_mode(
                modality, payload.get("normalization_mode", "linear")
            )
            if actual_mode != expected_mode:
                raise ValueError(
                    f"Latent cache normalization mismatch for modality {modality}: "
                    f"config requires {expected_mode}, cache contains {actual_mode}"
                )
            payload["_wsg_modality"] = str(modality)
            self.payloads[modality] = payload
            self.latents[modality] = payload["latents"]

        payload_sample_ids = {
            modality: list(
                payload.get(
                    "sample_ids",
                    [str(index) for index in range(self._payload_length(payload))],
                )
            )
            for modality, payload in self.payloads.items()
        }
        payload_lengths = {
            modality: self._payload_length(payload)
            for modality, payload in self.payloads.items()
        }
        for modality, sample_ids in payload_sample_ids.items():
            if len(sample_ids) != payload_lengths[modality]:
                raise ValueError(
                    f"Latent cache sample_ids length mismatch for {modality}: "
                    f"{len(sample_ids)} vs {payload_lengths[modality]}"
                )

        first_payload = self.payloads[self.modality_keys[0]]
        reference_modality = self.modality_keys[0]
        reference_sample_ids = payload_sample_ids[reference_modality]
        if all(payload_sample_ids[modality] == reference_sample_ids for modality in self.modality_keys):
            self.sample_ids = reference_sample_ids
            self.sample_index_by_modality = {
                modality: list(range(len(reference_sample_ids)))
                for modality in self.modality_keys
            }
        else:
            index_lookup = {
                modality: {sample_id: index for index, sample_id in enumerate(sample_ids)}
                for modality, sample_ids in payload_sample_ids.items()
            }
            self.sample_ids = reference_sample_ids
            self.sample_index_by_modality = {
                modality: [index_lookup[modality].get(sample_id, -1) for sample_id in reference_sample_ids]
                for modality in self.modality_keys
            }
            print(
                "[WSGLatentDataset] aligned optional condition caches to target sample_ids: "
                f"target={reference_modality} samples={len(reference_sample_ids)} from "
                + ", ".join(f"{modality}={len(payload_sample_ids[modality])}" for modality in self.modality_keys),
                flush=True,
            )

        self.pair_owner_by_modality = {}
        self.sample_dataset_families = self._reference_dataset_families(
            first_payload,
            reference_modality,
        )
        self._apply_modality_dataset_blacklists()
        self._filter_samples_with_requested_modalities()
        if (
            self.pair_dataset_policy == "prefer_avis"
            and self.include_conditions
            and len(self.requested_modalities) == 1
            and len(self.modality_keys) > 1
        ):
            self._apply_pair_dataset_ownership(reference_modality)

        self.length = len(self.sample_ids)
        crop_info_presence = {
            modality: ("crop_infos" in payload)
            for modality, payload in self.payloads.items()
        }
        self.crop_infos = None
        if first_payload.get("crop_infos") is not None:
            self.crop_infos = [
                first_payload.get("crop_infos")[index]
                for index in self.sample_index_by_modality[reference_modality]
            ]
        if self.crop_infos is not None and len(self.crop_infos) != self.length:
            raise ValueError(
                f"Latent cache crop_infos length mismatch: {len(self.crop_infos)} vs {self.length}"
            )
        self.sample_sources = None
        if first_payload.get("sample_sources") is not None:
            self.sample_sources = [
                first_payload.get("sample_sources")[index]
                for index in self.sample_index_by_modality[reference_modality]
            ]
        self.height = int(first_payload.get("height", first_payload.get("patch_hw", 480)))
        self.width = int(first_payload.get("width", first_payload.get("patch_hw", 480)))
        self.num_frames = int(first_payload.get("num_frames", 1))
        self.original_num_frames = self.num_frames
        self.original_latent_time_steps = self._payload_latent_time_steps(first_payload)
        self.active_latent_time_steps = self._resolve_active_latent_time_steps(self.original_latent_time_steps)
        if self.active_latent_time_steps != self.original_latent_time_steps:
            # Wan2.1 VAE maps 21 video frames to 6 latent time steps:
            # latent_T = (num_frames - 1) // 4 + 1, so decoding T latent
            # steps gives 4 * (T - 1) + 1 frames.  Cropping the same front
            # fraction for every modality keeps all modality lanes aligned.
            self.num_frames = 4 * (self.active_latent_time_steps - 1) + 1
            print(
                "[WSGLatentDataset] using front latent time window: "
                f"{self.active_latent_time_steps}/{self.original_latent_time_steps} "
                f"latent steps, num_frames {self.original_num_frames}->{self.num_frames}",
                flush=True,
            )
        self.prompt = first_payload.get("prompt", "")
        self.prompts = [self.prompt for _ in range(self.length)]
        self.text_embeddings = None
        self.text_embedding_shards = None
        self.prompt_indices = None
        self.prompt_shard_indices = None
        self.modality_text_embeddings = None
        self.modality_text_embeddings_by_role = None
        self._load_text_cache()

    @staticmethod
    def _dataset_family(sample_source: Optional[dict], sample_id: str) -> str:
        dataset_name = ""
        if isinstance(sample_source, dict):
            dataset_name = str(
                sample_source.get("dataset_name")
                or sample_source.get("dataset")
                or ""
            )
            if not dataset_name:
                dataset_name = Path(str(sample_source.get("dataset_root") or "")).name
        if not dataset_name:
            normalized = str(sample_id).replace("\\", "/")
            for known in (
                "AVIS_train_results",
                "AVIS_val_results",
                "AVIS_test_results",
                "Articulation_results",
                "ShapeNetSem_results",
                "MatSynth_results",
                "MMFi_Dataset_results",
                "XPose_results",
                "human_signer_480",
            ):
                if f"/{known}/" in normalized:
                    dataset_name = known
                    break
        if dataset_name.startswith("AVIS_"):
            return "AVIS"
        if dataset_name in {"XPose", "XPose_results"}:
            return "XPose"
        return dataset_name or "unknown"

    def _reference_dataset_families(self, payload: dict, reference_modality: str) -> list[str]:
        sources = payload.get("sample_sources")
        families = []
        for local_index, sample_id in zip(
            self.sample_index_by_modality[reference_modality],
            self.sample_ids,
        ):
            source = None
            if sources is not None and 0 <= int(local_index) < len(sources):
                source = sources[int(local_index)]
            families.append(self._dataset_family(source, sample_id))
        return families

    def _apply_modality_dataset_blacklists(self) -> None:
        masked_counts = {}
        for modality, blacklisted_families in MODALITY_DATASET_FAMILY_BLACKLISTS.items():
            if modality not in self.sample_index_by_modality:
                continue
            blacklisted = set(blacklisted_families)
            indices = self.sample_index_by_modality[modality]
            masked = 0
            for sample_index, family in enumerate(self.sample_dataset_families):
                if int(indices[sample_index]) >= 0 and family in blacklisted:
                    indices[sample_index] = -1
                    masked += 1
            if masked:
                masked_counts[modality] = masked
        if masked_counts:
            summary = ", ".join(
                f"{modality}:{count}"
                for modality, count in sorted(masked_counts.items())
            )
            print(
                "[WSGLatentDataset] modality dataset blacklists masked sample entries: "
                f"{summary}",
                flush=True,
            )

    def _filter_samples_with_requested_modalities(self) -> None:
        keep_indices = [
            sample_index
            for sample_index in range(len(self.sample_ids))
            if all(
                int(self.sample_index_by_modality[modality][sample_index]) >= 0
                for modality in self.requested_modalities
            )
        ]
        original_length = len(self.sample_ids)
        if len(keep_indices) == original_length:
            return
        if not keep_indices:
            raise ValueError(
                "Modality dataset blacklists removed all samples for requested modalities "
                f"{self.requested_modalities}"
            )
        self.sample_ids = [self.sample_ids[index] for index in keep_indices]
        self.sample_dataset_families = [
            self.sample_dataset_families[index]
            for index in keep_indices
        ]
        self.sample_index_by_modality = {
            modality: [indices[index] for index in keep_indices]
            for modality, indices in self.sample_index_by_modality.items()
        }
        print(
            "[WSGLatentDataset] filtered samples after modality dataset blacklists: "
            f"kept_samples={len(self.sample_ids)}/{original_length}; "
            f"requested={self.requested_modalities}",
            flush=True,
        )

    def _apply_pair_dataset_ownership(self, reference_modality: str) -> None:
        owner_counts = {}
        reference_overrides = self.pair_dataset_owner_overrides.get(reference_modality, {})
        for modality in self.modality_keys:
            if modality == reference_modality:
                continue
            counts = {}
            for sample_index, local_index in enumerate(self.sample_index_by_modality[modality]):
                if int(local_index) < 0:
                    continue
                family = self.sample_dataset_families[sample_index]
                counts[family] = counts.get(family, 0) + 1
            if not counts:
                continue
            override_owner = reference_overrides.get(modality)
            if override_owner:
                if override_owner not in counts:
                    raise ValueError(
                        f"pair_dataset_owner_overrides requested {reference_modality}->{modality} "
                        f"owner {override_owner!r}, but available families are {sorted(counts)}"
                    )
                owner = override_owner
            else:
                owner = "AVIS" if counts.get("AVIS", 0) > 0 else max(
                    counts,
                    key=lambda family: (counts[family], family),
                )
            self.pair_owner_by_modality[modality] = owner
            owner_counts[modality] = counts

        keep_indices = []
        for sample_index, family in enumerate(self.sample_dataset_families):
            has_owned_condition = any(
                int(self.sample_index_by_modality[modality][sample_index]) >= 0
                and self.pair_owner_by_modality.get(modality) == family
                for modality in self.modality_keys
                if modality != reference_modality
            )
            if has_owned_condition:
                keep_indices.append(sample_index)

        original_length = len(self.sample_ids)
        if len(keep_indices) != original_length:
            self.sample_ids = [self.sample_ids[index] for index in keep_indices]
            self.sample_dataset_families = [
                self.sample_dataset_families[index]
                for index in keep_indices
            ]
            self.sample_index_by_modality = {
                modality: [indices[index] for index in keep_indices]
                for modality, indices in self.sample_index_by_modality.items()
            }

        summary = ", ".join(
            f"{reference_modality}->{modality}:{owner}"
            for modality, owner in self.pair_owner_by_modality.items()
        )
        print(
            "[WSGLatentDataset] pair dataset ownership enabled: "
            f"kept_samples={len(self.sample_ids)}/{original_length}; {summary}",
            flush=True,
        )

    @staticmethod
    def _normalize_pair_dataset_owner_overrides(
        overrides: Optional[Mapping[str, Mapping[str, str]]],
    ) -> dict[str, dict[str, str]]:
        if not overrides:
            return {}
        if not isinstance(overrides, Mapping):
            raise TypeError("pair_dataset_owner_overrides must be a mapping of target -> condition -> family")
        normalized: dict[str, dict[str, str]] = {}
        for target, condition_map in overrides.items():
            if not isinstance(condition_map, Mapping):
                raise TypeError(f"pair_dataset_owner_overrides[{target!r}] must be a mapping")
            target_key = str(target)
            normalized[target_key] = {
                str(condition): str(owner)
                for condition, owner in condition_map.items()
                if str(owner)
            }
        return normalized

    def _load_text_cache(self) -> None:
        prompts_path = self.latent_cache_dir / "prompts.pt"
        if prompts_path.exists():
            payload = torch.load(prompts_path, map_location="cpu", weights_only=False)
            prompts = payload.get("prompts")
            if prompts is not None:
                prompt_sample_ids = payload.get("sample_ids")
                if prompt_sample_ids is not None and list(prompt_sample_ids) != self.sample_ids:
                    prompt_by_sample_id = {
                        sample_id: prompts[index]
                        for index, sample_id in enumerate(prompt_sample_ids)
                    }
                    missing = [sample_id for sample_id in self.sample_ids if sample_id not in prompt_by_sample_id]
                    if missing:
                        raise ValueError(f"{prompts_path} is missing prompts for sample_ids: {missing[:10]}")
                    self.prompts = [prompt_by_sample_id[sample_id] for sample_id in self.sample_ids]
                else:
                    if len(prompts) != self.length:
                        raise ValueError(f"{prompts_path} contains {len(prompts)} prompts but latent cache has {self.length} samples")
                    self.prompts = list(prompts)

        embeddings_path = self.latent_cache_dir / "text_embeddings.pt"
        embedding_shard_paths = sorted((self.latent_cache_dir / "text_embeddings").glob("shard_*_of_*.pt"))
        if embeddings_path.exists():
            payload = self._torch_load_payload(embeddings_path)
            embeddings = payload.get("text_embeddings")
            prompt_indices = payload.get("prompt_indices")
            if embeddings is None or prompt_indices is None:
                raise ValueError(f"{embeddings_path} must contain text_embeddings and prompt_indices")
            prompt_sample_ids = payload.get("sample_ids")
            if prompt_sample_ids is not None and list(prompt_sample_ids) != self.sample_ids:
                prompt_index_by_sample_id = {
                    sample_id: int(prompt_indices[index])
                    for index, sample_id in enumerate(prompt_sample_ids)
                }
                missing = [sample_id for sample_id in self.sample_ids if sample_id not in prompt_index_by_sample_id]
                if missing:
                    raise ValueError(f"{embeddings_path} is missing prompt embeddings for sample_ids: {missing[:10]}")
                prompt_indices = torch.tensor(
                    [prompt_index_by_sample_id[sample_id] for sample_id in self.sample_ids],
                    dtype=torch.long,
                )
            elif len(prompt_indices) != self.length:
                raise ValueError(f"{embeddings_path} contains {len(prompt_indices)} prompt indices but latent cache has {self.length} samples")
            self.text_embeddings = embeddings
            self.prompt_indices = torch.as_tensor(prompt_indices, dtype=torch.long)
        elif embedding_shard_paths:
            shard_payloads = [
                torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                for path in embedding_shard_paths
            ]
            self.text_embedding_shards = [payload.get("text_embeddings") for payload in shard_payloads]
            if any(embeddings is None for embeddings in self.text_embedding_shards):
                raise ValueError(f"Text embedding shards under {embedding_shard_paths[0].parent} must contain text_embeddings")
            location_by_sample_id = {}
            for shard_index, payload in enumerate(shard_payloads):
                shard_sample_ids = list(payload.get("sample_ids") or [])
                shard_prompt_indices = torch.as_tensor(payload.get("prompt_indices"), dtype=torch.long)
                if len(shard_sample_ids) != len(shard_prompt_indices):
                    raise ValueError(f"Text embedding shard sample length mismatch: {len(shard_sample_ids)} vs {len(shard_prompt_indices)}")
                for sample_id, prompt_index in zip(shard_sample_ids, shard_prompt_indices.tolist()):
                    location_by_sample_id[sample_id] = (shard_index, int(prompt_index))
            missing = [sample_id for sample_id in self.sample_ids if sample_id not in location_by_sample_id]
            if missing:
                raise ValueError(f"Text embedding shards are missing prompt embeddings for sample_ids: {missing[:10]}")
            self.prompt_shard_indices = torch.tensor(
                [location_by_sample_id[sample_id] for sample_id in self.sample_ids],
                dtype=torch.long,
            )

        modality_embeddings_path = self.latent_cache_dir / f"modality_prompt_embeddings_{self.modality_prompt_style}.pt"
        if modality_embeddings_path.exists():
            payload = torch.load(modality_embeddings_path, map_location="cpu", weights_only=False)
            modality_text_embeddings = payload.get("modality_text_embeddings")
            if not isinstance(modality_text_embeddings, dict):
                raise ValueError(f"{modality_embeddings_path} must contain modality_text_embeddings")
            missing = [modality for modality in self.modality_keys if modality not in modality_text_embeddings]
            if missing:
                raise ValueError(f"{modality_embeddings_path} is missing modality prompt embeddings for {missing}")
            self.modality_text_embeddings = {
                modality: modality_text_embeddings[modality]
                for modality in self.modality_keys
            }
            role_embeddings = payload.get("modality_text_embeddings_by_role")
            if isinstance(role_embeddings, dict):
                missing_by_role = {
                    role: [modality for modality in self.modality_keys if modality not in (role_embeddings.get(role) or {})]
                    for role in ("condition", "target")
                }
                missing_by_role = {role: missing for role, missing in missing_by_role.items() if missing}
                if missing_by_role:
                    raise ValueError(
                        f"{modality_embeddings_path} is missing role-aware modality prompt embeddings: {missing_by_role}"
                    )
                self.modality_text_embeddings_by_role = {
                    role: {
                        modality: role_embeddings[role][modality]
                        for modality in self.modality_keys
                    }
                    for role in ("condition", "target")
                }

    @staticmethod
    def _torch_load_payload(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _is_compact_sparse_payload(payload: dict) -> bool:
        return payload.get("sparse_cache_format") == "compact_lanes_v1"

    def _load_modality_payload(self, modality: str, path: Path) -> dict:
        if path.exists():
            payload = self._torch_load_payload(path)
            if "latents" not in payload:
                raise ValueError(f"{path} must contain key 'latents'")
            mode_map = payload.get("normalization_modes")
            if isinstance(mode_map, dict) and modality in mode_map:
                payload["normalization_mode"] = mode_map[modality]
            payload.setdefault("normalization_mode", "linear")
            source_mode = str(payload["normalization_mode"]).lower()
            payload["source_normalization_mode"] = source_mode
            payload["normalization_mode"] = _canonical_normalization_mode(source_mode)
            return payload

        shard_dir = self.latent_cache_dir / "shards" / safe_modality_name(modality)
        shard_paths = sorted(shard_dir.glob("shard_*_of_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"Missing latent cache for modality {modality}: {path} or shards in {shard_dir}")

        shards = [self._torch_load_payload(shard_path) for shard_path in shard_paths]
        source_normalization_modes = set()
        for shard in shards:
            mode_map = shard.get("normalization_modes")
            mode = mode_map.get(modality) if isinstance(mode_map, dict) else None
            source_normalization_modes.add(str(mode or shard.get("normalization_mode", "linear")).lower())
        normalization_modes = {
            _canonical_normalization_mode(mode)
            for mode in source_normalization_modes
        }
        if not normalization_modes <= {"linear", "log"}:
            raise ValueError(
                f"Shard cache for modality {modality} has invalid normalization modes: "
                f"{sorted(source_normalization_modes)}"
            )
        if len(normalization_modes) != 1:
            raise ValueError(
                f"Shard cache for modality {modality} mixes normalization modes: "
                f"{sorted(normalization_modes)}"
            )
        for shard in shards:
            mode_map = shard.get("normalization_modes")
            source_mode = str(
                (mode_map.get(modality) if isinstance(mode_map, dict) else None)
                or shard.get("normalization_mode", "linear")
            ).lower()
            shard["source_normalization_mode"] = source_mode
            shard["normalization_mode"] = _canonical_normalization_mode(source_mode)
        global_num_samples = int(shards[0].get("global_num_samples", 0))
        if global_num_samples <= 0:
            raise ValueError(f"Shard cache for modality {modality} is missing global_num_samples")

        compact_flags = [self._is_compact_sparse_payload(shard) for shard in shards]
        filtered_flags = [bool(shard.get("filtered_sample_ids")) for shard in shards]
        if any(filtered_flags):
            if any(compact_flags):
                if not all(compact_flags):
                    raise ValueError(f"Filtered shard cache for modality {modality} mixes compact and dense sparse formats")
                return self._combine_filtered_compact_sparse_shards(modality, shards, shard_paths)
            return self._combine_filtered_shards(modality, shards)

        if any(compact_flags):
            if not all(compact_flags):
                raise ValueError(f"Shard cache for modality {modality} mixes compact and dense sparse formats")
            return self._combine_compact_sparse_shards(modality, shards, shard_paths, global_num_samples)

        first_latents = shards[0]["latents"]
        latents = torch.empty((global_num_samples, *first_latents.shape[1:]), dtype=first_latents.dtype)
        sample_ids = [None] * global_num_samples
        filled = torch.zeros(global_num_samples, dtype=torch.bool)
        for shard in shards:
            shard_latents = shard["latents"]
            shard_indices = shard.get("shard_indices")
            if shard_indices is None:
                raise ValueError(f"Shard cache for modality {modality} is missing shard_indices")
            if len(shard_indices) != shard_latents.shape[0]:
                raise ValueError(f"Shard indices/latents length mismatch for modality {modality}")
            latents[shard_indices] = shard_latents
            for index, sample_id in zip(shard_indices, shard.get("sample_ids", [])):
                sample_ids[index] = sample_id
            filled[shard_indices] = True
        if not bool(filled.all()):
            missing = torch.nonzero(~filled, as_tuple=False).flatten()[:10].tolist()
            raise ValueError(f"Shard cache for modality {modality} is incomplete; first missing indices: {missing}")

        payload = dict(shards[0])
        payload["num_samples"] = global_num_samples
        payload["sample_ids"] = [sample_id if sample_id is not None else str(index) for index, sample_id in enumerate(sample_ids)]
        if any("crop_infos" in shard for shard in shards):
            crop_infos = [None] * global_num_samples
            for shard in shards:
                shard_indices = shard.get("shard_indices")
                shard_crop_infos = shard.get("crop_infos")
                if shard_crop_infos is None:
                    raise ValueError(f"Some shards for modality {modality} have crop_infos but another shard is missing them")
                if len(shard_crop_infos) != len(shard_indices):
                    raise ValueError(f"Shard crop_infos length mismatch for modality {modality}")
                for index, crop_info in zip(shard_indices, shard_crop_infos):
                    crop_infos[index] = crop_info
            payload["crop_infos"] = crop_infos
        payload["latents"] = latents
        return payload

    def _combine_filtered_shards(self, modality: str, shards: list[dict]) -> dict:
        sample_ids = []
        sample_sources = []
        crop_infos = []
        shard_offsets = [0]
        has_crop_infos = any("crop_infos" in shard for shard in shards)
        has_sample_sources = any("sample_sources" in shard for shard in shards)
        for shard in shards:
            latents = shard["latents"]
            shard_sample_ids = list(shard.get("sample_ids") or [])
            if len(shard_sample_ids) != int(latents.shape[0]):
                raise ValueError(f"Filtered shard cache for {modality} has sample_ids/latents length mismatch")
            sample_ids.extend(shard_sample_ids)
            shard_offsets.append(len(sample_ids))
            if has_sample_sources:
                sample_sources.extend(list(shard.get("sample_sources") or [{} for _ in shard_sample_ids]))
            if has_crop_infos:
                shard_crop_infos = shard.get("crop_infos")
                if shard_crop_infos is None:
                    raise ValueError(f"Some filtered shards for {modality} have crop_infos but another shard is missing them")
                crop_infos.extend(list(shard_crop_infos))
        payload = dict(shards[0])
        payload["num_samples"] = len(sample_ids)
        payload["global_num_samples"] = len(sample_ids)
        payload["sample_ids"] = sample_ids
        payload["filtered_dense_shards"] = shards
        payload["filtered_dense_shard_offsets"] = shard_offsets
        payload["latents"] = torch.empty((len(sample_ids), 0), dtype=shards[0]["latents"].dtype)
        if has_sample_sources:
            payload["sample_sources"] = sample_sources
        if has_crop_infos:
            payload["crop_infos"] = crop_infos
        return payload

    def _combine_filtered_compact_sparse_shards(
        self,
        modality: str,
        shards: list[dict],
        shard_paths: list[Path],
    ) -> dict:
        first = shards[0]
        lane_count = int(first.get("latent_lane_count", 0))
        lane_channels = int(first.get("latent_channels_per_lane", 0))
        empty_lane_latent = first.get("empty_lane_latent")
        if lane_count <= 0 or lane_channels <= 0 or empty_lane_latent is None:
            raise ValueError(f"Filtered compact sparse cache for {modality} has invalid lane metadata")
        chunks = []
        lane_indices = []
        sample_lane_offsets = [0]
        sample_ids = []
        sample_sources = []
        crop_infos = []
        has_crop_infos = any("crop_infos" in shard for shard in shards)
        has_sample_sources = any("sample_sources" in shard for shard in shards)
        for shard, shard_path in zip(shards, shard_paths):
            if int(shard.get("latent_lane_count", 0)) != lane_count:
                raise ValueError(f"{shard_path} latent_lane_count does not match other filtered compact shards")
            if int(shard.get("latent_channels_per_lane", 0)) != lane_channels:
                raise ValueError(f"{shard_path} latent_channels_per_lane does not match other filtered compact shards")
            latents = shard.get("latents")
            offsets = torch.as_tensor(shard.get("sample_lane_offsets"), dtype=torch.long)
            shard_lane_indices = torch.as_tensor(shard.get("lane_indices"), dtype=torch.long)
            shard_sample_ids = list(shard.get("sample_ids") or [])
            if len(shard_sample_ids) != offsets.numel() - 1:
                raise ValueError(f"{shard_path} filtered compact sample_ids/offsets length mismatch")
            if int(offsets[-1].item()) != int(shard_lane_indices.numel()) or int(offsets[-1].item()) != int(latents.shape[0]):
                raise ValueError(f"{shard_path} filtered compact active-lane count mismatch")
            if latents.shape[0] > 0:
                chunks.append(latents)
            start_offset = sample_lane_offsets[-1]
            for local_index, sample_id in enumerate(shard_sample_ids):
                start = int(offsets[local_index].item())
                end = int(offsets[local_index + 1].item())
                lane_indices.extend(int(item) for item in shard_lane_indices[start:end])
                sample_lane_offsets.append(start_offset + len(lane_indices) - sample_lane_offsets[-1])
                sample_ids.append(sample_id)
            if has_sample_sources:
                sample_sources.extend(list(shard.get("sample_sources") or [{} for _ in shard_sample_ids]))
            if has_crop_infos:
                shard_crop_infos = shard.get("crop_infos")
                if shard_crop_infos is None:
                    raise ValueError(f"Some filtered compact shards for {modality} have crop_infos but another shard is missing them")
                crop_infos.extend(list(shard_crop_infos))
        if chunks:
            latents = torch.cat(chunks, dim=0)
        else:
            latents = torch.empty((0, *empty_lane_latent.shape), dtype=first["latents"].dtype)
        payload = dict(first)
        payload["num_samples"] = len(sample_ids)
        payload["global_num_samples"] = len(sample_ids)
        payload["dense_num_samples"] = len(sample_ids)
        payload["sample_ids"] = sample_ids
        payload["latents"] = latents
        payload["lane_indices"] = torch.as_tensor(lane_indices, dtype=torch.long)
        payload["sample_lane_offsets"] = torch.as_tensor(sample_lane_offsets, dtype=torch.long)
        payload["compact_sparse_shard_paths"] = [str(path) for path in shard_paths]
        if has_sample_sources:
            payload["sample_sources"] = sample_sources
        if has_crop_infos:
            payload["crop_infos"] = crop_infos
        return payload

    def _combine_compact_sparse_shards(
        self,
        modality: str,
        shards: list[dict],
        shard_paths: list[Path],
        global_num_samples: int,
    ) -> dict:
        first = shards[0]
        lane_count = int(first.get("latent_lane_count", 0))
        lane_channels = int(first.get("latent_channels_per_lane", 0))
        empty_lane_latent = first.get("empty_lane_latent")
        if lane_count <= 0 or lane_channels <= 0 or empty_lane_latent is None:
            raise ValueError(f"Compact sparse shard cache for modality {modality} has invalid lane metadata")

        global_to_shard = torch.full((global_num_samples,), -1, dtype=torch.long)
        global_to_local = torch.full((global_num_samples,), -1, dtype=torch.long)
        sample_ids = [None] * global_num_samples
        crop_infos = [None] * global_num_samples
        has_crop_infos = any("crop_infos" in shard for shard in shards)

        for shard_idx, (shard, shard_path) in enumerate(zip(shards, shard_paths)):
            if int(shard.get("latent_lane_count", 0)) != lane_count:
                raise ValueError(f"{shard_path} latent_lane_count does not match other compact shards")
            if int(shard.get("latent_channels_per_lane", 0)) != lane_channels:
                raise ValueError(f"{shard_path} latent_channels_per_lane does not match other compact shards")
            shard_empty = shard.get("empty_lane_latent")
            if shard_empty is None or tuple(shard_empty.shape) != tuple(empty_lane_latent.shape):
                raise ValueError(f"{shard_path} empty_lane_latent shape does not match other compact shards")
            offsets = torch.as_tensor(shard.get("sample_lane_offsets"), dtype=torch.long)
            lane_indices = torch.as_tensor(shard.get("lane_indices"), dtype=torch.long)
            latents = shard.get("latents")
            shard_indices = shard.get("shard_indices")
            if offsets.ndim != 1 or offsets.numel() < 1:
                raise ValueError(f"{shard_path} compact sparse sample_lane_offsets must be a non-empty 1D tensor")
            if shard_indices is None:
                raise ValueError(f"Shard cache for modality {modality} is missing shard_indices")
            shard_indices = [int(index) for index in shard_indices]
            if len(shard_indices) != offsets.numel() - 1:
                raise ValueError(f"Shard indices/compact offsets length mismatch for modality {modality}: {shard_path}")
            if int(offsets[-1].item()) != int(lane_indices.numel()) or int(offsets[-1].item()) != int(latents.shape[0]):
                raise ValueError(f"{shard_path} compact sparse active-lane count mismatch")
            if has_crop_infos and "crop_infos" not in shard:
                raise ValueError(f"Some shards for modality {modality} have crop_infos but {shard_path} is missing them")
            shard_crop_infos = shard.get("crop_infos")
            if shard_crop_infos is not None and len(shard_crop_infos) != len(shard_indices):
                raise ValueError(f"Shard crop_infos length mismatch for modality {modality}: {shard_path}")

            for local_idx, global_idx in enumerate(shard_indices):
                if not 0 <= global_idx < global_num_samples:
                    raise ValueError(f"{shard_path} has invalid global sample index {global_idx}")
                if int(global_to_shard[global_idx].item()) != -1:
                    raise ValueError(f"Duplicate latent shard entry for modality {modality}, sample {global_idx}")
                global_to_shard[global_idx] = shard_idx
                global_to_local[global_idx] = local_idx
                shard_sample_ids = shard.get("sample_ids", [])
                if local_idx < len(shard_sample_ids):
                    sample_ids[global_idx] = shard_sample_ids[local_idx]
                if shard_crop_infos is not None:
                    crop_infos[global_idx] = shard_crop_infos[local_idx]

        if not bool((global_to_shard >= 0).all()):
            missing = torch.nonzero(global_to_shard < 0, as_tuple=False).flatten()[:10].tolist()
            raise ValueError(f"Shard cache for modality {modality} is incomplete; first missing indices: {missing}")

        payload = dict(first)
        payload["num_samples"] = global_num_samples
        payload["dense_num_samples"] = global_num_samples
        payload["sample_ids"] = [sample_id if sample_id is not None else str(index) for index, sample_id in enumerate(sample_ids)]
        if has_crop_infos:
            payload["crop_infos"] = crop_infos
        payload["compact_sparse_shards"] = shards
        payload["compact_sparse_shard_paths"] = [str(path) for path in shard_paths]
        payload["global_to_shard"] = global_to_shard
        payload["global_to_local"] = global_to_local
        payload["latents"] = torch.empty((global_num_samples, 0), dtype=first["latents"].dtype)
        return payload

    def _payload_length(self, payload: dict) -> int:
        if "filtered_dense_shards" in payload:
            return int(payload["num_samples"])
        if self._is_compact_sparse_payload(payload):
            if "compact_sparse_shards" in payload:
                return int(payload["num_samples"])
            offsets = payload.get("sample_lane_offsets")
            if offsets is None:
                raise ValueError("Compact sparse cache is missing sample_lane_offsets")
            return int(payload.get("dense_num_samples", len(offsets) - 1))
        return int(payload["latents"].shape[0])

    def _compact_sparse_training_lane_count(self, payload: dict, lane_count: int) -> int:
        if self.sparse_train_max_persons is None:
            return lane_count
        if str(payload.get("_wsg_modality", "")) != "64-66":
            return lane_count
        mode = str(payload.get("sparse_codec_mode", self.sparse_codec_mode)).lower()
        max_persons = max(1, int(self.sparse_train_max_persons))
        if mode in {"dwpose_body_partition_heatmap", "body_partition_heatmap"}:
            return min(lane_count, max_persons * 4)
        if mode in {"dwpose_body_only_heatmap", "body_only_heatmap"}:
            return min(lane_count, max_persons)
        if mode in {"dwpose_person_keypoint_pack3_heatmap", "person_keypoint_pack3_heatmap"}:
            return min(lane_count, (max_persons * 128 + 2) // 3)
        return lane_count

    def _compact_sparse_sample_latent(self, payload: dict, index: int) -> torch.Tensor:
        source = "<compact sparse cache>"
        local_index = int(index)
        source_modality = payload.get("_wsg_modality")
        if "compact_sparse_shards" in payload:
            shard_idx = int(payload["global_to_shard"][index].item())
            local_index = int(payload["global_to_local"][index].item())
            payload = payload["compact_sparse_shards"][shard_idx]
            if source_modality is not None:
                payload["_wsg_modality"] = source_modality
            source = str(payload.get("shard_path", f"compact sparse shard {shard_idx}"))

        latents = payload.get("latents")
        offsets = payload.get("sample_lane_offsets")
        lane_indices = payload.get("lane_indices")
        lane_count = int(payload.get("latent_lane_count", 0))
        training_lane_count = self._compact_sparse_training_lane_count(payload, lane_count)
        lane_channels = int(payload.get("latent_channels_per_lane", 0))
        empty_lane_latent = payload.get("empty_lane_latent")
        if latents is None or offsets is None or lane_indices is None:
            raise ValueError(f"{source} compact sparse cache must contain latents, lane_indices, and sample_lane_offsets")
        if lane_count <= 0 or lane_channels <= 0:
            raise ValueError(f"{source} compact sparse cache has invalid lane metadata")
        offsets = torch.as_tensor(offsets, dtype=torch.long)
        lane_indices = torch.as_tensor(lane_indices, dtype=torch.long)
        latents = torch.as_tensor(latents)
        if offsets.ndim != 1 or offsets.numel() < 1:
            raise ValueError(f"{source} compact sparse sample_lane_offsets must be a non-empty 1D tensor")
        num_samples = int(payload.get("dense_num_samples", offsets.numel() - 1))
        if offsets.numel() != num_samples + 1:
            raise ValueError(
                f"{source} compact sparse offset length mismatch: offsets={offsets.numel()} samples={num_samples}"
            )
        if int(offsets[-1].item()) != int(lane_indices.numel()) or int(offsets[-1].item()) != int(latents.shape[0]):
            raise ValueError(
                f"{source} compact sparse active-lane count mismatch: "
                f"offsets={int(offsets[-1].item())} lane_indices={lane_indices.numel()} latents={latents.shape[0]}"
            )
        if empty_lane_latent is None:
            empty_shape = tuple(latents.shape[1:])
            if not empty_shape:
                raise ValueError(f"{source} compact sparse cache is missing empty_lane_latent and latent shape")
            empty_lane_latent = torch.zeros(empty_shape, dtype=latents.dtype)
        else:
            empty_lane_latent = torch.as_tensor(empty_lane_latent, dtype=latents.dtype)
        if int(empty_lane_latent.shape[0]) != lane_channels:
            raise ValueError(
                f"{source} empty_lane_latent channel mismatch: "
                f"{empty_lane_latent.shape[0]} vs latent_channels_per_lane={lane_channels}"
            )
        if not 0 <= local_index < num_samples:
            raise IndexError(f"{source} compact sparse sample index {local_index} is outside [0, {num_samples})")

        dense = empty_lane_latent.unsqueeze(0).expand(training_lane_count, *empty_lane_latent.shape).clone()
        start = int(offsets[local_index].item())
        end = int(offsets[local_index + 1].item())
        if end > start:
            active_lanes = lane_indices[start:end]
            if bool(((active_lanes < 0) | (active_lanes >= lane_count)).any()):
                bad_lane = int(active_lanes[((active_lanes < 0) | (active_lanes >= lane_count)).nonzero()[0]].item())
                raise ValueError(f"{source} compact sparse lane index {bad_lane} is outside [0, {lane_count})")
            keep = active_lanes < training_lane_count
            if bool(keep.any()):
                dense[active_lanes[keep]] = latents[start:end][keep]
        return dense.reshape(training_lane_count * lane_channels, *empty_lane_latent.shape[1:])

    def _payload_latent_time_steps(self, payload: dict) -> int:
        if "filtered_dense_shards" in payload:
            shards = payload.get("filtered_dense_shards") or []
            if not shards:
                return 1
            latents = shards[0].get("latents")
        elif self._is_compact_sparse_payload(payload):
            if "compact_sparse_shards" in payload:
                shards = payload.get("compact_sparse_shards") or []
                payload = shards[0] if shards else payload
            latents = payload.get("empty_lane_latent")
            if latents is None:
                latents = payload.get("latents")
        else:
            latents = payload.get("latents")
        if latents is None:
            return 1
        shape = tuple(torch.as_tensor(latents).shape)
        if len(shape) < 3:
            return 1
        return max(1, int(shape[-3]))

    def _resolve_active_latent_time_steps(self, total_steps: int) -> int:
        total_steps = max(1, int(total_steps))
        if self.latent_time_steps is not None:
            return min(total_steps, max(1, int(self.latent_time_steps)))
        if self.latent_time_fraction is not None:
            return min(total_steps, max(1, int(math.ceil(total_steps * self.latent_time_fraction))))
        return total_steps

    def _crop_latent_time_window(self, latent: torch.Tensor) -> torch.Tensor:
        if self.active_latent_time_steps >= self.original_latent_time_steps:
            return latent
        if latent.ndim < 3:
            return latent
        time_dim = latent.ndim - 3
        if int(latent.shape[time_dim]) <= self.active_latent_time_steps:
            return latent
        return latent.narrow(time_dim, 0, self.active_latent_time_steps)

    def _modality_latent_at(self, key: str, index: int) -> torch.Tensor:
        payload = self.payloads[key]
        local_index = int(self.sample_index_by_modality[key][index])
        if local_index < 0:
            raise KeyError(f"Modality {key} is unavailable for sample_id={self.sample_ids[index]}")
        if "filtered_dense_shards" in payload:
            offsets = payload["filtered_dense_shard_offsets"]
            shard_index = bisect.bisect_right(offsets, local_index) - 1
            if shard_index < 0 or shard_index >= len(payload["filtered_dense_shards"]):
                raise IndexError(f"Filtered dense sample index {local_index} is out of range for modality {key}")
            shard_local_index = local_index - offsets[shard_index]
            return self._crop_latent_time_window(
                payload["filtered_dense_shards"][shard_index]["latents"][shard_local_index]
            )
        if self._is_compact_sparse_payload(payload):
            return self._crop_latent_time_window(self._compact_sparse_sample_latent(payload, local_index))
        return self._crop_latent_time_window(self.latents[key][local_index])

    def _reference_image_at(self, index: int) -> torch.Tensor:
        if self.sample_sources is None or self.crop_infos is None:
            raise ValueError(
                "Reference image loading requires sample_sources and crop_infos in the latent cache"
            )
        source = self.sample_sources[index]
        crop_info = self.crop_infos[index]
        if not isinstance(source, dict):
            raise TypeError(f"Invalid sample_source for reference image: {type(source)}")
        path = Path(str(source.get("path") or "")).expanduser()
        frame_keys = list(source.get("frame_keys") or [])
        if not path.is_file() or not frame_keys:
            raise ValueError(
                f"Reference image source must be a WSV window with frame keys: {source}"
            )
        with np.load(path) as payload:
            frame = np.asarray(payload[frame_keys[0]])
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError(f"Invalid reference WSG frame shape {frame.shape} from {path}")
        rgb = apply_resize_crop_info(frame[..., :3], crop_info)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.float32, copy=False)
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        sample_family = self.sample_dataset_families[index]
        available_modalities = [
            key
            for key in self.modality_keys
            if int(self.sample_index_by_modality[key][index]) >= 0
            and (
                key == self.modality_keys[0]
                or key not in self.pair_owner_by_modality
                or self.pair_owner_by_modality[key] == sample_family
            )
        ]
        item = {
            "modality_latents": {
                key: self._modality_latent_at(key, index)
                for key in available_modalities
            },
            "sample_id": self.sample_ids[index],
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "prompt": self.prompts[index],
            "dataset_family": sample_family,
        }
        if self.crop_infos is not None:
            item["crop_info"] = self.crop_infos[index]
        if self.sample_sources is not None:
            item["sample_source"] = self.sample_sources[index]
        if self.load_reference_image:
            item["reference_image"] = self._reference_image_at(index)
        if self.text_embeddings is not None and self.prompt_indices is not None:
            item["text_embeddings"] = self.text_embeddings[int(self.prompt_indices[index])]
        elif self.text_embedding_shards is not None and self.prompt_shard_indices is not None:
            shard_index, prompt_index = self.prompt_shard_indices[index].tolist()
            item["text_embeddings"] = self.text_embedding_shards[shard_index][prompt_index]
        if self.modality_text_embeddings is not None:
            item["modality_text_embeddings"] = {
                key: self.modality_text_embeddings[key]
                for key in available_modalities
            }
        if self.modality_text_embeddings_by_role is not None:
            item["modality_text_embeddings_by_role"] = {
                role: {
                    key: self.modality_text_embeddings_by_role[role][key]
                    for key in available_modalities
                }
                for role in ("condition", "target")
            }
        return item
