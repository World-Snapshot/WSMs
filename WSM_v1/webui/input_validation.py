"""Validation and canonical loading for ControlWorld WSM inputs.

The browser accepts visual previews for Paper Demo, but model inputs must obey
the selected task's data contract.  This module deliberately does not use
pickle and keeps validation independent from the GPU inference stack.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import wave
from typing import Any, Mapping

import numpy as np
from PIL import Image


MASK_VALUE = -1.0
STANDARD_WSG_CHANNELS = 256
CANONICAL_DATA_KEYS = (
    "wsg_data",
    "data",
    "arr_0",
    "modality_data",
    "values",
    "array",
    "tensor",
)
READER_DATA_KEYS = (
    "channel_map",
    "depth_data",
    "normal_data",
    "curvature_data",
    "flow_data",
    "edge_data",
    "segmentation_mask",
    "xpose_pose_rgb",
)
METADATA_KEYS = {
    "metadata",
    "metadata_json",
    "__metadata__",
    "modality",
    "modality_id",
    "modality_name",
    "token",
    "channel_key",
    "channels",
    "channel_ids",
    "channel_indices",
    "layout",
    "value_space",
    "normalized",
    "normalization",
    "value_range",
}
TOKEN_PATTERN = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")


class InputValidationError(ValueError):
    """Raised when an uploaded input cannot satisfy the selected task."""


def parse_numeric_token(token: str) -> list[int] | None:
    token = str(token).strip()
    if not TOKEN_PATTERN.fullmatch(token):
        return None
    channels: list[int] = []
    for part in token.split(","):
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise InputValidationError(f"Invalid descending channel range: {token}")
            channels.extend(range(start, end + 1))
        else:
            channels.append(int(part))
    if len(channels) != len(set(channels)):
        raise InputValidationError(f"Input modality repeats a channel: {token}")
    return channels


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise InputValidationError("Object metadata is not accepted; use JSON or plain arrays")
        if value.ndim == 0:
            return value.item()
        if value.size <= 1024:
            return value.tolist()
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_metadata(archive: Any, keys: list[str], warnings: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("metadata_json", "metadata", "__metadata__"):
        if key not in keys:
            continue
        try:
            value = _json_scalar(archive[key])
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str):
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    metadata.update(parsed)
            elif isinstance(value, dict):
                metadata.update(value)
        except (InputValidationError, UnicodeDecodeError, ValueError, TypeError) as exc:
            warnings.append(f"Ignored unreadable {key}: {exc}")
    for key in METADATA_KEYS - {"metadata_json", "metadata", "__metadata__"}:
        if key not in keys:
            continue
        try:
            value = _json_scalar(archive[key])
        except (InputValidationError, ValueError) as exc:
            warnings.append(f"Ignored unreadable metadata key {key}: {exc}")
            continue
        if value is not None:
            metadata[key] = value
    return metadata


def _safe_data_candidate(archive: Any, key: str) -> bool:
    if key in METADATA_KEYS:
        return False
    try:
        value = archive[key]
    except (ValueError, OSError):
        return False
    return (
        isinstance(value, np.ndarray)
        and value.ndim in {2, 3, 4}
        and value.dtype.kind in "uif"
    )


def _select_data_key(archive: Any, keys: list[str], token: str) -> str:
    token_aliases = (
        token,
        token.replace("-", "_").replace(",", "_"),
        f"modality_{token.replace('-', '_').replace(',', '_')}",
    )
    for key in token_aliases + CANONICAL_DATA_KEYS:
        if key in keys and _safe_data_candidate(archive, key):
            return key
    reader_candidates = [
        key for key in READER_DATA_KEYS if key in keys and _safe_data_candidate(archive, key)
    ]
    if len(reader_candidates) == 1:
        return reader_candidates[0]
    candidates = [key for key in keys if _safe_data_candidate(archive, key)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise InputValidationError(
            "No numeric 2D/3D snapshot array was found. Expected wsg_data, data, "
            "arr_0, the modality token, or one unambiguous Reader array."
        )
    raise InputValidationError(
        "The archive contains multiple possible snapshot arrays "
        f"({', '.join(candidates)}); name the intended array {token!r} or 'wsg_data'."
    )


def load_array_container(
    path: str | Path,
    token: str,
    *,
    data_key: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], list[str]]:
    """Load one safe numeric array and return array, container, metadata, warnings."""
    path = Path(path)
    warnings: list[str] = []
    if path.suffix.lower() == ".npy":
        try:
            array = np.load(path, allow_pickle=False)
        except ValueError as exc:
            raise InputValidationError(f"Unsafe or invalid NPY input: {exc}") from exc
        return np.asarray(array), {
            "container": "npy",
            "keys": [],
            "data_key": None,
        }, {}, warnings

    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"Invalid WSG/NPZ archive: {exc}") from exc
    if isinstance(archive, np.ndarray):
        return np.asarray(archive), {
            "container": "npy",
            "keys": [],
            "data_key": None,
        }, {}, warnings
    try:
        keys = list(archive.files)
        metadata = _parse_metadata(archive, keys, warnings)
        selected = data_key or _select_data_key(archive, keys, token)
        if selected not in keys:
            raise InputValidationError(f"Recorded array key {selected!r} is missing")
        try:
            array = np.asarray(archive[selected])
        except ValueError as exc:
            raise InputValidationError(
                f"Array {selected!r} requires pickle or has an unsupported dtype"
            ) from exc
        return array, {
            "container": "wsg" if path.suffix.lower() == ".wsg" else "npz",
            "keys": keys,
            "data_key": selected,
        }, metadata, warnings
    finally:
        archive.close()


def _parse_declared_channels(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_numeric_token(value.replace(" ", ""))
        if parsed is None:
            raise InputValidationError(f"Invalid channels metadata: {value!r}")
        return parsed
    if isinstance(value, (list, tuple)):
        try:
            channels = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise InputValidationError("channels metadata must contain integers") from exc
        if len(channels) != len(set(channels)):
            raise InputValidationError("channels metadata contains duplicates")
        return channels
    raise InputValidationError("channels metadata must be a channel string or integer list")


def _declared_channels(metadata: Mapping[str, Any]) -> list[int] | None:
    for key in ("channel_ids", "channel_indices", "channels"):
        if key in metadata:
            return _parse_declared_channels(metadata[key])
    return None


def _declared_token(metadata: Mapping[str, Any]) -> str:
    for key in ("modality_id", "token", "channel_key", "modality"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_config_contract(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalization = payload.get("normalization") or {}
    ranges = {
        int(channel): (float(bounds[0]), float(bounds[1]))
        for channel, bounds in (normalization.get("channel_ranges") or {}).items()
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    }
    specs = payload.get("modality_specs") or {}
    if not isinstance(specs, dict):
        raise InputValidationError(f"{config_path} modality_specs must be an object")
    return {"ranges": ranges, "specs": specs}


def _layout_name(value: Any) -> str:
    return str(value or "").upper().replace(" ", "")


def _canonical_view(
    array: np.ndarray,
    *,
    expected_count: int,
    numeric_channels: list[int] | None,
    declared_layout: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if array.dtype.kind not in "uif":
        raise InputValidationError(f"Snapshot dtype must be numeric, got {array.dtype}")
    if array.ndim not in {2, 3, 4}:
        raise InputValidationError(
            f"Expected one H×W, H×W×C, or C×H×W snapshot, got shape {array.shape}"
        )

    batch_axis: int | None = None
    layout = _layout_name(declared_layout)
    if array.ndim == 4:
        if layout in {"NHWC", "NCHW"}:
            if array.shape[0] != 1:
                raise InputValidationError(
                    f"ControlWorld expects one snapshot; declared {layout} input has {array.shape[0]} frames"
                )
            batch_axis = 0
            array = array[0]
            layout = layout[1:]
        elif array.shape[0] == 1:
            batch_axis = 0
            array = array[0]
        else:
            raise InputValidationError(
                f"ControlWorld expects one snapshot, not a batch/video with shape {array.shape}"
            )

    if array.ndim == 2:
        canonical = array[..., None]
        inferred_layout = "HW"
    else:
        if layout and layout not in {"HWC", "CHW"}:
            raise InputValidationError(f"Unsupported layout metadata {declared_layout!r}")
        if not layout:
            maximum_channel = max(numeric_channels) if numeric_channels else -1

            def score(axis: int) -> float:
                channels = int(array.shape[axis])
                spatial = [int(array.shape[index]) for index in range(3) if index != axis]
                if min(spatial) < 8 or channels <= 0 or channels > 4096:
                    return -math.inf
                value = 0.0
                if channels == expected_count:
                    value += 8.0
                if channels == STANDARD_WSG_CHANNELS:
                    value += 10.0
                if numeric_channels and channels > maximum_channel:
                    value += 4.0
                if channels <= STANDARD_WSG_CHANNELS:
                    value += 1.0
                if axis == 2:
                    value += 0.1
                return value

            scores = {"CHW": score(0), "HWC": score(2)}
            layout = max(scores, key=scores.get)
            if not math.isfinite(scores[layout]):
                raise InputValidationError(
                    f"Could not infer a channel axis for shape {array.shape}; add layout='HWC' or 'CHW' metadata"
                )
        inferred_layout = layout
        canonical = np.transpose(array, (1, 2, 0)) if layout == "CHW" else array

    height, width, channel_count = map(int, canonical.shape)
    if min(height, width) < 8:
        raise InputValidationError(
            f"Snapshot spatial dimensions are too small: {height}×{width}"
        )
    return canonical, {
        "layout": inferred_layout,
        "batch_axis": batch_axis,
        "canonical_shape": [height, width, channel_count],
        "source_shape": [int(value) for value in (array.shape if batch_axis is None else (1, *array.shape))],
    }


def _sample_plane(plane: np.ndarray, limit: int = 200_000) -> np.ndarray:
    flat = np.asarray(plane).reshape(-1)
    if flat.size <= limit:
        return flat
    stride = max(1, flat.size // limit)
    return flat[::stride][:limit]


def _range_report(
    canonical: np.ndarray,
    source_indices: Mapping[int, int],
    expected_channels: list[int],
    ranges: Mapping[int, tuple[float, float]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    reports: list[dict[str, Any]] = []
    informative_values = 0
    for channel in expected_channels:
        source_index = source_indices[channel]
        sampled = _sample_plane(canonical[..., source_index]).astype(np.float64, copy=False)
        finite = np.isfinite(sampled)
        nonfinite_fraction = float(1.0 - finite.mean()) if sampled.size else 1.0
        if nonfinite_fraction > 0:
            errors.append(
                f"channel {channel} contains {nonfinite_fraction:.2%} NaN/Inf values"
            )
        valid = sampled[finite & (sampled != MASK_VALUE)]
        informative_values += int(valid.size)
        item: dict[str, Any] = {
            "channel": channel,
            "valid_samples": int(valid.size),
            "nonfinite_fraction": nonfinite_fraction,
        }
        if valid.size:
            item["observed_min"] = float(valid.min())
            item["observed_max"] = float(valid.max())
        bounds = ranges.get(channel)
        if bounds is not None and valid.size:
            lower, upper = bounds
            span = max(abs(upper - lower), 1e-6)
            tolerance = max(1e-4, 0.02 * span)
            outside = (valid < lower - tolerance) | (valid > upper + tolerance)
            outside_fraction = float(outside.mean())
            item["expected_range"] = [lower, upper]
            item["outside_fraction"] = outside_fraction
            if outside_fraction > 0.05:
                warnings.append(
                    f"channel {channel} has {outside_fraction:.1%} values outside its training range "
                    f"[{lower:g}, {upper:g}]; inference will use the checkpoint's training normalization"
                )
            elif outside_fraction > 0:
                warnings.append(
                    f"channel {channel} has {outside_fraction:.2%} mild range outliers"
                )
        elif bounds is None:
            warnings.append(f"No portable training range is registered for channel {channel}")
        reports.append(item)
    if informative_values == 0:
        errors.append("All requested modality values are unavailable/masked (-1)")
    return reports, errors


def validate_tensor_input(
    path: str | Path,
    token: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate a WSG/NPZ/NPY against one selected input modality."""
    path = Path(path)
    if path.suffix.lower() not in {".wsg", ".npz", ".npy"}:
        raise InputValidationError("Non-RGB inputs must use .wsg, .npz, or .npy")
    array, container, metadata, warnings = load_array_container(path, token)
    red_reasons: list[str] = []
    config = _load_config_contract(config_path)
    numeric_channels = parse_numeric_token(token)
    modality_spec = dict(config["specs"].get(token) or {})
    if numeric_channels is None:
        if not modality_spec:
            raise InputValidationError(
                f"String modality {token!r} needs a modality_specs entry in {Path(config_path).name}"
            )
        expected_count = int(modality_spec.get("channels") or 0)
        if expected_count <= 0:
            raise InputValidationError(f"String modality {token!r} has no positive channel count")
    else:
        expected_count = len(numeric_channels)

    declared_token = _declared_token(metadata)
    if declared_token and declared_token != token:
        red_reasons.append(
            f"The file declares modality {declared_token!r}, but this input requires {token!r}"
        )
    declared_channels = _declared_channels(metadata)
    canonical, shape_contract = _canonical_view(
        array,
        expected_count=expected_count,
        numeric_channels=numeric_channels,
        declared_layout=str(metadata.get("layout") or modality_spec.get("layout") or ""),
    )
    channel_count = int(canonical.shape[-1])

    value_space = str(metadata.get("value_space") or "raw").strip().lower()
    normalized = metadata.get("normalized")
    normalization = metadata.get("normalization")
    if normalized is True or value_space in {"normalized", "unit", "minus1_1"}:
        red_reasons.append(
            "This tensor declares already-normalized values. Upload raw WSG values so the selected Head's training normalization is applied exactly once."
        )
    if isinstance(normalization, str) and normalization.lower() in {"normalized", "unit", "minus1_1"}:
        red_reasons.append(
            "This tensor declares normalized values; raw modality values are required"
        )
    declared_training_mode = str(
        metadata.get("training_normalization_mode") or ""
    ).lower()
    if declared_training_mode and declared_training_mode not in {
        "linear", "log", "xpose_rgb_minus1_1"
    }:
        red_reasons.append(
            f"Unsupported training normalization mode {declared_training_mode!r}"
        )
    raw_training_ranges = metadata.get("training_channel_ranges") or {}
    training_ranges: dict[int, list[float]] = {}
    if isinstance(raw_training_ranges, dict):
        for raw_channel, raw_bounds in raw_training_ranges.items():
            try:
                channel = int(raw_channel)
                bounds = [float(raw_bounds[0]), float(raw_bounds[1])]
            except (TypeError, ValueError, IndexError):
                red_reasons.append(
                    f"Invalid cached training range for channel {raw_channel!r}"
                )
                continue
            if not all(math.isfinite(value) for value in bounds) or bounds[1] < bounds[0]:
                red_reasons.append(
                    f"Invalid cached training range for channel {channel}: {bounds}"
                )
                continue
            training_ranges[channel] = bounds

    declared_image_like = bool(modality_spec.get("storage") == "image_like")
    visualization_markers = {"visualization", "visualisation", "display", "preview", "pseudo_rgb", "rgb_preview"}
    if not declared_image_like and value_space in visualization_markers:
        red_reasons.append(
            "This file declares a display visualization rather than raw modality values. "
            "Use the original WSG/NPZ/NPY data; Reader colors must never be encoded as the condition."
        )

    if numeric_channels is not None:
        maximum = max(numeric_channels)
        if declared_channels is not None:
            if len(declared_channels) != channel_count:
                raise InputValidationError(
                    f"channels metadata has {len(declared_channels)} entries but the array has {channel_count} channels"
                )
            missing = [channel for channel in numeric_channels if channel not in declared_channels]
            if missing:
                raise InputValidationError(
                    f"The declared channel map does not contain required channels: {missing}"
                )
            source_indices = {
                channel: declared_channels.index(channel) for channel in numeric_channels
            }
            storage = (
                "full_wsg"
                if channel_count == STANDARD_WSG_CHANNELS
                and declared_channels == list(range(channel_count))
                else "modality_scoped"
            )
            channel_ids = declared_channels
            semantic_source = "channels metadata"
        elif channel_count > maximum:
            source_indices = {channel: channel for channel in numeric_channels}
            storage = "full_wsg"
            channel_ids = list(range(channel_count))
            semantic_source = "global WSG channel axis"
            if channel_count != STANDARD_WSG_CHANNELS:
                warnings.append(
                    f"Full-layout input has {channel_count}, not {STANDARD_WSG_CHANNELS}, channels"
                )
        elif channel_count == expected_count:
            source_indices = {
                channel: index for index, channel in enumerate(numeric_channels)
            }
            storage = "modality_scoped"
            channel_ids = list(numeric_channels)
            semantic_source = "current task inference"
            if not declared_token and container.get("data_key") != token:
                warnings.append(
                    f"Inferred ordered channels {token} from the selected task; add modality/token and channels metadata for a self-describing file"
                )
        else:
            raise InputValidationError(
                f"Input {token} needs either a full array containing global channel {maximum} "
                f"or exactly {expected_count} scoped channels; received {channel_count}"
            )
        range_reports, errors = _range_report(
            canonical, source_indices, numeric_channels, config["ranges"], warnings
        )
    else:
        if channel_count != expected_count:
            raise InputValidationError(
                f"String modality {token!r} expects {expected_count} channels, got {channel_count}"
            )
        storage = "string_modality"
        channel_ids = list(range(channel_count))
        source_indices = {index: index for index in range(channel_count)}
        semantic_source = "modality metadata" if declared_token else "custom modality specification"
        spec_range = modality_spec.get("value_range")
        local_ranges = (
            {index: (float(spec_range[0]), float(spec_range[1])) for index in range(channel_count)}
            if isinstance(spec_range, (list, tuple)) and len(spec_range) == 2 else {}
        )
        range_reports, errors = _range_report(
            canonical, source_indices, list(range(channel_count)), local_ranges, warnings
        )
        if not declared_token and container.get("data_key") != token:
            warnings.append(
                f"The file does not name string modality {token!r}; add modality_id={token!r} for portability"
            )

    red_reasons.extend(errors)

    data_key = str(container.get("data_key") or "")
    numeric_bounds = [config["ranges"].get(channel) for channel in (numeric_channels or [])]
    range_implies_image_canvas = bool(
        numeric_channels
        and len(numeric_channels) == 3
        and all(
            bounds is not None and bounds[0] >= 0 and 128 <= bounds[1] <= 255
            for bounds in numeric_bounds
        )
    )
    image_like = declared_image_like or range_implies_image_canvas
    if not image_like and canonical.shape[-1] == 3 and canonical.dtype == np.uint8:
        if numeric_bounds and all(bounds is not None for bounds in numeric_bounds):
            red_reasons.append(
                "The payload looks like an 8-bit RGB visualization, but this modality is trained from raw numeric values. "
                "Upload the original modality array, not the WSG Reader preview."
            )
        else:
            warnings.append(
                "This is an 8-bit three-channel tensor, but no complete range contract proves whether it is raw data or a Reader visualization; verify it before inference"
            )
    if not image_like and data_key.lower().endswith(("_rgb", "_preview", "_visualization")):
        red_reasons.append(
            f"Array key {data_key!r} identifies a visualization. The model requires the raw modality array."
        )
    if container.get("data_key") in READER_DATA_KEYS:
        warnings.append(
            f"Accepted WSG Reader export key {container.get('data_key')!r}; a self-describing wsg_data archive is more portable"
        )
    if container.get("data_key") == "arr_0":
        warnings.append("Accepted generic NPZ key 'arr_0'; modality metadata is recommended")

    red_reasons = list(dict.fromkeys(red_reasons))
    non_runnable = any("NaN/Inf" in reason for reason in red_reasons)
    contract = {
        **shape_contract,
        "token": token,
        "data_key": container.get("data_key"),
        "container": container["container"],
        "storage": storage,
        "channel_ids": channel_ids,
        "expected_channels": numeric_channels,
        "value_space": "raw",
        "encoding": str(modality_spec.get("encoding") or "training_normalization"),
        "training_normalization_mode": declared_training_mode or None,
        "training_channel_ranges": training_ranges,
    }
    severity = "red" if red_reasons else ("orange" if warnings else "green")
    return {
        "valid": True,
        "accepted": True,
        "runnable": not non_runnable,
        "severity": severity,
        "summary": (
            f"{container['container'].upper()} · {storage.replace('_', ' ')} · "
            f"{canonical.shape[1]}×{canonical.shape[0]}×{canonical.shape[2]} · {array.dtype}"
        ),
        "token": token,
        "declared_token": declared_token or None,
        "semantic_source": semantic_source,
        "archive_keys": container.get("keys", []),
        "warnings": list(dict.fromkeys(warnings)),
        "red_reasons": red_reasons,
        "ranges": range_reports,
        "contract": contract,
    }


def inspect_tensor_input(
    path: str | Path,
    token: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """Diagnose an upload without deleting it when its contract is invalid."""
    try:
        return validate_tensor_input(path, token, config_path)
    except (InputValidationError, OSError, ValueError) as exc:
        return {
            "valid": False,
            "accepted": True,
            "runnable": False,
            "severity": "red",
            "summary": "Input retained, but its tensor contract cannot be mapped",
            "token": token,
            "warnings": [],
            "red_reasons": [str(exc)],
            "ranges": [],
            "contract": None,
        }


def canonical_array_from_contract(
    path: str | Path,
    contract: Mapping[str, Any],
) -> np.ndarray:
    """Reload and orient a tensor according to a previously validated contract."""
    token = str(contract["token"])
    array, _, _, _ = load_array_container(
        path, token, data_key=contract.get("data_key")
    )
    if contract.get("batch_axis") is not None:
        if array.ndim != 4 or array.shape[0] != 1:
            raise InputValidationError("Input batch shape changed after validation")
        array = array[0]
    layout = str(contract.get("layout") or "HWC")
    if layout == "HW":
        array = array[..., None]
    elif layout == "CHW":
        array = np.transpose(array, (1, 2, 0))
    elif layout != "HWC":
        raise InputValidationError(f"Unsupported validated layout: {layout}")
    return np.asarray(array)


def materialize_numeric_wsg(
    array: np.ndarray,
    contract: Mapping[str, Any],
) -> np.ndarray:
    """Map scoped channels back to their global WSG channel positions."""
    expected = contract.get("expected_channels")
    if expected is None:
        return array
    channel_ids = [int(value) for value in contract.get("channel_ids") or []]
    if channel_ids == list(range(array.shape[-1])) and array.shape[-1] > max(expected):
        return array
    needed = max(int(value) for value in expected) + 1
    output = np.full((*array.shape[:2], needed), MASK_VALUE, dtype=array.dtype)
    for source_index, channel in enumerate(channel_ids):
        if channel < needed:
            output[..., channel] = array[..., source_index]
    return output


def validate_rgb_image(path: str | Path, token: str) -> dict[str, Any]:
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format or path.suffix.lstrip(".").upper()
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"Invalid RGB image: {exc}") from exc
    if min(width, height) < 8:
        raise InputValidationError(f"RGB image is too small: {width}×{height}")
    return {
        "valid": True,
        "accepted": True,
        "runnable": True,
        "severity": "green",
        "summary": f"{image_format} · RGB input · {width}×{height} · {mode}",
        "token": token,
        "warnings": [],
        "red_reasons": [],
        "contract": {"token": token, "kind": "image", "width": width, "height": height},
    }


def validate_wav_audio(path: str | Path, token: str) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.getnframes()
            width = handle.getsampwidth()
    except (wave.Error, OSError) as exc:
        raise InputValidationError(f"Invalid WAV audio: {exc}") from exc
    if channels <= 0 or rate <= 0 or frames <= 0:
        raise InputValidationError("WAV audio is empty or has an invalid header")
    return {
        "valid": True,
        "accepted": True,
        "runnable": True,
        "severity": "green",
        "summary": f"WAV · {channels} channel(s) · {rate} Hz · {frames / rate:.2f}s",
        "token": token,
        "warnings": [],
        "red_reasons": [],
        "contract": {
            "token": token,
            "kind": "audio",
            "channels": channels,
            "sample_rate": rate,
            "sample_width": width,
        },
    }
