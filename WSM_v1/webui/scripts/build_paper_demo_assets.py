#!/usr/bin/env python3
"""Build the distributable ControlWorld demo bundle from paper Figures 14--39."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import tempfile
import sys
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper/evaluation"
MANIFEST_ROOT = PAPER_ROOT / "qualitative/manifests"
SELECTION_PATH = PAPER_ROOT / "configs/paper_selection.json"
OUTPUT_ROOT = PROJECT_ROOT / "webui/demo_assets"
VISUALIZATION_ASSET_ROOT = OUTPUT_ROOT / "input_visualizations"
VARIABLE_ASSET_ROOT = OUTPUT_ROOT / "input_variables"
LATENT_ASSET_ROOT = OUTPUT_ROOT / "input_latents"
REFERENCE_TOKENS = {
    "Reference_IMG",
    "FirstFrame_RGB",
    "CurrentFrame_RGB_Reference",
}
RGB_VARIABLE_TOKENS = REFERENCE_TOKENS | {"0-2", "CurrentFrame_RGB_Condition"}
class RawInputResolver:
    """Use cache metadata to recover the exact raw frame used for evaluation."""
    CACHE_ROOTS = {
        "dense": PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset",
        "custom": PROJECT_ROOT / "data/mp100_universal_pose_flux2_instance_crop_latents",
        "xpose": PROJECT_ROOT / "data/xpose_flux2_frame_latents_sample10pct_2frames",
    }
    # These dense caches were encoded before AVIS WSV archives were rewritten
    # from the historical lexicographic crop-frame order into natural AVIS JSON
    # order. Later cache rebuilds (notably RGB, surface normals, and channel
    # 195) already use the WSV's current keys and must not be remapped.
    LEGACY_AVIS_FRAME_ORDER_MODALITIES = {
        "32-33",
        "34-35",
        "36-38",
        "39",
        "40",
        "41-42",
        "67-69",
        "99-101",
        "103",
    }

    def __init__(self) -> None:
        self._indices: dict[tuple[str, str], dict[str, tuple[Path, int]]] = {}
        self._reference_payload: dict[str, Any] | None = None
        self._reference_index: dict[str, int] | None = None
        self._avis_frame_maps: dict[tuple[Path, str], dict[int, int]] = {}
    @staticmethod
    def _safe_modality(token: str) -> str:
        return token.replace("-", "_")
    def _index(self, source: str, token: str) -> dict[str, tuple[Path, int]]:
        key = (source, token)
        if key in self._indices:
            return self._indices[key]
        root = self.CACHE_ROOTS.get(source)
        if root is None:
            raise KeyError(f"No cache metadata is registered for Paper Demo source {source!r}")
        shard_dir = root / "shards" / self._safe_modality(token)
        index: dict[str, tuple[Path, int]] = {}
        for shard in sorted(shard_dir.glob("shard_*.pt")):
            payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
            for offset, sample_id in enumerate(payload.get("sample_ids") or []):
                index[str(sample_id)] = (shard, offset)
            del payload
        if not index:
            raise FileNotFoundError(f"No cached samples found in {shard_dir}")
        self._indices[key] = index
        return index
    def _metadata_at(self, payload: dict[str, Any], offset: int) -> dict[str, Any]:
        source = dict((payload.get("sample_sources") or [])[offset])
        frame_keys = payload.get("frame_keys") or []
        crop_infos = payload.get("crop_infos") or []
        if frame_keys:
            source["frame_key"] = frame_keys[offset]
        if crop_infos:
            source["crop_info"] = crop_infos[offset]
        modality = str(payload.get("modality") or "")
        mode_map = payload.get("normalization_modes") or {}
        recorded_mode = str(
            mode_map.get(modality) or payload.get("normalization_mode") or "linear"
        )
        # Dense RGB caches created under the historical global-log setup still
        # carry a "log" metadata label, although channels 0--2 have always
        # taken the fixed value/255 path in WSGNormalizationStore.
        source["training_normalization_mode"] = (
            "linear" if modality == "0-2" else recorded_mode
        )
        config_value = str(payload.get("modality_config_path") or "").strip()
        if config_value:
            config_path = Path(config_value).expanduser()
            if not config_path.is_absolute():
                config_path = PROJECT_ROOT / config_path
            if config_path.is_file():
                normalization = load_json(config_path).get("normalization") or {}
                source["training_channel_ranges"] = normalization.get("channel_ranges") or {}
        return source

    def _align_legacy_avis_raw_frame(
        self, metadata: dict[str, Any], token: str
    ) -> dict[str, Any]:
        """Resolve the WSV key whose content produced a legacy cached latent."""
        canonical = "0-2" if token in RGB_VARIABLE_TOKENS else token
        if canonical not in self.LEGACY_AVIS_FRAME_ORDER_MODALITIES:
            return metadata
        source = Path(str(metadata.get("path") or ""))
        split_root = source.parents[2] if len(source.parents) >= 3 else None
        if split_root is None or not split_root.name.startswith("AVIS_"):
            return metadata
        split = split_root.name.removeprefix("AVIS_").removesuffix("_results")
        index_path = split_root / f"{split}.json"
        video_name = source.parents[1].name
        cache_key = (index_path, video_name)
        mapping = self._avis_frame_maps.get(cache_key)
        if mapping is None:
            payload = load_json(index_path)
            video = next(
                (
                    item
                    for item in payload.get("videos", [])
                    if item.get("file_names")
                    and str(item["file_names"][0]).split("/", 1)[0] == video_name
                ),
                None,
            )
            if video is None:
                raise KeyError(f"No AVIS frame-order record for {video_name} in {index_path}")
            file_names = [str(value) for value in video["file_names"]]
            natural_index = {name: index + 1 for index, name in enumerate(file_names)}
            mapping = {
                old_index + 1: natural_index[name]
                for old_index, name in enumerate(sorted(file_names))
            }
            self._avis_frame_maps[cache_key] = mapping
        frame_key = str(metadata.get("frame_key") or "")
        if not frame_key.startswith("index_"):
            raise KeyError(f"Cannot align legacy AVIS frame key {frame_key!r}")
        cache_index = int(frame_key.removeprefix("index_"))
        wsv_index = int(mapping.get(cache_index, cache_index))
        aligned = dict(metadata)
        aligned["cache_frame_key"] = frame_key
        aligned["frame_key"] = f"index_{wsv_index:08d}"
        aligned["raw_frame_alignment"] = "cached_latent_to_current_wsv"
        return aligned

    def current(self, source: str, token: str, sample_id: str) -> dict[str, Any]:
        lookup_source = "xpose" if sample_id.startswith("xpose::") else source
        location = self._index(lookup_source, token).get(sample_id)
        if location is None:
            raise KeyError(f"No exact {source}:{token} source metadata for {sample_id}")
        shard, offset = location
        payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
        metadata = self._metadata_at(payload, offset)
        if source == "dense" and "AVIS_" in sample_id:
            metadata = self._align_legacy_avis_raw_frame(metadata, token)
        return metadata
    def current_latent(self, source: str, token: str, sample_id: str) -> torch.Tensor:
        lookup_source = "xpose" if sample_id.startswith("xpose::") else source
        location = self._index(lookup_source, token).get(sample_id)
        if location is None:
            raise KeyError(f"No exact {source}:{token} latent for {sample_id}")
        shard, offset = location
        payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
        return payload["latents"][offset].clone()

    def historical_rgb(self, sample_id: str) -> dict[str, Any]:
        if "AVIS_test_results" not in sample_id:
            return self.current("dense", "0-2", sample_id)
        if self._reference_payload is None:
            path = self.CACHE_ROOTS["dense"] / "reference_frames/0_2_avis_v2_first_frame.pt"
            self._reference_payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
            keys = self._reference_payload.get("group_keys") or self._reference_payload["video_ids"]
            self._reference_index = {str(key): index for index, key in enumerate(keys)}
        video_id = sample_id.split("/AVIS_test_results/", 1)[1].split("/", 1)[0]
        key = f"AVIS_test_results\x1f{video_id}"
        offset = (self._reference_index or {}).get(key)
        if offset is None:
            raise KeyError(f"No exact historical RGB source metadata for {sample_id}")
        return self._metadata_at(self._reference_payload, offset)
    def historical_rgb_latent(self, sample_id: str) -> torch.Tensor:
        if "AVIS_test_results" not in sample_id:
            return self.current_latent("dense", "0-2", sample_id)
        self.historical_rgb(sample_id)
        video_id = sample_id.split("/AVIS_test_results/", 1)[1].split("/", 1)[0]
        key = f"AVIS_test_results\x1f{video_id}"
        offset = (self._reference_index or {}).get(key)
        if offset is None or self._reference_payload is None:
            raise KeyError(f"No exact historical RGB latent for {sample_id}")
        return self._reference_payload["latents"][offset].clone()

class ExactPromptResolver:
    """Resolve the exact text condition consumed by formal inference."""
    DENSE_ROOT = PROJECT_ROOT / "data/wsg_flux2_caption_embeddings_all_datasets"
    PROMPT_FILES = {
        "custom": PROJECT_ROOT / "data/mp100_universal_pose_flux2_instance_crop_latents/prompts.pt",
        "xpose": PROJECT_ROOT / "data/xpose_flux2_frame_latents_sample10pct_2frames/prompts.pt",
    }
    def __init__(self) -> None:
        self._dense_index: dict[str, tuple[Path, int]] | None = None
        self._prompt_indices: dict[str, dict[str, str]] = {}
    def _build_dense_index(self) -> dict[str, tuple[Path, int]]:
        if self._dense_index is None:
            index: dict[str, tuple[Path, int]] = {}
            for shard in sorted(self.DENSE_ROOT.glob("shard_*.pt")):
                payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
                sample_ids = payload.get("sample_ids") or []
                text_indices = payload.get("sample_text_indices") or []
                index.update({
                    str(sample_id): (shard, int(text_index))
                    for sample_id, text_index in zip(sample_ids, text_indices)
                })
                del payload
            if not index:
                raise FileNotFoundError(f"No exact text cache found in {self.DENSE_ROOT}")
            self._dense_index = index
        return self._dense_index
    def _prompt_index(self, source: str) -> dict[str, str]:
        if source not in self._prompt_indices:
            path = self.PROMPT_FILES[source]
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            sample_ids = payload.get("sample_ids") or []
            prompts = payload.get("prompts") or payload.get("descriptions") or []
            index = {str(sample_id): str(prompt) for sample_id, prompt in zip(sample_ids, prompts)}
            if not index:
                raise FileNotFoundError(f"No exact text prompts found in {path}")
            self._prompt_indices[source] = index
        return self._prompt_indices[source]
    def exact(self, source: str, sample_id: str) -> str:
        if sample_id.startswith("xpose::"):
            index = self._prompt_index("xpose")
            return index[sample_id]
        if source == "custom":
            index = self._prompt_index("custom")
            return index[sample_id]
        location = self._build_dense_index().get(sample_id)
        if location is None:
            raise KeyError(f"No exact text condition for {source}:{sample_id}")
        shard, text_index = location
        payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
        return str(payload["unique_prompts"][text_index])
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
def resolve_path(value: str, evaluation_root: Path) -> Path | None:
    value = str(value or "")
    if not value:
        return None
    if value.startswith("project://"):
        return PROJECT_ROOT / value.removeprefix("project://")
    return evaluation_root / value
def square_crop(image: Image.Image, framing: dict[str, Any] | None) -> Image.Image:
    """Apply the same framing contract used by the qualitative paper figures."""
    framing = framing or {}
    width, height = image.size
    bottom_padding = max(
        0,
        min(
            height,
            int(round(height * float(framing.get("bottom_padding_fraction", 0.0)))),
        ),
    )
    shift = bottom_padding // 2
    if shift:
        recentered = Image.new(image.mode, image.size)
        recentered.paste(image.crop((0, height - shift, width, height)), (0, 0))
        recentered.paste(image.crop((0, 0, width, height - shift)), (0, shift))
        image = recentered
    side = max(1, int(round(min(width, height) * float(framing.get("crop_scale", 1.0)))))
    center_x = float(framing.get("center_x", 0.5)) * width
    center_y = float(framing.get("center_y", 0.5)) * height
    left = max(0, min(width - side, int(round(center_x - side / 2))))
    top = max(0, min(height - side, int(round(center_y - side / 2))))
    return image.crop((left, top, left + side, top + side))
def _paper_reader_display(image: Image.Image, source: Path, token: str) -> Image.Image:
    """Match the display transform used by the final qualitative paper figures."""
    canonical = "0-2" if token in RGB_VARIABLE_TOKENS else token
    if canonical != "39" or "/wsg_reader/" in source.as_posix():
        return image
    script = PAPER_ROOT / "qualitative/scripts/build_expanded_galleries.py"
    spec = importlib.util.spec_from_file_location("wsm_paper_gallery_display", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load paper display transforms from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._reader_curvature(image)

def save_asset(source: Path, framing: dict[str, Any] | None, token: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        image = square_crop(opened.convert("RGB"), framing)
        image = _paper_reader_display(image, source, token)
        if image.width > 512 or image.height > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()[:20]
        relative = Path("input_visualizations") / f"{digest}.png"
        destination = OUTPUT_ROOT / relative
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG", optimize=True)
    return relative.as_posix()
def save_latent(latent: torch.Tensor) -> str:
    latent = latent.detach().cpu().contiguous()
    fingerprint = (
        latent.view(torch.uint8).numpy().tobytes()
        + str(tuple(latent.shape)).encode()
        + str(latent.dtype).encode()
    )
    digest = hashlib.sha256(fingerprint).hexdigest()[:20]
    relative = Path("input_latents") / f"{digest}.pt"
    destination = OUTPUT_ROOT / relative
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent, destination)
    return relative.as_posix()


def _numeric_channels(token: str) -> list[int]:
    canonical = "0-2" if token in RGB_VARIABLE_TOKENS else token
    channels: list[int] = []
    for part in canonical.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            channels.extend(range(start, end + 1))
        else:
            channels.append(int(part))
    return channels
def _letterboxed_rgb(source: Path, crop_info: dict[str, Any]) -> np.ndarray:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    resize_width = int(crop_info.get("resize_width") or image.width)
    resize_height = int(crop_info.get("resize_height") or image.height)
    image = image.resize((resize_width, resize_height), Image.Resampling.BICUBIC)
    width = resize_width + int(crop_info.get("pad_left", 0)) + int(crop_info.get("pad_right", 0))
    height = resize_height + int(crop_info.get("pad_top", 0)) + int(crop_info.get("pad_bottom", 0))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(image, (int(crop_info.get("pad_left", 0)), int(crop_info.get("pad_top", 0))))
    return np.asarray(canvas)
def _raw_frame(metadata: dict[str, Any], token: str) -> np.ndarray:
    if metadata.get("rgb_path"):
        return _letterboxed_rgb(Path(metadata["rgb_path"]), dict(metadata.get("crop_info") or {}))
    if metadata.get("zip_path") and metadata.get("video_member"):
        canonical = "0-2" if token in RGB_VARIABLE_TOKENS else token
        archive_path = Path(str(metadata["zip_path"]))
        if canonical == "122-124" and metadata.get("pose_members"):
            members = list(metadata["pose_members"])
            member = str(members[int(metadata.get("selected_frame_offset", len(members) - 1))])
            with zipfile.ZipFile(archive_path) as archive:
                with archive.open(member) as handle, Image.open(handle) as opened:
                    return np.asarray(opened.convert("RGB"))
        import cv2
        member = str(metadata["video_member"])
        frame_index = int(metadata.get("selected_frame_index", 0))
        with tempfile.TemporaryDirectory(prefix="wsm_demo_xpose_") as temp:
            with zipfile.ZipFile(archive_path) as archive:
                extracted = Path(archive.extract(member, temp))
            capture = cv2.VideoCapture(str(extracted))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode frame {frame_index} from {member}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source = Path(str(metadata.get("path") or ""))
    if not source.is_file():
        raise FileNotFoundError(f"Raw World Snapshot source is missing: {source}; metadata={metadata}")
    frame_key = str(metadata.get("frame_key") or "")
    if not frame_key:
        frame_keys = metadata.get("frame_keys") or []
        frame_key = str(frame_keys[0]) if frame_keys else ""
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    archive = np.load(source, allow_pickle=False)
    try:
        if isinstance(archive, np.ndarray):
            return np.asarray(archive)
        if not frame_key:
            if len(archive.files) != 1:
                raise KeyError(f"No frame key recorded for {source}")
            frame_key = archive.files[0]
        return np.asarray(archive[frame_key])
    finally:
        if hasattr(archive, "close"):
            archive.close()
def save_raw_variable(metadata: dict[str, Any], token: str) -> str:
    channels = _numeric_channels(token)
    frame = _raw_frame(metadata, token)
    if frame.ndim == 2:
        frame = frame[..., None]
    maximum = max(channels)
    if frame.ndim != 3:
        raise ValueError(f"Expected one raw HWC/CHW frame, got {frame.shape}")
    if frame.shape[-1] == len(channels):
        scoped = frame
    elif frame.shape[0] == len(channels):
        scoped = np.transpose(frame, (1, 2, 0))
    elif frame.shape[-1] > maximum:
        scoped = frame[..., channels]
    elif frame.shape[0] > maximum:
        scoped = np.transpose(frame[channels, ...], (1, 2, 0))
    else:
        raise ValueError(f"Raw frame {frame.shape} does not contain channels {channels}")
    scoped = np.ascontiguousarray(scoped)
    canonical = "0-2" if token in RGB_VARIABLE_TOKENS else token
    cache_contract = {
        "training_normalization_mode": str(
            metadata.get("training_normalization_mode") or "linear"
        ).lower(),
        "training_channel_ranges": {
            str(channel): list((metadata.get("training_channel_ranges") or {}).get(str(channel), []))
            for channel in channels
            if len((metadata.get("training_channel_ranges") or {}).get(str(channel), [])) == 2
        },
    }
    contract_json = json.dumps(cache_contract, sort_keys=True, separators=(",", ":"))
    fingerprint = (
        scoped.tobytes() + canonical.encode() + str(tuple(scoped.shape)).encode()
        + contract_json.encode()
    )
    digest = hashlib.sha256(fingerprint).hexdigest()[:20]
    safe_token = canonical.replace(",", "_")
    relative = Path("input_variables") / f"{safe_token}_{digest}.wsg.npz"
    destination = OUTPUT_ROOT / relative
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            wsg_data=scoped,
            modality_id=np.asarray(canonical),
            channels=np.asarray(channels, dtype=np.int16),
            layout=np.asarray("HWC"),
            value_space=np.asarray("raw"),
            metadata_json=np.asarray(contract_json),
        )
    return relative.as_posix()
def task_specs() -> dict[tuple[str, int, str, str], str]:
    specs: dict[tuple[str, int, str, str], str] = {}
    config_names = {
        "dense": "wsg_v1_modalities_dense.json",
        "sparse": "wsg_v1_modalities_sparse.json",
        "custom": "wsg_v1_modalities_custom.json",
    }
    for source, name in config_names.items():
        entries = load_json(PROJECT_ROOT / "configs" / name).get("wsg_v1", {})
        for head, entry in entries.items():
            condition = entry.get("condition", [])
            if not isinstance(condition, list):
                condition = [condition]
            condition_key = ",".join(str(token) for token in condition)
            index = int(head.rsplit("_", 1)[-1])
            target = str(entry.get("target") or "")
            specs[(source, index, condition_key, target)] = f"{source}:{head}"
    return specs
def resolve_task_id(record: dict[str, Any], specs: dict[tuple[str, int, str, str], str]) -> tuple[str, str]:
    preferred_sources = (
        ("custom", "dense", "sparse")
        if str(record.get("figure_id", "")).startswith("custom_")
        else ("dense", "custom", "sparse")
    )
    key_tail = (
        int(record["head_index"]),
        str(record["condition"]),
        str(record["target"]),
    )
    matches = [(source, specs[(source, *key_tail)]) for source in preferred_sources if (source, *key_tail) in specs]
    if len(matches) != 1:
        raise ValueError(f"Expected one task for {record['figure_id']}, found {matches}")
    return matches[0]
def prompt_for(record: dict[str, Any], evaluation_root: Path) -> str:
    if "caption" not in str(record.get("condition", "")).split(","):
        return ""
    paths = record.get("paths") or {}
    for key in ("condition", "summary", "reference", "inference"):
        candidate = resolve_path(paths.get(key, ""), evaluation_root)
        prompt_path = candidate.with_name("prompt.txt") if candidate else None
        if prompt_path and prompt_path.is_file():
            return prompt_path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"No prompt.txt found for {record['figure_id']}")
def input_sources(record: dict[str, Any], evaluation_root: Path) -> list[tuple[str, Path]]:
    tokens = [token.strip() for token in str(record["condition"]).split(",") if token.strip()]
    paths = record.get("paths") or {}
    condition = resolve_path(paths.get("condition", ""), evaluation_root)
    reference = resolve_path(paths.get("reference", ""), evaluation_root)
    first_condition = resolve_path(paths.get("first_condition", ""), evaluation_root)
    ground_truth = resolve_path(paths.get("gt", ""), evaluation_root)
    result: list[tuple[str, Path]] = []
    visual_index = 0
    for token in tokens:
        if token == "caption":
            continue
        if token in REFERENCE_TOKENS:
            # Some single-image OBJ examples have no historical frame. Their
            # RGB context is the same observed image that produced the WSG
            # condition, so the evaluator records it as GT rather than as a
            # separate reference file.
            source = reference or first_condition or ground_truth
        elif visual_index == 0:
            source = condition
        else:
            source = first_condition or reference
        if source is None:
            raise FileNotFoundError(f"No input source for {record['figure_id']} token {token}")
        result.append((token, source))
        visual_index += 1
    return result
def sheet_sizes(part_config: dict[str, Any], count: int) -> list[int]:
    if part_config.get("sheet_sizes"):
        sizes = [int(value) for value in part_config["sheet_sizes"]]
    else:
        size = max(1, int(part_config.get("sheet_size", 6)))
        sizes = [min(size, count - start) for start in range(0, count, size)]
    if sum(sizes) != count:
        raise ValueError(f"Part {part_config['part']} sheet sizes do not match {count} records")
    return sizes
def build_record(
    record: dict[str, Any], *, part: str, part_title: str, ordinal: int,
    figure: int, figure_row: int, evaluation_root: Path,
    specs: dict[tuple[str, int, str, str], str],
    raw_resolver: RawInputResolver,
    prompt_resolver: ExactPromptResolver,
) -> dict[str, Any]:
    source, task_id = resolve_task_id(record, specs)
    sample_id = str(record["sample_id"])
    inputs = []
    for visual_index, (token, path) in enumerate(input_sources(record, evaluation_root)):
        historical_rgb = token in {"Reference_IMG", "FirstFrame_RGB"} or (
            visual_index > 0 and token == "0-2"
        )
        metadata = (
            raw_resolver.historical_rgb(sample_id)
            if historical_rgb
            else raw_resolver.current(source, token, sample_id)
        )
        latent = (
            raw_resolver.historical_rgb_latent(sample_id)
            if historical_rgb
            else raw_resolver.current_latent(source, token, sample_id)
        )
        inputs.append({
            "token": token,
            "visualization_asset": save_asset(path, record.get("framing"), token),
            "raw_asset": save_raw_variable(metadata, token),
            "latent_asset": save_latent(latent),
            "raw_kind": "wsg",
            "visualization_label": "WSG Reader visualization · display only",
        })
    has_caption_input = "caption" in {
        token.strip() for token in str(record["condition"]).split(",")
    }
    if has_caption_input:
        prompt = prompt_resolver.exact(source, sample_id)
    else:
        # Formal WSGOminiLatentDataset only uses cached captions when caption
        # is an explicit task input. Other tasks encode their auto-generated
        # modality route prompt at runtime.
        prompt = ""
    label = (
        f"Figure {figure} · Part {part} · {record.get('condition_label') or record['condition']} "
        f"→ {record.get('target_label') or record['target']}"
    )
    paper_inference = str((record.get("paths") or {}).get("inference") or "")
    paper_output_view = (
        "reader" if "/wsg_reader/" in paper_inference.replace("\\", "/") else "native"
    )
    return {
        "id": f"fig{figure:02d}_row{figure_row:02d}_{record['figure_id']}",
        "figure": figure,
        "figure_row": figure_row,
        "part": part,
        "part_title": part_title,
        "ordinal": ordinal,
        "label": label,
        "task_id": task_id,
        "source": source,
        "head_index": int(record["head_index"]),
        "task_type": str(record.get("task_type") or ""),
        "condition": str(record["condition"]),
        "condition_label": str(record.get("condition_label") or record["condition"]),
        "target": str(record["target"]),
        "target_label": str(record.get("target_label") or record["target"]),
        "prompt": prompt,
        "exact_prompt_only": has_caption_input,
        "paper_output_view": paper_output_view,
        "steps": 10,
        "size": 480,
        "seed": 20260812 + int(record["head_index"]),
        "inputs": inputs,
    }
def main() -> None:
    overview = load_json(MANIFEST_ROOT / "qualitative_materials.manifest.json")
    expanded = load_json(MANIFEST_ROOT / "expanded_galleries.manifest.json")
    selection = load_json(SELECTION_PATH)
    evaluation_root = PROJECT_ROOT / overview["evaluation_root"]
    specs = task_specs()
    raw_resolver = RawInputResolver()
    prompt_resolver = ExactPromptResolver()
    records: list[dict[str, Any]] = []
    overview_records = overview["main_four_interfaces"]
    for row, record in enumerate(overview_records, 1):
        records.append(build_record(
            record, part="I", part_title="Anatomy of the four WSM interfaces",
            ordinal=row, figure=14, figure_row=row,
            evaluation_root=evaluation_root, specs=specs, raw_resolver=raw_resolver,
            prompt_resolver=prompt_resolver,
        ))
    part_configs = {part["id"]: part for part in selection["expanded_qualitative_parts"]}
    next_figure = 15
    for part_id, part in expanded["parts"].items():
        source_records = part["records"]
        sizes = sheet_sizes(part_configs[part_id], len(source_records))
        start = 0
        ordinal = 1
        for size in sizes:
            for row, record in enumerate(source_records[start:start + size], 1):
                records.append(build_record(
                    record, part=str(part["part"]), part_title=str(part["title"]),
                    ordinal=ordinal, figure=next_figure, figure_row=row,
                    evaluation_root=evaluation_root, specs=specs, raw_resolver=raw_resolver,
                    prompt_resolver=prompt_resolver,
                ))
                ordinal += 1
            start += size
            next_figure += 1
    if next_figure != 39:
        raise ValueError(f"Expanded galleries ended at Figure {next_figure - 1}, expected Figure 38")
    for row, record in enumerate(overview["categorical_refiner"], 1):
        records.append(build_record(
            record, part="XI", part_title="Categorical outputs",
            ordinal=row, figure=39, figure_row=row,
            evaluation_root=evaluation_root, specs=specs, raw_resolver=raw_resolver,
            prompt_resolver=prompt_resolver,
        ))
    payload = {
        "schema_version": 3,
        "title": "WSM v1 paper demo",
        "description": "Locked raw inputs, exact latent caches, and display-only WSG Reader visualizations for Figures 14--39.",
        "paper_figures": [14, 39],
        "inference": {"steps": 10, "size": 480, "seed": 20260812},
        "record_count": len(records),
        "task_count": len({record["task_id"] for record in records}),
        "asset_count": len({item["visualization_asset"] for record in records for item in record["inputs"]}),
        "variable_asset_count": len({
            item["raw_asset"] for record in records for item in record["inputs"]
        }),
        "latent_asset_count": len({
            item["latent_asset"] for record in records for item in record["inputs"]
        }),
        "records": records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    referenced = {
        OUTPUT_ROOT / item["visualization_asset"]
        for record in records
        for item in record["inputs"]
    }
    for path in VISUALIZATION_ASSET_ROOT.glob("*.png"):
        if path not in referenced:
            path.unlink()
    referenced_variables = {
        OUTPUT_ROOT / item["raw_asset"]
        for record in records
        for item in record["inputs"]
    }
    for path in VARIABLE_ASSET_ROOT.glob("*.npz"):
        if path not in referenced_variables:
            path.unlink()
    referenced_latents = {
        OUTPUT_ROOT / item["latent_asset"]
        for record in records
        for item in record["inputs"]
    }
    for path in LATENT_ASSET_ROOT.glob("*.pt"):
        if path not in referenced_latents:
            path.unlink()
    print(json.dumps({
        key: payload[key]
        for key in (
            "record_count", "task_count", "asset_count", "variable_asset_count", "latent_asset_count",
        )
    }, indent=2))
if __name__ == "__main__":
    main()
