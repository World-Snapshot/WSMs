"""Canonical raw-input inference entry for TunnelingNet Channel Heads."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps
from peft import set_peft_model_state_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEGACY_MODALITY_CONFIG_NAMES = {
    "wsg_v1_modalities_dense_single_condition_channel_head_condition_first.json":
        "wsg_v1_modalities_dense.json",
    "wsg_v1_modalities_sparse_single_condition_channel_head_condition_first.json":
        "wsg_v1_modalities_sparse.json",
    "wsg_v1_modalities_custom_single_condition_channel_head_condition_first.json":
        "wsg_v1_modalities_custom.json",
}

TRAINING_SCRIPT = PROJECT_ROOT / "scripts/train_TunnelingNet_Flux2_4B_wsg.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webui.input_validation import (
    canonical_array_from_contract,
    inspect_tensor_input,
    materialize_numeric_wsg,
    parse_numeric_token,
    validate_rgb_image,
    validate_wav_audio,
)

from src.pipelines.categorical_refiner import (
    AudioRegionSelectorRuntime,
    CategoricalRefinerRuntime,
    CATEGORICAL_TARGET_MODALITIES,
    load_channel_head_checkpoint,
    channel_head_run_dir,
    resolve_audio_region_selector,
    resolve_categorical_refiner,
)


def log(message: str) -> None:
    print(f"[WSM Inference] {message}", flush=True)


def load_training_module():
    spec = importlib.util.spec_from_file_location("wsm_tunnelingnet_training", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_run_config(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_dir = checkpoint.parents[2]
    yaml_path = run_dir / "resolved_config.yaml"
    runtime_path = run_dir / "resolved_runtime_config.json"
    if not yaml_path.exists() or not runtime_path.exists():
        raise FileNotFoundError(f"Checkpoint run lacks resolved configuration: {run_dir}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle) or {}
    with runtime_path.open("r", encoding="utf-8") as handle:
        runtime = json.load(handle)
    return resolved, runtime, run_dir


def resolve_base_model_path(resolved: dict[str, Any], runtime: dict[str, Any]) -> str:
    configured = str(resolved.get("base_model") or runtime.get("flux_path") or "")
    candidates = [
        os.environ.get("WSM_FLUX2_MODEL_PATH"),
        os.environ.get("FLUX2_MODEL_PATH"),
        configured,
        PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_dir():
            return str(path.resolve())
    return configured


def resolve_modality_config_path(configured: Any) -> str:
    raw = str(configured or "").strip()
    if not raw:
        return str((PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json").resolve())

    path = Path(raw).expanduser()
    replacement = LEGACY_MODALITY_CONFIG_NAMES.get(path.name)
    if replacement:
        return str((PROJECT_ROOT / "configs" / replacement).resolve())

    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    return str(candidate.resolve()) if candidate.exists() else raw


def make_args(resolved: dict[str, Any], runtime: dict[str, Any]) -> SimpleNamespace:
    model = runtime.get("model") or {}
    train = runtime.get("train") or {}
    wsg = runtime.get("wsg") or {}
    rank = int(resolved.get("channel_head_rank") or train.get("channel_head_config", {}).get("r", 8))
    raw_alpha = resolved.get("channel_head_alpha")
    if raw_alpha in (None, "", "auto"):
        raw_alpha = train.get("channel_head_config", {}).get("lora_alpha", rank)
    return SimpleNamespace(
        model_path=resolve_base_model_path(resolved, runtime),
        modality_config=resolve_modality_config_path(resolved.get("modality_config")),
        channel_head_rank=rank,
        channel_head_alpha=float(raw_alpha),
        separate_condition_adapters=bool(model.get("separate_condition_adapters", False)),
        audio_condition_dim=int(resolved.get("audio_condition_dim") or 0),
        audio_conditioning_mode=str(resolved.get("audio_conditioning_mode") or "attention"),
        audio_global_condition=bool(resolved.get("audio_global_condition", False)),
        max_sequence_length=128,
        seed=int(resolved.get("seed") or 20260812),
        condition_modalities=str(resolved.get("condition_modalities") or ""),
        target_modality=str(resolved.get("target_modality") or wsg.get("entry", {}).get("target") or ""),
        target_visual_modality=str(resolved.get("target_visual_modality") or ""),
    )


def configure_runtime(module, resolved: dict[str, Any], runtime: dict[str, Any]) -> None:
    model = runtime.get("model") or {}
    module.set_condition_attention_mode(
        resolved.get("resolved_condition_attention_mode")
        or ("native_joint" if "native_joint" in model.get("branches", []) else "modality_tunneling")
    )
    module.set_separate_condition_adapters(bool(model.get("separate_condition_adapters", False)))
    module.set_trainable_target(bool(resolved.get("resolved_trainable_target", True)))
    module.set_trainable_condition(bool(resolved.get("resolved_trainable_condition", True)))
    module.set_literature_ablations(
        single_step_x0=bool(resolved.get("resolved_single_step_x0", False)),
        high_noise_anchor=bool(resolved.get("resolved_high_noise_anchor", False)),
        multires_noise=bool(resolved.get("resolved_multires_noise", False)),
    )


def load_weights(model, checkpoint: Path) -> tuple[Mapping[str, Any] | None, str]:
    raw_state, embedded_refiner, checkpoint_format = load_channel_head_checkpoint(checkpoint)
    state = {
        key.removeprefix("transformer."): value
        for key, value in raw_state.items()
    }
    result = set_peft_model_state_dict(model.transformer, state, adapter_name="default")
    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        print(f"[WSM WebUI] warning: unexpected Channel Head keys: {unexpected[:5]}", flush=True)

    condition_path = checkpoint.parent / "condition_lora_adapters.pt"
    if model.extra_lora_adapter_names and condition_path.exists():
        condition_states = torch.load(condition_path, map_location="cpu")
        for name, adapter_state in condition_states.items():
            set_peft_model_state_dict(model.transformer, adapter_state, adapter_name=name)

    for filename, attribute in (
        ("audio_condition_projector.pt", "audio_condition_projector"),
        ("audio_modulation_projector.pt", "audio_modulation_projector"),
    ):
        path = checkpoint.parent / filename
        target = getattr(model, attribute, None)
        if target is not None and path.exists():
            target.load_state_dict(torch.load(path, map_location="cpu"))
    gate_path = checkpoint.parent / "audio_modulation_gate.pt"
    if model.audio_modulation_gate is not None and gate_path.exists():
        model.audio_modulation_gate.data.copy_(
            torch.load(gate_path, map_location=model.audio_modulation_gate.device)
        )
    return embedded_refiner, checkpoint_format


def image_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def encode_audio_for_refiner(selector_runtime, path: Path) -> torch.Tensor:
    """Encode uploaded WAV with the exact AST contract stored by Head 010."""
    adapter_dir = PROJECT_ROOT / "third_party/ScoreHMR/wsm_adapters"
    if str(adapter_dir) not in sys.path:
        sys.path.insert(0, str(adapter_dir))
    from build_wsv_audio_dataset import AudioFeatureEncoder, _read_audio_file

    config = selector_runtime.config
    encoder_name = str(config.get("audio_encoder") or "ast")
    model_name = str(
        config.get("audio_model_name")
        or "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    waveform, sample_rate = _read_audio_file(Path(path))
    encoder = AudioFeatureEncoder(
        encoder_name,
        int(config.get("audio_dim", 768)),
        model_name,
        "cpu",
        1,
        10.0,
        6.0,
    )
    feature = encoder.encode_one(waveform, sample_rate=sample_rate, full_audio=False)
    return torch.from_numpy(np.asarray(feature, dtype=np.float32))[None]


@torch.no_grad()
def encode_image(model, path: Path, size: int, dtype: torch.dtype) -> torch.Tensor:
    model._move_frozen_component("vae", model.device)
    try:
        tensor = image_tensor(path, size).to(model.device, dtype=dtype)
        return model.flux_pipe.vae.encode(tensor).latent_dist.sample()
    finally:
        model._move_frozen_component("vae", "cpu")
        torch.cuda.empty_cache()


def load_wsg_array(path: Path, input_contract: Mapping[str, Any] | None = None) -> np.ndarray:
    if input_contract:
        return canonical_array_from_contract(path, input_contract)
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.ndarray):
        array = payload
    else:
        try:
            for key in ("data", "wsg_data", "arr_0"):
                if key in payload.files:
                    array = np.asarray(payload[key])
                    break
            else:
                if len(payload.files) != 1:
                    raise ValueError(
                        "A WSG snapshot archive must contain data, wsg_data, arr_0, "
                        "or exactly one array"
                    )
                array = np.asarray(payload[payload.files[0]])
        finally:
            payload.close()
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"WebUI expects one World Snapshot, but {path.name} contains "
                f"{array.shape[0]} frames"
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected an H×W×C World Snapshot array, got {array.shape}")
    return array


@torch.no_grad()
def encode_wsg_modality(
    model, path: Path, token: str, size: int, dtype: torch.dtype,
    input_contract: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    from src.dataset.wsg_dataset import (
        WSGModalityDataset,
        WSGNormalizationStore,
        effective_cached_normalization_mode,
        load_wsg_normalization_contract,
        load_wsg_normalization_ranges,
        normalization_mode_for_modality,
        random_resize_crop,
        sparse_modalities_from_config,
    )

    configured = getattr(model.args, "modality_config", None) or getattr(
        model.args, "modality_config_path", None
    )
    config_path = Path(resolve_modality_config_path(configured)).expanduser().resolve()
    contract = load_wsg_normalization_contract(config_path)
    ranges = load_wsg_normalization_ranges(config_path)
    cached_ranges = (input_contract or {}).get("training_channel_ranges") or {}
    if isinstance(cached_ranges, Mapping):
        for channel, bounds in cached_ranges.items():
            try:
                ranges[int(channel)] = (float(bounds[0]), float(bounds[1]))
            except (TypeError, ValueError, IndexError):
                continue
    canonical_token = "0-2" if token in RGB_INPUT_TOKENS else token
    recorded_mode = str(
        (input_contract or {}).get("training_normalization_mode") or ""
    ).lower()
    mode = (
        effective_cached_normalization_mode(canonical_token, recorded_mode)
        if recorded_mode
        else normalization_mode_for_modality(contract, canonical_token)
    )
    if mode == "xpose_rgb_minus1_1":
        mode = "linear"
    if mode not in {"linear", "log"}:
        raise ValueError(f"Unsupported training normalization mode {mode!r} for {token}")
    if recorded_mode or cached_ranges:
        log(f"Raw modality {token} cache preprocessing: mode={mode}, ranges={len(cached_ranges)}")
    helper = WSGModalityDataset.__new__(WSGModalityDataset)
    helper.normalization_contract = contract
    helper.portable_normalization_ranges = ranges
    helper.normalization_dataset_paths = []
    helper.normalization_show_progress = False
    helper.normalization = WSGNormalizationStore(
        normalization_mode=mode,
        portable_channel_ranges=ranges,
    )
    helper._normalization_stores_by_mode = {mode: helper.normalization}
    helper.modality_normalization_modes = {canonical_token: mode}
    helper.sparse_modalities = sparse_modalities_from_config(config_path)
    helper.sparse_codec_mode = "disabled"
    helper.sparse_heatmap_sigma = 3.0
    helper.sparse_heatmap_radius = 8
    helper.sparse_heatmap_peak_threshold = 0.10
    helper.sparse_heatmap_nms_radius = 6
    helper.num_frames = 1
    helper.output_dtype = "float32"
    helper.augment = False

    snapshot = load_wsg_array(path, input_contract)
    numeric_channels = parse_numeric_token(canonical_token)
    if numeric_channels is not None:
        if input_contract:
            snapshot = materialize_numeric_wsg(snapshot, input_contract)
        snapshot = random_resize_crop(snapshot, size, crop_mode="center_short_side_square")
        modality = helper._modality_to_video(snapshot, canonical_token)
        image = modality[:, 0].unsqueeze(0).to(model.device, dtype=dtype)
    else:
        snapshot = random_resize_crop(snapshot, size, crop_mode="center_short_side_square")
        encoding = str((input_contract or {}).get("encoding") or "")
        if encoding != "rgb_0_255_to_minus1_1":
            raise ValueError(f"Unsupported string-modality encoding {encoding!r} for {token}")
        if snapshot.shape[-1] != 3:
            raise ValueError(f"String image-like modality {token} requires three channels")
        normalized = np.asarray(snapshot, dtype=np.float32) / 127.5 - 1.0
        image = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(model.device, dtype=dtype)
    model._move_frozen_component("vae", model.device)
    try:
        return model.flux_pipe.vae.encode(image).latent_dist.sample()
    finally:
        model._move_frozen_component("vae", "cpu")
        torch.cuda.empty_cache()


def load_input_latent(model, path: Path, dtype: torch.dtype) -> torch.Tensor:
    """Load an exact cached VAE latent for reproducible Paper Demo inference."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("latent", "latents"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError(f"Cached input latent must be a tensor, got {type(payload)!r}: {path}")
    latent = payload
    if latent.ndim == 4 and latent.shape[0] == 32 and latent.shape[1] == 1:
        latent = latent.unsqueeze(0)
    elif latent.ndim == 3 and latent.shape[0] == 32:
        latent = latent.unsqueeze(0)
    if latent.ndim not in (4, 5) or latent.shape[1] != 32:
        raise ValueError(f"Unexpected cached input latent shape {tuple(latent.shape)}: {path}")
    return latent.to(model.device, dtype=dtype)



def collect_reader_images(reader_dir: Path) -> list[dict[str, str]]:
    labels: dict[Path, str] = {}
    summary_path = reader_dir / "outputs_summary.txt"
    if summary_path.exists():
        for raw_line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line.startswith("- ") or ": " not in line:
                continue
            label, relative = line[2:].rsplit(": ", 1)
            candidate = (reader_dir / relative).resolve()
            labels[candidate] = label.strip()

    records = []
    for path in sorted(reader_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        resolved = path.resolve()
        relative = path.relative_to(reader_dir)
        records.append({
            "label": labels.get(resolved, relative.as_posix()),
            "path": str(resolved),
            "filename": path.name,
        })
    return records


def select_reader_image(records: list[dict[str, str]], target: str) -> dict[str, str]:
    preferred_reader = {
        "32-33": "primedepth",
        "34-35": "metricdepth",
    }.get(target, "")

    def score(record: dict[str, str]) -> tuple[int, int, int, int, str]:
        path = Path(record["path"])
        lowered = path.as_posix().lower()
        name = path.name.lower()
        return (
            int(bool(preferred_reader) and preferred_reader in lowered),
            int(name.endswith("_vis.png")),
            int("overlay" not in name),
            int(name != "summary.png"),
            record["label"],
        )

    return max(records, key=score)


@torch.no_grad()
def generate(module, model, request: dict[str, Any], output_path: Path) -> list[dict[str, str]]:
    dtype = torch.bfloat16
    size = int(request.get("size", 256))
    if size < 128 or size > 1024 or size % 32:
        raise ValueError("size must be a multiple of 32 between 128 and 1024")

    input_records = request.get("inputs") or []
    image_records = [
        item for item in input_records if item.get("kind") in {"image", "wsg"}
    ]
    audio_records = [item for item in input_records if item.get("kind") == "audio"]
    log(f"Preparing {len(image_records)} visual input(s) at {size} × {size}")
    input_seed = int(request.get("seed", model.args.seed))
    torch.manual_seed(input_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(input_seed)
    encoded = {}
    for item in image_records:
        latent_path = item.get("latent_path")
        if latent_path:
            log("Loading exact cached latent for input modality %s" % item["token"])
            encoded[item["token"]] = load_input_latent(model, Path(latent_path), dtype)
        elif item.get("kind") == "wsg":
            log("Encoding raw WSG input modality %s with its training contract" % item["token"])
            encoded[item["token"]] = encode_wsg_modality(
                model, Path(item["path"]), str(item["token"]), size, dtype,
                item.get("input_contract"),
            )
        else:
            log("Encoding RGB input modality %s" % item["token"])
            encoded[item["token"]] = encode_image(model, Path(item["path"]), size, dtype)
    shape_source = next(iter(encoded.values()), None)
    if shape_source is None:
        black = output_path.parent / "_black.png"
        Image.new("RGB", (size, size), color=(0, 0, 0)).save(black)
        shape_source = encode_image(model, black, size, dtype)

    target_clean, target_ids = model.normalize_and_pack(model.flux_pipe, shape_source)
    condition_cleans = []
    condition_ids = []
    reference_clean = None
    reference_ids = None

    reference_tokens = {"Reference_IMG", "CurrentFrame_RGB_Reference"}
    historical_condition_tokens = {
        str(token) for token in request.get("historical_condition_tokens", [])
    }
    for item in input_records:
        token = item.get("token")
        if token not in encoded:
            continue
        packed, ids = model.normalize_and_pack(model.flux_pipe, encoded[token])
        if token in reference_tokens:
            reference_clean = packed
            reference_ids = ids.clone()
            reference_ids[..., 0] = 10
        else:
            if token in historical_condition_tokens:
                ids = ids.clone()
                ids[..., 0] = 10
            condition_cleans.append(packed)
            condition_ids.append(ids)

    prompt = str(request.get("prompt") or "").strip()
    modality_prompt = str(request.get("modality_prompt") or "").strip()
    exact_prompt_only = bool(request.get("exact_prompt_only", False))
    full_prompt = prompt if exact_prompt_only else " ".join(
        item for item in (prompt, modality_prompt) if item
    )
    log(
        "Encoding paper text condition"
        if exact_prompt_only else "Encoding text and modality prompt"
    )
    prompt_embeds, text_ids = model._encode_prompt_offloaded([full_prompt], dtype)

    # Match Flux2TunnelingNetModel.run_inference(): the true-inference noise stream uses seed + 17.
    generator = torch.Generator(device=model.device).manual_seed(
        int(request.get("seed", model.args.seed)) + 17
    )
    latents = torch.randn(target_clean.shape, generator=generator, device=model.device, dtype=dtype)
    steps = max(1, min(50, int(request.get("steps", 10))))
    single_step = bool(getattr(module, "_SINGLE_STEP_X0", False))
    timesteps = (
        torch.ones(1, device=model.device, dtype=dtype)
        if single_step
        else torch.linspace(1.0, 0.0, steps + 1, device=model.device, dtype=dtype)[:-1]
    )
    log(f"Starting denoising with {len(timesteps)} step(s)")
    for index, value in enumerate(timesteps):
        log(f"Denoising step {index + 1}/{len(timesteps)}")
        timestep = torch.full((latents.shape[0],), value, device=model.device, dtype=dtype)
        prediction = module.flux2_tunneling_forward(
            model.transformer,
            latents,
            condition_cleans,
            reference_clean,
            prompt_embeds,
            target_ids,
            condition_ids,
            reference_ids,
            text_ids,
            timestep,
            model.specify_lora,
            False,
        )
        if single_step:
            latents = prediction
        elif index < len(timesteps) - 1:
            latents = latents - float(timesteps[index] - timesteps[index + 1]) * prediction
        else:
            latents = latents - timestep[:, None, None] * prediction

    log("Decoding generated latent into WSG channels")
    raw = model.unpack_normalized_latents(latents, target_ids)
    target = str(request.get("target") or "")
    # Formal evaluation always saves this native pseudo-RGB decode as
    # inference.png. Paper Demo uses it as the default view; Reader-rendered
    # WSG views remain available as additional diagnostics.
    baseline_image = model.decode_raw_latents(raw, target, use_denorm=True)
    baseline_path = output_path.parent / "generated_pseudo_rgb.png"
    baseline_image.save(baseline_path)

    refiner_runtime = getattr(model, "categorical_refiner_runtime", None)
    categorical_refinement = None
    selector_runtime = getattr(model, "audio_region_selector_runtime", None)
    refined_wsg = None
    refiner_records = []
    if refiner_runtime is not None:
        baseline_path = output_path.parent / "generated_pseudo_rgb_unrefined.png"
        baseline_image.save(baseline_path)
        refined_path = output_path.parent / "generated_pseudo_rgb_refined.png"
        aliases = {
            "Reference_IMG": "0-2",
            "FirstFrame_RGB": "0-2",
            "CurrentFrame_RGB_Reference": "0-2",
            "CurrentFrame_RGB_Condition": "0-2",
        }
        encoded_by_modality = {}
        for token, value in encoded.items():
            encoded_by_modality.setdefault(aliases.get(token, token), value)
        required_latents = []
        for token in refiner_runtime.condition_modalities:
            canonical = aliases.get(token, token)
            if canonical not in encoded_by_modality:
                raise RuntimeError(
                    f"Categorical refiner requires input {token!r} ({canonical!r}); "
                    f"available={sorted(encoded_by_modality)}"
                )
            required_latents.append(encoded_by_modality[canonical])
        audio_feature = None
        if selector_runtime is not None:
            if len(audio_records) != 1:
                raise RuntimeError(
                    f"Head 010 requires exactly one WAV input; received {len(audio_records)}"
                )
            log("Encoding frame audio for candidate-mask selection")
            audio_feature = encode_audio_for_refiner(
                selector_runtime, Path(audio_records[0]["path"])
            )
        categorical_refinement = refiner_runtime.refine(
            baseline_image,
            required_latents,
            minimum_component_pixels=int(request.get("categorical_refiner_minimum_pixels", 32)),
            audio_selector=selector_runtime,
            audio_feature=audio_feature,
        )
        categorical_refinement["image"].save(refined_path)
        refined_wsg = refiner_runtime.to_wsg_array(
            categorical_refinement["category_map"]
        )
        metadata_path = output_path.parent / "categorical_refiner_result.json"
        metadata_path.write_text(
            json.dumps(
                {
                    **categorical_refinement["provenance"],
                    "regions": categorical_refinement["regions"],
                    "audio_region_selector": categorical_refinement.get("audio_region_selector"),
                    "baseline_artifact": baseline_path.name,
                    "refined_artifact": refined_path.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        refiner_records = [
            {
                "label": "Generated pseudo-RGB (before category refinement)",
                "path": str(baseline_path.resolve()),
                "filename": baseline_path.name,
                "selected": False,
            },
            {
                "label": "Generated pseudo-RGB (category-refined)",
                "path": str(refined_path.resolve()),
                "filename": refined_path.name,
                "selected": False,
            },
        ]
        log(
            f"Applied categorical refiner to {len(categorical_refinement['regions'])} region(s)"
        )
    try:
        if module.visualize_single_wsg is None:
            raise RuntimeError("WSG reader visualization module is unavailable")
        wsg = refined_wsg if refined_wsg is not None else model.decode_raw_latents_to_wsg_array(raw, target)
        if wsg is None:
            raise RuntimeError(f"Cannot decode target {target} into a WSG array")
        wsg_path = output_path.parent / "generated.wsg.npz"
        reader_dir = output_path.parent / "wsg_reader_outputs"
        np.savez_compressed(wsg_path, wsg_data=wsg)
        log(f"Running WSG reader for target {target}")
        module.visualize_single_wsg(wsg_path, reader_dir)
        records = collect_reader_images(reader_dir)
        if not records:
            raise RuntimeError("WSG reader produced no viewable images")
        selected = select_reader_image(records, target)
        paper_demo = bool(request.get("paper_demo", False))
        paper_output_view = str(request.get("paper_output_view") or "native").lower()
        use_native_paper = paper_demo and paper_output_view != "reader"
        paper_path = (
            output_path.parent / "generated_pseudo_rgb_refined.png"
            if categorical_refinement is not None
            else baseline_path
        )
        paper_record = {
            "label": "Paper inference view (native pseudo-RGB decode)",
            "path": str(paper_path.resolve()),
            "filename": paper_path.name,
            "selected": use_native_paper,
        }
        if use_native_paper:
            shutil.copy2(paper_path, output_path)
            for record in records:
                record["selected"] = False
            log("Selected formal paper inference view")
        else:
            shutil.copy2(selected["path"], output_path)
            for record in records:
                record["selected"] = record["path"] == selected["path"]
            if paper_demo:
                log(f"Selected formal paper WSG reader output: {selected['label']}")
            else:
                log(f"Selected WSG reader output: {selected['label']}")
        log(f"WSG reader produced {len(records)} viewable image(s)")
        return [paper_record] + records + refiner_records
    except Exception as exc:
        log(f"WSG reader failed; using pseudo-RGB fallback: {type(exc).__name__}: {exc}")
        result = (
            categorical_refinement["image"]
            if categorical_refinement is not None
            else model.decode_raw_latents(raw, target, use_denorm=True)
        )
        result.save(output_path)
        return [{
            "label": "Pseudo-RGB fallback",
            "path": str(output_path),
            "filename": output_path.name,
            "selected": True,
        }]



RGB_INPUT_TOKENS = {
    "0-2",
    "Reference_IMG",
    "FirstFrame_RGB",
    "CurrentFrame_RGB_Reference",
    "CurrentFrame_RGB_Condition",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
TENSOR_SUFFIXES = {".wsg", ".npz", ".npy"}
INPUT_KINDS = {"image", "wsg", "audio"}


def _condition_tokens(resolved: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    entry = ((runtime.get("wsg") or {}).get("entry") or {})
    raw = entry.get("condition")
    if raw in (None, ""):
        raw = resolved.get("condition_modalities") or ""
    if isinstance(raw, (list, tuple)):
        return [str(token).strip() for token in raw if str(token).strip()]
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_input_argument(value: str) -> tuple[str, str, Path]:
    label, separator, path_text = str(value).partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise ValueError(
            f"Invalid --input {value!r}; expected TOKEN=PATH or TOKEN:KIND=PATH"
        )
    label = label.strip()
    kind = ""
    if ":" in label:
        token, candidate_kind = label.rsplit(":", 1)
        if candidate_kind not in INPUT_KINDS:
            raise ValueError(
                f"Invalid input kind {candidate_kind!r}; choose image, wsg, or audio"
            )
        label, kind = token.strip(), candidate_kind
    path = Path(path_text.strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, kind, path


def _infer_input_kind(token: str, path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio"
    if token in RGB_INPUT_TOKENS:
        if suffix in IMAGE_SUFFIXES:
            return "image"
        if suffix in TENSOR_SUFFIXES:
            return "wsg"
        raise ValueError(
            f"RGB input {token!r} requires an image or raw WSG tensor; received {path.name}"
        )
    if suffix in TENSOR_SUFFIXES:
        return "wsg"
    if suffix in IMAGE_SUFFIXES:
        raise ValueError(
            f"Non-RGB input {token!r} requires raw .wsg, .npz, or .npy data. "
            "A WSG Reader image is a visualization and cannot be encoded as the condition."
        )
    raise ValueError(f"Cannot infer the input kind for {path}; use TOKEN:KIND=PATH")


def _validated_direct_inputs(
    values: list[str],
    resolved: dict[str, Any],
    runtime: dict[str, Any],
    self_preprocess_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    parsed: dict[str, tuple[str, Path]] = {}
    for value in values:
        token, kind, path = _parse_input_argument(value)
        if token in parsed:
            raise ValueError(f"Input modality {token!r} was supplied more than once")
        parsed[token] = (kind or _infer_input_kind(token, path), path)

    self_preprocessed: dict[str, Path] = {}
    for value in self_preprocess_values or []:
        token, kind, path = _parse_input_argument(value)
        if token in parsed or token in self_preprocessed:
            raise ValueError(f"Input modality {token!r} was supplied more than once")
        if kind and kind != "image":
            raise ValueError(
                f"--self-preprocess {token} expects an RGB image, not kind {kind!r}"
            )
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"--self-preprocess {token} expects a PNG, JPEG, or WebP image"
            )
        self_preprocessed[token] = path

    expected = [token for token in _condition_tokens(resolved, runtime) if token != "caption"]
    missing = [
        token for token in expected
        if token not in parsed and token not in self_preprocessed
    ]
    if missing:
        raise ValueError(
            "Checkpoint requires missing input modality/modalities: " + ", ".join(missing)
        )
    supplied_tokens = list(parsed) + list(self_preprocessed)
    ordered_tokens = expected + [token for token in supplied_tokens if token not in expected]
    config_path = resolve_modality_config_path(resolved.get("modality_config"))
    records: list[dict[str, Any]] = []
    for token in ordered_tokens:
        if token in self_preprocessed:
            path = self_preprocessed[token]
            report = validate_rgb_image(path, "0-2")
            if not report.get("runnable", False):
                reasons = report.get("red_reasons") or ["invalid RGB image"]
                raise ValueError(
                    f"Self-Preprocessor RGB input for {token!r} is not runnable: "
                    + "; ".join(reasons)
                )
            records.append({
                "token": token,
                "kind": "image",
                "path": str(path),
                "self_preprocess": True,
            })
            continue
        kind, path = parsed[token]
        if kind == "image":
            report = validate_rgb_image(path, token)
        elif kind == "audio":
            report = validate_wav_audio(path, token)
        else:
            validation_token = "0-2" if token in RGB_INPUT_TOKENS else token
            report = inspect_tensor_input(path, validation_token, config_path)
        severity = str(report.get("severity") or "red").upper()
        log(f"Input check {severity} {token}: {report.get('summary', '')}")
        for message in report.get("warnings", []):
            log(f"Input warning {token}: {message}")
        red_reasons = list(report.get("red_reasons") or [])
        if not report.get("runnable", False):
            reasons = red_reasons or ["input cannot be mapped to the training contract"]
            raise ValueError(f"Input {token!r} is not runnable: {'; '.join(reasons)}")
        record: dict[str, Any] = {
            "token": token,
            "kind": kind,
            "path": str(path),
        }
        if report.get("contract"):
            record["input_contract"] = report["contract"]
        records.append(record)
    return records


def _build_direct_request(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    # Preserve the logical Hub snapshot hierarchy: Hub checkpoint files are
    # symlinks whose resolved targets live in the flat ``blobs/`` directory.
    checkpoint = Path(args.checkpoint).expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    resolved, runtime, _ = read_run_config(checkpoint)
    target = str(
        resolved.get("target_modality")
        or (((runtime.get("wsg") or {}).get("entry") or {}).get("target"))
        or ""
    )
    if not target:
        raise ValueError("The checkpoint run configuration does not declare a target modality")
    if args.text_to_modality and (args.input or args.self_preprocess):
        raise ValueError(
            "--text-to-modality generates its RGB input and cannot be combined with "
            "--input or --self-preprocess"
        )
    request = {
        "mode": "text_to_modality" if args.text_to_modality else "channel_head",
        "checkpoint": str(checkpoint),
        "inputs": (
            [] if args.text_to_modality else _validated_direct_inputs(
                args.input, resolved, runtime, args.self_preprocess
            )
        ),
        "prompt": str(args.prompt or ""),
        "exact_prompt_only": bool(args.exact_prompt_only),
        "paper_demo": bool(args.paper_demo),
        "target": target,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "size": int(args.size),
        "categorical_refiner": str(args.categorical_refiner),
        "categorical_refiner_minimum_pixels": int(
            args.categorical_refiner_minimum_pixels
        ),
    }
    output_path = _resolve_output_path(args)
    return request, output_path


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    output_dir = Path(args.output_dir or "inference_output").expanduser().resolve()
    return output_dir / "result.png"


def _validated_request_inputs(
    request: Mapping[str, Any],
    resolved: dict[str, Any],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    """Revalidate internal JSON inputs against the checkpoint training contract."""
    values: list[str] = []
    for item in request.get("inputs") or []:
        token = str(item.get("token") or "").strip()
        path = str(item.get("path") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not token or not path:
            raise ValueError("Each request input requires token and path")
        label = f"{token}:{kind}" if kind else token
        values.append(f"{label}={path}")
    validated = _validated_direct_inputs(values, resolved, runtime)
    originals = {str(item.get("token") or ""): item for item in request.get("inputs") or []}
    for record in validated:
        latent_path = str(originals.get(record["token"], {}).get("latent_path") or "")
        if latent_path:
            path = Path(latent_path).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() != ".pt":
                raise FileNotFoundError(f"Cached input latent is missing or invalid: {path}")
            record["latent_path"] = str(path)
    return validated


class ChannelHeadInferenceSession:
    """Keep one TunnelingNet Channel Head loaded for repeated inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        categorical_refiner: str = "auto",
    ) -> None:
        # Preserve Hugging Face's logical snapshot path. Resolving the
        # checkpoint symlink would replace the run hierarchy with blobs/<hash>
        # and make the adjacent resolved configuration undiscoverable.
        self.checkpoint = Path(checkpoint).expanduser().absolute()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

        log(f"Reading run configuration for {self.checkpoint}")
        self.resolved, self.runtime, self.run_dir = read_run_config(self.checkpoint)
        log("Loading WSM training module")
        self.module = load_training_module()
        configure_runtime(self.module, self.resolved, self.runtime)
        self.model_args = make_args(self.resolved, self.runtime)
        log(f"Initializing base model from {self.model_args.model_path}")
        self.model = self.module.Flux2TunnelingNetModel(
            self.model_args,
            {"params": {"lr": 0.0, "weight_decay": 0.0}},
            self.module.specify_lora,
        )
        self.model._device = next(self.model.transformer.parameters()).device

        log("Loading Channel Head checkpoint weights")
        embedded_refiner, self.checkpoint_format = load_weights(
            self.model, self.checkpoint
        )
        resolved_refiner = resolve_categorical_refiner(
            self.checkpoint,
            embedded=embedded_refiner,
            selection=str(categorical_refiner or "auto"),
        )
        if (
            str(self.model_args.target_modality) in CATEGORICAL_TARGET_MODALITIES
            and resolved_refiner is None
        ):
            raise RuntimeError(
                "This categorical output requires a trained categorical refiner. "
                f"Install {channel_head_run_dir(self.checkpoint) / 'categorical_refiner' / 'best.pt'} "
                "or select a combined checkpoint containing the refiner."
            )
        self.model.categorical_refiner_runtime = (
            CategoricalRefinerRuntime(resolved_refiner, self.model.device)
            if resolved_refiner is not None
            else None
        )
        if resolved_refiner is not None:
            log(
                f"Loaded categorical refiner ({resolved_refiner.source_kind}): "
                f"{resolved_refiner.source_path}"
            )
        else:
            log("No categorical refiner selected or available")

        resolved_audio_selector = resolve_audio_region_selector(self.checkpoint)
        adapter_task = str(
            (((self.runtime.get("wsg") or {}).get("entry") or {}).get("adapter_task"))
            or ""
        )
        if (
            adapter_task == "rgb_to_semantic_mask_audio_region_selector"
            and resolved_audio_selector is None
        ):
            raise RuntimeError(
                "Head 010 requires audio_region_selector/best.pt beside its ckpt directory"
            )
        self.model.audio_region_selector_runtime = (
            AudioRegionSelectorRuntime(resolved_audio_selector, self.model.device)
            if resolved_audio_selector is not None
            else None
        )
        if resolved_audio_selector is not None:
            log(f"Loaded audio region selector: {resolved_audio_selector.source_path}")

        self.modality_prompt = (self.runtime.get("train") or {}).get(
            "modality_type_prompt", ""
        )
        first_frame_match = re.search(
            r"first-frame condition modality\s+([^.;]+)",
            str(self.modality_prompt),
            flags=re.IGNORECASE,
        )
        self.historical_condition_tokens = (
            [first_frame_match.group(1).strip()] if first_frame_match else []
        )
        self.model.eval()
        log(f"Checkpoint format: {self.checkpoint_format}")

    @property
    def device(self) -> torch.device:
        return next(self.model.transformer.parameters()).device

    def to(self, device: str | torch.device) -> "ChannelHeadInferenceSession":
        """Move the active tunnel while frozen VAE/text components stay offloaded."""
        resolved = torch.device(device)
        self.model.transformer.to(resolved)
        for name in ("audio_condition_projector", "audio_modulation_projector"):
            component = getattr(self.model, name, None)
            if component is not None:
                component.to(resolved)
        for name in (
            "categorical_refiner_runtime",
            "audio_region_selector_runtime",
        ):
            runtime = getattr(self.model, name, None)
            if runtime is not None:
                runtime.device = resolved
                runtime.model.to(resolved)
        self.model._device = resolved
        return self

    def __call__(
        self,
        request: Mapping[str, Any],
        output_path: str | Path,
    ) -> dict[str, str]:
        payload = dict(request)
        requested_checkpoint = payload.get("checkpoint")
        if requested_checkpoint is not None:
            requested = Path(str(requested_checkpoint)).expanduser().absolute()
            if requested != self.checkpoint:
                raise ValueError(
                    "A ChannelHeadInferenceSession can only serve its loaded checkpoint"
                )
        payload["checkpoint"] = str(self.checkpoint)
        payload["inputs"] = _validated_request_inputs(
            payload, self.resolved, self.runtime
        )
        payload.setdefault("target", self.model_args.target_modality)
        payload["modality_prompt"] = self.modality_prompt
        payload["historical_condition_tokens"] = self.historical_condition_tokens

        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        outputs = list(payload.pop("orchestrator_outputs", [])) + generate(
            self.module, self.model, payload, destination
        )
        manifest_path = destination.parent / "result_manifest.json"
        manifest_path.write_text(
            json.dumps({"outputs": outputs}, indent=2), encoding="utf-8"
        )
        result = {"output": str(destination), "checkpoint": str(self.checkpoint)}
        log("Inference completed")
        print(json.dumps(result), flush=True)
        return result


def run_base_rgb_inference(request: dict[str, Any], output_path: Path) -> dict[str, str]:
    """Run the unmodified FLUX.2 text-to-RGB stage of a composite route."""
    checkpoint = Path(request["checkpoint"]).expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    resolved, runtime, _ = read_run_config(checkpoint)
    model_path = resolve_base_model_path(resolved, runtime)
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Text-to-RGB generation requires a non-empty prompt")
    size = int(request.get("size", 256))
    if size < 128 or size > 1024 or size % 32:
        raise ValueError("size must be a multiple of 32 between 128 and 1024")
    steps = max(1, min(50, int(request.get("steps", 10))))
    seed = int(request.get("seed", 20260812))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"Initializing unmodified base model from {model_path}")
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    log(f"Generating stage-1 RGB at {size} × {size} with {steps} step(s)")
    image = pipeline(
        prompt=prompt,
        height=size,
        width=size,
        num_inference_steps=steps,
        generator=generator,
        output_type="pil",
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    manifest_path = output_path.parent / "result_manifest.json"
    manifest_path.write_text(json.dumps({"outputs": [{
        "label": "Stage 1 · Generated RGB",
        "path": str(output_path.resolve()),
        "filename": output_path.name,
        "selected": True,
    }]}, indent=2), encoding="utf-8")
    log("Stage-1 RGB generation completed")
    return {"output": str(output_path), "checkpoint": str(checkpoint)}


def run_channel_head_inference(
    request: dict[str, Any], output_path: Path,
) -> dict[str, str]:
    checkpoint = Path(request["checkpoint"]).expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    session = ChannelHeadInferenceSession(
        checkpoint,
        categorical_refiner=str(request.get("categorical_refiner") or "auto"),
    )
    return session(request, output_path)

def _released_rgb_preprocessor(token: str) -> dict[str, Any]:
    """Resolve the official RGB estimator for one WSG modality."""
    dense_path = PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json"
    registry_path = (
        PROJECT_ROOT / "configs/wsm_v1_recommended_dense_channel_heads.json"
    )
    dense_document = json.loads(dense_path.read_text(encoding="utf-8"))
    dense = dense_document.get("wsg_v1", dense_document)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    head_indices = []
    for head_name, entry in dense.items():
        match = re.fullmatch(r"Channel_Head_(\d{6})", str(head_name))
        conditions = entry.get("condition") if isinstance(entry, Mapping) else None
        if isinstance(conditions, str):
            conditions = [conditions]
        if (
            match
            and list(conditions or []) == ["0-2"]
            and str(entry.get("target")) == token
        ):
            head_indices.append(int(match.group(1)))
    entries = {
        int(entry["head_index"]): entry
        for entry in registry.get("entries", [])
        if entry.get("head_source") == "dense" and "head_index" in entry
    }
    candidates = []
    for index in head_indices:
        entry = entries.get(index, {})
        candidate = (entry.get("recommended") or {}).get("candidate") or {}
        relative = str(
            candidate.get("checkpoint")
            or (entry.get("evaluated") or {}).get("checkpoint_relative")
            or ""
        )
        checkpoint = (PROJECT_ROOT / relative).absolute() if relative else None
        if checkpoint is not None and checkpoint.is_file():
            candidates.append({
                "id": f"dense:Channel_Head_{index:06d}",
                "source": "dense",
                "head": f"Channel_Head_{index:06d}",
                "index": index,
                "target": token,
                "default_checkpoint": str(checkpoint),
            })
    if not candidates:
        raise ValueError(
            f"No released WSM RGB estimator can self-preprocess modality {token!r}"
        )
    candidates.sort(key=lambda task: int(task["index"]))
    return candidates[0]


def materialize_self_preprocessed_inputs(
    request: dict[str, Any], output_path: Path,
) -> None:
    """Replace RGB proxy uploads with WSM-estimated raw WSG conditions."""
    materialized = []
    intermediate_outputs = list(request.get("orchestrator_outputs", []))
    for index, item in enumerate(request.get("inputs") or []):
        if not bool(item.get("self_preprocess", False)):
            materialized.append(item)
            continue
        token = str(item.get("token") or "").strip()
        rgb_path = Path(str(item.get("path") or "")).expanduser().resolve()
        if not token or not rgb_path.is_file():
            raise ValueError("Each self-preprocessed input requires TOKEN and an RGB image path")
        if rgb_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"WSM Self-Preprocessor for {token} expects a PNG, JPEG, or WebP image"
            )
        rgb_report = validate_rgb_image(rgb_path, "0-2")
        if not rgb_report.get("runnable", False):
            reasons = rgb_report.get("red_reasons") or ["invalid RGB image"]
            raise ValueError(
                f"Self-Preprocessor RGB input for {token} cannot run: "
                + "; ".join(reasons)
            )
        preprocessor = _released_rgb_preprocessor(token)
        stage_dir = output_path.parent / f"self_preprocessor_{index:02d}_{token.replace('/', '_')}"
        stage_output = stage_dir / "result.png"
        log(
            f"WSM Self-Preprocessor: RGB → {token} with {preprocessor['head']}"
        )
        run_channel_head_inference({
            "checkpoint": preprocessor["default_checkpoint"],
            "inputs": [{"token": "0-2", "kind": "image", "path": str(rgb_path)}],
            "prompt": "",
            "exact_prompt_only": False,
            "paper_demo": False,
            "target": token,
            "steps": int(request.get("steps", 10)),
            "seed": int(request.get("seed", 20260812)),
            "size": int(request.get("size", 256)),
        }, stage_output)
        generated_wsg = stage_dir / "generated.wsg.npz"
        if not generated_wsg.is_file():
            raise RuntimeError(
                f"WSM Self-Preprocessor {preprocessor['head']} did not produce raw WSG values"
            )
        materialized.append({
            "token": token,
            "kind": "wsg",
            "path": str(generated_wsg),
        })
        intermediate_outputs.append({
            "label": f"Self-Preprocessor · RGB → {token}",
            "path": str(stage_output.resolve()),
            "filename": stage_output.name,
            "selected": False,
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    request["inputs"] = materialized
    request["orchestrator_outputs"] = intermediate_outputs


def run_text_to_modality_chain(
    request: dict[str, Any], output_path: Path,
) -> dict[str, str]:
    """Run Text → RGB → target as one canonical inference command."""
    checkpoint = Path(request["checkpoint"]).expanduser().absolute()
    resolved, runtime, _ = read_run_config(checkpoint)
    visual_inputs = [
        token for token in _condition_tokens(resolved, runtime) if token != "caption"
    ]
    if len(visual_inputs) != 1 or visual_inputs[0] not in RGB_INPUT_TOKENS:
        raise ValueError(
            "Text → RGB → modality requires a final Head with exactly one RGB input"
        )
    stage_dir = output_path.parent / "text_to_rgb"
    stage_output = stage_dir / "generated_rgb.png"
    prompt = str(request.get("prompt") or "").strip()
    run_base_rgb_inference({
        "checkpoint": str(checkpoint),
        "prompt": prompt,
        "steps": int(request.get("steps", 10)),
        "seed": int(request.get("seed", 20260812)),
        "size": int(request.get("size", 256)),
    }, stage_output)
    final_request = dict(request)
    final_request["mode"] = "channel_head"
    final_request["prompt"] = ""
    final_request["inputs"] = [{
        "token": visual_inputs[0], "kind": "image", "path": str(stage_output),
    }]
    final_request["orchestrator_outputs"] = [{
        "label": "Stage 1 · Generated RGB",
        "path": str(stage_output.resolve()),
        "filename": stage_output.name,
        "selected": False,
    }]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_channel_head_inference(final_request, output_path)


def run_inference(request: dict[str, Any], output_path: Path) -> dict[str, str]:
    mode = str(request.get("mode") or "channel_head")
    if mode == "base_rgb":
        return run_base_rgb_inference(request, output_path)
    if mode == "text_to_modality":
        return run_text_to_modality_chain(request, output_path)
    materialize_self_preprocessed_inputs(request, output_path)
    return run_channel_head_inference(request, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained TunnelingNet Channel Head on raw RGB, WSG, or audio inputs. "
            "Head architecture and routing are restored from the checkpoint run."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help="Channel Head pytorch_lora_weights.safetensors checkpoint.",
    )
    source.add_argument(
        "--request",
        help="Internal JSON request used by ControlWorld and legacy integrations.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="TOKEN=PATH",
        help=(
            "Input modality and path; repeat for multi-input Heads. Kind is inferred, "
            "or specify TOKEN:image=PATH, TOKEN:wsg=PATH, or TOKEN:audio=PATH."
        ),
    )
    parser.add_argument(
        "--self-preprocess",
        action="append",
        default=[],
        metavar="TOKEN=RGB_PATH",
        help=(
            "Use WSM's official RGB estimator to create this non-RGB condition; "
            "repeat for multiple inputs. The generated raw WSG values, not the preview, "
            "are passed to the final Head."
        ),
    )
    parser.add_argument(
        "--text-to-modality",
        "--chain-of-visual",
        action="store_true",
        help="Run the complete Text → RGB → selected target-modality chain.",
    )
    parser.add_argument("--prompt", default="", help="Optional text condition.")
    parser.add_argument(
        "--exact-prompt-only",
        action="store_true",
        help="Do not append the modality prompt stored with the Head.",
    )
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps.")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Square inference size; a multiple of 32 between 128 and 1024.",
    )
    parser.add_argument(
        "--categorical-refiner",
        default="auto",
        help="auto, off, or a categorical-refiner checkpoint/folder.",
    )
    parser.add_argument("--categorical-refiner-minimum-pixels", type=int, default=32)
    parser.add_argument(
        "--paper-demo",
        action="store_true",
        help="Select the native pseudo-RGB view instead of the WSG Reader view.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", help="Final selected output image path.")
    destination.add_argument(
        "--output-dir",
        help="Artifact directory; the selected image is written as result.png.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    log("Inference process started")
    if args.request:
        request_path = Path(args.request).expanduser().resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output_path = _resolve_output_path(args)
    else:
        request, output_path = _build_direct_request(args)
    run_inference(request, output_path)


if __name__ == "__main__":
    main()
