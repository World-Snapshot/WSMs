from __future__ import annotations

import json
from functools import lru_cache
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


from src.pipelines.categorical_refiner import (
    CATEGORICAL_TARGET_MODALITIES,
    audio_region_selector_revision,
    categorical_refiner_revision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DENSE_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json"
RECOMMENDED_DENSE_CONFIG = (
    PROJECT_ROOT / "configs/wsm_v1_recommended_dense_channel_heads.json"
)
SPARSE_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_sparse.json"
CUSTOM_CONFIG = PROJECT_ROOT / "configs/wsg_v1_modalities_custom.json"
SPARSE_LAUNCH_SCRIPT = PROJECT_ROOT / "scripts/launch_scripts/train_sparse_structure_models.sh"
CHECKPOINT_NAME = "pytorch_lora_weights.safetensors"
HEAD_PATTERN = re.compile(r"Channel_Head_(\d{6})$")
STEP_PATTERN = re.compile(r"(?:final_step_|step_|^)(\d+)$")
SPARSE_HEADS_PATTERN = re.compile(r'^CHANNEL_HEADS="\x24\{CHANNEL_HEADS:-([^}]*)\}"', re.MULTILINE)
SPARSE_RAW_INFERENCE_ADAPTERS = {
    "smpl",
    "bbox",
    "dwpose",
    "mediapipe",
    "scene_audio_rgb_to_semantic_instance_localization",
    "scene_audio_to_semantic_category",
    "rgb_to_semantic_mask_audio_region_selector",
}
NON_FILE_INPUTS = {"caption"}
AUDIO_INPUTS = {"175-184"}
REFERENCE_INPUTS = {"Reference_IMG", "CurrentFrame_RGB_Reference"}
SPECIAL_INPUT_NAMES = {
    "caption": "Caption text",
    "Reference_IMG": "First-frame RGB reference",
    "FirstFrame_RGB": "First-frame RGB condition",
    "CurrentFrame_RGB_Reference": "Current-frame RGB reference",
    "CurrentFrame_RGB_Condition": "Current-frame RGB condition",
    "175-184": "Scene audio",
}


def condition_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def condition_signature(entry: dict[str, Any]) -> str:
    return " + ".join(condition_list(entry.get("condition"))) or "caption"


def input_details(
    inputs: list[str],
    modality_names: dict[str, str],
    task_type: str = "",
) -> list[dict[str, str]]:
    visual_inputs = [
        token for token in inputs if token != "caption" and token not in AUDIO_INPUTS
    ]
    historical_rgb_pair = (
        task_type == "inverse_rendering"
        and "0-2" in visual_inputs
        and len(visual_inputs) > 1
    )
    details = []
    for token in inputs:
        name = SPECIAL_INPUT_NAMES.get(token, modality_names.get(token, "Unknown modality"))
        prefix = f"{token}:"
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
        role = "current_frame"
        role_label = "Current frame"
        hint = "Upload this modality for the frame being predicted."
        upload_prompt = "Drop the current-frame image here"
        semantic_name = name

        if token == "caption":
            role = "text"
            role_label = "Text"
            hint = "Text guidance for the requested output."
            upload_prompt = ""
            input_kind = "text"
        elif token in AUDIO_INPUTS:
            role = "audio"
            role_label = "Frame audio"
            hint = "Upload the frame-aligned WAV audio used to select sounding regions."
            upload_prompt = "Drop the frame-aligned WAV audio here"
            semantic_name = name
            input_kind = "audio"
        elif token in {"Reference_IMG", "FirstFrame_RGB"}:
            role = "first_frame"
            role_label = "First frame"
            hint = "Upload an earlier or first-frame RGB image from the same sequence."
            upload_prompt = "Drop the first-frame RGB image here"
            input_kind = "image"
        elif token == "CurrentFrame_RGB_Reference":
            role = "current_frame_reference"
            role_label = "Current frame reference"
            hint = "Upload the RGB reference from the frame being predicted."
            upload_prompt = "Drop the current-frame RGB reference here"
            input_kind = "image"
        elif token == "CurrentFrame_RGB_Condition":
            role = "current_frame_condition"
            role_label = "Current frame condition"
            hint = "Upload the RGB condition from the frame being predicted."
            upload_prompt = "Drop the current-frame RGB condition here"
            input_kind = "image"
        elif historical_rgb_pair and token == "0-2":
            role = "first_frame_or_single_image"
            role_label = "First frame"
            semantic_name = f"{name} (first frame)"
            hint = (
                "Use an earlier or first-frame RGB image from the same sequence. "
                "For a single-image input with no history, use its current-frame RGB."
            )
            upload_prompt = "Drop first-frame RGB here"
            input_kind = "image"
        elif token == "0-2":
            semantic_name = f"{name} (current frame)"
            input_kind = "image"
        else:
            semantic_name = f"{name} (current frame)"
            input_kind = "wsg"

        details.append(
            {
                "token": token,
                "name": name,
                "label": f"{token}: {semantic_name}",
                "role": role,
                "role_label": role_label,
                "hint": hint,
                "upload_prompt": upload_prompt,
                "kind": input_kind,
            }
        )
    return details


def _step_from_checkpoint(path: Path) -> int:
    match = STEP_PATTERN.search(path.parent.name)
    if match:
        return int(match.group(1))
    numbered_model = re.fullmatch(r"model-(?:step_)?(\d+)\.pt", path.name)
    return int(numbered_model.group(1)) if numbered_model else -1


@lru_cache(maxsize=4096)
def _run_checkpoint_quality(
    run_dir_text: str,
) -> tuple[int, int, float | None, tuple[tuple[int, float], ...]]:
    """Return max persisted step and validation-best checkpoint in one run."""
    run_dir = Path(run_dir_text)
    available_steps = set()
    ckpt_root = run_dir / "ckpt"
    if ckpt_root.exists():
        for candidate in ckpt_root.rglob("*"):
            if candidate.is_file() and (
                candidate.name == CHECKPOINT_NAME
                or candidate.name.startswith("model-")
            ):
                step = _step_from_checkpoint(candidate)
                if step >= 0:
                    available_steps.add(step)
    val_pairs: list[tuple[int, float]] = []
    stats_path = run_dir / "stats.npz"
    if stats_path.is_file():
        try:
            with np.load(stats_path, allow_pickle=False) as stats:
                val_steps = np.asarray(stats.get("val_step", []), dtype=np.float64)
                val_losses = np.asarray(stats.get("val_loss", []), dtype=np.float64)
                val_pairs = [
                    (int(round(step)), float(loss))
                    for step, loss in zip(val_steps.tolist(), val_losses.tolist())
                    if np.isfinite(step) and np.isfinite(loss)
                ]
        except (OSError, TypeError, ValueError):
            val_pairs = []
    exact = [item for item in val_pairs if item[0] in available_steps]
    if exact:
        best_step, best_loss = min(exact, key=lambda item: (item[1], -item[0]))
    elif available_steps:
        best_step, best_loss = max(available_steps), None
    else:
        best_step, best_loss = -1, None
    return (
        max(available_steps, default=-1),
        int(best_step),
        best_loss,
        tuple(val_pairs),
    )


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    run_dir: Path
    step: int
    run_step: int
    validation_loss: float | None
    validation_best: bool
    final: bool
    modified: float
    kind: str = "channel_head"
    preference: int = 0

    def as_dict(self) -> dict[str, Any]:
        marker = f"step {self.step}" if self.step >= 0 else self.path.name
        quality = (
            f" | best val {self.validation_loss:.6f}"
            if self.validation_best and self.validation_loss is not None else ""
        )
        return {
            "id": str(self.path),
            "path": str(self.path),
            "run": self.run_dir.name,
            "step": self.step,
            "run_step": self.run_step,
            "validation_loss": self.validation_loss,
            "validation_best": self.validation_best,
            "final": self.final,
            "modified": self.modified,
            "kind": self.kind,
            "label": f"{self.run_dir.name} | {marker}{quality}"
            + (" | final" if self.final else ""),
        }


def _checkpoint(
    path: Path,
    run_dir: Path,
    *,
    kind: str,
    preference: int = 0,
) -> Checkpoint:
    parent_name = path.parent.name
    step = _step_from_checkpoint(path)
    logical_run_dir = run_dir.expanduser().absolute()
    run_step, best_step, _best_loss, val_pairs = _run_checkpoint_quality(
        str(logical_run_dir)
    )
    validation_by_step = dict(val_pairs)
    return Checkpoint(
        path=path.expanduser().absolute(),
        run_dir=logical_run_dir,
        step=step,
        run_step=max(run_step, step),
        validation_loss=validation_by_step.get(step),
        validation_best=step >= 0 and step == best_step,
        final=parent_name.startswith("final_step_"),
        modified=path.stat().st_mtime,
        kind=kind,
        preference=preference,
    )

def _iter_dense_checkpoints(head_dir: Path) -> Iterable[Checkpoint]:
    if not head_dir.exists():
        return
    for path in head_dir.glob(f"*/ckpt/*/{CHECKPOINT_NAME}"):
        run_dir = path.parents[2]
        yield _checkpoint(path, run_dir, kind="channel_head")


def _sparse_default_heads() -> set[str]:
    text = SPARSE_LAUNCH_SCRIPT.read_text(encoding="utf-8")
    match = SPARSE_HEADS_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Cannot read default CHANNEL_HEADS from {SPARSE_LAUNCH_SCRIPT}")
    return {
        f"Channel_Head_{int(raw.strip()):06d}"
        for raw in match.group(1).split(",")
        if raw.strip()
    }


def _run_matches_sparse_entry(run_dir: Path, entry: dict[str, Any]) -> bool:
    runtime_path = run_dir / "resolved_runtime_config.json"
    if not runtime_path.exists():
        return True
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        recorded = runtime.get("wsg", {}).get("entry", {})
    except (OSError, ValueError, TypeError):
        return False
    keys = ("adapter_task", "task_type", "condition", "target", "condition_preprocess")
    return all(recorded.get(key) == entry.get(key) for key in keys)


def _iter_sparse_checkpoints(
    head_dir: Path,
    entry: dict[str, Any],
) -> Iterable[Checkpoint]:
    if not head_dir.exists():
        return
    adapter = str(entry.get("adapter_task") or "")
    if adapter == "smpl":
        for path in head_dir.glob("*/scorehmr/checkpoints/run/model-*.pt"):
            run_dir = path.parents[3]
            yield _checkpoint(path, run_dir, kind="scorehmr")
        return
    if adapter in {
        "scene_audio_to_semantic_mask_dense",
        "rgb_audio_to_semantic_mask_dense_prototype_ce",
        "rgb_to_semantic_mask_audio_region_selector",
    }:
        for path in head_dir.glob(f"*/ckpt/*/{CHECKPOINT_NAME}"):
            run_dir = path.parents[2]
            if _run_matches_sparse_entry(run_dir, entry):
                yield _checkpoint(path, run_dir, kind="channel_head")
        return
    preferences = {
        "model-best.pt": 3,
        "model-latest.pt": 2,
        "model-last.pt": 1,
    }
    for path in head_dir.glob("*/ckpt/model-*.pt"):
        run_dir = path.parents[1]
        if _run_matches_sparse_entry(run_dir, entry):
            yield _checkpoint(
                path,
                run_dir,
                kind="sparse_adapter",
                preference=preferences.get(path.name, 0),
            )


def _checkpoint_sort_key(checkpoint: Checkpoint) -> tuple[Any, ...]:
    # First choose the intended run by its furthest persisted training step;
    # timestamp only breaks run-level ties.  Only then choose that run's
    # validation-best checkpoint and, independently, its refiner best.pt.
    validation_score = (
        -float(checkpoint.validation_loss)
        if checkpoint.validation_loss is not None else float("-inf")
    )
    return (
        checkpoint.run_step,
        checkpoint.run_dir.name,
        int(checkpoint.validation_best),
        validation_score,
        checkpoint.preference,
        checkpoint.step,
        checkpoint.path.parent.name,
    )


def _checkpoint_modality_prompt(checkpoint: Path | None) -> str:
    """Read the exact automatic task-semantic prompt stored by training."""
    if checkpoint is None:
        return ""
    # Keep the logical Hugging Face snapshot path.  Checkpoint files in the Hub
    # cache are symlinks into ``blobs/``; resolving one here discards the run
    # directory that contains the associated runtime configuration.
    current = checkpoint.expanduser().absolute().parent
    for _ in range(5):
        runtime_path = current / "resolved_runtime_config.json"
        if runtime_path.is_file():
            try:
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                return str((runtime.get("train") or {}).get("modality_type_prompt") or "")
            except (OSError, ValueError, TypeError):
                return ""
        if current.parent == current:
            break
        current = current.parent
    return ""


def _load_entries(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("wsg_v1")
    if not isinstance(entries, dict):
        raise ValueError(f"{config_path} has no wsg_v1 task mapping")
    return entries


def _configured_recommended_checkpoint(entry: dict[str, Any]) -> Path | None:
    """Resolve a sparse/custom Head config-local deployment choice."""
    checkpoint_text = str(entry.get("recommended_checkpoint") or "").strip()
    if not checkpoint_text:
        return None
    checkpoint = Path(checkpoint_text).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    return checkpoint.absolute()


@lru_cache(maxsize=1)
def _official_recommendations(
) -> tuple[dict[tuple[str, int], Path], dict[tuple[str, int], dict[str, Any]]]:
    """Load deployment choices, keyed by Head family and index."""
    if not RECOMMENDED_DENSE_CONFIG.is_file():
        return {}, {}
    payload = json.loads(RECOMMENDED_DENSE_CONFIG.read_text(encoding="utf-8"))
    default_source = str(payload.get("head_source_default") or "dense")
    preferred: dict[tuple[str, int], Path] = {}
    registry: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_entry in payload.get("entries") or []:
        entry = dict(raw_entry)
        source = str(entry.get("head_source") or default_source)
        index = int(entry["head_index"])
        key = (source, index)
        registry[key] = entry
        candidate = dict((entry.get("recommended") or {}).get("candidate") or {})
        checkpoint_text = str(candidate.get("checkpoint") or "")
        if not checkpoint_text:
            continue
        checkpoint = Path(checkpoint_text)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        checkpoint = checkpoint.absolute()
        if checkpoint.is_file():
            preferred[key] = checkpoint
    return preferred, registry


def _dense_preferred_checkpoints() -> dict[int, Path]:
    """Prefer frozen dense recommendations; discover only as a fallback."""
    all_preferred, all_registry = _official_recommendations()
    preferred = {
        index: path
        for (source, index), path in all_preferred.items()
        if source == "dense"
    }
    registry = {
        index: entry
        for (source, index), entry in all_registry.items()
        if source == "dense"
    }
    if preferred and len(preferred) == len(registry):
        return preferred
    from src.evaluation.internal.channel_head_evaluator import discover_candidates

    candidates, _audit = discover_candidates(
        [PROJECT_ROOT / "train_log"], selection_policy="canonical-coverage"
    )
    discovered = {
        int(candidate.head_index): Path(candidate.checkpoint).expanduser().absolute()
        for candidate in candidates
    }
    discovered.update(preferred)
    return discovered


def _task_record(
    source: str,
    head_name: str,
    entry: dict[str, Any],
    checkpoints: list[Checkpoint],
    modality_names: dict[str, str],
    preferred_checkpoint: Path | None = None,
    recommended_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = condition_list(entry.get("condition"))
    details = input_details(inputs, modality_names, str(entry.get("task_type") or ""))
    sparse_adapter = str(entry.get("adapter_task") or "")
    target = str(entry.get("target") or "")
    categorical_refiner_required = (
        target in CATEGORICAL_TARGET_MODALITIES
        and sparse_adapter != "scene_audio_to_semantic_category"
    )
    audio_selector_required = sparse_adapter == "rgb_to_semantic_mask_audio_region_selector"
    base_checkpoint_count = len(checkpoints)
    preferred_checkpoint = (
        preferred_checkpoint.expanduser().absolute()
        if preferred_checkpoint is not None else None
    )
    if preferred_checkpoint is not None:
        checkpoints = sorted(
            checkpoints,
            key=lambda item: item.path == preferred_checkpoint,
            reverse=True,
        )
    preferred_base = next(
        (item for item in checkpoints if item.path == preferred_checkpoint), None
    )
    configured_checkpoint_missing = (
        preferred_checkpoint is not None and preferred_base is None
    )
    preferred_refiner_available = (
        preferred_base is not None
        and categorical_refiner_required
        and categorical_refiner_revision(preferred_base.path) is not None
    )
    preferred_selector_available = (
        preferred_base is not None
        and audio_selector_required
        and audio_region_selector_revision(preferred_base.path) is not None
    )
    if categorical_refiner_required:
        checkpoints = [
            item for item in checkpoints
            if categorical_refiner_revision(item.path) is not None
        ]
    if audio_selector_required:
        checkpoints = [
            item for item in checkpoints
            if audio_region_selector_revision(item.path) is not None
        ]
    trained = base_checkpoint_count > 0
    categorical_refiner_available = (
        preferred_refiner_available
        if preferred_checkpoint is not None else bool(checkpoints)
    ) if categorical_refiner_required else False
    audio_selector_available = (
        preferred_selector_available
        if preferred_checkpoint is not None else bool(checkpoints)
    ) if audio_selector_required else False
    supported = (
        source in {"dense", "custom"}
        or sparse_adapter in SPARSE_RAW_INFERENCE_ADAPTERS
    ) and trained
    reason = ""
    if source == "sparse" and not supported:
        reason = "This sparse task has trained weights, but its input codec and WebUI inference adapter are not connected yet."
    if configured_checkpoint_missing:
        supported = False
        reason = (
            "The recommended_checkpoint declared by this Head is not available locally: "
            f"{preferred_checkpoint}"
        )
    if categorical_refiner_required and trained and not categorical_refiner_available:
        supported = False
        reason = (
            "The selected default run has base weights but no deployed categorical refiner. "
            "Install categorical_refiner/best.pt beside ckpt in that run."
        )
    if audio_selector_required and trained and not audio_selector_available:
        supported = False
        reason = (
            "The selected Head 010 run has no deployed audio selector. "
            "Install audio_region_selector/best.pt beside ckpt in that run."
        )
    if not trained:
        supported = False
        reason = "No compatible Channel Head checkpoint was found."
    default_checkpoint = ""
    if preferred_checkpoint is not None:
        if any(item.path == preferred_checkpoint for item in checkpoints):
            default_checkpoint = str(preferred_checkpoint)
    elif checkpoints:
        default_checkpoint = str(checkpoints[0].path)
    modality_prompt = _checkpoint_modality_prompt(
        Path(default_checkpoint) if default_checkpoint else None
    )
    quality_notice = dict((recommended_entry or {}).get("quality_notice") or {})
    checkpoint_records = []
    for item in checkpoints:
        checkpoint_record = item.as_dict()
        is_official = (
            preferred_checkpoint is not None
            and item.path == preferred_checkpoint
        )
        checkpoint_record["official_recommended"] = is_official
        if is_official:
            marker = "⚠ Recommended" if quality_notice else "★ Recommended"
            checkpoint_record["label"] = (
                f"{marker} · {checkpoint_record['label']}"
            )
        checkpoint_records.append(checkpoint_record)
    return {
        "id": f"{source}:{head_name}",
        "source": source,
        "head": head_name,
        "index": int(HEAD_PATTERN.match(head_name).group(1)),
        "task_type": str(entry.get("task_type") or ""),
        "pair_exists": bool(entry.get("pair_exists", False)),
        "inputs": inputs,
        "input_details": details,
        "input_signature": " + ".join(inputs) or "caption",
        "input_description": str(entry.get("condition_description") or ""),
        "modality_prompt": modality_prompt,
        "target": target,
        "target_description": str(entry.get("target_description") or ""),
        "adapter_task": sparse_adapter,
        "trained": trained,
        "base_checkpoint_count": base_checkpoint_count,
        "categorical_refiner_required": categorical_refiner_required,
        "categorical_refiner_available": categorical_refiner_available,
        "supported": supported,
        "unsupported_reason": reason,
        "official_recommended": recommended_entry is not None,
        "recommendation_status": str(
            (recommended_entry or {}).get("recommendation_status") or ""
        ),
        "quality_notice": quality_notice or None,
        "checkpoints": checkpoint_records,
        "default_checkpoint": default_checkpoint,
    }


def build_catalog() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    specs = (
        ("dense", DENSE_CONFIG, PROJECT_ROOT / "train_log"),
        ("sparse", SPARSE_CONFIG, PROJECT_ROOT / "train_log_sparse"),
        ("custom", CUSTOM_CONFIG, PROJECT_ROOT / "train_log_custom"),
    )
    loaded_specs = [(source, _load_entries(config_path), log_root) for source, config_path, log_root in specs]
    dense_preferred = _dense_preferred_checkpoints()
    official_paths, official_registry = _official_recommendations()
    sparse_default_heads = _sparse_default_heads()
    modality_names: dict[str, str] = {}
    for _, entries, _ in loaded_specs:
        for entry in entries.values():
            target = str(entry.get("target") or "")
            description = str(entry.get("target_description") or "").strip()
            if target and description:
                modality_names.setdefault(target, description)
            conditions = condition_list(entry.get("condition"))
            condition_description = str(entry.get("condition_description") or "").strip()
            channel_conditions = [token for token in conditions if token not in SPECIAL_INPUT_NAMES]
            for token in channel_conditions:
                matches = [part.strip() for part in condition_description.split("|") if part.strip().startswith(f"{token}:")]
                if matches:
                    modality_names.setdefault(token, matches[0])
                elif len(channel_conditions) == 1 and condition_description:
                    modality_names.setdefault(token, condition_description)

    for source, entries, log_root in loaded_specs:
        for head_name, entry in entries.items():
            if source == "sparse" and head_name not in sparse_default_heads:
                continue
            match = HEAD_PATTERN.match(head_name)
            if not match:
                continue
            checkpoint_iter = (
                _iter_dense_checkpoints(log_root / head_name)
                if source in {"dense", "custom"}
                else _iter_sparse_checkpoints(log_root / head_name, entry)
            )
            checkpoints = sorted(
                checkpoint_iter,
                key=_checkpoint_sort_key,
                reverse=True,
            )
            tasks.append(
                _task_record(
                    source, head_name, entry, checkpoints, modality_names,
                    (
                        dense_preferred.get(int(match.group(1)))
                        if source == "dense"
                        else _configured_recommended_checkpoint(entry)
                    ),
                    (
                        official_registry.get((source, int(match.group(1))))
                        if source == "dense"
                        else (
                            {"recommendation_status": "config_local"}
                            if entry.get("recommended_checkpoint") else None
                        )
                    ),
                )
            )

    source_order = {"dense": 0, "custom": 1, "sparse": 2}
    tasks.sort(key=lambda item: (source_order.get(item["source"], 9), item["index"]))
    available = [task for task in tasks if task["supported"]]
    trained = [task for task in tasks if task["trained"]]
    signatures: dict[str, dict[str, Any]] = {}
    for task in tasks:
        signature = task["input_signature"]
        item = signatures.setdefault(
            signature,
            {
                "id": signature,
                "label": " + ".join(detail["label"] for detail in task["input_details"]),
                "inputs": task["inputs"],
                "input_details": task["input_details"],
                "configured_count": 0,
                "available_count": 0,
            },
        )
        item["configured_count"] += 1
        item["available_count"] += int(task["supported"])

    return {
        "project": "WSM_v1",
        "tasks": tasks,
        "input_signatures": sorted(signatures.values(), key=lambda item: item["label"]),
        "counts": {
            "configured": len(tasks),
            "available": len(available),
            "trained": len(trained),
            "inference_available": len(available),
            "dense_available": sum(task["supported"] and task["source"] == "dense" for task in tasks),
            "sparse_available": sum(task["supported"] and task["source"] == "sparse" for task in tasks),
            "custom_available": sum(task["supported"] and task["source"] == "custom" for task in tasks),
            "dense_trained": sum(task["trained"] and task["source"] == "dense" for task in tasks),
            "sparse_trained": sum(task["trained"] and task["source"] == "sparse" for task in tasks),
            "custom_trained": sum(task["trained"] and task["source"] == "custom" for task in tasks),
            "official_recommended": sum(
                bool(task["official_recommended"]) for task in tasks
            ),
            "quality_needs_improvement": sum(
                bool(task["quality_notice"]) for task in tasks
            ),
        },
    }


def resolve_task(catalog: dict[str, Any], task_id: str, checkpoint_path: str) -> dict[str, Any]:
    task = next((item for item in catalog["tasks"] if item["id"] == task_id), None)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    if not task["supported"]:
        raise ValueError(task["unsupported_reason"] or f"{task_id} is not available")
    allowed = {item["path"] for item in task["checkpoints"]}
    selected = checkpoint_path or task["default_checkpoint"]
    if selected not in allowed:
        raise ValueError("Selected checkpoint is not registered for this Channel Head")
    return task
