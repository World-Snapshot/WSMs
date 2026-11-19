#!/usr/bin/env python
"""TunnelingNet training entry for FLUX.2 4B dense structure models."""
from __future__ import annotations

import argparse
import bisect
from collections import OrderedDict
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Keep startup logs readable in tmux; this must run before importing diffusers.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from diffusers.models.embeddings import apply_rotary_emb
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.checkpoint import checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader, default_collate

from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    categorical_ids_to_rgb,
    categorical_rgb_to_ids,
)
try:
    from wsg_reader.pipeline16_read_psy_from_wsg import (
        VISUALIZATION_CHANNEL_RANGES as READER_VISUALIZATION_CHANNEL_RANGES,
        VISUALIZATION_SPECS as READER_VISUALIZATION_SPECS,
        apply_continuous_palette as reader_apply_continuous_palette,
        normalize_scalar_for_display as reader_normalize_scalar_for_display,
        refine_reader_valid_mask,
    )
except Exception:
    READER_VISUALIZATION_CHANNEL_RANGES = {}
    READER_VISUALIZATION_SPECS = {}
    reader_apply_continuous_palette = None
    reader_normalize_scalar_for_display = None
    refine_reader_valid_mask = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEGACY_MODALITY_CONFIG_NAMES = {
    "wsg_v1_modalities_dense_single_condition_channel_head_condition_first.json":
        "wsg_v1_modalities_dense.json",
    "wsg_v1_modalities_sparse_single_condition_channel_head_condition_first.json":
        "wsg_v1_modalities_sparse.json",
    "wsg_v1_modalities_custom_single_condition_channel_head_condition_first.json":
        "wsg_v1_modalities_custom.json",
}

from src.pipelines.categorical_refiner import (
    AudioRegionSelectorRuntime,
    CategoricalRefinerRuntime,
    CATEGORICAL_TARGET_MODALITIES,
    load_channel_head_checkpoint,
    resolve_audio_region_selector,
    resolve_categorical_refiner,
)

try:
    from src.visualization.single_wsg_visualization import visualize_single_wsg
except Exception:
    visualize_single_wsg = None
try:
    from src.dataset.wsg_dataset import (
        WSGNormalizationStore,
        load_wsg_normalization_contract,
        load_wsg_normalization_ranges,
        normalization_mode_for_modality,
    )
except Exception:
    WSGNormalizationStore = None
    load_wsg_normalization_contract = None
    load_wsg_normalization_ranges = None
    normalization_mode_for_modality = None

DEFAULT_PREVIEW_SAMPLE_ID = os.environ.get("WSM_PREVIEW_SAMPLE_ID", "")
DEFAULT_PREVIEW_VIDEO_ID = ""
DEFAULT_PREVIEW_FRAME_INDEX = 29


class WSGReaderNoPrimaryOutput(RuntimeError):
    """The generated sample has no valid primary Reader visualization."""


try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass
try:
    from diffusers.utils import logging as diffusers_logging
    diffusers_logging.set_verbosity_error()
except Exception:
    pass


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    replacement = LEGACY_MODALITY_CONFIG_NAMES.get(path.name)
    if replacement:
        return PROJECT_ROOT / "configs" / replacement
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_modality_name(value: str) -> str:
    return str(value).replace("-", "_")

def focal_regression_weights(prediction: torch.Tensor, target: torch.Tensor, gamma: float,
                             minimum: float = 0.25, maximum: float = 4.0) -> torch.Tensor:
    """Emphasize relatively hard latent tokens without differentiating the weights."""
    error = (prediction.float() - target.float()).pow(2).mean(dim=-1, keepdim=True)
    relative = error.detach() / error.detach().mean(dim=1, keepdim=True).clamp_min(1e-8)
    weights = relative.clamp(minimum, maximum).pow(float(gamma))
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)


def gradient_domain_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on horizontal and vertical latent finite differences."""
    prediction = prediction.float()
    target = target.float()
    loss_x = F.l1_loss(prediction[..., :, 1:] - prediction[..., :, :-1],
                       target[..., :, 1:] - target[..., :, :-1])
    loss_y = F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :],
                       target[..., 1:, :] - target[..., :-1, :])
    return 0.5 * (loss_x + loss_y)


def latent_foreground_iou_loss(prediction: torch.Tensor, target: torch.Tensor,
                               foreground_threshold: float = 0.1) -> torch.Tensor:
    """Estimate black-background foreground from the target border and compute soft IoU."""
    prediction = prediction.float()
    target = target.float()
    border = torch.cat(
        (target[..., 0, :], target[..., -1, :], target[..., 1:-1, 0], target[..., 1:-1, -1]),
        dim=-1,
    )
    background = border.mean(dim=-1, keepdim=True).unsqueeze(-1).detach()
    target_distance = (target - background).abs().mean(dim=1)
    prediction_distance = (prediction - background).abs().mean(dim=1)
    scale = target_distance.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6).detach()
    cutoff = scale * float(foreground_threshold)
    target_mask = (target_distance / cutoff).clamp(0.0, 1.0).detach()
    prediction_mask = (prediction_distance / cutoff).clamp(0.0, 1.0)
    intersection = (prediction_mask * target_mask).sum(dim=(-2, -1))
    union = (prediction_mask + target_mask - prediction_mask * target_mask).sum(dim=(-2, -1))
    return (1.0 - (intersection + 1e-6) / (union + 1e-6)).mean()

def categorical_latent_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.25,
    margin: float = 0.25,
    num_clusters: int = 8,
    kmeans_iterations: int = 4,
) -> dict[str, torch.Tensor]:
    """Cluster each target latent online and balance losses across its regions.

    A uniform RGB background is not spatially uniform after Flux VAE encoding, so
    foreground detection from a single border prototype is invalid. Deterministic
    target-only k-means avoids that assumption and also supports multiple semantic
    regions instead of reducing a categorical map to foreground/background.
    """
    prediction = prediction.float()
    target = target.float().detach()
    batch, channels, _, _ = target.shape
    prediction_flat = prediction.reshape(batch, channels, -1).transpose(1, 2)
    target_flat = target.reshape(batch, channels, -1).transpose(1, 2)
    token_count = target_flat.shape[1]
    cluster_count = max(2, min(int(num_clusters), token_count))

    feature_mean = target_flat.mean(dim=1, keepdim=True)
    feature_scale = target_flat.std(dim=1, keepdim=True).clamp_min(1e-4)
    target_normalized = (target_flat - feature_mean) / feature_scale
    prediction_normalized = (prediction_flat - feature_mean) / feature_scale

    # Deterministic farthest-point initialization avoids duplicate centroids and
    # keeps otherwise identical ablations reproducible.
    centered = target_normalized - target_normalized.mean(dim=1, keepdim=True)
    first_indices = centered.pow(2).mean(dim=-1).argmax(dim=1)
    batch_indices = torch.arange(batch, device=target.device)
    centroids = target_normalized[batch_indices, first_indices].unsqueeze(1)
    minimum_distance = (target_normalized - centroids[:, :1]).pow(2).mean(dim=-1)
    for _ in range(1, cluster_count):
        next_indices = minimum_distance.argmax(dim=1)
        next_centroid = target_normalized[batch_indices, next_indices].unsqueeze(1)
        centroids = torch.cat((centroids, next_centroid), dim=1)
        next_distance = (target_normalized - next_centroid).pow(2).mean(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, next_distance)

    for _ in range(max(1, int(kmeans_iterations))):
        target_distances = (target_normalized[:, :, None] - centroids[:, None]).pow(2).mean(dim=-1)
        labels = target_distances.argmin(dim=-1)
        assignments = F.one_hot(labels, cluster_count).to(target_normalized.dtype)
        counts = assignments.sum(dim=1)
        updated = torch.einsum("bnk,bnc->bkc", assignments, target_normalized)
        updated = updated / counts.clamp_min(1.0).unsqueeze(-1)
        centroids = torch.where((counts > 0).unsqueeze(-1), updated, centroids)

    target_distances = (target_normalized[:, :, None] - centroids[:, None]).pow(2).mean(dim=-1)
    labels = target_distances.argmin(dim=-1)
    assignments = F.one_hot(labels, cluster_count).to(target_normalized.dtype)
    counts = assignments.sum(dim=1)
    active = counts > 0
    active_count = active.sum(dim=1).clamp_min(1)
    # Recompute prototypes for the final fixed assignment so an exact target
    # prediction is a true zero-loss fixed point.
    centroids = torch.einsum("bnk,bnc->bkc", assignments, target_normalized)
    centroids = centroids / counts.clamp_min(1.0).unsqueeze(-1)
    target_distances = (target_normalized[:, :, None] - centroids[:, None]).pow(2).mean(dim=-1)

    prediction_distances = (prediction_normalized[:, :, None] - centroids[:, None]).pow(2).mean(dim=-1)
    inverse_temperature = 1.0 / max(float(temperature), 1e-4)
    target_probabilities = (-target_distances * inverse_temperature).softmax(dim=-1).detach()
    prediction_log_probabilities = (-prediction_distances * inverse_temperature).log_softmax(dim=-1)
    token_ce = F.kl_div(
        prediction_log_probabilities, target_probabilities, reduction="none"
    ).sum(dim=-1)
    cluster_ce = torch.einsum("bn,bnk->bk", token_ce, assignments) / counts.clamp_min(1.0)
    prototype_ce = ((cluster_ce * active).sum(dim=1) / active_count).mean()

    token_error = (prediction_flat - target_flat).pow(2).mean(dim=-1)
    cluster_error = torch.einsum("bn,bnk->bk", token_error, assignments) / counts.clamp_min(1.0)
    balanced = ((cluster_error * active).sum(dim=1) / active_count).mean()

    target_positive = target_distances.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_negative = target_distances.masked_fill(assignments.bool(), torch.inf).amin(dim=-1)
    prediction_positive = prediction_distances.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    prediction_negative = prediction_distances.masked_fill(assignments.bool(), torch.inf).amin(dim=-1)
    target_separation = (target_negative - target_positive).detach()
    prediction_separation = prediction_negative - prediction_positive
    token_margin = F.smooth_l1_loss(
        prediction_separation, target_separation, reduction="none", beta=max(float(margin), 1e-4)
    )
    cluster_margin = torch.einsum("bn,bnk->bk", token_margin, assignments) / counts.clamp_min(1.0)
    margin_loss = ((cluster_margin * active).sum(dim=1) / active_count).mean()

    prediction_cluster_means = torch.einsum("bnk,bnc->bkc", assignments, prediction_normalized)
    prediction_cluster_means = prediction_cluster_means / counts.clamp_min(1.0).unsqueeze(-1)
    cluster_alignment = (prediction_cluster_means - centroids).pow(2).mean(dim=-1)
    global_loss = ((cluster_alignment * active).sum(dim=1) / active_count).mean()

    prediction_probabilities = prediction_log_probabilities.exp()
    intersection = torch.minimum(prediction_probabilities, target_probabilities).sum(dim=1)
    union = torch.maximum(prediction_probabilities, target_probabilities).sum(dim=1)
    cluster_iou = 1.0 - (intersection + 1e-6) / (union + 1e-6)
    iou_loss = ((cluster_iou * active).sum(dim=1) / active_count).mean()

    fractions = counts / float(token_count)
    minimum_fraction = fractions.masked_fill(~active, 1.0).min(dim=1).values

    return {
        "prototype_ce": prototype_ce,
        "balanced": balanced,
        "margin": margin_loss,
        "global": global_loss,
        "iou": iou_loss,
        "active_clusters": active_count.float().mean(),
        "min_cluster_fraction": minimum_fraction.mean(),
        "max_cluster_fraction": fractions.max(dim=1).values.mean(),
    }


def dense_shard_path(root: Path, modality: str, shard_id: int, num_shards: int) -> Path:
    return root / "shards" / safe_modality_name(modality) / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"


def discover_dense_num_shards(root: Path, modality: str) -> int:
    shard_dir = root / "shards" / safe_modality_name(modality)
    candidates = sorted(shard_dir.glob("shard_*_of_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No dense latent shards found for modality={modality!r} under {shard_dir}")
    suffix = candidates[0].stem.split("_of_", 1)[-1]
    return int(suffix)


class MMapShardedRows:
    """Index rows across mmap-backed tensors without concatenating the shards."""

    def __init__(self, tensors: list[torch.Tensor]):
        if not tensors:
            raise ValueError("MMapShardedRows requires at least one tensor")
        trailing_shape = tuple(tensors[0].shape[1:])
        dtype = tensors[0].dtype
        cumulative = 0
        self.tensors = tensors
        self.ends = []
        for tensor in tensors:
            if tuple(tensor.shape[1:]) != trailing_shape or tensor.dtype != dtype:
                raise ValueError("Mmap tensor shards must have matching shape and dtype")
            cumulative += int(tensor.shape[0])
            self.ends.append(cumulative)
        self.shape = (cumulative, *trailing_shape)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor) and index.ndim == 0:
            index = int(index.item())
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return torch.stack([self[row] for row in range(start, stop, step)])
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        shard_start = self.ends[shard_index - 1] if shard_index else 0
        return self.tensors[shard_index][index - shard_start]


class LazyMMapShardedRows:
    """Open mmap-backed latent shards on demand with a bounded per-process cache."""

    def __init__(self, paths: list[Path], lengths: list[int], trailing_shape: tuple[int, ...], dtype: torch.dtype, max_open_shards: int = 1):
        if not paths or len(paths) != len(lengths):
            raise ValueError("LazyMMapShardedRows requires matching shard paths and lengths")
        self.paths = [Path(path) for path in paths]
        self.ends = []
        cumulative = 0
        for length in lengths:
            cumulative += int(length)
            self.ends.append(cumulative)
        self.shape = (cumulative, *trailing_shape)
        self.dtype = dtype
        self.max_open_shards = max(1, int(max_open_shards))
        self._tensors: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return self.shape[0]

    def _tensor_for_shard(self, shard_index: int) -> torch.Tensor:
        tensor = self._tensors.pop(shard_index, None)
        if tensor is None:
            payload = torch.load(self.paths[shard_index], map_location="cpu", weights_only=False, mmap=True)
            tensor = payload["latents"]
            del payload
        self._tensors[shard_index] = tensor
        while len(self._tensors) > self.max_open_shards:
            self._tensors.popitem(last=False)
        return tensor

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor) and index.ndim == 0:
            index = int(index.item())
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return torch.stack([self[row] for row in range(start, stop, step)])
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        shard_start = self.ends[shard_index - 1] if shard_index else 0
        return self._tensor_for_shard(shard_index)[index - shard_start].clone()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tensors"] = OrderedDict()
        return state


def merge_dense_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if len(payloads) == 1:
        return payloads[0]
    merged = dict(payloads[0])
    keys = set().union(*(payload.keys() for payload in payloads))
    for key in keys:
        values = [payload.get(key) for payload in payloads]
        if key == "latents" and all(torch.is_tensor(value) for value in values):
            merged[key] = MMapShardedRows(values)
        elif all(torch.is_tensor(value) for value in values):
            merged[key] = torch.cat(values, dim=0)
        elif all(isinstance(value, list) for value in values):
            out = []
            for value in values:
                out.extend(value)
            merged[key] = out
    if "latents" in merged:
        merged["num_samples"] = int(merged["latents"].shape[0])
    return merged


def load_dense_payload(root: Path, modality: str | None, train_rank: int, train_world_size: int, lazy_latents: bool = False) -> dict[str, Any] | None:
    if not modality:
        return None
    physical_num_shards = discover_dense_num_shards(root, modality)
    selected = [sid for sid in range(physical_num_shards) if sid % max(train_world_size, 1) == train_rank]
    paths = [dense_shard_path(root, modality, sid, physical_num_shards) for sid in selected]
    if lazy_latents:
        payloads = []
        lengths = []
        trailing_shape = None
        dtype = None
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            latent = payload["latents"]
            lengths.append(int(latent.shape[0]))
            trailing_shape = tuple(latent.shape[1:]) if trailing_shape is None else trailing_shape
            dtype = latent.dtype if dtype is None else dtype
            metadata = dict(payload)
            metadata.pop("latents", None)
            payloads.append(metadata)
            del latent, payload
        merged = merge_dense_payloads(payloads)
        merged["latents"] = LazyMMapShardedRows(paths, lengths, trailing_shape, dtype)
        merged["num_samples"] = len(merged["latents"])
    else:
        payloads = [
            torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            for path in paths
        ]
        merged = merge_dense_payloads(payloads)
    sources = merged.get("sample_sources") or [{} for _ in merged.get("sample_ids", [])]
    if not merged.get("video_ids"):
        merged["video_ids"] = [
            str(source.get("video_id") or source.get("video_name") or "") for source in sources
        ]
    if not merged.get("frame_indices"):
        merged["frame_indices"] = [
            source.get("selected_frame_index", source.get("frame_index")) for source in sources
        ]
    return merged


def robust_rmtree(path, attempts: int = 4, delay: float = 0.5) -> None:
    """Best-effort recursive delete that never crashes training.

    Checkpoint rotation is pure cleanup, but on network filesystems
    (NFS/Lustre) deleting a directory whose files are still held open leaves
    ``.nfs*`` silly-rename entries behind, so the final ``rmdir`` fails with
    ``OSError: [Errno 39] Directory not empty``. This was fatal when resuming
    into a run dir that already contained the resumed step's checkpoint. Retry a
    few times, then give up quietly rather than killing the run.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(delay)


# Condition-attention topology switch.
#   "native_joint": Native Joint Attention. Original FLUX/Omini-style single-softmax
#     layout: every lane (incl. the target) attends to text+target+condition+
#     reference in one SDPA.
#   "modality_tunneling": Modality Tunneling Attention. For the target lane, we
#     compute self-attention (text+target) and additive target->condition/
#     reference cross-attention as two separate SDPA calls, then sum them. Other
#     lanes stay on the Native Joint Attention path. No trainable parameters are introduced.
_NATIVE_JOINT_ATTENTION = "native_joint"
_MODALITY_TUNNELING_ATTENTION = "modality_tunneling"
_ATTENTION_ALIASES = {
    "joint": _NATIVE_JOINT_ATTENTION,
    "native_joint": _NATIVE_JOINT_ATTENTION,
    "native-joint": _NATIVE_JOINT_ATTENTION,
    "native_joint_attention": _NATIVE_JOINT_ATTENTION,
    "dual": _MODALITY_TUNNELING_ATTENTION,
    "modality_tunneling": _MODALITY_TUNNELING_ATTENTION,
    "modality-tunneling": _MODALITY_TUNNELING_ATTENTION,
    "modality_tunneling_attention": _MODALITY_TUNNELING_ATTENTION,
}
_ATTENTION_CHOICES = ("auto", "native_joint", "modality_tunneling", "joint", "dual")
_CONDITION_ATTENTION_MODE = _MODALITY_TUNNELING_ATTENTION
_SEPARATE_CONDITION_ADAPTERS = False


def normalize_attention_mode(mode: str) -> str:
    norm = _ATTENTION_ALIASES.get(str(mode).strip().lower())
    if norm is None:
        raise ValueError(f"Unknown condition attention mode: {mode!r}")
    return norm


def set_condition_attention_mode(mode: str) -> None:
    global _CONDITION_ATTENTION_MODE
    _CONDITION_ATTENTION_MODE = normalize_attention_mode(mode)


def set_separate_condition_adapters(flag: bool) -> None:
    global _SEPARATE_CONDITION_ADAPTERS
    _SEPARATE_CONDITION_ADAPTERS = bool(flag)


@contextmanager
def specify_lora(lora_modules, specified_lora):
    """Temporarily activate exactly one PEFT adapter for the given modules.

    This is the only third-party Omini utility TunnelingNet depended on. Keeping it
    local makes Channel Head routing explicit and removes the third-party runtime
    dependency. Passing None disables all active adapters for those modules.
    """
    valid_lora_modules = [module for module in lora_modules if isinstance(module, BaseTunerLayer)]
    original_scales = [
        {
            adapter: scale
            for adapter, scale in module.scaling.items()
        }
        for module in valid_lora_modules
    ]
    for module in valid_lora_modules:
        for adapter in module.scaling:
            module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter, scale in scales.items():
                if adapter in module.scaling:
                    module.scaling[adapter] = scale


# Target-lane trainability is set per task_type at runtime via the recipes below.
# (The original Omini-style recipe freezes the target lane and only adapts condition lanes;
# for new-modality tasks we make it trainable so it can learn the output instead of
# collapsing to the mean -- but the narrow generation task keeps it frozen, see
# the recipe notes.)
_TRAINABLE_TARGET = True
_TRAINABLE_CONDITION = True


def set_trainable_target(flag: bool) -> None:
    global _TRAINABLE_TARGET
    _TRAINABLE_TARGET = bool(flag)


def set_trainable_condition(flag: bool) -> None:
    global _TRAINABLE_CONDITION
    _TRAINABLE_CONDITION = bool(flag)


# ============================================================================
# UNIVERSAL DENSE-PREDICTION LOSS ENHANCEMENT (modality-agnostic).
# One enhancement, three pure objective-shaping levers, no new parameters and no
# change to inference (we still predict velocity). Gated per task_type via the
# recipes' "dense_enhance" field, fully reversible by turning it off:
#   A. Timestep low-noise shift. Sample timestep = sigmoid(randn*scale + shift)
#      with shift<0, biasing toward the low-noise end where the target (a narrow
#      colour manifold like normal, or a thin structure like edge) is still
#      clearly visible -- so its supervision is not drowned out by noise at high t.
#   B. Min-SNR-gamma weighting. Weight the per-sample velocity loss by
#      min(SNR(t), gamma)/gamma with SNR(t)=((1-t)/t)^2, concentrating gradient on
#      informative low-t steps and suppressing the "predicting the mean is safest"
#      degeneracy that washes outputs out. gamma caps the low-t explosion.
#   C. Huber (smooth-L1) loss replacing pure L2. L2 rewards smoothing toward the
#      mean (faded output); Huber keeps sharp structure for both narrow colour
#      manifolds and thin structures.
# This is deliberately NOT modality-specific: the same three levers help normal,
# edge, depth, etc. Hyperparameters are env-tunable; enable is per-task.
# ============================================================================
_DENSE_ENHANCE = False
# The three levers can be toggled independently (for ablation). By default each
# follows the master _DENSE_ENHANCE flag; set_dense_enhance() can override any one.
_DE_TIMESTEP_SHIFT = False   # A
_DE_MINSNR = False           # B
_DE_HUBER = False            # C
_DENSE_ENHANCE_SHIFT = float(os.environ.get("WSG_DENSE_ENHANCE_SHIFT", "-0.5"))
_DENSE_ENHANCE_SCALE = float(os.environ.get("WSG_DENSE_ENHANCE_SCALE", "1.0"))
_DENSE_ENHANCE_GAMMA = float(os.environ.get("WSG_DENSE_ENHANCE_GAMMA", "5.0"))
_DENSE_ENHANCE_HUBER_DELTA = float(os.environ.get("WSG_DENSE_ENHANCE_HUBER_DELTA", "1.0"))

# Independent literature ablations for deterministic dense prediction.
_SINGLE_STEP_X0 = False
_HIGH_NOISE_ANCHOR = False
_MULTIRES_NOISE = False
_HIGH_NOISE_ANCHOR_PROB = float(os.environ.get("WSG_HIGH_NOISE_ANCHOR_PROB", "0.5"))
_HIGH_NOISE_ANCHOR_T = float(os.environ.get("WSG_HIGH_NOISE_ANCHOR_T", "0.999"))
_MULTIRES_LEVELS = int(os.environ.get("WSG_MULTIRES_LEVELS", "4"))
_MULTIRES_DISCOUNT = float(os.environ.get("WSG_MULTIRES_DISCOUNT", "0.5"))

def set_literature_ablations(*, single_step_x0: bool, high_noise_anchor: bool, multires_noise: bool) -> None:
    global _SINGLE_STEP_X0, _HIGH_NOISE_ANCHOR, _MULTIRES_NOISE
    _SINGLE_STEP_X0 = bool(single_step_x0)
    _HIGH_NOISE_ANCHOR = bool(high_noise_anchor)
    _MULTIRES_NOISE = bool(multires_noise)


def set_dense_enhance(flag: bool, *, timestep: bool | None = None,
                      minsnr: bool | None = None, huber: bool | None = None) -> None:
    """Enable the dense-prediction enhancement. `flag` is the master switch; each
    of the three levers (A timestep shift, B Min-SNR weighting, C Huber loss)
    follows it unless individually overridden (for ablation)."""
    global _DENSE_ENHANCE, _DE_TIMESTEP_SHIFT, _DE_MINSNR, _DE_HUBER
    _DENSE_ENHANCE = bool(flag)
    base = bool(flag)
    _DE_TIMESTEP_SHIFT = base if timestep is None else bool(timestep)
    _DE_MINSNR = base if minsnr is None else bool(minsnr)
    _DE_HUBER = base if huber is None else bool(huber)


# ============================================================================
# PER-TASK RECIPES -- the four task types, four explicit configurations.
# Everything that differs by task lives here, one block per task, tuned by
# ablation (see debug/modality_tunneling_attention_notes.txt). Fields:
#   attention        : "native_joint" (Native Joint Attention)
#                      | "modality_tunneling" (Modality Tunneling Attention)
#   trainable_target : bool -- does the target lane carry the trainable Channel_Head?
#   trainable_condition : bool -- do condition lanes carry the trainable Channel_Head?
#   first_frame      : None | "native" | "condition" -- how the first-frame RGB
#                      (if the task has one) enters: frozen native reference, or
#                      the trainable Channel_Head condition channel.
#   dense_enhance    : bool -- master switch for the A(timestep shift) /
#                      B(Min-SNR) / C(Huber) loss enhancement.
# Per-task CLI/env switches and global overrides can still override these; with
# everything on "auto" these recipes are exactly what runs.
#
# Shared-condition finding (003/275/313/270, 2026-06-24): freezing
# condition-lane Channel_Head does not obviously break training in the four task recipes.
# This keeps condition hidden states adapter-independent, which is the prerequisite
# for future packed multi-Channel_Head training with one shared condition forward.
#
# Ablation finding (275/003, 2026-06-24): the enhancement is left OFF by default.
#   - Huber (C): visually the nicest, strong EARLY in training.
#   - Min-SNR (B): the val/training loss does NOT go down (it down-weights the
#     high-noise steps that set global colour), but it holds up BETTER LATE -- it
#     resists the late mean-collapse that plain training drifts into.
#   - With all three on, outputs slowly collapse toward the mean (normal keeps
#     shape but fades; edge gets swamped by the black background) as training
#     goes on. So none is a clear default win yet -- enable per-run via
#     --dense-enhance / --de-timestep / --de-minsnr / --de-huber when ablating.
# ============================================================================
_TASK_RECIPES = {
    # RGB -> Normal, etc. (e.g. Channel_Head_000003). No first frame.
    "estimation_or_reconstruction": {
        "attention": "modality_tunneling",
        "trainable_target": True,
        "first_frame": None,
        "trainable_condition": False,
        "high_noise_anchor": True,
        # OFF by default -- see ablation finding above (Huber strong early, Min-SNR
        # better late, all-three collapses toward the mean over time).
        "dense_enhance": False,
    },
    # Normal -> Edge, etc. (e.g. Channel_Head_000275). No first frame. (Still being tuned.)
    "conditional_control": {
        "attention": "modality_tunneling",
        "trainable_target": True,
        "first_frame": None,
        "trainable_condition": False,
        "high_noise_anchor": True,
        "dense_enhance": False,
    },
    # [Normal, RGB] -> Edge, etc. (e.g. Channel_Head_000313). First-frame RGB enters through
    # a condition branch, but the condition branch is frozen by default for
    # shared-condition training compatibility.
    "inverse_rendering": {
        "attention": "modality_tunneling",
        "trainable_target": True,
        "first_frame": "condition",
        "trainable_condition": False,
        "high_noise_anchor": True,
        "dense_enhance": False,
    },
    # [Normal, caption, first-frame] -> RGB (e.g. Channel_Head_000270). First frame is the
    # frozen native reference. With frozen condition lanes, the target lane is
    # trainable so this recipe still has an effective Channel_Head update path.
    # Narrow generation is a true generation task (not deterministic dense
    # prediction), so the low-noise/Min-SNR/Huber enhancement is left OFF here.
    "conditional_control+narrow_generation": {
        "attention": "native_joint",
        "trainable_target": True,
        "trainable_condition": False,
        "first_frame": "native",
        "high_noise_anchor": True,
        "dense_enhance": False,
    },
}
_DEFAULT_RECIPE = {
    "attention": "modality_tunneling",
    "trainable_target": True,
    "first_frame": None,
    "trainable_condition": True,
    "high_noise_anchor": False,
    "dense_enhance": False,
}


def task_recipe(task_type: str) -> dict:
    return _TASK_RECIPES.get(task_type, _DEFAULT_RECIPE)


# task_type -> the argparse attribute holding its per-task attention switch.
_TASK_ATTENTION_ARG = {
    "estimation_or_reconstruction": "attention_estimation",
    "conditional_control": "attention_conditional_control",
    "inverse_rendering": "attention_inverse_rendering",
    "conditional_control+narrow_generation": "attention_narrowgen",
}


def resolve_task_attention_mode(task_type: str, args: argparse.Namespace) -> tuple[str, str]:
    """Resolve Native Joint / Modality Tunneling Attention for a task_type.
    Precedence: global force > per-task switch > task recipe. Returns (mode,
    source) for logging."""
    glob = getattr(args, "condition_attention_mode", "auto")
    if glob != "auto":
        return normalize_attention_mode(glob), "global-override"
    attr = _TASK_ATTENTION_ARG.get(task_type)
    if attr:
        val = getattr(args, attr, "auto")
        if val != "auto":
            return normalize_attention_mode(val), f"task-switch({attr})"
    return normalize_attention_mode(task_recipe(task_type)["attention"]), "task-recipe"


def resolve_task_trainable_target(task_type: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Resolve trainable-target for a task_type. Precedence: global force
    (on/off) > task recipe. Returns (flag, source) for logging."""
    glob = getattr(args, "trainable_target", "auto")
    if glob in ("on", "off"):
        return glob == "on", "global-override"
    return task_recipe(task_type)["trainable_target"], "task-recipe"


def resolve_task_trainable_condition(task_type: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Resolve condition-lane trainability. Precedence: global force
    (on/off) > task recipe. Returns (flag, source) for logging."""
    glob = getattr(args, "trainable_condition", "auto")
    if glob in ("on", "off"):
        return glob == "on", "global-override"
    return bool(task_recipe(task_type).get("trainable_condition", True)), "task-recipe"


def resolve_task_dense_enhance(task_type: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Resolve the universal dense-prediction enhancement for a task_type.
    Precedence: global force (on/off) > task recipe. Returns (flag, source)."""
    glob = getattr(args, "dense_enhance", "auto")
    if glob in ("on", "off"):
        return glob == "on", "global-override"
    return bool(task_recipe(task_type).get("dense_enhance", False)), "task-recipe"


def resolve_high_noise_anchor(task_type: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Resolve high-noise anchor. Precedence: explicit on/off > task recipe."""
    raw = str(getattr(args, "high_noise_anchor", "auto")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True, "global-override"
    if raw in {"0", "false", "no", "off"}:
        return False, "global-override"
    if raw != "auto":
        raise ValueError(f"Invalid --high-noise-anchor value: {raw!r}; use auto/on/off")
    return bool(task_recipe(task_type).get("high_noise_anchor", False)), "task-recipe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--modality-config", required=True)
    parser.add_argument("--dense-latent-cache-dir", required=True)
    parser.add_argument("--xpose-latent-cache-dir", default=None)
    parser.add_argument("--audio-dense-condition-train", default="",
                        help="NPZ with frame-level audio features aligned to dense target latents; used by audio-conditioned dense tasks such as sparse Channel_Head_000008.")
    parser.add_argument("--audio-dense-condition-val", default="",
                        help="Validation NPZ for --audio-dense-condition-train.")
    parser.add_argument("--audio-dense-target-cache-dir", default=os.environ.get("WSG_AUDIO_DENSE_TARGET_CACHE_DIR", ""),
                        help="Optional dense latent cache root used only for audio-dense target latents. Conditions still come from --dense-latent-cache-dir.")
    parser.add_argument("--audio-dense-target-modality", default=os.environ.get("WSG_AUDIO_DENSE_TARGET_MODALITY", ""),
                        help="Optional target modality name inside --audio-dense-target-cache-dir. Empty => --target-modality.")
    parser.add_argument("--target-visual-modality", default=os.environ.get("WSG_TARGET_VISUAL_MODALITY", ""),
                        help="Optional modality used only for pseudo visualization decode of target/prediction. Empty => --target-modality.")
    parser.add_argument("--audio-condition-dim", type=int, default=0,
                        help="Audio feature dimension. >0 enables the audio-condition projector in TunnelingNet.")
    parser.add_argument("--audio-conditioning-mode", choices=("attention", "modulation"),
                        default=os.environ.get("WSG_AUDIO_CONDITIONING_MODE", "attention"),
                        help="How to inject audio_condition_feature. attention appends it as a condition lane; modulation uses FiLM-style target modulation and does not add an audio KV lane.")
    parser.add_argument("--separate-condition-adapters", action="store_true",
                        help="Route each condition lane through its own Channel Head adapter (condition_0, condition_1, ...), so Q/K/V/out/FFN adapter weights are not shared across condition modalities. Compatible with Native Joint and Modality Tunneling Attention.")
    parser.add_argument("--audio-global-condition", action="store_true",
                        help="Represent audio_condition_feature as one global condition token instead of expanding it to every target token.")
    parser.add_argument("--edit-loss-weight", type=float, default=float(os.environ.get("WSG_EDIT_LOSS_WEIGHT", "0")),
                        help="Training-only image-edit weighting. >0 upweights latent tokens where target differs from the current RGB/reference condition; validation stays on the fixed base metric.")
    parser.add_argument("--edit-loss-power", type=float, default=float(os.environ.get("WSG_EDIT_LOSS_POWER", "1")),
                        help="Power applied to normalized edit saliency before weighting.")
    parser.add_argument("--gradient-domain-loss-weight", type=float,
                        default=float(os.environ.get("WSG_GRADIENT_DOMAIN_LOSS_WEIGHT", "0")),
                        help="Optional clean-latent finite-difference loss weight; 0 disables it.")
    parser.add_argument("--focal-regression-gamma", type=float,
                        default=float(os.environ.get("WSG_FOCAL_REGRESSION_GAMMA", "0")),
                        help="Optional detached hard-token regression weighting exponent; 0 disables it.")
    parser.add_argument("--foreground-iou-loss-weight", type=float,
                        default=float(os.environ.get("WSG_FOREGROUND_IOU_LOSS_WEIGHT", "0")),
                        help="Optional latent foreground soft-IoU loss weight; 0 disables it.")
    parser.add_argument(
        "--categorical-loss-mode",
        choices=("off", "prototype_ce", "balanced_ce", "margin_global", "combined"),
        default=os.environ.get("WSG_CATEGORICAL_LOSS_MODE", "off"),
        help="Online categorical latent objective; off preserves the default training loss.",
    )
    parser.add_argument("--categorical-loss-weight", type=float,
                        default=float(os.environ.get("WSG_CATEGORICAL_LOSS_WEIGHT", "0.5")))
    parser.add_argument("--categorical-temperature", type=float,
                        default=float(os.environ.get("WSG_CATEGORICAL_TEMPERATURE", "0.25")))
    parser.add_argument("--categorical-margin", type=float,
                        default=float(os.environ.get("WSG_CATEGORICAL_MARGIN", "0.25")))
    parser.add_argument("--categorical-num-clusters", type=int,
                        default=int(os.environ.get("WSG_CATEGORICAL_NUM_CLUSTERS", "8")))
    parser.add_argument("--categorical-kmeans-iterations", type=int,
                        default=int(os.environ.get("WSG_CATEGORICAL_KMEANS_ITERATIONS", "4")))
    parser.add_argument("--channel-head-index", type=int, required=True)
    parser.add_argument("--target-modality", required=True)
    parser.add_argument("--condition-modalities", required=True)
    parser.add_argument("--channel-head-rank", type=int, default=8)
    parser.add_argument("--channel-head-alpha", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--eval-every-steps", type=int, default=100, help="Validation loss interval")
    parser.add_argument("--viz-every-steps", type=int, default=None, help="Visualization interval (defaults to eval-every-steps)")
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--caption-drop-probability", type=float, default=0.0)
    parser.add_argument("--max-local-samples", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--disable-preflight", action="store_true")
    parser.add_argument("--text-embedding-cache-dir", default=None)
    parser.add_argument(
        "--historical-reference-policy",
        choices=("strict", "tolerant"),
        default=os.environ.get("WSG_HISTORICAL_REFERENCE_POLICY", "tolerant"),
        help="Historical RGB policy for inverse/narrow generation. strict drops samples without an earlier frame; tolerant omits the native reference for RGB targets or falls back to current RGB for non-RGB targets.",
    )
    parser.add_argument(
        "--target-dataset-whitelist",
        default=os.environ.get("WSG_TARGET_DATASET_WHITELIST", ""),
        help="Optional comma-separated dataset_name whitelist applied after target/condition pair intersection. "
             "Empty => use target_dataset_whitelist_by_modality from the modality config, if present.",
    )
    parser.add_argument("--save-optimizer-state-8bit", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to trainer state directory to resume from")
    parser.add_argument("--inference-only", action="store_true", help="Load a trained Channel Head, regenerate its fixed inference preview, and exit without training.")
    parser.add_argument("--inference-checkpoint", type=str, default=None, help="Channel Head safetensors file used by --inference-only.")
    parser.add_argument(
        "--categorical-refiner", default="auto",
        help="Categorical refiner selection: auto (embedded then run-level sidecar beside ckpt), off, or an explicit checkpoint/folder.",
    )
    parser.add_argument(
        "--categorical-refiner-minimum-pixels", type=int, default=32,
        help="Minimum generated connected-component area recolored by a categorical refiner.",
    )
    parser.add_argument(
        "--categorical-refiner-audio-dataset", default="",
        help="Optional frame-audio NPZ used to supply Head-010 selector features during evaluation.",
    )
    parser.add_argument("--inference-label", default="latest", help="Output label under pseudo-visualization-inference.")
    parser.add_argument("--inference-steps", type=int, default=10, help="Denoising steps used by --inference-only.")
    parser.add_argument("--inference-shard-id", type=int, default=0, help="Logical latent shard processed by --inference-only.")
    parser.add_argument("--inference-num-shards", type=int, default=1, help="Number of logical latent shards processed across parallel inference workers.")
    parser.add_argument("--inference-save-raw-latent", action="store_true", help="Save each generated raw VAE latent beside inference metadata.")
    parser.add_argument("--inference-cache-only", action="store_true", help="Skip decoded/WSG visual artifacts and write only generated raw latents plus metadata.")
    parser.add_argument("--val-split-ratio", type=float, default=0.05)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--val-loss-batches", type=int, default=24,
                        help="Number of batches per validation-loss evaluation (3x the old default of 8).")
    parser.add_argument("--val-timestep-grid",
                        default=os.environ.get("WSG_VAL_TIMESTEP_GRID", "0.1,0.25,0.5,0.75,0.9"),
                        help="Comma-separated fixed timesteps tiled across val batches by batch_idx "
                             "(deterministic, low-variance, comparable across runs). Empty => sample a "
                             "fixed random timestep per batch (still step-independent).")
    parser.add_argument("--early-stop-seconds", type=float, default=7200,
                        help="Stop training if validation loss has not improved within this many seconds (default 7200 = 2h).")
    parser.add_argument("--preview-sample-id", default="", help="Exact sample id for fixed visualization preview. Empty => pick by video id + frame fraction (recommended) so the current frame sits late in the clip, well separated from the first-frame reference.")
    parser.add_argument(
        "--preview-sample-id-file", default="",
        help="Optional UTF-8 file containing one exact sample id per line. In inference-only mode, select all listed samples in file order during one model load.",
    )
    parser.add_argument("--preview-video-id", default=DEFAULT_PREVIEW_VIDEO_ID, help="Preferred video id for fixed visualization preview")
    parser.add_argument("--preview-frame-index", type=int, default=-1, help="Exact frame index for fixed visualization preview. Negative => use --preview-frame-fraction instead.")
    parser.add_argument("--preview-frame-fraction", type=float, default=0.75, help="When no exact sample id/frame index is given, deterministically pick the frame at this fraction along the clip (0.75 => a late frame, large motion vs the first-frame reference). Stays fixed across steps.")
    parser.add_argument("--preview-split", choices=("train", "val", "test", "all"), default="test",
                        help="Dataset split used for fixed previews. Defaults to held-out test samples.")
    parser.add_argument(
        "--preview-require-split", action="store_true",
        help="Fail instead of falling back to the training split when --preview-split has no valid samples.",
    )
    parser.add_argument("--preview-num-samples", type=int, default=int(os.environ.get("WSG_PREVIEW_NUM_SAMPLES", "1")),
                        help="Number of deterministic held-out preview samples. >1 writes sample_00/sample_01/... under each visualization step.")
    parser.add_argument(
        "--preview-random", action="store_true",
        help="Select deterministic random preview samples even when only one sample is requested.",
    )
    parser.add_argument(
        "--benchmark-metrics", action="store_true",
        help="Attach decoded-image and semantic categorical metrics to inference metadata.",
    )
    parser.add_argument(
        "--benchmark-categorical-tolerance", type=float, default=0.20,
        help="Fractional class-anchor tolerance used by --benchmark-metrics.",
    )
    parser.add_argument(
        "--benchmark-latent-diagnostics", action="store_true",
        help="Additionally compute normalized/raw latent MSE for benchmark debugging.",
    )
    # Attention topology is resolved per task_type. 'native_joint' = Native
    # Joint Attention; 'modality_tunneling' = Modality Tunneling Attention
    # (target self-attention + additive target->condition/reference attention).
    # The global switch forces all tasks when set; 'auto' (default) defers to
    # per-task switches/defaults. Legacy aliases 'joint'/'dual' are accepted.
    parser.add_argument(
        "--condition-attention-mode",
        choices=_ATTENTION_CHOICES,
        default=os.environ.get("WSG_FLUX_CONDITION_ATTENTION_MODE", "auto"),
        help="Global attention override. 'auto' uses per-task defaults; 'native_joint'/'modality_tunneling' force every task. Legacy 'joint'/'dual' are accepted.",
    )
    parser.add_argument(
        "--attention-estimation", choices=_ATTENTION_CHOICES,
        default=os.environ.get("WSG_ATTENTION_ESTIMATION", "auto"),
        help="Attention for task_type=estimation_or_reconstruction (auto=>modality_tunneling).",
    )
    parser.add_argument(
        "--attention-conditional-control", choices=_ATTENTION_CHOICES,
        default=os.environ.get("WSG_ATTENTION_CONTROL", "auto"),
        help="Attention for task_type=conditional_control (auto=>modality_tunneling).",
    )
    parser.add_argument(
        "--attention-inverse-rendering", choices=_ATTENTION_CHOICES,
        default=os.environ.get("WSG_ATTENTION_INVERSE_RENDERING", "auto"),
        help="Attention for task_type=inverse_rendering (auto=>modality_tunneling).",
    )
    parser.add_argument(
        "--attention-narrowgen", choices=_ATTENTION_CHOICES,
        default=os.environ.get("WSG_ATTENTION_NARROWGEN", "auto"),
        help="Attention for task_type=conditional_control+narrow_generation (auto=>native_joint).",
    )
    parser.add_argument(
        "--trainable-target", choices=("auto", "on", "off"),
        default=os.environ.get("WSG_TRAINABLE_TARGET", "auto"),
        help="Target-lane trainability. 'auto' (default) uses per-task defaults "
             "(all four task recipes=>on); 'on'/'off' force all tasks.",
    )
    parser.add_argument(
        "--trainable-condition", choices=("auto", "on", "off"),
        default=os.environ.get("WSG_TRAINABLE_CONDITION", "auto"),
        help="Condition-lane trainability. auto uses per-task defaults; current shared-condition recipes freeze condition lanes. on/off force all tasks.",
    )
    parser.add_argument(
        "--dense-enhance", choices=("auto", "on", "off"),
        default=os.environ.get("WSG_DENSE_ENHANCE", "auto"),
        help="Universal dense-prediction loss enhancement (timestep low-noise shift "
             "+ Min-SNR-gamma weighting + Huber loss). 'auto' (default) uses per-task "
             "defaults (estimation/control/inverse_rendering=>on, narrow_generation=>off); "
             "'on'/'off' force all tasks.",
    )
    # Per-lever ablation switches. 'auto' follows --dense-enhance; 'on'/'off'
    # override that single lever so each of A/B/C can be trained in isolation.
    parser.add_argument(
        "--de-timestep", choices=("auto", "on", "off"),
        default=os.environ.get("WSG_DE_TIMESTEP", "auto"),
        help="Lever A: low-noise timestep shift (auto=>follow --dense-enhance).",
    )
    parser.add_argument(
        "--de-minsnr", choices=("auto", "on", "off"),
        default=os.environ.get("WSG_DE_MINSNR", "auto"),
        help="Lever B: Min-SNR-gamma loss weighting (auto=>follow --dense-enhance).",
    )
    parser.add_argument(
        "--de-huber", choices=("auto", "on", "off"),
        default=os.environ.get("WSG_DE_HUBER", "auto"),
        help="Lever C: Huber loss instead of L2 (auto=>follow --dense-enhance).",
    )
    parser.add_argument("--single-step-x0", action="store_true",
                        help="GED/YOSO: fixed t=1 and direct clean-latent prediction.")
    parser.add_argument(
        "--high-noise-anchor", nargs="?", const="on",
        default=os.environ.get("WSG_HIGH_NOISE_ANCHOR", "auto"),
        help="NormalCrafter high-noise anchor: auto follows task recipe; on/off force all tasks.",
    )
    parser.add_argument("--multires-noise", action="store_true",
                        help="Marigold: annealed multi-resolution training noise.")
    return parser.parse_args()


def load_channel_head_entry(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    entries = config.get("wsg_v1", config)
    key = f"Channel_Head_{index:06d}"
    if key not in entries:
        raise KeyError(f"{key} is missing from {path}")
    return dict(entries[key])


def _parse_dataset_whitelist(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def load_target_dataset_whitelist(path: Path, target_modality: str, override: str = "") -> set[str]:
    explicit = _parse_dataset_whitelist(override)
    if explicit:
        return explicit
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    mapping = config.get("target_dataset_whitelist_by_modality") or {}
    return _parse_dataset_whitelist(mapping.get(str(target_modality)))


def build_prompts(sample_sources: list[dict[str, Any]]) -> list[str]:
    from src.dataset.wsg_dataset import WSGModalityDataset

    helper = WSGModalityDataset.__new__(WSGModalityDataset)
    helper.caption_file_name = "video_captions.csv"
    roots = sorted({str(item.get("dataset_root") or "") for item in sample_sources if item.get("dataset_root")})
    helper.caption_map = helper._load_caption_map(roots)
    prompts = []
    for source in sample_sources:
        path = Path(str(source.get("path") or ""))
        prompt = helper._caption_for_video(str(source.get("video_name") or path.stem), path)
        prompts.append(prompt)
    return prompts


def source_dataset_name(source: dict[str, Any]) -> str:
    return str(source.get("dataset_name") or source.get("dataset") or "")


def reference_group_key(source: dict[str, Any], video_id: Any) -> str:
    """Namespace video ids by dataset so unrelated datasets cannot share a reference."""
    normalized_video_id = "" if video_id is None else str(video_id)
    return f"{source_dataset_name(source)}\x1f{normalized_video_id}"


def build_first_frame_reference_cache(
    cache_dir: Path,
    modality: str,
    num_shards: int = 8,
    *,
    dataset_prefixes: tuple[str, ...] | None = ("AVIS_",),
    cache_tag: str = "avis",
) -> Path:
    reference_dir = cache_dir / "reference_frames"
    reference_path = reference_dir / f"{safe_modality_name(modality)}_{cache_tag}_first_frame.pt"
    if reference_path.exists():
        return reference_path
    # A single-dataset cache may contain no samples matching the requested
    # dataset prefixes (e.g. a custom-pilot cache with no AVIS videos, or a
    # non-video image dataset with frame_index=None). There is then no
    # cross-frame first frame to build, so return None instead of raising and
    # let callers fall back to the tolerant current-frame condition. All ranks
    # scan the same shards and reach the same decision without filesystem sync.
    if dataset_prefixes is not None:
        has_match = False
        for shard_id in range(num_shards):
            path = cache_dir / "shards" / safe_modality_name(modality) / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
            if not path.exists():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            sources = payload.get("sample_sources") or []
            frame_indices = payload.get("frame_indices") or [None] * len(sources)
            for idx, source in enumerate(sources):
                frame_index = frame_indices[idx] if idx < len(frame_indices) else None
                if source_dataset_name(source).startswith(dataset_prefixes) and frame_index is not None:
                    has_match = True
                    break
            del payload
            if has_match:
                break
        if not has_match:
            return None
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank == 0:
        reference_dir.mkdir(parents=True, exist_ok=True)
        earliest: dict[str, tuple[int, torch.Tensor, dict[str, Any], str]] = {}
        for shard_id in range(num_shards):
            path = cache_dir / "shards" / safe_modality_name(modality) / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            sources = payload.get("sample_sources") or [{} for _ in payload["sample_ids"]]
            video_ids = payload.get("video_ids") or [
                str(source.get("video_id") or source.get("video_name") or "") for source in sources
            ]
            frame_indices = payload.get("frame_indices") or [
                source.get("selected_frame_index", source.get("frame_index")) for source in sources
            ]
            for index, (video_id, frame_index, source) in enumerate(zip(video_ids, frame_indices, sources)):
                dataset_name = source_dataset_name(source)
                if dataset_prefixes and not dataset_name.startswith(dataset_prefixes):
                    continue
                if not video_id or frame_index is None:
                    continue
                frame_index = int(frame_index)
                group_key = reference_group_key(source, video_id)
                old = earliest.get(group_key)
                if old is None or frame_index < old[0]:
                    earliest[group_key] = (frame_index, payload["latents"][index].clone(), dict(source), str(video_id))
            del payload
        if not earliest:
            scope = ",".join(dataset_prefixes) if dataset_prefixes else "all datasets"
            raise RuntimeError(f"No first frames found in RGB cache for {scope}")
        group_keys = sorted(earliest)
        temporary = reference_path.with_suffix(".tmp")
        torch.save({
            "modality": modality,
            "group_keys": group_keys,
            "video_ids": [earliest[item][3] for item in group_keys],
            "frame_indices": [earliest[item][0] for item in group_keys],
            "latents": torch.stack([earliest[item][1] for item in group_keys]),
            "sample_sources": [earliest[item][2] for item in group_keys],
            "selection": f"earliest_frame_per_{cache_tag}_video",
        }, temporary)
        temporary.replace(reference_path)
        print(f"[WSM_v1] built native reference cache: {reference_path}", flush=True)
    else:
        deadline = time.time() + 1800
        while not reference_path.exists():
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for {reference_path}")
            time.sleep(2)
    return reference_path


class WSGOminiLatentDataset(Dataset):
    """Normal is the sole Omini condition; first-frame RGB is native FLUX.2 context."""
    def __init__(
        self,
        cache_dir,
        target_modality,
        condition_modality,
        reference_path,
        current_reference_modality,
        extra_condition_path,
        extra_condition_modality,
        current_extra_condition_modality,
        historical_reference_policy,
        local_historical_reference,
        shard_id,
        num_shards,
        max_samples,
        channel_head_entry,
        text_cache_dir=None,
        split="train",
        val_split_ratio=0.05,
        target_dataset_whitelist: set[str] | None = None,
        lazy_latents: bool = False,
    ):
        train_rank = int(shard_id)
        train_world_size = max(int(num_shards), 1)
        target = load_dense_payload(cache_dir, target_modality, train_rank, train_world_size, lazy_latents=lazy_latents)
        condition = load_dense_payload(cache_dir, condition_modality, train_rank, train_world_size, lazy_latents=lazy_latents) if condition_modality else None
        reference = torch.load(reference_path, map_location="cpu", weights_only=False) if reference_path is not None else None
        current_reference = load_dense_payload(cache_dir, current_reference_modality, train_rank, train_world_size, lazy_latents=lazy_latents) if current_reference_modality else None
        current_extra_condition = load_dense_payload(cache_dir, current_extra_condition_modality, train_rank, train_world_size, lazy_latents=lazy_latents) if current_extra_condition_modality else None
        historical_extra_condition = torch.load(extra_condition_path, map_location="cpu", weights_only=False) if extra_condition_path is not None else None
        target_by_id = {sample_id: index for index, sample_id in enumerate(target["sample_ids"])}
        condition_by_id = {sample_id: index for index, sample_id in enumerate(condition["sample_ids"])} if condition is not None else target_by_id
        reference_by_group = {
            str(group_key): index
            for index, group_key in enumerate(reference.get("group_keys") or reference["video_ids"])
        } if reference is not None else {}
        current_reference_by_id = {sample_id: index for index, sample_id in enumerate(current_reference["sample_ids"])} if current_reference is not None else {}
        current_reference_by_group = {}
        if current_reference is not None and local_historical_reference:
            local_sources = current_reference.get("sample_sources") or [{} for _ in current_reference["sample_ids"]]
            local_videos = current_reference.get("video_ids") or ["" for _ in current_reference["sample_ids"]]
            local_frames = current_reference.get("frame_indices") or [None for _ in current_reference["sample_ids"]]
            for local_index, (local_source, local_video, local_frame) in enumerate(zip(local_sources, local_videos, local_frames)):
                if local_frame is None:
                    continue
                local_key = reference_group_key(local_source, local_video)
                old_index = current_reference_by_group.get(local_key)
                if old_index is None or int(local_frame) < int(local_frames[old_index]):
                    current_reference_by_group[local_key] = local_index
        extra_by_group = {
            str(group_key): index
            for index, group_key in enumerate(historical_extra_condition.get("group_keys") or historical_extra_condition["video_ids"])
        } if historical_extra_condition is not None else {}
        extra_by_id = {sample_id: index for index, sample_id in enumerate(current_extra_condition["sample_ids"])} if current_extra_condition is not None else {}
        sources = target.get("sample_sources") or [{} for _ in target["sample_ids"]]
        video_ids = target.get("video_ids") or ["" for _ in target["sample_ids"]]
        frame_indices = target.get("frame_indices") or [None for _ in target["sample_ids"]]
        records = []
        strict_dropped = 0
        for sample_id in sorted(set(target_by_id).intersection(condition_by_id)):
            target_index = target_by_id[sample_id]
            source = sources[target_index]
            if not self._source_is_valid_for_target(target_modality, source, target_dataset_whitelist):
                continue
            reference_index = -1
            extra_condition_index = -1
            uses_current_extra = False
            current_frame = frame_indices[target_index]
            video_id = str(video_ids[target_index])
            group_key = reference_group_key(source, video_id)
            if reference is not None:
                reference_index = reference_by_group.get(group_key, -1)
                if (reference_index < 0 or current_frame is None
                        or int(current_frame) <= int(reference["frame_indices"][reference_index])):
                    if historical_reference_policy == "strict":
                        strict_dropped += 1
                        continue
                    # RGB is the narrow-generation target. Tolerant mode must omit
                    # a missing reference instead of leaking current target RGB.
                    reference_index = -1
            elif current_reference is not None:
                if local_historical_reference:
                    reference_index = current_reference_by_group.get(group_key, -1)
                    if (reference_index < 0 or current_frame is None
                            or int(current_frame) <= int(current_reference["frame_indices"][reference_index])):
                        if historical_reference_policy == "strict":
                            strict_dropped += 1
                            continue
                        reference_index = -1
                else:
                    reference_index = current_reference_by_id.get(sample_id, -1)
                    if reference_index < 0:
                        continue
            if historical_extra_condition is not None:
                extra_condition_index = extra_by_group.get(group_key, -1)
                if (extra_condition_index < 0 or current_frame is None
                        or int(current_frame) <= int(historical_extra_condition["frame_indices"][extra_condition_index])):
                    if historical_reference_policy == "strict":
                        strict_dropped += 1
                        continue
                    # The inverse-rendering target is non-RGB, so current RGB is a
                    # leakage-free fallback when no historical frame exists.
                    extra_condition_index = extra_by_id.get(sample_id, -1)
                    if extra_condition_index < 0:
                        continue
                    uses_current_extra = True
            elif current_extra_condition is not None:
                extra_condition_index = extra_by_id.get(sample_id, -1)
                if extra_condition_index < 0:
                    continue
                uses_current_extra = True
            if split != "all" and self._record_split(sample_id, source, val_split_ratio) != split:
                continue
            records.append((sample_id, target_index, condition_by_id[sample_id], reference_index, extra_condition_index, source, uses_current_extra))

        uses_historical_native_route = reference is not None or local_historical_reference
        uses_historical_extra_route = historical_extra_condition is not None
        fallback_candidates = [
            record for record in records
            if ((uses_historical_native_route and record[3] < 0)
                or (uses_historical_extra_route and record[6]))
        ]
        historical_records = [
            record for record in records
            if not ((uses_historical_native_route and record[3] < 0)
                    or (uses_historical_extra_route and record[6]))
        ]
        fallback_mode_activated = False
        if historical_reference_policy == "tolerant" and fallback_candidates:
            # Tolerance is a dataset-level rescue, not per-sample augmentation.
            # Never mix degraded samples into a split that already has valid
            # historical references.
            if historical_records:
                records = historical_records
            else:
                records = fallback_candidates
                fallback_mode_activated = True
            strict_dropped = len(fallback_candidates)
        if max_samples is not None and int(max_samples) > 0:
            records = records[:max_samples]
        if not records:
            raise RuntimeError(f"No pairs for {condition_modality} -> {target_modality}")
        self.target_latents = target["latents"]
        self.condition_latents = condition["latents"] if condition is not None else None
        self.reference_latents = current_reference["latents"] if current_reference is not None else (reference["latents"] if reference is not None else None)
        self.extra_condition_latents = historical_extra_condition["latents"] if historical_extra_condition is not None else None
        self.current_extra_condition_latents = current_extra_condition["latents"] if current_extra_condition is not None else None
        self.records = records
        self.split = str(split)
        self.reference_usage = {
            "policy": historical_reference_policy,
            "selection_scope": "split",
            "fallback_rule": "only_when_no_historical_samples_in_split",
            "effective_mode": (
                "strict_historical_only" if historical_reference_policy == "strict"
                else "tolerant_fallback_only" if fallback_mode_activated
                else "tolerant_historical_only"
            ),
            "historical_candidates": len(historical_records),
            "samples": len(records),
            "historical_native_reference": sum(1 for record in records if (reference is not None or local_historical_reference) and record[3] >= 0),
            "omitted_native_reference_fallback": sum(1 for record in records if (reference is not None or local_historical_reference) and record[3] < 0),
            "historical_rgb_condition": sum(1 for record in records if historical_extra_condition is not None and not record[6]),
            "current_rgb_condition_fallback": sum(1 for record in records if record[6]),
            "strict_dropped": strict_dropped,
            "fallback_candidates": len(fallback_candidates),
            "fallback_mode_activated": fallback_mode_activated,
        }
        self.target_frame_indices = frame_indices
        self.target_video_ids = video_ids
        self.condition_frame_indices = (
            condition.get("frame_indices") if condition is not None else frame_indices
        )
        self.condition_video_ids = (
            condition.get("video_ids") if condition is not None else video_ids
        )
        self.reference_frame_indices = current_reference.get("frame_indices") if current_reference is not None else (reference["frame_indices"] if reference is not None else None)
        self.reference_video_ids = current_reference.get("video_ids") if current_reference is not None else (reference["video_ids"] if reference is not None else None)
        self.extra_condition_frame_indices = historical_extra_condition["frame_indices"] if historical_extra_condition is not None else None
        self.extra_condition_video_ids = historical_extra_condition["video_ids"] if historical_extra_condition is not None else None
        self.current_extra_condition_frame_indices = current_extra_condition["frame_indices"] if current_extra_condition is not None else None
        self.current_extra_condition_video_ids = current_extra_condition["video_ids"] if current_extra_condition is not None else None
        self.extra_condition_modality = current_extra_condition_modality or extra_condition_modality
        self.channel_head_entry = dict(channel_head_entry or {})
        raw_conditions = self.channel_head_entry.get("condition") or [condition_modality]
        self.condition_tokens = raw_conditions if isinstance(raw_conditions, list) else [raw_conditions]
        self.use_caption = "caption" in self.condition_tokens
        self.modality_type_prompt = self._build_modality_type_prompt(self.channel_head_entry, condition_modality, target_modality, extra_condition_modality, current_reference_modality)
        self.prompts = ["" for _ in records]
        if self.use_caption:
            fallback_prompts = build_prompts([record[5] for record in records])
            prompts_path = Path(cache_dir) / "prompts.pt"
            if prompts_path.is_file():
                prompt_payload = torch.load(prompts_path, map_location="cpu", weights_only=False)
                prompt_sample_ids = list(prompt_payload.get("sample_ids") or [])
                prompt_values = list(prompt_payload.get("prompts") or [])
                if len(prompt_sample_ids) != len(prompt_values):
                    raise ValueError(
                        f"{prompts_path} contains {len(prompt_sample_ids)} sample_ids "
                        f"but {len(prompt_values)} prompts"
                    )
                prompt_by_sample_id = dict(zip(prompt_sample_ids, prompt_values))
                missing_prompt_ids = [record[0] for record in records if record[0] not in prompt_by_sample_id]
                if missing_prompt_ids:
                    raise ValueError(
                        f"{prompts_path} is missing prompts for sample_ids: {missing_prompt_ids[:10]}"
                    )
                self.prompts = [str(prompt_by_sample_id[record[0]]) for record in records]
            else:
                self.prompts = fallback_prompts
        self.text_cache = None
        self.text_prompt_embed_shards = None
        self.text_id_shards = None
        if self.use_caption and text_cache_dir is not None:
            text_root = Path(text_cache_dir)
            text_shards = sorted(text_root.glob("shard_*_of_*.pt"))
            if text_shards:
                try:
                    physical_total = int(text_shards[0].stem.split("_of_", 1)[-1])
                    selected_text_shards = [
                        text_root / f"shard_{sid:05d}_of_{physical_total:05d}.pt"
                        for sid in range(physical_total)
                        if sid % max(int(num_shards), 1) == int(shard_id)
                    ]
                except Exception:
                    selected_text_shards = text_shards
                payloads = [
                    torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                    for path in selected_text_shards
                    if path.exists()
                ]
                if payloads:
                    required_sample_ids = {sample_id for sample_id, *_ in records}
                    required_by_cache_id: dict[str, list[str]] = {}
                    for required_sample_id in required_sample_ids:
                        cache_id = required_sample_id
                        if required_sample_id.startswith("mp100::") and "::ann_" in required_sample_id:
                            cache_id = required_sample_id.rsplit("::ann_", 1)[0]
                        required_by_cache_id.setdefault(cache_id, []).append(required_sample_id)
                    locations = {}
                    for payload_index, payload in enumerate(payloads):
                        sample_ids = payload["sample_ids"]
                        sample_text_indices = payload.get("sample_text_indices")
                        if sample_text_indices is None:
                            sample_text_indices = range(len(sample_ids))
                        for sample_id, text_index in zip(sample_ids, sample_text_indices):
                            for required_sample_id in required_by_cache_id.get(sample_id, ()):
                                locations[required_sample_id] = (payload_index, int(text_index))
                    self.text_cache = locations
                    self.text_prompt_embed_shards = [payload["prompt_embeds"] for payload in payloads]
                    self.text_id_shards = [payload["text_ids"] for payload in payloads]
                    print(
                        f"[WSM_v1] memory-mapped text cache {len(payloads)} shard(s): "
                        f"{len(self.text_cache)}/{len(records)} samples",
                        flush=True,
                    )
                else:
                    print(f"[WSM_v1] text cache shards not readable under {text_root}, falling back to online encoding", flush=True)
            else:
                print(f"[WSM_v1] text cache not found under {text_root}, falling back to online encoding", flush=True)

    @staticmethod
    def _source_is_valid_for_target(
        target_modality: str,
        source: dict[str, Any],
        target_dataset_whitelist: set[str] | None = None,
    ) -> bool:
        # Second-stage data selection after target/condition pair intersection.
        # Some modalities have multiple valid caches with incompatible semantics
        # or trivial defaults (e.g. alpha=1 for ordinary opaque frames). Keep the
        # rule dataset-name based so WSG/WSV storage format does not decide semantic validity.
        if target_dataset_whitelist:
            return source_dataset_name(source) in target_dataset_whitelist
        return True

    @staticmethod
    def _record_split(sample_id: str, source: dict[str, Any], val_split_ratio: float) -> str:
        explicit_split = str(source.get("split") or "").strip().lower()
        if explicit_split in {"train", "val", "test"}:
            return explicit_split
        dataset_name = source_dataset_name(source)
        if dataset_name.startswith("AVIS_train"):
            return "train"
        if dataset_name.startswith("AVIS_val"):
            return "val"
        if dataset_name.startswith("AVIS_test"):
            return "test"
        key = f"{dataset_name}:{sample_id}".encode("utf-8")
        bucket = int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / float(2**64)
        holdout = max(0.0, min(0.49, val_split_ratio))
        if bucket < holdout:
            return "val"
        if bucket < holdout * 2.0:
            return "test"
        return "train"


    @staticmethod
    def _build_modality_type_prompt(entry: dict[str, Any], condition_modality: str | None, target_modality: str, extra_condition_modality: str | None = None, current_reference_modality: str | None = None) -> str:
        raw_conditions = entry.get("condition") or [condition_modality]
        conditions = raw_conditions if isinstance(raw_conditions, list) else [raw_conditions]
        tags = []
        if "caption" in conditions:
            tags.append("caption text")
        if "Reference_IMG" in conditions:
            tags.append("reference image")
        if "CurrentFrame_RGB_Reference" in conditions:
            tags.append(f"current-frame native reference modality {current_reference_modality or '0-2'}")
        if condition_modality:
            tags.append(f"condition modality {condition_modality}")
        if extra_condition_modality:
            tags.append(f"first-frame condition modality {extra_condition_modality}")
        tags.append(f"target modality {target_modality}")
        return ". ".join(tags) + "."

    def __len__(self):
        return len(self.records)

    def find_preview_index(
        self,
        sample_id: str | None = None,
        video_id: str | None = None,
        frame_index: int | None = None,
        frame_fraction: float | None = None,
    ) -> int:
        if sample_id:
            for index, record in enumerate(self.records):
                if record[0] == sample_id:
                    return index
        if video_id:
            matching_video = []
            for index, record in enumerate(self.records):
                target_index = record[1]
                if str(self.target_video_ids[target_index]) != str(video_id):
                    continue
                matching_video.append((index, int(self.target_frame_indices[target_index])))
                if frame_index is not None and int(frame_index) >= 0 and int(self.target_frame_indices[target_index]) == int(frame_index):
                    return index
            if matching_video:
                if frame_fraction is not None:
                    # Pick a frame from the back of the clip so the current frame is
                    # well separated from the fixed first-frame reference (larger
                    # motion => clearer modality-transfer signal). Deterministic, so
                    # the preview stays fixed across steps for progress tracking.
                    ordered = sorted(matching_video, key=lambda item: item[1])
                    pick = int(round(max(0.0, min(1.0, frame_fraction)) * (len(ordered) - 1)))
                    return ordered[pick][0]
                return matching_video[0][0]
        if frame_fraction is not None and self.records:
            pick = int(round(max(0.0, min(1.0, frame_fraction)) * (len(self.records) - 1)))
            return pick
        return 0

    @staticmethod
    def _safe_int(value, default: int = -1) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def __getitem__(self, index):
        sample_id, target_index, condition_index, reference_index, extra_condition_index, source, uses_current_extra = self.records[index]
        prompt = self.prompts[index]
        has_reference = self.reference_latents is not None and reference_index >= 0
        suffix = "Preserve the reference appearance and follow the supplied condition modality." if has_reference else ""
        parts = [prompt, self.modality_type_prompt, suffix]
        prompt = " ".join(part.strip() for part in parts if part and part.strip())
        item = {
            "target_latent": self.target_latents[target_index],
            "description": prompt,
            "sample_id": sample_id,
            "sample_source": source,
            "video_id": str(self.target_video_ids[target_index]),
            "target_frame": self._safe_int(self.target_frame_indices[target_index]),
            "dataset_split": self.split,
        }
        if self.condition_latents is not None:
            item["omini_condition_latent"] = self.condition_latents[condition_index]
            item["condition_frame"] = self._safe_int(
                self.condition_frame_indices[condition_index]
            )
            item["condition_video_id"] = str(
                self.condition_video_ids[condition_index]
            )
        if has_reference:
            item["native_reference_latent"] = self.reference_latents[reference_index]
            item["reference_frame"] = self._safe_int(self.reference_frame_indices[reference_index])
            item["reference_video_id"] = str(self.reference_video_ids[reference_index])
        if self.extra_condition_latents is not None or self.current_extra_condition_latents is not None:
            if uses_current_extra:
                item["extra_condition_latent"] = self.current_extra_condition_latents[extra_condition_index]
                extra_frames = self.current_extra_condition_frame_indices
                extra_videos = self.current_extra_condition_video_ids
                item["extra_condition_reference_kind"] = "current_rgb_fallback"
            else:
                item["extra_condition_latent"] = self.extra_condition_latents[extra_condition_index]
                extra_frames = self.extra_condition_frame_indices
                extra_videos = self.extra_condition_video_ids
                item["extra_condition_reference_kind"] = "historical_first_frame"
            item["extra_condition_modality"] = self.extra_condition_modality or ""
            item["extra_condition_frame"] = self._safe_int(extra_frames[extra_condition_index])
            item["extra_condition_video_id"] = str(extra_videos[extra_condition_index])
        if self.text_cache is not None and sample_id in self.text_cache:
            shard_index, text_index = self.text_cache[sample_id]
            item["prompt_embeds"] = self.text_prompt_embed_shards[shard_index][text_index]
            item["text_ids"] = self.text_id_shards[shard_index][text_index]
        return item


class WSGAudioConditionDenseDataset(Dataset):
    """Dense target latent dataset with frame-level AST audio as a condition lane."""
    def __init__(
        self,
        cache_dir: Path,
        target_modality: str,
        audio_dataset_path: Path,
        current_reference_modality: str | None,
        current_rgb_as_condition: bool,
        shard_id: int,
        num_shards: int,
        max_samples: int | None,
        channel_head_entry: dict[str, Any],
        target_dataset_whitelist: set[str] | None = None,
        target_cache_dir: Path | None = None,
        target_cache_modality: str | None = None,
    ) -> None:
        train_rank = int(shard_id)
        train_world_size = max(int(num_shards), 1)
        target_root = target_cache_dir or cache_dir
        target_name = target_cache_modality or target_modality
        target = load_dense_payload(target_root, target_name, train_rank, train_world_size)
        current_reference = load_dense_payload(cache_dir, current_reference_modality, train_rank, train_world_size) if current_reference_modality else None
        audio_npz = np.load(audio_dataset_path, allow_pickle=False)
        audio_keys = [
            f"{str(wsv_path)}::{str(frame_key)}-{str(frame_key)}"
            for wsv_path, frame_key in zip(audio_npz["wsv_path"].astype(str), audio_npz["frame_key"].astype(str))
        ]
        audio_by_id = {sample_id: index for index, sample_id in enumerate(audio_keys)}
        target_by_id = {sample_id: index for index, sample_id in enumerate(target["sample_ids"])}
        reference_by_id = {sample_id: index for index, sample_id in enumerate(current_reference["sample_ids"])} if current_reference is not None else {}
        sources = target.get("sample_sources") or [{} for _ in target["sample_ids"]]
        video_ids = target.get("video_ids") or ["" for _ in target["sample_ids"]]
        frame_indices = target.get("frame_indices") or [None for _ in target["sample_ids"]]
        records = []
        for sample_id in sorted(set(target_by_id).intersection(audio_by_id)):
            target_index = target_by_id[sample_id]
            source = sources[target_index]
            if not WSGOminiLatentDataset._source_is_valid_for_target(target_modality, source, target_dataset_whitelist):
                continue
            reference_index = -1
            if current_reference is not None:
                reference_index = reference_by_id.get(sample_id, -1)
                if reference_index < 0:
                    continue
            records.append((sample_id, target_index, audio_by_id[sample_id], reference_index, source))
        if max_samples is not None and int(max_samples) > 0:
            records = records[:max_samples]
        if not records:
            raise RuntimeError(f"No audio-conditioned dense pairs for {audio_dataset_path} -> {target_modality}")
        self.target_latents = target["latents"]
        self.target_crop_infos = target.get("crop_infos")
        self.reference_latents = current_reference["latents"] if current_reference is not None else None
        self.current_rgb_as_condition = bool(current_rgb_as_condition)
        self.reference_frame_indices = current_reference.get("frame_indices") if current_reference is not None else None
        self.reference_video_ids = current_reference.get("video_ids") if current_reference is not None else None
        self.target_frame_indices = frame_indices
        self.target_video_ids = video_ids
        self.audio_feat = audio_npz["audio_feat"].astype(np.float32)
        self.audio_rms_dbfs = audio_npz["audio_rms_dbfs"].astype(np.float32) if "audio_rms_dbfs" in audio_npz else np.zeros((len(self.audio_feat),), dtype=np.float32)
        self.audio_category = audio_npz["category"].astype(np.int64) if "category" in audio_npz else np.full((len(self.audio_feat),), -1, dtype=np.int64)
        self.wsv_path = audio_npz["wsv_path"].astype(str)
        self.frame_key = audio_npz["frame_key"].astype(str)
        self.records = records
        self.channel_head_entry = dict(channel_head_entry or {})
        self.modality_type_prompt = self._build_prompt(channel_head_entry, target_modality)

    @staticmethod
    def _build_prompt(entry: dict[str, Any], target_modality: str) -> str:
        desc = str((entry or {}).get("target_description") or "semantic mask")
        return f"audio-conditioned dense structure prediction. target modality {target_modality}. {desc}"

    @staticmethod
    def _safe_int(value, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def __len__(self) -> int:
        return len(self.records)

    def find_preview_index(self, sample_id: str | None = None, video_id: str | None = None, frame_index: int | None = None, frame_fraction: float | None = None) -> int:
        if sample_id:
            for index, record in enumerate(self.records):
                if record[0] == sample_id:
                    return index
        if video_id:
            matches = []
            for index, record in enumerate(self.records):
                target_index = record[1]
                if str(self.target_video_ids[target_index]) != str(video_id):
                    continue
                frame = self._safe_int(self.target_frame_indices[target_index])
                if frame_index is not None and frame_index >= 0 and frame == frame_index:
                    return index
                matches.append((index, frame))
            if matches:
                ordered = sorted(matches, key=lambda item: item[1])
                if frame_fraction is None:
                    return ordered[0][0]
                pick = int(round(max(0.0, min(1.0, frame_fraction)) * (len(ordered) - 1)))
                return ordered[pick][0]
        if frame_fraction is not None and self.records:
            pick = int(round(max(0.0, min(1.0, frame_fraction)) * (len(self.records) - 1)))
            return pick
        return 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, target_index, audio_index, reference_index, source = self.records[index]
        item = {
            "target_latent": self.target_latents[target_index],
            "audio_condition_feature": torch.from_numpy(self.audio_feat[audio_index]),
            "description": self.modality_type_prompt,
            "sample_id": sample_id,
            "sample_source": source,
            "video_id": str(self.target_video_ids[target_index]),
            "target_frame": self._safe_int(self.target_frame_indices[target_index]),
            "audio_rms_dbfs": torch.tensor(float(self.audio_rms_dbfs[audio_index]), dtype=torch.float32),
            "audio_category": torch.tensor(int(self.audio_category[audio_index]), dtype=torch.long),
            "wsv_path": str(self.wsv_path[audio_index]),
            "frame_key": str(self.frame_key[audio_index]),
        }
        if self.target_crop_infos is not None:
            item["crop_info"] = dict(self.target_crop_infos[target_index])
        if self.reference_latents is not None:
            reference_key = "omini_condition_latent" if self.current_rgb_as_condition else "native_reference_latent"
            item[reference_key] = self.reference_latents[reference_index]
            item["reference_frame"] = self._safe_int(self.reference_frame_indices[reference_index])
            item["reference_video_id"] = str(self.reference_video_ids[reference_index])
        return item


def project_qkv(attn, hidden_state, adapter, added, specify_lora):
    if added:
        modules = (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
    else:
        modules = (attn.to_q, attn.to_k, attn.to_v)
    with specify_lora(modules, adapter):
        query, key, value = (module(hidden_state) for module in modules)
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    if added:
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)
    else:
        query, key = attn.norm_q(query), attn.norm_k(key)
    return query, key, value


def branch_attention(attn, image_states, text_states, adapters, position_embs, specify_lora):
    text_count = len(text_states)
    queries, keys, values = [], [], []
    for index, hidden_state in enumerate(text_states):
        q, k, v = project_qkv(attn, hidden_state, adapters[index], True, specify_lora)
        queries.append(q); keys.append(k); values.append(v)
    for index, hidden_state in enumerate(image_states):
        adapter = adapters[text_count + index]
        q, k, v = project_qkv(attn, hidden_state, adapter, False, specify_lora)
        queries.append(q); keys.append(k); values.append(v)
    for index in range(len(queries)):
        queries[index] = apply_rotary_emb(queries[index], position_embs[index], sequence_dim=1)
        keys[index] = apply_rotary_emb(keys[index], position_embs[index], sequence_dim=1)
    all_keys = torch.cat(keys, dim=1).transpose(1, 2)
    all_values = torch.cat(values, dim=1).transpose(1, 2)
    # Lane order is [text..., target, condition..., reference]; the target is the
    # first image lane. In Modality Tunneling Attention mode the target's attention is split into a
    # self branch (text+target) plus an additive condition/reference cross branch.
    target_idx = text_count
    use_tunneling = _CONDITION_ATTENTION_MODE == _MODALITY_TUNNELING_ATTENTION and len(queries) > target_idx + 1
    if use_tunneling:
        self_idx = list(range(text_count)) + [target_idx]
        cross_idx = list(range(target_idx + 1, len(queries)))
        self_keys = torch.cat([keys[i] for i in self_idx], dim=1).transpose(1, 2)
        self_values = torch.cat([values[i] for i in self_idx], dim=1).transpose(1, 2)
        cross_keys = torch.cat([keys[i] for i in cross_idx], dim=1).transpose(1, 2)
        cross_values = torch.cat([values[i] for i in cross_idx], dim=1).transpose(1, 2)
    outputs = []
    for index, query in enumerate(queries):
        q = query.transpose(1, 2)
        if use_tunneling and index == target_idx:
            output = F.scaled_dot_product_attention(q, self_keys, self_values) + \
                F.scaled_dot_product_attention(q, cross_keys, cross_values)
        else:
            output = F.scaled_dot_product_attention(q, all_keys, all_values)
        outputs.append(output.transpose(1, 2).flatten(2, 3).to(query.dtype))
    text_outputs, image_outputs = [], []
    for index in range(text_count):
        with specify_lora((attn.to_add_out,), adapters[index]):
            text_outputs.append(attn.to_add_out(outputs[index]))
    for index in range(len(image_states)):
        adapter = adapters[text_count + index]
        with specify_lora((attn.to_out[0],), adapter):
            image_outputs.append(attn.to_out[0](outputs[text_count + index]))
    return image_outputs, text_outputs


def double_block_forward(block, image_states, text_states, image_mods, text_mods, adapters, position_embs, specify_lora):
    text_count = len(text_states)
    image_norms, text_norms = [], []
    for state, mods in zip(image_states, image_mods):
        (shift_msa, scale_msa, _), _ = mods
        image_norms.append((1 + scale_msa) * block.norm1(state) + shift_msa)
    for state, mods in zip(text_states, text_mods):
        (shift_msa, scale_msa, _), _ = mods
        text_norms.append((1 + scale_msa) * block.norm1_context(state) + shift_msa)
    image_attn, text_attn = branch_attention(
        block.attn, image_norms, text_norms, adapters, position_embs, specify_lora
    )
    image_out = []
    for index, (state, attn_out, mods) in enumerate(zip(image_states, image_attn, image_mods)):
        (_, _, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = mods
        state = state + gate_msa * attn_out
        norm = block.norm2(state) * (1 + scale_mlp) + shift_mlp
        adapter = adapters[text_count + index]
        with specify_lora((block.ff.linear_in, block.ff.linear_out), adapter):
            state = state + gate_mlp * block.ff(norm)
        image_out.append(state)
    text_out = []
    for index, (state, attn_out, mods) in enumerate(zip(text_states, text_attn, text_mods)):
        (_, _, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = mods
        state = state + gate_msa * attn_out
        norm = block.norm2_context(state) * (1 + scale_mlp) + shift_mlp
        with specify_lora((block.ff_context.linear_in, block.ff_context.linear_out), adapters[index]):
            state = state + gate_mlp * block.ff_context(norm)
        text_out.append(state)
    return image_out, text_out


def single_block_forward(block, states, mods, adapters, position_embs, specify_lora):
    queries, keys, values, mlps = [], [], [], []
    for state, mod, adapter in zip(states, mods, adapters):
        shift, scale, _ = mod
        norm = (1 + scale) * block.norm(state) + shift
        with specify_lora((block.attn.to_qkv_mlp_proj,), adapter):
            projected = block.attn.to_qkv_mlp_proj(norm)
        qkv, mlp = torch.split(
            projected,
            [3 * block.attn.inner_dim, block.attn.mlp_hidden_dim * block.attn.mlp_mult_factor],
            dim=-1,
        )
        q, k, v = qkv.chunk(3, dim=-1)
        q = block.attn.norm_q(q.unflatten(-1, (block.attn.heads, -1)))
        k = block.attn.norm_k(k.unflatten(-1, (block.attn.heads, -1)))
        v = v.unflatten(-1, (block.attn.heads, -1))
        queries.append(q); keys.append(k); values.append(v)
        mlps.append(block.attn.mlp_act_fn(mlp))
    for index in range(len(states)):
        queries[index] = apply_rotary_emb(queries[index], position_embs[index], sequence_dim=1)
        keys[index] = apply_rotary_emb(keys[index], position_embs[index], sequence_dim=1)
    all_keys = torch.cat(keys, dim=1).transpose(1, 2)
    all_values = torch.cat(values, dim=1).transpose(1, 2)
    # In the single stream the state order is [text, target, condition..., ref].
    # Mirror branch_attention's Modality Tunneling Attention topology for the target lane (index 1).
    target_idx = 1
    use_tunneling = _CONDITION_ATTENTION_MODE == _MODALITY_TUNNELING_ATTENTION and len(states) > target_idx + 1
    if use_tunneling:
        self_idx = [0, target_idx]
        cross_idx = list(range(target_idx + 1, len(states)))
        self_keys = torch.cat([keys[i] for i in self_idx], dim=1).transpose(1, 2)
        self_values = torch.cat([values[i] for i in self_idx], dim=1).transpose(1, 2)
        cross_keys = torch.cat([keys[i] for i in cross_idx], dim=1).transpose(1, 2)
        cross_values = torch.cat([values[i] for i in cross_idx], dim=1).transpose(1, 2)
    outputs = []
    for index, (state, query, mlp, mod, adapter) in enumerate(zip(states, queries, mlps, mods, adapters)):
        q = query.transpose(1, 2)
        if use_tunneling and index == target_idx:
            attention = F.scaled_dot_product_attention(q, self_keys, self_values) + \
                F.scaled_dot_product_attention(q, cross_keys, cross_values)
        else:
            attention = F.scaled_dot_product_attention(q, all_keys, all_values)
        attention = attention.transpose(1, 2).flatten(2, 3).to(query.dtype)
        with specify_lora((block.attn.to_out,), adapter):
            update = block.attn.to_out(torch.cat([attention, mlp], dim=-1))
        outputs.append(state + mod[2] * update)
    return outputs


def apply_audio_modulation_state(state: torch.Tensor, audio_modulation):
    if audio_modulation is None:
        return state
    scale, shift, gate = audio_modulation
    gate = gate.to(device=state.device, dtype=state.dtype)
    scale = torch.tanh(scale).to(device=state.device, dtype=state.dtype).unsqueeze(1)
    shift = shift.to(device=state.device, dtype=state.dtype).unsqueeze(1)
    return state * (1 + gate * scale) + gate * shift


def split_flux2_modulation(modulation, parameter_sets: int):
    """Normalize both released FLUX.2 modulation APIs to tuple form.

    Earlier Diffusers revisions returned tuples of ``(shift, scale, gate)``
    directly.  Current revisions keep the same values in one tensor and split
    them inside the stock transformer block.  TunnelingNet runs the block
    branches explicitly, so it performs that purely structural split here.
    """
    if not torch.is_tensor(modulation):
        return modulation
    if modulation.ndim == 2:
        modulation = modulation.unsqueeze(1)
    values = torch.chunk(modulation, 3 * parameter_sets, dim=-1)
    return tuple(
        values[3 * index : 3 * (index + 1)]
        for index in range(parameter_sets)
    )


def flux2_tunneling_forward(transformer, target, conditions, reference, prompt_embeds, target_ids, condition_ids, reference_ids, text_ids, timestep, specify_lora, use_checkpoint, audio_modulation=None):
    # Official Omini routing generalized to FLUX.2. Multiple condition branches
    # are supported and all use the same trainable "default" adapter, matching
    # TunnelingNet multi-condition routing. The target lane (adapters[1])
    # additionally carries the "default" adapter when _TRAINABLE_TARGET is set, so
    # it can learn to produce new modalities instead of drifting to the mean; see
    # set_trainable_target. proj_out has no Channel_Head so it stays base regardless.
    if not isinstance(conditions, (list, tuple)):
        conditions = [conditions]
    if not isinstance(condition_ids, (list, tuple)):
        condition_ids = [condition_ids]
    filtered = [(cond, ids) for cond, ids in zip(conditions, condition_ids) if cond is not None and ids is not None]
    conditions = [item[0] for item in filtered]
    condition_ids = [item[1] for item in filtered]
    has_reference = reference is not None
    target_adapter = "default" if _TRAINABLE_TARGET else None
    if _TRAINABLE_CONDITION and _SEPARATE_CONDITION_ADAPTERS:
        condition_adapters = [f"condition_{index}" for index in range(len(conditions))]
    else:
        condition_adapter = "default" if _TRAINABLE_CONDITION else None
        condition_adapters = [condition_adapter] * len(conditions)
    adapters = [None, target_adapter] + condition_adapters + ([None] if has_reference else [])
    image_inputs = [target, *conditions] + ([reference] if has_reference else [])
    image_ids = [target_ids, *condition_ids] + ([reference_ids] if has_reference else [])
    target_temb = transformer.time_guidance_embed(timestep.to(target.dtype) * 1000, None)
    clean_temb = transformer.time_guidance_embed(torch.zeros_like(timestep).to(target.dtype), None)
    tembs = [target_temb, target_temb] + [clean_temb] * len(conditions) + ([target_temb] if has_reference else [])
    image_states = []
    for value, adapter in zip(image_inputs, adapters[1:]):
        with specify_lora((transformer.x_embedder,), adapter):
            image_states.append(transformer.x_embedder(value))
    image_states[0] = apply_audio_modulation_state(image_states[0], audio_modulation)
    with specify_lora((transformer.context_embedder,), adapters[0]):
        text_states = [transformer.context_embedder(prompt_embeds)]
    image_mods = []
    for temb, adapter in zip(tembs[1:], adapters[1:]):
        with specify_lora((transformer.double_stream_modulation_img.linear,), adapter):
            image_mods.append(split_flux2_modulation(
                transformer.double_stream_modulation_img(temb), 2,
            ))
    with specify_lora((transformer.double_stream_modulation_txt.linear,), adapters[0]):
        text_mods = [split_flux2_modulation(
            transformer.double_stream_modulation_txt(tembs[0]), 2,
        )]
    position_embs = [transformer.pos_embed(text_ids[0] if text_ids.ndim == 3 else text_ids)]
    position_embs.extend(transformer.pos_embed(ids[0] if ids.ndim == 3 else ids) for ids in image_ids)
    n_image_states = len(image_states)
    for block in transformer.transformer_blocks:
        fn = lambda *values, block=block: double_block_forward(
            block, values[:n_image_states], values[n_image_states:], image_mods, text_mods, adapters, position_embs, specify_lora
        )
        if use_checkpoint and torch.is_grad_enabled():
            image_states, text_states = checkpoint(fn, *image_states, *text_states, use_reentrant=False)
        else:
            image_states, text_states = double_block_forward(
                block, image_states, text_states, image_mods, text_mods, adapters, position_embs, specify_lora
            )
    states = [*text_states, *image_states]
    single_mods = []
    for temb, adapter in zip(tembs, adapters):
        with specify_lora((transformer.single_stream_modulation.linear,), adapter):
            single_mods.append(split_flux2_modulation(
                transformer.single_stream_modulation(temb), 1,
            )[0])
    for block in transformer.single_transformer_blocks:
        fn = lambda *values, block=block: single_block_forward(block, list(values), single_mods, adapters, position_embs, specify_lora)
        states = checkpoint(fn, *states, use_reentrant=False) if use_checkpoint and torch.is_grad_enabled() else fn(*states)
    target_hidden = states[1]
    with specify_lora((transformer.proj_out,), None):
        return transformer.proj_out(transformer.norm_out(target_hidden, target_temb))

class Flux2TunnelingNetModel(L.LightningModule):
    """TunnelingNet model lifecycle for FLUX.2 Channel Head training."""
    def __init__(self, args, optimizer_config, specify_lora):
        super().__init__()
        self.args = args
        self.optimizer_config = optimizer_config
        self.specify_lora = specify_lora
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        self.flux_pipe = Flux2KleinPipeline.from_pretrained(
            str(resolve_path(args.model_path)), torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.transformer = self.flux_pipe.transformer.to(device)
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self._move_frozen_component("text_encoder", "cpu")
        self._move_frozen_component("vae", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.transformer.requires_grad_(False)
        target_modules = (
            r"(^x_embedder$|^double_stream_modulation_(img|txt)\.linear$|"
            r"^single_stream_modulation\.linear$|"
            r"^(?<!single_)transformer_blocks\.[0-9]+\.attn\."
            r"(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_add_out|to_out\.0)$|"
            r"^(?<!single_)transformer_blocks\.[0-9]+\.ff(_context)?\.(linear_in|linear_out)$|"
            r"^single_transformer_blocks\.[0-9]+\.attn\.(to_qkv_mlp_proj|to_out)$)"
        )
        channel_head_alpha = args.channel_head_rank if args.channel_head_alpha is None else args.channel_head_alpha
        lora_config = LoraConfig(
            r=args.channel_head_rank,
            lora_alpha=channel_head_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.transformer.add_adapter(lora_config, adapter_name="default")
        self.extra_lora_adapter_names = []
        if getattr(args, "separate_condition_adapters", False):
            # Current audio-dense tasks need two lanes (current RGB + Audio_AST).
            # Keep a few spare adapters for future 3-condition experiments; unused
            # adapters receive no gradient.
            self.extra_lora_adapter_names = [f"condition_{idx}" for idx in range(4)]
            for adapter_name in self.extra_lora_adapter_names:
                self.transformer.add_adapter(lora_config, adapter_name=adapter_name)
            try:
                self.transformer.set_adapter(["default", *self.extra_lora_adapter_names])
            except Exception:
                self.transformer.set_adapter("default")
        self.audio_condition_projector = None
        self.audio_modulation_projector = None
        self.audio_modulation_gate = None
        if int(getattr(args, "audio_condition_dim", 0) or 0) > 0:
            packed_dim = int(self.transformer.x_embedder.in_features)
            hidden_dim = int(self.transformer.x_embedder.out_features)
            audio_dim = int(args.audio_condition_dim)
            audio_mode = getattr(args, "audio_conditioning_mode", "attention")
            if audio_mode == "modulation":
                self.audio_modulation_projector = nn.Sequential(
                    nn.LayerNorm(audio_dim),
                    nn.Linear(audio_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                ).to(device)
                self.audio_modulation_gate = nn.Parameter(torch.tensor(-4.0, device=device))
            else:
                self.audio_condition_projector = nn.Sequential(
                    nn.LayerNorm(audio_dim),
                    nn.Linear(audio_dim, packed_dim),
                    nn.SiLU(),
                    nn.Linear(packed_dim, packed_dim),
                ).to(device)
        self.channel_head_layers = [parameter for parameter in self.transformer.parameters() if parameter.requires_grad]
        if self.audio_condition_projector is not None:
            self.channel_head_layers.extend(self.audio_condition_projector.parameters())
        if self.audio_modulation_projector is not None:
            self.channel_head_layers.extend(self.audio_modulation_projector.parameters())
        if self.audio_modulation_gate is not None:
            self.channel_head_layers.append(self.audio_modulation_gate)
        if not self.channel_head_layers:
            raise RuntimeError("FLUX.2 Channel_Head regex selected no parameters")

        # Load the same portable normalization contract used by the WSG dataset.
        # Pseudo RGB needs channel ranges; WSG-reader previews need the full
        # linear/log inverse transform so readers receive original-scale values.
        self.channel_ranges = {}
        self.normalization_contract = {"__default__": "linear"}
        self._wsg_denormalizers = {}
        if hasattr(args, "modality_config"):
            try:
                config_path = resolve_path(args.modality_config)
                if load_wsg_normalization_ranges is not None:
                    self.channel_ranges = load_wsg_normalization_ranges(config_path)
                else:
                    import json
                    with config_path.open("r", encoding="utf-8") as f:
                        config = json.load(f)
                    ranges = (config.get("normalization") or {}).get("channel_ranges") or {}
                    for channel, value_range in ranges.items():
                        if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
                            self.channel_ranges[int(channel)] = (float(value_range[0]), float(value_range[1]))
                if load_wsg_normalization_contract is not None:
                    self.normalization_contract = load_wsg_normalization_contract(config_path)
            except Exception:
                pass  # Fallback to RGB-only visualization if config loading fails

    def _move_frozen_component(self, name: str, device: str | torch.device) -> None:
        module = getattr(self.flux_pipe, name, None)
        if module is not None:
            module.to(device)

    def _encode_prompt_offloaded(self, prompts: list[str], dtype: torch.dtype):
        self._move_frozen_component("text_encoder", self.device)
        try:
            with torch.no_grad():
                prompt_embeds, text_ids = self.flux_pipe.encode_prompt(
                    prompts, device=self.device, max_sequence_length=self.args.max_sequence_length
                )
            return prompt_embeds.to(dtype=dtype), text_ids
        finally:
            self._move_frozen_component("text_encoder", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def normalize_and_pack(pipe, latents):
        if latents.ndim == 5:
            latents = latents[:, :, 0]
        latents = pipe._patchify_latents(latents)
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + pipe.vae.config.batch_norm_eps)
        latents = (latents - mean) / std
        ids = pipe._prepare_latent_ids(latents).to(latents.device)
        return pipe._pack_latents(latents), ids

    def append_audio_condition(self, batch, condition_cleans, condition_ids, target_clean, target_ids, dtype):
        feature = batch.get("audio_condition_feature")
        if feature is None or getattr(self.args, "audio_conditioning_mode", "attention") != "attention":
            return condition_cleans, condition_ids
        if self.audio_condition_projector is None:
            raise RuntimeError("Batch contains audio_condition_feature but audio_condition_projector is disabled; set --audio-condition-dim.")
        audio = feature.to(self.device, dtype=torch.float32)
        packed = self.audio_condition_projector(audio).to(dtype).unsqueeze(1)
        if getattr(self.args, "audio_global_condition", False):
            ids = target_ids[:, :1].clone()
            if ids.shape[-1] > 1:
                ids[..., 1:] = 0
            packed = packed.contiguous()
        else:
            packed = packed.expand(-1, target_clean.shape[1], -1).contiguous()
            ids = target_ids.clone()
        condition_cleans.append(packed)
        condition_ids.append(ids)
        return condition_cleans, condition_ids

    def build_audio_modulation(self, batch, dtype):
        if getattr(self.args, "audio_conditioning_mode", "attention") != "modulation":
            return None
        feature = batch.get("audio_condition_feature")
        if feature is None:
            return None
        if self.audio_modulation_projector is None or self.audio_modulation_gate is None:
            raise RuntimeError("Audio modulation requested but audio_modulation_projector is disabled; set --audio-condition-dim and --audio-conditioning-mode modulation.")
        audio = feature.to(self.device, dtype=torch.float32)
        scale_shift = self.audio_modulation_projector(audio).to(dtype)
        scale, shift = scale_shift.chunk(2, dim=-1)
        gate = torch.sigmoid(self.audio_modulation_gate)
        return scale, shift, gate

    def configure_optimizers(self):
        for parameter in self.channel_head_layers:
            parameter.requires_grad_(True)
        return torch.optim.AdamW(
            self.channel_head_layers,
            lr=self.optimizer_config["params"]["lr"],
            weight_decay=self.optimizer_config["params"].get("weight_decay", 0.01),
        )

    def save_channel_head(self, path):
        Flux2KleinPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer, adapter_name="default"),
            safe_serialization=True,
        )
        if getattr(self, "extra_lora_adapter_names", None):
            torch.save(
                {name: get_peft_model_state_dict(self.transformer, adapter_name=name)
                 for name in self.extra_lora_adapter_names},
                Path(path) / "condition_lora_adapters.pt",
            )
        if self.audio_condition_projector is not None:
            torch.save(self.audio_condition_projector.state_dict(), Path(path) / "audio_condition_projector.pt")
        if self.audio_modulation_projector is not None:
            torch.save(self.audio_modulation_projector.state_dict(), Path(path) / "audio_modulation_projector.pt")
        if self.audio_modulation_gate is not None:
            torch.save(self.audio_modulation_gate.detach().cpu(), Path(path) / "audio_modulation_gate.pt")

    def decode_raw_latents(
        self,
        latents: torch.Tensor,
        modality: str | None = None,
        use_denorm: bool = True
    ) -> Image.Image:
        """Decode latents with optional denormalization for pseudo-visualization."""
        if latents.ndim == 5:
            latents = latents[:, :, 0]
        self._move_frozen_component("vae", self.device)
        try:
            vae_dtype = next(self.flux_pipe.vae.parameters()).dtype
            with torch.no_grad():
                decoded = self.flux_pipe.vae.decode(latents.to(self.device, dtype=vae_dtype)).sample
        finally:
            self._move_frozen_component("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if use_denorm and modality and hasattr(self, "channel_ranges"):
            # Parse base channel from modality (e.g., "0-2" -> 0, "36-38" -> 36)
            parts = modality.split("-")
            try:
                base_channel = int(parts[0])
                normalization_mode = normalization_mode_for_modality(
                    self.normalization_contract, modality
                ) if normalization_mode_for_modality is not None else "linear"
                return tensor_to_pil_with_denorm(
                    decoded[0],
                    self.channel_ranges,
                    base_channel,
                    modality,
                    normalization_mode=normalization_mode,
                )
            except (ValueError, IndexError):
                pass

        return tensor_to_pil(decoded[0])

    def unpack_normalized_latents(self, packed: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        latents = self.flux_pipe._unpack_latents_with_ids(packed, ids)
        mean = self.flux_pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(
            self.flux_pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            + self.flux_pipe.vae.config.batch_norm_eps
        )
        latents = latents * std + mean
        return self.flux_pipe._unpatchify_latents(latents)

    def decode_raw_latents_to_wsg_array(self, latents: torch.Tensor, modality: str) -> np.ndarray | None:
        """Decode target latents into a sparse 256-channel WSG array for reader visualization."""
        if not modality:
            return None
        if latents.ndim == 5:
            latents = latents[:, :, 0]
        self._move_frozen_component("vae", self.device)
        try:
            vae_dtype = next(self.flux_pipe.vae.parameters()).dtype
            with torch.no_grad():
                decoded = self.flux_pipe.vae.decode(latents.to(self.device, dtype=vae_dtype)).sample[0]
        finally:
            self._move_frozen_component("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        decoded = decoded.detach().float().cpu()
        if modality in {"111", "195"}:
            rgb = ((decoded[:3] + 1.0) * 127.5).clamp(0, 255).permute(1, 2, 0).numpy()
            wsg = np.full((rgb.shape[0], rgb.shape[1], 256), -1.0, dtype=np.float32)
            channel = int(modality)
            colors = AVIS_CATEGORY_COLORS if channel == 195 else None
            if colors is None:
                wsg[..., channel] = categorical_rgb_to_ids(rgb)
            else:
                wsg[..., channel] = categorical_rgb_to_ids(rgb, colors)
            return wsg

        image01 = (decoded + 1.0) / 2.0
        channels = parse_modality_channels(modality, 0)
        if not channels:
            return None

        height, width = int(decoded.shape[-2]), int(decoded.shape[-1])
        # Sparse WSG lanes use -1 as the absent-value sentinel. Zero-filling
        # makes every unpredicted lane look like valid data to downstream readers.
        wsg = np.full((height, width, 256), -1.0, dtype=np.float32)
        for i, channel in enumerate(channels[: min(len(channels), decoded.shape[0])]):
            if channel < 0 or channel >= 256:
                continue
            normalized = image01[i].numpy().astype(np.float32, copy=False)
            if WSGNormalizationStore is not None and normalization_mode_for_modality is not None:
                mode = normalization_mode_for_modality(self.normalization_contract, modality)
                normalizer = self._wsg_denormalizers.get(mode)
                if normalizer is None:
                    normalizer = WSGNormalizationStore(
                        normalization_mode=mode,
                        portable_channel_ranges=self.channel_ranges,
                    )
                    self._wsg_denormalizers[mode] = normalizer
                values = normalizer.denormalize_channel(normalized, channel, [])
            elif hasattr(self, "channel_ranges") and channel in self.channel_ranges:
                lo, hi = self.channel_ranges[channel]
                values = normalized * (float(hi) - float(lo)) + float(lo)
            else:
                values = normalized * 255.0
            wsg[..., channel] = values.astype(np.float32, copy=False)
        if modality == "41-42" and decoded.shape[0] >= 2:
            # Restore the paired WSG sentinel before Reader visualization.
            normalized_pair = image01[:2].numpy()
            invalid_flow = np.all(normalized_pair <= 0.03, axis=0)
            wsg[..., 41][invalid_flow] = -1.0
            wsg[..., 42][invalid_flow] = -1.0
        elif modality == "36-38" and decoded.shape[0] >= 3:
            # Padded normal pixels use the same all-low VAE sentinel. Without
            # restoring it, a zero/near-zero vector maps to neutral gray and
            # creates a white haze around otherwise valid surface normals.
            normalized_triplet = image01[:3].numpy()
            invalid_normal = np.all(normalized_triplet <= 0.03, axis=0)
            for channel in (36, 37, 38):
                wsg[..., channel][invalid_normal] = -1.0
        return wsg

    def save_wsg_reader_visualization(
        self,
        items: list[tuple[str, torch.Tensor]],
        modality: str,
        pseudo_out_dir: Path,
    ) -> None:
        enabled = os.environ.get("WSG_READER_VISUALIZATION", "1") not in {"0", "false", "False"}
        required = os.environ.get("WSG_READER_REQUIRED", "0") in {"1", "true", "True"}
        if not enabled:
            if required:
                raise RuntimeError("Formal evaluation requires WSG Reader visualization, but it is disabled.")
            return
        if visualize_single_wsg is None:
            if required:
                raise RuntimeError("Formal evaluation requires WSG Reader, but its visualizer could not be imported.")
            return

        # Keep the authoritative Reader outputs beside the sample that produced
        # them.  The old path inference depended on a fixed directory depth and
        # silently misplaced benchmark/sample_xx artifacts.
        reader_dir = pseudo_out_dir / "wsg_reader"
        raw_dir = reader_dir / "raw"
        tensors_dir = reader_dir / "tensors"
        raw_dir.mkdir(parents=True, exist_ok=True)
        tensors_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = reader_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            manifest = {"schema_version": 1, "renderer": "single_wsg_visualization", "items": {}}

        for name, latents in items:
            try:
                wsg = self.decode_raw_latents_to_wsg_array(latents, modality)
                if wsg is None:
                    raise RuntimeError(f"Could not decode modality {modality!r} to a WSG array.")
                # Persist the reversible, denormalized WSG payload. Formal
                # figures can now be re-rendered by future Reader versions
                # without loading the model or repeating diffusion inference.
                wsg_archive = tensors_dir / f"{name}.wsg.npz"
                np.savez_compressed(
                    wsg_archive,
                    wsg_data=wsg,
                    modality=np.asarray(str(modality)),
                    renderer_contract=np.asarray("denormalized_wsg_v1"),
                )
                # Preserve the VAE-decoded [0, 1] lanes as a separate audit
                # artifact.  This is important for relative quantities such as
                # channel 32: the portable Reader contract maps that lane back
                # to its dataset range, while the direct network output remains
                # the normalized representation seen during training.
                normalized_wsg = np.full_like(wsg, -1.0, dtype=np.float32)
                mode = normalization_mode_for_modality(
                    self.normalization_contract, modality
                ) if normalization_mode_for_modality is not None else "linear"
                normalizer = self._wsg_denormalizers.get(mode)
                if normalizer is None and WSGNormalizationStore is not None:
                    normalizer = WSGNormalizationStore(
                        normalization_mode=mode,
                        portable_channel_ranges=self.channel_ranges,
                    )
                    self._wsg_denormalizers[mode] = normalizer
                for channel in parse_modality_channels(modality, 0):
                    if channel >= wsg.shape[-1]:
                        continue
                    if normalizer is not None:
                        normalized_wsg[..., channel] = normalizer.normalize_channel(
                            wsg[..., channel], channel, []
                        )
                    elif channel in self.channel_ranges:
                        lo, hi = self.channel_ranges[channel]
                        normalized_wsg[..., channel] = np.clip(
                            (wsg[..., channel] - float(lo)) /
                            max(float(hi) - float(lo), 1e-6),
                            0.0, 1.0,
                        )
                normalized_archive = tensors_dir / f"{name}.normalized.wsg.npz"
                np.savez_compressed(
                    normalized_archive,
                    wsg_data=normalized_wsg,
                    modality=np.asarray(str(modality)),
                    renderer_contract=np.asarray("normalized_wsg_v1"),
                )
                if os.environ.get("WSG_READER_TENSORS_ONLY", "0") in {
                    "1", "true", "True"
                }:
                    manifest["items"][name] = {
                        "modality": modality,
                        "denormalized_wsg": str(wsg_archive.relative_to(reader_dir)),
                        "normalized_wsg": str(normalized_archive.relative_to(reader_dir)),
                        "rendering": "skipped_tensors_only",
                    }
                    continue
                with tempfile.TemporaryDirectory(prefix="wsg_reader_vis_") as tmp:
                    wsg_path = Path(tmp) / f"{name}.wsg.npz"
                    np.savez_compressed(wsg_path, wsg_data=wsg)
                    item_dir = raw_dir / name
                    visualize_single_wsg(
                        wsg_path, item_dir,
                        declared_channels=parse_modality_channels(modality, 0),
                    )
                primary = self._select_wsg_reader_primary(item_dir, modality)
                if primary is None:
                    raise WSGReaderNoPrimaryOutput(
                        f"WSG Reader produced no primary PNG for {name} modality={modality!r}."
                    )
                canonical = reader_dir / f"{name}.png"
                shutil.copy2(primary, canonical)
                manifest["items"][name] = {
                    "modality": modality,
                    "canonical": canonical.name,
                    "reader_source": str(primary.relative_to(reader_dir)),
                    "denormalized_wsg": str(wsg_archive.relative_to(reader_dir)),
                    "normalized_wsg": str(normalized_archive.relative_to(reader_dir)),
                }
            except Exception as exc:
                if required:
                    if isinstance(exc, WSGReaderNoPrimaryOutput):
                        raise
                    raise RuntimeError(
                        f"Required WSG Reader visualization failed for {name} ({modality}): {exc}"
                    ) from exc
                print(f"[WSM_v1] WSG reader visualization skipped for {name}: {exc}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _select_wsg_reader_primary(item_dir: Path, modality: str) -> Path | None:
        files = sorted(item_dir.rglob("*.png"))
        if not files:
            return None
        channels = parse_modality_channels(modality, 0)
        channel_tokens = tuple(f"channel{channel}" for channel in channels)

        def score(path: Path) -> tuple[int, str]:
            name = path.name.lower()
            value = 0
            if modality == "0-2" and "front_rgb" in name:
                value += 120
            if "semantic_mask" in name:
                value += 110
            if channel_tokens and any(token in name for token in channel_tokens):
                value += 80
            if name.endswith("_map.png"):
                value += 50
            if any(token in name for token in ("normal", "depth", "flow", "edge", "pose", "mask")):
                value += 30
            if "overlay" in name:
                value -= 60
            if name == "summary.png" or "summary" in name:
                value -= 100
            return value, str(path)

        return max(files, key=score)

    @staticmethod
    def _first_batch_value(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return value.flatten()[0].item()
        return value

    @staticmethod
    def _batch_crop_info(value) -> dict | None:
        if not isinstance(value, dict):
            return None
        crop_info = {}
        for key, raw in value.items():
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            if torch.is_tensor(raw):
                raw = raw.flatten()[0].item() if raw.numel() else None
            if raw is not None:
                crop_info[key] = raw
        required = {"patch_hw", "crop_y", "crop_x"}
        return crop_info if required.issubset(crop_info) else None

    @staticmethod
    def _apply_spatial_crop(frame: np.ndarray, crop_info: dict | None) -> np.ndarray:
        if not crop_info:
            return frame
        patch_hw = int(crop_info["patch_hw"])
        crop_y = int(crop_info["crop_y"])
        crop_x = int(crop_info["crop_x"])
        h, w = frame.shape[:2]
        if h < patch_hw or w < patch_hw:
            pad_width = ((0, max(0, patch_hw - h)), (0, max(0, patch_hw - w)), (0, 0))
            frame = np.pad(frame, pad_width, mode="constant", constant_values=-1.0)
        return frame[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :]

    @staticmethod
    def _render_avis_semantic_mask(values: np.ndarray) -> Image.Image:
        semantic = np.asarray(values, dtype=np.float32)
        return Image.fromarray(categorical_ids_to_rgb(semantic, AVIS_CATEGORY_COLORS), mode="RGB")

    def load_original_audio_mask_image(self, batch: dict[str, Any]) -> Image.Image | None:
        """Load the true channel-195 mask from the source WSV frame.

        For audio-dense mask training, the latent GT is useful for loss but a bad
        visualization reference: VAE round-trip turns a discrete semantic channel
        into continuous category mixtures. The visual GT should come from the
        original WSV frame instead.
        """
        wsv_path = self._first_batch_value(batch.get("wsv_path"))
        frame_key = self._first_batch_value(batch.get("frame_key"))
        if not wsv_path or not frame_key:
            return None
        try:
            with np.load(str(wsv_path), allow_pickle=False) as data:
                if str(frame_key) not in data:
                    return None
                frame = data[str(frame_key)].astype(np.float32, copy=False)
            if frame.ndim != 3 or frame.shape[-1] <= 195:
                return None
            frame = self._apply_spatial_crop(frame, self._batch_crop_info(batch.get("crop_info")))
            return self._render_avis_semantic_mask(frame[..., 195])
        except Exception as exc:
            print(f"[WSM_v1] original WSV GT mask load skipped: {exc}", flush=True)
            return None

    def maybe_original_audio_mask_image(self, batch: dict[str, Any]) -> Image.Image | None:
        if getattr(self.args, "target_visual_modality", ""):
            return None
        if getattr(self.args, "target_modality", "") != "195" or not getattr(self.args, "audio_dense_condition_train", ""):
            return None
        return self.load_original_audio_mask_image(batch)

    def save_audio_mask_reader_comparison(
        self,
        pseudo_out_dir: Path,
        condition_image: Image.Image | None,
        pred_name: str,
        gt_name: str,
        batch: dict[str, Any] | None = None,
    ) -> None:
        if condition_image is None or os.environ.get("WSG_READER_VISUALIZATION", "1") in {"0", "false", "False"}:
            return
        if getattr(self.args, "target_modality", "") != "195" or not getattr(self.args, "audio_dense_condition_train", ""):
            return
        parent_name = pseudo_out_dir.parent.name
        if parent_name == "pseudo-visualization-inference":
            reader_root = pseudo_out_dir.parent.parent / "wsg-reader-visualization-inference"
        elif parent_name == "pseudo-visualization":
            reader_root = pseudo_out_dir.parent.parent / "wsg-reader-visualization"
        else:
            reader_root = pseudo_out_dir.parent.parent / f"wsg-reader-{parent_name}"
        reader_dir = reader_root / pseudo_out_dir.name
        pred_path = reader_dir / pred_name / "Audio" / f"{pred_name}.wsg_Audio_semantic_mask.png"
        latent_gt_path = reader_dir / gt_name / "Audio" / f"{gt_name}.wsg_Audio_semantic_mask.png"
        if not pred_path.exists():
            return
        try:
            if not latent_gt_path.exists():
                return
            condition = condition_image.convert("RGB")
            panels = [
                ("Current RGB condition", condition),
                ("Predicted semantic mask", Image.open(pred_path).convert("RGB").resize(condition.size, Image.Resampling.NEAREST)),
                ("Latent-decoded GT semantic mask", Image.open(latent_gt_path).convert("RGB").resize(condition.size, Image.Resampling.NEAREST)),
            ]
            original_gt = self.load_original_audio_mask_image(batch or {})
            if original_gt is not None:
                gt_dir = reader_dir / "original_gt" / "Audio"
                gt_dir.mkdir(parents=True, exist_ok=True)
                original_gt_path = gt_dir / "original_gt.wsg_Audio_semantic_mask.png"
                original_gt.save(original_gt_path)
                panels.append(("Original WSV GT debug", original_gt.convert("RGB").resize(condition.size, Image.Resampling.NEAREST)))
            fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 4, 4.6))
            for ax, (title, img) in zip(axes, panels):
                ax.imshow(img)
                ax.set_title(title, fontsize=11)
                ax.axis("off")
            fig.tight_layout()
            out_path = reader_dir / "summary_rgb_pred_gt.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception as exc:
            print(f"[WSM_v1] audio mask comparison visualization skipped: {exc}", flush=True)

    def sample_training_noise(self, target_clean: torch.Tensor, target_ids: torch.Tensor,
                              timestep: torch.Tensor, generator, use_multires: bool = True) -> torch.Tensor:
        if not _MULTIRES_NOISE or not use_multires:
            return torch.randn(target_clean.shape, generator=generator,
                               device=target_clean.device, dtype=target_clean.dtype)

        spatial = self.flux_pipe._unpack_latents_with_ids(target_clean, target_ids)
        noise = torch.randn(spatial.shape, generator=generator,
                            device=spatial.device, dtype=spatial.dtype)
        height, width = spatial.shape[-2:]
        anneal = timestep.float().view(-1, 1, 1, 1)
        for level in range(1, _MULTIRES_LEVELS):
            scale = 2 ** level
            coarse = torch.randn(
                (spatial.shape[0], spatial.shape[1], max(1, height // scale), max(1, width // scale)),
                generator=generator, device=spatial.device, dtype=spatial.dtype,
            )
            coarse = F.interpolate(coarse.float(), size=(height, width), mode="bilinear", align_corners=False)
            weight = (_MULTIRES_DISCOUNT ** level) * anneal.pow(level)
            noise = noise + coarse.to(noise.dtype) * weight.to(noise.dtype)

        reduce_dims = tuple(range(1, noise.ndim))
        noise = noise / noise.float().std(dim=reduce_dims, keepdim=True).clamp_min(1e-6).to(noise.dtype)
        return self.flux_pipe._pack_latents(noise)

    @torch.no_grad()
    def run_inference(self, batch: dict[str, Any], out_dir: Path, label: str, num_steps: int = 10) -> None:
        """True inference with multi-step denoising from pure noise."""
        dtype = torch.bfloat16
        normal_raw = batch.get("omini_condition_latent")
        normal_raw = normal_raw.to(self.device, dtype=dtype) if normal_raw is not None else None
        reference_raw = batch.get("native_reference_latent")
        reference_raw = reference_raw.to(self.device, dtype=dtype) if reference_raw is not None else None
        extra_raw = batch.get("extra_condition_latent")
        extra_raw = extra_raw.to(self.device, dtype=dtype) if extra_raw is not None else None
        target_raw = batch["target_latent"].to(self.device, dtype=dtype)  # For GT comparison only
        prompts = list(batch["description"])

        target_clean, target_ids = self.normalize_and_pack(self.flux_pipe, target_raw)
        if normal_raw is not None:
            normal_clean, normal_ids = self.normalize_and_pack(self.flux_pipe, normal_raw)
        else:
            normal_clean, normal_ids = None, None
        if reference_raw is not None:
            reference_clean, reference_ids = self.normalize_and_pack(self.flux_pipe, reference_raw)
            reference_ids = reference_ids.clone(); reference_ids[..., 0] = 10
        else:
            reference_clean, reference_ids = None, None
        condition_cleans = [] if normal_clean is None else [normal_clean]
        condition_ids = [] if normal_ids is None else [normal_ids]
        if extra_raw is not None:
            extra_clean, extra_ids = self.normalize_and_pack(self.flux_pipe, extra_raw)
            extra_ids = extra_ids.clone(); extra_ids[..., 0] = 10
            condition_cleans.append(extra_clean)
            condition_ids.append(extra_ids)
        condition_cleans, condition_ids = self.append_audio_condition(batch, condition_cleans, condition_ids, target_clean, target_ids, dtype)
        audio_modulation = self.build_audio_modulation(batch, dtype)

        if "prompt_embeds" in batch and "text_ids" in batch:
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=dtype)
            text_ids = batch["text_ids"].to(self.device)
        else:
            prompt_embeds, text_ids = self._encode_prompt_offloaded(prompts, dtype)
        # Start from pure noise
        generator = torch.Generator(device=self.device).manual_seed(self.args.seed + 17)
        target_shape = target_clean.shape
        latents = torch.randn(target_shape, generator=generator, device=self.device, dtype=dtype)

        # Single-step data prediction or the original multi-step flow sampler.
        timesteps = (torch.ones(1, device=self.device, dtype=dtype) if _SINGLE_STEP_X0 else
                     torch.linspace(1.0, 0.0, num_steps + 1, device=self.device, dtype=dtype)[:-1])
        for i, t in enumerate(timesteps):
            timestep = torch.full((latents.shape[0],), t, device=self.device, dtype=dtype)

            # Predict noise
            prediction = flux2_tunneling_forward(
                self.transformer, latents, condition_cleans, reference_clean, prompt_embeds,
                target_ids, condition_ids, reference_ids, text_ids, timestep,
                self.specify_lora, False, audio_modulation=audio_modulation,
            )

            if _SINGLE_STEP_X0:
                latents = prediction
            elif i < len(timesteps) - 1:
                dt = float(timesteps[i] - timesteps[i + 1])
                latents = latents - dt * prediction
            else:
                # Last step: predict clean
                latents = latents - timestep[:, None, None] * prediction

        generated_raw = self.unpack_normalized_latents(latents, target_ids)

        if bool(getattr(self.args, "inference_save_raw_latent", False)):
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_id = str(batch.get("sample_id", [""])[0]) if "sample_id" in batch else ""
            video_id_for_cache = str(batch.get("video_id", [""])[0]) if "video_id" in batch else ""
            frame_for_cache = int(batch.get("target_frame", [-1])[0]) if "target_frame" in batch else -1
            torch.save({
                "schema_version": 1,
                "sample_id": sample_id,
                "video_id": video_id_for_cache,
                "frame_index": frame_for_cache,
                "target_modality": str(getattr(self.args, "target_modality", "")),
                "condition_modalities": str(getattr(self.args, "condition_modalities", "")),
                "inference_steps": int(num_steps),
                "seed": int(getattr(self.args, "seed", 0)),
                "latent": generated_raw.detach().to(device="cpu").contiguous(),
            }, out_dir / "generated_raw_latent.pt")
            if bool(getattr(self.args, "inference_cache_only", False)):
                with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                    json.dump({
                        "label": label,
                        "sample_id": sample_id,
                        "video_id": video_id_for_cache,
                        "target_frame": frame_for_cache,
                        "num_inference_steps": int(num_steps),
                        "method": f"{num_steps}-step Euler from pure noise",
                        "artifact": "generated_raw_latent.pt",
                    }, handle, indent=2, ensure_ascii=False)
                return

        # Get modality names for proper denormalization. For audio-dense mask
        # tasks the first declared condition is Audio_AST, while normal_raw holds
        # the current-frame RGB latent. Decode that visual condition as RGB.
        reference_modality = "0-2"
        condition_tokens = [item.strip() for item in getattr(self.args, "condition_modalities", "").split(",") if item.strip()]
        condition_modality = "0-2" if "CurrentFrame_RGB_Condition" in condition_tokens else (condition_tokens[0] if condition_tokens else "")
        target_modality = getattr(self.args, "target_visual_modality", "") or getattr(self.args, "target_modality", "")

        target_frame = int(batch.get("target_frame", [0])[0]) if "target_frame" in batch else None
        condition_frame = int(batch.get("condition_frame", [0])[0]) if "condition_frame" in batch else None
        condition_video_id = (
            str(batch.get("condition_video_id", [""])[0])
            if "condition_video_id" in batch else ""
        )
        reference_frame = int(batch.get("reference_frame", [0])[0]) if "reference_frame" in batch else None
        video_id = str(batch.get("video_id", [""])[0]) if "video_id" in batch else ""
        reference_video_id = (
            str(batch.get("reference_video_id", [""])[0])
            if "reference_video_id" in batch else ""
        )
        extra_condition_frame = (
            int(batch.get("extra_condition_frame", [0])[0])
            if "extra_condition_frame" in batch else None
        )
        extra_condition_video_id = (
            str(batch.get("extra_condition_video_id", [""])[0])
            if "extra_condition_video_id" in batch else ""
        )
        extra_condition_reference_kind = (
            str(batch.get("extra_condition_reference_kind", [""])[0])
            if "extra_condition_reference_kind" in batch else ""
        )

        images = []
        output_names = []
        if reference_raw is not None:
            images.append((f"reference f{reference_frame}" if reference_frame is not None else "reference",
                           self.decode_raw_latents(reference_raw, reference_modality)))
            output_names.append("reference")
        if extra_raw is not None:
            extra_modality = str(batch.get("extra_condition_modality", [""])[0]) if "extra_condition_modality" in batch else "0-2"
            extra_frame = int(batch.get("extra_condition_frame", [0])[0]) if "extra_condition_frame" in batch else None
            images.append((f"first-condition f{extra_frame}" if extra_frame is not None else "first-condition",
                           self.decode_raw_latents(extra_raw, extra_modality)))
            output_names.append("first_condition")
        if normal_raw is not None:
            images.append((f"condition f{target_frame}" if target_frame is not None else "condition",
                           self.decode_raw_latents(normal_raw, condition_modality)))
            output_names.append("condition")
        baseline_inference_image = self.decode_raw_latents(generated_raw, target_modality)
        inference_image = baseline_inference_image
        categorical_refinement = None
        refiner_runtime = getattr(self, "categorical_refiner_runtime", None)
        if refiner_runtime is not None:
            condition_latent_by_modality = {}
            if normal_raw is not None:
                condition_latent_by_modality[condition_modality] = normal_raw
            if reference_raw is not None:
                condition_latent_by_modality.setdefault("0-2", reference_raw)
            if extra_raw is not None:
                condition_latent_by_modality.setdefault(extra_modality, extra_raw)
            aliases = {
                "Reference_IMG": "0-2",
                "FirstFrame_RGB": "0-2",
                "CurrentFrame_RGB_Reference": "0-2",
                "CurrentFrame_RGB_Condition": "0-2",
            }
            required_latents = []
            for token in refiner_runtime.condition_modalities:
                canonical = aliases.get(token, token)
                if canonical not in condition_latent_by_modality:
                    raise RuntimeError(
                        f"Categorical refiner requires condition {token!r} ({canonical!r}); "
                        f"available={sorted(condition_latent_by_modality)}"
                    )
                required_latents.append(condition_latent_by_modality[canonical])
            categorical_refinement = refiner_runtime.refine(
                baseline_inference_image,
                required_latents,
                minimum_component_pixels=int(
                    getattr(self.args, "categorical_refiner_minimum_pixels", 32)
                ),
                audio_selector=getattr(self, "audio_region_selector_runtime", None),
                audio_feature=batch.get("audio_condition_feature"),
            )
            inference_image = categorical_refinement["image"]
        images.extend([
            (f"inference f{target_frame}" if target_frame is not None else "inference",
             inference_image),
            (f"gt f{target_frame}" if target_frame is not None else "gt",
             self.decode_raw_latents(target_raw, target_modality)),
        ])
        output_names.extend(["inference", "gt"])

        out_dir.mkdir(parents=True, exist_ok=True)
        if categorical_refinement is not None:
            baseline_inference_image.save(out_dir / "inference_unrefined.png")
        for name, image in zip(output_names, [item[1] for item in images]):
            image.save(out_dir / f"{name}.png")
        save_image_grid(images, out_dir / "summary.png")
        self.save_wsg_reader_visualization([("inference", generated_raw), ("gt", target_raw)], target_modality, out_dir)
        if normal_raw is not None and condition_modality:
            self.save_wsg_reader_visualization([("condition", normal_raw)], condition_modality, out_dir)
        if extra_raw is not None and extra_modality:
            self.save_wsg_reader_visualization([("first_condition", extra_raw)], extra_modality, out_dir)
        if reference_raw is not None:
            self.save_wsg_reader_visualization([("reference", reference_raw)], reference_modality, out_dir)
        condition_image = images[output_names.index("condition")][1] if "condition" in output_names else None
        self.save_audio_mask_reader_comparison(out_dir, condition_image, "inference", "gt", batch)

        metadata = {
            "label": label,
            "num_inference_steps": 1 if _SINGLE_STEP_X0 else num_steps,
            "method": "single-step direct x0" if _SINGLE_STEP_X0 else f"{num_steps}-step Euler from pure noise",
            "sample_id": str(batch.get("sample_id", [""])[0]) if "sample_id" in batch else "",
            "dataset_split": str(batch.get("dataset_split", [""])[0]) if "dataset_split" in batch else "",
            # Preserve the exact temporal routing used by the model. These
            # fields let evaluation distinguish genuine historical-reference
            # control from a same-frame reconstruction.
            "video_id": video_id,
            "condition_video_id": condition_video_id,
            "condition_frame": condition_frame,
            "target_frame": target_frame,
            "reference_video_id": reference_video_id,
            "reference_frame": reference_frame,
            "reference_frame_delta": (
                target_frame - reference_frame
                if target_frame is not None and reference_frame is not None else None
            ),
            "reference_is_historical": (
                bool(reference_video_id == video_id and reference_frame < target_frame)
                if reference_video_id and video_id
                and reference_frame is not None and target_frame is not None
                else None
            ),
            "extra_condition_video_id": extra_condition_video_id,
            "extra_condition_frame": extra_condition_frame,
            "extra_condition_frame_delta": (
                target_frame - extra_condition_frame
                if target_frame is not None and extra_condition_frame is not None else None
            ),
            "extra_condition_reference_kind": extra_condition_reference_kind,
            "audio_category": int(batch.get("audio_category", torch.tensor([-1]))[0]) if "audio_category" in batch else -1,
            "audio_rms_dbfs": float(batch.get("audio_rms_dbfs", torch.tensor([0.0]))[0]) if "audio_rms_dbfs" in batch else None,
            "categorical_refiner": (
                None if categorical_refinement is None else {
                    **categorical_refinement["provenance"],
                    "regions": categorical_refinement["regions"],
                    "baseline_artifact": "inference_unrefined.png",
                    "audio_region_selector": categorical_refinement.get("audio_region_selector"),
                }
            ),
            "note": (
                "True inference from pure noise. Current-frame WSG conditions, "
                "historical/native RGB references, and auxiliary RGB conditions "
                "are recorded separately above."
            ),
        }
        if bool(getattr(self.args, "benchmark_metrics", False)):
            # Evaluate the same final pseudo-RGB images users inspect. Conversion
            # to [-1, 1] preserves the historical decoded-MSE scale while also
            # respecting modality denormalization used by the visualization.
            from src.evaluation.internal.channel_head_evaluator import (
                continuous_prediction_metrics,
                decoded_prediction_metrics,
                discrete_prediction_metrics,
            )

            decoded_by_name = {
                name: image for name, image in zip(output_names, [item[1] for item in images])
            }

            def image_tensor(image: Image.Image) -> torch.Tensor:
                array = np.asarray(image.convert("RGB"), dtype=np.float32)
                return torch.from_numpy(array).permute(2, 0, 1)[None] / 127.5 - 1.0

            generated_decoded = image_tensor(decoded_by_name["inference"])
            target_decoded = image_tensor(decoded_by_name["gt"])
            if bool(getattr(self.args, "benchmark_latent_diagnostics", False)):
                benchmark_metrics = continuous_prediction_metrics(
                    latents, target_clean, generated_raw, target_raw,
                    generated_decoded, target_decoded,
                )
            else:
                benchmark_metrics = decoded_prediction_metrics(
                    generated_decoded, target_decoded
                )
            benchmark_metrics["decoded_mse_definition"] = (
                "MSE between final inference and GT pseudo-RGB decodes in [-1,1] "
                "visualization space"
            )
            categorical = discrete_prediction_metrics(
                generated_decoded,
                batch,
                target_modality,
                str(getattr(self.args, "target_description", "") or ""),
                self.channel_ranges,
                float(getattr(self.args, "benchmark_categorical_tolerance", 0.20)),
            )
            if categorical is not None:
                benchmark_metrics["categorical"] = categorical
            if categorical_refinement is not None:
                baseline_decoded = image_tensor(baseline_inference_image)
                baseline_metrics = decoded_prediction_metrics(
                    baseline_decoded, target_decoded
                )
                baseline_categorical = discrete_prediction_metrics(
                    baseline_decoded,
                    batch,
                    target_modality,
                    str(getattr(self.args, "target_description", "") or ""),
                    self.channel_ranges,
                    float(getattr(self.args, "benchmark_categorical_tolerance", 0.20)),
                )
                if baseline_categorical is not None:
                    baseline_metrics["categorical"] = baseline_categorical
                metadata["benchmark_metrics_unrefined"] = baseline_metrics
            metadata["benchmark_metrics"] = benchmark_metrics
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        with (out_dir / "prompt.txt").open("w", encoding="utf-8") as handle:
            handle.write(prompts[0] if prompts else "")

    @torch.no_grad()
    def run_preview(self, batch: dict[str, Any], out_dir: Path, label: str) -> None:
        dtype = torch.bfloat16
        target_raw = batch["target_latent"].to(self.device, dtype=dtype)
        normal_raw = batch.get("omini_condition_latent")
        normal_raw = normal_raw.to(self.device, dtype=dtype) if normal_raw is not None else None
        reference_raw = batch.get("native_reference_latent")
        reference_raw = reference_raw.to(self.device, dtype=dtype) if reference_raw is not None else None
        extra_raw = batch.get("extra_condition_latent")
        extra_raw = extra_raw.to(self.device, dtype=dtype) if extra_raw is not None else None
        prompts = list(batch["description"])
        target_clean, target_ids = self.normalize_and_pack(self.flux_pipe, target_raw)
        if normal_raw is not None:
            normal_clean, normal_ids = self.normalize_and_pack(self.flux_pipe, normal_raw)
        else:
            normal_clean, normal_ids = None, None
        if reference_raw is not None:
            reference_clean, reference_ids = self.normalize_and_pack(self.flux_pipe, reference_raw)
            reference_ids = reference_ids.clone(); reference_ids[..., 0] = 10
        else:
            reference_clean, reference_ids = None, None
        condition_cleans = [] if normal_clean is None else [normal_clean]
        condition_ids = [] if normal_ids is None else [normal_ids]
        if extra_raw is not None:
            extra_clean, extra_ids = self.normalize_and_pack(self.flux_pipe, extra_raw)
            extra_ids = extra_ids.clone(); extra_ids[..., 0] = 10
            condition_cleans.append(extra_clean)
            condition_ids.append(extra_ids)
        condition_cleans, condition_ids = self.append_audio_condition(batch, condition_cleans, condition_ids, target_clean, target_ids, dtype)
        audio_modulation = self.build_audio_modulation(batch, dtype)
        if "prompt_embeds" in batch and "text_ids" in batch:
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=dtype)
            text_ids = batch["text_ids"].to(self.device)
        else:
            prompt_embeds, text_ids = self._encode_prompt_offloaded(prompts, dtype)
        generator = torch.Generator(device=self.device).manual_seed(self.args.seed + 17)
        timestep_value = 1.0 if _SINGLE_STEP_X0 else 0.5
        timestep = torch.full((target_clean.shape[0],), timestep_value, device=self.device, dtype=dtype)
        noise = torch.randn(target_clean.shape, generator=generator, device=self.device, dtype=dtype)
        target_noisy = (1 - timestep[:, None, None]) * target_clean + timestep[:, None, None] * noise
        prediction = flux2_tunneling_forward(
            self.transformer, target_noisy, condition_cleans, reference_clean, prompt_embeds,
            target_ids, condition_ids, reference_ids, text_ids, timestep,
            self.specify_lora, False, audio_modulation=audio_modulation,
        )
        clean_estimate = prediction if _SINGLE_STEP_X0 else target_noisy - timestep[:, None, None] * prediction
        generated_raw = self.unpack_normalized_latents(clean_estimate, target_ids)
        target_frame = int(batch.get("target_frame", [0])[0]) if "target_frame" in batch else None
        reference_frame = int(batch.get("reference_frame", [0])[0]) if "reference_frame" in batch else None
        condition_label = f"condition f{target_frame}" if target_frame is not None else "condition"
        generated_label = f"generated f{target_frame}" if target_frame is not None else "generated"
        gt_label = f"gt f{target_frame}" if target_frame is not None else "gt"
        reference_label = f"reference f{reference_frame}" if reference_frame is not None else "reference"
        # Get modality names for proper denormalization. For audio-dense mask
        # tasks the first declared condition is Audio_AST, while normal_raw holds
        # the current-frame RGB latent. Decode that visual condition as RGB.
        reference_modality = "0-2"
        condition_tokens = [item.strip() for item in getattr(self.args, "condition_modalities", "").split(",") if item.strip()]
        condition_modality = "0-2" if "CurrentFrame_RGB_Condition" in condition_tokens else (condition_tokens[0] if condition_tokens else "")
        target_modality = getattr(self.args, "target_visual_modality", "") or getattr(self.args, "target_modality", "")

        images = []
        output_names = []
        if reference_raw is not None:
            images.append((reference_label, self.decode_raw_latents(reference_raw, reference_modality)))
            output_names.append("reference")
        if extra_raw is not None:
            extra_modality = str(batch.get("extra_condition_modality", [""])[0]) if "extra_condition_modality" in batch else "0-2"
            extra_frame = int(batch.get("extra_condition_frame", [0])[0]) if "extra_condition_frame" in batch else None
            images.append((f"first-condition f{extra_frame}" if extra_frame is not None else "first-condition",
                           self.decode_raw_latents(extra_raw, extra_modality)))
            output_names.append("first_condition")
        if normal_raw is not None:
            images.append((condition_label, self.decode_raw_latents(normal_raw, condition_modality)))
            output_names.append("condition")
        images.extend([
            (generated_label, self.decode_raw_latents(generated_raw, target_modality)),
            (gt_label, self.decode_raw_latents(target_raw, target_modality)),
        ])
        output_names.extend(["generated", "gt"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image in zip(output_names, [item[1] for item in images]):
            image.save(out_dir / f"{name}.png")
        save_image_grid(images, out_dir / "summary.png")
        self.save_wsg_reader_visualization([("generated", generated_raw), ("gt", target_raw)], target_modality, out_dir)
        if normal_raw is not None and condition_modality:
            self.save_wsg_reader_visualization([("condition", normal_raw)], condition_modality, out_dir)
        if extra_raw is not None and extra_modality:
            self.save_wsg_reader_visualization([("first_condition", extra_raw)], extra_modality, out_dir)
        if reference_raw is not None:
            self.save_wsg_reader_visualization([("reference", reference_raw)], reference_modality, out_dir)
        condition_image = images[output_names.index("condition")][1] if "condition" in output_names else None
        self.save_audio_mask_reader_comparison(out_dir, condition_image, "generated", "gt", batch)
        metadata = {
            "label": label,
            "sample_id": str(batch.get("sample_id", [""])[0]) if "sample_id" in batch else "",
            "video_id": str(batch.get("video_id", [""])[0]) if "video_id" in batch else "",
            "reference_video_id": str(batch.get("reference_video_id", [""])[0]) if "reference_video_id" in batch else "",
            "reference_frame": reference_frame,
            "target_frame": target_frame,
            "audio_category": int(batch.get("audio_category", torch.tensor([-1]))[0]) if "audio_category" in batch else -1,
            "audio_rms_dbfs": float(batch.get("audio_rms_dbfs", torch.tensor([0.0]))[0]) if "audio_rms_dbfs" in batch else None,
            "note": "Reference is the current RGB frame when present; audio-conditioned dense tasks use AST features as a condition lane.",
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        with (out_dir / "prompt.txt").open("w", encoding="utf-8") as handle:
            handle.write(prompts[0] if prompts else "")

    def _compute_loss(self, batch, deterministic_seed: int | None = None,
                      force_timestep: float | None = None, val_metric: bool = False):
        dtype = torch.bfloat16
        target_raw = batch["target_latent"].to(self.device, dtype=dtype)
        normal_raw = batch.get("omini_condition_latent")
        normal_raw = normal_raw.to(self.device, dtype=dtype) if normal_raw is not None else None
        reference_raw = batch.get("native_reference_latent")
        reference_raw = reference_raw.to(self.device, dtype=dtype) if reference_raw is not None else None
        extra_raw = batch.get("extra_condition_latent")
        extra_raw = extra_raw.to(self.device, dtype=dtype) if extra_raw is not None else None
        prompts = list(batch["description"])
        with torch.no_grad():
            target_clean, target_ids = self.normalize_and_pack(self.flux_pipe, target_raw)
            if normal_raw is not None:
                normal_clean, normal_ids = self.normalize_and_pack(self.flux_pipe, normal_raw)
            else:
                normal_clean, normal_ids = None, None
            if reference_raw is not None:
                reference_clean, reference_ids = self.normalize_and_pack(self.flux_pipe, reference_raw)
                reference_ids = reference_ids.clone(); reference_ids[..., 0] = 10
            else:
                reference_clean, reference_ids = None, None
            condition_cleans = [] if normal_clean is None else [normal_clean]
            condition_ids = [] if normal_ids is None else [normal_ids]
            if extra_raw is not None:
                extra_clean, extra_ids = self.normalize_and_pack(self.flux_pipe, extra_raw)
                extra_ids = extra_ids.clone(); extra_ids[..., 0] = 10
                condition_cleans.append(extra_clean)
                condition_ids.append(extra_ids)
            if "prompt_embeds" in batch and "text_ids" in batch:
                prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=dtype)
                text_ids = batch["text_ids"].to(self.device)
            else:
                prompt_embeds, text_ids = self._encode_prompt_offloaded(prompts, dtype)
            generator = None
            if deterministic_seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(deterministic_seed))
            base_t = torch.randn((target_clean.shape[0],), generator=generator, device=self.device, dtype=dtype)
            if _SINGLE_STEP_X0:
                timestep = torch.ones((target_clean.shape[0],), device=self.device, dtype=dtype)
            elif force_timestep is not None:
                # Validation: evaluate at a fixed timestep (drives the fixed t-grid),
                # independent of the A-shift, so the metric is comparable across runs.
                timestep = torch.full((target_clean.shape[0],), float(force_timestep),
                                      device=self.device, dtype=dtype)
            elif _DE_TIMESTEP_SHIFT:
                # A. Low-noise timestep shift: bias toward the clear-target end.
                timestep = torch.sigmoid(base_t * _DENSE_ENHANCE_SCALE + _DENSE_ENHANCE_SHIFT)
            else:
                timestep = torch.sigmoid(base_t)
            if _HIGH_NOISE_ANCHOR and not _SINGLE_STEP_X0 and not val_metric:
                anchor_mask = torch.rand((target_clean.shape[0],), generator=generator,
                                         device=self.device) < _HIGH_NOISE_ANCHOR_PROB
                timestep = torch.where(anchor_mask, torch.full_like(timestep, _HIGH_NOISE_ANCHOR_T), timestep)
            noise = self.sample_training_noise(target_clean, target_ids, timestep, generator, use_multires=not val_metric)
            target_noisy = (1 - timestep[:, None, None]) * target_clean + timestep[:, None, None] * noise
            velocity = noise - target_clean
        condition_cleans, condition_ids = self.append_audio_condition(batch, condition_cleans, condition_ids, target_clean, target_ids, dtype)
        audio_modulation = self.build_audio_modulation(batch, dtype)
        prediction = flux2_tunneling_forward(
            self.transformer, target_noisy, condition_cleans, reference_clean, prompt_embeds,
            target_ids, condition_ids, reference_ids, text_ids, timestep,
            self.specify_lora, self.args.gradient_checkpointing and torch.is_grad_enabled(),
            audio_modulation=audio_modulation,
        )
        edit_source_clean = None
        if normal_clean is not None:
            edit_source_clean = normal_clean
        elif reference_clean is not None:
            edit_source_clean = reference_clean
        edit_weight = None
        if (not val_metric and float(getattr(self.args, "edit_loss_weight", 0.0) or 0.0) > 0.0
                and edit_source_clean is not None and edit_source_clean.shape == target_clean.shape):
            saliency = (target_clean.float() - edit_source_clean.float()).abs().mean(dim=-1, keepdim=True)
            denom = saliency.amax(dim=1, keepdim=True).clamp_min(1e-6)
            saliency = (saliency / denom).clamp(0.0, 1.0)
            power = float(getattr(self.args, "edit_loss_power", 1.0) or 1.0)
            if power != 1.0:
                saliency = saliency.pow(power)
            edit_weight = 1.0 + float(self.args.edit_loss_weight) * saliency

        if val_metric:
            clean_prediction = (prediction if _SINGLE_STEP_X0 else
                                target_noisy - timestep[:, None, None] * prediction)
            loss = F.mse_loss(clean_prediction.float(), target_clean.float())
        else:
            objective = target_clean if _SINGLE_STEP_X0 else velocity
            if _DE_HUBER and not _SINGLE_STEP_X0:
                element_loss = F.huber_loss(
                    prediction.float(), objective.float(), reduction="none",
                    delta=_DENSE_ENHANCE_HUBER_DELTA,
                )
            else:
                element_loss = (prediction.float() - objective.float()).pow(2)
            if float(self.args.focal_regression_gamma) > 0:
                element_loss = element_loss * focal_regression_weights(
                    prediction, objective, self.args.focal_regression_gamma
                )
            if edit_weight is not None:
                element_loss = element_loss * edit_weight
            per_sample = element_loss.flatten(1).mean(dim=1)
            if _DE_MINSNR and not _SINGLE_STEP_X0:
                t = timestep.float().clamp(1e-4, 1.0 - 1e-4)
                snr = ((1.0 - t) / t) ** 2
                w = torch.clamp(snr, max=_DENSE_ENHANCE_GAMMA) / _DENSE_ENHANCE_GAMMA
                loss = (w * per_sample).mean()
            else:
                loss = per_sample.mean()
            clean_prediction = (prediction if _SINGLE_STEP_X0 else
                                target_noisy - timestep[:, None, None] * prediction)
            auxiliary = {}
            prediction_spatial = target_spatial = None
            if float(self.args.gradient_domain_loss_weight) > 0:
                prediction_spatial = self.flux_pipe._unpack_latents_with_ids(clean_prediction, target_ids)
                target_spatial = self.flux_pipe._unpack_latents_with_ids(target_clean, target_ids)
                auxiliary["gradient"] = gradient_domain_loss(prediction_spatial, target_spatial)
                loss = loss + float(self.args.gradient_domain_loss_weight) * auxiliary["gradient"]
            if float(self.args.foreground_iou_loss_weight) > 0:
                if prediction_spatial is None:
                    prediction_spatial = self.flux_pipe._unpack_latents_with_ids(clean_prediction, target_ids)
                    target_spatial = self.flux_pipe._unpack_latents_with_ids(target_clean, target_ids)
                auxiliary["foreground_iou"] = latent_foreground_iou_loss(prediction_spatial, target_spatial)
                loss = loss + float(self.args.foreground_iou_loss_weight) * auxiliary["foreground_iou"]
            categorical_mode = str(self.args.categorical_loss_mode)
            if categorical_mode != "off":
                if prediction_spatial is None:
                    prediction_spatial = self.flux_pipe._unpack_latents_with_ids(clean_prediction, target_ids)
                    target_spatial = self.flux_pipe._unpack_latents_with_ids(target_clean, target_ids)
                categorical = categorical_latent_losses(
                    prediction_spatial,
                    target_spatial,
                    temperature=self.args.categorical_temperature,
                    margin=self.args.categorical_margin,
                    num_clusters=self.args.categorical_num_clusters,
                    kmeans_iterations=self.args.categorical_kmeans_iterations,
                )
                if categorical_mode == "prototype_ce":
                    categorical_loss = categorical["prototype_ce"]
                elif categorical_mode == "balanced_ce":
                    categorical_loss = categorical["prototype_ce"] + 0.5 * categorical["balanced"]
                elif categorical_mode == "margin_global":
                    categorical_loss = (categorical["prototype_ce"] + 0.5 * categorical["margin"]
                                        + 0.5 * categorical["global"])
                else:
                    categorical_loss = (categorical["prototype_ce"] + 0.5 * categorical["balanced"]
                                        + 0.5 * categorical["margin"] + 0.5 * categorical["global"]
                                        + 0.25 * categorical["iou"])
                auxiliary.update({f"categorical_{name}": value for name, value in categorical.items()})
                auxiliary["categorical_total"] = categorical_loss
                loss = loss + float(self.args.categorical_loss_weight) * categorical_loss
            self.last_auxiliary_losses = {name: value.detach().item() for name, value in auxiliary.items()}
        if not val_metric:
            # Only training updates the running log EMA / last_t (validation must not
            # pollute the training-loss display).
            self.last_t = timestep.mean().item()
            self.log_loss = loss.item() if not hasattr(self, "log_loss") else self.log_loss * 0.95 + loss.item() * 0.05
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch)

    @property
    def _val_timestep_grid(self) -> list[float]:
        if not hasattr(self, "_val_tgrid_cache"):
            raw = getattr(self.args, "val_timestep_grid", "") or ""
            self._val_tgrid_cache = [float(x) for x in raw.split(",") if x.strip()]
        return self._val_tgrid_cache

    @torch.no_grad()
    def validation_loss_for_batch(self, batch, batch_idx: int, step: int) -> torch.Tensor:
        # Validation is a FIXED, comparable metric:
        #  - seed independent of `step` => identical noise every evaluation, so the
        #    val curve reflects the model, not which timesteps happened to be drawn.
        #  - a fixed timestep grid is tiled across batches by batch_idx, so the 24
        #    val batches deterministically span [0,1] at no extra forward-pass cost.
        #  - val_metric=True => plain velocity-MSE regardless of A/B/C levers, so
        #    runs with different training objectives are still directly comparable.
        seed = self.args.seed + 1_000_000 + batch_idx
        grid = self._val_timestep_grid
        ft = grid[batch_idx % len(grid)] if grid else None
        return self._compute_loss(batch, deterministic_seed=seed, force_timestep=ft, val_metric=True)

def to_preview_batch(sample: dict[str, Any]) -> dict[str, Any]:
    return default_collate([sample])


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Legacy RGB-only converter. Use tensor_to_pil_with_denorm for proper modality handling."""
    image = image.detach().float().cpu()
    image = (image.clamp(-1, 1) + 1) * 127.5
    image = image.clamp(0, 255).to(torch.uint8)
    if image.ndim == 4:
        image = image[0]
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def tensor_to_pil_with_denorm(
    image: torch.Tensor,
    channel_ranges: dict[int, tuple[float, float]] | None = None,
    base_channel: int = 0,
    modality: str | None = None,
    normalization_mode: str = "linear",
) -> Image.Image:
    """Generic WSG visualization for decoded FLUX latents."""
    image = image.detach().float().cpu()
    modality = str(modality or "")

    if modality in {"111", "195"}:
        rgb = ((image[:3] + 1.0) * 127.5).clamp(0, 255).permute(1, 2, 0).numpy()
        colors = AVIS_CATEGORY_COLORS if modality == "195" else None
        if colors is None:
            labels = categorical_rgb_to_ids(rgb)
            rendered = categorical_ids_to_rgb(labels)
        else:
            labels = categorical_rgb_to_ids(rgb, colors)
            rendered = categorical_ids_to_rgb(labels, colors)
        return Image.fromarray(rendered, mode="RGB")

    if channel_ranges is None or base_channel not in channel_ranges:
        image = (image.clamp(-1, 1) + 1) * 127.5
        image = image.clamp(0, 255).to(torch.uint8)
    else:
        image01 = (image + 1) / 2
        denormalized = torch.zeros_like(image01)
        for i in range(min(3, image01.shape[0])):
            ch = base_channel + i
            if ch in channel_ranges:
                min_val, max_val = channel_ranges[ch]
                value_range = max(float(max_val) - float(min_val), 1e-6)
                if str(normalization_mode).lower() == "log":
                    denormalized[i] = (
                        torch.expm1(
                            image01[i].clamp(0, 1)
                            * max(math.log1p(value_range), 1e-6)
                        )
                        + float(min_val)
                    )
                else:
                    denormalized[i] = image01[i] * value_range + float(min_val)
            else:
                denormalized[i] = image01[i] * 255.0
        channels = parse_modality_channels(modality, base_channel)
        image = render_denormalized_modal_tensor(denormalized, channels, channel_ranges).to(torch.uint8)

    if image.ndim == 4:
        image = image[0]
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def parse_modality_channels(modality: str, base_channel: int) -> list[int]:
    if not modality:
        return [base_channel + i for i in range(3)]
    parts = str(modality).split("-")
    try:
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
        return [int(parts[0])]
    except ValueError:
        return [base_channel + i for i in range(3)]


def render_denormalized_modal_tensor(
    denormalized: torch.Tensor,
    channels: list[int],
    channel_ranges: dict[int, tuple[float, float]],
) -> torch.Tensor:
    channel_count = max(1, min(len(channels), denormalized.shape[0]))
    values = denormalized[:channel_count]
    ranges = [channel_ranges.get(ch) for ch in channels[:channel_count]]

    if channel_count == 3 and is_rgb_like_range(ranges):
        out = values
        max_range = max(float(item[1]) for item in ranges if item is not None)
        if max_range <= 1.5:
            out = out * 255.0
        return out.clamp(0, 255)

    if channel_count == 3 and is_signed_vector_range(ranges):
        return render_signed_vector_rgb(values)

    if channel_count == 1:
        value_range = ranges[0] if ranges else None
        if is_mask_like_scalar_range(value_range):
            return render_scalar_grayscale(values[0], value_range)
        channel = channels[0] if channels else -1
        if (
            channel in READER_VISUALIZATION_SPECS
            and READER_VISUALIZATION_SPECS[channel].get("nodes")
            and reader_apply_continuous_palette is not None
            and reader_normalize_scalar_for_display is not None
            and refine_reader_valid_mask is not None
        ):
            value_map = values[0].float().numpy()
            valid_mask = np.isfinite(value_map) & (value_map != -1)
            valid_mask = refine_reader_valid_mask(
                value_map, valid_mask, channel
            )
            normalized, _ = reader_normalize_scalar_for_display(
                value_map,
                valid_mask,
                READER_VISUALIZATION_CHANNEL_RANGES.get(channel),
            )
            rendered = reader_apply_continuous_palette(
                normalized, channel, valid_mask
            )
            return torch.from_numpy(rendered).permute(2, 0, 1)
        # Curvature keeps the Reader's contrast-oriented signed rendering; all
        # other scalar modalities use their dataset-wide calibration range.
        # This prevents identical physical values from receiving different
        # colors merely because they occur in different samples.
        if channel == 39:
            return render_scalar_pseudorgb(values[0])
        return render_scalar_pseudorgb(values[0], value_range)

    if channel_count == 2:
        # Generic two-channel visualization:
        # - scalar + auxiliary/confidence channel: colorize the calibrated scalar.
        # - real 2D vector: HSV encodes angle as hue and magnitude as brightness.
        if len(ranges) > 1 and ranges[1] is None and ranges[0] is not None:
            return render_scalar_pseudorgb(values[0], ranges[0])
        return render_vector2_hsv_rgb(values[:2])

    # Fallback for arbitrary 3+ channel non-RGB tensors: normalize first three
    # channels independently into RGB so every modality remains inspectable.
    rendered = [
        normalize_for_visual(channel, ranges[index] if index < len(ranges) else None) * 255.0
        for index, channel in enumerate(values[:3])
    ]
    while len(rendered) < 3:
        rendered.append(rendered[-1])
    return torch.stack(rendered[:3], dim=0).clamp(0, 255)


def normalize_for_visual(
    value: torch.Tensor,
    value_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    finite = torch.isfinite(value)
    if value_range is not None:
        lo = torch.as_tensor(float(value_range[0]), dtype=torch.float32, device=value.device)
        hi = torch.as_tensor(float(value_range[1]), dtype=torch.float32, device=value.device)
    elif finite.any():
        lo, hi = torch.quantile(value[finite].float(), torch.tensor([0.01, 0.99]))
    else:
        lo, hi = value.float().min(), value.float().max()
    span = hi - lo
    if float(span) <= 1e-6:
        out = torch.full_like(value.float(), 0.5)
    else:
        out = ((value.float() - lo) / span).clamp(0, 1)
    return torch.where(finite, out, torch.zeros_like(out))


def render_scalar_pseudorgb(
    value: torch.Tensor,
    value_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    return turbo_colormap(normalize_for_visual(value, value_range)) * 255.0


def render_scalar_grayscale(value: torch.Tensor, value_range: tuple[float, float] | None) -> torch.Tensor:
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
        denom = max(hi - lo, 1e-6)
        gray = ((value.float() - lo) / denom).clamp(0, 1)
    else:
        gray = normalize_for_visual(value)
    return gray.expand(3, -1, -1) * 255.0


def is_mask_like_scalar_range(value_range: tuple[float, float] | None) -> bool:
    if value_range is None:
        return False
    lo, hi = float(value_range[0]), float(value_range[1])
    return (abs(lo) < 1e-6 and abs(hi - 255.0) < 1e-6) or (abs(lo) < 1e-6 and abs(hi - 1.0) < 1e-6) or (abs(lo + 1.0) < 1e-6 and abs(hi - 1.0) < 1e-6)



def turbo_colormap(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp(0, 1)
    powers = torch.stack([x ** i for i in range(6)], dim=0)
    coeffs = torch.tensor([
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ], dtype=powers.dtype, device=powers.device)
    rgb = torch.einsum("cp,phw->chw", coeffs, powers).clamp(0, 1)
    return rgb


def render_vector2_hsv_rgb(values: torch.Tensor) -> torch.Tensor:
    x, y = values[0].float(), values[1].float()
    finite = torch.isfinite(x) & torch.isfinite(y)
    hue = ((torch.atan2(y, x) + torch.pi) / (2 * torch.pi)).clamp(0, 1)
    mag = normalize_for_visual(torch.sqrt(torch.where(finite, x * x + y * y, torch.zeros_like(x))))
    sat = torch.where(finite, torch.ones_like(mag), torch.zeros_like(mag))
    val = torch.where(finite, mag, torch.zeros_like(mag))
    return hsv_to_rgb(hue, sat, val) * 255.0


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h6 = (h * 6.0) % 6.0
    c = v * s
    x = c * (1.0 - torch.abs((h6 % 2.0) - 1.0))
    m = v - c
    z = torch.zeros_like(h)
    r = torch.where(h6 < 1, c, torch.where(h6 < 2, x, torch.where(h6 < 4, z, torch.where(h6 < 5, x, c))))
    g = torch.where(h6 < 1, x, torch.where(h6 < 3, c, torch.where(h6 < 4, x, z)))
    b = torch.where(h6 < 2, z, torch.where(h6 < 3, x, torch.where(h6 < 5, c, x)))
    return torch.stack([r + m, g + m, b + m], dim=0).clamp(0, 1)


def render_signed_vector_rgb(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    abs_values = values[finite].abs().float()
    if abs_values.numel() == 0:
        scale = torch.tensor(1.0, dtype=values.dtype)
    else:
        scale = torch.quantile(abs_values, 0.99).to(values.dtype)
        scale = torch.clamp(scale, min=torch.tensor(0.25, dtype=values.dtype), max=torch.tensor(1.0, dtype=values.dtype))
    return ((values / scale).clamp(-1, 1) + 1.0) * 127.5


def is_rgb_like_range(ranges: list[tuple[float, float] | None]) -> bool:
    if len(ranges) != 3 or any(item is None for item in ranges):
        return False
    mins = [float(item[0]) for item in ranges if item is not None]
    maxs = [float(item[1]) for item in ranges if item is not None]
    return all(0.0 <= lo <= 1.5 for lo in mins) and (
        all(0.0 <= hi <= 1.5 for hi in maxs) or all(10.0 <= hi <= 255.5 for hi in maxs)
    )


def is_signed_vector_range(ranges: list[tuple[float, float] | None]) -> bool:
    if len(ranges) != 3 or any(item is None for item in ranges):
        return False
    return all(-2.0 <= float(item[0]) <= 0.25 and 0.25 <= float(item[1]) <= 2.0 for item in ranges if item is not None)


def save_image_grid(images: list[tuple[str, Image.Image]], path: Path) -> None:
    if not images:
        return
    widths = [img.width for _, img in images]
    heights = [img.height for _, img in images]
    label_h = 28
    canvas = Image.new("RGB", (sum(widths), max(heights) + label_h), "white")
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
    except Exception:
        draw = None
    x = 0
    for label, img in images:
        canvas.paste(img.convert("RGB"), (x, label_h))
        if draw is not None:
            draw.text((x + 4, 6), label, fill=(0, 0, 0))
        x += img.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def quantize_tensor_int8_for_state(tensor: torch.Tensor) -> dict[str, Any] | torch.Tensor:
    tensor = tensor.detach().cpu()
    if not tensor.is_floating_point() or tensor.numel() == 0:
        return tensor
    values = tensor.float()
    max_abs = values.abs().max()
    if not torch.isfinite(max_abs) or float(max_abs) == 0.0:
        scale = 1.0
        quantized = torch.zeros_like(values, dtype=torch.int8)
    else:
        scale = float(max_abs / 127.0)
        quantized = torch.clamp(torch.round(values / scale), -127, 127).to(torch.int8)
    return {
        "__wsm_quantized_int8_tensor__": True,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "scale": scale,
        "data": quantized,
    }


def quantize_state_int8_for_save(value: Any) -> Any:
    if torch.is_tensor(value):
        return quantize_tensor_int8_for_state(value)
    if isinstance(value, dict):
        return {key: quantize_state_int8_for_save(item) for key, item in value.items()}
    if isinstance(value, list):
        return [quantize_state_int8_for_save(item) for item in value]
    if isinstance(value, tuple):
        return tuple(quantize_state_int8_for_save(item) for item in value)
    return value


def dequantize_state_int8_after_load(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__wsm_quantized_int8_tensor__"):
        dtype_name = str(value.get("dtype", "float32"))
        dtype = getattr(torch, dtype_name, torch.float32)
        data = value["data"].to(torch.float32)
        tensor = data * float(value.get("scale", 1.0))
        return tensor.reshape(tuple(value.get("shape", tensor.shape))).to(dtype)
    if isinstance(value, dict):
        return {key: dequantize_state_int8_after_load(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dequantize_state_int8_after_load(item) for item in value]
    if isinstance(value, tuple):
        return tuple(dequantize_state_int8_after_load(item) for item in value)
    return value


class Flux2WSMTrainingCallback(L.Callback):
    def __init__(self, run_name: str, training_config: dict[str, Any], test_function=None, val_loader=None):
        self.run_name = run_name
        self.training_config = training_config
        self.test_function = test_function
        self.val_loader = val_loader
        self.save_path = Path(training_config.get("save_path", "./output"))
        self.print_every = int(training_config.get("print_every_n_steps", 10))
        self.save_interval = int(training_config.get("save_interval", 1000))
        self.sample_interval = int(training_config.get("sample_interval", 1000))
        self.val_interval = int(training_config.get("val_interval", self.sample_interval))
        self.total_steps = 0
        self.train_start_time: float | None = None
        self.last_progress_update_step = 0
        # Checkpoint only on validation-loss improvement; early-stop if no
        # improvement within early_stop_seconds.
        self.best_val_loss = float("inf")
        self.last_improve_time: float | None = None
        self.early_stop_seconds = float(training_config.get("early_stop_seconds", 7200))
        self.history: dict[str, list[float]] = {"step": [], "loss": [], "smoothed_loss": [], "t": [], "grad_avg": [], "grad_max": [], "val_step": [], "val_loss": []}

    @property
    def run_dir(self) -> Path:
        return self.save_path / self.run_name

    def _save_stats(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in self.history.items()}
        np.savez(self.run_dir / "stats.npz", **arrays)
        if len(self.history["step"]) == 0:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(self.history["step"], self.history["loss"], label="loss", alpha=0.45)
        ax.plot(self.history["step"], self.history["smoothed_loss"], label="smoothed_loss")
        if self.history.get("val_step") and self.history.get("val_loss"):
            ax.plot(self.history["val_step"], self.history["val_loss"], label="val_loss", marker="o", linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.run_dir / "stats.png", dpi=160)
        plt.close(fig)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "?"
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{seconds:02d}s"
        if minutes:
            return f"{minutes}m{seconds:02d}s"
        return f"{seconds}s"

    def _progress_context(self, trainer) -> dict[str, Any]:
        max_steps = int(self.training_config.get("max_steps", getattr(trainer, "max_steps", -1)) or -1)
        total = max_steps if max_steps > 0 else max(int(getattr(trainer, "estimated_stepping_batches", 0) or 0), self.total_steps)
        steps_per_epoch = int(getattr(trainer, "num_training_batches", 0) or 0)
        if steps_per_epoch <= 0:
            try:
                steps_per_epoch = len(trainer.train_dataloader)
            except Exception:
                steps_per_epoch = 0
        batch_size = int(self.training_config.get("batch_size", 1) or 1)
        grad_accum = int(self.training_config.get("accumulate_grad_batches", 1) or 1)
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        effective_bs = batch_size * grad_accum * world_size
        return {
            "total": max(total, self.total_steps),
            "steps_per_epoch": steps_per_epoch,
            "epoch_float": (self.total_steps / steps_per_epoch) if steps_per_epoch > 0 else float(trainer.current_epoch),
            "effective_bs": effective_bs,
        }

    def _print_progress_line(self, trainer, smoothed: float) -> None:
        ctx = self._progress_context(trainer)
        total = ctx["total"]
        fraction = min(max(self.total_steps / total, 0.0), 1.0) if total > 0 else 0.0
        bar_width = 10
        filled = int(fraction * bar_width)
        bar = "█" * filled + " " * (bar_width - filled)
        elapsed = 0.0 if self.train_start_time is None else time.time() - self.train_start_time
        seconds_per_step = elapsed / max(self.total_steps, 1)
        remaining = max(total - self.total_steps, 0) * seconds_per_step if total > 0 else float("nan")
        rate = f"{seconds_per_step:.2f}s/it" if self.total_steps else "?it/s"
        epoch = f"{ctx['epoch_float']:.3f}" if ctx["steps_per_epoch"] else str(trainer.current_epoch)
        print(
            f"eta~{self._format_duration(remaining)} Steps {self.total_steps}/{total} "
            f"elapsed={self._format_duration(elapsed)}: {fraction * 100:3.0f}%|{bar}| "
            f"{self.total_steps}/{total} [{self._format_duration(elapsed)}<{self._format_duration(remaining)}, {rate}, "
            f"loss={smoothed:.4f}, steps/epoch={ctx['steps_per_epoch']}, epoch={epoch}, BS={ctx['effective_bs']} ]",
            flush=True,
        )

    def _save_state(self, trainer, step_dir: Path) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        optimizer_states = [optimizer.state_dict() for optimizer in trainer.optimizers]
        optimizer_state_format = "torch"
        if bool(self.training_config.get("save_optimizer_state_8bit", False)):
            optimizer_states = quantize_state_int8_for_save(optimizer_states)
            optimizer_state_format = "int8_symmetric_per_tensor"
        state = {
            "global_step": int(self.total_steps),
            "trainer_global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "run_name": self.run_name,
            "training_config": self.training_config,
            "optimizer_state_format": optimizer_state_format,
            "optimizer_states": optimizer_states,
            "lr_scheduler_states": [scheduler["scheduler"].state_dict() for scheduler in trainer.lr_scheduler_configs] if getattr(trainer, "lr_scheduler_configs", None) else [],
            "stats": self.history,
        }
        state_pt = step_dir / "trainer_state.pt"
        torch.save(state, state_pt)
        shutil.copy2(state_pt, self.run_dir / "latest_trainer_state.pt")
        self._save_stats()
        for name in ("stats.npz", "stats.png"):
            source = self.run_dir / name
            if source.exists():
                shutil.copy2(source, step_dir / name)

    @staticmethod
    def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, dict):
            return {key: Flux2WSMTrainingCallback._move_batch_to_device(value, device) for key, value in batch.items()}
        if isinstance(batch, list):
            return [Flux2WSMTrainingCallback._move_batch_to_device(value, device) for value in batch]
        if isinstance(batch, tuple):
            return tuple(Flux2WSMTrainingCallback._move_batch_to_device(value, device) for value in batch)
        return batch

    def _compute_val_loss(self, trainer, pl_module) -> float | None:
        if self.val_loader is None:
            return None
        max_batches = int(self.training_config.get("val_loss_batches", 0) or 0)
        if max_batches <= 0:
            return None
        was_training = pl_module.training
        pl_module.eval()
        losses = []
        try:
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                batch = self._move_batch_to_device(batch, pl_module.device)
                loss = pl_module.validation_loss_for_batch(batch, batch_idx, self.total_steps)
                losses.append(float(loss.detach().item()))
        finally:
            if was_training:
                pl_module.train()
        if not losses:
            return None
        val_loss = float(np.mean(losses))
        self.history["val_step"].append(float(self.total_steps))
        self.history["val_loss"].append(val_loss)
        print(f"[WSM_v1] val_loss step={self.total_steps} loss={val_loss:.4f} batches={len(losses)}", flush=True)
        return val_loss

    def _preview(self, pl_module, label: str) -> None:
        if self.test_function is None:
            return
        out_dir = self.run_dir / "pseudo-visualization" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        was_training = pl_module.training
        pl_module.eval()
        try:
            self.test_function(pl_module, out_dir, label)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WSM_v1] preview {label} OOM; skipped", flush=True)
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            if was_training:
                pl_module.train()

    def _inference_preview(self, pl_module, label: str, num_steps: int = 10) -> None:
        """True inference visualization with multi-step denoising."""
        if not hasattr(pl_module, 'run_inference'):
            return
        out_dir = self.run_dir / "pseudo-visualization-inference" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        was_training = pl_module.training
        pl_module.eval()
        try:
            if self.test_function is None:
                return
            inference_samples = list(getattr(pl_module, "inference_samples", None) or [])
            if not inference_samples and getattr(pl_module, "inference_sample", None) is not None:
                inference_samples = [pl_module.inference_sample]
            if not inference_samples:
                print(f"[WSM_v1] inference preview {label} skipped (no sample)", flush=True)
                return
            for sample_index, inference_sample in enumerate(inference_samples):
                sample_out_dir = out_dir / f"sample_{sample_index:02d}" if len(inference_samples) > 1 else out_dir
                pl_module.run_inference(inference_sample, sample_out_dir, label, num_steps=num_steps)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[WSM_v1] inference preview {label} OOM; skipped", flush=True)
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            if was_training:
                pl_module.train()

    def on_train_start(self, trainer, pl_module):
        if not bool(self.training_config.get("disable_preflight", False)):
            preflight_label = "preflight" if self.total_steps == 0 else f"preflight_resume_step_{self.total_steps:07d}"
            self._preview(pl_module, preflight_label)
            self._inference_preview(pl_module, preflight_label)
        self.train_start_time = time.time()
        ctx = self._progress_context(trainer)
        print(
            f"[WSM_v1] training progress: max_steps={ctx['total']} "
            f"steps_per_epoch={ctx['steps_per_epoch']} effective_bs={ctx['effective_bs']} "
            f"log_every={self.print_every} save_every={self.save_interval} val_every={self.val_interval} viz_every={self.sample_interval}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss_tensor = outputs.get("loss") if isinstance(outputs, dict) else outputs
        loss = float(loss_tensor.detach().item()) if torch.is_tensor(loss_tensor) else float("nan")
        grad_sum = 0.0
        grad_max = 0.0
        grad_count = 0
        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.norm(2).detach().item())
            grad_sum += norm
            grad_max = max(grad_max, norm)
            grad_count += 1
        grad_avg = grad_sum / grad_count if grad_count else 0.0
        self.total_steps += 1
        smoothed = loss if not self.history["smoothed_loss"] else self.history["smoothed_loss"][-1] * 0.95 + loss * 0.05
        self.history["step"].append(float(self.total_steps))
        self.history["loss"].append(loss)
        self.history["smoothed_loss"].append(smoothed)
        self.history["t"].append(float(getattr(pl_module, "last_t", 0.0)))
        self.history["grad_avg"].append(grad_avg)
        self.history["grad_max"].append(grad_max)
        should_log = self.total_steps % self.print_every == 0
        if should_log:
            self._print_progress_line(trainer, smoothed)
            print(f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {smoothed:.4f}, Gradient size: {grad_avg:.4f}, Max gradient size: {grad_max:.4f}", flush=True)
            auxiliary = getattr(pl_module, "last_auxiliary_losses", {})
            if auxiliary:
                values = " ".join(f"{name}={value:.6g}" for name, value in sorted(auxiliary.items()))
                print(f"[WSM_v1] auxiliary step={self.total_steps} {values}", flush=True)
            self._save_stats()
        if self.last_improve_time is None:
            self.last_improve_time = time.time()
        if self.total_steps % self.val_interval == 0:
            val_loss = self._compute_val_loss(trainer, pl_module)
            self._save_stats()
            if val_loss is not None:
                if val_loss < self.best_val_loss - 1e-6:
                    # New best validation loss -> this is the only thing that saves.
                    self.best_val_loss = val_loss
                    self.last_improve_time = time.time()
                    self._save_checkpoint(trainer, pl_module, val_loss)
                    if not should_log:
                        self._print_progress_line(trainer, smoothed)
                elif (time.time() - self.last_improve_time) > self.early_stop_seconds:
                    mins = self.early_stop_seconds / 60.0
                    print(
                        f"[WSM_v1] early stop: no val_loss improvement for "
                        f"{mins:.0f} min (best={self.best_val_loss:.4f}) -> stopping",
                        flush=True,
                    )
                    trainer.should_stop = True
        if self.total_steps % self.sample_interval == 0:
            self._preview(pl_module, f"step_{self.total_steps:07d}")
            self._inference_preview(pl_module, f"step_{self.total_steps:07d}")
            if not should_log:
                self._print_progress_line(trainer, smoothed)

    def _save_checkpoint(self, trainer, pl_module, val_loss: float) -> None:
        # Keep only the latest (best) checkpoint: delete old ckpt/state dirs first.
        for old_ckpt in sorted((self.run_dir / "ckpt").glob("[0-9]*")):
            if old_ckpt.is_dir():
                robust_rmtree(old_ckpt)
        for old_state in sorted((self.run_dir / "state").glob("step_*")):
            if old_state.is_dir():
                robust_rmtree(old_state)
        ckpt_dir = self.run_dir / "ckpt" / f"{self.total_steps:07d}"
        print(f"[WSM_v1] new best val_loss={val_loss:.4f} -> saving {ckpt_dir}", flush=True)
        pl_module.save_channel_head(str(ckpt_dir))
        self._save_state(trainer, self.run_dir / "state" / f"step_{self.total_steps:07d}")

    def on_train_end(self, trainer, pl_module):
        if self.history["smoothed_loss"]:
            self._print_progress_line(trainer, self.history["smoothed_loss"][-1])
        final_dir = self.run_dir / "ckpt" / f"final_step_{self.total_steps:07d}"
        pl_module.save_channel_head(str(final_dir))
        self._save_state(trainer, self.run_dir / "state" / f"final_step_{self.total_steps:07d}")


def train_with_wsm_monitoring(dataset, val_dataset, trainable_model, config, test_function, resume_checkpoint_path=None):
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    is_main = rank == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    training_config = config["train"]
    run_name = training_config.get("run_name")
    if run_name in (None, "", "."):
        run_name = "."
    loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config.get("dataloader_workers", 0),
    )
    val_loader = None
    if is_main and val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get("batch_size", 1),
            shuffle=False,
            num_workers=training_config.get("val_dataloader_workers", 0),
        )
    callbacks = [Flux2WSMTrainingCallback(run_name, training_config, test_function, val_loader)] if is_main else []
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )
    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)
    if is_main:
        run_dir = Path(training_config.get("save_path", "./output")) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print("Rank:", rank, flush=True)
        print("Config:", config, flush=True)
    print("Dataset length:", len(dataset), flush=True)
    if is_main and val_dataset is not None:
        print("Validation dataset length:", len(val_dataset), flush=True)

    # Resume from checkpoint if specified
    if resume_checkpoint_path is not None:
        resume_path = Path(resume_checkpoint_path)
        if resume_path.exists():
            print(f"[WSM_v1] Resuming from checkpoint: {resume_path}", flush=True)
            state_file = resume_path / "trainer_state.pt"
            if state_file.exists():
                state = torch.load(state_file, map_location="cpu")
                resume_weights = None
                if resume_path.parent.name == "state" and resume_path.name.startswith("step_"):
                    weight_dir = resume_path.parent.parent / "ckpt" / resume_path.name.removeprefix("step_")
                    weight_file = weight_dir / "pytorch_lora_weights.safetensors"
                    if weight_file.exists():
                        resume_weights = weight_file
                elif resume_path.parent.name == "state" and resume_path.name.startswith("final_step_"):
                    weight_dir = resume_path.parent.parent / "ckpt" / resume_path.name
                    weight_file = weight_dir / "pytorch_lora_weights.safetensors"
                    if weight_file.exists():
                        resume_weights = weight_file
                elif (resume_path / "pytorch_lora_weights.safetensors").exists():
                    resume_weights = resume_path / "pytorch_lora_weights.safetensors"
                if resume_weights is not None:
                    channel_head_state = torch.load(resume_weights, map_location="cpu") if resume_weights.suffix == ".pt" else None
                    if channel_head_state is None:
                        from safetensors.torch import load_file as load_safetensors
                        channel_head_state = load_safetensors(str(resume_weights))
                    channel_head_state = {
                        key.removeprefix("transformer."): value
                        for key, value in channel_head_state.items()
                    }
                    load_result = set_peft_model_state_dict(
                        trainable_model.transformer,
                        channel_head_state,
                        adapter_name="default",
                    )
                    unexpected = list(getattr(load_result, "unexpected_keys", []))
                    if unexpected:
                        print(
                            f"[WSM_v1] WARNING: {len(unexpected)} unexpected Channel_Head "
                            f"resume keys; first keys: {unexpected[:5]}",
                            flush=True,
                        )
                    condition_adapter_file = resume_weights.parent / "condition_lora_adapters.pt"
                    if getattr(trainable_model, "extra_lora_adapter_names", None) and condition_adapter_file.exists():
                        condition_states = torch.load(condition_adapter_file, map_location="cpu")
                        for adapter_name, adapter_state in condition_states.items():
                            set_peft_model_state_dict(trainable_model.transformer, adapter_state, adapter_name=adapter_name)
                        print(f"[WSM_v1] Restored condition Channel Head adapters from {condition_adapter_file}", flush=True)
                    print(f"[WSM_v1] Restored Channel_Head weights from {resume_weights}", flush=True)
                    projector_file = resume_weights.parent / "audio_condition_projector.pt"
                    if trainable_model.audio_condition_projector is not None and projector_file.exists():
                        trainable_model.audio_condition_projector.load_state_dict(torch.load(projector_file, map_location="cpu"))
                        print(f"[WSM_v1] Restored audio condition projector from {projector_file}", flush=True)
                    modulation_file = resume_weights.parent / "audio_modulation_projector.pt"
                    if trainable_model.audio_modulation_projector is not None and modulation_file.exists():
                        trainable_model.audio_modulation_projector.load_state_dict(torch.load(modulation_file, map_location="cpu"))
                        print(f"[WSM_v1] Restored audio modulation projector from {modulation_file}", flush=True)
                    gate_file = resume_weights.parent / "audio_modulation_gate.pt"
                    if trainable_model.audio_modulation_gate is not None and gate_file.exists():
                        trainable_model.audio_modulation_gate.data.copy_(torch.load(gate_file, map_location=trainable_model.audio_modulation_gate.device))
                        print(f"[WSM_v1] Restored audio modulation gate from {gate_file}", flush=True)
                else:
                    raise RuntimeError(f"No Channel_Head weights found for resume path {resume_path}")
                # Restore optimizer states
                if "optimizer_states" in state and trainer.optimizers:
                    optimizer_state_format = state.get("optimizer_state_format", "torch")
                    if optimizer_state_format == "int8_symmetric_per_tensor":
                        optimizer_states = dequantize_state_int8_after_load(state["optimizer_states"])
                    else:
                        optimizer_states = state["optimizer_states"]
                    for optimizer, opt_state in zip(trainer.optimizers, optimizer_states):
                        optimizer.load_state_dict(opt_state)
                # Restore learning rate scheduler states
                if "lr_scheduler_states" in state and trainer.lr_scheduler_configs:
                    for scheduler_config, sched_state in zip(trainer.lr_scheduler_configs, state["lr_scheduler_states"]):
                        scheduler_config["scheduler"].load_state_dict(sched_state)
                # Restore callback stats
                if is_main and callbacks and "stats" in state:
                    callbacks[0].history = state["stats"]
                    callbacks[0].total_steps = state.get("global_step", 0)
                resumed_step = int(state.get("global_step", 0))
                target_steps = int(training_config.get("max_steps", -1))
                if target_steps >= 0:
                    remaining_steps = max(target_steps - resumed_step, 0)
                    trainer.fit_loop.epoch_loop.max_steps = remaining_steps
                    print(
                        f"[WSM_v1] Resumed from step {resumed_step}; "
                        f"running {remaining_steps} remaining step(s) toward {target_steps}",
                        flush=True,
                    )
                else:
                    print(f"[WSM_v1] Resumed from step {resumed_step}", flush=True)
        else:
            print(f"[WSM_v1] WARNING: Checkpoint path does not exist: {resume_path}", flush=True)

    trainer.fit(trainable_model, loader)


def main():
    args = parse_args()
    # Attention topology and target trainability are resolved per task_type once
    # the entry is loaded (below).
    try:
        import wandb  # noqa: F401
    except ImportError:
        sys.modules["wandb"] = types.SimpleNamespace()
    try:
        import prodigyopt  # noqa: F401
    except ImportError:
        sys.modules["prodigyopt"] = types.SimpleNamespace(Prodigy=None)

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    modality_config_path = resolve_path(args.modality_config)
    entry = load_channel_head_entry(modality_config_path, args.channel_head_index)
    target_dataset_whitelist = load_target_dataset_whitelist(
        modality_config_path, args.target_modality, args.target_dataset_whitelist
    )
    conditions = [item.strip() for item in args.condition_modalities.split(",") if item.strip()]

    # Reference routing is task-type driven, not inferred from a single condition.
    # - estimation_or_reconstruction / conditional_control: visual condition only.
    # - conditional_control+narrow_generation with Reference_IMG: RGB first frame is
    #   the NATIVE (frozen) reference by DEFAULT. Ablation (Channel_Head_000270) showed native +
    #   Native Joint Attention + frozen target was the fastest/best for this narrow generation task;
    #   routing the first frame through the trainable Channel_Head condition channel instead
    #   made the condition hard to follow. Set WSG_REFERENCE_IMG_CONDITION=1 to route
    #   the first frame through the Channel_Head condition channel instead (opt-in).
    # - inverse_rendering with RGB plus another modality: RGB first frame is an extra condition branch; the other modality is condition.
    task_type = str(entry.get("task_type") or "")
    # Resolve attention topology for THIS task_type (four per-task switches +
    # per-task defaults; a global --condition-attention-mode native_joint/modality_tunneling forces all).
    _attn_mode, _attn_src = resolve_task_attention_mode(task_type, args)
    set_condition_attention_mode(_attn_mode)
    set_separate_condition_adapters(args.separate_condition_adapters)
    # Resolve target-lane trainability for THIS task_type (per-task default; a
    # global --trainable-target on/off forces all).
    _tt_flag, _tt_src = resolve_task_trainable_target(task_type, args)
    set_trainable_target(_tt_flag)
    _tc_flag, _tc_src = resolve_task_trainable_condition(task_type, args)
    if getattr(args, "separate_condition_adapters", False):
        _tc_flag = True
        _tc_src = f"{_tc_src}+separate-condition-adapters"
    set_trainable_condition(_tc_flag)
    # Resolve the universal dense-prediction loss enhancement for THIS task_type
    # (per-task default; a global --dense-enhance on/off forces all).
    _de_flag, _de_src = resolve_task_dense_enhance(task_type, args)

    def _lever(val: str, master: bool) -> bool | None:
        # 'auto' => follow master (None lets set_dense_enhance default it); on/off override.
        if val == "on":
            return True
        if val == "off":
            return False
        return None

    set_dense_enhance(
        _de_flag,
        timestep=_lever(getattr(args, "de_timestep", "auto"), _de_flag),
        minsnr=_lever(getattr(args, "de_minsnr", "auto"), _de_flag),
        huber=_lever(getattr(args, "de_huber", "auto"), _de_flag),
    )
    _hna_flag, _hna_src = resolve_high_noise_anchor(task_type, args)
    set_literature_ablations(
        single_step_x0=args.single_step_x0,
        high_noise_anchor=_hna_flag,
        multires_noise=args.multires_noise,
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[WSM_v1] attention={_attn_mode} task_type={task_type or '?'} via={_attn_src}", flush=True)
        print(f"[WSM_v1] trainable_target={_tt_flag} via={_tt_src}", flush=True)
        print(f"[WSM_v1] trainable_condition={_tc_flag} via={_tc_src}", flush=True)
        print(
            f"[WSM_v1] target_dataset_whitelist="
            f"{sorted(target_dataset_whitelist) if target_dataset_whitelist else '<none>'}",
            flush=True,
        )
        print(
            f"[WSM_v1] dense_enhance={_de_flag} via={_de_src} | levers: "
            f"A.timestep_shift={_DE_TIMESTEP_SHIFT} B.minsnr={_DE_MINSNR} C.huber={_DE_HUBER} "
            f"(shift={_DENSE_ENHANCE_SHIFT} scale={_DENSE_ENHANCE_SCALE} "
            f"gamma={_DENSE_ENHANCE_GAMMA} huber_delta={_DENSE_ENHANCE_HUBER_DELTA})",
            flush=True,
        )
        print(
            f"[WSM_v1] literature_ablations single_step_x0={_SINGLE_STEP_X0} "
            f"high_noise_anchor={_HIGH_NOISE_ANCHOR} via={_hna_src} multires_noise={_MULTIRES_NOISE}",
            flush=True,
        )
        print(
            f"[WSM_v1] audio_condition_options separate_condition_adapters={args.separate_condition_adapters} "
            f"audio_global_condition={args.audio_global_condition} audio_conditioning_mode={args.audio_conditioning_mode}",
            flush=True,
        )
    control_tokens = {"caption", "Reference_IMG", "FirstFrame_RGB", "CurrentFrame_RGB_Reference", "CurrentFrame_RGB_Condition"}
    visual_conditions = [c for c in conditions if c not in control_tokens]
    first_frame_condition_modality = "0-2" if "FirstFrame_RGB" in conditions else None
    current_reference_modality = None
    current_rgb_condition_modality = None
    current_extra_condition_modality = None
    local_historical_reference = False

    if "CurrentFrame_RGB_Condition" in conditions:
        reference_modality = None
        current_rgb_condition_modality = "0-2"
        latent_conditions = visual_conditions
        use_reference = False
    elif "CurrentFrame_RGB_Reference" in conditions:
        reference_modality = "0-2"
        current_reference_modality = "0-2"
        latent_conditions = visual_conditions
        use_reference = True
    elif "Reference_IMG" in conditions:
        # First-frame routing comes from this task's recipe ("native" by default
        # for narrow_generation); WSG_REFERENCE_IMG_CONDITION=1 forces the
        # trainable Channel_Head condition channel for ablation.
        ff_mode = task_recipe(task_type)["first_frame"] or "native"
        if os.environ.get("WSG_REFERENCE_IMG_CONDITION") == "1":
            ff_mode = "condition"
        if ff_mode == "condition":
            reference_modality = None
            latent_conditions = visual_conditions
            first_frame_condition_modality = "0-2"
            use_reference = False
        else:  # "native"
            reference_modality = "0-2"
            latent_conditions = visual_conditions
            use_reference = True
    elif task_type == "inverse_rendering" and "0-2" in visual_conditions and len(visual_conditions) > 1:
        reference_modality = None
        latent_conditions = [c for c in visual_conditions if c != "0-2"]
        first_frame_condition_modality = "0-2"
        current_extra_condition_modality = "0-2" if args.historical_reference_policy == "tolerant" else None
        use_reference = False
    elif len(visual_conditions) == 1:
        reference_modality = None
        latent_conditions = visual_conditions
        use_reference = False
    elif len(visual_conditions) == 0:
        reference_modality = None
        latent_conditions = []
        use_reference = False
    else:
        raise ValueError(f"Unsupported condition configuration for task_type={task_type!r}: {conditions}")

    if len(latent_conditions) > 1:
        raise ValueError(f"Expected at most 1 latent condition after reference routing, got {latent_conditions}")
    if use_reference and len(latent_conditions) > 1:
        raise ValueError(f"Expected at most 1 latent condition with reference, got {latent_conditions}")

    # For caption-only mode, use None as condition_modality.
    condition_modality = latent_conditions[0] if latent_conditions else None

    # XPose was encoded into its own paired RGB/pose cache. The main dense cache
    # intentionally has empty 122-124 shards, so route the complete task pair to
    # the XPose cache whenever either side uses that modality.
    default_cache_dir = resolve_path(args.dense_latent_cache_dir)
    uses_xpose_cache = "122-124" in {args.target_modality, condition_modality}
    if uses_xpose_cache:
        if not args.xpose_latent_cache_dir:
            raise ValueError("Modality 122-124 requires --xpose-latent-cache-dir")
        cache_dir = resolve_path(args.xpose_latent_cache_dir)
        reference_dataset_prefixes = None
        reference_cache_tag = "xpose_v2"
    else:
        cache_dir = default_cache_dir
        reference_dataset_prefixes = ("AVIS_",)
        reference_cache_tag = "avis_v2"

    # XPose stores exactly two frames from each video in the same physical shard.
    # Reuse that mmap payload for its historical first frame instead of copying a
    # separate ~1GB reference cache.
    if uses_xpose_cache and use_reference and current_reference_modality is None:
        local_historical_reference = True
        current_reference_modality = reference_modality

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = (
        build_first_frame_reference_cache(
            cache_dir, reference_modality,
            dataset_prefixes=reference_dataset_prefixes, cache_tag=reference_cache_tag,
        )
        if use_reference and current_reference_modality is None else None
    )
    extra_condition_path = (
        build_first_frame_reference_cache(
            cache_dir, first_frame_condition_modality,
            dataset_prefixes=reference_dataset_prefixes, cache_tag=reference_cache_tag,
        )
        if first_frame_condition_modality else None
    )
    text_cache_dir = resolve_path(args.text_embedding_cache_dir) if args.text_embedding_cache_dir else None

    val_dataset = None
    try:
        use_audio_dense_condition = bool(args.audio_dense_condition_train)
        if use_audio_dense_condition:
            if condition_modality not in {"175-184", "Audio_AST", "audio_ast"}:
                raise ValueError(f"Audio dense condition requires condition 175-184/Audio_AST, got {condition_modality!r}")
            rgb_condition_modality = current_rgb_condition_modality or current_reference_modality
            audio_target_cache_dir = resolve_path(args.audio_dense_target_cache_dir) if args.audio_dense_target_cache_dir else None
            audio_target_modality = args.audio_dense_target_modality or None
            dataset = WSGAudioConditionDenseDataset(
                cache_dir, args.target_modality, resolve_path(args.audio_dense_condition_train),
                rgb_condition_modality, bool(current_rgb_condition_modality), rank, world_size, args.max_local_samples, entry,
                target_dataset_whitelist=target_dataset_whitelist,
                target_cache_dir=audio_target_cache_dir,
                target_cache_modality=audio_target_modality,
            )
            val_dataset = None
            if rank == 0 and args.val_loss_batches > 0:
                val_audio_path = resolve_path(args.audio_dense_condition_val or args.audio_dense_condition_train)
                val_dataset = WSGAudioConditionDenseDataset(
                    cache_dir, args.target_modality, val_audio_path,
                    rgb_condition_modality, bool(current_rgb_condition_modality), 0, 1, args.max_val_samples, entry,
                    target_dataset_whitelist=target_dataset_whitelist,
                    target_cache_dir=audio_target_cache_dir,
                    target_cache_modality=audio_target_modality,
                )
            condition_modality = "Audio_AST"
        else:
            inference_data_rank = int(args.inference_shard_id) if args.inference_only else rank
            inference_data_world = int(args.inference_num_shards) if args.inference_only else world_size
            inference_data_split = args.preview_split if args.inference_only else "train"
            dataset = WSGOminiLatentDataset(
                cache_dir, args.target_modality, condition_modality, reference_path,
                current_reference_modality, extra_condition_path, first_frame_condition_modality, current_extra_condition_modality,
                args.historical_reference_policy, local_historical_reference, inference_data_rank, inference_data_world, args.max_local_samples, entry, text_cache_dir,
                split=inference_data_split, val_split_ratio=args.val_split_ratio,
                target_dataset_whitelist=target_dataset_whitelist,
            )
            val_dataset = None
            if rank == 0 and args.val_loss_batches > 0:
                val_dataset = WSGOminiLatentDataset(
                    cache_dir, args.target_modality, condition_modality, reference_path,
                    current_reference_modality, extra_condition_path, first_frame_condition_modality, current_extra_condition_modality,
                    args.historical_reference_policy, local_historical_reference, 0, 1, args.max_val_samples, entry, text_cache_dir,
                    split="val", val_split_ratio=args.val_split_ratio,
                    target_dataset_whitelist=target_dataset_whitelist,
                    lazy_latents=True,
                )
    except RuntimeError as exc:
        no_pair_message = (
            "No pairs for" in str(exc)
            or "No audio-conditioned dense pairs for" in str(exc)
        )
        if not no_pair_message:
            raise
        if "dataset" in locals():
            print(f"[WSM_v1] validation split unavailable after runtime filtering ({exc}); continuing without validation", flush=True)
            val_dataset = None
        else:
            if rank == 0:
                marker = output_dir / "NO_TRAINING_NO_PAIRS.txt"
                marker.write_text(
                    "No training was run because runtime filters left zero target/condition pairs.\n"
                    f"channel_head_index={args.channel_head_index}\n"
                    f"target_modality={args.target_modality}\n"
                    f"condition_modalities={args.condition_modalities}\n"
                    f"target_dataset_whitelist={sorted(target_dataset_whitelist)}\n"
                    f"reason={exc}\n",
                    encoding="utf-8",
                )
                with (output_dir / "resolved_runtime_config.json").open("w", encoding="utf-8") as handle:
                    json.dump({
                        "skipped": True,
                        "skip_reason": str(exc),
                        "channel_head_index": args.channel_head_index,
                        "target_modality": args.target_modality,
                        "condition_modalities": args.condition_modalities,
                        "target_dataset_whitelist": sorted(target_dataset_whitelist),
                        "entry": entry,
                    }, handle, indent=2, ensure_ascii=False)
            print(f"[WSM_v1] skipped Channel_Head_{args.channel_head_index:06d}: {exc}", flush=True)
            return

    if not use_audio_dense_condition and (reference_path is not None or extra_condition_path is not None or local_historical_reference):
        local_usage = dict(dataset.reference_usage)
        gathered_usage = [local_usage]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_usage = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_usage, local_usage)
        elif world_size > 1:
            usage_part = output_dir / f".historical_reference_usage_rank_{rank}.json"
            usage_tmp = usage_part.with_suffix(".tmp")
            with usage_tmp.open("w", encoding="utf-8") as handle:
                json.dump(local_usage, handle)
            usage_tmp.replace(usage_part)
            if rank == 0:
                usage_parts = [
                    output_dir / f".historical_reference_usage_rank_{item_rank}.json"
                    for item_rank in range(world_size)
                ]
                deadline = time.time() + 600
                while not all(path.exists() for path in usage_parts):
                    if time.time() >= deadline:
                        missing = [str(path) for path in usage_parts if not path.exists()]
                        raise TimeoutError(f"Timed out waiting for reference usage ranks: {missing}")
                    time.sleep(0.2)
                gathered_usage = [
                    json.loads(path.read_text(encoding="utf-8")) for path in usage_parts
                ]
                for path in usage_parts:
                    path.unlink(missing_ok=True)
        usage = {
            "channel_head_index": args.channel_head_index,
            "task_type": task_type,
            "policy": args.historical_reference_policy,
        }
        count_keys = (
            "samples",
            "historical_candidates",
            "historical_native_reference",
            "omitted_native_reference_fallback",
            "historical_rgb_condition",
            "current_rgb_condition_fallback",
            "strict_dropped",
            "fallback_candidates",
        )
        for key in count_keys:
            usage[key] = sum(int(item.get(key, 0)) for item in gathered_usage if item)
        fallback_count = usage["omitted_native_reference_fallback"] + usage["current_rgb_condition_fallback"]
        usage["fallback_used"] = fallback_count > 0
        usage["fallback_count"] = fallback_count
        usage["fallback_fraction"] = fallback_count / max(usage["samples"], 1)
        usage["fallback_mode_activated"] = any(
            bool(item.get("fallback_mode_activated", False)) for item in gathered_usage if item
        )
        rank_effective_modes = [
            str(item.get("effective_mode", "unknown")) for item in gathered_usage if item
        ]
        usage["selection_scope"] = "split"
        usage["fallback_rule"] = "only_when_no_historical_samples_in_split"
        usage["rank_effective_modes"] = rank_effective_modes
        usage["fallback_mode_activated_ranks"] = sum(
            mode == "tolerant_fallback_only" for mode in rank_effective_modes
        )
        usage["effective_mode"] = (
            rank_effective_modes[0]
            if rank_effective_modes and len(set(rank_effective_modes)) == 1
            else "mixed_across_ranks"
        )
        if rank == 0:
            with (output_dir / "historical_reference_usage.json").open("w", encoding="utf-8") as handle:
                json.dump(usage, handle, indent=2, ensure_ascii=False)
    val_count = len(val_dataset) if val_dataset is not None else 0

    # Log different modes
    if not use_reference and condition_modality is None:
        mode_str = f"caption-only->{args.target_modality}"
    elif not use_reference:
        current_mode = f"+current_frame_condition({current_rgb_condition_modality})" if current_rgb_condition_modality else ""
        extra_mode = f"+first_frame_condition({first_frame_condition_modality})" if first_frame_condition_modality else ""
        mode_str = f"condition({condition_modality}){current_mode}{extra_mode}->{args.target_modality}"
    else:
        if condition_modality:
            mode_str = f"native_reference({reference_modality})+condition({condition_modality})->{args.target_modality}"
        else:
            mode_str = f"native_reference({reference_modality})->{args.target_modality}"

    print(f"[WSM_v1] rank={rank}/{world_size} train_samples={len(dataset)} val_samples={val_count} {mode_str}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure branches based on mode
    if use_reference:
        if condition_modality:
            branches = ["text", "target", condition_modality, "native_reference"]
            condition_type = f"{condition_modality}+native_reference({reference_modality})"
        else:
            branches = ["text", "target", "native_reference"]
            condition_type = f"native_reference({reference_modality})"
    elif condition_modality:
        current_rgb_branch = [f"current_frame_condition_{current_rgb_condition_modality}"] if current_rgb_condition_modality else []
        if first_frame_condition_modality and current_extra_condition_modality:
            extra_condition_branch = [f"historical_or_current_rgb_{first_frame_condition_modality}"]
        elif first_frame_condition_modality:
            extra_condition_branch = [f"first_frame_{first_frame_condition_modality}"]
        elif current_extra_condition_modality:
            extra_condition_branch = [f"current_frame_condition_{current_extra_condition_modality}"]
        else:
            extra_condition_branch = []
        if use_audio_dense_condition and current_rgb_branch:
            # Audio-dense batches put current RGB in omini_condition_latent and append
            # projected audio afterward; keep branch names aligned with that token order.
            branches = ["text", "target"] + current_rgb_branch + [condition_modality] + extra_condition_branch
            condition_type = f"current_frame_condition({current_rgb_condition_modality})+{condition_modality}"
        else:
            branches = ["text", "target", condition_modality] + current_rgb_branch + extra_condition_branch
            condition_type = condition_modality + (f"+current_frame_condition({current_rgb_condition_modality})" if current_rgb_condition_modality else "")
        condition_type += (f"+first_frame_condition({first_frame_condition_modality})" if first_frame_condition_modality else "")
        condition_type += (f"+current_frame_condition({current_extra_condition_modality})" if current_extra_condition_modality else "")
    else:
        branches = ["text", "target"]
        condition_type = "caption_only"

    condition_lane_count = ((1 if condition_modality else 0)
                            + (1 if current_rgb_condition_modality else 0)
                            + (1 if (first_frame_condition_modality or current_extra_condition_modality) else 0))
    condition_adapter = "default" if _tc_flag else None
    target_adapter = "default" if _tt_flag else None
    if getattr(args, "separate_condition_adapters", False):
        condition_adapters = [f"condition_{idx}" for idx in range(condition_lane_count)]
    else:
        condition_adapters = [condition_adapter] * condition_lane_count
    adapter_routing = [None, target_adapter] + condition_adapters
    if use_reference:
        adapter_routing.append("default" if current_reference_modality and not local_historical_reference else None)

    config = {
        "flux_path": str(resolve_path(args.model_path)),
        "dtype": "bfloat16",
        "model": {
            "base": "FLUX.2-klein-4B",
            "tunnelingnet_strategy": "channel_head_routing",
            "adapter_routing": adapter_routing,
            "branches": branches,
            "reference_position_t": 10,
            "current_reference_modality": current_reference_modality,
            "current_rgb_condition_modality": current_rgb_condition_modality,
            "audio_dense_condition": bool(args.audio_dense_condition_train),
            "separate_condition_adapters": bool(args.separate_condition_adapters),
        },
        "train": {
            "accumulate_grad_batches": args.gradient_accumulation_steps,
            "dataloader_workers": args.num_workers,
            "val_dataloader_workers": args.val_num_workers,
            "batch_size": args.batch_size,
            "save_interval": args.save_every_steps,
            "val_interval": args.eval_every_steps,
            "early_stop_seconds": args.early_stop_seconds,
            "sample_interval": args.viz_every_steps if args.viz_every_steps is not None else args.eval_every_steps,
            "print_every_n_steps": args.log_every_steps,
            "max_steps": 2 if args.smoke_test else args.max_steps,
            "max_epochs": -1,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_clip_val": 0.5,
            "save_path": str(output_dir),
            "condition_type": condition_type,
            "modality_type_prompt": dataset.modality_type_prompt,
            "channel_head_config": {"r": args.channel_head_rank, "lora_alpha": args.channel_head_rank if args.channel_head_alpha is None else args.channel_head_alpha},
            "optimizer": {"type": "AdamW", "params": {"lr": args.learning_rate, "weight_decay": 0.01}},
            "disable_preflight": args.disable_preflight,
            "text_embedding_cache_dir": str(resolve_path(args.text_embedding_cache_dir)) if args.text_embedding_cache_dir else None,
            "run_name": ".",
            "display_name": f"Channel_Head_{args.channel_head_index:06d}_rank{args.channel_head_rank}",
            "save_optimizer_state_8bit": bool(args.save_optimizer_state_8bit),
            "val_split_ratio": args.val_split_ratio,
            "max_val_samples": args.max_val_samples,
            "val_loss_batches": args.val_loss_batches,
            "preview_sample_id": args.preview_sample_id,
            "preview_video_id": args.preview_video_id,
            "preview_frame_index": args.preview_frame_index,
            "preview_frame_fraction": args.preview_frame_fraction,
            "preview_num_samples": args.preview_num_samples,
            "target_dataset_whitelist": sorted(target_dataset_whitelist),
            "edit_loss_weight": float(args.edit_loss_weight),
            "edit_loss_power": float(args.edit_loss_power),
            "gradient_domain_loss_weight": float(args.gradient_domain_loss_weight),
            "focal_regression_gamma": float(args.focal_regression_gamma),
            "foreground_iou_loss_weight": float(args.foreground_iou_loss_weight),
            "categorical_loss_mode": str(args.categorical_loss_mode),
            "categorical_loss_weight": float(args.categorical_loss_weight),
            "categorical_temperature": float(args.categorical_temperature),
            "categorical_margin": float(args.categorical_margin),
            "categorical_num_clusters": int(args.categorical_num_clusters),
            "categorical_kmeans_iterations": int(args.categorical_kmeans_iterations),
            "audio_conditioning_mode": str(args.audio_conditioning_mode),
            "historical_reference_policy": args.historical_reference_policy,
        },
        "wsg": {"channel_head_index": args.channel_head_index, "entry": entry, "effective_latent_cache": str(cache_dir), "uses_xpose_cache": uses_xpose_cache, "reference_cache": str(reference_path), "extra_condition_cache": str(extra_condition_path), "current_extra_condition_modality": current_extra_condition_modality, "local_historical_reference": local_historical_reference},
    }
    with (output_dir / "resolved_runtime_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    # Keep evaluation-only semantic metadata on args without changing the
    # training/checkpoint format.
    args.target_description = str(entry.get("target_description") or "")
    model = Flux2TunnelingNetModel(args, config["train"]["optimizer"], specify_lora)
    preview_batches = []
    if rank == 0:
        selector_audio_by_sample = {}
        if args.categorical_refiner_audio_dataset:
            selector_audio = np.load(
                resolve_path(args.categorical_refiner_audio_dataset), allow_pickle=False
            )
            selector_ids = [
                f"{str(wsv_path)}::{str(frame_key)}-{str(frame_key)}"
                for wsv_path, frame_key in zip(
                    selector_audio["wsv_path"], selector_audio["frame_key"]
                )
            ]
            selector_audio_by_sample = {
                sample_id: torch.from_numpy(
                    np.asarray(selector_audio["audio_feat"][index], dtype=np.float32)
                )
                for index, sample_id in enumerate(selector_ids)
            }
        # The fixed preview is meant to track a held-out sample, so draw it from the
        # test split (loaded the same way as train/val). Fall back to the train
        # dataset when this Channel_Head has no test pairs.
        preview_dataset = dataset
        if use_audio_dense_condition:
            # The audio-conditioned dense path is driven by train/val NPZ files, not
            # by ordinary visual-latent condition shards. Use val when available so
            # preview stays held-out without trying to load a non-existent Audio_AST shard.
            if val_dataset is not None:
                preview_dataset = val_dataset
        elif args.inference_only:
            # Batch annotation already loaded exactly one logical shard and split.
            preview_dataset = dataset
        else:
            try:
                preview_dataset = WSGOminiLatentDataset(
                    cache_dir, args.target_modality, condition_modality, reference_path,
                    current_reference_modality, extra_condition_path, first_frame_condition_modality, current_extra_condition_modality,
                    args.historical_reference_policy, local_historical_reference, 0, 8, None, entry, text_cache_dir,
                    split=args.preview_split, val_split_ratio=args.val_split_ratio,
                    target_dataset_whitelist=target_dataset_whitelist,
                    lazy_latents=True,
                )
            except RuntimeError as exc:
                if args.preview_require_split:
                    raise RuntimeError(
                        f"No valid {args.preview_split!r} samples for required preview split"
                    ) from exc
                print(f"[WSM_v1] no test split for preview ({exc}); using train split", flush=True)
        preview_num_samples = max(1, int(args.preview_num_samples or 1))
        if args.preview_sample_id_file:
            sample_id_path = resolve_path(args.preview_sample_id_file)
            requested_sample_ids = [
                line.strip() for line in sample_id_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if not requested_sample_ids:
                raise RuntimeError(f"No sample ids found in {sample_id_path}")
            preview_indices = [
                preview_dataset.find_preview_index(sample_id, "", -1, args.preview_frame_fraction)
                for sample_id in requested_sample_ids
            ]
            resolved_sample_ids = [
                str(preview_dataset.records[index][0])
                for index in preview_indices
            ]
            mismatches = [
                (requested, resolved)
                for requested, resolved in zip(
                    requested_sample_ids, resolved_sample_ids
                )
                if requested != resolved
            ]
            if mismatches:
                raise RuntimeError(
                    f"Requested preview sample ids are absent from the selected "
                    f"cache/split: {mismatches[:5]}"
                )
            if len(set(preview_indices)) != len(preview_indices):
                raise RuntimeError(
                    f"Duplicate dataset rows resolved from {sample_id_path}: {preview_indices}"
                )
        elif args.preview_random:
            rng = random.Random(args.seed + args.channel_head_index * 1009)
            population = list(range(len(preview_dataset)))
            if selector_audio_by_sample and hasattr(preview_dataset, "records"):
                population = [
                    index for index, record in enumerate(preview_dataset.records)
                    if str(record[0]) in selector_audio_by_sample
                ]
                if not population:
                    raise RuntimeError(
                        "No preview samples overlap categorical-refiner audio dataset"
                    )
            preview_indices = rng.sample(population, k=min(preview_num_samples, len(population)))
        elif preview_num_samples == 1 or args.preview_sample_id or args.preview_video_id or args.preview_frame_index >= 0:
            preview_indices = [
                preview_dataset.find_preview_index(
                    args.preview_sample_id,
                    args.preview_video_id,
                    args.preview_frame_index,
                    args.preview_frame_fraction,
                )
            ]
        else:
            rng = random.Random(args.seed + args.channel_head_index * 1009)
            population = list(range(len(preview_dataset)))
            preview_indices = rng.sample(population, k=min(preview_num_samples, len(population)))
        for preview_slot, preview_index in enumerate(preview_indices):
            preview_source = preview_dataset[preview_index]
            if selector_audio_by_sample:
                sample_id = str(preview_source.get("sample_id") or "")
                if sample_id not in selector_audio_by_sample:
                    raise RuntimeError(
                        f"No categorical-refiner audio feature for preview {sample_id}"
                    )
                preview_source["audio_condition_feature"] = selector_audio_by_sample[sample_id]
            preview_batches.append(to_preview_batch(preview_source))
            print(
                f"[WSM_v1] fixed preview sample[{preview_slot}] split={args.preview_split if preview_dataset is not dataset else 'train'} "
                f"index={preview_index} "
                f"sample_id={preview_source.get('sample_id', '')} "
                f"video_id={preview_source.get('video_id', '')} frame={preview_source.get('target_frame', '')}",
                flush=True,
            )

    def preview_fn(pl_module, out_dir, label):
        if preview_batches:
            pl_module.inference_samples = preview_batches
            pl_module.inference_sample = preview_batches[0]
            for preview_slot, preview_batch in enumerate(preview_batches):
                sample_out_dir = Path(out_dir) / f"sample_{preview_slot:02d}" if len(preview_batches) > 1 else Path(out_dir)
                pl_module.run_preview(preview_batch, sample_out_dir, label)

    if args.inference_only:
        if rank != 0:
            return
        if not args.inference_checkpoint:
            raise ValueError("--inference-only requires --inference-checkpoint")
        checkpoint = resolve_path(args.inference_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        raw_channel_head_state, embedded_refiner, checkpoint_format = (
            load_channel_head_checkpoint(checkpoint)
        )
        channel_head_state = {
            key.removeprefix("transformer."): value
            for key, value in raw_channel_head_state.items()
        }
        load_result = set_peft_model_state_dict(
            model.transformer, channel_head_state, adapter_name="default"
        )
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if unexpected:
            raise RuntimeError(
                f"Unexpected Channel Head inference keys in {checkpoint}: {unexpected[:5]}"
            )
        condition_adapter_file = checkpoint.parent / "condition_lora_adapters.pt"
        if getattr(model, "extra_lora_adapter_names", None) and condition_adapter_file.exists():
            condition_states = torch.load(condition_adapter_file, map_location="cpu")
            for adapter_name, adapter_state in condition_states.items():
                set_peft_model_state_dict(model.transformer, adapter_state, adapter_name=adapter_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        resolved_refiner = resolve_categorical_refiner(
            checkpoint,
            embedded=embedded_refiner,
            selection=args.categorical_refiner,
        )
        if (
            str(args.target_modality) in CATEGORICAL_TARGET_MODALITIES
            and str(args.categorical_refiner) == "auto"
            and resolved_refiner is None
        ):
            raise RuntimeError(
                f"Categorical target {args.target_modality} requires an embedded or "
                f"run-level categorical_refiner sidecar bound to this checkpoint; "
                "use --categorical-refiner off only for explicit baseline debugging."
            )
        model.categorical_refiner_runtime = (
            CategoricalRefinerRuntime(resolved_refiner, device)
            if resolved_refiner is not None else None
        )
        if resolved_refiner is not None:
            print(
                f"[WSM_v1] categorical_refiner={resolved_refiner.source_kind} "
                f"source={resolved_refiner.source_path}",
                flush=True,
            )
        else:
            print("[WSM_v1] categorical_refiner=off_or_unavailable", flush=True)
        resolved_audio_selector = resolve_audio_region_selector(checkpoint)
        audio_selector_required = (
            str(entry.get("adapter_task") or "")
            == "rgb_to_semantic_mask_audio_region_selector"
        )
        if audio_selector_required and resolved_audio_selector is None:
            raise RuntimeError(
                "This sparse audio-region-selection task requires the run-level "
                "audio_region_selector sidecar"
            )
        model.audio_region_selector_runtime = (
            AudioRegionSelectorRuntime(resolved_audio_selector, device)
            if resolved_audio_selector is not None else None
        )
        if resolved_audio_selector is not None:
            print(
                f"[WSM_v1] audio_region_selector={resolved_audio_selector.source_path}",
                flush=True,
            )
        print(f"[WSM_v1] checkpoint_format={checkpoint_format}", flush=True)
        model.eval()
        if not preview_batches:
            raise RuntimeError("Inference-only mode could not select a preview sample")
        model.inference_samples = preview_batches
        model.inference_sample = preview_batches[0]
        inference_dir = output_dir / "pseudo-visualization-inference" / args.inference_label
        inference_dir.mkdir(parents=True, exist_ok=True)
        skipped_reader_samples = 0
        for preview_slot, preview_batch in enumerate(preview_batches):
            sample_dir = (
                inference_dir / f"sample_{preview_slot:02d}"
                if len(preview_batches) > 1
                else inference_dir
            )
            try:
                model.run_inference(
                    preview_batch,
                    sample_dir,
                    args.inference_label,
                    num_steps=args.inference_steps,
                )
            except WSGReaderNoPrimaryOutput as exc:
                skipped_reader_samples += 1
                sample_dir.mkdir(parents=True, exist_ok=True)
                with (sample_dir / "skipped.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "status": "skipped",
                            "reason": "no_primary_wsg_reader_png",
                            "message": str(exc),
                            "sample_slot": preview_slot,
                            "sample_id": str(preview_batch.get("sample_id", [""])[0]),
                        },
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(
                    f"[WSM_v1] skipped inference sample slot={preview_slot}: {exc}",
                    flush=True,
                )
        if skipped_reader_samples:
            print(
                f"[WSM_v1] skipped {skipped_reader_samples}/{len(preview_batches)} "
                "samples without a primary WSG Reader PNG",
                flush=True,
            )
        print(
            f"[WSM_v1] inference-only complete head={args.channel_head_index:06d} "
            f"checkpoint={checkpoint} output={inference_dir}",
            flush=True,
        )
        return

    train_with_wsm_monitoring(dataset, val_dataset, model, config, test_function=preview_fn, resume_checkpoint_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
