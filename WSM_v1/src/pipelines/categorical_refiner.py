#!/usr/bin/env python3
"""Train and evaluate a lightweight region-category refiner for WSM heads.

The generative Channel Head remains responsible for mask geometry.  This
optional, independently trained module applies one complete foreground region
to the Head's already-encoded condition latent, predicts one category for that
region, and recolors the original-resolution generated region with an exact
target-palette color.  No WSM generation parameters are changed.

The module is deliberately generic over condition modality and palette.  The
first pilot targets AVIS channel 195 for Head 038 (RGB) and Head 308 (normals).
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wsg_reader.categorical_visualization import (  # noqa: E402
    AVIS_CATEGORY_COLORS,
)


DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT / "data/wsg_flux2_frame_latents_dense_multidataset/shards"
)
DEFAULT_CATEGORICAL_ROOT = DEFAULT_CACHE_ROOT.parent / "categorical_targets"
DEFAULT_FEATURE_ROOT = DEFAULT_CACHE_ROOT.parent / "categorical_refiner_features"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/black-forest-labs/FLUX.2-klein-4B"
BASE_STEP_PATTERN = re.compile(r"(?:final_step_|step_|^)(\d+)$")
CATEGORICAL_REFINER_DIR = "categorical_refiner"
CATEGORICAL_REFINER_MANIFEST = "manifest.json"
CATEGORICAL_REFINER_FORMAT = "wsm_categorical_refiner_sidecar_v1"
AUDIO_REGION_SELECTOR_DIR = "audio_region_selector"
AUDIO_REGION_SELECTOR_MANIFEST = "manifest.json"
AUDIO_REGION_SELECTOR_FORMAT = "wsm_audio_region_selector_sidecar_v1"
COMBINED_CHECKPOINT_FORMAT = "wsm_channel_head_with_categorical_refiner_v1"
CATEGORICAL_TARGET_MODALITIES = frozenset({"111", "131", "135", "136", "195"})
SCALAR_CATEGORY_DISPLAY_RANGES = {
    "135": (0.0, 1.0),
}


def scalar_category_display_contract(
    target_modality: str,
    category_values: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the exact pseudo-RGB palette used by scalar WSG VAE inputs.

    Invalid WSG pixels remain -1 before VAE encoding and render as black. The
    category colors must therefore use the normalization range from the dense
    WSG contract rather than only the range of valid category IDs. Black is
    declared as background for geometry recovery during refinement.
    """
    value_range = SCALAR_CATEGORY_DISPLAY_RANGES.get(str(target_modality))
    if value_range is None:
        return None
    values = torch.as_tensor(category_values, dtype=torch.float32)
    low, high = value_range
    fractions = ((values - float(low)) / max(float(high - low), 1e-6)).clamp(0.0, 1.0)
    grayscale = torch.round(fractions * 255.0).to(torch.uint8)
    prototypes = grayscale[:, None].expand(-1, 3).contiguous()
    return prototypes, torch.zeros(3, dtype=torch.uint8)


@lru_cache(maxsize=4096)
def _sha256_file_revision(
    path_text: str, size_bytes: int, mtime_ns: int, chunk_size: int
) -> str:
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_file_revision(
        str(resolved), int(stat.st_size), int(stat.st_mtime_ns), int(chunk_size)
    )


def _json_list(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_categorical_target_contract(
    target_modality: str,
    categorical_root: Path = DEFAULT_CATEGORICAL_ROOT,
) -> dict[str, Any]:
    """Read the portable category/palette contract shared by all target shards."""
    shard_paths = sorted((Path(categorical_root) / str(target_modality)).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No categorical target sidecars for modality {target_modality}: {categorical_root}"
        )
    payload = torch.load(
        shard_paths[0], map_location="cpu", mmap=True, weights_only=False
    )
    category_values = [int(value) for value in _json_list(payload["category_values"])]
    category_prototypes_rgb = _json_list(payload["category_prototypes_rgb"])
    background_rgb = _json_list(payload.get("background_rgb"))
    scalar_contract = scalar_category_display_contract(target_modality, category_values)
    if scalar_contract is not None:
        prototypes, background = scalar_contract
        category_prototypes_rgb = prototypes.tolist()
        background_rgb = background.tolist()
    return {
        "category_values": category_values,
        "category_prototypes_rgb": category_prototypes_rgb,
        "background_rgb": background_rgb,
        "spatial_encoding": str(
            (payload.get("contract") or {}).get(
                "spatial_encoding", "zero_background_one_based_category_index"
            )
        ),
    }


def enrich_refiner_payload(
    payload: Mapping[str, Any],
    categorical_root: Path = DEFAULT_CATEGORICAL_ROOT,
) -> dict[str, Any]:
    """Make a classifier checkpoint self-contained without changing its weights."""
    enriched = dict(payload)
    config = dict(enriched.get("config") or {})
    target = str(config.get("target_modality") or "")
    if target in SCALAR_CATEGORY_DISPLAY_RANGES:
        contract = load_categorical_target_contract(target, Path(categorical_root))
        for key, value in contract.items():
            config[key] = value
    elif target and (
        not config.get("category_prototypes_rgb")
        or "background_rgb" not in config
    ):
        contract = load_categorical_target_contract(target, Path(categorical_root))
        for key, value in contract.items():
            config.setdefault(key, value)
    condition_modalities = config.get("condition_modalities")
    if not condition_modalities:
        condition = str(config.get("condition_modality") or "")
        config["condition_modalities"] = (
            parse_condition_modalities(condition) if condition else []
        )
    config["enabled_by_default"] = True
    enriched["config"] = config
    return enriched


def safe_modality_name(modality: str) -> str:
    return str(modality).replace("/", "_").replace(",", "_").replace("-", "_")


def parse_condition_modalities(
    value: str | Sequence[str],
) -> list[str]:
    """Normalize one or more ordered condition modalities."""
    raw_values = value.split(",") if isinstance(value, str) else list(value)
    modalities = [str(item).strip() for item in raw_values if str(item).strip()]
    modalities = list(dict.fromkeys(modalities))
    if not modalities:
        raise ValueError("At least one condition modality is required")
    return modalities


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_training_curves(metrics_path: Path, output_dir: Path) -> None:
    """Render loss and accuracy histories recorded by a classifier run."""
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    train_loss = np.asarray([float(row["train_loss"]) for row in rows])
    val_loss = np.asarray([float(row["val_loss"]) for row in rows])
    val_accuracy = np.asarray([float(row["val_accuracy"]) for row in rows])
    val_balanced = np.asarray([float(row["val_balanced_accuracy"]) for row in rows])
    best_index = int(np.argmax(val_balanced))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(steps, train_loss, label="train loss", linewidth=1.8)
    axes[0].plot(steps, val_loss, label="validation loss", linewidth=1.8)
    axes[0].set(xlabel="Step", ylabel="Cross-entropy loss", title="Classifier loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, val_accuracy, label="validation accuracy", linewidth=1.8)
    axes[1].plot(steps, val_balanced, label="balanced accuracy", linewidth=1.8)
    axes[1].scatter(
        [steps[best_index]], [val_balanced[best_index]], s=42, zorder=3,
        label=f"best balanced @ {steps[best_index]}",
    )
    axes[1].set(xlabel="Step", ylabel="Accuracy", title="Classifier validation accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    figure.savefig(output_dir / "training_curves.pdf")
    plt.close(figure)


def load_modality_shards(
    cache_root: Path, modality: str
) -> list[tuple[Path, dict[str, Any]]]:
    safe_modality = safe_modality_name(modality)
    paths = sorted((cache_root / safe_modality).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No latent shards found for modality {modality}: {cache_root}")
    return [
        (path, torch.load(path, map_location="cpu", mmap=True, weights_only=False))
        for path in paths
    ]


def build_sample_index(
    shards: list[tuple[Path, dict[str, Any]]]
) -> dict[str, tuple[int, int]]:
    output: dict[str, tuple[int, int]] = {}
    for shard_index, (_path, payload) in enumerate(shards):
        for row_index, sample_id in enumerate(payload["sample_ids"]):
            output[str(sample_id)] = (shard_index, row_index)
    return output


def load_flux_vae(model_path: Path, device: torch.device):
    from diffusers import AutoencoderKLFlux2

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKLFlux2.from_pretrained(
        str(model_path), subfolder="vae", torch_dtype=dtype
    )
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    vae.to(device=device, dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae, dtype


@torch.inference_mode()
def decode_condition_latents(
    vae: nn.Module, latents: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return vae.decode(latents.to(device=device, dtype=dtype)).sample.float().clamp(-1.0, 1.0)


def decoded_modality_masks(decoded: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Apply the complete category mask to a VAE-decoded pseudo-RGB modality."""
    masks = F.interpolate(
        masks.float(), size=decoded.shape[-2:], mode="nearest-exact"
    ).clamp(0.0, 1.0)
    return decoded * masks - (1.0 - masks)


def masked_condition_latents(latents: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Keep one complete spatial region in the structured VAE condition latent."""
    if masks.shape[-2:] != latents.shape[-2:]:
        masks = F.interpolate(
            masks.float(), size=latents.shape[-2:], mode="nearest-exact"
        )
    return latents.float() * masks.float().clamp(0.0, 1.0)


class MaskedLatentClassifier(nn.Module):
    """Fast spatial classifier over the exact condition representation seen by a Head."""

    def __init__(
        self,
        latent_channels: int = 32,
        num_classes: int = 26,
        embedding_dim: int = 512,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(15.0), dtype=torch.float32))

    def forward(self, masked_latents: torch.Tensor) -> torch.Tensor:
        spatial = self.encoder(masked_latents.float())
        pooled = torch.cat(
            (F.adaptive_avg_pool2d(spatial, 1), F.adaptive_max_pool2d(spatial, 1)),
            dim=1,
        ).flatten(1)
        embedding = F.normalize(pooled + self.adapter(pooled), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        return self.logit_scale.exp().clamp(max=100.0) * embedding @ prototypes.T


class AudioRegionSelector(nn.Module):
    """Select zero or more generated regions using frame-aligned audio.

    Geometry remains the responsibility of the RGB-to-mask Channel Head. This
    audio selector compares every candidate RGB region with the AST audio
    embedding and predicts whether the region is sounding.  The already
    trained RGB categorical refiner independently assigns the exact category.
    """

    def __init__(
        self,
        latent_channels: int = 32,
        audio_dim: int = 768,
        embedding_dim: int = 512,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256), nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256), nn.SiLU(),
        )
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim),
        )
        self.audio_encoder = nn.Sequential(
            nn.LayerNorm(audio_dim), nn.Linear(audio_dim, embedding_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim),
        )
        self.selection_scale = nn.Parameter(torch.tensor(math.log(15.0)))
        self.selection_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.visual_adapter[-1].weight)
        nn.init.zeros_(self.visual_adapter[-1].bias)

    def forward(
        self, masked_latents: torch.Tensor, audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one independent selection logit per candidate.

        ``masked_latents`` is ``B,K,C,H,W`` and ``audio_features`` is ``B,D``.
        K remains independent so one sound may select several regions at once.
        """
        if masked_latents.ndim != 5:
            raise ValueError(
                f"Expected B,K,C,H,W candidate latents, got {tuple(masked_latents.shape)}"
            )
        batch, candidates, channels, height, width = masked_latents.shape
        flat = masked_latents.reshape(batch * candidates, channels, height, width)
        spatial = self.visual_encoder(flat.float())
        visual = torch.cat(
            (F.adaptive_avg_pool2d(spatial, 1), F.adaptive_max_pool2d(spatial, 1)),
            dim=1,
        ).flatten(1)
        visual = F.normalize(visual + self.visual_adapter(visual), dim=1)
        visual = visual.reshape(batch, candidates, -1)
        audio = F.normalize(self.audio_encoder(audio_features.float()), dim=1)
        expanded_audio = audio[:, None, :].expand(-1, candidates, -1)
        selection = (
            self.selection_scale.exp().clamp(max=100.0)
            * (visual * expanded_audio).sum(dim=-1) + self.selection_bias
        )
        return selection


class RegionPrototypeClassifier(nn.Module):
    """Small per-head classifier over one decoded three-channel modality mask."""

    def __init__(
        self,
        num_classes: int = 26,
        embedding_dim: int = 512,
        dropout: float = 0.20,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        visual_dim = int(backbone.fc.in_features)
        if visual_dim != embedding_dim:
            raise ValueError(f"Expected ResNet18 feature dim {embedding_dim}, got {visual_dim}")
        backbone.fc = nn.Identity()
        self.visual_backbone = backbone
        for parameter in self.visual_backbone.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(15.0), dtype=torch.float32))

    def base_features(self, masked_image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            masked_image.float(), size=(224, 224), mode="bilinear", align_corners=False
        )
        image = ((image + 1.0) * 0.5 - self.imagenet_mean) / self.imagenet_std
        self.visual_backbone.eval()
        with torch.no_grad():
            return self.visual_backbone(image)

    def forward(self, masked_image: torch.Tensor) -> torch.Tensor:
        features = self.base_features(masked_image)
        return self.forward_features(features)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify cached features produced by :meth:`base_features`."""
        embedding = F.normalize(features + self.adapter(features), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        return self.logit_scale.exp().clamp(max=100.0) * embedding @ prototypes.T


def compact_classifier_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Exclude the shared frozen ImageNet backbone from every per-head checkpoint."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("visual_backbone.")
    }


def load_compact_classifier_state(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Load a per-head checkpoint while reconstructing the shared backbone locally."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in missing if not key.startswith("visual_backbone.")
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Invalid compact categorical-refiner state: "
            f"missing={invalid_missing}, unexpected={list(unexpected)}"
        )


def categorical_record_split(
    sample_id: str, source: Mapping[str, Any], holdout_ratio: float = 0.05
) -> str:
    """Mirror the Channel-Head train/val/test split for every source dataset."""
    dataset_name = str(source.get("dataset_name") or source.get("dataset") or "")
    if dataset_name.startswith("AVIS_train"):
        return "train"
    if dataset_name.startswith("AVIS_val"):
        return "val"
    if dataset_name.startswith("AVIS_test"):
        return "test"
    key = f"{dataset_name}:{sample_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / float(2**64)
    holdout = max(0.0, min(0.49, float(holdout_ratio)))
    if bucket < holdout:
        return "val"
    if bucket < holdout * 2.0:
        return "test"
    return "train"


class DenseRegionDataset(Dataset):
    def __init__(
        self,
        categorical_sidecar_dir: Path,
        cache_root: Path,
        condition_modality: str,
        target_modality: str,
        split: str,
        minimum_region_cells: int,
    ) -> None:
        sidecar_paths = sorted(categorical_sidecar_dir.glob("shard_*.pt"))
        if not sidecar_paths:
            raise FileNotFoundError(
                f"No categorical sidecars found: {categorical_sidecar_dir}"
            )
        self.sidecars = [
            torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            for path in sidecar_paths
        ]
        first_values = torch.as_tensor(self.sidecars[0]["category_values"], dtype=torch.long)
        for sidecar in self.sidecars:
            contract = sidecar.get("contract") or {}
            if int(contract.get("version", 0)) < 3:
                raise RuntimeError(
                    f"Categorical sidecar v3 with spatial maps is required: {categorical_sidecar_dir}"
                )
            if "category_spatial_index_map" not in sidecar:
                raise RuntimeError(
                    f"Categorical sidecar has no spatial map: {categorical_sidecar_dir}"
                )
            if not torch.equal(
                first_values,
                torch.as_tensor(sidecar["category_values"], dtype=torch.long),
            ):
                raise RuntimeError("Category values changed across sidecar shards")
        self.category_values = first_values
        self.category_prototypes_rgb = torch.as_tensor(
            self.sidecars[0]["category_prototypes_rgb"], dtype=torch.uint8
        )
        self.background_rgb = self.sidecars[0].get("background_rgb")
        if self.background_rgb is not None:
            self.background_rgb = torch.as_tensor(self.background_rgb, dtype=torch.uint8)
        scalar_contract = scalar_category_display_contract(
            target_modality, self.category_values
        )
        if scalar_contract is not None:
            self.category_prototypes_rgb, self.background_rgb = scalar_contract
        self.spatial_encoding = str(
            (self.sidecars[0].get("contract") or {}).get("spatial_encoding", "")
        )
        self.num_classes = len(first_values)
        self.condition_modalities = parse_condition_modalities(condition_modality)
        self.condition_shards_by_modality = {
            modality: load_modality_shards(cache_root, modality)
            for modality in self.condition_modalities
        }
        condition_indices = {
            modality: build_sample_index(shards)
            for modality, shards in self.condition_shards_by_modality.items()
        }
        # Preserve the historical attribute for single-condition callers.
        self.condition_shards = self.condition_shards_by_modality[
            self.condition_modalities[0]
        ]
        target_shards = load_modality_shards(cache_root, target_modality)
        target_sources: dict[str, Mapping[str, Any]] = {}
        for _path, payload in target_shards:
            sources = payload.get("sample_sources") or [{} for _ in payload["sample_ids"]]
            target_sources.update(
                {str(sample_id): source for sample_id, source in zip(payload["sample_ids"], sources)}
            )
        self.records: list[tuple[int, ...]] = []
        for sidecar_index, sidecar in enumerate(self.sidecars):
            spatial_maps = sidecar["category_spatial_index_map"]
            failed_sample_ids = {
                str(sample_id) for sample_id in sidecar.get("failed_sample_ids", [])
            }
            for sidecar_row, (sample_id, spatial_map) in enumerate(
                zip(sidecar["sample_ids"], spatial_maps)
            ):
                sample_id = str(sample_id)
                if sample_id in failed_sample_ids:
                    continue
                source = target_sources.get(sample_id) or {}
                if split and categorical_record_split(sample_id, source) != split:
                    continue
                if any(
                    sample_id not in condition_indices[modality]
                    for modality in self.condition_modalities
                ):
                    continue
                condition_locations = [
                    condition_indices[modality][sample_id]
                    for modality in self.condition_modalities
                ]
                for one_based_index in torch.unique(spatial_map).tolist():
                    one_based_index = int(one_based_index)
                    if one_based_index <= 0:
                        continue
                    if int((spatial_map == one_based_index).sum()) < int(minimum_region_cells):
                        continue
                    class_index = one_based_index - 1
                    flattened_locations = tuple(
                        value
                        for shard_and_row in condition_locations
                        for value in shard_and_row
                    )
                    self.records.append(
                        (sidecar_index, sidecar_row, class_index, *flattened_locations)
                    )
        if not self.records:
            raise RuntimeError(f"No {split} regions for condition {condition_modality}")
        self.labels = np.asarray([item[2] for item in self.records], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        sidecar_index, sidecar_row, class_index = record[:3]
        locations = record[3:]
        latents = []
        for modality_index, modality in enumerate(self.condition_modalities):
            shard_index, row_index = locations[
                modality_index * 2 : modality_index * 2 + 2
            ]
            latent = self.condition_shards_by_modality[modality][shard_index][1][
                "latents"
            ][row_index]
            if latent.ndim == 4:
                latent = latent[:, 0]
            latents.append(latent.float())
        spatial_shapes = {tuple(latent.shape[-2:]) for latent in latents}
        if len(spatial_shapes) != 1:
            raise RuntimeError(
                f"Condition latent shapes differ for {self.condition_modalities}: "
                f"{sorted(spatial_shapes)}"
            )
        latent = torch.cat(latents, dim=0)
        spatial_map = self.sidecars[sidecar_index]["category_spatial_index_map"][sidecar_row]
        mask = (spatial_map == class_index + 1)[None]
        return latent, mask.float(), torch.tensor(class_index, dtype=torch.long)

    def records_signature(self) -> str:
        records = np.asarray(self.records, dtype=np.int32)
        return hashlib.sha256(records.tobytes()).hexdigest()


def _crop_audio_source_mask(
    source_mask: np.ndarray,
    crop_info: Mapping[str, Any],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """Map the legacy full-frame 224 mask into the dense center-crop frame."""
    mask = torch.as_tensor(np.asarray(source_mask) > 0, dtype=torch.float32)[None, None]
    source_h, source_w = mask.shape[-2:]
    original_h = max(1, int(crop_info.get("original_height") or source_h))
    original_w = max(1, int(crop_info.get("original_width") or source_w))
    crop_y = float(crop_info.get("crop_y") or 0)
    crop_x = float(crop_info.get("crop_x") or 0)
    patch_h = float(crop_info.get("patch_hw") or min(original_h, original_w))
    y0 = max(0, min(source_h - 1, int(math.floor(crop_y * source_h / original_h))))
    x0 = max(0, min(source_w - 1, int(math.floor(crop_x * source_w / original_w))))
    y1 = max(y0 + 1, min(source_h, int(math.ceil((crop_y + patch_h) * source_h / original_h))))
    x1 = max(x0 + 1, min(source_w, int(math.ceil((crop_x + patch_h) * source_w / original_w))))
    cropped = mask[..., y0:y1, x0:x1]
    return F.interpolate(cropped, size=output_shape, mode="nearest-exact")[0, 0]


class AudioRegionSelectionDataset(Dataset):
    """One frame with all candidate channel-195 regions and audio supervision."""

    def __init__(
        self,
        categorical_sidecar_dir: Path,
        cache_root: Path,
        audio_dataset: Path,
        minimum_region_cells: int = 4,
        maximum_candidates: int = 24,
    ) -> None:
        sidecar_paths = sorted(Path(categorical_sidecar_dir).glob("shard_*.pt"))
        if not sidecar_paths:
            raise FileNotFoundError(categorical_sidecar_dir)
        self.sidecars = [
            torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            for path in sidecar_paths
        ]
        self.category_values = torch.as_tensor(
            self.sidecars[0]["category_values"], dtype=torch.long
        )
        self.category_prototypes_rgb = torch.as_tensor(
            self.sidecars[0]["category_prototypes_rgb"], dtype=torch.uint8
        )
        self.background_rgb = torch.as_tensor(
            self.sidecars[0].get("background_rgb", (0, 0, 0)), dtype=torch.uint8
        )
        self.rgb_shards = load_modality_shards(Path(cache_root), "0-2")
        rgb_index = build_sample_index(self.rgb_shards)
        categorical_index: dict[str, tuple[int, int]] = {}
        for shard_index, payload in enumerate(self.sidecars):
            categorical_index.update(
                {
                    str(sample_id): (shard_index, row_index)
                    for row_index, sample_id in enumerate(payload["sample_ids"])
                }
            )
        audio = np.load(Path(audio_dataset), allow_pickle=False)
        self.audio_features = np.asarray(audio["audio_feat"], dtype=np.float32)
        self.audio_masks = np.asarray(audio["mask"], dtype=np.uint8)
        self.audio_dim = int(self.audio_features.shape[1])
        self.audio_encoder = str(audio["audio_encoder"].item())
        self.audio_model_name = str(audio["audio_model_name"].item())
        audio_sample_ids = [
            f"{str(wsv_path)}::{str(frame_key)}-{str(frame_key)}"
            for wsv_path, frame_key in zip(audio["wsv_path"], audio["frame_key"])
        ]
        self.records: list[dict[str, Any]] = []
        skipped_without_positive = 0
        for audio_index, sample_id in enumerate(audio_sample_ids):
            if sample_id not in rgb_index or sample_id not in categorical_index:
                continue
            rgb_shard, rgb_row = rgb_index[sample_id]
            sidecar_shard, sidecar_row = categorical_index[sample_id]
            spatial = self.sidecars[sidecar_shard]["category_spatial_index_map"][
                sidecar_row
            ].long()
            crop_infos = self.rgb_shards[rgb_shard][1].get("crop_infos") or []
            crop_info = crop_infos[rgb_row] if rgb_row < len(crop_infos) else {}
            source = _crop_audio_source_mask(
                self.audio_masks[audio_index], crop_info, tuple(spatial.shape)
            ).numpy() > 0
            if not source.any():
                continue
            candidates: list[tuple[np.ndarray, int, bool, int]] = []
            for one_based_index in torch.unique(spatial).tolist():
                one_based_index = int(one_based_index)
                if one_based_index <= 0:
                    continue
                category_mask = spatial.numpy() == one_based_index
                components, count = ndimage.label(
                    category_mask, structure=np.ones((3, 3), dtype=bool)
                )
                for component_id in range(1, count + 1):
                    component = components == component_id
                    pixels = int(component.sum())
                    if pixels < int(minimum_region_cells):
                        continue
                    overlap = int(np.logical_and(component, source).sum())
                    candidates.append(
                        (component, one_based_index - 1, overlap > 0, pixels)
                    )
            positives = [item for item in candidates if item[2]]
            if not positives:
                skipped_without_positive += 1
                continue
            negatives = sorted(
                (item for item in candidates if not item[2]),
                key=lambda item: item[3], reverse=True,
            )
            candidates = positives + negatives[: max(0, int(maximum_candidates) - len(positives))]
            self.records.append(
                {
                    "sample_id": sample_id,
                    "rgb_shard": rgb_shard,
                    "rgb_row": rgb_row,
                    "audio_index": audio_index,
                    "masks": torch.from_numpy(
                        np.stack([item[0] for item in candidates]).astype(np.float32)
                    ),
                    "selection": torch.tensor(
                        [item[2] for item in candidates], dtype=torch.float32
                    ),
                    "categories": torch.tensor(
                        [item[1] for item in candidates], dtype=torch.long
                    ),
                }
            )
        if not self.records:
            raise RuntimeError(
                f"No aligned audio/candidate-mask records in {audio_dataset}"
            )
        print(
            f"[audio-region-selector] dataset={Path(audio_dataset).name} "
            f"frames={len(self.records)} skipped_without_positive={skipped_without_positive} "
            f"audio_dim={self.audio_dim}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latent = self.rgb_shards[record["rgb_shard"]][1]["latents"][record["rgb_row"]]
        if latent.ndim == 4:
            latent = latent[:, 0]
        masks = record["masks"][:, None]
        masked = latent.float()[None] * masks
        return {
            "sample_id": record["sample_id"],
            "candidate_latents": masked,
            "audio": torch.from_numpy(self.audio_features[record["audio_index"]]),
            "selection": record["selection"],
            "categories": record["categories"],
        }


def collate_audio_region_selection(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    maximum = max(int(item["candidate_latents"].shape[0]) for item in batch)
    sample = batch[0]["candidate_latents"]
    latents = sample.new_zeros((len(batch), maximum, *sample.shape[1:]))
    selection = torch.zeros((len(batch), maximum), dtype=torch.float32)
    categories = torch.full((len(batch), maximum), -1, dtype=torch.long)
    valid = torch.zeros((len(batch), maximum), dtype=torch.bool)
    for row, item in enumerate(batch):
        count = int(item["candidate_latents"].shape[0])
        latents[row, :count] = item["candidate_latents"]
        selection[row, :count] = item["selection"]
        categories[row, :count] = item["categories"]
        valid[row, :count] = True
    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "candidate_latents": latents,
        "audio": torch.stack([item["audio"] for item in batch]),
        "selection": selection,
        "categories": categories,
        "valid": valid,
    }


class RegionFeatureDataset(Dataset):
    """Compact frozen features for fast repeated classifier experiments."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.features = torch.as_tensor(payload["features"])
        self.labels_tensor = torch.as_tensor(payload["labels"], dtype=torch.long)
        self.labels = self.labels_tensor.numpy()
        self.category_values = torch.as_tensor(payload["category_values"], dtype=torch.long)
        self.num_classes = len(self.category_values)
        if len(self.features) != len(self.labels_tensor):
            raise RuntimeError("Categorical feature and label counts do not match")

    def __len__(self) -> int:
        return len(self.labels_tensor)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index].float(), self.labels_tensor[index]


def feature_cache_path(
    feature_root: Path, target_modality: str, condition_modality: str, split: str
) -> Path:
    return (
        feature_root
        / f"target_{safe_modality_name(target_modality)}"
        / f"condition_{safe_modality_name(condition_modality)}"
        / f"{split}.pt"
    )


def load_valid_feature_cache(
    path: Path,
    dense_dataset: DenseRegionDataset,
    condition_modality: str,
    target_modality: str,
    split: str,
) -> RegionFeatureDataset | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    contract = payload.get("contract") or {}
    expected = {
        "version": 1,
        "kind": "decoded_mask_resnet18_features",
        "condition_modality": str(condition_modality),
        "target_modality": str(target_modality),
        "split": str(split),
        "records_signature": dense_dataset.records_signature(),
        "records": len(dense_dataset),
        "feature_dim": 512,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        print(f"[categorical-refiner] stale feature cache will be rebuilt: {path}", flush=True)
        return None
    category_values = torch.as_tensor(payload.get("category_values"), dtype=torch.long)
    if not torch.equal(category_values, dense_dataset.category_values):
        print(f"[categorical-refiner] category values changed; rebuilding: {path}", flush=True)
        return None
    print(
        f"[categorical-refiner] loaded feature cache split={split} "
        f"records={len(dense_dataset)} -> {path}",
        flush=True,
    )
    return RegionFeatureDataset(payload)


@torch.inference_mode()
def _build_feature_cache_unlocked(
    path: Path,
    dense_dataset: DenseRegionDataset,
    extractor: RegionPrototypeClassifier,
    vae: nn.Module,
    vae_dtype: torch.dtype,
    device: torch.device,
    condition_modality: str,
    target_modality: str,
    split: str,
    batch_size: int,
) -> RegionFeatureDataset:
    # Feature rows must retain exactly the same order as ``dense_dataset.records``.
    loader = _loader(dense_dataset, batch_size, sampler=None, shuffle=False)
    features = torch.empty((len(dense_dataset), 512), dtype=torch.float16)
    labels = torch.empty(len(dense_dataset), dtype=torch.int16)
    cursor = 0
    started = time.monotonic()
    extractor.eval()
    for batch_index, (latent, mask, label) in enumerate(loader, start=1):
        decoded = decode_condition_latents(vae, latent, device, vae_dtype)
        batch_features = extractor.base_features(
            decoded_modality_masks(decoded, mask.to(device, non_blocking=True))
        )
        count = len(label)
        features[cursor : cursor + count] = batch_features.detach().cpu().to(torch.float16)
        labels[cursor : cursor + count] = label.to(torch.int16)
        cursor += count
        if batch_index == 1 or batch_index % 20 == 0 or cursor == len(dense_dataset):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = cursor / elapsed
            remaining = (len(dense_dataset) - cursor) / max(rate, 1e-6)
            print(
                f"[categorical-refiner] caching {condition_modality}->{target_modality} "
                f"{split}: {cursor}/{len(dense_dataset)} ({100.0 * cursor / len(dense_dataset):.1f}%) "
                f"rate={rate:.1f} regions/s eta={remaining / 60.0:.1f}m",
                flush=True,
            )
    records = torch.as_tensor(np.asarray(dense_dataset.records, dtype=np.int32))
    contract = {
        "version": 1,
        "kind": "decoded_mask_resnet18_features",
        "condition_modality": str(condition_modality),
        "target_modality": str(target_modality),
        "split": str(split),
        "records_signature": dense_dataset.records_signature(),
        "records": len(dense_dataset),
        "feature_dim": 512,
        "source_representation": "flux2_vae_decoded_complete_modality_mask_image",
        "feature_extractor": "torchvision_resnet18_imagenet1k_v1",
    }
    payload = {
        "contract": contract,
        "features": features,
        "labels": labels,
        "records": records,
        "category_values": dense_dataset.category_values,
    }
    atomic_torch_save(payload, path)
    print(
        f"[categorical-refiner] saved feature cache records={len(dense_dataset)} -> {path}",
        flush=True,
    )
    return RegionFeatureDataset(payload)


def build_feature_cache(
    path: Path,
    dense_dataset: DenseRegionDataset,
    extractor: RegionPrototypeClassifier,
    vae: nn.Module,
    vae_dtype: torch.dtype,
    device: torch.device,
    condition_modality: str,
    target_modality: str,
    split: str,
    batch_size: int,
) -> RegionFeatureDataset:
    """Build once per modality signature; concurrent Heads wait and then reuse it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached = load_valid_feature_cache(
            path,
            dense_dataset,
            condition_modality,
            target_modality,
            split,
        )
        if cached is not None:
            return cached
        return _build_feature_cache_unlocked(
            path,
            dense_dataset,
            extractor,
            vae,
            vae_dtype,
            device,
            condition_modality,
            target_modality,
            split,
            batch_size,
        )


def _loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Any = None,
    shuffle: bool | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = sampler is None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


@torch.inference_mode()
def evaluate_classifier(
    model: RegionPrototypeClassifier,
    vae: nn.Module,
    vae_dtype: torch.dtype,
    loader: DataLoader,
    device: torch.device,
    category_values: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    per_class_correct = Counter()
    per_class_total = Counter()
    confusion = torch.zeros(
        (len(category_values), len(category_values)), dtype=torch.long
    )
    for latent, mask, label in loader:
        latent = latent.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        decoded = decode_condition_latents(vae, latent, device, vae_dtype)
        logits = model(decoded_modality_masks(decoded, mask))
        loss_sum += float(F.cross_entropy(logits, label, reduction="sum").item())
        pred = logits.argmax(dim=1)
        correct += int((pred == label).sum())
        total += label.numel()
        flat_pairs = label.detach().cpu() * len(category_values) + pred.detach().cpu()
        confusion += torch.bincount(
            flat_pairs, minlength=len(category_values) ** 2
        ).reshape(len(category_values), len(category_values))
        for target, is_correct in zip(label.cpu().tolist(), (pred == label).cpu().tolist()):
            category_value = int(category_values[int(target)])
            per_class_total[category_value] += 1
            per_class_correct[category_value] += int(is_correct)
    per_class = {
        str(class_id): per_class_correct[class_id] / max(per_class_total[class_id], 1)
        for class_id in sorted(per_class_total)
    }
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "correct": correct,
        "samples": total,
        "per_class_accuracy": per_class,
        "per_class_support": {str(key): value for key, value in sorted(per_class_total.items())},
        "confusion_category_values": category_values.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate_feature_classifier(
    model: RegionPrototypeClassifier,
    loader: DataLoader,
    device: torch.device,
    category_values: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    per_class_correct = Counter()
    per_class_total = Counter()
    confusion = torch.zeros(
        (len(category_values), len(category_values)), dtype=torch.long
    )
    for features, label in loader:
        features = features.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model.forward_features(features)
        loss_sum += float(F.cross_entropy(logits, label, reduction="sum").item())
        pred = logits.argmax(dim=1)
        correct += int((pred == label).sum())
        total += label.numel()
        flat_pairs = label.detach().cpu() * len(category_values) + pred.detach().cpu()
        confusion += torch.bincount(
            flat_pairs, minlength=len(category_values) ** 2
        ).reshape(len(category_values), len(category_values))
        for target, is_correct in zip(label.cpu().tolist(), (pred == label).cpu().tolist()):
            category_value = int(category_values[int(target)])
            per_class_total[category_value] += 1
            per_class_correct[category_value] += int(is_correct)
    per_class = {
        str(class_id): per_class_correct[class_id] / max(per_class_total[class_id], 1)
        for class_id in sorted(per_class_total)
    }
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "correct": correct,
        "samples": total,
        "per_class_accuracy": per_class,
        "per_class_support": {str(key): value for key, value in sorted(per_class_total.items())},
        "confusion_category_values": category_values.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate_latent_classifier(
    model: MaskedLatentClassifier,
    loader: DataLoader,
    device: torch.device,
    category_values: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    per_class_correct = Counter()
    per_class_total = Counter()
    confusion = torch.zeros(
        (len(category_values), len(category_values)), dtype=torch.long
    )
    for latent, mask, label in loader:
        latent = latent.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(masked_condition_latents(latent, mask))
        loss_sum += float(F.cross_entropy(logits, label, reduction="sum").item())
        pred = logits.argmax(dim=1)
        correct += int((pred == label).sum())
        total += label.numel()
        flat_pairs = label.detach().cpu() * len(category_values) + pred.detach().cpu()
        confusion += torch.bincount(
            flat_pairs, minlength=len(category_values) ** 2
        ).reshape(len(category_values), len(category_values))
        for target, is_correct in zip(label.cpu().tolist(), (pred == label).cpu().tolist()):
            category_value = int(category_values[int(target)])
            per_class_total[category_value] += 1
            per_class_correct[category_value] += int(is_correct)
    per_class = {
        str(class_id): per_class_correct[class_id] / max(per_class_total[class_id], 1)
        for class_id in sorted(per_class_total)
    }
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "correct": correct,
        "samples": total,
        "per_class_accuracy": per_class,
        "per_class_support": {str(key): value for key, value in sorted(per_class_total.items())},
        "confusion_category_values": category_values.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def train_latent_classifier(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Each sample performs several tiny 60x60 tensor operations. Letting every
    # worker operation fan out over all host threads costs substantially more
    # than the operations themselves.
    torch.set_num_threads(1)
    device = torch.device(args.device)
    categorical_sidecar_dir = Path(args.categorical_root) / str(args.target_modality)
    train_set = DenseRegionDataset(
        categorical_sidecar_dir,
        Path(args.cache_root),
        args.condition_modality,
        args.target_modality,
        "train",
        args.minimum_region_cells,
    )
    val_set = DenseRegionDataset(
        categorical_sidecar_dir,
        Path(args.cache_root),
        args.condition_modality,
        args.target_modality,
        "val",
        args.minimum_region_cells,
    )
    if not torch.equal(train_set.category_values, val_set.category_values):
        raise RuntimeError("Train/validation category values do not match")
    if train_set.condition_modalities != val_set.condition_modalities:
        raise RuntimeError("Train/validation condition modalities do not match")
    sample_latent, _sample_mask, _sample_label = train_set[0]
    latent_channels = int(sample_latent.shape[0])
    condition_modalities = list(train_set.condition_modalities)
    condition_signature = ",".join(condition_modalities)
    print(
        f"[categorical-refiner] fast latent conditions={condition_signature} "
        f"latent_channels={latent_channels} "
        f"train_regions={len(train_set)} val_regions={len(val_set)}",
        flush=True,
    )
    counts = np.bincount(train_set.labels, minlength=train_set.num_classes).astype(np.float64)
    # A normal epoch shuffle prevents the shard/sample ordering from creating
    # long, class-correlated stretches. Class imbalance is handled once, by the
    # weighted CE below; using a weighted sampler as well would double-correct.
    effective_max_steps = 2 if args.smoke_test else int(args.max_steps)
    effective_eval_every = 1 if args.smoke_test else int(args.eval_every)
    train_loader_dataset: Dataset = train_set
    val_loader_dataset: Dataset = val_set
    if args.smoke_test:
        smoke_regions = max(1, int(args.smoke_regions))
        train_count = min(smoke_regions, len(train_set))
        val_count = min(smoke_regions, len(val_set))
        train_indices = np.linspace(0, len(train_set) - 1, train_count, dtype=np.int64).tolist()
        val_indices = np.linspace(0, len(val_set) - 1, val_count, dtype=np.int64).tolist()
        train_loader_dataset = Subset(train_set, train_indices)
        val_loader_dataset = Subset(val_set, val_indices)
        print(
            f"[categorical-refiner] smoke subset train={train_count} val={val_count} "
            f"steps={effective_max_steps}",
            flush=True,
        )
    train_loader = _loader(
        train_loader_dataset, args.batch_size, sampler=None, shuffle=True
    )
    val_loader = _loader(
        val_loader_dataset, args.eval_batch_size, sampler=None, shuffle=False
    )
    model = MaskedLatentClassifier(
        latent_channels=latent_channels,
        num_classes=train_set.num_classes,
        dropout=args.dropout,
    ).to(device)
    class_weights = torch.from_numpy(
        np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    ).float().to(device)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 3,
        "mode": "independent_region_category_refiner",
        "enabled_by_default": False,
        "condition_modality": condition_signature,
        "condition_modalities": condition_modalities,
        "target_modality": str(args.target_modality),
        "head_index": int(args.head_index),
        "categorical_sidecar_dir": str(categorical_sidecar_dir.resolve()),
        "train_regions": len(train_set),
        "val_regions": len(val_set),
        "latent_channels": latent_channels,
        "embedding_dim": 512,
        "input_representation": "masked_structured_condition_latent",
        "num_classes": train_set.num_classes,
        "category_values": train_set.category_values.tolist(),
        "category_prototypes_rgb": train_set.category_prototypes_rgb.tolist(),
        "background_rgb": (
            train_set.background_rgb.tolist()
            if train_set.background_rgb is not None else None
        ),
        "spatial_encoding": train_set.spatial_encoding,
        "minimum_region_cells": int(args.minimum_region_cells),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "max_steps": int(effective_max_steps),
        "smoke_test": bool(args.smoke_test),
        "smoke_regions": int(args.smoke_regions),
        "latent_noise_std": float(args.latent_noise_std),
        "seed": int(args.seed),
        "architecture": (
            "complete category mask applied to the existing structured condition latent + "
            "small spatial CNN + residual MLP adapter + cosine prototypes"
        ),
    }
    atomic_json_dump(config, output_dir / "config.json")
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy"),
        )
        writer.writeheader()
        handle.flush()
        step = 0
        best_balanced_accuracy = -1.0
        best_overall_accuracy = -1.0
        best_validation_loss = float("inf")
        train_iterator = iter(train_loader)
        running_loss = 0.0
        running_batches = 0
        started = time.monotonic()
        while step < effective_max_steps:
            try:
                latent, mask, label = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                latent, mask, label = next(train_iterator)
            model.train()
            latent = latent.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            masked_latent = masked_condition_latents(latent, mask)
            if args.latent_noise_std > 0:
                masked_latent = masked_latent + (
                    torch.randn_like(masked_latent)
                    * float(args.latent_noise_std)
                    * mask.float()
                )
            logits = model(masked_latent)
            loss = F.cross_entropy(
                logits, label, weight=class_weights, label_smoothing=args.label_smoothing
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running_loss += float(loss.detach())
            running_batches += 1
            if step % effective_eval_every == 0 or step == effective_max_steps:
                val = evaluate_latent_classifier(
                    model, val_loader, device, train_set.category_values
                )
                train_loss = running_loss / max(running_batches, 1)
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": f"{train_loss:.10e}",
                        "val_loss": f"{val['loss']:.10e}",
                        "val_accuracy": f"{val['accuracy']:.10e}",
                        "val_balanced_accuracy": f"{val['balanced_accuracy']:.10e}",
                    }
                )
                handle.flush()
                checkpoint = {
                    "config": config,
                    "step": step,
                    "model": compact_classifier_state_dict(model),
                    "validation": val,
                }
                if args.save_alternative_checkpoints:
                    atomic_torch_save(checkpoint, output_dir / "last.pt")
                if args.keep_step_checkpoints:
                    atomic_torch_save(checkpoint, output_dir / f"step_{step:07d}.pt")
                metric_record = {"step": step, **val}
                if float(val["balanced_accuracy"]) > best_balanced_accuracy:
                    best_balanced_accuracy = float(val["balanced_accuracy"])
                    atomic_torch_save(checkpoint, output_dir / "best.pt")
                    atomic_json_dump(metric_record, output_dir / "best_metrics.json")
                if float(val["accuracy"]) > best_overall_accuracy:
                    best_overall_accuracy = float(val["accuracy"])
                    if args.save_alternative_checkpoints:
                        atomic_torch_save(checkpoint, output_dir / "best_accuracy.pt")
                    atomic_json_dump(
                        metric_record, output_dir / "best_accuracy_metrics.json"
                    )
                if float(val["loss"]) < best_validation_loss:
                    best_validation_loss = float(val["loss"])
                    if args.save_alternative_checkpoints:
                        atomic_torch_save(checkpoint, output_dir / "best_val_loss.pt")
                    atomic_json_dump(
                        metric_record, output_dir / "best_val_loss_metrics.json"
                    )
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[categorical-refiner] step={step} train={train_loss:.4f} "
                    f"val_acc={val['accuracy']:.4f} val_balanced={val['balanced_accuracy']:.4f} "
                    f"speed={step / elapsed:.1f} steps/s",
                    flush=True,
                )
                running_loss = 0.0
                running_batches = 0
    print(f"[categorical-refiner] best balanced accuracy={best_balanced_accuracy:.4f} -> {output_dir}", flush=True)


def train_decoded_feature_classifier(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    categorical_sidecar_dir = Path(args.categorical_root) / str(args.target_modality)
    dense_train_set = DenseRegionDataset(
        categorical_sidecar_dir,
        Path(args.cache_root),
        args.condition_modality,
        args.target_modality,
        "train",
        args.minimum_region_cells,
    )
    dense_val_set = DenseRegionDataset(
        categorical_sidecar_dir,
        Path(args.cache_root),
        args.condition_modality,
        args.target_modality,
        "val",
        args.minimum_region_cells,
    )
    if not torch.equal(dense_train_set.category_values, dense_val_set.category_values):
        raise RuntimeError("Train/validation category values do not match")
    print(
        f"[categorical-refiner] condition={args.condition_modality} "
        f"train_regions={len(dense_train_set)} val_regions={len(dense_val_set)}",
        flush=True,
    )
    model = RegionPrototypeClassifier(
        num_classes=dense_train_set.num_classes,
        dropout=args.dropout,
    ).to(device)
    feature_root = Path(args.feature_root)
    train_feature_path = feature_cache_path(
        feature_root, args.target_modality, args.condition_modality, "train"
    )
    val_feature_path = feature_cache_path(
        feature_root, args.target_modality, args.condition_modality, "val"
    )
    train_feature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_lock_path = train_feature_path.parent / "build_signature.lock"
    with signature_lock_path.open("a+b") as signature_lock:
        fcntl.flock(signature_lock.fileno(), fcntl.LOCK_EX)
        train_set = load_valid_feature_cache(
            train_feature_path,
            dense_train_set,
            args.condition_modality,
            args.target_modality,
            "train",
        )
        val_set = load_valid_feature_cache(
            val_feature_path,
            dense_val_set,
            args.condition_modality,
            args.target_modality,
            "val",
        )
        if train_set is None or val_set is None:
            print("[categorical-refiner] loading frozen FLUX.2 VAE for feature caching", flush=True)
            vae, vae_dtype = load_flux_vae(Path(args.model_path), device)
            if train_set is None:
                train_set = _build_feature_cache_unlocked(
                    train_feature_path,
                    dense_train_set,
                    model,
                    vae,
                    vae_dtype,
                    device,
                    args.condition_modality,
                    args.target_modality,
                    "train",
                    args.feature_cache_batch_size,
                )
            if val_set is None:
                val_set = _build_feature_cache_unlocked(
                    val_feature_path,
                    dense_val_set,
                    model,
                    vae,
                    vae_dtype,
                    device,
                    args.condition_modality,
                    args.target_modality,
                    "val",
                    args.feature_cache_batch_size,
                )
            del vae
            torch.cuda.empty_cache()
    # Feature extraction is frozen and complete.  Keep the shared backbone on
    # CPU so the actual per-Head training only occupies its tiny adapter.
    model.visual_backbone.to("cpu")
    categorical_contract = load_categorical_target_contract(args.target_modality, Path(args.categorical_root))
    del dense_train_set, dense_val_set
    counts = np.bincount(
        train_set.labels, minlength=train_set.num_classes
    ).astype(np.float64)
    sample_weights = np.asarray(
        [1.0 / math.sqrt(max(counts[label], 1.0)) for label in train_set.labels],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), num_samples=len(train_set), replacement=True
    )
    train_loader = _loader(train_set, args.batch_size, sampler=sampler)
    val_loader = _loader(val_set, args.eval_batch_size, sampler=None, shuffle=False)
    class_weights = torch.from_numpy(
        np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    ).float().to(device)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 2,
        "mode": "independent_region_category_refiner",
        "enabled_by_default": False,
        "condition_modality": str(args.condition_modality),
        "condition_modalities": [str(args.condition_modality)],
        "target_modality": str(args.target_modality),
        "head_index": int(args.head_index),
        "categorical_sidecar_dir": str(categorical_sidecar_dir.resolve()),
        "feature_cache_root": str(feature_root.resolve()),
        "train_feature_cache": str(train_feature_path.resolve()),
        "val_feature_cache": str(val_feature_path.resolve()),
        "train_regions": len(train_set),
        "val_regions": len(val_set),
        "model_path": str(Path(args.model_path).resolve()),
        "embedding_dim": 512,
        "classifier_input_channels": 3,
        "input_representation": "frozen_flux2_vae_decoded_condition_mask_image",
        "shared_backbone": "torchvision_resnet18_imagenet1k_v1",
        "checkpoint_excludes_shared_backbone": True,
        "num_classes": train_set.num_classes,
        "category_values": train_set.category_values.tolist(),
        "category_prototypes_rgb": categorical_contract["category_prototypes_rgb"],
        "background_rgb": categorical_contract["background_rgb"],
        "spatial_encoding": categorical_contract["spatial_encoding"],
        "minimum_region_cells": int(args.minimum_region_cells),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "feature_cache_batch_size": int(args.feature_cache_batch_size),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "architecture": (
            "frozen FLUX.2 VAE decode + complete three-channel modality mask image + "
            "shared frozen ResNet18 + per-head residual MLP adapter + cosine prototypes"
        ),
    }
    atomic_json_dump(config, output_dir / "config.json")
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy"),
        )
        writer.writeheader()
        step = 0
        best_balanced_accuracy = -1.0
        best_overall_accuracy = -1.0
        best_validation_loss = float("inf")
        train_iterator = iter(train_loader)
        running_loss = 0.0
        running_batches = 0
        while step < args.max_steps:
            try:
                features, label = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                features, label = next(train_iterator)
            model.train()
            features = features.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model.forward_features(features)
            loss = F.cross_entropy(
                logits, label, weight=class_weights, label_smoothing=args.label_smoothing
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running_loss += float(loss.detach())
            running_batches += 1
            if step % args.eval_every == 0 or step == args.max_steps:
                val = evaluate_feature_classifier(
                    model,
                    val_loader,
                    device,
                    train_set.category_values,
                )
                train_loss = running_loss / max(running_batches, 1)
                writer.writerow(
                    {
                        "step": step,
                        "train_loss": f"{train_loss:.10e}",
                        "val_loss": f"{val['loss']:.10e}",
                        "val_accuracy": f"{val['accuracy']:.10e}",
                        "val_balanced_accuracy": f"{val['balanced_accuracy']:.10e}",
                    }
                )
                handle.flush()
                checkpoint = {
                    "config": config,
                    "step": step,
                    "model": compact_classifier_state_dict(model),
                    "validation": val,
                }
                if args.save_alternative_checkpoints:
                    atomic_torch_save(checkpoint, output_dir / "last.pt")
                if args.keep_step_checkpoints:
                    atomic_torch_save(checkpoint, output_dir / f"step_{step:07d}.pt")
                metric_record = {"step": step, **val}
                if float(val["balanced_accuracy"]) > best_balanced_accuracy:
                    best_balanced_accuracy = float(val["balanced_accuracy"])
                    atomic_torch_save(checkpoint, output_dir / "best.pt")
                    atomic_json_dump(metric_record, output_dir / "best_metrics.json")
                if float(val["accuracy"]) > best_overall_accuracy:
                    best_overall_accuracy = float(val["accuracy"])
                    if args.save_alternative_checkpoints:
                        atomic_torch_save(checkpoint, output_dir / "best_accuracy.pt")
                    atomic_json_dump(
                        metric_record, output_dir / "best_accuracy_metrics.json"
                    )
                if float(val["loss"]) < best_validation_loss:
                    best_validation_loss = float(val["loss"])
                    if args.save_alternative_checkpoints:
                        atomic_torch_save(checkpoint, output_dir / "best_val_loss.pt")
                    atomic_json_dump(
                        metric_record, output_dir / "best_val_loss_metrics.json"
                    )
                print(
                    f"[categorical-refiner] step={step} train={train_loss:.4f} "
                    f"val_acc={val['accuracy']:.4f} val_balanced={val['balanced_accuracy']:.4f}",
                    flush=True,
                )
                running_loss = 0.0
                running_batches = 0
    print(f"[categorical-refiner] best balanced accuracy={best_balanced_accuracy:.4f} -> {output_dir}", flush=True)


def _audio_selector_loss(
    selection_logits: torch.Tensor,
    selection_target: torch.Tensor,
    valid: torch.Tensor,
    ranking_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positives = valid & (selection_target > 0.5)
    negatives = valid & ~positives
    positive_count = positives.sum().clamp_min(1)
    negative_count = negatives.sum().clamp_min(1)
    positive_weight = (negative_count.float() / positive_count.float()).clamp(1.0, 20.0)
    binary = F.binary_cross_entropy_with_logits(
        selection_logits[valid], selection_target[valid],
        pos_weight=positive_weight,
    )
    ranking_terms = []
    for row in range(selection_logits.shape[0]):
        row_positive = positives[row]
        row_negative = negatives[row]
        if row_positive.any() and row_negative.any():
            ranking_terms.append(
                F.relu(
                    0.25
                    - selection_logits[row][row_positive].min()
                    + selection_logits[row][row_negative].max()
                )
            )
    ranking = (
        torch.stack(ranking_terms).mean()
        if ranking_terms else selection_logits.sum() * 0.0
    )
    total = binary + float(ranking_weight) * ranking
    return total, {"binary": binary, "ranking": ranking}


@torch.inference_mode()
def evaluate_audio_region_selector(
    model: AudioRegionSelector,
    loader: DataLoader,
    device: torch.device,
    ranking_weight: float,
) -> dict[str, Any]:
    model.eval()
    stored: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    loss_sum = 0.0
    batches = 0
    for batch in loader:
        candidate_latents = batch["candidate_latents"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        selection_target = batch["selection"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        selection_logits = model(candidate_latents, audio)
        loss, _parts = _audio_selector_loss(
            selection_logits, selection_target, valid, ranking_weight,
        )
        loss_sum += float(loss.item())
        batches += 1
        stored.append((selection_logits.cpu(), selection_target.cpu(), valid.cpu()))
    best: dict[str, Any] | None = None
    for threshold in np.linspace(0.20, 0.80, 13).tolist():
        positive_correct = negative_correct = positive_total = negative_total = 0
        selected_true = selected_false = missed = 0
        frame_exact = frame_total = 0
        for selection_logits, target, valid in stored:
            predicted = selection_logits.sigmoid() >= float(threshold)
            positives = valid & (target > 0.5)
            negatives = valid & ~positives
            positive_correct += int((predicted & positives).sum())
            negative_correct += int((~predicted & negatives).sum())
            positive_total += int(positives.sum())
            negative_total += int(negatives.sum())
            selected_true += int((predicted & positives).sum())
            selected_false += int((predicted & negatives).sum())
            missed += int((~predicted & positives).sum())
            for row in range(valid.shape[0]):
                frame_exact += int(torch.equal(predicted[row][valid[row]], positives[row][valid[row]]))
                frame_total += 1
        precision = selected_true / max(selected_true + selected_false, 1)
        recall = selected_true / max(selected_true + missed, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        positive_recall = positive_correct / max(positive_total, 1)
        negative_recall = negative_correct / max(negative_total, 1)
        balanced = 0.5 * (positive_recall + negative_recall)
        accuracy = (positive_correct + negative_correct) / max(
            positive_total + negative_total, 1
        )
        record = {
            "loss": loss_sum / max(batches, 1),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "selection_precision": precision,
            "selection_recall": recall,
            "selection_f1": f1,
            "frame_exact_accuracy": frame_exact / max(frame_total, 1),
            "selection_threshold": float(threshold),
            "positive_candidates": positive_total,
            "negative_candidates": negative_total,
        }
        key = (record["selection_f1"], record["balanced_accuracy"], record["accuracy"])
        if best is None or key > (
            best["selection_f1"], best["balanced_accuracy"], best["accuracy"]
        ):
            best = record
    assert best is not None
    return best


def train_audio_region_selector(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    categorical_dir = Path(args.categorical_root) / str(args.target_modality)
    train_set = AudioRegionSelectionDataset(
        categorical_dir, Path(args.cache_root), Path(args.audio_train),
        args.minimum_region_cells, args.maximum_candidates,
    )
    val_set = AudioRegionSelectionDataset(
        categorical_dir, Path(args.cache_root), Path(args.audio_val),
        args.minimum_region_cells, args.maximum_candidates,
    )
    effective_steps = 2 if args.smoke_test else int(args.max_steps)
    if args.smoke_test:
        train_set = Subset(train_set, range(min(len(train_set), int(args.smoke_frames))))
        val_set = Subset(val_set, range(min(len(val_set), int(args.smoke_frames))))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_audio_region_selection,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_audio_region_selection,
    )
    base_train_set = train_set.dataset if isinstance(train_set, Subset) else train_set
    sample = base_train_set[0]["candidate_latents"]
    model = AudioRegionSelector(
        latent_channels=int(sample.shape[1]),
        audio_dim=int(base_train_set.audio_dim),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 4,
        "mode": "audio_region_selector",
        "enabled_by_default": False,
        "condition_modality": "0-2",
        "condition_modalities": ["0-2", "175-184"],
        "generation_condition_modalities": ["0-2"],
        "selector_condition_modalities": ["175-184"],
        "auxiliary_condition": "audio_ast",
        "target_modality": str(args.target_modality),
        "head_index": int(args.head_index),
        "latent_channels": int(sample.shape[1]),
        "audio_dim": int(base_train_set.audio_dim),
        "audio_encoder": base_train_set.audio_encoder,
        "audio_model_name": base_train_set.audio_model_name,
        "embedding_dim": 512,
        "input_representation": "masked_rgb_latent_candidate_set+audio_ast",
        "selection_mode": "independent_multiselect",
        "minimum_region_cells": int(args.minimum_region_cells),
        "maximum_candidates": int(args.maximum_candidates),
        "max_steps": int(effective_steps),
        "learning_rate": float(args.learning_rate),
        "ranking_loss_weight": float(args.ranking_loss_weight),
        "seed": int(args.seed),
        "architecture": "RGB candidate-set encoder + AST audio encoder + independent multi-selection head",
    }
    atomic_json_dump(config, output_dir / "config.json")
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step", "train_loss", "val_loss", "val_accuracy",
                "val_balanced_accuracy", "selection_precision", "selection_recall",
                "selection_f1", "frame_exact_accuracy",
            ),
        )
        writer.writeheader()
        train_iterator = iter(train_loader)
        running_loss = 0.0
        running_batches = 0
        best_score = -1.0
        for step in range(1, effective_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            model.train()
            candidate_latents = batch["candidate_latents"].to(device)
            audio = batch["audio"].to(device)
            selection_target = batch["selection"].to(device)
            valid = batch["valid"].to(device)
            selection_logits = model(candidate_latents, audio)
            loss, _parts = _audio_selector_loss(
                selection_logits, selection_target, valid, args.ranking_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach())
            running_batches += 1
            eval_every = 1 if args.smoke_test else int(args.eval_every)
            if step % eval_every == 0 or step == effective_steps:
                val = evaluate_audio_region_selector(
                    model, val_loader, device, args.ranking_loss_weight,
                )
                train_loss = running_loss / max(running_batches, 1)
                writer.writerow({
                    "step": step, "train_loss": f"{train_loss:.10e}",
                    "val_loss": f"{val['loss']:.10e}",
                    "val_accuracy": f"{val['accuracy']:.10e}",
                    "val_balanced_accuracy": f"{val['balanced_accuracy']:.10e}",
                    "selection_precision": f"{val['selection_precision']:.10e}",
                    "selection_recall": f"{val['selection_recall']:.10e}",
                    "selection_f1": f"{val['selection_f1']:.10e}",
                    "frame_exact_accuracy": f"{val['frame_exact_accuracy']:.10e}",
                })
                handle.flush()
                checkpoint_config = {**config, "selection_threshold": val["selection_threshold"]}
                checkpoint = {
                    "config": checkpoint_config, "step": step,
                    "model": compact_classifier_state_dict(model), "validation": val,
                }
                score = float(val["selection_f1"])
                if score > best_score:
                    best_score = score
                    atomic_torch_save(checkpoint, output_dir / "best.pt")
                    atomic_json_dump({"step": step, **val}, output_dir / "best_metrics.json")
                print(
                    f"[audio-region-selector] step={step} train={train_loss:.4f} "
                    f"select_f1={val['selection_f1']:.4f} "
                    f"frame_exact={val['frame_exact_accuracy']:.4f}", flush=True,
                )
                running_loss = 0.0
                running_batches = 0
    plot_training_curves(metrics_path, output_dir)
    print(f"[audio-region-selector] best selection F1={best_score:.4f} -> {output_dir}", flush=True)


def train_classifier(args: argparse.Namespace) -> None:
    if args.input_representation == "masked-latent":
        train_latent_classifier(args)
    elif args.input_representation == "decoded-mask-features":
        condition_modalities = parse_condition_modalities(args.condition_modality)
        if len(condition_modalities) != 1:
            raise ValueError(
                "decoded-mask-features supports exactly one condition modality; "
                "use the default masked-latent representation for multi-condition Heads"
            )
        train_decoded_feature_classifier(args)
    else:
        raise ValueError(args.input_representation)


@dataclass(frozen=True)
class ResolvedCategoricalRefiner:
    payload: Mapping[str, Any]
    source_kind: str
    source_path: Path
    manifest: Mapping[str, Any] | None = None

    def provenance(self) -> dict[str, Any]:
        config = dict(self.payload.get("config") or {})
        validation = dict(self.payload.get("validation") or {})
        return {
            "enabled": True,
            "source_kind": self.source_kind,
            "source_path": str(self.source_path.resolve()),
            "step": int(self.payload.get("step", -1)),
            "head_index": int(config.get("head_index", -1)),
            "condition_modalities": list(config.get("condition_modalities") or []),
            "target_modality": str(config.get("target_modality") or ""),
            "validation_accuracy": validation.get("accuracy"),
            "validation_balanced_accuracy": validation.get("balanced_accuracy"),
        }


@dataclass(frozen=True)
class ResolvedAudioRegionSelector:
    payload: Mapping[str, Any]
    source_path: Path
    manifest: Mapping[str, Any]

    def provenance(self) -> dict[str, Any]:
        config = dict(self.payload.get("config") or {})
        validation = dict(self.payload.get("validation") or {})
        return {
            "enabled": True,
            "source_path": str(self.source_path.resolve()),
            "step": int(self.payload.get("step", -1)),
            "head_index": int(config.get("head_index", -1)),
            "condition_modalities": list(config.get("condition_modalities") or []),
            "target_modality": str(config.get("target_modality") or ""),
            "selection_f1": validation.get("selection_f1"),
            "frame_exact_accuracy": validation.get("frame_exact_accuracy"),
        }


def load_channel_head_checkpoint(
    checkpoint: Path,
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any] | None, str]:
    """Load either a standard Channel Head safetensors file or an attached WSM bundle."""
    checkpoint = Path(checkpoint).expanduser().absolute()
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors

        state = load_safetensors(str(checkpoint), device="cpu")
        return dict(state), None, "safetensors"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != COMBINED_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Channel Head checkpoint format: {checkpoint}. "
            f"Expected safetensors or {COMBINED_CHECKPOINT_FORMAT}."
        )
    state = payload.get("channel_head_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Combined checkpoint has no channel_head_state_dict: {checkpoint}")
    refiner = payload.get("categorical_refiner")
    return dict(state), refiner if isinstance(refiner, Mapping) else None, str(payload["format"])


def _load_refiner_file(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and payload.get("format") == COMBINED_CHECKPOINT_FORMAT:
        payload = payload.get("categorical_refiner")
    if not isinstance(payload, Mapping) or "model" not in payload or "config" not in payload:
        raise ValueError(f"Not a categorical-refiner checkpoint: {path}")
    return payload


def channel_head_run_dir(checkpoint: Path) -> Path:
    """Return the training-run directory containing the checkpoint ckpt folder."""
    checkpoint = Path(checkpoint).expanduser().absolute()
    for parent in checkpoint.parents:
        if parent.name == "ckpt":
            return parent.parent
    raise ValueError(f"Checkpoint is not inside a training run ckpt directory: {checkpoint}")


def categorical_refiner_search_dirs(checkpoint: Path) -> list[Path]:
    """Return the run-level sidecar directory next to the ckpt folder."""
    return [channel_head_run_dir(checkpoint) / CATEGORICAL_REFINER_DIR]


def _canonical_condition_token(value: str) -> str:
    aliases = {
        "Reference_IMG": "0-2",
        "FirstFrame_RGB": "0-2",
        "CurrentFrame_RGB_Reference": "0-2",
        "CurrentFrame_RGB_Condition": "0-2",
    }
    return aliases.get(str(value), str(value))


def _run_compatible_refiner(
    checkpoint: Path,
    metadata: Mapping[str, Any],
) -> bool:
    """Return whether a run-level classifier fits this Head configuration.

    The region classifier is independent of the WSM generation parameters. Its
    compatibility boundary is the Head/input/target contract, rather than one
    exact checkpoint step inside the already selected training run.
    """
    run_dir = channel_head_run_dir(checkpoint)
    head_match = re.fullmatch(r"Channel_Head_(\d{6})", run_dir.parent.name)
    expected_head = int(head_match.group(1)) if head_match else None
    try:
        recorded_head = int(metadata.get("head_index"))
    except (TypeError, ValueError):
        return False
    if expected_head is None or recorded_head != expected_head:
        return False

    runtime_path = run_dir / "resolved_runtime_config.json"
    if not runtime_path.is_file():
        return True
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        entry = (runtime.get("wsg") or {}).get("entry") or {}
    except (OSError, TypeError, ValueError):
        return False
    if str(metadata.get("target_modality") or "") != str(entry.get("target") or ""):
        return False
    configured = metadata.get("condition_modalities") or []
    if isinstance(configured, str):
        configured = parse_condition_modalities(configured)
    declared = entry.get("condition") or []
    if isinstance(declared, str):
        declared = [declared]
    declared_tokens = {
        _canonical_condition_token(token)
        for token in declared
        if str(token) != "caption"
    }
    return all(
        _canonical_condition_token(token) in declared_tokens
        for token in configured
    )


def active_sidecar(refiner_dir: Path) -> tuple[Path, Mapping[str, Any] | None]:
    manifest_path = refiner_dir / CATEGORICAL_REFINER_MANIFEST
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else None
    )
    active = str((manifest or {}).get("active_checkpoint") or "best.pt")
    return refiner_dir / active, manifest


def resolve_categorical_refiner(
    checkpoint: Path,
    embedded: Mapping[str, Any] | None = None,
    selection: str | Path | None = "auto",
) -> ResolvedCategoricalRefiner | None:
    """Resolve embedded-first, then the run-level sidecar bound by manifest."""
    checkpoint = Path(checkpoint).expanduser().absolute()
    raw_selection = "auto" if selection is None else str(selection)
    if raw_selection.lower() in {"off", "none", "false", "0"}:
        return None
    if raw_selection == "auto" and embedded is not None:
        try:
            embedded_payload = enrich_refiner_payload(embedded, DEFAULT_CATEGORICAL_ROOT)
        except FileNotFoundError:
            embedded_payload = dict(embedded)
        return ResolvedCategoricalRefiner(
            payload=embedded_payload,
            source_kind="embedded",
            source_path=checkpoint,
        )

    manifest: Mapping[str, Any] | None = None
    if raw_selection == "auto":
        refiner_dir = categorical_refiner_search_dirs(checkpoint)[0]
        refiner_path, manifest = active_sidecar(refiner_dir)
        if not refiner_path.is_file() or manifest is None:
            return None
        source_kind = "run_sidecar"
    else:
        source_kind = "explicit"
        refiner_path = Path(raw_selection).expanduser().resolve()
        if refiner_path.is_dir():
            manifest_path = refiner_path / CATEGORICAL_REFINER_MANIFEST
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refiner_path = refiner_path / str(
                (manifest or {}).get("active_checkpoint") or "best.pt"
            )
        if not refiner_path.is_file():
            raise FileNotFoundError(refiner_path)
    raw_payload = _load_refiner_file(refiner_path)
    try:
        payload = enrich_refiner_payload(raw_payload, DEFAULT_CATEGORICAL_ROOT)
    except FileNotFoundError:
        payload = dict(raw_payload)
    expected_hash = str((manifest or {}).get("active_checkpoint_sha256") or "")
    if expected_hash and sha256_file(refiner_path) != expected_hash:
        raise RuntimeError(f"Categorical-refiner checksum mismatch: {refiner_path}")
    compatibility_metadata = dict(manifest or {})
    if not compatibility_metadata:
        compatibility_metadata = dict(payload.get("config") or {})
    run_compatible = _run_compatible_refiner(checkpoint, compatibility_metadata)
    expected_base_relative = str((manifest or {}).get("base_checkpoint_relative") or "")
    if expected_base_relative:
        expected_base_path = (
            channel_head_run_dir(checkpoint) / expected_base_relative
        ).resolve()
        if expected_base_path != checkpoint and not run_compatible:
            raise RuntimeError(
                "Categorical refiner manifest selects a different checkpoint: "
                f"{expected_base_path}"
            )
    expected_base_hash = str((manifest or {}).get("base_checkpoint_sha256") or "")
    if (
        expected_base_hash
        and checkpoint.suffix == ".safetensors"
        and not run_compatible
        and sha256_file(checkpoint) != expected_base_hash
    ):
        raise RuntimeError(
            "Categorical refiner is bound to a different base checkpoint: "
            f"{refiner_path}"
        )
    return ResolvedCategoricalRefiner(
        payload=payload,
        source_kind=source_kind,
        source_path=refiner_path,
        manifest=manifest,
    )


def resolve_audio_region_selector(
    checkpoint: Path,
    selection: str | Path | None = "auto",
) -> ResolvedAudioRegionSelector | None:
    """Resolve the optional Head-010 audio selector beside categorical_refiner."""
    checkpoint = Path(checkpoint).expanduser().absolute()
    raw_selection = "auto" if selection is None else str(selection)
    if raw_selection.lower() in {"off", "none", "false", "0"}:
        return None
    if raw_selection == "auto":
        selector_dir = channel_head_run_dir(checkpoint) / AUDIO_REGION_SELECTOR_DIR
    else:
        selector_dir = Path(raw_selection).expanduser().resolve()
    manifest_path = selector_dir / AUDIO_REGION_SELECTOR_MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selector_path = selector_dir / str(manifest.get("active_checkpoint") or "best.pt")
    if not selector_path.is_file():
        return None
    if str(manifest.get("format") or "") != AUDIO_REGION_SELECTOR_FORMAT:
        raise RuntimeError(f"Invalid audio-region-selector manifest: {manifest_path}")
    expected_hash = str(manifest.get("active_checkpoint_sha256") or "")
    if expected_hash and sha256_file(selector_path) != expected_hash:
        raise RuntimeError(f"Audio-region-selector checksum mismatch: {selector_path}")
    if not _run_compatible_refiner(checkpoint, manifest):
        raise RuntimeError(f"Audio region selector is incompatible with {checkpoint}")
    payload = _load_refiner_file(selector_path)
    return ResolvedAudioRegionSelector(
        payload=payload, source_path=selector_path, manifest=manifest,
    )


def categorical_refiner_revision(checkpoint: Path) -> dict[str, Any] | None:
    """Small cache identity used by the incremental evaluator."""
    checkpoint = Path(checkpoint).expanduser().absolute()
    if checkpoint.suffix != ".safetensors":
        stat = checkpoint.stat()
        return {
            "source_kind": "embedded_or_combined",
            "path": str(checkpoint),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    refiner_dir = categorical_refiner_search_dirs(checkpoint)[0]
    best_path, manifest = active_sidecar(refiner_dir)
    if not best_path.is_file() or manifest is None:
        return None
    run_compatible = _run_compatible_refiner(checkpoint, manifest or {})
    expected_base_relative = str((manifest or {}).get("base_checkpoint_relative") or "")
    if expected_base_relative:
        expected_base_path = (
            channel_head_run_dir(checkpoint) / expected_base_relative
        ).resolve()
        if expected_base_path != checkpoint and not run_compatible:
            return None
    expected_base_hash = str((manifest or {}).get("base_checkpoint_sha256") or "")
    if (
        expected_base_hash
        and not run_compatible
        and sha256_file(checkpoint) != expected_base_hash
    ):
        return None
    expected_refiner_hash = str((manifest or {}).get("active_checkpoint_sha256") or "")
    if expected_refiner_hash and sha256_file(best_path) != expected_refiner_hash:
        return None
    stat = best_path.stat()
    return {
        "source_kind": "run_sidecar",
        "path": str(best_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": str((manifest or {}).get("active_checkpoint_sha256") or ""),
        "validation_accuracy": (manifest or {}).get("validation_accuracy"),
        "validation_balanced_accuracy": (manifest or {}).get(
            "validation_balanced_accuracy"
        ),
    }


def audio_region_selector_revision(checkpoint: Path) -> dict[str, Any] | None:
    """Small cache identity for the optional Head-010 selector sidecar."""
    resolved = resolve_audio_region_selector(checkpoint)
    if resolved is None:
        return None
    stat = resolved.source_path.stat()
    return {
        "source_kind": "run_sidecar",
        "path": str(resolved.source_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": str(resolved.manifest.get("active_checkpoint_sha256") or ""),
        "selection_f1": resolved.manifest.get("selection_f1"),
        "frame_exact_accuracy": resolved.manifest.get("frame_exact_accuracy"),
    }


def export_sidecar(args: argparse.Namespace) -> None:
    """Install a portable classifier beside ckpt in the selected training run."""
    base_checkpoint = Path(args.base_checkpoint).resolve()
    source = Path(args.refiner_checkpoint).resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = enrich_refiner_payload(
        _load_refiner_file(source), Path(args.categorical_root)
    )
    config = dict(payload["config"])
    if getattr(args, "head_index_override", None) is not None:
        config["head_index"] = int(args.head_index_override)
        payload = {**dict(payload), "config": config}
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir else channel_head_run_dir(base_checkpoint) / CATEGORICAL_REFINER_DIR
    )
    best_path = destination / "best.pt"
    manifest_path = destination / CATEGORICAL_REFINER_MANIFEST
    if (best_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Refusing to replace categorical-refiner sidecar: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, best_path)
    atomic_json_dump(config, destination / "config.json")
    validation = dict(payload.get("validation") or {})
    validation_record = {"step": int(payload.get("step", -1)), **validation}
    atomic_json_dump(validation_record, destination / "best_metrics.json")
    source_metrics = source.parent / "metrics.csv"
    destination_metrics = destination / "metrics.csv"
    if (
        source_metrics.is_file()
        and source_metrics.resolve() != destination_metrics.resolve()
    ):
        shutil.copy2(source_metrics, destination_metrics)
    plot_training_curves(destination_metrics, destination)
    manifest = {
        "format": CATEGORICAL_REFINER_FORMAT,
        "schema_version": 1,
        "enabled_by_default": True,
        "active_checkpoint": "best.pt",
        "active_checkpoint_sha256": sha256_file(best_path),
        "base_checkpoint_filename": base_checkpoint.name,
        "base_checkpoint_relative": str(
            base_checkpoint.relative_to(channel_head_run_dir(base_checkpoint))
        ),
        "base_checkpoint_step": int(
            (BASE_STEP_PATTERN.search(base_checkpoint.parent.name) or [None, -1])[1]
        ),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "head_index": int(config.get("head_index", -1)),
        "condition_modalities": list(config.get("condition_modalities") or []),
        "target_modality": str(config.get("target_modality") or ""),
        "refiner_step": int(payload.get("step", -1)),
        "input_representation": str(config.get("input_representation") or ""),
        "validation_accuracy": validation.get("accuracy"),
        "validation_balanced_accuracy": validation.get("balanced_accuracy"),
        "source_refiner_checkpoint": str(source),
    }
    atomic_json_dump(manifest, manifest_path)
    print(f"[categorical-refiner] installed sidecar -> {destination}", flush=True)


def export_audio_selector(args: argparse.Namespace) -> None:
    """Install the independently trained audio selector beside a run ckpt."""
    base_checkpoint = Path(args.base_checkpoint).resolve()
    source = Path(args.selector_checkpoint).resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    payload = dict(_load_refiner_file(source))
    config = dict(payload.get("config") or {})
    if str(config.get("mode") or "") != "audio_region_selector":
        raise ValueError(f"Not an audio region selector: {source}")
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir else channel_head_run_dir(base_checkpoint) / AUDIO_REGION_SELECTOR_DIR
    )
    best_path = destination / "best.pt"
    manifest_path = destination / AUDIO_REGION_SELECTOR_MANIFEST
    if (best_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Refusing to replace audio selector: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, best_path)
    atomic_json_dump(config, destination / "config.json")
    validation = dict(payload.get("validation") or {})
    atomic_json_dump(
        {"step": int(payload.get("step", -1)), **validation},
        destination / "best_metrics.json",
    )
    source_metrics = source.parent / "metrics.csv"
    destination_metrics = destination / "metrics.csv"
    if source_metrics.is_file() and source_metrics.resolve() != destination_metrics.resolve():
        shutil.copy2(source_metrics, destination_metrics)
    plot_training_curves(destination_metrics, destination)
    manifest = {
        "format": AUDIO_REGION_SELECTOR_FORMAT,
        "schema_version": 1,
        "enabled_by_default": True,
        "active_checkpoint": "best.pt",
        "active_checkpoint_sha256": sha256_file(best_path),
        "base_checkpoint_relative": str(
            base_checkpoint.relative_to(channel_head_run_dir(base_checkpoint))
        ),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "head_index": int(config.get("head_index", -1)),
        "condition_modalities": list(config.get("condition_modalities") or []),
        "target_modality": str(config.get("target_modality") or ""),
        "selector_step": int(payload.get("step", -1)),
        "input_representation": str(config.get("input_representation") or ""),
        "selection_f1": validation.get("selection_f1"),
        "frame_exact_accuracy": validation.get("frame_exact_accuracy"),
        "source_selector_checkpoint": str(source),
    }
    atomic_json_dump(manifest, manifest_path)
    print(f"[audio-region-selector] installed sidecar -> {destination}", flush=True)


def attach_classifier(args: argparse.Namespace) -> None:
    """Create a self-contained Head copy with an attached categorical refiner."""
    base_checkpoint = Path(args.base_checkpoint).resolve()
    if base_checkpoint.name != "pytorch_lora_weights.safetensors":
        raise ValueError(
            "The base checkpoint must be a pytorch_lora_weights.safetensors file"
        )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if args.refiner_checkpoint:
        refiner_checkpoint = Path(args.refiner_checkpoint).resolve()
        refiner = _load_refiner_file(refiner_checkpoint)
    else:
        resolved = resolve_categorical_refiner(base_checkpoint, selection="auto")
        if resolved is None:
            raise FileNotFoundError(
                f"No {CATEGORICAL_REFINER_DIR}/best.pt next to {base_checkpoint}"
            )
        refiner_checkpoint = resolved.source_path
        refiner = resolved.payload
    refiner = enrich_refiner_payload(refiner, Path(args.categorical_root))
    refiner_config = dict(refiner.get("config") or {})
    base_match = BASE_STEP_PATTERN.search(base_checkpoint.parent.name)
    base_step = int(base_match.group(1)) if base_match else -1
    refiner_step = int(refiner.get("step", -1))
    if args.output is None:
        base_marker = f"{base_step:07d}" if base_step >= 0 else base_checkpoint.parent.name
        refiner_marker = f"{refiner_step:07d}" if refiner_step >= 0 else "unknown"
        output = base_checkpoint.parent / (
            f"step_{base_marker}_with_categorical_refiner_step_{refiner_marker}.pt"
        )
    else:
        output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing attached checkpoint: {output}")
    state, _embedded, _format = load_channel_head_checkpoint(base_checkpoint)
    bundle = {
        "format": COMBINED_CHECKPOINT_FORMAT,
        "base_checkpoint": {
            "source": str(base_checkpoint),
            "step": base_step,
            "sha256": sha256_file(base_checkpoint),
        },
        "channel_head_state_dict": {
            key: value.detach().cpu() for key, value in state.items()
        },
        "categorical_refiner": refiner,
        "categorical_refiner_config": refiner_config,
        "categorical_refiner_step": refiner_step,
        "categorical_refiner_source": str(refiner_checkpoint),
    }
    atomic_torch_save(bundle, output)
    print(
        f"[categorical-refiner] attached Head copy: base_step={base_step} "
        f"refiner_step={refiner_step} -> {output}",
        flush=True,
    )


def image_to_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1)


def _otsu_foreground_mask(prediction_rgb: np.ndarray) -> np.ndarray:
    """Separate pseudo-RGB foreground when decoded background is not pure black."""
    rgb = np.asarray(prediction_rgb, dtype=np.uint8)
    intensity = rgb.max(axis=2)
    histogram = np.bincount(intensity.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    weights = np.cumsum(probabilities)
    means = np.cumsum(probabilities * np.arange(256, dtype=np.float64))
    total_mean = float(means[-1])
    denominator = weights * (1.0 - weights)
    score = np.zeros(256, dtype=np.float64)
    valid = denominator > 1e-12
    score[valid] = (
        (total_mean * weights[valid] - means[valid]) ** 2
        / denominator[valid]
    )
    threshold = int(np.argmax(score))
    return intensity > threshold


def _palette_partitioned_components(
    foreground: np.ndarray,
    rgb: np.ndarray,
    palette_rgb: Sequence[Sequence[int]],
    minimum_pixels: int,
    minimum_partition_fraction: float = 0.08,
    maximum_partitions: int = 8,
) -> list[np.ndarray]:
    """Split foreground components only at stable generated color boundaries.

    The categorical classifier consumes one spatial proposal at a time and
    predicts one class for that proposal. Collapsing all non-background
    pixels before connected-component analysis therefore destroys valid
    boundaries whenever two semantic masks touch. Conversely, directly
    taking connected components of every nearest-palette label fragments VAE
    color noise into many tiny proposals.

    Keep the binary foreground as the geometry envelope, retain only color
    islands that occupy a meaningful fraction of their parent component, and
    assign remaining noisy pixels to the nearest retained island. The returned
    masks are disjoint and preserve the complete foreground envelope.
    """
    palette = np.asarray(palette_rgb, dtype=np.float32)
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        return []

    smoothed = ndimage.gaussian_filter(
        np.asarray(rgb, dtype=np.float32), sigma=(1.2, 1.2, 0.0)
    )
    flat = smoothed.reshape(-1, 3)
    color_distances = (
        np.sum(flat * flat, axis=1, keepdims=True)
        + np.sum(palette * palette, axis=1)[None, :]
        - 2.0 * flat @ palette.T
    )
    nearest_palette = np.argmin(color_distances, axis=1).reshape(foreground.shape)

    parents, parent_count = ndimage.label(
        foreground, structure=np.ones((3, 3), dtype=bool)
    )
    output: list[np.ndarray] = []
    for parent_id in range(1, parent_count + 1):
        parent = parents == parent_id
        parent_pixels = int(parent.sum())
        if parent_pixels < int(minimum_pixels):
            continue

        stable_minimum = max(
            int(minimum_pixels),
            int(math.ceil(parent_pixels * float(minimum_partition_fraction))),
        )
        seeds: list[tuple[int, np.ndarray]] = []
        for palette_index in np.unique(nearest_palette[parent]).tolist():
            islands, island_count = ndimage.label(
                parent & (nearest_palette == int(palette_index)),
                structure=np.ones((3, 3), dtype=bool),
            )
            for island_id in range(1, island_count + 1):
                island = islands == island_id
                pixels = int(island.sum())
                if pixels >= stable_minimum:
                    seeds.append((pixels, island))

        seeds.sort(key=lambda item: item[0], reverse=True)
        seeds = seeds[: max(2, int(maximum_partitions))]
        if len(seeds) < 2:
            output.append(parent)
            continue

        seed_labels = np.zeros(parent.shape, dtype=np.int16)
        for seed_id, (_pixels, seed) in enumerate(seeds, start=1):
            seed_labels[seed] = seed_id
        nearest_seed_indices = ndimage.distance_transform_edt(
            seed_labels == 0, return_distances=False, return_indices=True
        )
        assigned = seed_labels[tuple(nearest_seed_indices)]
        output.extend(
            parent & (assigned == seed_id)
            for seed_id in range(1, len(seeds) + 1)
        )
    return output


def _component_masks(
    prediction_rgb: np.ndarray,
    minimum_pixels: int,
    background_rgb: Sequence[int] | None = None,
    palette_rgb: Sequence[Sequence[int]] | None = None,
    preserve_palette_boundaries: bool = False,
) -> list[np.ndarray]:
    """Recover generated geometry without trusting its final category labels."""
    rgb = np.asarray(prediction_rgb, dtype=np.float32)
    if background_rgb is not None:
        background = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
        foreground = np.linalg.norm(rgb - background, axis=2) > 8.0
        # Scalar categorical WSG channels are decoded through a pseudo-RGB
        # visualization. Their invalid background can therefore be dark purple
        # rather than exact black. If the declared background marks virtually
        # the entire canvas as foreground, recover the high-intensity semantic
        # lobe instead of treating the full frame as one component.
        if float(foreground.mean()) > 0.98:
            recovered = _otsu_foreground_mask(prediction_rgb)
            coverage = float(recovered.mean())
            if 0.001 <= coverage <= 0.95:
                foreground = recovered
        if preserve_palette_boundaries and palette_rgb is not None:
            output = _palette_partitioned_components(
                foreground, rgb, palette_rgb, minimum_pixels
            )
            if output:
                output.sort(key=lambda mask: int(mask.sum()), reverse=True)
                return output
        label_maps = [foreground]
    elif palette_rgb is not None and len(palette_rgb):
        palette = np.asarray(palette_rgb, dtype=np.float32)
        flat = rgb.reshape(-1, 3)
        nearest = np.argmin(
            np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2), axis=1
        ).reshape(rgb.shape[:2])
        label_maps = [nearest == index for index in range(len(palette))]
    else:
        label_maps = [np.ones(rgb.shape[:2], dtype=bool)]

    output: list[np.ndarray] = []
    for foreground in label_maps:
        components, count = ndimage.label(
            foreground, structure=np.ones((3, 3), dtype=bool)
        )
        for component_id in range(1, count + 1):
            mask = components == component_id
            if int(mask.sum()) >= int(minimum_pixels):
                output.append(mask)
    output.sort(key=lambda mask: int(mask.sum()), reverse=True)
    return output


def _confirm_palette_partitions(
    parent_components: Sequence[np.ndarray],
    partition_components: Sequence[np.ndarray],
    partition_probabilities: np.ndarray,
    minimum_class_confidence: float = 0.75,
) -> list[np.ndarray]:
    """Keep a color split only when the classifier confirms distinct classes.

    Generated category colors are useful boundary proposals, but they are not
    reliable labels. A parent foreground component is split only when at least
    two proposed regions independently predict different classes with adequate
    confidence. Proposals assigned to the same class are merged, and uncertain
    pixels are absorbed by the nearest confirmed class region.
    """
    parents = [np.asarray(mask, dtype=bool) for mask in parent_components]
    partitions = [np.asarray(mask, dtype=bool) for mask in partition_components]
    probabilities = np.asarray(partition_probabilities, dtype=np.float32)
    if not parents or not partitions:
        return parents
    if probabilities.ndim != 2 or probabilities.shape[0] != len(partitions):
        raise ValueError(
            "Partition probabilities must have shape "
            f"({len(partitions)}, classes), got {probabilities.shape}"
        )

    parent_labels = np.zeros(parents[0].shape, dtype=np.int32)
    for parent_index, parent in enumerate(parents, start=1):
        parent_labels[parent] = parent_index
    grouped: list[list[int]] = [[] for _ in parents]
    for partition_index, partition in enumerate(partitions):
        labels, counts = np.unique(parent_labels[partition], return_counts=True)
        valid = labels > 0
        if not np.any(valid):
            continue
        labels = labels[valid]
        counts = counts[valid]
        grouped[int(labels[int(np.argmax(counts))]) - 1].append(partition_index)

    output: list[np.ndarray] = []
    for parent_index, parent in enumerate(parents):
        indices = grouped[parent_index]
        if len(indices) < 2:
            output.append(parent)
            continue

        local_probabilities = probabilities[indices]
        predicted_classes = local_probabilities.argmax(axis=1)
        confidences = local_probabilities.max(axis=1)
        confirmed_classes = sorted(
            {
                int(class_index)
                for class_index, confidence in zip(predicted_classes, confidences)
                if float(confidence) >= float(minimum_class_confidence)
            }
        )
        if len(confirmed_classes) < 2:
            output.append(parent)
            continue

        seed_labels = np.zeros(parent.shape, dtype=np.int16)
        for seed_id, class_index in enumerate(confirmed_classes, start=1):
            class_seed = np.zeros(parent.shape, dtype=bool)
            for local_index, partition_index in enumerate(indices):
                if (
                    int(predicted_classes[local_index]) == class_index
                    and float(confidences[local_index]) >= float(minimum_class_confidence)
                ):
                    class_seed |= partitions[partition_index]
            seed_labels[class_seed] = seed_id

        nearest_seed_indices = ndimage.distance_transform_edt(
            seed_labels == 0, return_distances=False, return_indices=True
        )
        assigned = seed_labels[tuple(nearest_seed_indices)]
        output.extend(
            parent & (assigned == seed_id)
            for seed_id in range(1, len(confirmed_classes) + 1)
        )

    output.sort(key=lambda mask: int(mask.sum()), reverse=True)
    return output


def _prepare_condition_latents(
    condition_latents: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    values = (
        list(condition_latents)
        if isinstance(condition_latents, (list, tuple)) else [condition_latents]
    )
    prepared = []
    for latent in values:
        if latent is None:
            continue
        if latent.ndim == 5:
            latent = latent[:, :, 0]
        elif latent.ndim == 3:
            latent = latent[None]
        if latent.ndim != 4:
            raise ValueError(f"Expected BCHW condition latent, got {tuple(latent.shape)}")
        prepared.append(latent)
    if not prepared:
        raise ValueError("Categorical refiner received no condition latent")
    spatial_shapes = {tuple(item.shape[-2:]) for item in prepared}
    if len(spatial_shapes) != 1:
        raise ValueError(f"Categorical condition latent shapes differ: {spatial_shapes}")
    return torch.cat(prepared, dim=1)


class AudioRegionSelectorRuntime:
    """Frozen Head-010 selector that decides which candidate regions sound."""

    def __init__(self, resolved: ResolvedAudioRegionSelector, device: torch.device | str):
        self.resolved = resolved
        self.payload = dict(resolved.payload)
        self.config = dict(self.payload.get("config") or {})
        self.device = torch.device(device)
        if str(self.config.get("mode") or "") != "audio_region_selector":
            raise RuntimeError("Invalid audio region selector mode")
        self.model = AudioRegionSelector(
            latent_channels=int(self.config.get("latent_channels", 32)),
            audio_dim=int(self.config.get("audio_dim", 768)),
            embedding_dim=int(self.config.get("embedding_dim", 512)),
        )
        load_compact_classifier_state(self.model, self.payload["model"])
        self.model.to(self.device).eval()

    @property
    def condition_modalities(self) -> list[str]:
        values = self.config.get("generation_condition_modalities") or ["0-2"]
        return parse_condition_modalities(values)

    @torch.inference_mode()
    def select(
        self, candidate_latents: torch.Tensor, audio_feature: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = torch.as_tensor(audio_feature, dtype=torch.float32, device=self.device)
        if audio.ndim == 1:
            audio = audio[None]
        if audio.shape[0] != 1:
            raise ValueError("Audio region selection expects one sample at a time")
        logits = self.model(candidate_latents.to(self.device)[None], audio)
        probabilities = logits.sigmoid()[0]
        threshold = float(self.config.get("selection_threshold", 0.5))
        return probabilities, probabilities >= threshold


class CategoricalRefinerRuntime:
    """Frozen RGB category head, optionally paired with an audio selector."""

    def __init__(self, resolved: ResolvedCategoricalRefiner, device: torch.device | str):
        self.resolved = resolved
        self.payload = dict(resolved.payload)
        self.config = dict(self.payload.get("config") or {})
        self.device = torch.device(device)
        self.category_values = [int(value) for value in self.config["category_values"]]
        palette = self.config.get("category_prototypes_rgb")
        if palette is None:
            target = str(self.config.get("target_modality") or "")
            if target == "195":
                palette = [AVIS_CATEGORY_COLORS[value] for value in self.category_values]
            else:
                raise RuntimeError("Categorical refiner has no portable category palette")
        self.palette_rgb = np.asarray(palette, dtype=np.uint8)
        background = self.config.get("background_rgb")
        self.background_rgb = None if background is None else np.asarray(background, dtype=np.uint8)
        representation = str(self.config.get("input_representation") or "")
        if representation != "masked_structured_condition_latent":
            raise RuntimeError(f"Unsupported categorical representation {representation!r}")
        self.model = MaskedLatentClassifier(
            latent_channels=int(self.config.get("latent_channels", 32)),
            num_classes=len(self.category_values),
            embedding_dim=int(self.config.get("embedding_dim", 512)),
        )
        load_compact_classifier_state(self.model, self.payload["model"])
        self.model.to(self.device).eval()

    @property
    def condition_modalities(self) -> list[str]:
        values = self.config.get("condition_modalities") or self.config.get("condition_modality") or []
        return parse_condition_modalities(values)

    @torch.inference_mode()
    def refine(
        self,
        prediction: Image.Image,
        condition_latents: torch.Tensor | Sequence[torch.Tensor],
        minimum_component_pixels: int = 32,
        audio_selector: AudioRegionSelectorRuntime | None = None,
        audio_feature: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        prediction_rgb = np.asarray(prediction.convert("RGB"), dtype=np.uint8)
        parent_components = _component_masks(
            prediction_rgb,
            minimum_component_pixels,
            self.background_rgb,
            self.palette_rgb,
        )
        # Palette-aware region discovery is the default for every categorical
        # target. Single-region outputs naturally fall back to the complete
        # foreground component: a split is accepted only when the classifier
        # confidently confirms at least two distinct classes. This keeps the
        # deployment rule generic while fixing touching multi-class masks such
        # as semantic target 195.
        partition_components = _component_masks(
            prediction_rgb,
            minimum_component_pixels,
            self.background_rgb,
            self.palette_rgb,
            preserve_palette_boundaries=True,
        )
        components = parent_components
        condition = _prepare_condition_latents(condition_latents).to(
            self.device, dtype=torch.float32,
        )
        expected_channels = int(self.config.get("latent_channels", condition.shape[1]))
        if condition.shape[1] != expected_channels:
            raise RuntimeError(
                f"Categorical refiner expected {expected_channels} channels, received {condition.shape[1]}"
            )
        if condition.shape[0] != 1:
            raise ValueError("Categorical refinement expects one sample at a time")
        if self.background_rgb is None:
            refined_rgb = prediction_rgb.copy()
        else:
            refined_rgb = np.broadcast_to(
                self.background_rgb.reshape(1, 1, 3), prediction_rgb.shape,
            ).copy()
        category_map = np.full(prediction_rgb.shape[:2], -1, dtype=np.float32)
        regions: list[dict[str, Any]] = []
        if not components:
            return {
                "image": Image.fromarray(refined_rgb),
                "category_map": category_map,
                "regions": regions,
                "provenance": self.resolved.provenance(),
                "audio_region_selector": None,
            }
        def encode_components(component_masks: Sequence[np.ndarray]) -> torch.Tensor:
            candidate_latents = []
            for component in component_masks:
                mask = torch.from_numpy(component.astype(np.float32))[None, None].to(
                    self.device
                )
                candidate_latents.append(masked_condition_latents(condition, mask)[0])
            return torch.stack(candidate_latents, dim=0)

        if partition_components is not parent_components:
            partition_tensor = encode_components(partition_components)
            partition_probabilities = self.model(partition_tensor).softmax(dim=-1)
            components = _confirm_palette_partitions(
                parent_components,
                partition_components,
                partition_probabilities.detach().cpu().numpy(),
            )

        candidate_tensor = encode_components(components)

        # These two predictions consume the same confirmed candidate set and
        # have no dependency on one another: RGB assigns class; audio selects regions.
        category_probabilities = self.model(candidate_tensor).softmax(dim=-1)
        if audio_selector is None:
            selection_probabilities = torch.ones(len(components), device=self.device)
            selected = torch.ones(len(components), dtype=torch.bool, device=self.device)
            selector_provenance = None
        else:
            if audio_feature is None:
                raise RuntimeError("Head 010 requires frame audio for mask selection")
            selection_probabilities, selected = audio_selector.select(
                candidate_tensor, audio_feature,
            )
            selector_provenance = audio_selector.resolved.provenance()

        for region_index, component in enumerate(components):
            class_index = int(category_probabilities[region_index].argmax().item())
            category_value = self.category_values[class_index]
            is_selected = bool(selected[region_index].item())
            if is_selected:
                refined_rgb[component] = self.palette_rgb[class_index]
                category_map[component] = float(category_value)
            regions.append({
                "region": region_index,
                "pixels": int(component.sum()),
                "selected": is_selected,
                "selection_confidence": float(selection_probabilities[region_index].item()),
                "predicted_class": category_value,
                "class_confidence": float(category_probabilities[region_index, class_index].item()),
            })
        return {
            "image": Image.fromarray(refined_rgb),
            "category_map": category_map,
            "regions": regions,
            "provenance": self.resolved.provenance(),
            "audio_region_selector": selector_provenance,
        }

    def to_wsg_array(self, category_map: np.ndarray) -> np.ndarray:
        target = str(self.config.get("target_modality") or "")
        if not target.isdigit():
            raise ValueError(f"Categorical target is not a scalar WSG channel: {target}")
        channel = int(target)
        height, width = category_map.shape
        wsg = np.full((height, width, 256), -1.0, dtype=np.float32)
        wsg[..., channel] = np.asarray(category_map, dtype=np.float32)
        return wsg


def _save_comparison(
    condition: Path, baseline: Path, refined: Path, ground_truth: Path, output: Path
) -> None:
    paths = (condition, baseline, refined, ground_truth)
    titles = ("Condition", "Baseline inference", "Refined categories", "GT latent decode")
    images = [Image.open(path).convert("RGB") for path in paths]
    height = max(image.height for image in images)
    width = max(image.width for image in images)
    canvas = Image.new("RGB", (width * 4, height + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (image, title) in enumerate(zip(images, titles)):
        image = image.resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(image, (column * width, 30))
        draw.text((column * width + 6, 8), title, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _aggregate_sample_metrics(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    categorical = [sample[key]["categorical"] for sample in samples]
    valid = sum(int(item.get("valid_pixels", 0)) for item in categorical)
    strict = sum(int(item.get("strict_correct_pixels", 0)) for item in categorical)
    tolerant = sum(int(item.get("tolerant_correct_pixels", 0)) for item in categorical)
    return {
        "samples": len(samples),
        "decoded_mse_mean": float(np.mean([sample[key]["decoded_mse"] for sample in samples])),
        "strict_pixel_accuracy_micro": strict / max(valid, 1),
        "tolerant_pixel_accuracy_micro": tolerant / max(valid, 1),
        "balanced_strict_accuracy_macro": float(np.mean([
            item.get("balanced_strict_accuracy", 0.0) for item in categorical
        ])),
        "dominant_strict_accuracy": float(np.mean([
            item.get("dominant_strict_accuracy", 0.0) for item in categorical
        ])),
        "foreground_iou_mean": float(np.mean([
            item.get("foreground_iou", 0.0) for item in categorical
        ])),
    }


@torch.inference_mode()
def refine_benchmark(args: argparse.Namespace) -> None:
    from src.evaluation.internal.channel_head_evaluator import (
        palette_categorical_metrics,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configured_conditions = (
        config.get("condition_modalities") or config.get("condition_modality")
    )
    condition_modalities = parse_condition_modalities(configured_conditions)
    condition_modality = ",".join(condition_modalities)
    target_modality = str(config["target_modality"])
    category_values = [int(value) for value in config["category_values"]]
    device = torch.device(args.device)
    input_representation = str(
        config.get("input_representation", "frozen_flux2_vae_decoded_condition_mask_image")
    )
    if input_representation == "masked_structured_condition_latent":
        model = MaskedLatentClassifier(
            latent_channels=int(config.get("latent_channels", 32)),
            num_classes=len(category_values),
            embedding_dim=int(config.get("embedding_dim", 512)),
        )
        vae = None
        vae_dtype = None
    else:
        if len(condition_modalities) != 1:
            raise ValueError(
                "Decoded-image categorical-refiner benchmarks support only one "
                "condition; multi-condition refiners must use masked structured latents"
            )
        model = RegionPrototypeClassifier(
            num_classes=len(category_values),
            embedding_dim=int(config.get("embedding_dim", 512)),
        )
        model_path = Path(args.model_path or config.get("model_path", DEFAULT_MODEL_PATH))
        vae, vae_dtype = load_flux_vae(model_path, device)
    load_compact_classifier_state(model, checkpoint["model"])
    model.to(device).eval()
    cache_root = Path(args.cache_root)
    condition_shards_by_modality = {
        modality: load_modality_shards(cache_root, modality)
        for modality in condition_modalities
    }
    condition_indices = {
        modality: build_sample_index(shards)
        for modality, shards in condition_shards_by_modality.items()
    }
    categorical_paths = sorted(
        (cache_root.parent / "categorical_targets" / target_modality).glob("shard_*.pt")
    )
    categorical_shards = [
        (path, torch.load(path, map_location="cpu", mmap=True, weights_only=False))
        for path in categorical_paths
    ]
    categorical_index = build_sample_index(categorical_shards)
    benchmark_dir = Path(args.benchmark_dir)
    sample_dirs = sorted(path for path in benchmark_dir.glob("sample_*") if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No benchmark samples below {benchmark_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        sample_id = str(metadata["sample_id"])
        missing_modalities = [
            modality
            for modality in condition_modalities
            if sample_id not in condition_indices[modality]
        ]
        if missing_modalities:
            print(
                f"[categorical-refiner] missing condition latent(s) "
                f"{missing_modalities}: {sample_id}",
                flush=True,
            )
            continue
        latents = []
        for modality in condition_modalities:
            shard_index, row_index = condition_indices[modality][sample_id]
            payload = condition_shards_by_modality[modality][shard_index][1]
            latent = payload["latents"][row_index]
            if latent.ndim == 4:
                latent = latent[:, 0]
            latents.append(latent.float())
        spatial_shapes = {tuple(latent.shape[-2:]) for latent in latents}
        if len(spatial_shapes) != 1:
            raise RuntimeError(
                f"Condition latent shapes differ for {condition_modalities}: "
                f"{sorted(spatial_shapes)}"
            )
        latent = torch.cat(latents, dim=0)
        if vae is not None:
            decoded = decode_condition_latents(
                vae, latent[None].float(), device, vae_dtype
            )
        unrefined_path = sample_dir / "inference_unrefined.png"
        baseline_path = (
            unrefined_path if unrefined_path.is_file() else sample_dir / "inference.png"
        )
        baseline_rgb = np.asarray(Image.open(baseline_path).convert("RGB"), dtype=np.uint8)
        palette_rgb = [
            AVIS_CATEGORY_COLORS[key] for key in sorted(AVIS_CATEGORY_COLORS)
        ]
        parent_components = _component_masks(
            baseline_rgb,
            args.minimum_component_pixels,
            background_rgb=(0, 0, 0),
            palette_rgb=palette_rgb,
        )
        partition_components = _component_masks(
            baseline_rgb,
            args.minimum_component_pixels,
            background_rgb=(0, 0, 0),
            palette_rgb=palette_rgb,
            preserve_palette_boundaries=True,
        )

        def classify_component(component: np.ndarray) -> torch.Tensor:
            mask = torch.from_numpy(component.astype(np.float32))[None, None].to(device)
            if vae is None:
                return model(
                    masked_condition_latents(latent[None].float().to(device), mask)
                )
            return model(decoded_modality_masks(decoded, mask))

        if partition_components and not args.binary_regions_only:
            partition_probabilities = torch.cat(
                [classify_component(component).softmax(dim=1)
                 for component in partition_components],
                dim=0,
            )
            components = _confirm_palette_partitions(
                parent_components,
                partition_components,
                partition_probabilities.detach().cpu().numpy(),
            )
        else:
            components = parent_components
        refined_rgb = np.zeros_like(baseline_rgb)
        region_predictions = []
        for component_index, component in enumerate(components):
            logits = classify_component(component)
            class_id = category_values[int(logits.argmax(dim=1).item())]
            refined_rgb[component] = np.asarray(AVIS_CATEGORY_COLORS[class_id], dtype=np.uint8)
            region_predictions.append(
                {
                    "region": component_index,
                    "pixels": int(component.sum()),
                    "predicted_class": class_id,
                    "confidence": float(logits.softmax(dim=1).max().item()),
                }
            )
        sample_output = output_dir / sample_dir.name
        sample_output.mkdir(parents=True, exist_ok=True)
        refined_path = sample_output / "inference_refined.png"
        Image.fromarray(refined_rgb).save(refined_path)
        if sample_id not in categorical_index:
            print(f"[categorical-refiner] exact category set unavailable: {sample_id}", flush=True)
            continue
        refined_tensor = image_to_tensor(refined_path)
        baseline_tensor = image_to_tensor(baseline_path)
        gt_tensor = image_to_tensor(sample_dir / "gt.png")
        category_shard, category_row = categorical_index[sample_id]
        sidecar = categorical_shards[category_shard][1]
        spatial_index = sidecar["category_spatial_index_map"][category_row].long()
        value_lookup = torch.tensor([-1] + category_values, dtype=torch.long)
        gt_ids = value_lookup[spatial_index]
        gt_ids = F.interpolate(
            gt_ids[None, None].float(),
            size=gt_tensor.shape[-2:],
            mode="nearest",
        )[0, 0].numpy()
        refined_metrics = {
            "decoded_mse": float(F.mse_loss(refined_tensor, gt_tensor).item()),
            "categorical": palette_categorical_metrics(
                refined_tensor, gt_ids, AVIS_CATEGORY_COLORS, args.categorical_tolerance
            ),
        }
        baseline_metrics = {
            "decoded_mse": float(F.mse_loss(baseline_tensor, gt_tensor).item()),
            "categorical": palette_categorical_metrics(
                baseline_tensor, gt_ids, AVIS_CATEGORY_COLORS, args.categorical_tolerance
            ),
        }
        sample_payload = {
            "sample": sample_dir.name,
            "sample_id": sample_id,
            "regions": region_predictions,
            "baseline": baseline_metrics,
            "refined": refined_metrics,
        }
        atomic_json_dump(sample_payload, sample_output / "metadata_refined.json")
        _save_comparison(
            sample_dir / "condition.png", baseline_path, refined_path,
            sample_dir / "gt.png", sample_output / "summary_refined.png",
        )
        samples.append(sample_payload)
        print(
            f"[categorical-refiner] {sample_dir.name}: regions={len(region_predictions)} "
            f"strict={refined_metrics['categorical'].get('strict_pixel_accuracy', 0.0):.4f}",
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "benchmark_dir": str(benchmark_dir.resolve()),
        "condition_modality": condition_modality,
        "condition_modalities": condition_modalities,
        "baseline": _aggregate_sample_metrics(samples, "baseline"),
        "refined": _aggregate_sample_metrics(samples, "refined"),
        "samples": samples,
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps({key: summary[key] for key in ("baseline", "refined")}, indent=2), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one condition-specific refiner")
    train.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    train.add_argument("--categorical-root", type=Path, default=DEFAULT_CATEGORICAL_ROOT)
    train.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    train.add_argument(
        "--condition-modality",
        required=True,
        help="One modality or an ordered comma-separated list, e.g. 36-38,0-2.",
    )
    train.add_argument("--target-modality", default="195")
    train.add_argument("--head-index", type=int, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda:6")
    train.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument(
        "--input-representation",
        choices=("masked-latent", "decoded-mask-features"),
        default="masked-latent",
    )
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--eval-batch-size", type=int, default=256)
    train.add_argument("--feature-cache-batch-size", type=int, default=64)
    train.add_argument("--max-steps", type=int, default=3000)
    train.add_argument(
        "--keep-step-checkpoints", action="store_true",
        help="Retain every evaluation snapshot; disabled by default.",
    )
    train.add_argument(
        "--save-alternative-checkpoints", action="store_true",
        help="Also save best_accuracy.pt, best_val_loss.pt, and last.pt; default stores their metrics but only keeps best.pt weights.",
    )
    train.add_argument("--eval-every", type=int, default=50)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--dropout", type=float, default=0.20)
    train.add_argument("--label-smoothing", type=float, default=0.05)
    train.add_argument("--latent-noise-std", type=float, default=0.0)
    train.add_argument("--minimum-region-cells", type=int, default=6)
    train.add_argument("--seed", type=int, default=20260814)
    train.add_argument(
        "--smoke-test", action="store_true",
        help="Run two train/validation/checkpoint steps on bounded region subsets.",
    )
    train.add_argument(
        "--smoke-regions", type=int, default=64,
        help="Maximum train and validation regions used by --smoke-test.",
    )

    selector = subparsers.add_parser(
        "train-audio-selector",
        help="Train the Head-010 audio-conditioned multi-region selector",
    )
    selector.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    selector.add_argument("--categorical-root", type=Path, default=DEFAULT_CATEGORICAL_ROOT)
    selector.add_argument("--audio-train", type=Path, required=True)
    selector.add_argument("--audio-val", type=Path, required=True)
    selector.add_argument("--target-modality", default="195")
    selector.add_argument("--head-index", type=int, default=10)
    selector.add_argument("--output-dir", type=Path, required=True)
    selector.add_argument("--device", default="cuda:0")
    selector.add_argument("--batch-size", type=int, default=8)
    selector.add_argument("--eval-batch-size", type=int, default=16)
    selector.add_argument("--max-steps", type=int, default=3000)
    selector.add_argument("--eval-every", type=int, default=50)
    selector.add_argument("--learning-rate", type=float, default=3e-4)
    selector.add_argument("--dropout", type=float, default=0.20)
    selector.add_argument("--ranking-loss-weight", type=float, default=0.25)
    selector.add_argument("--minimum-region-cells", type=int, default=4)
    selector.add_argument("--maximum-candidates", type=int, default=24)
    selector.add_argument("--seed", type=int, default=20260823)
    selector.add_argument("--smoke-test", action="store_true")
    selector.add_argument("--smoke-frames", type=int, default=64)

    refine = subparsers.add_parser("refine-benchmark", help="Recolor saved true-inference samples")
    refine.add_argument("--checkpoint", type=Path, required=True)
    refine.add_argument("--benchmark-dir", type=Path, required=True)
    refine.add_argument("--output-dir", type=Path, required=True)
    refine.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    refine.add_argument("--device", default="cuda:6")
    refine.add_argument("--model-path", type=Path)
    refine.add_argument("--minimum-component-pixels", type=int, default=32)
    refine.add_argument("--categorical-tolerance", type=float, default=0.20)
    refine.add_argument(
        "--binary-regions-only",
        action="store_true",
        help="Ablation: use foreground connected components without semantic splitting",
    )

    attach = subparsers.add_parser(
        "attach", help="Copy a trained Head into a new .pt and attach a refiner"
    )
    attach.add_argument("--base-checkpoint", type=Path, required=True)
    attach.add_argument("--categorical-root", type=Path, default=DEFAULT_CATEGORICAL_ROOT)
    attach.add_argument("--refiner-checkpoint", type=Path)
    attach.add_argument("--output", type=Path)
    attach.add_argument("--overwrite", action="store_true")
    export = subparsers.add_parser(
        "export-sidecar", help="Install a portable categorical_refiner folder beside a run ckpt directory"
    )
    export.add_argument("--base-checkpoint", type=Path, required=True)
    export.add_argument("--refiner-checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--categorical-root", type=Path, default=DEFAULT_CATEGORICAL_ROOT)
    export.add_argument("--head-index-override", type=int)
    export.add_argument("--overwrite", action="store_true")
    audio_export = subparsers.add_parser(
        "export-audio-selector",
        help="Install an audio_region_selector folder beside a run ckpt directory",
    )
    audio_export.add_argument("--base-checkpoint", type=Path, required=True)
    audio_export.add_argument("--selector-checkpoint", type=Path, required=True)
    audio_export.add_argument("--output-dir", type=Path)
    audio_export.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "train":
        train_classifier(args)
    elif args.command == "train-audio-selector":
        train_audio_region_selector(args)
    elif args.command == "refine-benchmark":
        refine_benchmark(args)
    elif args.command == "export-sidecar":
        export_sidecar(args)
    elif args.command == "export-audio-selector":
        export_audio_selector(args)
    elif args.command == "attach":
        attach_classifier(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
