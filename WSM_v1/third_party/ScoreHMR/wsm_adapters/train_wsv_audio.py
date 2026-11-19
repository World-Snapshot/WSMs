from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


AUDIO_CHANNELS = slice(175, 185)
MASK_VALUE = -1.0


class MetricLogger:
    def __init__(self, out_dir: Path) -> None:
        self.csv_path = out_dir / "metrics.csv"
        self.stats_path = out_dir / "stats.png"
        self.rows: list[tuple[int, float, float, float, float]] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.csv_path.write_text("step,train_loss,val_loss,train_metric,val_metric\n")

    def append(self, step: int, train_loss: float, val_loss: float, train_metric: float, val_metric: float) -> None:
        self.rows.append((int(step), float(train_loss), float(val_loss), float(train_metric), float(val_metric)))
        with self.csv_path.open("a") as f:
            f.write(
                f"{int(step)},{float(train_loss):.10e},{float(val_loss):.10e},"
                f"{float(train_metric):.10e},{float(val_metric):.10e}\n"
            )
        self.plot()

    def plot(self) -> None:
        if not self.rows:
            return
        steps = [row[0] for row in self.rows]
        train = [row[1] for row in self.rows]
        val = [row[2] for row in self.rows]
        train_metric = [row[3] for row in self.rows]
        val_metric = [row[4] for row in self.rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(steps, train, label="train_loss")
        axes[0].plot(steps, val, label="val_loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        axes[1].plot(steps, train_metric, label="train_metric")
        axes[1].plot(steps, val_metric, label="val_metric")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("IoU / accuracy")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(self.stats_path)
        plt.close(fig)


AVIS_CATEGORY_NAMES = {
    1: "person",
    2: "violin",
    3: "guitar",
    4: "cello",
    5: "flute",
    6: "piano",
    7: "ukulele",
    8: "accordion",
    9: "guzheng",
    10: "clarinet",
    11: "cat",
    12: "car",
    13: "saxophone",
    14: "dog",
    15: "lawn_mower",
    16: "tuba",
    17: "banjo",
    18: "pipa",
    19: "bassoon",
    20: "airplane",
    21: "tree_harvester",
    22: "trumpet",
    23: "lion",
    24: "bass",
    25: "erhu",
    26: "horse",
}


def category_label(category: int) -> str:
    return f"{category}:{AVIS_CATEGORY_NAMES.get(int(category), 'unknown')}"


class AudioSparseDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        image_size: int,
        task: str,
        min_audio_rms_dbfs: float | None = None,
        temporal_radius: int = 0,
        temporal_center_weight: float = 0.0,
        temporal_return_sequence: bool = False,
        category_load_image: bool = False,
        category_image_crop: bool = False,
    ) -> None:
        self.data = np.load(dataset_file)
        self.imgname = self.data["imgname"].astype(str)
        self.audio_feat = self.data["audio_feat"].astype(np.float32)
        self.audio_rms_dbfs = self.data["audio_rms_dbfs"].astype(np.float32) if "audio_rms_dbfs" in self.data else None
        self.bbox = self.data["bbox"].astype(np.float32)
        self.mask = self.data["mask"].astype(np.float32) if "mask" in self.data else None
        self.category = self.data["category"].astype(np.int64)
        self.wsv_path = self.data["wsv_path"].astype(str) if "wsv_path" in self.data else np.asarray([""] * len(self.imgname))
        self.frame_key = self.data["frame_key"].astype(str) if "frame_key" in self.data else np.asarray([""] * len(self.imgname))
        self.filtered_count = 0
        if min_audio_rms_dbfs is not None:
            if self.audio_rms_dbfs is None:
                print(f"[audio-filter] {dataset_file} has no audio_rms_dbfs; skip dynamic loudness filtering", flush=True)
            else:
                keep = self.audio_rms_dbfs >= float(min_audio_rms_dbfs)
                self.filtered_count = int((~keep).sum())
                self.imgname = self.imgname[keep]
                self.audio_feat = self.audio_feat[keep]
                self.audio_rms_dbfs = self.audio_rms_dbfs[keep]
                self.bbox = self.bbox[keep]
                self.category = self.category[keep]
                self.wsv_path = self.wsv_path[keep]
                self.frame_key = self.frame_key[keep]
                if self.mask is not None:
                    self.mask = self.mask[keep]
                print(
                    f"[audio-filter] {Path(dataset_file).name}: kept {int(keep.sum())}/{len(keep)} "
                    f"samples with audio_rms_dbfs >= {float(min_audio_rms_dbfs):.2f}",
                    flush=True,
                )
                if int(keep.sum()) == 0:
                    raise RuntimeError(f"No samples remain after audio loudness filtering: {dataset_file}")
        if self.audio_rms_dbfs is None:
            self.audio_rms_dbfs = np.zeros((len(self.audio_feat),), dtype=np.float32)
        self.audio_encoder = str(self.data["audio_encoder"].item()) if "audio_encoder" in self.data else "raw"
        self.audio_model_name = str(self.data["audio_model_name"].item()) if "audio_model_name" in self.data else ""
        self.audio_code_shape = tuple(int(x) for x in self.data["audio_code_shape"].tolist()) if "audio_code_shape" in self.data else (0, 0)
        self.audio_codebook_size = int(self.data["audio_codebook_size"].item()) if "audio_codebook_size" in self.data else 1024
        self.audio_codec_sample_rate = int(self.data["audio_codec_sample_rate"].item()) if "audio_codec_sample_rate" in self.data else 24000
        self.audio_codec_bandwidth = float(self.data["audio_codec_bandwidth"].item()) if "audio_codec_bandwidth" in self.data else 6.0
        self.image_size = image_size
        self.task = task
        self.temporal_radius = max(0, int(temporal_radius))
        self.temporal_center_weight = float(temporal_center_weight)
        self.temporal_return_sequence = bool(temporal_return_sequence)
        self.category_load_image = bool(category_load_image)
        self.category_image_crop = bool(category_image_crop)
        self.temporal_neighbors: list[np.ndarray] | None = None
        if self.temporal_radius > 0:
            groups: dict[str, list[int]] = {}
            for index, path in enumerate(self.wsv_path.tolist()):
                groups.setdefault(str(path), []).append(index)
            self.temporal_neighbors = [np.asarray([index], dtype=np.int64) for index in range(len(self.audio_feat))]
            for indices in groups.values():
                indices.sort(key=lambda index: str(self.frame_key[index]))
                for position, index in enumerate(indices):
                    if self.temporal_return_sequence:
                        neighbors = [
                            indices[min(max(position + offset, 0), len(indices) - 1)]
                            for offset in range(-self.temporal_radius, self.temporal_radius + 1)
                        ]
                        self.temporal_neighbors[index] = np.asarray(neighbors, dtype=np.int64)
                    else:
                        start = max(0, position - self.temporal_radius)
                        end = min(len(indices), position + self.temporal_radius + 1)
                        self.temporal_neighbors[index] = np.asarray(indices[start:end], dtype=np.int64)
            print(
                f"[audio-temporal] {Path(dataset_file).name}: "
                f"{'returning' if self.temporal_return_sequence else 'mean-pooling up to'} "
                f"{2 * self.temporal_radius + 1} frames within each video",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.audio_feat)

    def _load_image(self, path: str, bbox: np.ndarray | None = None) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if bbox is not None:
            width, height = image.size
            cx, cy, bw, bh = bbox.tolist()
            pad = 0.15
            x0 = max(0, int((cx - bw * (0.5 + pad)) * width))
            y0 = max(0, int((cy - bh * (0.5 + pad)) * height))
            x1 = min(width, int((cx + bw * (0.5 + pad)) * width))
            y1 = min(height, int((cy + bh * (0.5 + pad)) * height))
            if x1 > x0 and y1 > y0:
                image = image.crop((x0, y0, x1, y1))
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)))

    def _load_mask(self, index: int) -> torch.Tensor:
        if self.mask is None:
            cx, cy, bw, bh = self.bbox[index].tolist()
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            x0 = max(0, int((cx - bw / 2) * self.image_size))
            y0 = max(0, int((cy - bh / 2) * self.image_size))
            x1 = min(self.image_size, int((cx + bw / 2) * self.image_size))
            y1 = min(self.image_size, int((cy + bh / 2) * self.image_size))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0
            return torch.from_numpy(mask[None])
        mask_img = Image.fromarray((self.mask[index] > 0).astype(np.uint8) * 255)
        mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = (np.asarray(mask_img, dtype=np.float32) / 255.0)[None]
        return torch.from_numpy(mask)

    def __getitem__(self, index: int) -> dict:
        bbox = self.bbox[index]
        if self.temporal_neighbors is None:
            audio_feat = self.audio_feat[index]
        else:
            neighbors = self.temporal_neighbors[index]
            neighbor_audio = self.audio_feat[neighbors]
            if self.temporal_return_sequence:
                audio_feat = neighbor_audio
            elif self.temporal_center_weight > 0.0 and len(neighbors) > 1:
                center_weight = min(max(self.temporal_center_weight, 0.0), 1.0)
                weights = np.full(len(neighbors), (1.0 - center_weight) / (len(neighbors) - 1), dtype=np.float32)
                weights[np.where(neighbors == index)[0][0]] = center_weight
                audio_feat = (neighbor_audio * weights[:, None]).sum(axis=0, dtype=np.float32)
            else:
                audio_feat = neighbor_audio.mean(axis=0, dtype=np.float32)
        item = {
            "audio": torch.from_numpy(audio_feat),
            "bbox": torch.from_numpy(bbox),
            "category": torch.tensor(int(self.category[index]), dtype=torch.long),
            "audio_rms_dbfs": torch.tensor(float(self.audio_rms_dbfs[index]), dtype=torch.float32),
            "imgname": self.imgname[index],
            "wsv_path": self.wsv_path[index],
            "frame_key": self.frame_key[index],
        }
        if self.task == "visual_to_audio":
            item["image"] = self._load_image(self.imgname[index], bbox=bbox)
        elif self.task in {"localization", "segmentation"}:
            item["image"] = self._load_image(self.imgname[index], bbox=None)
            item["mask"] = self._load_mask(index)
        if self.task == "category" and self.mask is not None:
            item["mask"] = self._load_mask(index)
        if self.task == "category" and self.category_load_image:
            item["image"] = self._load_image(
                self.imgname[index],
                bbox=bbox if self.category_image_crop else None,
            )
        return item


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, out_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        ys = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return self.net(torch.cat((image, coord), dim=1))


class AudioEncoder(nn.Module):
    def __init__(self, audio_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, out_dim),
            nn.SiLU(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.net(audio)


class SpatialImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        ys = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return self.net(torch.cat((image, coord), dim=1))


class AudioLocalizationHead(nn.Module):
    def __init__(self, audio_dim: int) -> None:
        super().__init__()
        self.image = ImageEncoder(256)
        self.audio = AudioEncoder(audio_dim, 256)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 4))

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.image(image), self.audio(audio)), dim=1)).sigmoid()


class AudioConditionedResNetCenterBoxHead(nn.Module):
    """Head-001 spatial detector conditioned on the current-frame AST embedding."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        init_checkpoint: str,
        grounding_weight: float = 0.0,
        grounding_margin: float = 0.5,
        category_aux_weight: float = 0.15,
        crossmodal_matching: bool = False,
        crossmodal_contrastive_weight: float = 0.0,
        crossmodal_pixel_weight: float = 0.0,
        crossmodal_temperature: float = 0.10,
        grid_size: int = 14,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        network = resnet18(weights=ResNet18_Weights.DEFAULT if init_checkpoint else None)
        self.backbone = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.grid_size = int(grid_size)
        self.dense_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 5, 1),
        )
        self.audio = AudioEncoder(audio_dim, 256)
        self.film = nn.Linear(256, 512)
        self.category_head = nn.Linear(256, num_classes + 1)
        self.visual_projection = nn.Conv2d(256, 128, 1)
        self.audio_projection = nn.Linear(256, 128)
        self.crossmodal_gate = nn.Parameter(torch.tensor(0.0))
        self.grounding_weight = float(grounding_weight)
        self.grounding_margin = float(grounding_margin)
        self.category_aux_weight = float(category_aux_weight)
        self.crossmodal_matching = bool(crossmodal_matching)
        self.crossmodal_contrastive_weight = float(crossmodal_contrastive_weight)
        self.crossmodal_pixel_weight = float(crossmodal_pixel_weight)
        self.crossmodal_temperature = float(crossmodal_temperature)
        self.crossmodal_visual_query: torch.Tensor | None = None
        self.crossmodal_audio_query: torch.Tensor | None = None
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        if init_checkpoint:
            self.load_bbox_initialization(init_checkpoint)

    def load_bbox_initialization(self, checkpoint_path: str) -> None:
        if not checkpoint_path:
            raise ValueError("The ResNet-Gaussian localization modes require --bbox-init-checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint)
        current = self.state_dict()
        transferable = {
            key: value
            for key, value in source.items()
            if key in current
            and current[key].shape == value.shape
            and key.startswith(("backbone.", "dense_head.", "imagenet_"))
        }
        missing_dense = [
            key for key in current if key.startswith("dense_head.") and key not in transferable
        ]
        if missing_dense:
            raise RuntimeError(f"Head-001 checkpoint is missing detector tensors: {missing_dense}")
        self.load_state_dict(transferable, strict=False)
        print(
            f"[audio:localization] initialized {len(transferable)} spatial tensors from "
            f"{checkpoint_path} (step={checkpoint.get('step', 'unknown')})",
            flush=True,
        )

    def _visual_features(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone((image - self.imagenet_mean) / self.imagenet_std)

    def forward_dense(
        self,
        image: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self._visual_features(image)
        audio_feat = self.audio(audio)
        gamma, beta = self.film(audio_feat).chunk(2, dim=1)
        conditioned = visual * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        dense = self.dense_head(conditioned)
        center_logits = dense[:, 0]
        if self.crossmodal_matching:
            visual_query = F.normalize(self.visual_projection(visual), dim=1)
            audio_query = F.normalize(self.audio_projection(audio_feat), dim=1)
            self.crossmodal_visual_query = visual_query
            self.crossmodal_audio_query = audio_query
            similarity = (visual_query * audio_query[:, :, None, None]).sum(dim=1)
            center_logits = center_logits + self.crossmodal_gate * similarity
        logits = center_logits.flatten(1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size, device=image.device, dtype=image.dtype),
            torch.arange(self.grid_size, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        offsets = 0.5 * dense[:, 1:3].tanh()
        center_x = (grid_x[None] + 0.5 + offsets[:, 0]) / self.grid_size
        center_y = (grid_y[None] + 0.5 + offsets[:, 1]) / self.grid_size
        sizes = dense[:, 3:5].sigmoid()
        boxes = torch.stack((center_x, center_y, sizes[:, 0], sizes[:, 1]), dim=-1)
        boxes = boxes.reshape(image.shape[0], self.grid_size * self.grid_size, 4)
        return logits, boxes, self.category_head(audio_feat)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        logits, boxes, _category_logits = self.forward_dense(image, audio)
        best = logits.argmax(dim=1)
        return boxes[torch.arange(boxes.shape[0], device=boxes.device), best]



class AudioCLIPResNetCenterBoxHead(AudioConditionedResNetCenterBoxHead):
    """ResNet spatial localizer augmented with frozen CLIP patch semantics."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        init_checkpoint: str,
        category_aux_weight: float = 0.15,
        crossmodal_contrastive_weight: float = 0.10,
        crossmodal_pixel_weight: float = 0.10,
        crossmodal_temperature: float = 0.10,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        load_pretrained_clip: bool = True,
    ) -> None:
        super().__init__(
            audio_dim=audio_dim,
            num_classes=num_classes,
            init_checkpoint=init_checkpoint,
            category_aux_weight=category_aux_weight,
            crossmodal_matching=True,
            crossmodal_contrastive_weight=crossmodal_contrastive_weight,
            crossmodal_pixel_weight=crossmodal_pixel_weight,
            crossmodal_temperature=crossmodal_temperature,
        )
        from transformers import CLIPVisionConfig, CLIPVisionModel

        if load_pretrained_clip:
            self.clip_vision = CLIPVisionModel.from_pretrained(
                clip_model_name,
                local_files_only=True,
            )
        else:
            self.clip_vision = CLIPVisionModel(CLIPVisionConfig())
        self.clip_vision.eval()
        for parameter in self.clip_vision.parameters():
            parameter.requires_grad_(False)
        self.clip_projection = nn.Sequential(
            nn.Conv2d(self.clip_vision.config.hidden_size, 256, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.clip_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "clip_mean",
            torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1),
        )
        if load_pretrained_clip:
            print(
                f"[audio:localization] loaded frozen CLIP vision backbone {clip_model_name}",
                flush=True,
            )
        else:
            print(
                f"[audio:localization] initialized CLIP architecture for checkpoint restore",
                flush=True,
            )

    def _visual_features(self, image: torch.Tensor) -> torch.Tensor:
        resnet_features = super()._visual_features(image)
        self.clip_vision.eval()
        with torch.no_grad():
            clip_output = self.clip_vision(
                pixel_values=(image - self.clip_mean) / self.clip_std
            ).last_hidden_state
            patch_tokens = clip_output[:, 1:]
            patch_grid = int(round(patch_tokens.shape[1] ** 0.5))
            clip_features = patch_tokens.transpose(1, 2).reshape(
                image.shape[0],
                patch_tokens.shape[2],
                patch_grid,
                patch_grid,
            )
        clip_features = self.clip_projection(clip_features)
        clip_features = F.interpolate(
            clip_features,
            size=resnet_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return resnet_features + self.clip_scale.tanh() * clip_features


class AudioMaskToBoxLocalizationHead(AudioConditionedResNetCenterBoxHead):
    """Predict the native sound-source mask, then export its enclosing box."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        init_checkpoint: str,
        category_aux_weight: float = 0.10,
        mask_target_mode: str = "source",
    ) -> None:
        super().__init__(
            audio_dim=audio_dim,
            num_classes=num_classes,
            init_checkpoint=init_checkpoint,
            category_aux_weight=category_aux_weight,
        )
        self.mask_target_mode = str(mask_target_mode)
        self.mask_threshold = float(np.exp(-2.0)) if self.mask_target_mode == "gaussian" else 0.5
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.constant_(self.mask_decoder[-1].bias, -2.0)

    def mask_logits(self, image: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self._visual_features(image)
        audio_feat = self.audio(audio)
        gamma, beta = self.film(audio_feat).chunk(2, dim=1)
        conditioned = visual * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.mask_decoder(conditioned), self.category_head(audio_feat)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        logits, _category_logits = self.mask_logits(image, audio)
        mask = logits.sigmoid() > self.mask_threshold
        boxes: list[torch.Tensor] = []
        height, width = mask.shape[-2:]
        for sample_index, sample_mask in enumerate(mask[:, 0]):
            positions = torch.nonzero(sample_mask, as_tuple=False)
            if positions.numel() == 0:
                flat_index = logits[sample_index, 0].argmax()
                y0 = flat_index // width
                x0 = flat_index % width
                x1 = x0
                y1 = y0
            else:
                y0, x0 = positions.min(dim=0).values
                y1, x1 = positions.max(dim=0).values
            pixel_width = (x1 - x0 + 1).to(logits.dtype)
            pixel_height = (y1 - y0 + 1).to(logits.dtype)
            box_width = pixel_width / width
            box_height = pixel_height / height
            center_x = (x0.to(logits.dtype) + 0.5 * pixel_width) / width
            center_y = (y0.to(logits.dtype) + 0.5 * pixel_height) / height
            boxes.append(torch.stack((center_x, center_y, box_width, box_height)))
        return torch.stack(boxes, dim=0)



class AudioJointMaskBoxLocalizationHead(AudioMaskToBoxLocalizationHead):
    """Joint dense-box and native-mask localizer with a self-contained fused output."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        init_checkpoint: str,
        category_aux_weight: float = 0.10,
        grounding_weight: float = 0.35,
        grounding_margin: float = 0.5,
        crossmodal_contrastive_weight: float = 0.10,
        crossmodal_pixel_weight: float = 0.25,
        crossmodal_temperature: float = 0.10,
        mask_loss_weight: float = 0.50,
        fusion_cross_weight: float = 0.90,
        fusion_learned: bool = False,
        fused_box_loss_weight: float = 1.0,
        fusion_residual_scale: float = 0.0,
        mask_threshold: float = 0.57,
    ) -> None:
        super().__init__(
            audio_dim=audio_dim,
            num_classes=num_classes,
            init_checkpoint=init_checkpoint,
            category_aux_weight=category_aux_weight,
            mask_target_mode="source",
        )
        self.crossmodal_matching = True
        self.grounding_weight = float(grounding_weight)
        self.grounding_margin = float(grounding_margin)
        self.crossmodal_contrastive_weight = float(crossmodal_contrastive_weight)
        self.crossmodal_pixel_weight = float(crossmodal_pixel_weight)
        self.crossmodal_temperature = float(crossmodal_temperature)
        self.mask_loss_weight = float(mask_loss_weight)
        self.fusion_cross_weight = float(fusion_cross_weight)
        self.fusion_learned = bool(fusion_learned)
        self.fused_box_loss_weight = float(fused_box_loss_weight)
        self.fusion_residual_scale = float(fusion_residual_scale)
        self.mask_threshold = float(mask_threshold)
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 8, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )
        nn.init.zeros_(self.fusion_gate[-1].weight)
        initial_alpha = min(max(self.fusion_cross_weight, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.fusion_gate[-1].bias, float(np.log(initial_alpha / (1.0 - initial_alpha))))
        self.fusion_residual = nn.Sequential(
            nn.Linear(256 + 8, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )
        nn.init.zeros_(self.fusion_residual[-1].weight)
        nn.init.zeros_(self.fusion_residual[-1].bias)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        dense_box = AudioConditionedResNetCenterBoxHead.forward(self, image, audio)
        mask_box = AudioMaskToBoxLocalizationHead.forward(self, image, audio)
        if self.fusion_learned:
            audio_feat = self.audio(audio)
            fusion_input = torch.cat((audio_feat, dense_box, mask_box), dim=1)
            alpha = self.fusion_gate(fusion_input).sigmoid()
        else:
            alpha = min(max(self.fusion_cross_weight, 0.0), 1.0)
        fused_box = alpha * dense_box + (1.0 - alpha) * mask_box
        if self.fusion_learned and self.fusion_residual_scale > 0.0:
            residual = self.fusion_residual(fusion_input).tanh()
            fused_box = fused_box + self.fusion_residual_scale * residual
        return fused_box.clamp(0.0, 1.0)



class AudioCLIPJointMaskBoxLocalizationHead(AudioCLIPResNetCenterBoxHead):
    """Self-contained fixed fusion of the CLIP box and native-mask branches."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        init_checkpoint: str,
        mask_init_checkpoint: str,
        fusion_cross_weight: float = 0.875,
        mask_threshold: float = 0.57,
        load_pretrained_clip: bool = True,
    ) -> None:
        super().__init__(
            audio_dim=audio_dim,
            num_classes=num_classes,
            init_checkpoint=init_checkpoint,
            load_pretrained_clip=load_pretrained_clip,
        )
        self.fusion_cross_weight = float(fusion_cross_weight)
        self.mask_threshold = float(mask_threshold)
        self.mask_audio = AudioEncoder(audio_dim, 256)
        self.mask_film = nn.Linear(256, 512)
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),
        )
        if mask_init_checkpoint:
            self.load_mask_initialization(mask_init_checkpoint)

    def load_mask_initialization(self, checkpoint_path: str) -> None:
        if not checkpoint_path:
            raise ValueError("CLIP joint mask mode requires --mask-init-checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint)
        remapped = {}
        for source_key, target_prefix in (("audio.", "mask_audio."), ("film.", "mask_film.")):
            for key, value in source.items():
                if key.startswith(source_key):
                    remapped[target_prefix + key[len(source_key):]] = value
        remapped.update({
            key: value
            for key, value in source.items()
            if key.startswith("mask_decoder.")
        })
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in remapped.items()
            if key in current and current[key].shape == value.shape
        }
        if not any(key.startswith("mask_decoder.") for key in compatible):
            raise RuntimeError(f"No compatible mask decoder tensors in {checkpoint_path}")
        self.load_state_dict(compatible, strict=False)
        print(
            f"[audio:localization] initialized {len(compatible)} independent mask tensors from "
            f"{checkpoint_path} (step={checkpoint.get(chr(115) + chr(116) + chr(101) + chr(112), chr(117) + chr(110) + chr(107) + chr(110) + chr(111) + chr(119) + chr(110))})",
            flush=True,
        )

    def _mask_box(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        visual = AudioConditionedResNetCenterBoxHead._visual_features(self, image)
        audio_feat = self.mask_audio(audio)
        gamma, beta = self.mask_film(audio_feat).chunk(2, dim=1)
        conditioned = visual * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        logits = self.mask_decoder(conditioned)
        mask = logits.sigmoid() > self.mask_threshold
        boxes = []
        height, width = mask.shape[-2:]
        for sample_index, sample_mask in enumerate(mask[:, 0]):
            positions = torch.nonzero(sample_mask, as_tuple=False)
            if positions.numel() == 0:
                flat_index = logits[sample_index, 0].argmax()
                y0 = flat_index // width
                x0 = flat_index % width
                x1 = x0
                y1 = y0
            else:
                y0, x0 = positions.min(dim=0).values
                y1, x1 = positions.max(dim=0).values
            pixel_width = (x1 - x0 + 1).to(logits.dtype)
            pixel_height = (y1 - y0 + 1).to(logits.dtype)
            boxes.append(torch.stack((
                (x0.to(logits.dtype) + 0.5 * pixel_width) / width,
                (y0.to(logits.dtype) + 0.5 * pixel_height) / height,
                pixel_width / width,
                pixel_height / height,
            )))
        return torch.stack(boxes, dim=0)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        clip_box = AudioCLIPResNetCenterBoxHead.forward(self, image, audio)
        mask_box = self._mask_box(image, audio)
        alpha = min(max(self.fusion_cross_weight, 0.0), 1.0)
        return alpha * clip_box + (1.0 - alpha) * mask_box


class BalancedPrototypeAudioCategoryHead(nn.Module):
    """Cosine prototype classifier with a zero-init residual AST adapter."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        class_weights: torch.Tensor,
        label_smoothing: float = 0.05,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        hidden_dim = min(512, audio_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, audio_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.prototypes = nn.Parameter(torch.randn(num_classes + 1, audio_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(15.0), dtype=torch.float32))
        self.register_buffer("class_weights", class_weights.float())
        self.label_smoothing = float(label_smoothing)

    @torch.no_grad()
    def initialize_prototypes(self, audio_feat: np.ndarray, category: np.ndarray) -> None:
        audio = F.normalize(torch.from_numpy(np.asarray(audio_feat, dtype=np.float32)), dim=1)
        labels = torch.from_numpy(np.asarray(category, dtype=np.int64))
        initialized = 0
        for label in range(1, self.prototypes.shape[0]):
            selected = audio[labels == label]
            if selected.numel() == 0:
                continue
            self.prototypes[label].copy_(F.normalize(selected.mean(dim=0), dim=0))
            initialized += 1
        self.prototypes[0].copy_(F.normalize(self.prototypes[0], dim=0))
        print(f"[audio:category] initialized {initialized} class prototypes from train AST means", flush=True)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        embedding = F.normalize(audio + self.adapter(audio), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * embedding @ prototypes.t()



class TemporalPrototypeAudioCategoryHead(BalancedPrototypeAudioCategoryHead):
    """Prototype classifier with learnable, identity-stable temporal pooling."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        class_weights: torch.Tensor,
        temporal_length: int,
        label_smoothing: float = 0.05,
        dropout: float = 0.30,
    ) -> None:
        super().__init__(
            audio_dim=audio_dim,
            num_classes=num_classes,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            dropout=dropout,
        )
        self.temporal_position_logits = nn.Parameter(torch.zeros(temporal_length))
        self.temporal_similarity_scale = nn.Parameter(torch.tensor(0.0))
        self.temporal_pool_gate = nn.Parameter(torch.tensor(2.0))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"Temporal prototype head expects [batch,time,dim], got {tuple(audio.shape)}")
        center = audio[:, audio.shape[1] // 2]
        normalized = F.normalize(audio, dim=2)
        normalized_center = F.normalize(center, dim=1)
        similarity = (normalized * normalized_center[:, None]).sum(dim=2)
        position_logits = self.temporal_position_logits[: audio.shape[1]]
        weights = F.softmax(
            self.temporal_similarity_scale * similarity + position_logits[None],
            dim=1,
        )
        pooled = (weights[:, :, None] * audio).sum(dim=1)
        gate = self.temporal_pool_gate.sigmoid()
        aggregated = gate * pooled + (1.0 - gate) * center
        return super().forward(aggregated)


class AudioVisualFusionCategoryHead(BalancedPrototypeAudioCategoryHead):
    """Frozen temporal-audio prototype prior plus a trainable RGB discriminator."""

    def __init__(
        self,
        audio_dim: int,
        num_classes: int,
        class_weights: torch.Tensor,
        label_smoothing: float = 0.05,
        dropout: float = 0.30,
    ) -> None:
        super().__init__(
            audio_dim=audio_dim,
            num_classes=num_classes,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            dropout=dropout,
        )
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        visual_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.visual_backbone = backbone
        self.visual_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Dropout(0.20),
            nn.Linear(visual_dim, num_classes + 1),
        )
        self.visual_gate = nn.Parameter(torch.tensor(-1.0))
        nn.init.zeros_(self.visual_head[-1].weight)
        nn.init.zeros_(self.visual_head[-1].bias)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
        self.prototypes.requires_grad_(False)
        self.logit_scale.requires_grad_(False)
        for parameter in self.visual_backbone.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, audio: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        audio_logits = super().forward(audio)
        self.visual_backbone.eval()
        with torch.no_grad():
            visual = self.visual_backbone((image - self.imagenet_mean) / self.imagenet_std)
        visual_logits = self.visual_head(visual)
        return audio_logits + self.visual_gate.sigmoid() * visual_logits


class AudioSegmentationHead(nn.Module):
    def __init__(self, audio_dim: int) -> None:
        super().__init__()
        channels = 192
        self.image = SpatialImageEncoder(channels)
        self.audio = AudioEncoder(audio_dim, channels * 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 192, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(192, 128, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        feat = self.image(image)
        gamma, beta = self.audio(audio).chunk(2, dim=1)
        feat = feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.decoder(feat)


class AudioCategoryHead(nn.Module):
    def __init__(self, audio_dim: int, num_classes: int) -> None:
        super().__init__()
        self.audio = AudioEncoder(audio_dim, 384)
        self.head = nn.Sequential(nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, num_classes + 1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.audio(audio))


class VisualToAudioHead(nn.Module):
    def __init__(self, audio_dim: int) -> None:
        super().__init__()
        self.image = ImageEncoder(384)
        self.head = nn.Sequential(nn.Linear(384, 512), nn.SiLU(), nn.Linear(512, audio_dim), nn.Tanh())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.image(image))


def _cxcywh_to_xyxy(box: np.ndarray, width: int, height: int) -> list[float]:
    cx, cy, bw, bh = box.tolist()
    return [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ]


def _bbox_iou_np(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    p = np.asarray(pred_box, dtype=np.float32)
    g = np.asarray(gt_box, dtype=np.float32)
    p_xy = np.asarray([p[0] - p[2] / 2, p[1] - p[3] / 2, p[0] + p[2] / 2, p[1] + p[3] / 2], dtype=np.float32)
    g_xy = np.asarray([g[0] - g[2] / 2, g[1] - g[3] / 2, g[0] + g[2] / 2, g[1] + g[3] / 2], dtype=np.float32)
    ix0, iy0 = np.maximum(p_xy[:2], g_xy[:2])
    ix1, iy1 = np.minimum(p_xy[2:], g_xy[2:])
    inter = max(0.0, float(ix1 - ix0)) * max(0.0, float(iy1 - iy0))
    p_area = max(0.0, float(p_xy[2] - p_xy[0])) * max(0.0, float(p_xy[3] - p_xy[1]))
    g_area = max(0.0, float(g_xy[2] - g_xy[0])) * max(0.0, float(g_xy[3] - g_xy[1]))
    return inter / max(p_area + g_area - inter, 1e-8)


def _bbox_iou_torch(pred_box: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    p_xy = torch.stack((pred_box[:, 0] - pred_box[:, 2] / 2, pred_box[:, 1] - pred_box[:, 3] / 2, pred_box[:, 0] + pred_box[:, 2] / 2, pred_box[:, 1] + pred_box[:, 3] / 2), dim=1)
    g_xy = torch.stack((gt_box[:, 0] - gt_box[:, 2] / 2, gt_box[:, 1] - gt_box[:, 3] / 2, gt_box[:, 0] + gt_box[:, 2] / 2, gt_box[:, 1] + gt_box[:, 3] / 2), dim=1)
    inter_min = torch.maximum(p_xy[:, :2], g_xy[:, :2])
    inter_max = torch.minimum(p_xy[:, 2:], g_xy[:, 2:])
    inter_wh = (inter_max - inter_min).clamp_min(0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    p_area = (p_xy[:, 2] - p_xy[:, 0]).clamp_min(0.0) * (p_xy[:, 3] - p_xy[:, 1]).clamp_min(0.0)
    g_area = (g_xy[:, 2] - g_xy[:, 0]).clamp_min(0.0) * (g_xy[:, 3] - g_xy[:, 1]).clamp_min(0.0)
    return inter / (p_area + g_area - inter).clamp_min(1e-8)


def draw_localization(
    img_path: str,
    pred_box: np.ndarray,
    gt_box: np.ndarray,
    out: Path,
    category: int | None = None,
    rms_dbfs: float | None = None,
) -> None:
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle(_cxcywh_to_xyxy(gt_box, width, height), outline=(0, 255, 0), width=3)
    draw.rectangle(_cxcywh_to_xyxy(pred_box, width, height), outline=(255, 0, 0), width=2)
    draw.rectangle((0, 0, 620, 70), fill=(0, 0, 0))
    draw.text((8, 7), f"bbox IoU={_bbox_iou_np(pred_box, gt_box):.3f} | GT=green pred=red", fill=(255, 255, 255))
    if category is not None:
        draw.text((8, 29), f"category={category_label(int(category))}", fill=(80, 255, 80))
    if rms_dbfs is not None:
        draw.text((8, 51), f"audio_rms={float(rms_dbfs):.1f} dBFS", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def draw_segmentation(img_path: str, pred_mask: np.ndarray, gt_mask: np.ndarray, out: Path) -> None:
    image = Image.open(img_path).convert("RGB").resize((pred_mask.shape[1], pred_mask.shape[0]), Image.BICUBIC)
    base = np.asarray(image, dtype=np.float32)
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    overlay = base.copy()
    overlay[gt] = overlay[gt] * 0.35 + np.asarray([0, 255, 0], dtype=np.float32) * 0.65
    overlay[pred] = overlay[pred] * 0.35 + np.asarray([255, 0, 0], dtype=np.float32) * 0.65
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out)


def _decode_frame_audio(frame: np.ndarray, max_raw_bytes: int = 9000) -> np.ndarray | None:
    if frame.ndim != 3 or frame.shape[-1] < AUDIO_CHANNELS.stop:
        return None
    audio_channels = frame[:, :, AUDIO_CHANNELS]
    audio_mask = np.any(audio_channels != MASK_VALUE, axis=2)
    ys, xs = np.where(audio_mask)
    if ys.size == 0:
        return None
    raw: list[int] = []
    for y, x in sorted(zip(ys.tolist(), xs.tolist())):
        for value in audio_channels[y, x]:
            if value == MASK_VALUE:
                continue
            byte_value = int((float(value) / 100.0 + 0.5) * 255.0)
            raw.append(int(np.clip(byte_value, 0, 255)))
            if len(raw) >= 14:
                length = raw[6] | (raw[7] << 8) | (raw[8] << 16) | (raw[9] << 24)
                if length == 7743:
                    length = 8000
                expected_total = 14 + max(0, min(length, max_raw_bytes))
                if len(raw) >= expected_total:
                    audio_uint8 = np.asarray(raw[14:expected_total], dtype=np.uint8)
                    if audio_uint8.size > 0:
                        return (audio_uint8.astype(np.float32) - 128.0) / 127.0
            if len(raw) >= max_raw_bytes:
                break
        if len(raw) >= max_raw_bytes:
            break
    if len(raw) > 14:
        audio_uint8 = np.asarray(raw[14:], dtype=np.uint8)
        return (audio_uint8.astype(np.float32) - 128.0) / 127.0
    return None


def write_category_audio_clip(wsv_path: str, frame_key: str, out: Path) -> None:
    if not wsv_path or not frame_key:
        return
    try:
        data = np.load(wsv_path)
        if frame_key not in data:
            return
        audio = _decode_frame_audio(data[frame_key])
    except Exception as exc:
        print(f"[audio:category] failed to export audio clip for {wsv_path}:{frame_key}: {exc}", flush=True)
        return
    if audio is None or audio.size == 0:
        return
    _write_wav_pair(out, 8000, audio)


def draw_category(img_path: str, pred: int, gt: int, out: Path, gt_mask: np.ndarray | None = None, rms_dbfs: float | None = None) -> None:
    image = Image.open(img_path).convert("RGB")
    if gt_mask is not None:
        mask_img = Image.fromarray((gt_mask > 0.5).astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
        base = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask_img) > 0
        base[mask] = base[mask] * 0.35 + np.asarray([0, 255, 0], dtype=np.float32) * 0.65
        image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 620, 70), fill=(0, 0, 0))
    draw.text((8, 8), f"pred={category_label(pred)}", fill=(255, 80, 80))
    draw.text((8, 30), f"gt={category_label(gt)} | GT mask=green", fill=(80, 255, 80))
    if rms_dbfs is not None:
        draw.text((8, 52), f"audio_rms={rms_dbfs:.1f} dBFS", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


_ENCODEC_MODEL_CACHE: dict[tuple[str, str], nn.Module] = {}


def _decode_encodec_vector(vector: np.ndarray, dataset: AudioSparseDataset, device: torch.device) -> np.ndarray:
    if dataset.audio_code_shape[0] <= 0 or dataset.audio_code_shape[1] <= 0:
        raise ValueError("Missing EnCodec code shape in dataset metadata")
    model_name = dataset.audio_model_name or "facebook/encodec_24khz"
    cache_key = (model_name, str(device))
    model = _ENCODEC_MODEL_CACHE.get(cache_key)
    if model is None:
        from transformers import EncodecModel

        model = EncodecModel.from_pretrained(model_name).to(device).eval()
        _ENCODEC_MODEL_CACHE[cache_key] = model
    q, t = dataset.audio_code_shape
    codebook_size = dataset.audio_codebook_size
    codes = np.rint((np.clip(vector, -1.0, 1.0) + 1.0) * 0.5 * float(codebook_size - 1))
    codes = np.clip(codes, 0, codebook_size - 1).astype(np.int64).reshape(1, 1, q, t)
    code_tensor = torch.from_numpy(codes).to(device)
    with torch.no_grad():
        decoded = model.decode(code_tensor, [None]).audio_values
    return decoded[0, 0].detach().float().cpu().numpy()


def _write_wav_pair(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    wavfile.write(path, sample_rate, clipped)
    peak = float(np.max(np.abs(clipped))) if clipped.size else 0.0
    if peak > 1e-6:
        wavfile.write(path.with_name(path.stem + "_listen.wav"), sample_rate, clipped / peak)


def draw_audio(
    pred: np.ndarray,
    gt: np.ndarray,
    out: Path,
    dataset: AudioSparseDataset | None = None,
    device: torch.device | None = None,
    img_path: str | None = None,
    rms_dbfs: float | None = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if dataset is not None and dataset.audio_encoder == "encodec":
        assert device is not None
        pred_audio = _decode_encodec_vector(pred, dataset, device)
        gt_audio = _decode_encodec_vector(gt, dataset, device)
        _write_wav_pair(out.with_name(out.stem + "_pred.wav"), dataset.audio_codec_sample_rate, pred_audio)
        _write_wav_pair(out.with_name(out.stem + "_gt.wav"), dataset.audio_codec_sample_rate, gt_audio)
    if img_path is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]})
        image = Image.open(img_path).convert("RGB")
        axes[0].imshow(image)
        axes[0].axis("off")
        title = Path(img_path).name
        if rms_dbfs is not None:
            title += f" | audio_rms={rms_dbfs:.1f} dBFS"
        axes[0].set_title(title)
        ax = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(gt, label="gt", linewidth=0.7)
    ax.plot(pred, label="pred", linewidth=0.7, alpha=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _generalized_iou_loss_single(pred_box: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    pred = torch.stack(
        (pred_box[:, 0] - pred_box[:, 2] / 2, pred_box[:, 1] - pred_box[:, 3] / 2,
         pred_box[:, 0] + pred_box[:, 2] / 2, pred_box[:, 1] + pred_box[:, 3] / 2),
        dim=1,
    ).clamp(0.0, 1.0)
    gt = torch.stack(
        (gt_box[:, 0] - gt_box[:, 2] / 2, gt_box[:, 1] - gt_box[:, 3] / 2,
         gt_box[:, 0] + gt_box[:, 2] / 2, gt_box[:, 1] + gt_box[:, 3] / 2),
        dim=1,
    ).clamp(0.0, 1.0)
    inter_min = torch.maximum(pred[:, :2], gt[:, :2])
    inter_max = torch.minimum(pred[:, 2:], gt[:, 2:])
    inter_wh = (inter_max - inter_min).clamp_min(0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = (pred[:, 2] - pred[:, 0]).clamp_min(0.0) * (pred[:, 3] - pred[:, 1]).clamp_min(0.0)
    gt_area = (gt[:, 2] - gt[:, 0]).clamp_min(0.0) * (gt[:, 3] - gt[:, 1]).clamp_min(0.0)
    union = (pred_area + gt_area - inter).clamp_min(1e-6)
    iou = inter / union
    enclosing_min = torch.minimum(pred[:, :2], gt[:, :2])
    enclosing_max = torch.maximum(pred[:, 2:], gt[:, 2:])
    enclosing_wh = (enclosing_max - enclosing_min).clamp_min(0.0)
    enclosing_area = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp_min(1e-6)
    return (1.0 - iou + (enclosing_area - union) / enclosing_area).mean()


def _centernet_focal_loss_single(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    probability = logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = centers > 0.5
    negative_weight = (1.0 - heatmap).pow(4.0)
    positive_loss = -(1.0 - probability).pow(2.0) * probability.log() * positive
    negative_loss = -probability.pow(2.0) * (1.0 - probability).log()
    negative_loss = negative_loss * negative_weight * (~positive)
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1)


def _dense_localization_loss(
    model: AudioConditionedResNetCenterBoxHead,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    image = batch["image"].to(device)
    audio = batch["audio"].to(device)
    gt = batch["bbox"].to(device)
    category = batch["category"].to(device)
    logits, boxes, category_logits = model.forward_dense(image, audio)
    grid_size = model.grid_size
    cell_x = (gt[:, 0] * grid_size).long().clamp(0, grid_size - 1)
    cell_y = (gt[:, 1] * grid_size).long().clamp(0, grid_size - 1)
    target_index = cell_y * grid_size + cell_x
    batch_index = torch.arange(gt.shape[0], device=device)
    center_target = torch.zeros_like(logits)
    center_target[batch_index, target_index] = 1.0
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(grid_size, device=device, dtype=gt.dtype) + 0.5) / grid_size,
        (torch.arange(grid_size, device=device, dtype=gt.dtype) + 0.5) / grid_size,
        indexing="ij",
    )
    sigma_x = gt[:, 2, None, None].div(6.0).clamp_min(1.0 / grid_size)
    sigma_y = gt[:, 3, None, None].div(6.0).clamp_min(1.0 / grid_size)
    heatmap = torch.exp(
        -0.5 * (
            ((grid_x[None] - gt[:, 0, None, None]) / sigma_x).pow(2.0)
            + ((grid_y[None] - gt[:, 1, None, None]) / sigma_y).pow(2.0)
        )
    ).flatten(1)
    heatmap[batch_index, target_index] = 1.0
    matched = boxes[batch_index, target_index]
    center_loss = _centernet_focal_loss_single(logits, heatmap, center_target)
    l1_loss = F.l1_loss(matched, gt)
    giou_loss = _generalized_iou_loss_single(matched, gt)
    pred_area = (matched[:, 2].clamp_min(1e-4) * matched[:, 3].clamp_min(1e-4)).clamp_min(1e-6)
    gt_area = (gt[:, 2].clamp_min(1e-4) * gt[:, 3].clamp_min(1e-4)).clamp_min(1e-6)
    area_loss = (pred_area.log() - gt_area.log()).abs().mean()
    loss = center_loss + 5.0 * l1_loss + 5.0 * giou_loss + area_loss
    if model.category_aux_weight > 0.0:
        loss = loss + model.category_aux_weight * F.cross_entropy(category_logits, category)
    if model.crossmodal_contrastive_weight > 0.0:
        visual_query = model.crossmodal_visual_query
        audio_query = model.crossmodal_audio_query
        if visual_query is None or audio_query is None:
            raise RuntimeError("Cross-modal contrastive loss requires crossmodal_matching")
        visual_at_target = visual_query.permute(0, 2, 3, 1).reshape(gt.shape[0], -1, visual_query.shape[1])
        visual_at_target = visual_at_target[batch_index, target_index]
        similarity = visual_at_target @ audio_query.t()
        similarity = similarity / max(model.crossmodal_temperature, 1e-4)
        positive = (category[:, None] == category[None, :]).to(similarity.dtype)
        visual_log_prob = F.log_softmax(similarity, dim=1)
        audio_log_prob = F.log_softmax(similarity.t(), dim=1)
        visual_contrast = -(visual_log_prob * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)
        audio_contrast = -(audio_log_prob * positive.t()).sum(dim=1) / positive.t().sum(dim=1).clamp_min(1.0)
        contrastive_loss = 0.5 * (visual_contrast.mean() + audio_contrast.mean())
        loss = loss + model.crossmodal_contrastive_weight * contrastive_loss

    if model.crossmodal_pixel_weight > 0.0:
        visual_query = model.crossmodal_visual_query
        audio_query = model.crossmodal_audio_query
        if visual_query is None or audio_query is None:
            raise RuntimeError("Pixel grounding loss requires crossmodal_matching")
        pixel_logits = (visual_query * audio_query[:, :, None, None]).sum(dim=1)
        pixel_logits = pixel_logits / max(model.crossmodal_temperature, 1e-4)
        pixel_target = F.interpolate(
            batch["mask"].to(device),
            size=pixel_logits.shape[-2:],
            mode="area",
        )[:, 0]
        positive = pixel_target.sum().clamp_min(1.0)
        negative = (pixel_target.numel() - positive).clamp_min(1.0)
        pos_weight = (negative / positive).clamp(1.0, 20.0)
        pixel_loss = F.binary_cross_entropy_with_logits(
            pixel_logits,
            pixel_target,
            pos_weight=pos_weight,
        )
        loss = loss + model.crossmodal_pixel_weight * pixel_loss

    if model.grounding_weight > 0.0 and audio.shape[0] > 1:
        wrong_audio = audio.roll(1, dims=0)
        wrong_category = category.roll(1, dims=0)
        wrong_logits, _wrong_boxes, _wrong_category_logits = model.forward_dense(image, wrong_audio)
        valid = category != wrong_category
        if valid.any():
            correct_score = logits[batch_index, target_index]
            wrong_score = wrong_logits[batch_index, target_index]
            grounding = F.relu(model.grounding_margin - correct_score + wrong_score)[valid].mean()
            loss = loss + model.grounding_weight * grounding
    return loss


def compute_category_logits(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    audio = batch["audio"].to(device)
    if isinstance(model, AudioVisualFusionCategoryHead):
        return model(audio, batch["image"].to(device))
    return model(audio)


def audio_loss(task: str, model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    if task == "localization":

        if isinstance(model, AudioJointMaskBoxLocalizationHead):
            if getattr(model, "branches_frozen", False):
                fused_box = model(batch["image"].to(device), batch["audio"].to(device))
                target_box = batch["bbox"].to(device)
                fused_loss = 5.0 * F.l1_loss(fused_box, target_box)
                return fused_loss + 5.0 * _generalized_iou_loss_single(fused_box, target_box)
            dense_loss = _dense_localization_loss(model, batch, device)
            mask_logits, mask_category_logits = model.mask_logits(
                batch["image"].to(device), batch["audio"].to(device)
            )
            mask_target = batch["mask"].to(device)
            positive = mask_target.sum().clamp_min(1.0)
            negative = (mask_target.numel() - positive).clamp_min(1.0)
            pos_weight = (negative / positive).clamp(1.0, 20.0)
            mask_bce = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_target,
                pos_weight=pos_weight,
            )
            mask_probability = mask_logits.sigmoid()
            mask_dice = 1.0 - (
                2.0 * (mask_probability * mask_target).sum(dim=(1, 2, 3)) + 1.0
            ) / (
                mask_probability.sum(dim=(1, 2, 3))
                + mask_target.sum(dim=(1, 2, 3))
                + 1.0
            )
            mask_loss = mask_bce + mask_dice.mean()
            if model.category_aux_weight > 0.0:
                mask_loss = mask_loss + model.category_aux_weight * F.cross_entropy(
                    mask_category_logits,
                    batch["category"].to(device),
                )
            fused_box = model(batch["image"].to(device), batch["audio"].to(device))
            target_box = batch["bbox"].to(device)
            fused_loss = 5.0 * F.l1_loss(fused_box, target_box)
            fused_loss = fused_loss + 5.0 * _generalized_iou_loss_single(fused_box, target_box)
            return (
                dense_loss
                + model.mask_loss_weight * mask_loss
                + model.fused_box_loss_weight * fused_loss
            )
        if isinstance(model, AudioMaskToBoxLocalizationHead):
            logits, category_logits = model.mask_logits(
                batch["image"].to(device), batch["audio"].to(device)
            )
            target = batch["mask"].to(device)
            if model.mask_target_mode in {"box", "gaussian"}:
                gt_box = batch["bbox"].to(device)
                height, width = logits.shape[-2:]
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=device, dtype=logits.dtype) + 0.5) / height,
                    (torch.arange(width, device=device, dtype=logits.dtype) + 0.5) / width,
                    indexing="ij",
                )
                dx = grid_x[None] - gt_box[:, 0, None, None]
                dy = grid_y[None] - gt_box[:, 1, None, None]
                if model.mask_target_mode == "box":
                    target = (
                        (dx.abs() <= gt_box[:, 2, None, None] / 2.0)
                        & (dy.abs() <= gt_box[:, 3, None, None] / 2.0)
                    ).to(logits.dtype)[:, None]
                else:
                    sigma_x = (gt_box[:, 2, None, None] / 4.0).clamp_min(1.0 / width)
                    sigma_y = (gt_box[:, 3, None, None] / 4.0).clamp_min(1.0 / height)
                    target = torch.exp(-0.5 * ((dx / sigma_x).pow(2.0) + (dy / sigma_y).pow(2.0)))[:, None]
            positive = target.sum().clamp_min(1.0)
            negative = target.numel() - positive
            pos_weight = (negative / positive).clamp(1.0, 20.0)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            probability = logits.sigmoid()
            dice = 1.0 - (2.0 * (probability * target).sum(dim=(1, 2, 3)) + 1.0) / (
                probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
            )
            loss = bce + dice.mean()
            if model.category_aux_weight > 0.0:
                loss = loss + model.category_aux_weight * F.cross_entropy(
                    category_logits, batch["category"].to(device)
                )
            return loss
        if isinstance(model, AudioConditionedResNetCenterBoxHead):
            return _dense_localization_loss(model, batch, device)
        pred = model(batch["image"].to(device), batch["audio"].to(device))
        return F.smooth_l1_loss(pred, batch["bbox"].to(device))
    if task == "segmentation":
        logits = model(batch["image"].to(device), batch["audio"].to(device))
        target = batch["mask"].to(device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dice = 1.0 - (2.0 * (prob * target).sum(dim=(1, 2, 3)) + 1.0) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)
        return bce + dice.mean()
    if task == "category":
        logits = compute_category_logits(model, batch, device)
        target = batch["category"].to(device)
        if isinstance(model, BalancedPrototypeAudioCategoryHead):
            return F.cross_entropy(
                logits,
                target,
                weight=model.class_weights,
                label_smoothing=model.label_smoothing,
            )
        return F.cross_entropy(logits, target)
    pred = model(batch["image"].to(device))
    target = batch["audio"].to(device)
    return nn.functional.mse_loss(pred, target) + 0.25 * nn.functional.l1_loss(pred, target)


def audio_metric(task: str, model: nn.Module, batch: dict, device: torch.device) -> float:
    with torch.no_grad():
        if task == "localization":
            pred = model(batch["image"].to(device), batch["audio"].to(device))
            return float(_bbox_iou_torch(pred, batch["bbox"].to(device)).mean().detach().cpu())
        if task == "segmentation":
            logits = model(batch["image"].to(device), batch["audio"].to(device))
            pred = (torch.sigmoid(logits) > 0.5).float()
            target = batch["mask"].to(device).float()
            inter = (pred * target).sum(dim=(1, 2, 3))
            union = ((pred + target) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
            return float((inter / union).mean().detach().cpu())
        if task == "category":
            logits = compute_category_logits(model, batch, device)
            pred = logits.argmax(dim=1)
            return float((pred == batch["category"].to(device)).float().mean().detach().cpu())
    return float("nan")


def compute_val_stats(task: str, model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    metrics: list[float] = []
    with torch.no_grad():
        for batch in loader:
            losses.append(float(audio_loss(task, model, batch, device).detach().cpu()))
            metric = audio_metric(task, model, batch, device)
            if np.isfinite(metric):
                metrics.append(metric)
    model.train()
    return (float(np.mean(losses)) if losses else float("inf"), float(np.mean(metrics)) if metrics else float("nan"))


def _representative_visual_audio_batch(loader: DataLoader) -> dict:
    dataset = loader.dataset
    if not isinstance(dataset, AudioSparseDataset) or len(dataset) == 0:
        return next(iter(loader))
    order = np.argsort(dataset.audio_rms_dbfs)
    positions = [0.35, 0.6, 0.8, 0.95]
    indices = [int(order[min(len(order) - 1, max(0, round(pos * (len(order) - 1))))]) for pos in positions]
    samples = [dataset[index] for index in indices]
    return {
        "audio": torch.stack([sample["audio"] for sample in samples], dim=0),
        "bbox": torch.stack([sample["bbox"] for sample in samples], dim=0),
        "category": torch.stack([sample["category"] for sample in samples], dim=0),
        "audio_rms_dbfs": torch.stack([sample["audio_rms_dbfs"] for sample in samples], dim=0),
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "imgname": [sample["imgname"] for sample in samples],
    }


def _representative_category_batch(loader: DataLoader) -> dict:
    dataset = loader.dataset
    if not isinstance(dataset, AudioSparseDataset) or len(dataset) == 0:
        return next(iter(loader))
    categories = np.asarray(dataset.category)
    unique_categories = np.unique(categories)
    if unique_categories.size == 0:
        return next(iter(loader))
    positions = [0.0, 0.33, 0.66, 1.0]
    category_indices = [
        int(round(pos * (unique_categories.size - 1)))
        for pos in positions[: min(4, unique_categories.size)]
    ]
    selected: list[int] = []
    for category_index in category_indices:
        category = unique_categories[category_index]
        candidates = np.where(categories == category)[0]
        if candidates.size == 0:
            continue
        loudest = candidates[np.argmax(dataset.audio_rms_dbfs[candidates])]
        selected.append(int(loudest))
    if not selected:
        return next(iter(loader))
    samples = [dataset[index] for index in selected]
    batch = {
        "audio": torch.stack([sample["audio"] for sample in samples], dim=0),
        "bbox": torch.stack([sample["bbox"] for sample in samples], dim=0),
        "category": torch.stack([sample["category"] for sample in samples], dim=0),
        "audio_rms_dbfs": torch.stack([sample["audio_rms_dbfs"] for sample in samples], dim=0),
        "imgname": [sample["imgname"] for sample in samples],
        "wsv_path": [sample.get("wsv_path", "") for sample in samples],
        "frame_key": [sample.get("frame_key", "") for sample in samples],
    }
    if all("mask" in sample for sample in samples):
        batch["mask"] = torch.stack([sample["mask"] for sample in samples], dim=0)
    if all("image" in sample for sample in samples):
        batch["image"] = torch.stack([sample["image"] for sample in samples], dim=0)
    return batch




def _representative_audio_condition_batch(loader: DataLoader) -> dict:
    dataset = loader.dataset
    if not isinstance(dataset, AudioSparseDataset) or len(dataset) == 0:
        return next(iter(loader))
    categories = np.asarray(dataset.category)
    unique_categories = np.unique(categories)
    if unique_categories.size == 0:
        return next(iter(loader))
    positions = [0.0, 0.33, 0.66, 1.0]
    selected: list[int] = []
    used_categories: set[int] = set()
    for pos in positions[: min(4, unique_categories.size)]:
        category = int(unique_categories[int(round(pos * (unique_categories.size - 1)))])
        if category in used_categories:
            continue
        candidates = np.where(categories == category)[0]
        if candidates.size == 0:
            continue
        loudest = candidates[np.argmax(dataset.audio_rms_dbfs[candidates])]
        selected.append(int(loudest))
        used_categories.add(category)
    if len(selected) < min(4, len(dataset)):
        order = np.argsort(dataset.audio_rms_dbfs)[::-1]
        for index in order.tolist():
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= min(4, len(dataset)):
                break
    if not selected:
        return next(iter(loader))
    samples = [dataset[index] for index in selected[:4]]
    batch = {
        "audio": torch.stack([sample["audio"] for sample in samples], dim=0),
        "bbox": torch.stack([sample["bbox"] for sample in samples], dim=0),
        "category": torch.stack([sample["category"] for sample in samples], dim=0),
        "audio_rms_dbfs": torch.stack([sample["audio_rms_dbfs"] for sample in samples], dim=0),
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "imgname": [sample["imgname"] for sample in samples],
        "wsv_path": [sample.get("wsv_path", "") for sample in samples],
        "frame_key": [sample.get("frame_key", "") for sample in samples],
    }
    if all("mask" in sample for sample in samples):
        batch["mask"] = torch.stack([sample["mask"] for sample in samples], dim=0)
    return batch

def evaluate(task: str, model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, step: int) -> None:
    model.eval()
    if task == "visual_to_audio":
        batch = _representative_visual_audio_batch(loader)
    elif task == "category":
        batch = _representative_category_batch(loader)
    elif task in {"localization", "segmentation"}:
        batch = _representative_audio_condition_batch(loader)
    else:
        batch = next(iter(loader))
    with torch.no_grad():
        if task == "localization":
            pred = model(batch["image"].to(device), batch["audio"].to(device)).cpu().numpy()
            gt = batch["bbox"].numpy()
            categories = batch.get("category")
            category_np = categories.numpy() if categories is not None else [None] * pred.shape[0]
            rms = batch.get("audio_rms_dbfs")
            rms_np = rms.numpy() if rms is not None else [None] * pred.shape[0]
            for i in range(min(4, pred.shape[0])):
                out_path = out_dir / "visuals" / f"step_{step:06d}_sample_{i}.png"
                draw_localization(
                    batch["imgname"][i],
                    pred[i],
                    gt[i],
                    out_path,
                    category=int(category_np[i]) if categories is not None else None,
                    rms_dbfs=float(rms_np[i]) if rms is not None else None,
                )
                if "wsv_path" in batch and "frame_key" in batch:
                    write_category_audio_clip(
                        str(batch["wsv_path"][i]),
                        str(batch["frame_key"][i]),
                        out_path.with_name(out_path.stem + "_audio.wav"),
                    )
        elif task == "segmentation":
            logits = model(batch["image"].to(device), batch["audio"].to(device))
            pred = torch.sigmoid(logits).cpu().numpy()[:, 0]
            gt = batch["mask"].numpy()[:, 0]
            for i in range(min(4, pred.shape[0])):
                out_path = out_dir / "visuals" / f"step_{step:06d}_sample_{i}.png"
                draw_segmentation(batch["imgname"][i], pred[i], gt[i], out_path)
                if "wsv_path" in batch and "frame_key" in batch:
                    write_category_audio_clip(
                        str(batch["wsv_path"][i]),
                        str(batch["frame_key"][i]),
                        out_path.with_name(out_path.stem + "_audio.wav"),
                    )
        elif task == "category":
            logits = compute_category_logits(model, batch, device)
            pred = logits.argmax(dim=1).cpu().numpy()
            gt = batch["category"].numpy()
            masks = batch.get("mask")
            mask_np = masks.numpy()[:, 0] if masks is not None else [None] * pred.shape[0]
            rms = batch.get("audio_rms_dbfs")
            rms_np = rms.numpy() if rms is not None else [None] * pred.shape[0]
            for i in range(min(4, pred.shape[0])):
                out_path = out_dir / "visuals" / f"step_{step:06d}_sample_{i}.png"
                draw_category(
                    batch["imgname"][i],
                    int(pred[i]),
                    int(gt[i]),
                    out_path,
                    gt_mask=mask_np[i],
                    rms_dbfs=float(rms_np[i]) if rms is not None else None,
                )
                if "wsv_path" in batch and "frame_key" in batch:
                    write_category_audio_clip(
                        str(batch["wsv_path"][i]),
                        str(batch["frame_key"][i]),
                        out_path.with_name(out_path.stem + "_audio.wav"),
                    )
        else:
            pred = model(batch["image"].to(device)).cpu().numpy()
            gt = batch["audio"].numpy()
            rms = batch.get("audio_rms_dbfs")
            rms_np = rms.numpy() if rms is not None else [None] * pred.shape[0]
            for i in range(min(4, pred.shape[0])):
                draw_audio(
                    pred[i],
                    gt[i],
                    out_dir / "visuals" / f"step_{step:06d}_sample_{i}.png",
                    dataset=loader.dataset,
                    device=device,
                    img_path=batch["imgname"][i],
                    rms_dbfs=float(rms_np[i]) if rms is not None else None,
                )
    model.train()


def _balanced_category_weights(
    category: np.ndarray,
    num_classes: int,
    beta: float,
) -> tuple[torch.Tensor, np.ndarray]:
    labels = np.asarray(category, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes + 1).astype(np.float64)
    weights = np.zeros(num_classes + 1, dtype=np.float64)
    present = counts > 0
    safe_beta = min(max(float(beta), 0.0), 0.999999)
    if safe_beta > 0.0:
        weights[present] = (1.0 - safe_beta) / (1.0 - np.power(safe_beta, counts[present]))
    else:
        weights[present] = 1.0
    if present[1:].any():
        weights[present] /= weights[present].mean()
    weights[0] = 0.0
    sample_weights = np.zeros(labels.shape[0], dtype=np.float64)
    positive_counts = np.maximum(counts[labels], 1.0)
    sample_weights[:] = 1.0 / np.sqrt(positive_counts)
    return torch.from_numpy(weights.astype(np.float32)), sample_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", choices=["localization", "segmentation", "category", "visual_to_audio"], required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=500)
    parser.add_argument("--val-freq", type=int, default=100)
    parser.add_argument("--checkpoint-freq", type=int, default=500)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--clean-log-layout", action="store_true", help="Write ckpt/ and visuals/ directly under --output-dir.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-audio-rms-dbfs", type=float, default=None, help="Dynamically keep only samples at or above this RMS dBFS threshold.")
    parser.add_argument("--num-classes", type=int, default=26)
    parser.add_argument(
        "--localization-mode",
        choices=["legacy", "resnet_gaussian_film", "resnet_gaussian_grounded", "resnet_gaussian_crossmodal", "resnet_mask_bbox", "resnet_joint_mask_box", "clip_resnet_crossmodal", "clip_resnet_joint_mask"],
        default="legacy",
    )
    parser.add_argument("--bbox-init-checkpoint", default="")
    parser.add_argument("--adapter-init-checkpoint", default="")
    parser.add_argument("--mask-init-checkpoint", default="")
    parser.add_argument("--freeze-inherited-localizer", action="store_true")
    parser.add_argument("--freeze-clip-base", action="store_true")
    parser.add_argument("--clip-train-dense-head", action="store_true")
    parser.add_argument("--freeze-mask-conditioner", action="store_true")
    parser.add_argument("--mask-target-mode", choices=["source", "box", "gaussian"], default="source")
    parser.add_argument("--grounding-weight", type=float, default=0.35)
    parser.add_argument("--grounding-margin", type=float, default=0.5)
    parser.add_argument("--category-aux-weight", type=float, default=0.15)
    parser.add_argument("--crossmodal-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--crossmodal-pixel-weight", type=float, default=0.0)
    parser.add_argument("--crossmodal-temperature", type=float, default=0.10)
    parser.add_argument("--mask-loss-weight", type=float, default=0.50)
    parser.add_argument("--fusion-cross-weight", type=float, default=0.90)
    parser.add_argument("--fusion-learned", action="store_true")
    parser.add_argument("--fused-box-loss-weight", type=float, default=1.0)
    parser.add_argument("--fusion-residual-scale", type=float, default=0.0)
    parser.add_argument("--freeze-joint-branches", action="store_true")
    parser.add_argument("--mask-threshold", type=float, default=0.57)
    parser.add_argument(
        "--category-mode",
        choices=["mlp", "prototype_balanced", "prototype_temporal_attention", "visual_audio_fusion"],
        default="mlp",
    )
    parser.add_argument("--category-balanced-sampling", action="store_true")
    parser.add_argument("--video-balanced-sampling", action="store_true")
    parser.add_argument("--category-effective-beta", type=float, default=0.999)
    parser.add_argument("--category-label-smoothing", type=float, default=0.05)
    parser.add_argument("--category-dropout", type=float, default=0.30)
    parser.add_argument(
        "--category-visual-crop",
        action="store_true",
        help="Crop the RGB category input to the annotated object; at inference use the generated-mask box.",
    )
    parser.add_argument("--temporal-radius", type=int, default=0)
    parser.add_argument("--temporal-center-weight", type=float, default=0.0)
    parser.add_argument(
        "--validate-before-training",
        action="store_true",
        help="Evaluate and preserve the initialized model as the step-0 best candidate before fine-tuning.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many validations without metric improvement; zero disables it.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_set = AudioSparseDataset(
        args.train_dataset,
        args.image_size,
        args.task,
        args.min_audio_rms_dbfs,
        temporal_radius=args.temporal_radius if args.task == "category" else 0,
        temporal_center_weight=args.temporal_center_weight if args.task == "category" else 0.0,
        temporal_return_sequence=args.task == "category" and args.category_mode == "prototype_temporal_attention",
        category_load_image=args.task == "category" and args.category_mode == "visual_audio_fusion",
        category_image_crop=args.category_visual_crop,
    )
    val_set = AudioSparseDataset(
        args.val_dataset,
        args.image_size,
        args.task,
        args.min_audio_rms_dbfs,
        temporal_radius=args.temporal_radius if args.task == "category" else 0,
        temporal_center_weight=args.temporal_center_weight if args.task == "category" else 0.0,
        temporal_return_sequence=args.task == "category" and args.category_mode == "prototype_temporal_attention",
        category_load_image=args.task == "category" and args.category_mode == "visual_audio_fusion",
        category_image_crop=args.category_visual_crop,
    )
    class_weights, sample_weights = _balanced_category_weights(
        train_set.category,
        args.num_classes,
        args.category_effective_beta,
    )
    sampler = None
    shuffle = True
    sampling_weights = np.ones(len(train_set), dtype=np.float64)
    use_weighted_sampler = False
    if args.task == "category" and args.category_balanced_sampling:
        sampling_weights *= sample_weights
        use_weighted_sampler = True
        print("[audio:category] using inverse-sqrt class-balanced sampling", flush=True)
    if args.task == "category" and args.video_balanced_sampling:
        _video_names, video_inverse, video_counts = np.unique(
            train_set.wsv_path,
            return_inverse=True,
            return_counts=True,
        )
        sampling_weights /= np.maximum(video_counts[video_inverse], 1)
        use_weighted_sampler = True
        print(
            f"[audio:category] using video-balanced sampling across {len(video_counts)} videos",
            flush=True,
        )
    if use_weighted_sampler:
        generator = torch.Generator().manual_seed(0)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sampling_weights),
            num_samples=len(train_set),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    audio_dim = train_set.audio_feat.shape[1]
    if args.task == "localization":
        if args.localization_mode == "legacy":
            model: nn.Module = AudioLocalizationHead(audio_dim)
        elif args.localization_mode == "resnet_mask_bbox":
            model = AudioMaskToBoxLocalizationHead(
                audio_dim=audio_dim,
                num_classes=args.num_classes,
                init_checkpoint=args.bbox_init_checkpoint,
                category_aux_weight=args.category_aux_weight,
                mask_target_mode=args.mask_target_mode,
            )
        elif args.localization_mode == "resnet_joint_mask_box":
            model = AudioJointMaskBoxLocalizationHead(
                audio_dim=audio_dim,
                num_classes=args.num_classes,
                init_checkpoint=args.bbox_init_checkpoint,
                category_aux_weight=args.category_aux_weight,
                grounding_weight=args.grounding_weight,
                grounding_margin=args.grounding_margin,
                crossmodal_contrastive_weight=args.crossmodal_contrastive_weight,
                crossmodal_pixel_weight=args.crossmodal_pixel_weight,
                crossmodal_temperature=args.crossmodal_temperature,
                mask_loss_weight=args.mask_loss_weight,
                fusion_cross_weight=args.fusion_cross_weight,
                fusion_learned=args.fusion_learned,
                fused_box_loss_weight=args.fused_box_loss_weight,
                fusion_residual_scale=args.fusion_residual_scale,
                mask_threshold=args.mask_threshold,
            )
        elif args.localization_mode == "clip_resnet_joint_mask":
            model = AudioCLIPJointMaskBoxLocalizationHead(
                audio_dim=audio_dim,
                num_classes=args.num_classes,
                init_checkpoint=args.bbox_init_checkpoint,
                mask_init_checkpoint=args.mask_init_checkpoint,
                fusion_cross_weight=args.fusion_cross_weight,
                mask_threshold=args.mask_threshold,
            )
        elif args.localization_mode == "clip_resnet_crossmodal":
            model = AudioCLIPResNetCenterBoxHead(
                audio_dim=audio_dim,
                num_classes=args.num_classes,
                init_checkpoint=args.bbox_init_checkpoint,
                category_aux_weight=args.category_aux_weight,
                crossmodal_contrastive_weight=args.crossmodal_contrastive_weight,
                crossmodal_pixel_weight=args.crossmodal_pixel_weight,
                crossmodal_temperature=args.crossmodal_temperature,
            )
        else:
            grounding_weight = args.grounding_weight if args.localization_mode in {
                "resnet_gaussian_grounded", "resnet_gaussian_crossmodal"
            } else 0.0
            model = AudioConditionedResNetCenterBoxHead(
                audio_dim=audio_dim,
                num_classes=args.num_classes,
                init_checkpoint=args.bbox_init_checkpoint,
                grounding_weight=grounding_weight,
                grounding_margin=args.grounding_margin,
                category_aux_weight=args.category_aux_weight,
                crossmodal_matching=args.localization_mode == "resnet_gaussian_crossmodal",
                crossmodal_contrastive_weight=args.crossmodal_contrastive_weight,
                crossmodal_pixel_weight=args.crossmodal_pixel_weight,
                crossmodal_temperature=args.crossmodal_temperature,
            )
    elif args.task == "segmentation":
        model = AudioSegmentationHead(audio_dim)
    elif args.task == "category":
        if args.category_mode == "visual_audio_fusion":
            model = AudioVisualFusionCategoryHead(
                audio_dim,
                args.num_classes,
                class_weights,
                label_smoothing=args.category_label_smoothing,
                dropout=args.category_dropout,
            )
        elif args.category_mode == "prototype_balanced":
            model = BalancedPrototypeAudioCategoryHead(
                audio_dim,
                args.num_classes,
                class_weights,
                label_smoothing=args.category_label_smoothing,
                dropout=args.category_dropout,
            )
            model.initialize_prototypes(train_set.audio_feat, train_set.category)
        elif args.category_mode == "prototype_temporal_attention":
            if args.temporal_radius <= 0:
                raise ValueError("--category-mode prototype_temporal_attention requires --temporal-radius > 0")
            model = TemporalPrototypeAudioCategoryHead(
                audio_dim,
                args.num_classes,
                class_weights,
                temporal_length=2 * args.temporal_radius + 1,
                label_smoothing=args.category_label_smoothing,
                dropout=args.category_dropout,
            )
            model.initialize_prototypes(train_set.audio_feat, train_set.category)
        else:
            model = AudioCategoryHead(audio_dim, args.num_classes)
    else:
        model = VisualToAudioHead(audio_dim)
    if args.adapter_init_checkpoint:
        checkpoint = torch.load(args.adapter_init_checkpoint, map_location="cpu", weights_only=False)
        source_state = checkpoint.get("model", checkpoint)
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in source_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)
        print(
            f"[audio:{args.task}] initialized {len(compatible_state)} adapter tensors from "
            f"{args.adapter_init_checkpoint} (step={checkpoint.get('step', 'unknown')})",
            flush=True,
        )

    if args.mask_init_checkpoint and not isinstance(model, AudioCLIPJointMaskBoxLocalizationHead):
        if not isinstance(model, AudioJointMaskBoxLocalizationHead):
            raise ValueError("--mask-init-checkpoint requires --localization-mode resnet_joint_mask_box")
        checkpoint = torch.load(args.mask_init_checkpoint, map_location="cpu", weights_only=False)
        source_state = checkpoint.get("model", checkpoint)
        current_state = model.state_dict()
        mask_state = {
            key: value
            for key, value in source_state.items()
            if key.startswith("mask_decoder.")
            and key in current_state
            and current_state[key].shape == value.shape
        }
        if not mask_state:
            raise RuntimeError(f"No compatible mask decoder tensors in {args.mask_init_checkpoint}")
        model.load_state_dict(mask_state, strict=False)
        print(
            f"[audio:{args.task}] initialized {len(mask_state)} mask tensors from "
            f"{args.mask_init_checkpoint} (step={checkpoint.get(chr(115) + chr(116) + chr(101) + chr(112), chr(117) + chr(110) + chr(107) + chr(110) + chr(111) + chr(119) + chr(110))})",
            flush=True,
        )


    if args.freeze_joint_branches:
        if not isinstance(model, AudioJointMaskBoxLocalizationHead) or not model.fusion_learned:
            raise ValueError("--freeze-joint-branches requires learned joint fusion")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.fusion_gate.parameters():
            parameter.requires_grad_(True)
        for parameter in model.fusion_residual.parameters():
            parameter.requires_grad_(True)
        model.branches_frozen = True
        print(
            f"[audio:{args.task}] froze dense/mask branches; training adaptive fusion gate only",
            flush=True,
        )

    if args.freeze_clip_base:
        if not isinstance(model, AudioCLIPResNetCenterBoxHead):
            raise ValueError("--freeze-clip-base requires --localization-mode clip_resnet_crossmodal")
        trainable_prefixes = ("clip_projection.", "clip_scale")
        if args.clip_train_dense_head:
            trainable_prefixes = trainable_prefixes + ("dense_head.",)
        trainable_names = []
        for name, parameter in model.named_parameters():
            keep_trainable = name.startswith(trainable_prefixes)
            parameter.requires_grad_(keep_trainable)
            if keep_trainable:
                trainable_names.append(name)
        print(
            f"[audio:{args.task}] froze ResNet/audio base; training {len(trainable_names)} CLIP fusion tensors",
            flush=True,
        )
    if args.freeze_mask_conditioner:
        if not isinstance(model, AudioMaskToBoxLocalizationHead):
            raise ValueError("--freeze-mask-conditioner requires --localization-mode resnet_mask_bbox")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.mask_decoder.parameters():
            parameter.requires_grad_(True)
        print(f"[audio:{args.task}] froze inherited audio/visual conditioner; training mask decoder only", flush=True)
    if args.freeze_inherited_localizer:
        trainable_prefixes = ("visual_projection.", "audio_projection.", "crossmodal_gate")
        trainable_names: list[str] = []
        for name, parameter in model.named_parameters():
            keep_trainable = name.startswith(trainable_prefixes)
            parameter.requires_grad_(keep_trainable)
            if keep_trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError("--freeze-inherited-localizer found no cross-modal adapter parameters")
        print(
            f"[audio:{args.task}] froze inherited localizer; training {len(trainable_names)} cross-modal tensors",
            flush=True,
        )
    model = model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    output_root = Path(args.output_dir)
    if args.clean_log_layout:
        ckpt_dir = output_root / "ckpt"
        out_dir = output_root
    else:
        ckpt_dir = output_root / "checkpoints" / args.name
        out_dir = output_root / "visuals" / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricLogger(output_root)
    best_val_loss = float("inf")
    best_val_metric = float("-inf")
    validations_without_improvement = 0
    stop_training = False

    def checkpoint_payload(current_step: int) -> dict:
        return {
            "step": current_step,
            "model": model.state_dict(),
            "task": args.task,
            "audio_dim": audio_dim,
            "best_val_loss": best_val_loss,
            "best_val_metric": best_val_metric,
            "localization_mode": args.localization_mode,
            "category_mode": args.category_mode,
            "category_visual_crop": args.category_visual_crop,
            "category_balanced_sampling": args.category_balanced_sampling,
            "temporal_radius": args.temporal_radius,
            "temporal_center_weight": args.temporal_center_weight,
            "video_balanced_sampling": args.video_balanced_sampling,
            "validate_before_training": args.validate_before_training,
            "mask_target_mode": args.mask_target_mode,
            "crossmodal_pixel_weight": args.crossmodal_pixel_weight,
            "mask_loss_weight": args.mask_loss_weight,
            "fusion_cross_weight": args.fusion_cross_weight,
            "fusion_learned": args.fusion_learned,
            "fused_box_loss_weight": args.fused_box_loss_weight,
            "fusion_residual_scale": args.fusion_residual_scale,
            "freeze_joint_branches": args.freeze_joint_branches,
            "freeze_clip_base": args.freeze_clip_base,
            "clip_train_dense_head": args.clip_train_dense_head,
            "mask_threshold": args.mask_threshold,
        }

    if args.validate_before_training:
        initial_val_loss, initial_val_metric = compute_val_stats(
            args.task,
            model,
            val_loader,
            device,
        )
        best_val_loss = initial_val_loss
        if np.isfinite(initial_val_metric):
            best_val_metric = initial_val_metric
        initial_payload = checkpoint_payload(0)
        initial_payload.update(
            {
                "train_loss": float("nan"),
                "val_loss": initial_val_loss,
                "val_metric": initial_val_metric,
                "initialized_best": True,
            }
        )
        torch.save(initial_payload, ckpt_dir / "model-best.pt")
        metrics.append(0, float("nan"), initial_val_loss, float("nan"), initial_val_metric)
        evaluate(args.task, model, val_loader, device, out_dir, 0)
        print(
            f"[audio:{args.task}] preserved initialized model at step=0 "
            f"val_loss={initial_val_loss:.10e} val_metric={initial_val_metric:.6f}",
            flush=True,
        )

    step = 0
    progress = tqdm(total=args.total_steps, desc=f"train audio {args.task}")
    while step < args.total_steps and not stop_training:
        for batch in train_loader:
            loss = audio_loss(args.task, model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            progress.update(1)
            progress.set_description(f"{args.task}_loss: {loss.item():.4f}")
            if step % args.val_freq == 0:
                train_loss = float(loss.detach().cpu())
                train_metric = audio_metric(args.task, model, batch, device)
                val_loss, val_metric = compute_val_stats(args.task, model, val_loader, device)
                print(
                    f"[audio:{args.task}] step={step} train_loss={train_loss:.10e} val_loss={val_loss:.10e} "
                    f"train_metric={train_metric:.6f} val_metric={val_metric:.6f}",
                    flush=True,
                )
                metrics.append(step, train_loss, val_loss, train_metric, val_metric)
                best_val_loss = min(best_val_loss, val_loss)
                if np.isfinite(val_metric):
                    improved = val_metric > best_val_metric + 1e-8
                else:
                    improved = val_loss <= best_val_loss
                if improved:
                    best_val_metric = val_metric if np.isfinite(val_metric) else best_val_metric
                    validations_without_improvement = 0
                    payload = checkpoint_payload(step)
                    payload.update({"train_loss": train_loss, "val_loss": val_loss, "val_metric": val_metric})
                    torch.save(payload, ckpt_dir / "model-best.pt")
                else:
                    validations_without_improvement += 1
                if step % args.log_freq == 0:
                    evaluate(args.task, model, val_loader, device, out_dir, step)
                if args.early_stop_patience > 0 and validations_without_improvement >= args.early_stop_patience:
                    print(
                        f"[audio:{args.task}] early stop at step={step}; no validation metric improvement "
                        f"for {validations_without_improvement} checks",
                        flush=True,
                    )
                    stop_training = True
            if step % args.checkpoint_freq == 0:
                torch.save(checkpoint_payload(step), ckpt_dir / "model-latest.pt")
            if step >= args.total_steps or stop_training:
                break

    progress.close()
    torch.save(checkpoint_payload(step), ckpt_dir / "model-last.pt")
    if args.clean_log_layout:
        print(f"[audio:{args.task}] finished; outputs under {args.output_dir}/ckpt and {args.output_dir}/visuals", flush=True)
    else:
        print(f"[audio:{args.task}] finished; outputs under {args.output_dir}/checkpoints/{args.name} and {args.output_dir}/visuals/{args.name}", flush=True)


if __name__ == "__main__":
    main()
