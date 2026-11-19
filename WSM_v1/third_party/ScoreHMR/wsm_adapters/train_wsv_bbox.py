from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - ScoreHMR normally installs scipy.
    linear_sum_assignment = None


class MetricLogger:
    def __init__(self, out_dir: Path) -> None:
        self.csv_path = out_dir / "metrics.csv"
        self.stats_path = out_dir / "stats.png"
        self.rows: list[dict[str, float]] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.csv_path.write_text(
                "step,train_loss,val_loss,val_precision,val_recall,val_f1,"
                "val_count_mae,val_mean_confidence,val_ece\n"
            )

    def append(self, step: int, train_loss: float, metrics: dict[str, float]) -> None:
        row = {
            "step": int(step),
            "train_loss": float(train_loss),
            **{key: float(value) for key, value in metrics.items()},
        }
        self.rows.append(row)
        with self.csv_path.open("a") as f:
            f.write(
                f"{row['step']},{row['train_loss']:.8f},{row['loss']:.8f},"
                f"{row['precision']:.8f},{row['recall']:.8f},{row['f1']:.8f},"
                f"{row['count_mae']:.8f},{row['mean_confidence']:.8f},{row['ece']:.8f}\n"
            )
        self.plot()

    def plot(self) -> None:
        if not self.rows:
            return
        steps = [row["step"] for row in self.rows]
        train = [row["train_loss"] for row in self.rows]
        val = [row["loss"] for row in self.rows]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, train, label="train_loss")
        plt.plot(steps, val, label="val_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.stats_path)
        plt.close()


class BBoxDataset(Dataset):
    def __init__(self, dataset_file: str, rgb_cache_file: str) -> None:
        self.data = np.load(dataset_file)
        self.cache = np.load(rgb_cache_file)
        self.images = self.cache["rgb_ch0_2_full"]
        self.boxes = self.data["boxes"].astype(np.float32)
        self.has_box = self.data["has_box"].astype(np.float32)
        self.imgname = self.data["imgname"].astype(str)

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.from_numpy(self.images[index].astype(np.float32)),
            "boxes": torch.from_numpy(self.boxes[index]),
            "has_box": torch.from_numpy(self.has_box[index]),
            "imgname": self.imgname[index],
        }


class ParallelQueryBoxHead(nn.Module):
    """LocateAnything-style parallel box decoder.

    A fixed set of learnable box queries attends to image tokens. Each query
    predicts one whole box atomically: objectness + cx,cy,w,h. This keeps the
    four coordinates coupled inside a single query head instead of flattening
    the entire image into slot-wise linear outputs.
    """

    def __init__(
        self,
        max_slots: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        use_count_head: bool = False,
    ) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.hidden_dim = hidden_dim
        self.use_count_head = use_count_head
        self.stem = nn.Sequential(
            nn.Conv2d(5, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 160, 3, stride=2, padding=1),
            nn.GroupNorm(8, 160),
            nn.SiLU(),
            nn.Conv2d(160, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, hidden_dim))
        self.query_embed = nn.Parameter(torch.randn(1, max_slots, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                        "self_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                        "norm_q1": nn.LayerNorm(hidden_dim),
                        "norm_q2": nn.LayerNorm(hidden_dim),
                        "norm_q3": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.count_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, max_slots + 1),
            )
            if use_count_head
            else None
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, boxes, _count_logits = self.forward_with_aux(image)
        return logits, boxes

    def forward_with_aux(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, _, height, width = image.shape
        ys = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        feats = self.stem(torch.cat((image, coord), dim=1))
        tokens = feats.flatten(2).transpose(1, 2) + self.pos_embed.to(dtype=feats.dtype)
        queries = self.query_embed.to(dtype=feats.dtype).expand(batch, -1, -1)

        for layer in self.layers:
            cross, _ = layer["cross_attn"](queries, tokens, tokens, need_weights=False)
            queries = layer["norm_q1"](queries + cross)
            self_out, _ = layer["self_attn"](queries, queries, queries, need_weights=False)
            queries = layer["norm_q2"](queries + self_out)
            queries = layer["norm_q3"](queries + layer["ffn"](queries))

        logits = self.cls_head(queries).squeeze(-1)
        boxes = self.box_head(queries).sigmoid()
        count_logits = self.count_head(tokens.mean(dim=1)) if self.count_head is not None else None
        return logits, boxes, count_logits


class DenseCenterBoxHead(nn.Module):
    """Anchor-free dense center detector with an explicit person count head."""

    def __init__(self, max_count: int, hidden_dim: int = 256, grid_size: int = 14) -> None:
        super().__init__()
        self.max_count = max_count
        self.grid_size = grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(5, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 160, 3, stride=2, padding=1),
            nn.GroupNorm(8, 160),
            nn.SiLU(),
            nn.Conv2d(160, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        # Per cell: center score, sub-cell x/y offsets, width and height.
        self.dense_head = nn.Conv2d(hidden_dim, 5, 1)
        self.count_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_count + 1),
        )
        with torch.no_grad():
            self.dense_head.bias[0] = -3.0

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, boxes, _count_logits = self.forward_with_aux(image)
        return logits, boxes

    def forward_with_aux(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = image.shape
        ys = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        feats = self.stem(torch.cat((image, coord), dim=1))
        dense = self.dense_head(feats)
        logits = dense[:, 0].flatten(1)
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
        boxes = boxes.reshape(batch, self.grid_size * self.grid_size, 4)
        count_logits = self.count_head(feats.mean(dim=(2, 3)))
        return logits, boxes, count_logits


class PretrainedResNetCenterBoxHead(nn.Module):
    """Dense center detector on frozen ImageNet ResNet-18 layer-3 features."""

    def __init__(
        self,
        max_count: int,
        grid_size: int = 14,
        load_pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        # Training starts from ImageNet weights. Deployment checkpoints already
        # contain the complete backbone, so inference can construct the module
        # without a network download and restore the saved tensors below.
        weights = ResNet18_Weights.DEFAULT if load_pretrained_backbone else None
        network = resnet18(weights=weights)
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
        self.grid_size = grid_size
        self.dense_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 5, 1),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, max_count + 1),
        )
        with torch.no_grad():
            self.dense_head[-1].bias[0] = -3.0
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, boxes, _count_logits = self.forward_with_aux(image)
        return logits, boxes

    def forward_with_aux(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Frozen batch-normalization statistics are part of the pretrained prior.
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone((image - self.imagenet_mean) / self.imagenet_std)
        dense = self.dense_head(feats)
        logits = dense[:, 0].flatten(1)
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
        count_logits = self.count_head(feats.mean(dim=(2, 3)))
        return logits, boxes, count_logits


# Backward-compatible alias for old checkpoints/imports.
TinyFullImageBoxHead = ParallelQueryBoxHead

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def _cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [boxes[:, i] for i in range(4)]
    return np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.35, max_keep: int | None = None) -> list[int]:
    if boxes.size == 0 or scores.size == 0:
        return []
    xyxy = np.clip(_cxcywh_to_xyxy_np(boxes), 0.0, 1.0)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if max_keep is not None and len(keep) >= max_keep:
            break
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[idx, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[idx, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[idx, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[idx, 3], xyxy[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_idx = np.maximum(0.0, xyxy[idx, 2] - xyxy[idx, 0]) * np.maximum(0.0, xyxy[idx, 3] - xyxy[idx, 1])
        area_rest = np.maximum(0.0, xyxy[rest, 2] - xyxy[rest, 0]) * np.maximum(0.0, xyxy[rest, 3] - xyxy[rest, 1])
        union = np.maximum(1e-6, area_idx + area_rest - inter)
        iou = inter / union
        order = rest[iou <= iou_threshold]
    return keep


def _generalized_iou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    pred = _cxcywh_to_xyxy(pred_boxes).clamp(0.0, 1.0)
    gt = _cxcywh_to_xyxy(gt_boxes).clamp(0.0, 1.0)

    inter_x1 = torch.maximum(pred[:, 0], gt[:, 0])
    inter_y1 = torch.maximum(pred[:, 1], gt[:, 1])
    inter_x2 = torch.minimum(pred[:, 2], gt[:, 2])
    inter_y2 = torch.minimum(pred[:, 3], gt[:, 3])
    inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)

    area_pred = (pred[:, 2] - pred[:, 0]).clamp_min(0) * (pred[:, 3] - pred[:, 1]).clamp_min(0)
    area_gt = (gt[:, 2] - gt[:, 0]).clamp_min(0) * (gt[:, 3] - gt[:, 1]).clamp_min(0)
    union = (area_pred + area_gt - inter).clamp_min(1e-6)
    iou = inter / union

    enc_x1 = torch.minimum(pred[:, 0], gt[:, 0])
    enc_y1 = torch.minimum(pred[:, 1], gt[:, 1])
    enc_x2 = torch.maximum(pred[:, 2], gt[:, 2])
    enc_y2 = torch.maximum(pred[:, 3], gt[:, 3])
    enc_area = ((enc_x2 - enc_x1).clamp_min(0) * (enc_y2 - enc_y1).clamp_min(0)).clamp_min(1e-6)
    giou = iou - (enc_area - union) / enc_area
    return (1.0 - giou).mean()


def _box_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    a = _cxcywh_to_xyxy(boxes_a).clamp(0.0, 1.0)
    b = _cxcywh_to_xyxy(boxes_b).clamp(0.0, 1.0)
    inter_x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    union = (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-6)
    return inter / union


def _box_iou_matrix_batched(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    a = _cxcywh_to_xyxy(boxes_a).clamp(0.0, 1.0)
    b = _cxcywh_to_xyxy(boxes_b).clamp(0.0, 1.0)
    inter_x1 = torch.maximum(a[:, :, None, 0], b[:, None, :, 0])
    inter_y1 = torch.maximum(a[:, :, None, 1], b[:, None, :, 1])
    inter_x2 = torch.minimum(a[:, :, None, 2], b[:, None, :, 2])
    inter_y2 = torch.minimum(a[:, :, None, 3], b[:, None, :, 3])
    inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    area_a = (a[:, :, 2] - a[:, :, 0]).clamp_min(0) * (a[:, :, 3] - a[:, :, 1]).clamp_min(0)
    area_b = (b[:, :, 2] - b[:, :, 0]).clamp_min(0) * (b[:, :, 3] - b[:, :, 1]).clamp_min(0)
    union = (area_a[:, :, None] + area_b[:, None, :] - inter).clamp_min(1e-6)
    return inter / union


def _greedy_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One-to-one greedy assignment for tiny bbox slot counts."""
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    cost_work = cost.clone()
    pred_indices: list[torch.Tensor] = []
    gt_indices: list[torch.Tensor] = []
    for _ in range(min(cost.shape)):
        flat_index = torch.argmin(cost_work)
        pred_idx = flat_index // cost_work.shape[1]
        gt_idx = flat_index % cost_work.shape[1]
        if not torch.isfinite(cost_work[pred_idx, gt_idx]):
            break
        pred_indices.append(pred_idx)
        gt_indices.append(gt_idx)
        cost_work[pred_idx, :] = float("inf")
        cost_work[:, gt_idx] = float("inf")
    if not pred_indices:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    return torch.stack(pred_indices).long(), torch.stack(gt_indices).long()


def _exact_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact bipartite assignment, with the old greedy matcher as a fallback."""
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    if linear_sum_assignment is None:
        return _greedy_match(cost)
    pred_np, gt_np = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(pred_np, dtype=torch.long, device=cost.device),
        torch.as_tensor(gt_np, dtype=torch.long, device=cost.device),
    )


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    normalize_by_positives: bool = False,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if normalize_by_positives:
        return loss.sum() / targets.sum().clamp_min(1.0)
    return loss.mean()


def _centernet_focal_loss(
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


def loss_fn(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    has_box: torch.Tensor,
    loss_mode: str = "set_focal_cardinality",
    count_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    target_has = torch.zeros_like(logits)
    target_quality = torch.zeros_like(logits)
    target_heatmap = torch.zeros_like(logits)
    matched_pred: list[torch.Tensor] = []
    matched_gt: list[torch.Tensor] = []

    with torch.no_grad():
        l1_cost_all = torch.cdist(boxes.detach(), gt_boxes.detach(), p=1)
        iou_cost_all = 1.0 - _box_iou_matrix_batched(boxes.detach(), gt_boxes.detach())
        match_cost_all = 5.0 * l1_cost_all + 2.0 * iou_cost_all

    for batch_idx in range(boxes.shape[0]):
        valid_gt = torch.nonzero(has_box[batch_idx] > 0.5, as_tuple=False).flatten()
        if valid_gt.numel() == 0:
            continue
        gt = gt_boxes[batch_idx, valid_gt]
        with torch.no_grad():
            if loss_mode in {"ordered_count_head", "ordered_count_quality"}:
                # Stable left-to-right person identities remove permutation
                # jitter while the count head selects the valid leading slots.
                gt_idx = torch.argsort(gt[:, 0])
                pred_idx = torch.arange(gt_idx.numel(), device=boxes.device)
            elif loss_mode in {
                "dense_center_count", "dense_center_gaussian", "pretrained_center_gaussian"
            }:
                grid_size = int(round(logits.shape[1] ** 0.5))
                cell_x = (gt[:, 0] * grid_size).long().clamp(0, grid_size - 1)
                cell_y = (gt[:, 1] * grid_size).long().clamp(0, grid_size - 1)
                pred_idx = cell_y * grid_size + cell_x
                gt_idx = torch.arange(gt.shape[0], device=boxes.device)
                # Extremely close people may share a cell; preserve both via
                # exact matching instead of supervising one cell twice.
                if torch.unique(pred_idx).numel() != pred_idx.numel():
                    pred_idx, gt_idx = _exact_match(match_cost_all[batch_idx, :, valid_gt])
            else:
                pred_idx, gt_idx = _exact_match(match_cost_all[batch_idx, :, valid_gt])
        if pred_idx.numel() == 0:
            continue
        target_has[batch_idx, pred_idx] = 1.0
        if loss_mode in {"dense_center_gaussian", "pretrained_center_gaussian"}:
            grid_size = int(round(logits.shape[1] ** 0.5))
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype) + 0.5) / grid_size,
                (torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype) + 0.5) / grid_size,
                indexing="ij",
            )
            gaussian = torch.zeros_like(grid_x)
            for gt_box in gt:
                sigma_x = torch.clamp(gt_box[2] / 6.0, min=1.0 / grid_size)
                sigma_y = torch.clamp(gt_box[3] / 6.0, min=1.0 / grid_size)
                current = torch.exp(
                    -0.5 * (((grid_x - gt_box[0]) / sigma_x).pow(2.0)
                            + ((grid_y - gt_box[1]) / sigma_y).pow(2.0))
                )
                gaussian = torch.maximum(gaussian, current)
            target_heatmap[batch_idx] = gaussian.flatten()
            target_heatmap[batch_idx, pred_idx] = 1.0
        matched_pred.append(boxes[batch_idx, pred_idx])
        matched_gt.append(gt[gt_idx])
        with torch.no_grad():
            matched_iou = _box_iou_matrix(boxes[batch_idx, pred_idx], gt[gt_idx]).diag()
            # A matched but still inaccurate box should not claim confidence 1.
            # The 0.5 floor keeps the target distinguishable from no-object.
            target_quality[batch_idx, pred_idx] = 0.5 + 0.5 * matched_iou.clamp(0.0, 1.0)

    max_count = count_logits.shape[-1] - 1 if count_logits is not None else logits.shape[1]
    gt_count = has_box.sum(dim=1).clamp(0, max_count)
    probabilities = logits.sigmoid()
    if loss_mode == "baseline":
        pos_weight = torch.full((), 3.0, dtype=logits.dtype, device=logits.device)
        cls_loss = F.binary_cross_entropy_with_logits(logits, target_has, pos_weight=pos_weight)
        count_loss = logits.sum() * 0.0
    elif loss_mode == "set_focal_cardinality":
        cls_loss = _sigmoid_focal_loss(logits, target_has)
        count_loss = F.smooth_l1_loss(probabilities.sum(dim=1), gt_count)
    elif loss_mode == "quality_focal":
        quality_bce = F.binary_cross_entropy_with_logits(logits, target_quality, reduction="none")
        quality_weight = (target_quality - probabilities).abs().pow(2.0)
        positive_weight = torch.where(target_has > 0.0, 2.0, 1.0)
        cls_loss = (quality_bce * quality_weight * positive_weight).mean()
        count_loss = F.smooth_l1_loss(probabilities.sum(dim=1), gt_count)
    elif loss_mode in {"count_head", "ordered_count_head"}:
        if count_logits is None:
            raise ValueError(f"{loss_mode} loss requires count_logits")
        cls_loss = _sigmoid_focal_loss(logits, target_has)
        count_loss = F.cross_entropy(count_logits, gt_count.long())
    elif loss_mode in {"dense_center_gaussian", "pretrained_center_gaussian"}:
        if count_logits is None:
            raise ValueError(f"{loss_mode} loss requires count_logits")
        cls_loss = _centernet_focal_loss(logits, target_heatmap, target_has)
        count_loss = F.cross_entropy(count_logits, gt_count.long())
    elif loss_mode == "dense_center_count":
        if count_logits is None:
            raise ValueError("dense_center_count loss requires count_logits")
        cls_loss = _sigmoid_focal_loss(
            logits, target_has, normalize_by_positives=True
        )
        count_loss = F.cross_entropy(count_logits, gt_count.long())
    elif loss_mode in {"count_head_quality", "ordered_count_quality"}:
        if count_logits is None:
            raise ValueError("count_head_quality loss requires count_logits")
        quality_bce = F.binary_cross_entropy_with_logits(logits, target_quality, reduction="none")
        quality_weight = (target_quality - probabilities).abs().pow(2.0)
        positive_weight = torch.where(target_has > 0.0, 2.0, 1.0)
        cls_loss = (quality_bce * quality_weight * positive_weight).mean()
        count_loss = F.cross_entropy(count_logits, gt_count.long())
    else:
        raise ValueError(f"Unsupported bbox loss mode: {loss_mode}")

    if not matched_pred:
        return cls_loss + count_loss + boxes.sum() * 0.0

    pred = torch.cat(matched_pred, dim=0)
    gt = torch.cat(matched_gt, dim=0)
    l1_loss = F.l1_loss(pred, gt)
    giou_loss = _generalized_iou_loss(pred, gt)
    pred_area = (pred[:, 2].clamp_min(1e-4) * pred[:, 3].clamp_min(1e-4)).clamp_min(1e-6)
    gt_area = (gt[:, 2].clamp_min(1e-4) * gt[:, 3].clamp_min(1e-4)).clamp_min(1e-6)
    area_loss = (pred_area.log() - gt_area.log()).abs().mean()
    pred_iou = _box_iou_matrix_batched(boxes, boxes)
    pair_mask = torch.triu(torch.ones_like(pred_iou, dtype=torch.bool), diagonal=1)
    pair_weight = probabilities[:, :, None] * probabilities[:, None, :]
    duplicate_loss = ((pred_iou - 0.35).clamp_min(0) * pair_weight)[pair_mask].mean()

    count_weight = 1.0 if loss_mode in {
        "count_head", "count_head_quality", "ordered_count_head", "ordered_count_quality",
        "dense_center_count", "dense_center_gaussian", "pretrained_center_gaussian"
    } else 0.35
    return (
        cls_loss
        + count_weight * count_loss
        + 5.0 * l1_loss
        + 5.0 * giou_loss
        + area_loss
        + 0.5 * duplicate_loss
    )

def _select_prediction_indices(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    score_threshold: float = 0.35,
    predicted_count: int | None = None,
    ordered_slots: bool = False,
) -> list[int]:
    if predicted_count is not None and predicted_count <= 0:
        return []
    if predicted_count is None:
        candidates = np.flatnonzero(pred_scores >= score_threshold)
        max_keep = None
    elif ordered_slots:
        candidates = np.arange(min(int(predicted_count), len(pred_scores)))
        max_keep = None
    else:
        candidates = np.arange(len(pred_scores))
        max_keep = min(int(predicted_count), len(pred_scores))
    if candidates.size == 0:
        return []
    local_keep = _nms_indices(
        pred_boxes[candidates],
        pred_scores[candidates],
        iou_threshold=0.35,
        max_keep=max_keep,
    )
    return [int(candidates[index]) for index in local_keep]


def draw_boxes(
    img_path: str,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    has_box: np.ndarray,
    out: Path,
    predicted_count: int | None = None,
    ordered_slots: bool = False,
) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for box, flag in zip(gt_boxes, has_box):
        if flag < 0.5:
            continue
        cx, cy, bw, bh = box.tolist()
        xy = [
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ]
        draw.rectangle(xy, outline=(0, 255, 0), width=3)

    # Never use the ground-truth count to choose predictions: doing so hid the
    # one/two-box slot collapse in the old qualitative previews.
    keep = _select_prediction_indices(
        pred_boxes,
        pred_scores,
        predicted_count=predicted_count,
        ordered_slots=ordered_slots,
    )
    for pred_idx in keep:
        box = pred_boxes[pred_idx]
        score = float(pred_scores[pred_idx])
        cx, cy, bw, bh = box.tolist()
        xy = [
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ]
        color = (255, 0, 0) if score >= 0.35 else (255, 165, 0)
        draw.rectangle(xy, outline=color, width=2)
        draw.text((xy[0], max(0, xy[1] - 12)), f"{score:.2f}", fill=color)
    count_label = f"pred={len(keep)}"
    if predicted_count is not None:
        count_label += f" count_head={predicted_count}"
    draw.text((4, 4), count_label, fill=(255, 255, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _detection_correctness(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    correct = np.zeros((len(pred_boxes),), dtype=np.float32)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    ious = _box_iou_matrix(
        torch.from_numpy(pred_boxes).float(),
        torch.from_numpy(gt_boxes).float(),
    ).numpy()
    work = ious.copy()
    for _ in range(min(work.shape)):
        pred_idx, gt_idx = np.unravel_index(np.argmax(work), work.shape)
        if work[pred_idx, gt_idx] < iou_threshold:
            break
        correct[pred_idx] = 1.0
        work[pred_idx, :] = -1.0
        work[:, gt_idx] = -1.0
    return correct


def _expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bins: int = 10,
) -> float:
    if confidences.size == 0:
        return 0.0
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (confidences >= edges[index]) & (
            confidences <= edges[index + 1] if upper_inclusive else confidences < edges[index + 1]
        )
        if not np.any(selected):
            continue
        result += float(selected.mean()) * abs(
            float(confidences[selected].mean()) - float(correctness[selected].mean())
        )
    return result


def compute_val_metrics(
    model: ParallelQueryBoxHead,
    loader: DataLoader,
    device: torch.device,
    loss_mode: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    true_positives = 0
    predicted_total = 0
    ground_truth_total = 0
    count_absolute_error = 0.0
    confidences: list[float] = []
    correctness: list[float] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            gt_boxes = batch["boxes"].to(device)
            has_box = batch["has_box"].to(device)
            logits, boxes, count_logits = model.forward_with_aux(image)
            losses.append(
                float(
                    loss_fn(
                        logits,
                        boxes,
                        gt_boxes,
                        has_box,
                        loss_mode=loss_mode,
                        count_logits=count_logits,
                    ).detach().cpu()
                )
            )
            scores_np = logits.sigmoid().cpu().numpy()
            boxes_np = boxes.cpu().numpy()
            gt_np = gt_boxes.cpu().numpy()
            has_np = has_box.cpu().numpy()
            predicted_counts = (
                count_logits.argmax(dim=1).cpu().numpy()
                if count_logits is not None
                else np.full((len(scores_np),), -1, dtype=np.int64)
            )
            for batch_idx in range(len(scores_np)):
                requested_count = (
                    int(predicted_counts[batch_idx])
                    if count_logits is not None
                    else None
                )
                keep = _select_prediction_indices(
                    boxes_np[batch_idx],
                    scores_np[batch_idx],
                    predicted_count=requested_count,
                    ordered_slots=loss_mode in {"ordered_count_head", "ordered_count_quality"},
                )
                valid_gt = np.flatnonzero(has_np[batch_idx] > 0.5)
                selected_boxes = boxes_np[batch_idx, keep]
                selected_scores = scores_np[batch_idx, keep]
                gt_current = gt_np[batch_idx, valid_gt]
                correct = _detection_correctness(selected_boxes, gt_current)
                true_positives += int(correct.sum())
                predicted_total += len(keep)
                ground_truth_total += len(valid_gt)
                count_absolute_error += abs(len(keep) - len(valid_gt))
                confidences.extend(selected_scores.tolist())
                correctness.extend(correct.tolist())

    precision = true_positives / max(1, predicted_total)
    recall = true_positives / max(1, ground_truth_total)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    confidence_array = np.asarray(confidences, dtype=np.float32)
    correctness_array = np.asarray(correctness, dtype=np.float32)
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_mae": count_absolute_error / max(1, len(loader.dataset)),
        "mean_confidence": float(confidence_array.mean()) if confidence_array.size else 0.0,
        "ece": _expected_calibration_error(confidence_array, correctness_array),
    }


def evaluate(
    model: ParallelQueryBoxHead,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    step: int,
    loss_mode: str,
) -> None:
    model.eval()
    batch = next(iter(loader))
    image = batch["image"].to(device)
    with torch.no_grad():
        logits, boxes, count_logits = model.forward_with_aux(image)
    pred_scores = logits.sigmoid().cpu().numpy()
    pred_boxes = boxes.cpu().numpy()
    predicted_counts = (
        count_logits.argmax(dim=1).cpu().numpy() if count_logits is not None else None
    )
    gt_boxes = batch["boxes"].numpy()
    has_box = batch["has_box"].numpy()
    for i in range(min(4, image.shape[0])):
        draw_boxes(
            batch["imgname"][i],
            pred_boxes[i],
            pred_scores[i],
            gt_boxes[i],
            has_box[i],
            out_dir / "visuals" / f"step_{step:06d}_sample_{i}.png",
            predicted_count=int(predicted_counts[i]) if predicted_counts is not None else None,
            ordered_slots=loss_mode in {"ordered_count_head", "ordered_count_quality"},
        )
    model.train()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--train-rgb-cache", required=True)
    parser.add_argument("--val-rgb-cache", required=True)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=500)
    parser.add_argument("--val-freq", type=int, default=100)
    parser.add_argument("--checkpoint-freq", type=int, default=500)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--clean-log-layout", action="store_true", help="Write ckpt/ and visuals/ directly under --output-dir.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--loss-mode",
        choices=[
            "baseline", "set_focal_cardinality", "quality_focal", "count_head",
            "count_head_quality", "ordered_count_head", "ordered_count_quality",
            "dense_center_count", "dense_center_gaussian", "pretrained_center_gaussian",
        ],
        default="pretrained_center_gaussian",
    )
    parser.add_argument("--resume", default="", help="Optional bbox checkpoint to resume from.")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_set = BBoxDataset(args.train_dataset, args.train_rgb_cache)
    val_set = BBoxDataset(args.val_dataset, args.val_rgb_cache)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, drop_last=False, num_workers=args.num_workers)

    max_slots = train_set.boxes.shape[1]
    use_count_head = args.loss_mode in {
        "count_head", "count_head_quality", "ordered_count_head", "ordered_count_quality",
        "dense_center_count", "dense_center_gaussian", "pretrained_center_gaussian"
    }
    if args.loss_mode in {"dense_center_count", "dense_center_gaussian"}:
        model = DenseCenterBoxHead(max_count=max_slots).to(device)
    elif args.loss_mode == "pretrained_center_gaussian":
        model = PretrainedResNetCenterBoxHead(max_count=max_slots).to(device)
    else:
        model = ParallelQueryBoxHead(max_slots=max_slots, use_count_head=use_count_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
    best_val_f1 = -1.0

    step = 0
    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_f1 = float(ckpt.get("best_val_f1", best_val_f1))
        checkpoint_mode = str(ckpt.get("loss_mode", args.loss_mode))
        if checkpoint_mode != args.loss_mode:
            raise ValueError(
                f"Checkpoint loss_mode={checkpoint_mode} does not match --loss-mode={args.loss_mode}"
            )
        print(
            f"[bbox] resumed from {resume_path} at step={step} "
            f"loss_mode={args.loss_mode} optimizer={'yes' if 'optimizer' in ckpt else 'no'}",
            flush=True,
        )

    progress = tqdm(total=args.total_steps, initial=step, desc="train bbox")
    while step < args.total_steps:
        for batch in train_loader:
            image = batch["image"].to(device)
            gt_boxes = batch["boxes"].to(device)
            has_box = batch["has_box"].to(device)
            logits, boxes, count_logits = model.forward_with_aux(image)
            loss = loss_fn(
                logits,
                boxes,
                gt_boxes,
                has_box,
                loss_mode=args.loss_mode,
                count_logits=count_logits,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            progress.update(1)
            progress.set_description(f"bbox_loss: {loss.item():.4f}")
            if step % args.val_freq == 0:
                train_loss = float(loss.detach().cpu())
                val_metrics = compute_val_metrics(model, val_loader, device, args.loss_mode)
                val_loss = val_metrics["loss"]
                print(
                    f"[bbox] step={step} mode={args.loss_mode} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} val_f1={val_metrics['f1']:.6f} "
                    f"precision={val_metrics['precision']:.6f} recall={val_metrics['recall']:.6f} "
                    f"count_mae={val_metrics['count_mae']:.6f} "
                    f"mean_conf={val_metrics['mean_confidence']:.6f} ece={val_metrics['ece']:.6f}",
                    flush=True,
                )
                metrics.append(step, train_loss, val_metrics)
                improved = (
                    val_metrics["f1"] > best_val_f1 + 1e-8
                    or (
                        abs(val_metrics["f1"] - best_val_f1) <= 1e-8
                        and val_loss < best_val_loss
                    )
                )
                best_val_loss = min(best_val_loss, val_loss)
                if improved:
                    best_val_f1 = val_metrics["f1"]
                    torch.save(
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "max_slots": max_slots,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_metrics": val_metrics,
                            "best_val_loss": best_val_loss,
                            "best_val_f1": best_val_f1,
                            "loss_mode": args.loss_mode,
                            "use_count_head": use_count_head,
                        },
                        ckpt_dir / "model-best.pt",
                    )
                if improved or step % args.log_freq == 0:
                    evaluate(model, val_loader, device, out_dir, step, args.loss_mode)
            if step % args.checkpoint_freq == 0:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "max_slots": max_slots,
                        "best_val_loss": best_val_loss,
                        "best_val_f1": best_val_f1,
                        "loss_mode": args.loss_mode,
                        "use_count_head": use_count_head,
                    },
                    ckpt_dir / "model-latest.pt",
                )
            if step >= args.total_steps:
                break
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "max_slots": max_slots,
            "best_val_loss": best_val_loss,
            "best_val_f1": best_val_f1,
            "loss_mode": args.loss_mode,
            "use_count_head": use_count_head,
        },
        ckpt_dir / "model-last.pt",
    )
    if args.clean_log_layout:
        print(f"[bbox] finished; outputs under {args.output_dir}/ckpt and {args.output_dir}/visuals", flush=True)
    else:
        print(f"[bbox] finished; outputs under {args.output_dir}/checkpoints/{args.name} and {args.output_dir}/visuals/{args.name}", flush=True)


if __name__ == "__main__":
    main()
