from __future__ import annotations

import argparse
import importlib.util
import sys
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
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

class MetricLogger:
    def __init__(self, out_dir: Path) -> None:
        self.csv_path = out_dir / "metrics.csv"
        self.stats_path = out_dir / "stats.png"
        self.rows: list[tuple[int, float, float]] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.csv_path.write_text("step,train_loss,val_loss\n")

    def append(self, step: int, train_loss: float, val_loss: float) -> None:
        self.rows.append((int(step), float(train_loss), float(val_loss)))
        with self.csv_path.open("a") as f:
            f.write(f"{int(step)},{float(train_loss):.8f},{float(val_loss):.8f}\n")
        self.plot()

    def plot(self) -> None:
        if not self.rows:
            return
        steps = [row[0] for row in self.rows]
        train = [row[1] for row in self.rows]
        val = [row[2] for row in self.rows]
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


WSM_ROOT = Path(__file__).resolve().parents[3]
if str(WSM_ROOT) not in sys.path:
    sys.path.insert(0, str(WSM_ROOT))

MASK_VALUE = -1.0
DWPOSE_PERSON_ID_CHANNEL = 64
DWPOSE_KEYPOINT_ID_CHANNEL = 65
DWPOSE_SCORE_CHANNEL = 66
MEDIAPIPE_PERSON_ID_CHANNEL = 95
MEDIAPIPE_JOINT_ID_CHANNEL = 96
MEDIAPIPE_Z_CHANNEL = 97
MEDIAPIPE_VISIBILITY_CHANNEL = 98
OBJ_ID_CHANNEL = 111
OBJ_VALUE_CHANNELS = list(range(112, 122))


class StructureDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        cache_file: str,
        condition: str,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ) -> None:
        self.data = np.load(dataset_file)
        self.cache = np.load(cache_file)
        self.condition = condition
        self.target_raw = self.data["target"].astype(np.float32)
        self.target_mask = self.data["target_mask"].astype(np.float32)
        if target_mean is None or target_std is None:
            denom = self.target_mask.sum(axis=0).clip(min=1.0)
            target_mean = (self.target_raw * self.target_mask).sum(axis=0) / denom
            target_var = (((self.target_raw - target_mean) ** 2) * self.target_mask).sum(axis=0) / denom
            target_std = np.sqrt(target_var).clip(min=1e-3)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        valid = self.target_mask > 0.0
        target_safe = np.where(valid, self.target_raw, self.target_mean[None, :]).astype(np.float64)
        target_norm = (target_safe - self.target_mean[None, :].astype(np.float64)) / self.target_std[None, :].astype(np.float64)
        target_norm = np.nan_to_num(target_norm, nan=0.0, posinf=20.0, neginf=-20.0)
        self.target = np.clip(target_norm, -20.0, 20.0).astype(np.float32)
        self.imgname = self.data["imgname"].astype(str)
        self.wsv_path = self.data["wsv_path"].astype(str)
        self.frame_key = self.data["frame_key"].astype(str)
        self.person_id = self.data["person_id"].astype(np.int32)
        self.valid_points = self.data["valid_points"].astype(np.int32)
        self.bbox = self.data["bbox"].astype(np.float32) if "bbox" in self.data.files else np.zeros((len(self.target_raw), 4), dtype=np.float32)
        self.max_keypoints = int(np.asarray(self.data.get("max_keypoints", 160)).item())
        self.max_slots = int(np.asarray(self.data.get("max_slots", 32)).item())
        if "target_format" in self.data.files:
            self.target_format = str(np.asarray(self.data["target_format"]).item())
        else:
            self.target_format = ""
        if condition == "crop_pare":
            pose_feats = self.cache["pose_feats"].reshape(self.target_raw.shape[0], -1)
            cam_shape_feats = self.cache["cam_shape_feats"].reshape(self.target_raw.shape[0], -1)
            self.features = np.concatenate([pose_feats, cam_shape_feats], axis=1).astype(np.float32)
            sample_count = self.target_raw.shape[0]
            if "pred_pose" in self.cache.files:
                pred_pose = self.cache["pred_pose"].astype(np.float32)
                self.pare_pose_6d = pred_pose[:, :, :, :2].transpose(0, 1, 3, 2).reshape(sample_count, -1)
            else:
                identity = np.eye(3, dtype=np.float32)[None, None].repeat(sample_count, axis=0).repeat(24, axis=1)
                self.pare_pose_6d = identity[:, :, :, :2].transpose(0, 1, 3, 2).reshape(sample_count, -1)
            self.pare_cam = (
                self.cache["pred_cam"].reshape(sample_count, -1).astype(np.float32)
                if "pred_cam" in self.cache.files
                else np.zeros((sample_count, 3), dtype=np.float32)
            )
            self.pare_betas = (
                self.cache["pred_betas"].reshape(sample_count, -1).astype(np.float32)
                if "pred_betas" in self.cache.files
                else np.zeros((sample_count, 10), dtype=np.float32)
            )
            self.images = None
        elif condition == "full_rgb":
            self.images = self.cache["rgb_ch0_2_full"].astype(np.float32)
            self.features = None
        elif condition == "crop_rgb":
            self.images = self.cache["rgb_ch0_2_crop"].astype(np.float32)
            self.features = None
        elif condition == "pointcloud":
            self.pointcloud = self.cache["pointcloud_norm"].astype(np.float32)
            self.pointcloud_mask = self.cache["pointcloud_mask"].astype(np.float32)
            if "init_bfs_delta_tree" in self.cache.files and "init_bfs_delta_tree_mask" in self.cache.files:
                self.init_bfs_delta_tree = self.cache["init_bfs_delta_tree"].astype(np.float32)
                self.init_bfs_delta_tree_mask = self.cache["init_bfs_delta_tree_mask"].astype(np.float32)
            else:
                self.init_bfs_delta_tree = np.zeros_like(self.target_raw, dtype=np.float32)
                self.init_bfs_delta_tree_mask = np.zeros_like(self.target_mask, dtype=np.float32)
            self.images = None
            self.features = None
        else:
            raise ValueError(f"Unsupported condition: {condition}")

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict:
        item = {
            "target": torch.from_numpy(self.target[index]),
            "target_raw": torch.from_numpy(self.target_raw[index]),
            "target_mask": torch.from_numpy(self.target_mask[index]),
            "imgname": self.imgname[index],
            "wsv_path": self.wsv_path[index],
            "frame_key": self.frame_key[index],
            "person_id": torch.tensor(self.person_id[index], dtype=torch.int32),
            "valid_points": torch.tensor(self.valid_points[index], dtype=torch.int32),
            "bbox": torch.from_numpy(self.bbox[index]),
        }
        if self.condition == "crop_pare":
            item["features"] = torch.from_numpy(self.features[index])
            item["pare_pose_6d"] = torch.from_numpy(self.pare_pose_6d[index])
            item["pare_cam"] = torch.from_numpy(self.pare_cam[index])
            item["pare_betas"] = torch.from_numpy(self.pare_betas[index])
        elif self.condition == "pointcloud":
            item["pointcloud"] = torch.from_numpy(self.pointcloud[index])
            item["pointcloud_mask"] = torch.from_numpy(self.pointcloud_mask[index])
            item["init_bfs_delta_tree"] = torch.from_numpy(self.init_bfs_delta_tree[index])
            item["init_bfs_delta_tree_mask"] = torch.from_numpy(self.init_bfs_delta_tree_mask[index])
        else:
            item["image"] = torch.from_numpy(self.images[index])
        return item


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["features"])


def _human_pose_topology(task: str, max_keypoints: int) -> tuple[np.ndarray, list[int], int, int]:
    """Return a rooted tree for DWPose-128 or MediaPipe-33 landmarks."""
    parents = np.full((max_keypoints,), -2, dtype=np.int64)
    if task == "dwpose":
        active_count = min(128, max_keypoints)
        root = 1  # OpenPose neck: central and present in >94% of samples.
        body_parents = {
            0: 1, 1: -1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 5, 7: 6,
            8: 1, 9: 8, 10: 9, 11: 1, 12: 11, 13: 12,
            14: 0, 15: 0, 16: 14, 17: 15,
        }
        for child, parent in body_parents.items():
            if child < active_count:
                parents[child] = parent

        def face(local_index: int) -> int:
            return 18 + local_index

        def set_face(child: int, parent: int) -> None:
            child_global = face(child)
            parent_global = face(parent) if parent >= 0 else 0
            if child_global < active_count:
                parents[child_global] = parent_global

        # Dlib-68 face tree. Loops are intentionally broken into short chains.
        set_face(30, -1)
        set_face(29, 30)
        set_face(28, 29)
        set_face(27, 28)
        set_face(31, 30)
        for child in range(32, 36):
            set_face(child, child - 1)
        set_face(8, 30)
        for child in range(7, -1, -1):
            set_face(child, child + 1)
        for child in range(9, 17):
            set_face(child, child - 1)
        set_face(21, 27)
        for child in range(20, 16, -1):
            set_face(child, child + 1)
        set_face(22, 27)
        for child in range(23, 27):
            set_face(child, child - 1)
        set_face(36, 27)
        for child in range(37, 42):
            set_face(child, child - 1)
        set_face(42, 27)
        for child in range(43, 48):
            set_face(child, child - 1)
        set_face(48, 33)
        for child in range(49, 60):
            set_face(child, child - 1)
        set_face(60, 48)
        for child in range(61, 68):
            set_face(child, child - 1)

        # Each 21-point hand is a wrist plus five four-joint finger chains.
        for hand_base, body_wrist in ((86, 7), (107, 4)):
            if hand_base >= active_count:
                continue
            parents[hand_base] = body_wrist
            for finger_start in (1, 5, 9, 13, 17):
                first = hand_base + finger_start
                if first < active_count:
                    parents[first] = hand_base
                for offset in range(1, 4):
                    child = first + offset
                    if child < active_count:
                        parents[child] = child - 1
    elif task == "mediapipe":
        active_count = min(33, max_keypoints)
        root = 11  # Left shoulder is central and present in >99% of samples.
        media_parents = {
            0: 11, 1: 0, 2: 1, 3: 2, 7: 3,
            4: 0, 5: 4, 6: 5, 8: 6, 9: 0, 10: 9,
            11: -1, 12: 11,
            13: 11, 15: 13, 17: 15, 19: 15, 21: 15,
            14: 12, 16: 14, 18: 16, 20: 16, 22: 16,
            23: 11, 24: 12,
            25: 23, 27: 25, 29: 27, 31: 29,
            26: 24, 28: 26, 30: 28, 32: 30,
        }
        for child, parent in media_parents.items():
            if child < active_count:
                parents[child] = parent
    else:
        raise ValueError(f"Unsupported human pose topology: {task}")

    # Any future/unknown active landmark remains usable as a short root branch.
    for joint in range(active_count):
        if parents[joint] == -2:
            parents[joint] = root

    order = [root]
    pending = set(range(active_count)) - {root}
    while pending:
        ready = sorted(joint for joint in pending if int(parents[joint]) in order)
        if not ready:
            raise RuntimeError(f"Invalid {task} skeleton topology; unresolved joints={sorted(pending)}")
        order.extend(ready)
        pending.difference_update(ready)
    return parents, order, root, active_count


def _frame_from_x_direction_np(direction: np.ndarray) -> np.ndarray:
    x_axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x_axis /= norm
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(reference, x_axis))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = np.cross(reference, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def _kinematic_priors(
    target_raw: np.ndarray,
    target_mask: np.ndarray,
    task: str,
    max_keypoints: int,
) -> dict[str, np.ndarray | list[int] | int]:
    parents, order, root, active_count = _human_pose_topology(task, max_keypoints)
    values_per_keypoint = 3 if task == "dwpose" else 4
    spatial_dim = 2 if task == "dwpose" else 3
    values = target_raw.reshape(-1, max_keypoints, values_per_keypoint)
    mask = target_mask.reshape(-1, max_keypoints, values_per_keypoint)
    coord_valid = np.all(mask[:, :, :spatial_dim] > 0.0, axis=-1)

    base_root = np.asarray([0.5, 0.5] if spatial_dim == 2 else [0.5, 0.5, 0.0], dtype=np.float32)
    root_samples = values[coord_valid[:, root], root, :spatial_dim]
    if len(root_samples):
        base_root = np.median(root_samples, axis=0).astype(np.float32)
        root_std = np.std(root_samples, axis=0).astype(np.float32)
    else:
        root_std = np.full((spatial_dim,), 0.2, dtype=np.float32)
    root_std = np.clip(root_std, 0.05, 0.5)

    base_lengths = np.full((max_keypoints,), 0.05, dtype=np.float32)
    base_local_angles = np.zeros((max_keypoints,), dtype=np.float32)
    base_local_rotations = np.tile(np.eye(3, dtype=np.float32), (max_keypoints, 1, 1))
    base_global_angles = np.zeros((max_keypoints,), dtype=np.float32)
    base_global_rotations = np.tile(np.eye(3, dtype=np.float32), (max_keypoints, 1, 1))

    for joint in order[1:]:
        parent = int(parents[joint])
        valid = coord_valid[:, joint] & coord_valid[:, parent]
        differences = values[valid, joint, :spatial_dim] - values[valid, parent, :spatial_dim]
        lengths = np.linalg.norm(differences, axis=1)
        usable = lengths > 1e-5
        differences = differences[usable]
        lengths = lengths[usable]
        if len(lengths):
            base_lengths[joint] = float(np.clip(np.median(lengths), 1e-3, 2.0))
            unit = differences / lengths[:, None]
            mean_direction = unit.mean(axis=0)
            if float(np.linalg.norm(mean_direction)) < 1e-5:
                mean_direction = unit[0]
        else:
            mean_direction = np.asarray(
                [1.0, 0.0] if spatial_dim == 2 else [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
        if spatial_dim == 2:
            global_angle = float(np.arctan2(mean_direction[1], mean_direction[0]))
            parent_angle = float(base_global_angles[parent]) if parent != root else 0.0
            local_angle = (global_angle - parent_angle + np.pi) % (2.0 * np.pi) - np.pi
            base_global_angles[joint] = global_angle
            base_local_angles[joint] = local_angle
        else:
            global_rotation = _frame_from_x_direction_np(mean_direction)
            parent_rotation = base_global_rotations[parent] if parent != root else np.eye(3, dtype=np.float32)
            base_global_rotations[joint] = global_rotation
            base_local_rotations[joint] = parent_rotation.T @ global_rotation

    score_index = spatial_dim
    score_prior = np.full((max_keypoints,), 0.5, dtype=np.float32)
    for joint in range(active_count):
        valid = mask[:, joint, score_index] > 0.0
        if np.any(valid):
            score_prior[joint] = float(np.mean(values[valid, joint, score_index]))
    score_prior = np.clip(score_prior, 0.02, 0.98)
    score_prior_logits = np.log(score_prior / (1.0 - score_prior)).astype(np.float32)
    return {
        "parents": parents,
        "order": order,
        "root": root,
        "active_count": active_count,
        "spatial_dim": spatial_dim,
        "values_per_keypoint": values_per_keypoint,
        "base_root": base_root,
        "root_std": root_std,
        "base_lengths": base_lengths,
        "base_local_angles": base_local_angles,
        "base_local_rotations": base_local_rotations,
        "score_prior_logits": score_prior_logits,
    }


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle_sq = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-12))
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    small = angle_sq < 1e-8
    a = torch.where(small, 1.0 - angle_sq / 6.0, torch.sin(angle) / angle)
    b = torch.where(
        small,
        0.5 - angle_sq / 24.0,
        (1.0 - torch.cos(angle)) / angle_sq.clamp_min(1e-12),
    )
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    identity = identity.view(*([1] * (axis_angle.ndim - 1)), 3, 3)
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


class KinematicPoseRegressor(nn.Module):
    """Rotation-sequence human decoder with an initialized canonical skeleton.

    PARE/SMPL rotations condition a small residual network. The output is root
    translation, global/person scale, local joint rotations, bounded bone-length
    residuals, and independently calibrated confidence/visibility. Forward
    kinematics exports the original DWPose or MediaPipe keypoint tensor.
    """

    def __init__(
        self,
        feature_dim: int,
        max_keypoints: int,
        priors: dict[str, np.ndarray | list[int] | int],
        hidden_dim: int = 512,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_keypoints = max_keypoints
        self.parents = np.asarray(priors["parents"], dtype=np.int64)
        self.order = [int(item) for item in priors["order"]]
        self.root = int(priors["root"])
        self.active_count = int(priors["active_count"])
        self.spatial_dim = int(priors["spatial_dim"])
        self.values_per_keypoint = int(priors["values_per_keypoint"])

        pare_condition_dim = 24 * 6 + 3 + 10
        input_dim = feature_dim + pare_condition_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        rotation_dim = 1 if self.spatial_dim == 2 else 3
        self.root_head = nn.Linear(hidden_dim, self.spatial_dim)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.rotation_head = nn.Linear(hidden_dim, max_keypoints * rotation_dim)
        self.length_head = nn.Linear(hidden_dim, max_keypoints)
        self.score_head = nn.Linear(hidden_dim, max_keypoints)
        for head in (
            self.root_head,
            self.scale_head,
            self.rotation_head,
            self.length_head,
            self.score_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.register_buffer("base_root", torch.as_tensor(priors["base_root"], dtype=torch.float32))
        self.register_buffer("root_std", torch.as_tensor(priors["root_std"], dtype=torch.float32))
        self.register_buffer("base_lengths", torch.as_tensor(priors["base_lengths"], dtype=torch.float32))
        self.register_buffer(
            "base_local_angles",
            torch.as_tensor(priors["base_local_angles"], dtype=torch.float32),
        )
        self.register_buffer(
            "base_local_rotations",
            torch.as_tensor(priors["base_local_rotations"], dtype=torch.float32),
        )
        self.register_buffer(
            "score_prior_logits",
            torch.as_tensor(priors["score_prior_logits"], dtype=torch.float32),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        condition = torch.cat(
            [
                batch["features"],
                batch["pare_pose_6d"],
                batch["pare_cam"],
                batch["pare_betas"],
            ],
            dim=1,
        )
        hidden = self.trunk(condition)
        root = self.base_root.unsqueeze(0) + 3.0 * self.root_std.unsqueeze(0) * torch.tanh(
            self.root_head(hidden)
        )
        global_scale = torch.exp(0.35 * torch.tanh(self.scale_head(hidden)))
        length_residual = torch.tanh(self.length_head(hidden))
        lengths = (
            self.base_lengths.unsqueeze(0)
            * global_scale
            * torch.exp(0.35 * length_residual)
        )
        score_logits = self.score_head(hidden) + self.score_prior_logits.unsqueeze(0)

        batch_size = hidden.shape[0]
        zero_point = torch.zeros(
            (batch_size, self.spatial_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        points: list[torch.Tensor | None] = [None] * self.max_keypoints
        points[self.root] = root

        if self.spatial_dim == 2:
            residual = torch.tanh(
                self.rotation_head(hidden).view(batch_size, self.max_keypoints)
            ) * np.pi
            local_angles = self.base_local_angles.unsqueeze(0) + residual
            global_angles: list[torch.Tensor | None] = [None] * self.max_keypoints
            global_angles[self.root] = torch.zeros(
                (batch_size,), device=hidden.device, dtype=hidden.dtype
            )
            for joint in self.order[1:]:
                parent = int(self.parents[joint])
                parent_angle = global_angles[parent]
                assert parent_angle is not None and points[parent] is not None
                angle = parent_angle + local_angles[:, joint]
                global_angles[joint] = angle
                direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
                points[joint] = points[parent] + lengths[:, joint : joint + 1] * direction
            rotation_values = local_angles.unsqueeze(-1)
        else:
            residual_axis_angle = torch.tanh(
                self.rotation_head(hidden).view(batch_size, self.max_keypoints, 3)
            ) * np.pi
            delta_rotation = _axis_angle_to_matrix(residual_axis_angle)
            global_rotations: list[torch.Tensor | None] = [None] * self.max_keypoints
            root_rotation = torch.eye(3, device=hidden.device, dtype=hidden.dtype)
            root_rotation = root_rotation.unsqueeze(0).expand(batch_size, -1, -1)
            global_rotations[self.root] = root_rotation
            for joint in self.order[1:]:
                parent = int(self.parents[joint])
                parent_rotation = global_rotations[parent]
                assert parent_rotation is not None and points[parent] is not None
                base_local = self.base_local_rotations[joint].unsqueeze(0).expand(batch_size, -1, -1)
                local_rotation = base_local @ delta_rotation[:, joint]
                global_rotation = parent_rotation @ local_rotation
                global_rotations[joint] = global_rotation
                direction = global_rotation[:, :, 0]
                points[joint] = points[parent] + lengths[:, joint : joint + 1] * direction
            rotation_values = residual_axis_angle

        point_tensor = torch.stack(
            [point if point is not None else zero_point for point in points],
            dim=1,
        )
        confidence = torch.sigmoid(score_logits).unsqueeze(-1)
        flat = torch.cat([point_tensor, confidence], dim=-1).reshape(batch_size, -1)
        return {
            "flat": flat,
            "points": point_tensor,
            "score_logits": score_logits,
            "rotation_values": rotation_values,
            "lengths": lengths,
            "length_residual": length_residual,
            "global_scale": global_scale,
        }


class FullImageRegressor(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.backbone(batch["image"]))


class PointCloudConditionEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pointcloud: torch.Tensor, pointcloud_mask: torch.Tensor) -> torch.Tensor:
        point_feat = self.point_mlp(pointcloud)
        valid = pointcloud_mask > 0.0
        masked = point_feat.masked_fill(~valid.unsqueeze(-1), 0.0)
        denom = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_feat = masked.sum(dim=1) / denom
        max_feat = point_feat.masked_fill(~valid.unsqueeze(-1), -1e4).max(dim=1).values
        max_feat = torch.where(torch.isfinite(max_feat), max_feat, torch.zeros_like(max_feat))
        return self.global_mlp(torch.cat([mean_feat, max_feat], dim=1))


class GrowingTreeRegressor(nn.Module):
    """Autoregressive-ish OBJ skeleton decoder.

    The previous dense OBJ head predicts all BFS slots independently in one
    vector, which makes the topology easy to scramble.  This head predicts the
    tree in slot order: each step consumes the previous skeleton token and
    emits existence, parent slot and absolute xyz for the next node.
    """

    def __init__(self, max_slots: int, hidden_dim: int = 384) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.hidden_dim = hidden_dim
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.image_proj = nn.Linear(192, hidden_dim)
        self.token_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cell = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.parent_head = nn.Linear(hidden_dim, max_slots)
        self.xyz_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.start_token = nn.Parameter(torch.zeros(8))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        batch_size = batch["image"].shape[0]
        image_feat = self.image_proj(self.image_backbone(batch["image"]))
        hidden = torch.tanh(image_feat)
        prev_token = self.start_token.unsqueeze(0).expand(batch_size, -1)
        teacher_force = self.training and "target_raw" in batch and "target_mask" in batch
        if teacher_force:
            target_slots = batch["target_raw"].view(batch_size, self.max_slots, 7)
            target_mask = batch["target_mask"].view(batch_size, self.max_slots, 7)
            target_exists = (target_mask[:, :, 0] > 0.0).float()
        else:
            target_slots = None
            target_exists = None

        exist_logits = []
        parent_logits = []
        xyz_values = []
        for slot in range(self.max_slots):
            token_feat = self.token_proj(prev_token)
            hidden = self.cell(torch.cat([image_feat, token_feat], dim=1), hidden)
            exist = self.exist_head(hidden).squeeze(-1)
            parent = self.parent_head(hidden)
            # Slots should not attach to future nodes.  Keep this in the model
            # output so both training and inference use the same topology bias.
            if slot + 1 < self.max_slots:
                parent = parent.clone()
                parent[:, slot + 1 :] = -1e4
            xyz = torch.sigmoid(self.xyz_head(hidden))
            exist_logits.append(exist)
            parent_logits.append(parent)
            xyz_values.append(xyz)

            if teacher_force and target_slots is not None and target_exists is not None:
                src = target_slots[:, slot]
                src_exists = target_exists[:, slot : slot + 1]
                prev_token = torch.cat([src, src_exists], dim=1)
            else:
                parent_idx = torch.argmax(parent, dim=1)
                parent_den = float(max(1, self.max_slots - 1))
                if xyz_values:
                    xyz_stack = torch.stack(xyz_values, dim=1)
                    parent_xyz = torch.gather(
                        xyz_stack,
                        1,
                        parent_idx.clamp(0, xyz_stack.shape[1] - 1).view(batch_size, 1, 1).expand(-1, 1, 3),
                    ).squeeze(1)
                else:
                    parent_xyz = xyz
                delta = torch.clamp(xyz - parent_xyz, -1.0, 1.0)
                prev_token = torch.cat(
                    [parent_idx.float().unsqueeze(1) / parent_den, delta, xyz, torch.sigmoid(exist).unsqueeze(1)],
                    dim=1,
                )

        return {
            "exist_logits": torch.stack(exist_logits, dim=1),
            "parent_logits": torch.stack(parent_logits, dim=1),
            "xyz": torch.stack(xyz_values, dim=1),
        }


class DiscreteGrowingTreeRegressor(nn.Module):
    """BFS OBJ skeleton decoder with CE-style xyz bins.

    This keeps the "grow a tree" ordering, but avoids smooth regression for
    joint locations.  Each x/y/z coordinate is predicted as a discrete token,
    closer to text generation and less prone to visually plausible low-MSE
    averages.
    """

    def __init__(
        self,
        max_slots: int,
        hidden_dim: int = 384,
        xyz_bins: int = 64,
        use_slot_embed: bool = False,
        condition: str = "image",
    ) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.hidden_dim = hidden_dim
        self.xyz_bins = xyz_bins
        self.use_slot_embed = use_slot_embed
        self.condition = condition
        if condition == "pointcloud":
            self.pointcloud_encoder = PointCloudConditionEncoder(hidden_dim)
            self.image_backbone = None
            self.image_proj = None
        else:
            self.image_backbone = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, 192, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.image_proj = nn.Linear(192, hidden_dim)
            self.pointcloud_encoder = None
        self.token_proj = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cell = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.slot_embed = nn.Embedding(max_slots, hidden_dim) if use_slot_embed else None
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.parent_head = nn.Linear(hidden_dim, max_slots)
        self.xyz_head = nn.Linear(hidden_dim, 3 * xyz_bins)
        self.start_token = nn.Parameter(torch.zeros(5))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if self.condition == "pointcloud":
            condition_feat = self.pointcloud_encoder(batch["pointcloud"], batch["pointcloud_mask"])
            batch_size = batch["pointcloud"].shape[0]
            device = batch["pointcloud"].device
        else:
            condition_feat = self.image_proj(self.image_backbone(batch["image"]))
            batch_size = batch["image"].shape[0]
            device = batch["image"].device
        hidden = torch.tanh(condition_feat)
        prev_token = self.start_token.unsqueeze(0).expand(batch_size, -1)
        teacher_force = self.training and "target_raw" in batch and "target_mask" in batch
        if teacher_force:
            target_slots = batch["target_raw"].view(batch_size, self.max_slots, 7)
            target_mask = batch["target_mask"].view(batch_size, self.max_slots, 7)
            target_exists = (target_mask[:, :, 0] > 0.0).float()
        else:
            target_slots = None
            target_exists = None

        exist_logits = []
        parent_logits = []
        xyz_logits = []
        xyz_values = []
        slot_den = float(max(1, self.max_slots - 1))
        bin_den = float(max(1, self.xyz_bins - 1))
        slot_ids = torch.arange(self.max_slots, device=device) if self.slot_embed is not None else None
        for slot in range(self.max_slots):
            token_feat = self.token_proj(prev_token)
            if self.slot_embed is not None and slot_ids is not None:
                slot_feat = self.slot_embed(slot_ids[slot]).unsqueeze(0).expand(batch_size, -1)
                token_feat = token_feat + slot_feat
            hidden = self.cell(torch.cat([condition_feat, token_feat], dim=1), hidden)
            exist = self.exist_head(hidden).squeeze(-1)
            parent = self.parent_head(hidden)
            if slot + 1 < self.max_slots:
                parent = parent.clone()
                parent[:, slot + 1 :] = -1e4
            logits = self.xyz_head(hidden).view(batch_size, 3, self.xyz_bins)
            bins = torch.argmax(logits, dim=-1)
            xyz = bins.float() / bin_den
            exist_logits.append(exist)
            parent_logits.append(parent)
            xyz_logits.append(logits)
            xyz_values.append(xyz)

            if teacher_force and target_slots is not None and target_exists is not None:
                src = target_slots[:, slot]
                src_exists = target_exists[:, slot : slot + 1]
                prev_token = torch.cat([src[:, 0:1], torch.clamp(src[:, 4:7], 0.0, 1.0), src_exists], dim=1)
            else:
                parent_idx = torch.argmax(parent, dim=1)
                prev_token = torch.cat(
                    [parent_idx.float().unsqueeze(1) / slot_den, xyz, torch.sigmoid(exist).unsqueeze(1)],
                    dim=1,
                )

        return {
            "exist_logits": torch.stack(exist_logits, dim=1),
            "parent_logits": torch.stack(parent_logits, dim=1),
            "xyz_logits": torch.stack(xyz_logits, dim=1),
            "xyz": torch.stack(xyz_values, dim=1),
        }


class DeltaGrowingTreeRegressor(nn.Module):
    """BFS OBJ skeleton decoder that grows positions from parent + delta.

    The absolute-coordinate CE head can learn each slot independently, so late
    training can reduce loss while visually breaking the tree into unrelated
    parts.  This head predicts a root position and then predicts local deltas
    for later slots; visual xyz is reconstructed recursively from the predicted
    parent, which makes connectivity a geometric constraint instead of only a
    line drawn after the fact.
    """

    def __init__(
        self,
        max_slots: int,
        hidden_dim: int = 384,
        xyz_bins: int = 64,
        delta_bins: int = 65,
        condition: str = "image",
    ) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.hidden_dim = hidden_dim
        self.xyz_bins = xyz_bins
        self.delta_bins = delta_bins
        self.condition = condition
        if condition == "pointcloud":
            self.pointcloud_encoder = PointCloudConditionEncoder(hidden_dim)
            self.image_backbone = None
            self.image_proj = None
        else:
            self.image_backbone = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, 192, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.image_proj = nn.Linear(192, hidden_dim)
            self.pointcloud_encoder = None
        self.token_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cell = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.parent_head = nn.Linear(hidden_dim, max_slots)
        self.root_xyz_head = nn.Linear(hidden_dim, 3 * xyz_bins)
        self.delta_head = nn.Linear(hidden_dim, 3 * delta_bins)
        self.start_token = nn.Parameter(torch.zeros(8))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if self.condition == "pointcloud":
            condition_feat = self.pointcloud_encoder(batch["pointcloud"], batch["pointcloud_mask"])
            batch_size = batch["pointcloud"].shape[0]
        else:
            condition_feat = self.image_proj(self.image_backbone(batch["image"]))
            batch_size = batch["image"].shape[0]
        hidden = torch.tanh(condition_feat)
        prev_token = self.start_token.unsqueeze(0).expand(batch_size, -1)
        teacher_force = self.training and "target_raw" in batch and "target_mask" in batch
        if teacher_force:
            target_slots = batch["target_raw"].view(batch_size, self.max_slots, 7)
            target_mask = batch["target_mask"].view(batch_size, self.max_slots, 7)
            target_exists = (target_mask[:, :, 0] > 0.0).float()
        else:
            target_slots = None
            target_exists = None

        exist_logits = []
        parent_logits = []
        root_xyz_logits = []
        delta_logits = []
        delta_values = []
        xyz_values = []
        slot_den = float(max(1, self.max_slots - 1))
        xyz_den = float(max(1, self.xyz_bins - 1))
        delta_den = float(max(1, self.delta_bins - 1))
        for slot in range(self.max_slots):
            token_feat = self.token_proj(prev_token)
            hidden = self.cell(torch.cat([condition_feat, token_feat], dim=1), hidden)
            exist = self.exist_head(hidden).squeeze(-1)
            parent = self.parent_head(hidden)
            if slot + 1 < self.max_slots:
                parent = parent.clone()
                parent[:, slot + 1 :] = -1e4
            root_logits = self.root_xyz_head(hidden).view(batch_size, 3, self.xyz_bins)
            d_logits = self.delta_head(hidden).view(batch_size, 3, self.delta_bins)
            root_bins = torch.argmax(root_logits, dim=-1)
            delta_bins = torch.argmax(d_logits, dim=-1)
            root_xyz = root_bins.float() / xyz_den
            delta = delta_bins.float() / delta_den * 2.0 - 1.0
            if slot == 0:
                xyz = root_xyz
            else:
                parent_idx = torch.argmax(parent, dim=1).clamp(0, slot - 1)
                xyz_stack = torch.stack(xyz_values, dim=1)
                parent_xyz = torch.gather(
                    xyz_stack,
                    1,
                    parent_idx.view(batch_size, 1, 1).expand(-1, 1, 3),
                ).squeeze(1)
                xyz = torch.clamp(parent_xyz + delta, 0.0, 1.0)

            exist_logits.append(exist)
            parent_logits.append(parent)
            root_xyz_logits.append(root_logits)
            delta_logits.append(d_logits)
            delta_values.append(delta)
            xyz_values.append(xyz)

            if teacher_force and target_slots is not None and target_exists is not None:
                src = target_slots[:, slot]
                src_exists = target_exists[:, slot : slot + 1]
                prev_token = torch.cat([src, src_exists], dim=1)
            else:
                parent_idx = torch.argmax(parent, dim=1)
                prev_token = torch.cat(
                    [parent_idx.float().unsqueeze(1) / slot_den, delta, xyz, torch.sigmoid(exist).unsqueeze(1)],
                    dim=1,
                )

        return {
            "exist_logits": torch.stack(exist_logits, dim=1),
            "parent_logits": torch.stack(parent_logits, dim=1),
            "root_xyz_logits": torch.stack(root_xyz_logits, dim=1),
            "delta_logits": torch.stack(delta_logits, dim=1),
            "delta": torch.stack(delta_values, dim=1),
            "xyz": torch.stack(xyz_values, dim=1),
        }


class InitRefineTreeRegressor(nn.Module):
    """Refine a deterministic kmeans-MST skeleton instead of regenerating it.

    The initialization already lives inside the point cloud.  This model keeps
    that geometry as the base representation and predicts only a bounded
    residual, with a zero-initialized output head so step 0 equals the geometric
    initializer.
    """

    def __init__(self, max_slots: int, hidden_dim: int = 384, max_delta: float = 0.01) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.hidden_dim = hidden_dim
        self.max_delta = max_delta
        self.pointcloud_encoder = PointCloudConditionEncoder(hidden_dim)
        self.init_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.slot_embed = nn.Embedding(max_slots, hidden_dim)
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        final = self.refine[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        batch_size = batch["pointcloud"].shape[0]
        device = batch["pointcloud"].device
        init = batch["init_bfs_delta_tree"].view(batch_size, self.max_slots, 7)
        init_mask = batch["init_bfs_delta_tree_mask"].view(batch_size, self.max_slots, 7)
        init_exists = (init_mask[:, :, 0] > 0.0).float()
        init_xyz = torch.clamp(init[:, :, 4:7], 0.0, 1.0)
        global_feat = self.pointcloud_encoder(batch["pointcloud"], batch["pointcloud_mask"])
        slot_ids = torch.arange(self.max_slots, device=device)
        init_token = torch.cat([init, init_exists.unsqueeze(-1)], dim=-1)
        init_feat = self.init_proj(init_token) + self.slot_embed(slot_ids).unsqueeze(0)
        global_feat = global_feat.unsqueeze(1).expand(-1, self.max_slots, -1)
        raw = self.refine(torch.cat([global_feat, init_feat], dim=-1))
        delta = torch.tanh(raw[:, :, :3]) * self.max_delta * init_exists.unsqueeze(-1)
        xyz = torch.clamp(init_xyz + delta, 0.0, 1.0)
        exist_logits = raw[:, :, 3] + (init_exists * 8.0 - 4.0)

        parent_norm = torch.clamp(init[:, :, 0], 0.0, 1.0)
        parent_idx = torch.clamp(
            torch.round(parent_norm * float(max(1, self.max_slots - 1))).long(),
            0,
            self.max_slots - 1,
        )
        parent_logits = torch.full((batch_size, self.max_slots, self.max_slots), -1e4, device=device, dtype=xyz.dtype)
        parent_logits.scatter_(2, parent_idx.unsqueeze(-1), 1e4)
        return {
            "exist_logits": exist_logits,
            "parent_logits": parent_logits,
            "xyz": xyz,
            "init_xyz": init_xyz,
            "refine_delta": delta,
            "init_exists": init_exists,
        }


class InitDiffusionTreeRegressor(nn.Module):
    """Conditional denoiser from geometric init to OBJ skeleton points.

    This is intentionally closer to the ScoreHMR idea than a plain regressor:
    during training we corrupt an intermediate skeleton state between the
    kmeans-MST initializer and the matched GT joints, then predict the clean
    joint locations under point-cloud conditioning.
    """

    def __init__(self, max_slots: int, hidden_dim: int = 384, max_delta: float = 0.20) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.hidden_dim = hidden_dim
        self.max_delta = max_delta
        self.pointcloud_encoder = PointCloudConditionEncoder(hidden_dim)
        self.init_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.noisy_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.slot_embed = nn.Embedding(max_slots, hidden_dim)
        self.denoise_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.xyz_exist_head = nn.Linear(hidden_dim, 4)
        self.parent_head = nn.Linear(hidden_dim, max_slots)
        nn.init.zeros_(self.xyz_exist_head.weight)
        nn.init.zeros_(self.xyz_exist_head.bias)
        nn.init.zeros_(self.parent_head.weight)
        nn.init.zeros_(self.parent_head.bias)

    def _init_state(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch["pointcloud"].shape[0]
        device = batch["pointcloud"].device
        init = batch["init_bfs_delta_tree"].view(batch_size, self.max_slots, 7)
        init_mask = batch["init_bfs_delta_tree_mask"].view(batch_size, self.max_slots, 7)
        init_exists = (init_mask[:, :, 0] > 0.0).float()
        init_xyz = torch.clamp(init[:, :, 4:7], 0.0, 1.0)
        parent_norm = torch.clamp(init[:, :, 0], 0.0, 1.0)
        parent_idx = torch.clamp(
            torch.round(parent_norm * float(max(1, self.max_slots - 1))).long(),
            0,
            self.max_slots - 1,
        )
        parent_logits = torch.full((batch_size, self.max_slots, self.max_slots), -1e4, device=device, dtype=init_xyz.dtype)
        parent_logits.scatter_(2, parent_idx.unsqueeze(-1), 1e4)
        return init, init_xyz, init_exists, parent_logits

    def forward(
        self,
        batch: dict,
        noisy_xyz: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = batch["pointcloud"].shape[0]
        device = batch["pointcloud"].device
        init, init_xyz, init_exists, init_parent_logits = self._init_state(batch)
        if noisy_xyz is None:
            noisy_xyz = init_xyz
        if timestep is None:
            timestep = torch.ones(batch_size, device=device, dtype=init_xyz.dtype)
        timestep = timestep.to(device=device, dtype=init_xyz.dtype).view(batch_size, 1, 1)

        global_feat = self.pointcloud_encoder(batch["pointcloud"], batch["pointcloud_mask"]).unsqueeze(1).expand(-1, self.max_slots, -1)
        slot_ids = torch.arange(self.max_slots, device=device)
        init_token = torch.cat([init, init_exists.unsqueeze(-1)], dim=-1)
        noisy_token = torch.cat([torch.clamp(noisy_xyz, 0.0, 1.0), init_exists.unsqueeze(-1)], dim=-1)
        init_feat = self.init_proj(init_token)
        noisy_feat = self.noisy_proj(noisy_token)
        time_feat = self.time_mlp(timestep.reshape(batch_size, 1)).unsqueeze(1).expand(-1, self.max_slots, -1)
        slot_feat = self.slot_embed(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.denoise_trunk(torch.cat([global_feat, init_feat + slot_feat, noisy_feat, time_feat], dim=-1))
        raw = self.xyz_exist_head(hidden)
        delta = torch.tanh(raw[:, :, :3]) * self.max_delta * init_exists.unsqueeze(-1)
        xyz = torch.clamp(noisy_xyz + delta, 0.0, 1.0)
        # Init slots are only geometric candidates.  The model must learn how
        # many of them should become real joints instead of inheriting init
        # existence, so start with a sparse prior rather than a hard init mask.
        exist_logits = raw[:, :, 3] - 2.0
        parent_norm = torch.clamp(init[:, :, 0], 0.0, 1.0)
        parent_idx = torch.clamp(
            torch.round(parent_norm * float(max(1, self.max_slots - 1))).long(),
            0,
            self.max_slots - 1,
        )
        parent_prior = torch.zeros((batch_size, self.max_slots, self.max_slots), device=device, dtype=xyz.dtype)
        parent_prior.scatter_(2, parent_idx.unsqueeze(-1), 2.0)
        parent_logits = parent_prior + self.parent_head(hidden)
        parent_logits = parent_logits.masked_fill(init_exists.unsqueeze(1) <= 0.0, -1e4)
        return {
            "exist_logits": exist_logits,
            "parent_logits": parent_logits,
            "xyz": xyz,
            "init_xyz": init_xyz,
            "refine_delta": xyz - init_xyz,
            "init_exists": init_exists,
            "init_parent_logits": init_parent_logits,
            "learned_exists": torch.ones((), device=device, dtype=torch.bool),
            "allow_any_parent": torch.ones((), device=device, dtype=torch.bool),
            "postprocess_mst_parent": torch.ones((), device=device, dtype=torch.bool),
        }


class InitAnchorTreeRegressor(nn.Module):
    """Predict a small correction from the geometric point-cloud skeleton.

    The kmeans/MST initializer is currently the strongest signal we have for
    OBJ skeletons.  This head treats it as an anchor: each candidate slot can
    move only within a bounded radius, and the network mainly learns which
    candidates survive plus how they connect.  This is deliberately more
    conservative than free autoregressive generation.
    """

    def __init__(self, max_slots: int, hidden_dim: int = 384, max_delta: float = 0.06) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.max_delta = max_delta
        self.pointcloud_encoder = PointCloudConditionEncoder(hidden_dim)
        self.init_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.slot_embed = nn.Embedding(max_slots, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.xyz_exist_head = nn.Linear(hidden_dim, 4)
        self.parent_head = nn.Linear(hidden_dim, max_slots)
        nn.init.zeros_(self.xyz_exist_head.weight)
        nn.init.zeros_(self.xyz_exist_head.bias)

    def _init_state(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch["pointcloud"].shape[0]
        device = batch["pointcloud"].device
        init = batch["init_bfs_delta_tree"].view(batch_size, self.max_slots, 7)
        init_mask = batch["init_bfs_delta_tree_mask"].view(batch_size, self.max_slots, 7)
        init_exists = (init_mask[:, :, 0] > 0.0).float()
        init_xyz = torch.clamp(init[:, :, 4:7], 0.0, 1.0)
        parent_norm = torch.clamp(init[:, :, 0], 0.0, 1.0)
        parent_idx = torch.clamp(
            torch.round(parent_norm * float(max(1, self.max_slots - 1))).long(),
            0,
            self.max_slots - 1,
        )
        parent_logits = torch.full((batch_size, self.max_slots, self.max_slots), -1e4, device=device, dtype=init_xyz.dtype)
        parent_logits.scatter_(2, parent_idx.unsqueeze(-1), 1e4)
        return init, init_xyz, init_exists, parent_logits

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        batch_size = batch["pointcloud"].shape[0]
        device = batch["pointcloud"].device
        init, init_xyz, init_exists, init_parent_logits = self._init_state(batch)
        global_feat = self.pointcloud_encoder(batch["pointcloud"], batch["pointcloud_mask"]).unsqueeze(1).expand(-1, self.max_slots, -1)
        slot_ids = torch.arange(self.max_slots, device=device)
        init_token = torch.cat([init, init_exists.unsqueeze(-1)], dim=-1)
        init_feat = self.init_proj(init_token) + self.slot_embed(slot_ids).unsqueeze(0)
        hidden = self.trunk(torch.cat([global_feat, init_feat], dim=-1))
        raw = self.xyz_exist_head(hidden)
        delta = torch.tanh(raw[:, :, :3]) * self.max_delta * init_exists.unsqueeze(-1)
        xyz = torch.clamp(init_xyz + delta, 0.0, 1.0)
        # Slightly sparse prior: init candidates are candidates, not all final
        # joints.  Training can still turn them on through matched GT slots.
        exist_logits = raw[:, :, 3] - 1.0
        parent_logits = self.parent_head(hidden)
        parent_logits = parent_logits.masked_fill(init_exists.unsqueeze(1) <= 0.0, -1e4)
        return {
            "exist_logits": exist_logits,
            "parent_logits": parent_logits,
            "xyz": xyz,
            "init_xyz": init_xyz,
            "refine_delta": delta,
            "init_exists": init_exists,
            "init_parent_logits": init_parent_logits,
            "learned_exists": torch.ones((), device=device, dtype=torch.bool),
            "allow_any_parent": torch.ones((), device=device, dtype=torch.bool),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


def kinematic_pose_loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    model: KinematicPoseRegressor,
) -> torch.Tensor:
    target = batch["target_raw"].view(
        -1,
        model.max_keypoints,
        model.values_per_keypoint,
    )
    mask = batch["target_mask"].view_as(target)
    target_points = target[:, :, : model.spatial_dim]
    coordinate_mask = mask[:, :, : model.spatial_dim]
    coordinate_denom = coordinate_mask.sum().clamp_min(1.0)
    coordinate_loss = (
        F.smooth_l1_loss(
            output["points"],
            target_points,
            reduction="none",
            beta=0.02,
        )
        * coordinate_mask
    ).sum() / coordinate_denom

    children = torch.as_tensor(
        model.order[1:],
        dtype=torch.long,
        device=target.device,
    )
    parent_indices = torch.as_tensor(
        [int(model.parents[joint]) for joint in model.order[1:]],
        dtype=torch.long,
        device=target.device,
    )
    target_delta = target_points[:, children] - target_points[:, parent_indices]
    predicted_delta = output["points"][:, children] - output["points"][:, parent_indices]
    target_length = torch.linalg.vector_norm(target_delta, dim=-1)
    predicted_length = torch.linalg.vector_norm(predicted_delta, dim=-1)
    point_valid = torch.all(coordinate_mask > 0.0, dim=-1)
    bone_valid = (
        point_valid[:, children]
        & point_valid[:, parent_indices]
        & (target_length > 1e-5)
    )
    if bone_valid.any():
        target_direction = F.normalize(target_delta[bone_valid], dim=-1, eps=1e-6)
        predicted_direction = F.normalize(predicted_delta[bone_valid], dim=-1, eps=1e-6)
        direction_loss = (1.0 - (target_direction * predicted_direction).sum(dim=-1)).mean()
        log_length_error = (
            torch.log(predicted_length[bone_valid].clamp_min(1e-5))
            - torch.log(target_length[bone_valid].clamp_min(1e-5))
        )
        length_loss = F.smooth_l1_loss(
            log_length_error,
            torch.zeros_like(log_length_error),
            beta=0.1,
        )
    else:
        direction_loss = output["points"].sum() * 0.0
        length_loss = output["lengths"].sum() * 0.0

    score_index = model.spatial_dim
    score_mask = mask[:, :, score_index] > 0.0
    if score_mask.any():
        score_target = target[:, :, score_index].clamp(0.0, 1.0)
        score_loss = F.binary_cross_entropy_with_logits(
            output["score_logits"][score_mask],
            score_target[score_mask],
        )
    else:
        score_loss = output["score_logits"].sum() * 0.0

    xy = output["points"][:, : model.active_count, :2]
    range_loss = (F.relu(-xy) + F.relu(xy - 1.0)).mean()
    prior_loss = output["length_residual"].pow(2).mean()
    scale_loss = torch.log(output["global_scale"].clamp_min(1e-6)).pow(2).mean()
    return (
        3.0 * coordinate_loss
        + 0.25 * direction_loss
        + 0.10 * length_loss
        + 0.10 * score_loss
        + 0.05 * range_loss
        + 0.005 * prior_loss
        + 0.005 * scale_loss
    )


def obj_bfs_targets(batch: dict, max_slots: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch["target_raw"].view(-1, max_slots, 7)
    mask = batch["target_mask"].view(-1, max_slots, 7)
    exists = (mask[:, :, 0] > 0.0).float()
    parent = torch.clamp(torch.round(target[:, :, 0] * float(max(1, max_slots - 1))).long(), 0, max_slots - 1)
    xyz = torch.clamp(target[:, :, 4:7], 0.0, 1.0)
    delta = torch.clamp(target[:, :, 1:4], -1.0, 1.0)
    return exists, parent, xyz, delta


def growing_tree_loss(output: dict[str, torch.Tensor], batch: dict, max_slots: int) -> torch.Tensor:
    exists, parent, xyz, delta = obj_bfs_targets(batch, max_slots)
    valid = exists > 0.0
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exists)
    if valid.any():
        parent_loss = F.cross_entropy(output["parent_logits"][valid], parent[valid])
        xyz_loss = F.smooth_l1_loss(output["xyz"][valid], xyz[valid])
        pred_parent_xyz = torch.gather(output["xyz"], 1, parent.clamp(0, max_slots - 1).unsqueeze(-1).expand(-1, -1, 3))
        pred_delta = torch.clamp(output["xyz"] - pred_parent_xyz, -1.0, 1.0)
        delta_loss = F.smooth_l1_loss(pred_delta[valid], delta[valid])
    else:
        parent_loss = output["parent_logits"].sum() * 0.0
        xyz_loss = output["xyz"].sum() * 0.0
        delta_loss = output["xyz"].sum() * 0.0
    return xyz_loss + 0.5 * delta_loss + 0.2 * parent_loss + 0.1 * exist_loss


def discrete_growing_tree_loss(output: dict[str, torch.Tensor], batch: dict, max_slots: int) -> torch.Tensor:
    exists, parent, xyz, _delta = obj_bfs_targets(batch, max_slots)
    valid = exists > 0.0
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exists)
    if valid.any():
        slot_ids = torch.arange(max_slots, device=xyz.device).view(1, -1)
        slot_weight = torch.ones_like(exists)
        slot_weight = torch.where(slot_ids < 16, torch.full_like(slot_weight, 4.0), slot_weight)
        slot_weight = torch.where((slot_ids >= 16) & (slot_ids < 32), torch.full_like(slot_weight, 2.0), slot_weight)
        valid_weight = slot_weight[valid]
        valid_weight = valid_weight / valid_weight.mean().clamp_min(1e-6)
        parent_ce = F.cross_entropy(output["parent_logits"][valid], parent[valid], reduction="none")
        parent_loss = (parent_ce * valid_weight).mean()
        xyz_bins = output["xyz_logits"].shape[-1]
        xyz_target = torch.clamp(torch.round(xyz * float(xyz_bins - 1)).long(), 0, xyz_bins - 1)
        xyz_ce = F.cross_entropy(output["xyz_logits"][valid].reshape(-1, xyz_bins), xyz_target[valid].reshape(-1), reduction="none")
        xyz_weight = valid_weight[:, None].expand(-1, 3).reshape(-1)
        xyz_loss = (xyz_ce * xyz_weight).mean()
        bin_values = torch.linspace(0.0, 1.0, xyz_bins, device=xyz.device, dtype=output["xyz_logits"].dtype)
        xyz_soft = (torch.softmax(output["xyz_logits"], dim=-1) * bin_values.view(1, 1, 1, -1)).sum(dim=-1)
        xyz_smooth = F.smooth_l1_loss(xyz_soft[valid], xyz[valid], reduction="none").mean(dim=1)
        xyz_smooth_loss = (xyz_smooth * valid_weight).mean()
    else:
        parent_loss = output["parent_logits"].sum() * 0.0
        xyz_loss = output["xyz_logits"].sum() * 0.0
        xyz_smooth_loss = output["xyz_logits"].sum() * 0.0
    return xyz_loss + 0.5 * parent_loss + 0.2 * xyz_smooth_loss + 0.1 * exist_loss


def discrete_growing_tree_loss_base(output: dict[str, torch.Tensor], batch: dict, max_slots: int) -> torch.Tensor:
    exists, parent, xyz, _delta = obj_bfs_targets(batch, max_slots)
    valid = exists > 0.0
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exists)
    if valid.any():
        parent_loss = F.cross_entropy(output["parent_logits"][valid], parent[valid])
        xyz_bins = output["xyz_logits"].shape[-1]
        xyz_target = torch.clamp(torch.round(xyz * float(xyz_bins - 1)).long(), 0, xyz_bins - 1)
        xyz_loss = F.cross_entropy(output["xyz_logits"][valid].reshape(-1, xyz_bins), xyz_target[valid].reshape(-1))
    else:
        parent_loss = output["parent_logits"].sum() * 0.0
        xyz_loss = output["xyz_logits"].sum() * 0.0
    return xyz_loss + 0.2 * parent_loss + 0.1 * exist_loss


def delta_growing_tree_loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    max_slots: int,
    xyz_weight: float = 0.25,
) -> torch.Tensor:
    exists, parent, xyz, delta = obj_bfs_targets(batch, max_slots)
    valid = exists > 0.0
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exists)
    if valid.any():
        parent_loss = F.cross_entropy(output["parent_logits"][valid], parent[valid])
        root_valid = valid[:, 0]
        if root_valid.any():
            xyz_bins = output["root_xyz_logits"].shape[-1]
            root_target = torch.clamp(torch.round(xyz[:, 0] * float(xyz_bins - 1)).long(), 0, xyz_bins - 1)
            root_loss = F.cross_entropy(output["root_xyz_logits"][:, 0][root_valid].reshape(-1, xyz_bins), root_target[root_valid].reshape(-1))
        else:
            root_loss = output["root_xyz_logits"].sum() * 0.0
        branch_valid = valid.clone()
        branch_valid[:, 0] = False
        if branch_valid.any():
            delta_bins = output["delta_logits"].shape[-1]
            delta_target = torch.clamp(torch.round((delta + 1.0) * 0.5 * float(delta_bins - 1)).long(), 0, delta_bins - 1)
            delta_loss = F.cross_entropy(output["delta_logits"][branch_valid].reshape(-1, delta_bins), delta_target[branch_valid].reshape(-1))
        else:
            delta_loss = output["delta_logits"].sum() * 0.0
        xyz_loss = F.smooth_l1_loss(output["xyz"][valid], xyz[valid])
    else:
        parent_loss = output["parent_logits"].sum() * 0.0
        root_loss = output["root_xyz_logits"].sum() * 0.0
        delta_loss = output["delta_logits"].sum() * 0.0
        xyz_loss = output["xyz"].sum() * 0.0
    return root_loss + delta_loss + 0.2 * parent_loss + xyz_weight * xyz_loss + 0.1 * exist_loss


def init_refine_tree_loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    max_slots: int,
    delta_weight: float = 20.0,
    outside_weight: float = 0.2,
) -> torch.Tensor:
    exists, _parent, xyz, _delta = obj_bfs_targets(batch, max_slots)
    valid = exists > 0.0
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exists)
    pred_valid = output.get("init_exists", torch.ones_like(exists)) > 0.0
    chamfer_losses = []
    for batch_idx in range(xyz.shape[0]):
        pred_points = output["xyz"][batch_idx][pred_valid[batch_idx]]
        target_points = xyz[batch_idx][valid[batch_idx]]
        if pred_points.numel() == 0 or target_points.numel() == 0:
            continue
        distances = torch.cdist(pred_points.float(), target_points.float(), p=2)
        chamfer_losses.append(distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    if chamfer_losses:
        xyz_loss = torch.stack(chamfer_losses).mean()
    else:
        xyz_loss = output["xyz"].sum() * 0.0
    if pred_valid.any():
        delta_loss = (output["refine_delta"][pred_valid] ** 2).mean()
        outside = F.relu(output["xyz"] - 1.0) + F.relu(-output["xyz"])
        outside_loss = outside[pred_valid].mean()
    else:
        delta_loss = output["refine_delta"].sum() * 0.0
        outside_loss = output["xyz"].sum() * 0.0
    return xyz_loss + delta_weight * delta_loss + outside_weight * outside_loss + 0.02 * exist_loss


def _assign_init_slots_to_targets(
    init_xyz: torch.Tensor,
    init_valid: torch.Tensor,
    target_xyz: torch.Tensor,
    target_valid: torch.Tensor,
    target_parent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matched = init_xyz.detach().clone()
    matched_valid = torch.zeros_like(init_valid, dtype=torch.bool)
    matched_parent = torch.zeros_like(target_parent)
    for batch_idx in range(init_xyz.shape[0]):
        src_idx = torch.nonzero(init_valid[batch_idx], as_tuple=False).flatten()
        dst_idx = torch.nonzero(target_valid[batch_idx], as_tuple=False).flatten()
        if src_idx.numel() == 0 or dst_idx.numel() == 0:
            continue
        src = init_xyz[batch_idx, src_idx]
        dst = target_xyz[batch_idx, dst_idx]
        distances = torch.cdist(src.float(), dst.float(), p=2)
        flat_order = torch.argsort(distances.reshape(-1))
        used_src: set[int] = set()
        used_dst: set[int] = set()
        target_to_init: dict[int, int] = {}
        for flat in flat_order.tolist():
            src_local = int(flat // distances.shape[1])
            dst_local = int(flat % distances.shape[1])
            if src_local in used_src or dst_local in used_dst:
                continue
            src_slot = int(src_idx[src_local].item())
            dst_slot = int(dst_idx[dst_local].item())
            used_src.add(src_local)
            used_dst.add(dst_local)
            target_to_init[dst_slot] = src_slot
            matched[batch_idx, src_slot] = target_xyz[batch_idx, dst_slot].to(dtype=matched.dtype)
            matched_valid[batch_idx, src_slot] = True
            if len(used_dst) == int(dst_idx.numel()):
                break

        for dst_slot, src_slot in target_to_init.items():
            parent_slot = int(target_parent[batch_idx, dst_slot].item())
            if parent_slot in target_to_init:
                matched_parent[batch_idx, src_slot] = target_to_init[parent_slot]
            elif target_to_init:
                parent_xyz = target_xyz[batch_idx, parent_slot].view(1, 3)
                assigned_init = torch.tensor(list(target_to_init.values()), device=init_xyz.device, dtype=torch.long)
                nearest = torch.cdist(init_xyz[batch_idx, assigned_init].float(), parent_xyz.float(), p=2).argmin(dim=0)
                matched_parent[batch_idx, src_slot] = assigned_init[int(nearest.item())]
            else:
                matched_parent[batch_idx, src_slot] = src_slot
    return matched, matched_valid, matched_parent


def init_diffusion_tree_loss(
    model: InitDiffusionTreeRegressor,
    batch: dict,
    max_slots: int,
    max_sigma: float = 0.08,
    delta_weight: float = 0.2,
) -> torch.Tensor:
    _exists, _parent, target_xyz, _delta = obj_bfs_targets(batch, max_slots)
    target_valid = _exists > 0.0
    with torch.no_grad():
        _init, init_xyz, init_exists, _parent_logits = model._init_state(batch)
        init_valid = init_exists > 0.0
        matched_xyz, matched_valid, matched_parent = _assign_init_slots_to_targets(init_xyz, init_valid, target_xyz, target_valid, _parent)
        t = torch.rand(init_xyz.shape[0], device=init_xyz.device, dtype=init_xyz.dtype)
        sigma = (0.005 + t * max_sigma).view(-1, 1, 1)
        bridge = (1.0 - t.view(-1, 1, 1)) * matched_xyz + t.view(-1, 1, 1) * init_xyz
        noisy_xyz = torch.clamp(bridge + torch.randn_like(init_xyz) * sigma * init_exists.unsqueeze(-1), 0.0, 1.0)

    output = model(batch, noisy_xyz=noisy_xyz, timestep=t)
    exist_target = matched_valid.float()
    valid_candidate = init_valid.float()
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exist_target, reduction="none")
    exist_loss = (exist_loss * valid_candidate).sum() / valid_candidate.sum().clamp_min(1.0)
    if matched_valid.any():
        xyz_loss = F.smooth_l1_loss(output["xyz"][matched_valid], matched_xyz[matched_valid])
        delta_loss = ((output["xyz"][matched_valid] - init_xyz[matched_valid]) ** 2).mean()
        parent_loss = F.cross_entropy(output["parent_logits"][matched_valid], matched_parent[matched_valid])
    else:
        xyz_loss = output["xyz"].sum() * 0.0
        delta_loss = output["xyz"].sum() * 0.0
        parent_loss = output["parent_logits"].sum() * 0.0
    return xyz_loss + 0.5 * parent_loss + delta_weight * delta_loss + 0.2 * exist_loss


def init_anchor_tree_loss(
    model: InitAnchorTreeRegressor,
    output: dict[str, torch.Tensor],
    batch: dict,
    max_slots: int,
    delta_weight: float = 1.0,
) -> torch.Tensor:
    _exists, _parent, target_xyz, _delta = obj_bfs_targets(batch, max_slots)
    target_valid = _exists > 0.0
    with torch.no_grad():
        _init, init_xyz, init_exists, _parent_logits = model._init_state(batch)
        init_valid = init_exists > 0.0
        matched_xyz, matched_valid, matched_parent = _assign_init_slots_to_targets(
            init_xyz,
            init_valid,
            target_xyz,
            target_valid,
            _parent,
        )

    candidate_weight = init_valid.float()
    exist_target = matched_valid.float()
    exist_loss = F.binary_cross_entropy_with_logits(output["exist_logits"], exist_target, reduction="none")
    exist_loss = (exist_loss * candidate_weight).sum() / candidate_weight.sum().clamp_min(1.0)
    if matched_valid.any():
        xyz_loss = F.smooth_l1_loss(output["xyz"][matched_valid], matched_xyz[matched_valid])
        parent_loss = F.cross_entropy(output["parent_logits"][matched_valid], matched_parent[matched_valid])
        delta_loss = (output["refine_delta"][matched_valid] ** 2).mean()
    else:
        xyz_loss = output["xyz"].sum() * 0.0
        parent_loss = output["parent_logits"].sum() * 0.0
        delta_loss = output["refine_delta"].sum() * 0.0
    return xyz_loss + 0.5 * parent_loss + delta_weight * delta_loss + 0.2 * exist_loss


def growing_tree_to_flat(
    output: dict[str, torch.Tensor],
    max_slots: int,
    threshold: float = 0.5,
    min_slots: int = 8,
    max_visible_slots: int = 16,
    preview_strategy: str = "topk",
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.clamp(output["xyz"], 0.0, 1.0)
    batch_size = xyz.shape[0]
    exist_prob = torch.sigmoid(output["exist_logits"])
    learned_exists = bool(output.get("learned_exists", False))
    allow_any_parent = bool(output.get("allow_any_parent", False))
    if learned_exists:
        keep = exist_prob >= threshold
        candidate_keep = output.get("init_exists", torch.ones_like(exist_prob)) > 0.0
        keep = keep & candidate_keep
        for batch_idx in range(batch_size):
            if int(keep[batch_idx].sum().item()) < min_slots:
                available = torch.nonzero(candidate_keep[batch_idx], as_tuple=False).flatten()
                if available.numel() == 0:
                    continue
                scores = exist_prob[batch_idx, available]
                top = available[torch.topk(scores, k=min(min_slots, int(available.numel())), dim=0).indices]
                keep[batch_idx, top] = True
    elif "init_exists" in output:
        keep = output["init_exists"] > 0.0
    elif preview_strategy == "first_slots":
        # For autoregressive BFS decoding, slot order is semantic.  This view
        # checks whether the early trunk itself is being learned.
        keep = torch.zeros((batch_size, max_slots), device=xyz.device, dtype=torch.bool)
        keep[:, : min(max_visible_slots, max_slots)] = True
    elif preview_strategy == "prefix":
        # BFS slots form a tree prefix: children depend on earlier parent
        # slots.  Top-k confidence can drop ancestors while keeping leaves,
        # which makes a good sequence look disconnected.  Keep a contiguous
        # prefix instead, stopping at the first low-existence slot after the
        # minimum preview length.
        keep = torch.zeros((batch_size, max_slots), device=xyz.device, dtype=torch.bool)
        max_count = min(max_visible_slots, max_slots)
        min_count = min(min_slots, max_count)
        for batch_idx in range(batch_size):
            count = max_count
            low_slots = torch.nonzero(exist_prob[batch_idx, :max_count] < threshold, as_tuple=False).flatten()
            low_slots = low_slots[low_slots >= min_count]
            if low_slots.numel() > 0:
                count = int(low_slots[0].item())
            count = max(min_count, min(count, max_count))
            keep[batch_idx, :count] = True
    else:
        keep = exist_prob >= threshold
        keep[:, 0] = True
        top_k = min(max(min_slots, max_visible_slots), max_slots)
        if top_k > 0:
            top_idx = torch.topk(exist_prob, k=top_k, dim=1).indices
            bounded_keep = torch.zeros_like(keep)
            bounded_keep.scatter_(1, top_idx, True)
            keep = keep & bounded_keep
            min_idx = torch.topk(exist_prob, k=min(min_slots, max_slots), dim=1).indices
            keep.scatter_(1, min_idx, True)
            keep[:, 0] = True

    if bool(output.get("postprocess_mst_parent", False)):
        parent = torch.arange(max_slots, device=xyz.device).view(1, -1).expand(batch_size, -1).clone()
        for batch_idx in range(batch_size):
            visible = torch.nonzero(keep[batch_idx], as_tuple=False).flatten()
            if visible.numel() <= 1:
                continue
            points = xyz[batch_idx, visible].float()
            centroid = points.mean(dim=0, keepdim=True)
            root_local = int(torch.cdist(points, centroid, p=2).argmin().item())
            selected = [root_local]
            remaining = set(range(int(visible.numel())))
            remaining.remove(root_local)
            parent[batch_idx, visible[root_local]] = visible[root_local]
            distances = torch.cdist(points, points, p=2)
            while remaining:
                best_src = selected[0]
                best_dst = next(iter(remaining))
                best_dist = float("inf")
                for src in selected:
                    rem = torch.tensor(sorted(remaining), device=xyz.device, dtype=torch.long)
                    dst_pos = int(torch.argmin(distances[src, rem]).item())
                    dst = int(rem[dst_pos].item())
                    dist = float(distances[src, dst].item())
                    if dist < best_dist:
                        best_dist = dist
                        best_src = src
                        best_dst = dst
                parent[batch_idx, visible[best_dst]] = visible[best_src]
                selected.append(best_dst)
                remaining.remove(best_dst)
    else:
        parent = torch.argmax(output["parent_logits"], dim=-1)
    slot_ids = torch.arange(max_slots, device=xyz.device).view(1, -1)
    if not allow_any_parent:
        parent = torch.minimum(parent, slot_ids)
    parent_norm = parent.float() / float(max(1, max_slots - 1))
    parent_xyz = torch.gather(xyz, 1, parent.clamp(0, max_slots - 1).unsqueeze(-1).expand(-1, -1, 3))
    delta = torch.clamp(xyz - parent_xyz, -1.0, 1.0)
    flat = torch.cat([parent_norm.unsqueeze(-1), delta, xyz], dim=-1).reshape(batch_size, max_slots * 7)
    mask = keep.float().unsqueeze(-1).expand(-1, -1, 7).reshape(batch_size, max_slots * 7)
    return flat, mask


def _select_fixed_eval_indices(dataset: StructureDataset, task: str, max_samples: int) -> list[int]:
    """Pick stable, useful qualitative eval samples.

    For human keypoint tasks, the first validation rows are often poor visual
    examples. Prefer a single frame that has multiple extracted people so the
    preview tests both crop conditioning and multi-person handling.
    """
    n = len(dataset)
    if n == 0:
        return []
    if task in {"dwpose", "mediapipe"}:
        groups: dict[tuple[str, str], list[int]] = {}
        for idx, (path, frame_key) in enumerate(zip(dataset.wsv_path, dataset.frame_key)):
            groups.setdefault((str(path), str(frame_key)), []).append(idx)
        candidates = []
        for key, indices in groups.items():
            if len(indices) < 2:
                continue
            sorted_indices = sorted(indices, key=lambda i: (-int(dataset.valid_points[i]), int(dataset.person_id[i]), i))
            score = sum(int(dataset.valid_points[i]) for i in sorted_indices[:max_samples])
            candidates.append((score, len(indices), key, sorted_indices[:max_samples]))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            return list(candidates[0][3])
    order = np.argsort(-dataset.valid_points)
    return [int(i) for i in order[: min(max_samples, n)]]


def _select_eval_indices(
    dataset: StructureDataset,
    task: str,
    step: int,
    fixed_samples: int = 2,
    random_samples: int = 2,
) -> list[int]:
    fixed = _select_fixed_eval_indices(dataset, task, fixed_samples)
    n = len(dataset)
    remaining = [idx for idx in range(n) if idx not in set(fixed)]
    if random_samples > 0 and remaining:
        rng = np.random.default_rng(20260523 + int(step))
        pick_count = min(random_samples, len(remaining))
        random_pick = rng.choice(np.asarray(remaining, dtype=np.int64), size=pick_count, replace=False)
        fixed.extend(int(idx) for idx in random_pick.tolist())
    return fixed


def _write_single_wsg(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".npz")
    np.savez_compressed(tmp_path, wsg_data=frame.astype(np.float32, copy=False))
    tmp_path.replace(path)


def _load_source_frame(path: str, frame_key: str) -> np.ndarray:
    with np.load(path) as data:
        if frame_key == "__single__":
            frame = data.get("wsg_data", data.get("data"))
        else:
            frame = data[frame_key]
        return np.asarray(frame, dtype=np.float32).copy()


def _base_visual_frame(source_frame: np.ndarray) -> np.ndarray:
    frame = np.full_like(source_frame, MASK_VALUE, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        frame[:, :, :3] = source_frame[:, :, :3]
    return frame


def _scatter_keypoints(
    frame: np.ndarray,
    task: str,
    flat_values: np.ndarray,
    flat_mask: np.ndarray,
    person_id: int,
    max_keypoints: int,
    bbox: np.ndarray | None = None,
) -> None:
    height, width = frame.shape[:2]
    if bbox is not None and bbox.shape[0] >= 4 and (bbox[2] > 0.0 and bbox[3] > 0.0):
        box_x, box_y, box_w, box_h = [float(v) for v in bbox[:4]]
    else:
        box_x, box_y, box_w, box_h = 0.0, 0.0, float(width - 1), float(height - 1)
    values_per_keypoint = 3 if task == "dwpose" else 4
    values = flat_values.reshape(max_keypoints, values_per_keypoint)
    mask = flat_mask.reshape(max_keypoints, values_per_keypoint)
    for key_id in range(max_keypoints):
        if mask[key_id, 0] <= 0.0 or mask[key_id, 1] <= 0.0:
            continue
        x = int(round(box_x + float(np.clip(values[key_id, 0], 0.0, 1.0)) * max(1.0, box_w - 1.0)))
        y = int(round(box_y + float(np.clip(values[key_id, 1], 0.0, 1.0)) * max(1.0, box_h - 1.0)))
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if task == "dwpose":
            frame[y, x, DWPOSE_PERSON_ID_CHANNEL] = float(person_id)
            frame[y, x, DWPOSE_KEYPOINT_ID_CHANNEL] = float(key_id)
            # For generated structure previews, confidence is a visibility gate.
            # Keep valid predicted/GT keypoints visible instead of letting a noisy
            # regressed score hide otherwise-correct coordinates.
            frame[y, x, DWPOSE_SCORE_CHANNEL] = 1.0
        else:
            frame[y, x, MEDIAPIPE_PERSON_ID_CHANNEL] = float(person_id)
            frame[y, x, MEDIAPIPE_JOINT_ID_CHANNEL] = float(key_id)
            frame[y, x, MEDIAPIPE_Z_CHANNEL] = float(values[key_id, 2]) if mask[key_id, 2] > 0.0 else 0.0
            frame[y, x, MEDIAPIPE_VISIBILITY_CHANNEL] = 1.0


def _scatter_obj(frame: np.ndarray, flat_values: np.ndarray, flat_mask: np.ndarray, max_slots: int) -> None:
    height, width = frame.shape[:2]
    values = flat_values.reshape(max_slots, 2 + len(OBJ_VALUE_CHANNELS))
    mask = flat_mask.reshape(max_slots, 2 + len(OBJ_VALUE_CHANNELS))
    for slot in range(max_slots):
        if mask[slot, 0] <= 0.0 or mask[slot, 1] <= 0.0:
            continue
        x = int(round(float(np.clip(values[slot, 0], 0.0, 1.0)) * max(1, width - 1)))
        y = int(round(float(np.clip(values[slot, 1], 0.0, 1.0)) * max(1, height - 1)))
        frame[y, x, OBJ_ID_CHANNEL] = 1.0
        for offset, channel in enumerate(OBJ_VALUE_CHANNELS):
            if mask[slot, 2 + offset] > 0.0:
                frame[y, x, channel] = float(values[slot, 2 + offset])


def _run_wsg_visualizer(wsg_path: Path, out_dir: Path, task: str) -> None:
    """Run only the relevant WSG reader pipeline.

    The one-stop visualizer executes every WSG reader, which is useful for
    manual inspection but far too slow during training eval. These sparse
    structure tasks only need their own reader output.
    """
    reader_root = WSM_ROOT / "wsg_reader"
    pipeline_by_task = {
        "dwpose": ("pipeline01_read_dwpose_from_wsg.py", ["vis", "vis_overlay", "save"]),
        "mediapipe": ("pipeline03_read_mediapipe_from_wsg.py", ["vis", "vis_overlay", "save"]),
        "obj": ("pipeline15_read_object_from_wsg.py", ["vis", "save"]),
    }
    if task not in pipeline_by_task:
        return
    pipeline_file, modes = pipeline_by_task[task]
    pipeline_path = reader_root / pipeline_file
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if str(reader_root) not in sys.path:
            sys.path.insert(0, str(reader_root))
        spec = importlib.util.spec_from_file_location(f"wsg_reader_{task}", pipeline_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {pipeline_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "process_single_wsg_file"):
            module.process_single_wsg_file((str(wsg_path), str(out_dir), modes))
        else:
            raise AttributeError(f"{pipeline_path} has no supported WSG reader entry point")
    except Exception as exc:
        print(f"[visualize] failed for {wsg_path}: {exc}", flush=True)


def _save_obj_preview(frame: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = frame[:, :, :3]
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    canvas = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    valid = frame[:, :, OBJ_VALUE_CHANNELS[0]] != MASK_VALUE
    ys, xs = np.where(valid)
    joint_positions: dict[int, tuple[int, int]] = {}
    bones: list[tuple[int, int]] = []

    for y, x in zip(ys, xs):
        class_value = int(round(float(frame[y, x, 112])))
        if class_value == 3:
            joint_id = int(round(float(frame[y, x, 113])))
            joint_positions[joint_id] = (int(x), int(y))
        elif class_value == 4:
            start_joint = int(round(float(frame[y, x, 114])))
            end_joint = int(round(float(frame[y, x, 115])))
            bones.append((start_joint, end_joint))

    for start_joint, end_joint in bones:
        if start_joint in joint_positions and end_joint in joint_positions:
            draw.line((joint_positions[start_joint], joint_positions[end_joint]), fill=(255, 80, 80), width=2)

    for joint_id, (x, y) in joint_positions.items():
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(64, 220, 255), outline=(0, 0, 0))
        draw.text((x + 4, y - 4), str(joint_id), fill=(255, 255, 255))

    if not joint_positions:
        palette = [
            (255, 64, 64),
            (64, 180, 255),
            (64, 255, 128),
            (255, 220, 64),
            (220, 64, 255),
        ]
        for y, x in zip(ys, xs):
            class_value = frame[y, x, OBJ_VALUE_CHANNELS[0]]
            color = palette[int(abs(round(float(class_value)))) % len(palette)]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    canvas.save(out_path)


def _rgb_image_from_source(source_frame: np.ndarray) -> Image.Image:
    rgb = source_frame[:, :, :3].astype(np.float32)
    valid = rgb != MASK_VALUE
    if np.any(valid) and float(np.nanmax(rgb[valid])) > 1.5:
        rgb[valid] = rgb[valid] / 255.0
    rgb = np.where(valid, rgb, 0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def _save_keypoint_overlay(frame: np.ndarray, source_frame: np.ndarray, task: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = _rgb_image_from_source(source_frame)
    draw = ImageDraw.Draw(canvas)
    if task == "dwpose":
        id_channel = DWPOSE_KEYPOINT_ID_CHANNEL
        score_channel = DWPOSE_SCORE_CHANNEL
    elif task == "mediapipe":
        id_channel = MEDIAPIPE_JOINT_ID_CHANNEL
        score_channel = MEDIAPIPE_VISIBILITY_CHANNEL
    else:
        return
    valid = frame[:, :, id_channel] != MASK_VALUE
    ys, xs = np.where(valid)
    palette = [(64, 200, 255), (255, 120, 80), (120, 255, 140), (255, 220, 80), (210, 130, 255)]
    for y, x in zip(ys, xs):
        score = frame[y, x, score_channel]
        if score == MASK_VALUE:
            continue
        key_id = int(round(float(frame[y, x, id_channel])))
        color = palette[key_id % len(palette)]
        draw.ellipse((int(x) - 3, int(y) - 3, int(x) + 3, int(y) + 3), fill=color, outline=(0, 0, 0))
        if key_id < 32:
            draw.text((int(x) + 4, int(y) - 4), str(key_id), fill=(255, 255, 255))
    canvas.save(out_path)


def _save_obj_points_overlay(
    values: np.ndarray,
    mask: np.ndarray,
    source_frame: np.ndarray,
    bbox: np.ndarray,
    out_path: Path,
    title: str,
    parents: dict[int, int] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if values.shape[-1] == 3:
        points_all = values.reshape(-1, 3)
        mask_all = mask.reshape(-1, 3)
        valid = np.all(mask_all[:, :3] > 0.0, axis=1)
        points = np.clip(points_all[:, :3], 0.0, 1.0)
    else:
        values_all = values.reshape(-1, 7)
        mask_all = mask.reshape(-1, 7)
        valid = np.all(mask_all[:, :7] > 0.0, axis=1)
        points = np.clip(values_all[:, 4:7], 0.0, 1.0)

    crop = _crop_rgb_from_source(source_frame, bbox).resize((224, 224), Image.BICUBIC)
    draw = ImageDraw.Draw(crop)
    valid_slots = np.where(valid)[0]
    projected: dict[int, tuple[int, int]] = {}
    for slot in valid_slots:
        point = points[int(slot)]
        x = int(round(float(point[0]) * 223.0))
        y = int(round((1.0 - float(point[1])) * 223.0))
        projected[int(slot)] = (x, y)

    if parents:
        for slot, parent_slot in parents.items():
            if slot == parent_slot:
                continue
            if slot in projected and parent_slot in projected:
                draw.line((projected[parent_slot], projected[slot]), fill=(255, 80, 80), width=2)

    palette = [(64, 200, 255), (255, 120, 80), (120, 255, 140), (255, 220, 80), (210, 130, 255)]
    for slot in valid_slots:
        x, y = projected[int(slot)]
        color = palette[int(slot) % len(palette)]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(0, 0, 0))
        if slot < 32:
            draw.text((x + 5, y - 5), str(int(slot)), fill=(255, 255, 255))
    draw.text((8, 8), f"{title}: xy overlay", fill=(255, 255, 255))
    crop.save(out_path)


def _extract_obj_vertex_pointcloud(source_frame: np.ndarray, person_id: int) -> np.ndarray:
    if source_frame.ndim != 3 or source_frame.shape[-1] <= 116:
        return np.zeros((0, 3), dtype=np.float32)
    object_channel = source_frame[:, :, OBJ_ID_CHANNEL]
    class_channel = source_frame[:, :, 112]
    valid = (object_channel != MASK_VALUE) & (np.rint(class_channel).astype(np.int32) == 1)
    if np.any(valid & (np.rint(object_channel).astype(np.int32) == int(person_id))):
        valid = valid & (np.rint(object_channel).astype(np.int32) == int(person_id))
    coords = np.asarray(source_frame[:, :, 114:117][valid], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    keep = np.isfinite(coords).all(axis=1) & ~np.any(coords == MASK_VALUE, axis=1)
    return coords[keep]


def _save_obj_pointcloud_tree_views(
    values: np.ndarray,
    mask: np.ndarray,
    source_frame: np.ndarray,
    person_id: int,
    out_path: Path,
    title: str,
    parents: dict[int, int] | None = None,
) -> None:
    """Save point-cloud + skeleton overlap in the same normalized 3D space."""
    values = values.reshape(-1, 7)
    mask = mask.reshape(-1, 7)
    valid = np.all(mask[:, :7] > 0.0, axis=1)
    points = np.clip(values[:, 4:7], 0.0, 1.0) if np.any(valid) else np.zeros((values.shape[0], 3), dtype=np.float32)
    valid_slots = np.where(valid)[0]

    pc_raw = _extract_obj_vertex_pointcloud(source_frame, person_id)
    if pc_raw.shape[0] >= 3:
        mins = pc_raw.min(axis=0)
        maxs = pc_raw.max(axis=0)
        scale = np.maximum(maxs - mins, 1e-6)
        pc = np.clip((pc_raw - mins) / scale, 0.0, 1.0)
    else:
        pc = np.zeros((0, 3), dtype=np.float32)

    views = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
    panel_w, panel_h = 256, 256
    canvas = Image.new("RGB", (panel_w * len(views), panel_h), (16, 16, 20))
    palette = [(64, 200, 255), (255, 120, 80), (120, 255, 140), (255, 220, 80), (210, 130, 255)]
    for view_idx, (name, axes) in enumerate(views):
        panel = Image.new("RGB", (panel_w, panel_h), (12, 12, 16))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((20, 20, panel_w - 20, panel_h - 20), outline=(72, 72, 80))
        draw.text((8, 8), f"{title}: pointcloud + skeleton {name}", fill=(255, 255, 255))

        if pc.shape[0] > 0:
            step = max(1, pc.shape[0] // 1600)
            for point in pc[::step]:
                x = int(round(20 + float(point[axes[0]]) * (panel_w - 40)))
                y = int(round(20 + (1.0 - float(point[axes[1]])) * (panel_h - 40)))
                draw.point((x, y), fill=(92, 92, 100))

        projected: dict[int, tuple[int, int]] = {}
        for slot in valid_slots:
            point = points[int(slot)]
            x = int(round(20 + float(point[axes[0]]) * (panel_w - 40)))
            y = int(round(20 + (1.0 - float(point[axes[1]])) * (panel_h - 40)))
            projected[int(slot)] = (x, y)

        if parents:
            for slot, parent_slot in parents.items():
                if slot == parent_slot:
                    continue
                if slot in projected and parent_slot in projected:
                    draw.line((projected[parent_slot], projected[slot]), fill=(255, 76, 76), width=2)

        for slot in valid_slots:
            x, y = projected[int(slot)]
            color = palette[int(slot) % len(palette)]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(0, 0, 0))
            if slot < 32:
                draw.text((x + 5, y - 5), str(int(slot)), fill=(230, 230, 230))

        canvas.paste(panel, (panel_w * view_idx, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _crop_rgb_from_source(source_frame: np.ndarray, bbox: np.ndarray) -> Image.Image:
    height, width = source_frame.shape[:2]
    if bbox is not None and bbox.shape[0] >= 4 and bbox[2] > 1.0 and bbox[3] > 1.0:
        x, y, box_w, box_h = [float(v) for v in bbox[:4]]
        x0 = int(np.clip(np.floor(x), 0, width - 1))
        y0 = int(np.clip(np.floor(y), 0, height - 1))
        x1 = int(np.clip(np.ceil(x + box_w - 1), x0, width - 1))
        y1 = int(np.clip(np.ceil(y + box_h - 1), y0, height - 1))
    else:
        x0, y0, x1, y1 = 0, 0, width - 1, height - 1
    rgb = source_frame[y0 : y1 + 1, x0 : x1 + 1, :3].astype(np.float32)
    valid = rgb != MASK_VALUE
    if np.any(valid) and float(np.nanmax(rgb[valid])) > 1.5:
        rgb[valid] = rgb[valid] / 255.0
    rgb = np.where(valid, rgb, 0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def _save_obj_joint_xyz_views(
    values: np.ndarray,
    mask: np.ndarray,
    source_frame: np.ndarray,
    bbox: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    values = values.reshape(-1, 3)
    mask = mask.reshape(-1, 3)
    valid = np.all(mask[:, :3] > 0.0, axis=1)
    points = np.clip(values[valid, :3], 0.0, 1.0)
    joint_ids = np.where(valid)[0]

    crop = _crop_rgb_from_source(source_frame, bbox).resize((224, 224), Image.BICUBIC)
    views = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
    panel_w, panel_h = 224, 224
    canvas = Image.new("RGB", (panel_w * 4, panel_h), (24, 24, 24))
    canvas.paste(crop, (0, 0))
    ImageDraw.Draw(canvas).text((8, 8), f"{title}: crop", fill=(255, 255, 255))

    palette = [(64, 200, 255), (255, 120, 80), (120, 255, 140), (255, 220, 80), (210, 130, 255)]
    for view_idx, (name, axes) in enumerate(views, start=1):
        panel = Image.new("RGB", (panel_w, panel_h), (12, 12, 16))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((20, 20, panel_w - 20, panel_h - 20), outline=(72, 72, 80))
        draw.text((8, 8), f"{title}: {name}", fill=(255, 255, 255))
        for local_idx, point in enumerate(points):
            x = int(round(20 + float(point[axes[0]]) * (panel_w - 40)))
            y = int(round(20 + (1.0 - float(point[axes[1]])) * (panel_h - 40)))
            color = palette[int(joint_ids[local_idx]) % len(palette)]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            if local_idx < 24:
                draw.text((x + 4, y - 4), str(int(joint_ids[local_idx])), fill=(220, 220, 220))
        canvas.paste(panel, (panel_w * view_idx, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    _save_obj_points_overlay(
        values,
        mask,
        source_frame,
        bbox,
        out_path.with_name("obj_joint_xyz_overlay_rgb.png"),
        title,
    )


def _save_obj_bfs_tree_views(
    values: np.ndarray,
    mask: np.ndarray,
    source_frame: np.ndarray,
    bbox: np.ndarray,
    out_path: Path,
    title: str,
    person_id: int,
) -> None:
    values = values.reshape(-1, 7)
    mask = mask.reshape(-1, 7)
    valid = np.all(mask[:, :7] > 0.0, axis=1)
    if np.any(valid):
        points = np.clip(values[:, 4:7], 0.0, 1.0)
    else:
        points = np.zeros((values.shape[0], 3), dtype=np.float32)
    valid_slots = np.where(valid)[0]

    parent_slots: dict[int, int] = {}
    max_parent_den = float(max(1, values.shape[0] - 1))
    for slot in valid_slots:
        parent_slot = int(round(float(np.clip(values[slot, 0], 0.0, 1.0)) * max_parent_den))
        if parent_slot not in valid_slots:
            parent_slot = int(slot)
        parent_slots[int(slot)] = parent_slot

    crop = _crop_rgb_from_source(source_frame, bbox).resize((224, 224), Image.BICUBIC)
    views = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
    panel_w, panel_h = 224, 224
    canvas = Image.new("RGB", (panel_w * 4, panel_h), (24, 24, 24))
    canvas.paste(crop, (0, 0))
    ImageDraw.Draw(canvas).text((8, 8), f"{title}: crop", fill=(255, 255, 255))

    palette = [(64, 200, 255), (255, 120, 80), (120, 255, 140), (255, 220, 80), (210, 130, 255)]
    for view_idx, (name, axes) in enumerate(views, start=1):
        panel = Image.new("RGB", (panel_w, panel_h), (12, 12, 16))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((20, 20, panel_w - 20, panel_h - 20), outline=(72, 72, 80))
        draw.text((8, 8), f"{title}: {name}", fill=(255, 255, 255))

        projected: dict[int, tuple[int, int]] = {}
        for slot in valid_slots:
            point = points[slot]
            x = int(round(20 + float(point[axes[0]]) * (panel_w - 40)))
            y = int(round(20 + (1.0 - float(point[axes[1]])) * (panel_h - 40)))
            projected[int(slot)] = (x, y)

        for slot, parent_slot in parent_slots.items():
            if slot == parent_slot:
                continue
            if slot in projected and parent_slot in projected:
                draw.line((projected[parent_slot], projected[slot]), fill=(130, 130, 150), width=1)

        for slot in valid_slots:
            x, y = projected[int(slot)]
            color = palette[int(slot) % len(palette)]
            radius = 4 if parent_slots.get(int(slot), int(slot)) == int(slot) else 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            if slot < 32:
                draw.text((x + 4, y - 4), str(int(slot)), fill=(220, 220, 220))
        canvas.paste(panel, (panel_w * view_idx, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    _save_obj_points_overlay(
        values,
        mask,
        source_frame,
        bbox,
        out_path.with_name("obj_bfs_tree_overlay_rgb.png"),
        title,
        parents=parent_slots,
    )
    _save_obj_pointcloud_tree_views(
        values,
        mask,
        source_frame,
        person_id,
        out_path.with_name("obj_bfs_tree_pointcloud_overlay.png"),
        title,
        parents=parent_slots,
    )


def _save_wsg_visualizations(
    task: str,
    out_dir: Path,
    step: int,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    batch: dict,
    dataset: StructureDataset,
    pred_mask: np.ndarray | None = None,
    extra_predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
) -> None:
    sample_count = min(4, pred.shape[0])
    for sample_idx in range(sample_count):
        source_frame = _load_source_frame(str(batch["wsv_path"][sample_idx]), str(batch["frame_key"][sample_idx]))
        person_id = int(batch["person_id"][sample_idx].item() if torch.is_tensor(batch["person_id"]) else batch["person_id"][sample_idx])
        bbox = batch["bbox"][sample_idx].detach().cpu().numpy() if torch.is_tensor(batch["bbox"]) else np.asarray(batch["bbox"][sample_idx], dtype=np.float32)
        label_values: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("generated", pred[sample_idx], pred_mask[sample_idx] if pred_mask is not None else mask[sample_idx]),
            ("gt", target[sample_idx], mask[sample_idx]),
        ]
        if extra_predictions:
            for extra_label, (extra_values, extra_mask) in extra_predictions.items():
                label_values.append((extra_label, extra_values[sample_idx], extra_mask[sample_idx] if extra_mask is not None else mask[sample_idx]))
        for label, values, current_mask in label_values:
            frame = _base_visual_frame(source_frame)
            if task in {"dwpose", "mediapipe"}:
                _scatter_keypoints(frame, task, values, current_mask, person_id, dataset.max_keypoints, bbox=bbox)
            elif task == "obj":
                if dataset.target_format == "obj_joint_xyz_norm":
                    sample_dir = out_dir / f"step_{step:06d}" / f"sample_{sample_idx:02d}" / label
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        sample_dir / f"{label}_{task}_joint_xyz_norm.npz",
                        joint_xyz_norm=values.reshape(dataset.max_slots, 3),
                        target_mask=current_mask.reshape(dataset.max_slots, 3),
                        bbox=bbox,
                        person_id=np.asarray(person_id, dtype=np.int32),
                    )
                    _save_obj_joint_xyz_views(
                        values,
                        current_mask,
                        source_frame,
                        bbox,
                        sample_dir / "obj_joint_xyz_views.png",
                        label,
                    )
                    continue
                if dataset.target_format == "obj_bfs_delta_tree":
                    sample_dir = out_dir / f"step_{step:06d}" / f"sample_{sample_idx:02d}" / label
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        sample_dir / f"{label}_{task}_bfs_delta_tree.npz",
                        bfs_delta_tree=values.reshape(dataset.max_slots, 7),
                        target_mask=current_mask.reshape(dataset.max_slots, 7),
                        bbox=bbox,
                        person_id=np.asarray(person_id, dtype=np.int32),
                    )
                    _save_obj_bfs_tree_views(
                        values,
                        current_mask,
                        source_frame,
                        bbox,
                        sample_dir / "obj_bfs_tree_views.png",
                        label,
                        person_id,
                    )
                    continue
                _scatter_obj(frame, values, current_mask, dataset.max_slots)
            else:
                continue
            sample_dir = out_dir / f"step_{step:06d}" / f"sample_{sample_idx:02d}" / label
            wsg_path = sample_dir / f"{label}_{task}.wsg"
            _write_single_wsg(wsg_path, frame)
            if task == "obj":
                _save_obj_preview(frame, sample_dir / "wsg_visualization" / f"{label}_{task}_skeleton.png")
                _save_obj_preview(frame, sample_dir / "wsg_visualization" / f"{label}_{task}_overlay_rgb.png")
            else:
                _save_keypoint_overlay(frame, source_frame, task, sample_dir / "wsg_visualization" / f"{label}_{task}_overlay_rgb.png")
                _run_wsg_visualizer(wsg_path, sample_dir / "wsg_visualization", task)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, step: int, task: str) -> None:
    model.eval()
    eval_indices = _select_eval_indices(loader.dataset, task, step, fixed_samples=2, random_samples=2)
    if eval_indices:
        batch = default_collate([loader.dataset[index] for index in eval_indices])
    else:
        batch = next(iter(loader))
    batch_device = {}
    for key, value in batch.items():
        batch_device[key] = value.to(device) if torch.is_tensor(value) else value
    with torch.no_grad():
        output = model(batch_device)
        extra_predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        pred_is_raw = False
        if isinstance(output, dict) and "flat" in output:
            pred = output["flat"].detach().cpu().numpy()
            pred_mask = None
            pred_is_raw = True
        elif isinstance(output, dict):
            pred_tensor, pred_mask_tensor = growing_tree_to_flat(
                output,
                loader.dataset.max_slots,
                min_slots=getattr(model, "preview_min_slots", 8),
                max_visible_slots=getattr(model, "preview_max_visible_slots", 16),
                preview_strategy=getattr(model, "preview_strategy", "topk"),
            )
            pred = pred_tensor.detach().cpu().numpy()
            pred_mask = pred_mask_tensor.detach().cpu().numpy()
            if "init_xyz" in output and "init_exists" in output:
                init_output = dict(output)
                init_output["xyz"] = output["init_xyz"]
                init_output["exist_logits"] = output["init_exists"] * 16.0 - 8.0
                if "init_parent_logits" in output:
                    init_output["parent_logits"] = output["init_parent_logits"]
                init_output["learned_exists"] = False
                init_output["allow_any_parent"] = False
                init_tensor, init_mask_tensor = growing_tree_to_flat(
                    init_output,
                    loader.dataset.max_slots,
                    min_slots=getattr(model, "preview_min_slots", 8),
                    max_visible_slots=getattr(model, "preview_max_visible_slots", 16),
                    preview_strategy=getattr(model, "preview_strategy", "topk"),
                )
                extra_predictions["init"] = (
                    init_tensor.detach().cpu().numpy(),
                    init_mask_tensor.detach().cpu().numpy(),
                )
        else:
            pred = output.detach().cpu().numpy()
            pred_mask = None
    target = batch["target_raw"].numpy()
    mask = batch["target_mask"].numpy()
    if pred_mask is None and not pred_is_raw:
        pred = pred * loader.dataset.target_std[None, :] + loader.dataset.target_mean[None, :]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"step_{step:06d}_predictions.npz",
        pred=pred[:8],
        target=target[:8],
        target_mask=mask[:8],
        imgname=np.asarray(batch["imgname"][:8]),
    )
    _save_wsg_visualizations(
        task,
        out_dir,
        step,
        pred,
        target,
        mask,
        batch,
        loader.dataset,
        pred_mask=pred_mask,
        extra_predictions=extra_predictions,
    )
    model.train()


def structure_loss(
    model: nn.Module,
    batch_device: dict,
    train_set: StructureDataset,
    use_growing_tree: bool,
    use_delta_tree: bool,
    use_init_refine: bool,
    use_init_diffusion: bool,
    use_init_anchor: bool,
) -> torch.Tensor:
    if isinstance(model, KinematicPoseRegressor):
        return kinematic_pose_loss(model(batch_device), batch_device, model)
    if use_init_diffusion:
        return init_diffusion_tree_loss(model, batch_device, train_set.max_slots)
    pred = model(batch_device)
    if use_growing_tree:
        return discrete_growing_tree_loss_base(pred, batch_device, train_set.max_slots)
    if use_delta_tree:
        return delta_growing_tree_loss(pred, batch_device, train_set.max_slots)
    if use_init_refine:
        return init_refine_tree_loss(pred, batch_device, train_set.max_slots)
    if use_init_anchor:
        return init_anchor_tree_loss(model, pred, batch_device, train_set.max_slots)
    return masked_mse(pred, batch_device["target"], batch_device["target_mask"])


def compute_val_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_set: StructureDataset,
    use_growing_tree: bool,
    use_delta_tree: bool,
    use_init_refine: bool,
    use_init_diffusion: bool,
    use_init_anchor: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_device = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            loss = structure_loss(model, batch_device, train_set, use_growing_tree, use_delta_tree, use_init_refine, use_init_diffusion, use_init_anchor)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True, choices=["dwpose", "mediapipe", "obj"])
    parser.add_argument("--condition", required=True, choices=["crop_pare", "full_rgb", "crop_rgb", "pointcloud"])
    parser.add_argument(
        "--obj-mode",
        choices=[
            "growing_tree_ce_base",
            "growing_tree_delta_ce",
            "growing_tree_init_refine",
            "growing_tree_init_diffusion",
            "growing_tree_init_anchor",
        ],
        default="growing_tree_ce_base",
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
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
        "--pose-mode",
        choices=["flat_mlp", "kinematic_rotations"],
        default="kinematic_rotations",
        help="Human pose decoder. kinematic_rotations initializes a skeleton and predicts local rotations.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_set = StructureDataset(args.train_dataset, args.train_cache, args.condition)
    val_set = StructureDataset(args.val_dataset, args.val_cache, args.condition, train_set.target_mean, train_set.target_std)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, drop_last=False, num_workers=args.num_workers)

    output_dim = train_set.target.shape[1]
    use_growing_tree = args.task == "obj" and train_set.target_format == "obj_bfs_delta_tree" and args.obj_mode == "growing_tree_ce_base"
    use_delta_tree = args.task == "obj" and train_set.target_format == "obj_bfs_delta_tree" and args.obj_mode == "growing_tree_delta_ce"
    use_init_refine = args.task == "obj" and train_set.target_format == "obj_bfs_delta_tree" and args.obj_mode == "growing_tree_init_refine"
    use_init_diffusion = args.task == "obj" and train_set.target_format == "obj_bfs_delta_tree" and args.obj_mode == "growing_tree_init_diffusion"
    use_init_anchor = args.task == "obj" and train_set.target_format == "obj_bfs_delta_tree" and args.obj_mode == "growing_tree_init_anchor"
    if use_growing_tree:
        model = DiscreteGrowingTreeRegressor(train_set.max_slots, use_slot_embed=False, condition=args.condition).to(device)
        model.preview_strategy = "prefix"
    elif use_delta_tree:
        model = DeltaGrowingTreeRegressor(train_set.max_slots, condition=args.condition).to(device)
        model.preview_strategy = "prefix"
    elif use_init_refine or use_init_diffusion or use_init_anchor:
        if args.condition != "pointcloud":
            raise ValueError(f"{args.obj_mode} requires --condition pointcloud")
        if use_init_diffusion:
            model = InitDiffusionTreeRegressor(train_set.max_slots).to(device)
        elif use_init_anchor:
            model = InitAnchorTreeRegressor(train_set.max_slots).to(device)
            model.preview_min_slots = min(32, train_set.max_slots)
            model.preview_max_visible_slots = min(48, train_set.max_slots)
        else:
            model = InitRefineTreeRegressor(train_set.max_slots).to(device)
        model.preview_strategy = "prefix"
    elif args.task in {"dwpose", "mediapipe"} and args.pose_mode == "kinematic_rotations":
        if args.condition != "crop_pare":
            raise ValueError("kinematic_rotations currently requires --condition crop_pare")
        priors = _kinematic_priors(
            train_set.target_raw,
            train_set.target_mask,
            args.task,
            train_set.max_keypoints,
        )
        model = KinematicPoseRegressor(
            train_set.features.shape[1],
            train_set.max_keypoints,
            priors,
        ).to(device)
        print(
            f"[{args.task}] initialized kinematic skeleton: "
            f"root={model.root} joints={model.active_count} spatial_dim={model.spatial_dim}",
            flush=True,
        )
    elif args.condition == "crop_pare":
        model = MLPRegressor(train_set.features.shape[1], output_dim).to(device)
    else:
        model = FullImageRegressor(output_dim).to(device)
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

    step = 0
    progress = tqdm(total=args.total_steps, desc=f"train {args.task}")
    while step < args.total_steps:
        for batch in train_loader:
            batch_device = {}
            for key, value in batch.items():
                batch_device[key] = value.to(device) if torch.is_tensor(value) else value
            loss = structure_loss(model, batch_device, train_set, use_growing_tree, use_delta_tree, use_init_refine, use_init_diffusion, use_init_anchor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            progress.update(1)
            progress.set_description(f"{args.task}_loss: {loss.item():.4f}")
            if step % args.val_freq == 0:
                train_loss = float(loss.detach().cpu())
                val_loss = compute_val_loss(model, val_loader, device, train_set, use_growing_tree, use_delta_tree, use_init_refine, use_init_diffusion, use_init_anchor)
                print(f"[{args.task}] step={step} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
                metrics.append(step, train_loss, val_loss)
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    torch.save({"step": step, "model": model.state_dict(), "output_dim": output_dim, "condition": args.condition, "task": args.task, "obj_mode": args.obj_mode, "pose_mode": args.pose_mode, "use_growing_tree": use_growing_tree, "use_delta_tree": use_delta_tree, "use_init_refine": use_init_refine, "use_init_diffusion": use_init_diffusion, "use_init_anchor": use_init_anchor, "target_mean": train_set.target_mean, "target_std": train_set.target_std, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val_loss}, ckpt_dir / "model-best.pt")
                if improved or step % args.log_freq == 0:
                    evaluate(model, val_loader, device, out_dir, step, args.task)
            if step % args.checkpoint_freq == 0:
                torch.save({"step": step, "model": model.state_dict(), "output_dim": output_dim, "condition": args.condition, "task": args.task, "obj_mode": args.obj_mode, "pose_mode": args.pose_mode, "use_growing_tree": use_growing_tree, "use_delta_tree": use_delta_tree, "use_init_refine": use_init_refine, "use_init_diffusion": use_init_diffusion, "use_init_anchor": use_init_anchor, "target_mean": train_set.target_mean, "target_std": train_set.target_std, "best_val_loss": best_val_loss}, ckpt_dir / "model-latest.pt")
            if step >= args.total_steps:
                break
    torch.save({"step": step, "model": model.state_dict(), "output_dim": output_dim, "condition": args.condition, "task": args.task, "obj_mode": args.obj_mode, "pose_mode": args.pose_mode, "use_growing_tree": use_growing_tree, "use_delta_tree": use_delta_tree, "use_init_refine": use_init_refine, "use_init_diffusion": use_init_diffusion, "use_init_anchor": use_init_anchor, "target_mean": train_set.target_mean, "target_std": train_set.target_std, "best_val_loss": best_val_loss}, ckpt_dir / "model-last.pt")
    if args.clean_log_layout:
        print(f"[{args.task}] finished; outputs under {args.output_dir}/ckpt and {args.output_dir}/visuals", flush=True)
    else:
        print(f"[{args.task}] finished; outputs under {args.output_dir}/checkpoints/{args.name} and {args.output_dir}/visuals/{args.name}", flush=True)


if __name__ == "__main__":
    main()
