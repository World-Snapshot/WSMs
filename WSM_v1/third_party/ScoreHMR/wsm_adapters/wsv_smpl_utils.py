from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


MASK_VALUE = -1.0

PERSON_ID_CHANNEL = 72
JOINT_ID_CHANNEL = 73
JOINT_X_CHANNEL = 74
JOINT_Y_CHANNEL = 75
JOINT_Z_CHANNEL = 76
BETAS_EXPRESSION_TYPE_CHANNEL = 77
BETAS_EXPRESSION_CHANNELS = list(range(78, 88))

MEDIAPIPE_PERSON_ID_CHANNEL = 95
MEDIAPIPE_JOINT_ID_CHANNEL = 96
MEDIAPIPE_VISIBILITY_CHANNEL = 98

DWPOSE_PERSON_ID_CHANNEL = 64
DWPOSE_KEYPOINT_ID_CHANNEL = 65
DWPOSE_SCORE_CHANNEL = 66

# ScoreHMR predicts SMPL global_orient + 23 body joints:
# pelvis-relative SMPL joints 1..21 map directly to the WSV SMPLer-X body ids.
# SMPL's terminal L_hand/R_hand joints do not exist as body joints in SMPLer-X,
# so we approximate them with the first left/right hand-pose rotations.
WSV_LEFT_HAND_ROOT_ID = 22
WSV_RIGHT_HAND_ROOT_ID = 37
SMPL_LEFT_HAND_JOINT_ID = 22
SMPL_RIGHT_HAND_JOINT_ID = 23
WSV_TO_SCOREHMR_JOINT_ID = {
    **{joint_id: joint_id for joint_id in range(0, 22)},
    WSV_LEFT_HAND_ROOT_ID: SMPL_LEFT_HAND_JOINT_ID,
    WSV_RIGHT_HAND_ROOT_ID: SMPL_RIGHT_HAND_JOINT_ID,
}


def _box_to_arrays(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min, y_min, x_max, y_max = box
    bbox_w = max(1, x_max - x_min + 1)
    bbox_h = max(1, y_max - y_min + 1)
    center = np.asarray([x_min + 0.5 * bbox_w, y_min + 0.5 * bbox_h], dtype=np.float32)
    scale = np.asarray([bbox_w, bbox_h], dtype=np.float32)
    bbox = np.asarray([x_min, y_min, bbox_w, bbox_h], dtype=np.float32)
    bbox_norm = np.asarray(
        [
            center[0] / max(1, width),
            center[1] / max(1, height),
            bbox_w / max(1, width),
            bbox_h / max(1, height),
        ],
        dtype=np.float32,
    )
    return center, scale, bbox, bbox_norm


def _expand_box_relative(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    width: int,
    height: int,
    expand_fraction: float,
) -> tuple[int, int, int, int]:
    box_w = max(1.0, float(x_max - x_min + 1))
    box_h = max(1.0, float(y_max - y_min + 1))
    pad_x = 0.5 * float(expand_fraction) * box_w
    pad_y = 0.5 * float(expand_fraction) * box_h
    return (
        max(0, int(np.floor(float(x_min) - pad_x))),
        max(0, int(np.floor(float(y_min) - pad_y))),
        min(width - 1, int(np.ceil(float(x_max) + pad_x))),
        min(height - 1, int(np.ceil(float(y_max) + pad_y))),
    )


def _xyxy_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1 + 1) * (ay2 - ay1 + 1))
    area_b = max(1, (bx2 - bx1 + 1) * (by2 - by1 + 1))
    return float(inter / max(1, area_a + area_b - inter))


def extract_mediapipe_boxes_from_wsg_frame(
    frame: np.ndarray,
    margin: int = 20,
    min_points: int = 3,
    visibility_threshold: float | None = None,
    max_area_fraction: float = 1.0,
    margin_mode: str = "pixels",
    margin_fraction: float = 0.2,
) -> list[dict]:
    """Recover cleaner human boxes from MediaPipe pose channels 95/96/98."""
    if frame.ndim != 3 or frame.shape[-1] <= MEDIAPIPE_VISIBILITY_CHANNEL:
        return []

    person_channel = frame[:, :, MEDIAPIPE_PERSON_ID_CHANNEL]
    joint_channel = frame[:, :, MEDIAPIPE_JOINT_ID_CHANNEL]
    visibility_channel = frame[:, :, MEDIAPIPE_VISIBILITY_CHANNEL]
    valid = (
        (person_channel != MASK_VALUE)
        & (joint_channel != MASK_VALUE)
        & (visibility_channel != MASK_VALUE)
    )
    if visibility_threshold is not None:
        valid &= visibility_channel >= float(visibility_threshold)
    if not np.any(valid):
        return []

    height, width = frame.shape[:2]
    boxes: list[dict] = []
    for person_id_float in np.unique(person_channel[valid]):
        person_mask = valid & (person_channel == person_id_float)
        ys, xs = np.where(person_mask)
        if ys.size < min_points:
            continue
        x_min_raw = int(xs.min())
        y_min_raw = int(ys.min())
        x_max_raw = int(xs.max())
        y_max_raw = int(ys.max())
        if margin_mode == "relative":
            x_min, y_min, x_max, y_max = _expand_box_relative(
                x_min_raw, y_min_raw, x_max_raw, y_max_raw, width, height, margin_fraction
            )
        elif margin_mode == "pixels":
            x_min = max(0, x_min_raw - margin)
            y_min = max(0, y_min_raw - margin)
            x_max = min(width - 1, x_max_raw + margin)
            y_max = min(height - 1, y_max_raw + margin)
        else:
            raise ValueError(f"Unsupported bbox margin_mode: {margin_mode}")
        box_area_fraction = float(((x_max - x_min + 1) * (y_max - y_min + 1)) / max(1, width * height))
        if box_area_fraction > float(max_area_fraction):
            continue
        center, scale, bbox, bbox_norm = _box_to_arrays((x_min, y_min, x_max, y_max), width, height)
        visibility_values = visibility_channel[person_mask].astype(np.float32)
        boxes.append(
            {
                "person_id": int(round(float(person_id_float))),
                "center": center,
                "scale": scale,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "xyxy": (x_min, y_min, x_max, y_max),
                "num_points": int(ys.size),
                "visibility_mean": float(visibility_values.mean()) if visibility_values.size else 0.0,
                "visibility_min": float(visibility_values.min()) if visibility_values.size else 0.0,
                "visibility_max": float(visibility_values.max()) if visibility_values.size else 0.0,
                "area_fraction": box_area_fraction,
            }
        )
    boxes.sort(key=lambda item: int(item["person_id"]))
    return boxes


def extract_smpl_boxes_from_wsg_frame(
    frame: np.ndarray,
    margin: int = 20,
    min_points: int = 8,
    max_area_fraction: float = 1.0,
    margin_mode: str = "relative",
    margin_fraction: float = 0.2,
) -> list[dict]:
    """Recover full-human boxes from dense SMPL/SMPLer-X person regions."""
    if frame.ndim != 3 or frame.shape[-1] <= PERSON_ID_CHANNEL:
        return []

    person_channel = frame[:, :, PERSON_ID_CHANNEL]
    valid = person_channel != MASK_VALUE
    if not np.any(valid):
        return []

    height, width = frame.shape[:2]
    boxes: list[dict] = []
    for person_id_float in np.unique(person_channel[valid]):
        person_mask = valid & (person_channel == person_id_float)
        ys, xs = np.where(person_mask)
        if ys.size < min_points:
            continue
        x_min_raw = int(xs.min())
        y_min_raw = int(ys.min())
        x_max_raw = int(xs.max())
        y_max_raw = int(ys.max())
        if margin_mode == "relative":
            x_min, y_min, x_max, y_max = _expand_box_relative(
                x_min_raw, y_min_raw, x_max_raw, y_max_raw, width, height, margin_fraction
            )
        elif margin_mode == "pixels":
            x_min = max(0, x_min_raw - margin)
            y_min = max(0, y_min_raw - margin)
            x_max = min(width - 1, x_max_raw + margin)
            y_max = min(height - 1, y_max_raw + margin)
        else:
            raise ValueError(f"Unsupported bbox margin_mode: {margin_mode}")
        box_area_fraction = float(((x_max - x_min + 1) * (y_max - y_min + 1)) / max(1, width * height))
        if box_area_fraction > float(max_area_fraction):
            continue
        center, scale, bbox, bbox_norm = _box_to_arrays((x_min, y_min, x_max, y_max), width, height)
        boxes.append(
            {
                "person_id": int(round(float(person_id_float))),
                "center": center,
                "scale": scale,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "xyxy": (x_min, y_min, x_max, y_max),
                "num_points": int(ys.size),
                "area_fraction": box_area_fraction,
            }
        )
    boxes.sort(key=lambda item: int(item["person_id"]))
    return boxes


def has_dwpose_human_evidence_in_wsg_frame(
    frame: np.ndarray,
    min_points: int = 8,
    score_threshold: float | None = None,
) -> bool:
    """Return whether DWPose channels provide independent human evidence."""
    if frame.ndim != 3 or frame.shape[-1] <= DWPOSE_SCORE_CHANNEL:
        return False

    person = frame[:, :, DWPOSE_PERSON_ID_CHANNEL]
    keypoint = frame[:, :, DWPOSE_KEYPOINT_ID_CHANNEL]
    score = frame[:, :, DWPOSE_SCORE_CHANNEL]
    valid = (person != MASK_VALUE) & (keypoint != MASK_VALUE) & (score != MASK_VALUE)
    if score_threshold is not None:
        valid &= score >= float(score_threshold)
    return int(np.count_nonzero(valid)) >= int(min_points)


def iter_wsv_files(roots: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        root_path = Path(root).expanduser()
        paths.extend(root_path.glob("**/*.wsv"))
        paths.extend(root_path.glob("**/*.wsv.npz"))
    return sorted({path.resolve() for path in paths})


def frame_key_to_image_path(wsv_path: Path, frame_key: str) -> Path:
    video_dir = wsv_path.parent.parent if wsv_path.parent.name.lower() == "wsv" else wsv_path.parent
    frame_id = frame_key
    if frame_id.startswith("index_"):
        frame_id = frame_id[len("index_"):]
    return video_dir / "crop_frame" / f"{int(frame_id):08d}.jpg"


def extract_people_from_wsg_frame(
    frame: np.ndarray,
    margin: int = 20,
    box_source: str = "mediapipe95",
) -> list[dict]:
    if frame.ndim != 3 or frame.shape[-1] <= JOINT_Z_CHANNEL:
        return []

    person_channel = frame[:, :, PERSON_ID_CHANNEL]
    valid = person_channel != MASK_VALUE
    if not np.any(valid):
        return []

    height, width = frame.shape[:2]
    mediapipe_boxes = extract_mediapipe_boxes_from_wsg_frame(frame, margin=margin)
    people: list[dict] = []
    for person_id_float in np.unique(person_channel[valid]):
        person_id = int(round(float(person_id_float)))
        person_mask = person_channel == person_id_float
        ys, xs = np.where(person_mask)
        if ys.size == 0:
            continue

        body_pose = np.zeros(72, dtype=np.float32)
        joint_channel = frame[:, :, JOINT_ID_CHANNEL]
        joint_values: dict[int, tuple[float, float, float]] = {}
        scorehmr_joint_values: dict[int, tuple[float, float, float]] = {}
        for y, x in zip(ys, xs):
            joint_id_value = joint_channel[y, x]
            if joint_id_value == MASK_VALUE:
                continue
            wsv_joint_id = int(round(float(joint_id_value)))
            if wsv_joint_id not in WSV_TO_SCOREHMR_JOINT_ID:
                continue
            scorehmr_joint_id = WSV_TO_SCOREHMR_JOINT_ID[wsv_joint_id]
            xyz = frame[y, x, [JOINT_X_CHANNEL, JOINT_Y_CHANNEL, JOINT_Z_CHANNEL]].astype(np.float32)
            if np.any(xyz == MASK_VALUE):
                continue
            xyz_tuple = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            joint_values.setdefault(wsv_joint_id, xyz_tuple)
            scorehmr_joint_values.setdefault(scorehmr_joint_id, xyz_tuple)

        if 0 in scorehmr_joint_values:
            body_pose[:3] = np.asarray(scorehmr_joint_values[0], dtype=np.float32)
        for joint_id in range(1, 24):
            if joint_id in scorehmr_joint_values:
                start = 3 + (joint_id - 1) * 3
                body_pose[start:start + 3] = np.asarray(scorehmr_joint_values[joint_id], dtype=np.float32)

        if len(scorehmr_joint_values) < 8:
            continue

        x_min = max(0, int(xs.min()) - margin)
        y_min = max(0, int(ys.min()) - margin)
        x_max = min(width - 1, int(xs.max()) + margin)
        y_max = min(height - 1, int(ys.max()) + margin)
        region_xyxy = (x_min, y_min, x_max, y_max)
        center, scale, bbox, bbox_norm = _box_to_arrays(region_xyxy, width, height)

        matched_box = None
        if box_source == "mediapipe95" and mediapipe_boxes:
            matched_box = max(mediapipe_boxes, key=lambda item: _xyxy_iou(region_xyxy, item["xyxy"]))
            if _xyxy_iou(region_xyxy, matched_box["xyxy"]) <= 0.0:
                matched_box = None
        if matched_box is not None:
            center = matched_box["center"]
            scale = matched_box["scale"]
            bbox = matched_box["bbox"]
            bbox_norm = matched_box["bbox_norm"]

        betas = np.zeros(10, dtype=np.float32)
        beta_mask = person_mask & (frame[:, :, BETAS_EXPRESSION_TYPE_CHANNEL] == 0)
        if np.any(beta_mask):
            for i, channel in enumerate(BETAS_EXPRESSION_CHANNELS):
                vals = frame[:, :, channel][beta_mask]
                vals = vals[vals != MASK_VALUE]
                if vals.size:
                    betas[i] = float(np.median(vals.astype(np.float32)))

        people.append(
            {
                "person_id": person_id,
                "body_pose": body_pose,
                "betas": betas,
                "center": center,
                "scale": scale,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "num_joints": len(scorehmr_joint_values),
                "box_source": box_source if matched_box is not None else "smplerx_region",
            }
        )
    return people


def split_train_val(paths: list[Path], val_fraction: float = 0.05, seed: int = 42) -> tuple[list[Path], list[Path]]:
    if not paths:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(paths) * val_fraction))) if len(paths) > 1 else 0
    val_indices = set(indices[:val_count].tolist())
    train, val = [], []
    for i, path in enumerate(paths):
        (val if i in val_indices else train).append(path)
    return train, val
