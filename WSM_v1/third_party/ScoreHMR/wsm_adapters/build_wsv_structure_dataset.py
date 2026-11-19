from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from wsv_smpl_utils import (
    MASK_VALUE,
    _box_to_arrays,
    extract_mediapipe_boxes_from_wsg_frame,
    frame_key_to_image_path,
    iter_wsv_files,
    split_train_val,
)


DWPOSE_PERSON_ID_CHANNEL = 64
DWPOSE_KEYPOINT_ID_CHANNEL = 65
DWPOSE_SCORE_CHANNEL = 66

MEDIAPIPE_PERSON_ID_CHANNEL = 95
MEDIAPIPE_JOINT_ID_CHANNEL = 96
MEDIAPIPE_Z_CHANNEL = 97
MEDIAPIPE_VISIBILITY_CHANNEL = 98

OBJ_ID_CHANNEL = 111
OBJ_CLASS_CHANNEL = 112
# Channel 111 is the dense object/category id and is used only for grouping.
# Channels 112-121 carry the sparse object/animal structure payload.
# Channels 122-125 are reserved and intentionally ignored for now.
OBJ_VALUE_CHANNELS = list(range(112, 122))


def iter_structure_files(roots: list[str | Path], task: str) -> list[Path]:
    if task != "obj":
        return iter_wsv_files(roots)
    paths: list[Path] = []
    for root in roots:
        root_path = Path(root).expanduser()
        paths.extend(root_path.glob("**/*.wsg"))
        paths.extend(root_path.glob("**/*.wsg.npz"))
    return sorted({path.resolve() for path in paths})


def iter_np_frames(data: np.lib.npyio.NpzFile, frame_stride: int, max_frames_per_video: int | None) -> list[tuple[str, np.ndarray]]:
    keys = sorted(data.files)
    if keys == ["wsg_data"]:
        return [("__single__", data["wsg_data"])]
    if frame_stride > 1:
        keys = keys[::frame_stride]
    if max_frames_per_video is not None:
        keys = keys[:max_frames_per_video]
    return [(key, data[key]) for key in keys]


def _person_box_from_points(xs: np.ndarray, ys: np.ndarray, width: int, height: int, margin: int) -> dict:
    x_min = max(0, int(xs.min()) - margin)
    y_min = max(0, int(ys.min()) - margin)
    x_max = min(width - 1, int(xs.max()) + margin)
    y_max = min(height - 1, int(ys.max()) + margin)
    center, scale, bbox, bbox_norm = _box_to_arrays((x_min, y_min, x_max, y_max), width, height)
    return {
        "center": center,
        "scale": scale,
        "bbox": bbox,
        "bbox_norm": bbox_norm,
        "xyxy": (x_min, y_min, x_max, y_max),
    }


def _xy_to_box_norm(x: int, y: int, box: dict) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = box["xyxy"]
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)
    return (
        float(np.clip((float(x) - x_min) / box_w, 0.0, 1.0)),
        float(np.clip((float(y) - y_min) / box_h, 0.0, 1.0)),
    )


def _extract_keypoint_rows(
    frame: np.ndarray,
    task: str,
    max_keypoints: int,
    margin: int,
) -> list[dict]:
    if task == "dwpose":
        person_channel = frame[:, :, DWPOSE_PERSON_ID_CHANNEL]
        key_channel = frame[:, :, DWPOSE_KEYPOINT_ID_CHANNEL]
        value_channels = [DWPOSE_SCORE_CHANNEL]
        values_per_keypoint = 3
    elif task == "mediapipe":
        person_channel = frame[:, :, MEDIAPIPE_PERSON_ID_CHANNEL]
        key_channel = frame[:, :, MEDIAPIPE_JOINT_ID_CHANNEL]
        value_channels = [MEDIAPIPE_Z_CHANNEL, MEDIAPIPE_VISIBILITY_CHANNEL]
        values_per_keypoint = 4
    else:
        raise ValueError(f"Unsupported keypoint task: {task}")

    valid = (person_channel != MASK_VALUE) & (key_channel != MASK_VALUE)
    if not np.any(valid):
        return []

    height, width = frame.shape[:2]
    rows: list[dict] = []

    if task == "dwpose":
        # DWPose person ids are known to be noisy in multi-person scenes,
        # especially for hands. Use cleaner MediaPipe person boxes to decide
        # which crop owns each DWPose point; the model learns keypoint geometry,
        # not the raw DWPose person-id label.
        media_boxes = extract_mediapipe_boxes_from_wsg_frame(frame, margin=margin)
        if media_boxes:
            for box in media_boxes:
                x_min, y_min, x_max, y_max = box["xyxy"]
                box_mask = np.zeros(valid.shape, dtype=bool)
                box_mask[y_min : y_max + 1, x_min : x_max + 1] = True
                person_mask = valid & box_mask
                ys, xs = np.where(person_mask)
                if ys.size < 3:
                    continue

                target = np.zeros((max_keypoints, values_per_keypoint), dtype=np.float32)
                target_mask = np.zeros((max_keypoints, values_per_keypoint), dtype=np.float32)
                best_score = np.full((max_keypoints,), -np.inf, dtype=np.float32)
                seen = 0
                for y, x in zip(ys, xs):
                    key_id = int(round(float(key_channel[y, x])))
                    if key_id < 0 or key_id >= max_keypoints:
                        continue
                    score = frame[y, x, DWPOSE_SCORE_CHANNEL]
                    score_value = float(score) if score != MASK_VALUE else 1.0
                    if score_value < best_score[key_id]:
                        continue
                    best_score[key_id] = score_value
                    target[key_id, 0], target[key_id, 1] = _xy_to_box_norm(x, y, box)
                    target[key_id, 2] = score_value
                    target_mask[key_id, :3] = 1.0
                    seen += 1
                if seen < 3:
                    continue

                rows.append(
                    {
                        "person_id": int(box["person_id"]),
                        "center": box["center"],
                        "scale": box["scale"],
                        "bbox": box["bbox"],
                        "bbox_norm": box["bbox_norm"],
                        "target": target.reshape(-1),
                        "target_mask": target_mask.reshape(-1),
                        "valid_points": int(np.count_nonzero(target_mask[:, 0])),
                    }
                )
            if rows:
                return rows

    for person_id_float in np.unique(person_channel[valid]):
        person_mask = valid & (person_channel == person_id_float)
        ys, xs = np.where(person_mask)
        if ys.size < 3:
            continue

        # Use MediaPipe 95 boxes when available because DWPose person ids can be
        # wrong for hands in multi-person scenes; fall back to this task's points.
        media_boxes = extract_mediapipe_boxes_from_wsg_frame(frame, margin=margin)
        person_id = int(round(float(person_id_float)))
        box = next((item for item in media_boxes if item["person_id"] == person_id), None)
        if box is None:
            box = _person_box_from_points(xs, ys, width, height, margin)

        target = np.zeros((max_keypoints, values_per_keypoint), dtype=np.float32)
        target_mask = np.zeros((max_keypoints, values_per_keypoint), dtype=np.float32)
        seen = 0
        for y, x in zip(ys, xs):
            key_id = int(round(float(key_channel[y, x])))
            if key_id < 0 or key_id >= max_keypoints:
                continue
            target[key_id, 0], target[key_id, 1] = _xy_to_box_norm(x, y, box)
            target_mask[key_id, :2] = 1.0
            if task == "dwpose":
                score = frame[y, x, DWPOSE_SCORE_CHANNEL]
                if score != MASK_VALUE:
                    target[key_id, 2] = float(score)
                    target_mask[key_id, 2] = 1.0
            else:
                z_value = frame[y, x, MEDIAPIPE_Z_CHANNEL]
                vis_value = frame[y, x, MEDIAPIPE_VISIBILITY_CHANNEL]
                if z_value != MASK_VALUE:
                    target[key_id, 2] = float(z_value)
                    target_mask[key_id, 2] = 1.0
                if vis_value != MASK_VALUE:
                    target[key_id, 3] = float(vis_value)
                    target_mask[key_id, 3] = 1.0
            seen += 1
        if seen < 3:
            continue

        rows.append(
            {
                "person_id": person_id,
                "center": box["center"],
                "scale": box["scale"],
                "bbox": box["bbox"],
                "bbox_norm": box["bbox_norm"],
                "target": target.reshape(-1),
                "target_mask": target_mask.reshape(-1),
                "valid_points": seen,
            }
        )
    return rows


def _extract_obj_row(frame: np.ndarray, max_slots: int, margin: int) -> dict | None:
    if frame.ndim != 3 or frame.shape[-1] <= max(OBJ_VALUE_CHANNELS):
        return None
    object_channel = frame[:, :, OBJ_ID_CHANNEL]
    class_channel = frame[:, :, OBJ_CLASS_CHANNEL]
    # The current structured OBJ task trains the object skeleton only.  Full
    # mesh vertices/faces can dominate a WSG file by hundreds of thousands of
    # pixels, which previously caused the rare joint/bone records to disappear
    # during fixed-slot sampling.  Keep class=3 joints and class=4 bones first.
    valid = (object_channel != MASK_VALUE) & np.isin(np.rint(class_channel).astype(np.int32), [3, 4])
    if not np.any(valid):
        return None

    # OBJ/animal skeleton data is not a fixed-template body like SMPL. Each
    # valid joint/bone WSG pixel is treated as a sparse structure token:
    # projected xy + channels 112-121. Packed metadata fields are masked out
    # below. Empirically, the WSG RGB render and object rig align best when
    # joint xyz is projected with the mesh-vertex extent on X-Y and both axes
    # flipped (debug: vertex_extent_xy_fx1_fy1).
    slots = np.zeros((max_slots, 2 + len(OBJ_VALUE_CHANNELS)), dtype=np.float32)
    slot_mask = np.zeros_like(slots, dtype=np.float32)

    ys, xs = np.where(valid)
    object_ids = object_channel[ys, xs].astype(np.float32)
    class_values = class_channel[ys, xs].astype(np.float32)

    vertex_mask = (object_channel != MASK_VALUE) & (np.rint(class_channel).astype(np.int32) == 1)
    vertex_coords = frame[:, :, 114:117][vertex_mask]
    if vertex_coords.size == 0:
        joint_mask = (object_channel != MASK_VALUE) & (np.rint(class_channel).astype(np.int32) == 3)
        vertex_coords = frame[:, :, 114:117][joint_mask]
    finite_coord_mask = np.isfinite(vertex_coords).all(axis=1) if vertex_coords.ndim == 2 else np.asarray([], dtype=bool)
    vertex_coords = vertex_coords[finite_coord_mask] if vertex_coords.ndim == 2 else np.zeros((0, 3), dtype=np.float32)
    if vertex_coords.size:
        min_x = float(vertex_coords[:, 0].min())
        max_x = float(vertex_coords[:, 0].max())
        min_y = float(vertex_coords[:, 1].min())
        max_y = float(vertex_coords[:, 1].max())
    else:
        min_x, max_x, min_y, max_y = 0.0, 1.0, 0.0, 1.0
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)

    joint_xyz_by_id: dict[int, np.ndarray] = {}
    for y, x in zip(ys, xs):
        class_value = int(round(float(frame[y, x, OBJ_CLASS_CHANNEL])))
        if class_value != 3:
            continue
        joint_id = int(round(float(frame[y, x, 113])))
        xyz = np.asarray(frame[y, x, 114:117], dtype=np.float32)
        if np.isfinite(xyz).all():
            joint_xyz_by_id[joint_id] = xyz

    def project_vertex_extent_xy_flip(xyz: np.ndarray, fallback_x: int, fallback_y: int) -> tuple[float, float]:
        if np.isfinite(xyz).all():
            proj_x = 1.0 - ((float(xyz[0]) - min_x) / range_x)
            proj_y = 1.0 - ((float(xyz[1]) - min_y) / range_y)
            return float(np.clip(proj_x, 0.0, 1.0)), float(np.clip(proj_y, 0.0, 1.0))
        height, width = frame.shape[:2]
        return float(fallback_x) / max(1, width - 1), float(fallback_y) / max(1, height - 1)

    joint_order = np.where(np.rint(class_values).astype(np.int32) == 3)[0]
    bone_order = np.where(np.rint(class_values).astype(np.int32) == 4)[0]
    joint_order = joint_order[np.lexsort((xs[joint_order], ys[joint_order], object_ids[joint_order]))]
    bone_order = bone_order[np.lexsort((xs[bone_order], ys[bone_order], object_ids[bone_order]))]
    order = np.concatenate([joint_order, bone_order])
    if order.size > max_slots:
        pick = np.linspace(0, order.size - 1, max_slots).round().astype(np.int64)
        order = order[pick]

    for slot, idx in enumerate(order[:max_slots]):
        y = int(ys[idx])
        x = int(xs[idx])
        class_value = int(round(float(frame[y, x, OBJ_CLASS_CHANNEL])))
        if class_value == 3:
            xyz = np.asarray(frame[y, x, 114:117], dtype=np.float32)
        elif class_value == 4:
            start_joint = int(round(float(frame[y, x, 114])))
            end_joint = int(round(float(frame[y, x, 115])))
            if start_joint in joint_xyz_by_id and end_joint in joint_xyz_by_id:
                xyz = (joint_xyz_by_id[start_joint] + joint_xyz_by_id[end_joint]) * 0.5
            else:
                xyz = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
        else:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
        slots[slot, 0], slots[slot, 1] = project_vertex_extent_xy_flip(xyz, x, y)
        slot_mask[slot, :2] = 1.0
        for offset, channel in enumerate(OBJ_VALUE_CHANNELS):
            value = frame[y, x, channel]
            if value == MASK_VALUE:
                continue
            # Keep only semantically meaningful, numerically stable fields:
            # class value, record id, xyz/edge endpoints/length, parent id.
            if class_value == 3 and channel not in {112, 113, 114, 115, 116, 118}:
                continue
            if class_value == 4 and channel not in {112, 113, 114, 115, 116}:
                continue
            if not np.isfinite(value):
                continue
            slots[slot, 2 + offset] = float(value)
            slot_mask[slot, 2 + offset] = 1.0
    if not np.any(slot_mask):
        return None
    return {
        "target": slots.reshape(-1),
        "target_mask": slot_mask.reshape(-1),
        "valid_points": int(np.count_nonzero(slot_mask[:, 0])),
    }


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.astype(np.float32, copy=False)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    scale = np.maximum(maxs - mins, 1e-6)
    return np.clip((xyz - mins[None, :]) / scale[None, :], 0.0, 1.0).astype(np.float32)


def _extract_obj_joint_xyz_rows(frame: np.ndarray, max_slots: int, margin: int) -> list[dict]:
    """Extract object/animal group crops and BFS tree targets.

    The fixed joint-id slot target is brittle because joint ids are not stable
    semantic labels across object assets.  Instead, build a local skeleton graph
    from class=3 joints and class=4 bones, choose a central root, then serialize
    nodes in BFS order.  Each slot predicts parent slot + relative delta + final
    normalized xyz, so the target behaves like a small structure growing outward
    from a root node.
    """
    if frame.ndim != 3 or frame.shape[-1] <= 118:
        return []
    object_channel = frame[:, :, OBJ_ID_CHANNEL]
    class_channel = frame[:, :, OBJ_CLASS_CHANNEL]
    valid_obj = object_channel != MASK_VALUE
    if not np.any(valid_obj):
        return []

    height, width = frame.shape[:2]
    rows: list[dict] = []
    for object_id_float in np.unique(object_channel[valid_obj]):
        object_mask = valid_obj & (object_channel == object_id_float)
        joint_mask = object_mask & (np.rint(class_channel).astype(np.int32) == 3)
        ys, xs = np.where(joint_mask)
        if ys.size < 3:
            continue

        joint_by_id: dict[int, np.ndarray] = {}
        for y, x in zip(ys, xs):
            joint_id = int(round(float(frame[y, x, 113])))
            if joint_id < 0:
                continue
            xyz = np.asarray(frame[y, x, 114:117], dtype=np.float32)
            if np.isfinite(xyz).all() and not np.any(xyz == MASK_VALUE):
                joint_by_id[joint_id] = xyz
        if len(joint_by_id) < 3:
            continue

        # Skeleton joints should live in the same normalized 3D volume as the
        # object point cloud.  Normalizing by joint extent alone makes the
        # skeleton self-consistent but loses the "inside the shape" constraint,
        # which can make generated bones drift outside the visible point cloud.
        vertex_mask = object_mask & (np.rint(class_channel).astype(np.int32) == 1)
        vertex_ys, vertex_xs = np.where(vertex_mask)
        vertex_xyz: list[np.ndarray] = []
        for y, x in zip(vertex_ys, vertex_xs):
            xyz = np.asarray(frame[y, x, 114:117], dtype=np.float32)
            if np.isfinite(xyz).all() and not np.any(xyz == MASK_VALUE):
                vertex_xyz.append(xyz)

        edge_set: set[tuple[int, int]] = set()
        bone_mask = object_mask & (np.rint(class_channel).astype(np.int32) == 4)
        bone_ys, bone_xs = np.where(bone_mask)
        for y, x in zip(bone_ys, bone_xs):
            start_joint = int(round(float(frame[y, x, 114])))
            end_joint = int(round(float(frame[y, x, 115])))
            if start_joint in joint_by_id and end_joint in joint_by_id and start_joint != end_joint:
                edge_set.add(tuple(sorted((start_joint, end_joint))))

        adjacency: dict[int, set[int]] = {joint_id: set() for joint_id in joint_by_id}
        for start_joint, end_joint in edge_set:
            adjacency[start_joint].add(end_joint)
            adjacency[end_joint].add(start_joint)

        joint_ids = sorted(joint_by_id)
        xyz_raw = np.stack([joint_by_id[joint_id] for joint_id in joint_ids], axis=0)
        if len(vertex_xyz) >= 3:
            norm_xyz_raw = np.stack(vertex_xyz, axis=0)
        else:
            norm_xyz_raw = xyz_raw
        mins = norm_xyz_raw.min(axis=0)
        maxs = norm_xyz_raw.max(axis=0)
        scale = np.maximum(maxs - mins, 1e-6)
        xyz_norm_by_id = {
            joint_id: np.clip((joint_by_id[joint_id] - mins) / scale, 0.0, 1.0).astype(np.float32)
            for joint_id in joint_ids
        }

        centroid = norm_xyz_raw.mean(axis=0)
        root_id = min(
            joint_ids,
            key=lambda joint_id: (
                -len(adjacency.get(joint_id, ())),
                float(np.linalg.norm(joint_by_id[joint_id] - centroid)),
                joint_id,
            ),
        )

        ordered_ids: list[int] = []
        parent_by_id: dict[int, int] = {root_id: root_id}
        visited: set[int] = set()
        components = [root_id] + [joint_id for joint_id in joint_ids if joint_id != root_id]
        for component_root in components:
            if component_root in visited:
                continue
            if component_root not in parent_by_id:
                parent_by_id[component_root] = component_root
            queue = [component_root]
            visited.add(component_root)
            while queue:
                current = queue.pop(0)
                ordered_ids.append(current)
                for neighbor in sorted(adjacency.get(current, ())):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    parent_by_id[neighbor] = current
                    queue.append(neighbor)

        if len(ordered_ids) > max_slots:
            ordered_ids = ordered_ids[:max_slots]
        slot_by_id = {joint_id: slot for slot, joint_id in enumerate(ordered_ids)}

        # slot fields: parent_slot_norm, delta_xyz_from_parent, absolute_xyz.
        target = np.zeros((max_slots, 7), dtype=np.float32)
        target_mask = np.zeros_like(target, dtype=np.float32)
        max_parent_den = float(max(1, max_slots - 1))
        for slot, joint_id in enumerate(ordered_ids):
            parent_id = parent_by_id.get(joint_id, joint_id)
            parent_slot = slot_by_id.get(parent_id, slot)
            xyz_norm = xyz_norm_by_id[joint_id]
            parent_xyz_norm = xyz_norm_by_id.get(parent_id, xyz_norm)
            delta = np.clip(xyz_norm - parent_xyz_norm, -1.0, 1.0)
            target[slot, 0] = float(parent_slot) / max_parent_den
            target[slot, 1:4] = delta
            target[slot, 4:7] = xyz_norm
            target_mask[slot, :7] = 1.0

        # Crop conditioning should see the object group, not the full scene.
        # Prefer dense mesh vertices for the crop; fall back to sparse rig pixels.
        crop_mask = object_mask & (np.rint(class_channel).astype(np.int32) == 1)
        if not np.any(crop_mask):
            crop_mask = object_mask & np.isin(np.rint(class_channel).astype(np.int32), [3, 4])
        crop_ys, crop_xs = np.where(crop_mask)
        if crop_ys.size == 0:
            crop_ys, crop_xs = ys, xs
        box = _person_box_from_points(crop_xs, crop_ys, width, height, margin)
        rows.append(
            {
                "person_id": int(round(float(object_id_float))),
                "center": box["center"],
                "scale": box["scale"],
                "bbox": box["bbox"],
                "bbox_norm": box["bbox_norm"],
                "target": target.reshape(-1),
                "target_mask": target_mask.reshape(-1),
                "valid_points": int(len(ordered_ids)),
            }
        )
    return rows


def build_dataset(
    task: str,
    wsv_paths: list[Path],
    out_path: Path,
    frame_stride: int,
    max_frames_per_video: int | None,
    max_keypoints: int,
    max_slots: int,
    margin: int,
) -> int:
    rows = {
        "imgname": [],
        "center": [],
        "scale": [],
        "bbox": [],
        "bbox_norm": [],
        "target": [],
        "target_mask": [],
        "wsv_path": [],
        "frame_key": [],
        "person_id": [],
        "valid_points": [],
    }
    target_format = "obj_bfs_delta_tree" if task == "obj" else f"{task}_keypoints"

    for wsv_path in tqdm(wsv_paths, desc=f"build {task} {out_path.name}"):
        try:
            data = np.load(wsv_path)
        except Exception as exc:
            print(f"[skip] failed to open {wsv_path}: {exc}", flush=True)
            continue
        with data:
            for key, frame in iter_np_frames(data, frame_stride, max_frames_per_video):
                image_path = frame_key_to_image_path(wsv_path, key) if key != "__single__" else wsv_path
                if task in {"dwpose", "mediapipe"} and not image_path.exists():
                    continue
                if task in {"dwpose", "mediapipe"}:
                    extracted = _extract_keypoint_rows(frame, task, max_keypoints, margin)
                    for item in extracted:
                        rows["imgname"].append(str(image_path))
                        rows["center"].append(item["center"])
                        rows["scale"].append(item["scale"])
                        rows["bbox"].append(item["bbox"])
                        rows["bbox_norm"].append(item["bbox_norm"])
                        rows["target"].append(item["target"])
                        rows["target_mask"].append(item["target_mask"])
                        rows["wsv_path"].append(str(wsv_path))
                        rows["frame_key"].append(str(key))
                        rows["person_id"].append(item["person_id"])
                        rows["valid_points"].append(item["valid_points"])
                elif task == "obj":
                    extracted = _extract_obj_joint_xyz_rows(frame, max_slots=max_slots, margin=margin)
                    for item in extracted:
                        rows["imgname"].append(str(image_path))
                        rows["center"].append(item["center"])
                        rows["scale"].append(item["scale"])
                        rows["bbox"].append(item["bbox"])
                        rows["bbox_norm"].append(item["bbox_norm"])
                        rows["target"].append(item["target"])
                        rows["target_mask"].append(item["target_mask"])
                        rows["wsv_path"].append(str(wsv_path))
                        rows["frame_key"].append(str(key))
                        rows["person_id"].append(item["person_id"])
                        rows["valid_points"].append(item["valid_points"])
                else:
                    raise ValueError(f"Unsupported task: {task}")

    if not rows["imgname"]:
        raise RuntimeError(f"No usable WSV {task} samples were extracted for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        task=np.asarray(task),
        imgname=np.asarray(rows["imgname"]),
        center=np.asarray(rows["center"], dtype=np.float32),
        scale=np.asarray(rows["scale"], dtype=np.float32),
        bbox=np.asarray(rows["bbox"], dtype=np.float32),
        bbox_norm=np.asarray(rows["bbox_norm"], dtype=np.float32),
        target=np.asarray(rows["target"], dtype=np.float32),
        target_mask=np.asarray(rows["target_mask"], dtype=np.float32),
        wsv_path=np.asarray(rows["wsv_path"]),
        frame_key=np.asarray(rows["frame_key"]),
        person_id=np.asarray(rows["person_id"], dtype=np.int32),
        valid_points=np.asarray(rows["valid_points"], dtype=np.int32),
        max_keypoints=np.asarray(max_keypoints, dtype=np.int32),
        max_slots=np.asarray(max_slots, dtype=np.int32),
        target_format=np.asarray(target_format),
    )
    print(f"Saved {len(rows['imgname'])} {task} samples -> {out_path}", flush=True)
    return len(rows["imgname"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["dwpose", "mediapipe", "obj"])
    parser.add_argument("--train-root", action="append", default=[])
    parser.add_argument("--val-root", action="append", default=[])
    parser.add_argument("--out-dir", default="data/datasets")
    parser.add_argument("--train-name", required=True)
    parser.add_argument("--val-name", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--max-keypoints", type=int, default=160)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--split-val-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    train_paths = iter_structure_files(args.train_root, args.task)
    val_paths = iter_structure_files(args.val_root, args.task)
    if not val_paths and train_paths:
        train_paths, val_paths = split_train_val(train_paths, args.split_val_fraction, args.split_seed)
    if args.max_videos is not None:
        train_paths = train_paths[: args.max_videos]
        val_paths = val_paths[: args.max_videos]

    out_dir = Path(args.out_dir)
    build_dataset(args.task, train_paths, out_dir / args.train_name, args.frame_stride, args.max_frames_per_video, args.max_keypoints, args.max_slots, args.margin)
    if val_paths:
        try:
            build_dataset(args.task, val_paths, out_dir / args.val_name, args.frame_stride, args.max_frames_per_video, args.max_keypoints, args.max_slots, args.margin)
        except RuntimeError as exc:
            print(f"[warn] {exc}; falling back to train paths for validation smoke/cache build", flush=True)
            build_dataset(args.task, train_paths, out_dir / args.val_name, args.frame_stride, args.max_frames_per_video, args.max_keypoints, args.max_slots, args.margin)


if __name__ == "__main__":
    main()
