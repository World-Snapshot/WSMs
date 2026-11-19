from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


MASK_VALUE = -1.0
OBJ_ID_CHANNEL = 111
OBJ_CLASS_CHANNEL = 112


def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def _farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int64)
    count = min(count, points.shape[0])
    center = points.mean(axis=0, keepdims=True)
    first = int(np.argmin(_pairwise_sq(points, center)[:, 0]))
    selected = [first]
    min_dist = _pairwise_sq(points, points[[first]])[:, 0]
    for _ in range(1, count):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, _pairwise_sq(points, points[[idx]])[:, 0])
    return np.asarray(selected, dtype=np.int64)


def _kmeans(points: np.ndarray, count: int, iterations: int = 24) -> np.ndarray:
    if points.shape[0] == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    count = min(count, points.shape[0])
    centers = points[_farthest_point_sample(points, count)].astype(np.float32)
    for _ in range(iterations):
        labels = np.argmin(_pairwise_sq(points, centers), axis=1)
        next_centers = centers.copy()
        for idx in range(count):
            member = points[labels == idx]
            if member.shape[0] > 0:
                next_centers[idx] = member.mean(axis=0)
        if np.max(np.abs(next_centers - centers)) < 1e-5:
            break
        centers = next_centers
    return np.clip(centers, 0.0, 1.0).astype(np.float32)


def _mst_parents(points: np.ndarray) -> dict[int, int]:
    n = points.shape[0]
    if n == 0:
        return {}
    root = int(np.argmin(_pairwise_sq(points, points.mean(axis=0, keepdims=True))[:, 0]))
    parents = {root: root}
    selected = {root}
    dist = _pairwise_sq(points, points)
    while len(selected) < n:
        best: tuple[int, int] | None = None
        best_dist = float("inf")
        for src in selected:
            for dst in range(n):
                if dst in selected:
                    continue
                value = float(dist[src, dst])
                if value < best_dist:
                    best = (src, dst)
                    best_dist = value
        if best is None:
            break
        src, dst = best
        parents[int(dst)] = int(src)
        selected.add(int(dst))
    return parents


def _kmeans_mst_init(pointcloud: np.ndarray, point_mask: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_slots = target_mask.reshape(-1, 7).shape[0]
    init = np.zeros((max_slots, 7), dtype=np.float32)
    init_mask = np.zeros((max_slots, 7), dtype=np.float32)
    points = pointcloud[point_mask > 0.0]
    # This initializer must not peek at the GT skeleton count.  It provides a
    # fixed candidate pool; the model learns which candidates exist.
    node_count = min(max_slots, points.shape[0])
    if points.shape[0] < 3 or node_count <= 0:
        return init.reshape(-1), init_mask.reshape(-1)

    centers = _kmeans(points, node_count)
    parents = _mst_parents(centers)
    parent_den = float(max(1, max_slots - 1))
    for slot in range(min(node_count, max_slots)):
        parent_slot = parents.get(slot, slot)
        parent_xyz = centers[parent_slot] if parent_slot < centers.shape[0] else centers[slot]
        xyz = centers[slot]
        init[slot, 0] = float(parent_slot) / parent_den
        init[slot, 1:4] = np.clip(xyz - parent_xyz, -1.0, 1.0)
        init[slot, 4:7] = xyz
        init_mask[slot, :7] = 1.0
    return init.reshape(-1), init_mask.reshape(-1)


def _load_frame(path: str, frame_key: str) -> np.ndarray:
    with np.load(path) as data:
        if frame_key == "__single__" and "wsg_data" in data.files:
            return data["wsg_data"]
        if frame_key in data.files:
            return data[frame_key]
        if "wsg_data" in data.files:
            return data["wsg_data"]
    raise KeyError(f"Cannot find frame {frame_key} in {path}")


def _object_vertex_pointcloud(frame: np.ndarray, object_id: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    pointcloud = np.zeros((max_points, 3), dtype=np.float32)
    mask = np.zeros((max_points,), dtype=np.float32)
    if frame.ndim != 3 or frame.shape[-1] <= 116:
        return pointcloud, mask

    object_channel = frame[:, :, OBJ_ID_CHANNEL]
    class_channel = frame[:, :, OBJ_CLASS_CHANNEL]
    object_ids = np.rint(object_channel).astype(np.int32)
    class_ids = np.rint(class_channel).astype(np.int32)
    valid = (object_channel != MASK_VALUE) & (class_ids == 1) & (object_ids == int(object_id))
    if not np.any(valid):
        valid = (object_channel != MASK_VALUE) & (class_ids == 1)
    coords = np.asarray(frame[:, :, 114:117][valid], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return pointcloud, mask
    keep = np.isfinite(coords).all(axis=1) & ~np.any(coords == MASK_VALUE, axis=1)
    coords = coords[keep]
    if coords.shape[0] < 3:
        return pointcloud, mask

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    scale = np.maximum(maxs - mins, 1e-6)
    coords = np.clip((coords - mins) / scale, 0.0, 1.0).astype(np.float32)

    # Deterministic subsampling keeps cache reproducible and avoids adding a
    # random data-processing dependency to these tiny structure experiments.
    if coords.shape[0] > max_points:
        indices = np.linspace(0, coords.shape[0] - 1, max_points, dtype=np.int64)
        coords = coords[indices]
    count = min(max_points, coords.shape[0])
    pointcloud[:count] = coords[:count]
    mask[:count] = 1.0
    return pointcloud, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-points", type=int, default=1024)
    args = parser.parse_args()

    with np.load(args.dataset) as data:
        paths = data["wsv_path"].astype(str)
        frame_keys = data["frame_key"].astype(str)
        object_ids = data["person_id"].astype(np.int32)
        target_masks = data["target_mask"].astype(np.float32)

    pointclouds = []
    masks = []
    init_targets = []
    init_masks = []
    iterator = zip(paths, frame_keys, object_ids)
    for index, (path, frame_key, object_id) in enumerate(tqdm(iterator, total=len(paths), desc=f"cache OBJ pointcloud {Path(args.dataset).name}")):
        pointcloud, mask = _object_vertex_pointcloud(_load_frame(path, frame_key), int(object_id), args.max_points)
        pointclouds.append(pointcloud.astype(np.float16))
        masks.append(mask.astype(np.float16))
        init_target, init_mask = _kmeans_mst_init(pointcloud, mask, target_masks[index])
        init_targets.append(init_target.astype(np.float16))
        init_masks.append(init_mask.astype(np.float16))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pointcloud_norm=np.stack(pointclouds, axis=0),
        pointcloud_mask=np.stack(masks, axis=0),
        init_bfs_delta_tree=np.stack(init_targets, axis=0),
        init_bfs_delta_tree_mask=np.stack(init_masks, axis=0),
        source_path=paths,
        frame_key=frame_keys,
        object_id=object_ids,
        max_points=np.asarray(args.max_points, dtype=np.int32),
    )
    print(f"Saved OBJ normalized pointcloud cache -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
