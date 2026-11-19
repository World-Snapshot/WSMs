from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


MASK_VALUE = -1.0


@dataclass(frozen=True)
class DWPoseHeatmapCodec:
    """Semantic 3-channel heatmap codec for WSG DWPose lanes.

    ``to_heatmap`` consumes raw WSG sparse values in C,T,H,W layout:
    channel 0 = person id, channel 1 = keypoint id, channel 2 = score.
    It emits a normalized heatmap image in [0, 1] for VAE/model input.

    ``from_heatmap`` intentionally maps that normalized heatmap directly back
    to raw WSG sparse values instead of using normal WSG denormalization.
    Sparse ids are discrete semantic labels; ordinary continuous/log
    denormalization can produce invalid labels such as 1.1 or 65.7. Here the
    codec owns the quantization and rounds/clamps ids back to legal labels.
    """

    sigma: float = 3.0
    radius: int = 8
    peak_threshold: float = 0.10
    nms_radius: int = 6
    max_points_per_frame: int = 512
    person_id_max: int = 20
    keypoint_id_max: int = 127

    modality: str = "64-66"

    def _kernel(self) -> np.ndarray:
        radius = max(1, int(self.radius))
        sigma = max(float(self.sigma), 1e-6)
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
        kernel[kernel < 1e-4] = 0.0
        return kernel

    def to_heatmap(self, sparse: np.ndarray) -> np.ndarray:
        if sparse.ndim != 4 or sparse.shape[0] < 3:
            raise ValueError(f"DWPoseHeatmapCodec expects C,T,H,W with at least 3 channels, got {sparse.shape}")
        sparse = sparse[:3].astype(np.float32, copy=False)
        _c, frames, height, width = sparse.shape
        heatmap = np.full((3, frames, height, width), MASK_VALUE, dtype=np.float32)
        heatmap[2] = 0.0
        weight_map = np.zeros((frames, height, width), dtype=np.float32)
        kernel = self._kernel()
        radius = kernel.shape[0] // 2

        valid = (
            (sparse[0] != MASK_VALUE)
            & (sparse[1] != MASK_VALUE)
            & (sparse[2] != MASK_VALUE)
            & np.isfinite(sparse[0])
            & np.isfinite(sparse[1])
            & np.isfinite(sparse[2])
        )
        for frame_idx in range(frames):
            ys, xs = np.where(valid[frame_idx])
            for y, x in zip(ys.tolist(), xs.tolist()):
                if float(sparse[2, frame_idx, y, x]) <= 0.0:
                    continue
                score = 1.0
                y0 = max(0, y - radius)
                y1 = min(height, y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)
                ky0 = y0 - (y - radius)
                ky1 = ky0 + (y1 - y0)
                kx0 = x0 - (x - radius)
                kx1 = kx0 + (x1 - x0)
                local_weight = kernel[ky0:ky1, kx0:kx1] * score
                if local_weight.size == 0:
                    continue
                current_weight = weight_map[frame_idx, y0:y1, x0:x1]
                replace = local_weight > current_weight
                if np.any(replace):
                    person_id = np.clip(np.rint(sparse[0, frame_idx, y, x]), 0, self.person_id_max)
                    keypoint_id = np.clip(np.rint(sparse[1, frame_idx, y, x]), 0, self.keypoint_id_max)
                    heatmap[0, frame_idx, y0:y1, x0:x1][replace] = person_id / float(self.person_id_max)
                    heatmap[1, frame_idx, y0:y1, x0:x1][replace] = keypoint_id / float(self.keypoint_id_max)
                    current_weight[replace] = local_weight[replace]
                score_view = heatmap[2, frame_idx, y0:y1, x0:x1]
                np.maximum(score_view, local_weight, out=score_view)
        return heatmap

    def _local_maxima(self, score: np.ndarray) -> np.ndarray:
        threshold = float(self.peak_threshold)
        nms_radius = max(1, int(self.nms_radius))
        score = np.asarray(score, dtype=np.float32)
        padded = np.pad(score, nms_radius, mode="constant", constant_values=-np.inf)
        local_max = score > threshold
        center = padded[nms_radius : nms_radius + score.shape[0], nms_radius : nms_radius + score.shape[1]]
        for dy in range(-nms_radius, nms_radius + 1):
            for dx in range(-nms_radius, nms_radius + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[
                    nms_radius + dy : nms_radius + dy + score.shape[0],
                    nms_radius + dx : nms_radius + dx + score.shape[1],
                ]
                local_max &= center >= shifted
        candidates = np.argwhere(local_max)
        if candidates.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        values = score[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(values)[::-1]
        selected: list[tuple[int, int]] = []
        suppressed = np.zeros(score.shape, dtype=bool)
        height, width = score.shape
        for idx in order.tolist():
            y, x = int(candidates[idx, 0]), int(candidates[idx, 1])
            if suppressed[y, x]:
                continue
            selected.append((y, x))
            if len(selected) >= int(self.max_points_per_frame):
                break
            y0 = max(0, y - nms_radius)
            y1 = min(height, y + nms_radius + 1)
            x0 = max(0, x - nms_radius)
            x1 = min(width, x + nms_radius + 1)
            suppressed[y0:y1, x0:x1] = True
        return np.asarray(selected, dtype=np.int64)

    def from_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.ndim != 4 or heatmap.shape[0] < 3:
            raise ValueError(f"DWPoseHeatmapCodec expects C,T,H,W with at least 3 channels, got {heatmap.shape}")
        heatmap = heatmap[:3].astype(np.float32, copy=False)
        _c, frames, height, width = heatmap.shape
        sparse = np.full((3, frames, height, width), MASK_VALUE, dtype=np.float32)
        for frame_idx in range(frames):
            score = np.clip(heatmap[2, frame_idx], 0.0, 1.0)
            valid = score > float(self.peak_threshold)
            if not np.any(valid):
                continue
            persons = np.rint(np.clip(heatmap[0, frame_idx], 0.0, 1.0) * self.person_id_max).astype(np.int32)
            keypoints = np.rint(np.clip(heatmap[1, frame_idx], 0.0, 1.0) * self.keypoint_id_max).astype(np.int32)
            pairs = np.unique(np.stack([persons[valid], keypoints[valid]], axis=1), axis=0)
            for person_id, keypoint_id in pairs.tolist():
                group = valid & (persons == int(person_id)) & (keypoints == int(keypoint_id))
                if not np.any(group):
                    continue
                group_score = np.where(group, score, 0.0)
                peaks = self._local_maxima(group_score)
                if peaks.size == 0:
                    y, x = np.unravel_index(int(np.argmax(group_score)), group_score.shape)
                else:
                    values = group_score[peaks[:, 0], peaks[:, 1]]
                    y, x = peaks[int(np.argmax(values))].tolist()
                sparse[0, frame_idx, y, x] = float(np.clip(person_id, 0, self.person_id_max))
                sparse[1, frame_idx, y, x] = float(np.clip(keypoint_id, 0, self.keypoint_id_max))
                sparse[2, frame_idx, y, x] = float(score[y, x])
        return sparse


@dataclass(frozen=True)
class DWPosePersonKeypointPack3HeatmapCodec:
    """Pack3 person-keypoint heatmap codec for DWPose sparse lanes.

    Unlike :class:`DWPoseHeatmapCodec`, this codec does not store person/keypoint
    ids as colors inside one image. Each fixed (person_id, keypoint_id) slot owns
    one heatmap lane, and three lanes are packed into RGB before Wan VAE encoding.
    The slot index carries the discrete semantics, while the VAE only has to
    preserve a smooth score blob and its location.
    """

    sigma: float = 3.0
    radius: int = 8
    peak_threshold: float = 0.10
    nms_radius: int = 6
    max_persons: int = 7
    keypoint_id_max: int = 127
    modality: str = "64-66"

    @property
    def heatmap_lanes(self) -> int:
        return int(self.max_persons) * (int(self.keypoint_id_max) + 1)

    @property
    def pack_count(self) -> int:
        return int(ceil(self.heatmap_lanes / 3.0))

    @property
    def encoded_channels(self) -> int:
        return self.pack_count * 3

    @property
    def latent_lane_count(self) -> int:
        return self.pack_count

    def slot_modality_names(self) -> list[str]:
        return [f"{self.modality}#pkpack3_{idx:03d}" for idx in range(self.pack_count)]

    def _kernel(self) -> np.ndarray:
        radius = max(1, int(self.radius))
        sigma = max(float(self.sigma), 1e-6)
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
        kernel[kernel < 1e-4] = 0.0
        return kernel

    def _slot_index(self, person_id: int, keypoint_id: int) -> int | None:
        if person_id < 0 or person_id >= int(self.max_persons):
            return None
        if keypoint_id < 0 or keypoint_id > int(self.keypoint_id_max):
            return None
        return person_id * (int(self.keypoint_id_max) + 1) + keypoint_id

    def _slot_ids(self, slot_idx: int) -> tuple[int, int]:
        keypoint_count = int(self.keypoint_id_max) + 1
        return slot_idx // keypoint_count, slot_idx % keypoint_count

    def to_heatmap(self, sparse: np.ndarray) -> np.ndarray:
        if sparse.ndim != 4 or sparse.shape[0] < 3:
            raise ValueError(
                f"DWPosePersonKeypointPack3HeatmapCodec expects C,T,H,W with at least 3 channels, got {sparse.shape}"
            )
        sparse = sparse[:3].astype(np.float32, copy=False)
        _c, frames, height, width = sparse.shape
        heatmap = np.zeros((self.encoded_channels, frames, height, width), dtype=np.float32)
        kernel = self._kernel()
        radius = kernel.shape[0] // 2
        valid = (
            (sparse[0] != MASK_VALUE)
            & (sparse[1] != MASK_VALUE)
            & (sparse[2] != MASK_VALUE)
            & np.isfinite(sparse[0])
            & np.isfinite(sparse[1])
            & np.isfinite(sparse[2])
        )
        for frame_idx in range(frames):
            ys, xs = np.where(valid[frame_idx])
            for y, x in zip(ys.tolist(), xs.tolist()):
                if float(sparse[2, frame_idx, y, x]) <= 0.0:
                    continue
                person_id = int(np.rint(sparse[0, frame_idx, y, x]))
                keypoint_id = int(np.rint(sparse[1, frame_idx, y, x]))
                slot_idx = self._slot_index(person_id, keypoint_id)
                if slot_idx is None:
                    continue
                pack_idx = slot_idx // 3
                channel_idx = pack_idx * 3 + (slot_idx % 3)
                y0 = max(0, y - radius)
                y1 = min(height, y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)
                ky0 = y0 - (y - radius)
                ky1 = ky0 + (y1 - y0)
                kx0 = x0 - (x - radius)
                kx1 = kx0 + (x1 - x0)
                local_weight = kernel[ky0:ky1, kx0:kx1]
                if local_weight.size == 0:
                    continue
                score_view = heatmap[channel_idx, frame_idx, y0:y1, x0:x1]
                np.maximum(score_view, local_weight, out=score_view)
        return heatmap

    def from_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.ndim != 4:
            raise ValueError(f"Expected heatmap in C,T,H,W layout, got {heatmap.shape}")
        heatmap = heatmap.astype(np.float32, copy=False)
        _channels, frames, height, width = heatmap.shape
        sparse = np.full((3, frames, height, width), MASK_VALUE, dtype=np.float32)
        slot_limit = min(self.heatmap_lanes, heatmap.shape[0])
        for slot_idx in range(slot_limit):
            pack_idx = slot_idx // 3
            channel_idx = pack_idx * 3 + (slot_idx % 3)
            if channel_idx >= heatmap.shape[0]:
                break
            person_id, keypoint_id = self._slot_ids(slot_idx)
            for frame_idx in range(frames):
                score = np.clip(heatmap[channel_idx, frame_idx], 0.0, 1.0)
                peak = float(np.max(score))
                if peak <= float(self.peak_threshold):
                    continue
                y, x = np.unravel_index(int(np.argmax(score)), score.shape)
                sparse[0, frame_idx, y, x] = float(person_id)
                sparse[1, frame_idx, y, x] = float(keypoint_id)
                sparse[2, frame_idx, y, x] = peak
        return sparse


@dataclass(frozen=True)
class DWPoseBodyPartitionHeatmapCodec:
    """Person/body-partition DWPose heatmap codec.

    This is the debug-proven intermediate scheme: each person and each coarse
    body partition gets its own RGB heatmap lane, so person identity is carried
    by the lane index while the local keypoint id is still encoded inside the
    lane. It is less precise than person-keypoint pack3, but much smaller.
    """

    sigma: float = 3.0
    radius: int = 8
    peak_threshold: float = 0.10
    nms_radius: int = 6
    max_persons: int = 7
    modality: str = "64-66"

    groups: tuple[tuple[str, int, int, int], ...] = (
        ("body", 0, 18, 18),
        ("face", 18, 68, 68),
        ("left_hand", 86, 21, 21),
        ("right_hand", 107, 21, 21),
    )

    @property
    def latent_lane_count(self) -> int:
        return int(self.max_persons) * len(self.groups)

    def slot_modality_names(self) -> list[str]:
        names = []
        for person_id in range(int(self.max_persons)):
            for name, _offset, _count, _scale_count in self.groups:
                names.append(f"{self.modality}#person{person_id}_{name}")
        return names

    def _kernel(self) -> np.ndarray:
        radius = max(1, int(self.radius))
        sigma = max(float(self.sigma), 1e-6)
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
        kernel[kernel < 1e-4] = 0.0
        return kernel

    def _codebook(self, count: int) -> np.ndarray:
        cols = int(np.ceil(np.sqrt(max(count, 1))))
        rows = int(np.ceil(float(max(count, 1)) / float(cols)))
        xs = np.linspace(0.08, 0.92, cols, dtype=np.float32)
        ys = np.linspace(0.08, 0.92, rows, dtype=np.float32)
        candidates = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
        if count <= 0:
            return np.empty((0, 2), dtype=np.float32)
        selected = [0]
        remaining = set(range(1, len(candidates)))
        while len(selected) < int(count) and remaining:
            best_idx = max(
                remaining,
                key=lambda idx: float(np.min(np.linalg.norm(candidates[idx] - candidates[selected], axis=1))),
            )
            selected.append(int(best_idx))
            remaining.remove(int(best_idx))
        return candidates[selected[: int(count)]]

    def to_heatmap(self, sparse: np.ndarray) -> np.ndarray:
        if sparse.ndim != 4 or sparse.shape[0] < 3:
            raise ValueError(f"DWPoseBodyPartitionHeatmapCodec expects C,T,H,W with at least 3 channels, got {sparse.shape}")
        sparse = sparse[:3].astype(np.float32, copy=False)
        _c, frames, height, width = sparse.shape
        heatmap = np.zeros((self.latent_lane_count * 3, frames, height, width), dtype=np.float32)
        winners = np.zeros((self.latent_lane_count, frames, height, width), dtype=np.float32)
        codebooks = [self._codebook(count) for _name, _offset, count, _scale_count in self.groups]
        kernel = self._kernel()
        radius = kernel.shape[0] // 2
        valid = (
            (sparse[0] != MASK_VALUE)
            & (sparse[1] != MASK_VALUE)
            & (sparse[2] != MASK_VALUE)
            & np.isfinite(sparse[0])
            & np.isfinite(sparse[1])
            & np.isfinite(sparse[2])
        )
        for frame_idx in range(frames):
            ys, xs = np.where(valid[frame_idx])
            for y, x in zip(ys.tolist(), xs.tolist()):
                if float(sparse[2, frame_idx, y, x]) <= 0.0:
                    continue
                person_id = int(np.rint(sparse[0, frame_idx, y, x]))
                keypoint_id = int(np.rint(sparse[1, frame_idx, y, x]))
                if person_id < 0 or person_id >= int(self.max_persons):
                    continue
                group_idx = None
                local_key = None
                scale_count = None
                for idx, (_name, offset, count, group_scale_count) in enumerate(self.groups):
                    if offset <= keypoint_id < offset + count:
                        group_idx = idx
                        local_key = keypoint_id - offset
                        scale_count = group_scale_count
                        break
                if group_idx is None or local_key is None or scale_count is None:
                    continue
                lane_idx = person_id * len(self.groups) + group_idx
                base_ch = lane_idx * 3
                code0, code1 = codebooks[group_idx][int(np.clip(local_key, 0, scale_count - 1))].tolist()
                y0 = max(0, y - radius)
                y1 = min(height, y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)
                ky0 = y0 - (y - radius)
                ky1 = ky0 + (y1 - y0)
                kx0 = x0 - (x - radius)
                kx1 = kx0 + (x1 - x0)
                local = kernel[ky0:ky1, kx0:kx1]
                replace = local > winners[lane_idx, frame_idx, y0:y1, x0:x1]
                if np.any(replace):
                    heatmap[base_ch + 0, frame_idx, y0:y1, x0:x1][replace] = code0
                    heatmap[base_ch + 1, frame_idx, y0:y1, x0:x1][replace] = code1
                    winners[lane_idx, frame_idx, y0:y1, x0:x1][replace] = local[replace]
                np.maximum(heatmap[base_ch + 2, frame_idx, y0:y1, x0:x1], local, out=heatmap[base_ch + 2, frame_idx, y0:y1, x0:x1])
        return heatmap

    def from_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.ndim != 4:
            raise ValueError(f"Expected heatmap in C,T,H,W layout, got {heatmap.shape}")
        heatmap = heatmap.astype(np.float32, copy=False)
        _channels, frames, height, width = heatmap.shape
        sparse = np.full((3, frames, height, width), MASK_VALUE, dtype=np.float32)
        codebooks = [self._codebook(count) for _name, _offset, count, _scale_count in self.groups]
        lanes = min(self.latent_lane_count, heatmap.shape[0] // 3)
        for lane_idx in range(lanes):
            person_id = lane_idx // len(self.groups)
            group_idx = lane_idx % len(self.groups)
            _name, offset, count, scale_count = self.groups[group_idx]
            codebook = codebooks[group_idx]
            base_ch = lane_idx * 3
            for frame_idx in range(frames):
                score = np.clip(heatmap[base_ch + 2, frame_idx], 0.0, 1.0)
                valid = score > float(self.peak_threshold)
                if not np.any(valid):
                    continue
                code = np.stack(
                    [
                        np.clip(heatmap[base_ch + 0, frame_idx], 0.0, 1.0),
                        np.clip(heatmap[base_ch + 1, frame_idx], 0.0, 1.0),
                    ],
                    axis=-1,
                )
                distances = np.linalg.norm(code[:, :, None, :] - codebook[None, None, :, :], axis=-1)
                local_keys = np.argmin(distances, axis=-1).astype(np.int32)
                for local_key in np.unique(local_keys[valid]).tolist():
                    if not (0 <= int(local_key) < count):
                        continue
                    group_mask = valid & (local_keys == int(local_key))
                    if not np.any(group_mask):
                        continue
                    group_score = np.where(group_mask, score, 0.0)
                    y, x = np.unravel_index(int(np.argmax(group_score)), group_score.shape)
                    sparse[0, frame_idx, y, x] = float(person_id)
                    sparse[1, frame_idx, y, x] = float(offset + int(local_key))
                    sparse[2, frame_idx, y, x] = float(score[y, x])
        return sparse


@dataclass(frozen=True)
class DWPoseBodyOnlyHeatmapCodec(DWPoseBodyPartitionHeatmapCodec):
    """Body-only DWPose heatmap codec with a 2D keypoint codebook.

    This codec keeps only the 18 body keypoints and drops face/hand points.
    Unlike the generic body-partition codec, local body key ids are encoded as
    well-spaced 2D codebook colors instead of a narrow one-dimensional ramp.
    Wan VAE can slightly shift reconstructed colors; nearest-codebook decoding
    gives the body-only lane more tolerance to that color drift.
    """

    groups: tuple[tuple[str, int, int, int], ...] = (("body", 0, 18, 18),)

    def _codebook(self, count: int) -> np.ndarray:
        xs = np.linspace(0.10, 0.90, 5, dtype=np.float32)
        ys = np.linspace(0.10, 0.90, 4, dtype=np.float32)
        candidates = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
        selected = [0]
        remaining = set(range(1, len(candidates)))
        while len(selected) < int(count) and remaining:
            best_idx = max(
                remaining,
                key=lambda idx: float(np.min(np.linalg.norm(candidates[idx] - candidates[selected], axis=1))),
            )
            selected.append(int(best_idx))
            remaining.remove(int(best_idx))
        return candidates[selected[: int(count)]]

    def to_heatmap(self, sparse: np.ndarray) -> np.ndarray:
        if sparse.ndim != 4 or sparse.shape[0] < 3:
            raise ValueError(f"DWPoseBodyOnlyHeatmapCodec expects C,T,H,W with at least 3 channels, got {sparse.shape}")
        sparse = sparse[:3].astype(np.float32, copy=False)
        _c, frames, height, width = sparse.shape
        heatmap = np.zeros((self.latent_lane_count * 3, frames, height, width), dtype=np.float32)
        winners = np.zeros((self.latent_lane_count, frames, height, width), dtype=np.float32)
        kernel = self._kernel()
        radius = kernel.shape[0] // 2
        _name, offset, count, _scale_count = self.groups[0]
        codebook = self._codebook(count)
        valid = (
            (sparse[0] != MASK_VALUE)
            & (sparse[1] != MASK_VALUE)
            & (sparse[2] != MASK_VALUE)
            & np.isfinite(sparse[0])
            & np.isfinite(sparse[1])
            & np.isfinite(sparse[2])
        )
        valid &= sparse[1] >= offset
        valid &= sparse[1] < offset + count
        for frame_idx in range(frames):
            ys, xs = np.where(valid[frame_idx])
            for y, x in zip(ys.tolist(), xs.tolist()):
                if float(sparse[2, frame_idx, y, x]) <= 0.0:
                    continue
                person_id = int(np.rint(sparse[0, frame_idx, y, x]))
                if person_id < 0 or person_id >= int(self.max_persons):
                    continue
                local_key = int(np.clip(np.rint(sparse[1, frame_idx, y, x] - offset), 0, count - 1))
                lane_idx = person_id
                base_ch = lane_idx * 3
                code0, code1 = codebook[local_key].tolist()
                y0 = max(0, y - radius)
                y1 = min(height, y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(width, x + radius + 1)
                ky0 = y0 - (y - radius)
                ky1 = ky0 + (y1 - y0)
                kx0 = x0 - (x - radius)
                kx1 = kx0 + (x1 - x0)
                local = kernel[ky0:ky1, kx0:kx1]
                replace = local > winners[lane_idx, frame_idx, y0:y1, x0:x1]
                if np.any(replace):
                    heatmap[base_ch + 0, frame_idx, y0:y1, x0:x1][replace] = code0
                    heatmap[base_ch + 1, frame_idx, y0:y1, x0:x1][replace] = code1
                    winners[lane_idx, frame_idx, y0:y1, x0:x1][replace] = local[replace]
                np.maximum(heatmap[base_ch + 2, frame_idx, y0:y1, x0:x1], local, out=heatmap[base_ch + 2, frame_idx, y0:y1, x0:x1])
        return heatmap

    def from_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.ndim != 4:
            raise ValueError(f"Expected heatmap in C,T,H,W layout, got {heatmap.shape}")
        heatmap = heatmap.astype(np.float32, copy=False)
        _channels, frames, height, width = heatmap.shape
        sparse = np.full((3, frames, height, width), MASK_VALUE, dtype=np.float32)
        _name, offset, count, _scale_count = self.groups[0]
        codebook = self._codebook(count)
        lanes = min(self.latent_lane_count, heatmap.shape[0] // 3)
        for lane_idx in range(lanes):
            person_id = lane_idx
            base_ch = lane_idx * 3
            for frame_idx in range(frames):
                score = np.clip(heatmap[base_ch + 2, frame_idx], 0.0, 1.0)
                valid = score > float(self.peak_threshold)
                if not np.any(valid):
                    continue
                code = np.stack(
                    [
                        np.clip(heatmap[base_ch + 0, frame_idx], 0.0, 1.0),
                        np.clip(heatmap[base_ch + 1, frame_idx], 0.0, 1.0),
                    ],
                    axis=-1,
                )
                distances = np.linalg.norm(code[:, :, None, :] - codebook[None, None, :, :], axis=-1)
                local_keys = np.argmin(distances, axis=-1).astype(np.int32)
                for local_key in np.unique(local_keys[valid]).tolist():
                    if not (0 <= int(local_key) < count):
                        continue
                    group_mask = valid & (local_keys == int(local_key))
                    if not np.any(group_mask):
                        continue
                    group_score = np.where(group_mask, score, 0.0)
                    y, x = np.unravel_index(int(np.argmax(group_score)), group_score.shape)
                    sparse[0, frame_idx, y, x] = float(person_id)
                    sparse[1, frame_idx, y, x] = float(offset + int(local_key))
                    sparse[2, frame_idx, y, x] = float(score[y, x])
        return sparse
