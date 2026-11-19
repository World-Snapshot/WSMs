from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from wsv_smpl_utils import (
    extract_mediapipe_boxes_from_wsg_frame,
    extract_smpl_boxes_from_wsg_frame,
    frame_key_to_image_path,
    has_dwpose_human_evidence_in_wsg_frame,
    iter_wsv_files,
    split_train_val,
)


def build_dataset(
    wsv_paths: list[Path],
    out_path: Path,
    frame_stride: int,
    max_frames_per_video: int | None,
    max_slots: int,
    box_source: str,
    min_visible_points: int,
    visibility_threshold: float,
    max_box_area_fraction: float,
    bbox_margin_mode: str,
    bbox_margin_pixels: int,
    bbox_margin_fraction: float,
    require_dwpose: bool,
    dwpose_min_points: int,
    dwpose_score_threshold: float | None,
) -> None:
    rows = {
        "imgname": [],
        "boxes": [],
        "has_box": [],
        "wsv_path": [],
        "frame_key": [],
    }

    for wsv_path in tqdm(wsv_paths, desc=f"build {out_path.name}"):
        try:
            data = np.load(wsv_path)
        except Exception as exc:
            print(f"[skip] failed to open {wsv_path}: {exc}", flush=True)
            continue
        with data:
            keys = sorted(data.files)
            if frame_stride > 1:
                keys = keys[::frame_stride]
            if max_frames_per_video is not None:
                keys = keys[:max_frames_per_video]
            for key in keys:
                image_path = frame_key_to_image_path(wsv_path, key)
                if not image_path.exists():
                    continue
                frame = data[key]
                if require_dwpose and not has_dwpose_human_evidence_in_wsg_frame(
                    frame,
                    min_points=dwpose_min_points,
                    score_threshold=dwpose_score_threshold,
                ):
                    continue
                if box_source == "smpl72":
                    person_boxes = extract_smpl_boxes_from_wsg_frame(
                        frame,
                        min_points=min_visible_points,
                        max_area_fraction=max_box_area_fraction,
                        margin_mode=bbox_margin_mode,
                        margin=bbox_margin_pixels,
                        margin_fraction=bbox_margin_fraction,
                    )
                elif box_source == "mediapipe95":
                    person_boxes = extract_mediapipe_boxes_from_wsg_frame(
                        frame,
                        min_points=min_visible_points,
                        visibility_threshold=visibility_threshold,
                        max_area_fraction=max_box_area_fraction,
                        margin_mode=bbox_margin_mode,
                        margin=bbox_margin_pixels,
                        margin_fraction=bbox_margin_fraction,
                    )
                else:
                    raise ValueError(f"Unsupported box_source: {box_source}")
                person_boxes = sorted(person_boxes, key=lambda item: item["person_id"])
                if not person_boxes:
                    continue
                boxes = np.zeros((max_slots, 4), dtype=np.float32)
                has_box = np.zeros((max_slots,), dtype=np.float32)
                for slot, person_box in enumerate(person_boxes[:max_slots]):
                    boxes[slot] = person_box["bbox_norm"]
                    has_box[slot] = 1.0
                rows["imgname"].append(str(image_path))
                rows["boxes"].append(boxes)
                rows["has_box"].append(has_box)
                rows["wsv_path"].append(str(wsv_path))
                rows["frame_key"].append(str(key))

    if not rows["imgname"]:
        raise RuntimeError(f"No usable WSV bbox samples were extracted for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        imgname=np.asarray(rows["imgname"]),
        boxes=np.asarray(rows["boxes"], dtype=np.float32),
        has_box=np.asarray(rows["has_box"], dtype=np.float32),
        wsv_path=np.asarray(rows["wsv_path"]),
        frame_key=np.asarray(rows["frame_key"]),
        max_slots=np.asarray(max_slots, dtype=np.int32),
        box_source=np.asarray(box_source),
        min_visible_points=np.asarray(min_visible_points, dtype=np.int32),
        visibility_threshold=np.asarray(visibility_threshold, dtype=np.float32),
        max_box_area_fraction=np.asarray(max_box_area_fraction, dtype=np.float32),
        bbox_margin_mode=np.asarray(bbox_margin_mode),
        bbox_margin_pixels=np.asarray(bbox_margin_pixels, dtype=np.int32),
        bbox_margin_fraction=np.asarray(bbox_margin_fraction, dtype=np.float32),
        require_dwpose=np.asarray(require_dwpose, dtype=np.bool_),
        dwpose_min_points=np.asarray(dwpose_min_points, dtype=np.int32),
        dwpose_score_threshold=np.asarray(-1.0 if dwpose_score_threshold is None else dwpose_score_threshold, dtype=np.float32),
    )
    print(f"Saved {len(rows['imgname'])} full-image bbox samples -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", action="append", default=[])
    parser.add_argument("--val-root", action="append", default=[])
    parser.add_argument("--out-dir", default="data/datasets")
    parser.add_argument("--train-name", default="wsv_bbox_train.npz")
    parser.add_argument("--val-name", default="wsv_bbox_val.npz")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--max-slots", type=int, default=10)
    parser.add_argument("--box-source", choices=("smpl72", "mediapipe95"), default="smpl72")
    parser.add_argument("--min-visible-points", type=int, default=8)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--max-box-area-fraction", type=float, default=0.8)
    parser.add_argument("--bbox-margin-mode", choices=("pixels", "relative"), default="pixels")
    parser.add_argument("--bbox-margin-pixels", type=int, default=20)
    parser.add_argument("--bbox-margin-fraction", type=float, default=0.2)
    parser.add_argument("--require-dwpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dwpose-min-points", type=int, default=8)
    parser.add_argument("--dwpose-score-threshold", type=float, default=0.5)
    parser.add_argument("--split-val-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    train_paths = iter_wsv_files(args.train_root)
    val_paths = iter_wsv_files(args.val_root)
    if not val_paths and train_paths:
        train_paths, val_paths = split_train_val(train_paths, args.split_val_fraction, args.split_seed)
    if args.max_videos is not None:
        train_paths = train_paths[: args.max_videos]
        val_paths = val_paths[: args.max_videos]

    out_dir = Path(args.out_dir)
    build_dataset(
        train_paths,
        out_dir / args.train_name,
        args.frame_stride,
        args.max_frames_per_video,
        args.max_slots,
        args.box_source,
        args.min_visible_points,
        args.visibility_threshold,
        args.max_box_area_fraction,
        args.bbox_margin_mode,
        args.bbox_margin_pixels,
        args.bbox_margin_fraction,
        args.require_dwpose,
        args.dwpose_min_points,
        args.dwpose_score_threshold,
    )
    if val_paths:
        build_dataset(
            val_paths,
            out_dir / args.val_name,
            args.frame_stride,
            args.max_frames_per_video,
            args.max_slots,
            args.box_source,
            args.min_visible_points,
            args.visibility_threshold,
            args.max_box_area_fraction,
            args.bbox_margin_mode,
            args.bbox_margin_pixels,
            args.bbox_margin_fraction,
            args.require_dwpose,
            args.dwpose_min_points,
            args.dwpose_score_threshold,
        )


if __name__ == "__main__":
    main()
