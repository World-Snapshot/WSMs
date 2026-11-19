from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from wsv_smpl_utils import extract_people_from_wsg_frame, frame_key_to_image_path, iter_wsv_files, split_train_val


def build_dataset(
    wsv_paths: list[Path],
    out_path: Path,
    frame_stride: int,
    max_frames_per_video: int | None,
    box_source: str,
) -> None:
    rows = {
        "imgname": [],
        "center": [],
        "scale": [],
        "bbox": [],
        "bbox_norm": [],
        "has_bbox": [],
        "body_pose": [],
        "has_body_pose": [],
        "betas": [],
        "has_betas": [],
        "body_keypoints_2d": [],
        "extra_keypoints_2d": [],
        "body_keypoints_3d": [],
        "extra_keypoints_3d": [],
        "wsv_path": [],
        "frame_key": [],
        "person_id": [],
    }

    for wsv_path in tqdm(wsv_paths, desc=f"build {out_path.name}"):
        try:
            data = np.load(wsv_path)
        except Exception as exc:
            print(f"[skip] failed to open {wsv_path}: {exc}")
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
                for person in extract_people_from_wsg_frame(frame, box_source=box_source):
                    rows["imgname"].append(str(image_path))
                    rows["center"].append(person["center"])
                    rows["scale"].append(person["scale"])
                    rows["bbox"].append(person["bbox"])
                    rows["bbox_norm"].append(person["bbox_norm"])
                    rows["has_bbox"].append(1.0)
                    rows["body_pose"].append(person["body_pose"])
                    rows["has_body_pose"].append(1.0)
                    rows["betas"].append(person["betas"])
                    rows["has_betas"].append(1.0)
                    rows["body_keypoints_2d"].append(np.zeros((25, 3), dtype=np.float32))
                    rows["extra_keypoints_2d"].append(np.zeros((19, 3), dtype=np.float32))
                    rows["body_keypoints_3d"].append(np.zeros((25, 4), dtype=np.float32))
                    rows["extra_keypoints_3d"].append(np.zeros((19, 4), dtype=np.float32))
                    rows["wsv_path"].append(str(wsv_path))
                    rows["frame_key"].append(str(key))
                    rows["person_id"].append(person["person_id"])

    if not rows["imgname"]:
        raise RuntimeError(f"No usable WSV/SMPL samples were extracted for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        imgname=np.asarray(rows["imgname"]),
        center=np.asarray(rows["center"], dtype=np.float32),
        scale=np.asarray(rows["scale"], dtype=np.float32),
        bbox=np.asarray(rows["bbox"], dtype=np.float32),
        bbox_norm=np.asarray(rows["bbox_norm"], dtype=np.float32),
        has_bbox=np.asarray(rows["has_bbox"], dtype=np.float32),
        body_pose=np.asarray(rows["body_pose"], dtype=np.float32),
        has_body_pose=np.asarray(rows["has_body_pose"], dtype=np.float32),
        betas=np.asarray(rows["betas"], dtype=np.float32),
        has_betas=np.asarray(rows["has_betas"], dtype=np.float32),
        body_keypoints_2d=np.asarray(rows["body_keypoints_2d"], dtype=np.float32),
        extra_keypoints_2d=np.asarray(rows["extra_keypoints_2d"], dtype=np.float32),
        body_keypoints_3d=np.asarray(rows["body_keypoints_3d"], dtype=np.float32),
        extra_keypoints_3d=np.asarray(rows["extra_keypoints_3d"], dtype=np.float32),
        wsv_path=np.asarray(rows["wsv_path"]),
        frame_key=np.asarray(rows["frame_key"]),
        person_id=np.asarray(rows["person_id"], dtype=np.int32),
        box_source=np.asarray(box_source),
    )
    print(f"Saved {len(rows['imgname'])} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", action="append", default=[])
    parser.add_argument("--val-root", action="append", default=[])
    parser.add_argument("--out-dir", default="data/datasets")
    parser.add_argument("--train-name", default="wsv_smpl_train.npz")
    parser.add_argument("--val-name", default="wsv_smpl_val.npz")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--box-source", default="mediapipe95", choices=["mediapipe95", "smplerx_region"])
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
    build_dataset(train_paths, out_dir / args.train_name, args.frame_stride, args.max_frames_per_video, args.box_source)
    if val_paths:
        build_dataset(val_paths, out_dir / args.val_name, args.frame_stride, args.max_frames_per_video, args.box_source)


if __name__ == "__main__":
    main()
