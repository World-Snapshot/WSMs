from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_frame(path: str, frame_key: str) -> np.ndarray:
    with np.load(path) as data:
        if frame_key == "__single__" and "wsg_data" in data.files:
            return data["wsg_data"]
        if frame_key in data.files:
            return data[frame_key]
        if "wsg_data" in data.files:
            return data["wsg_data"]
    raise KeyError(f"Cannot find frame {frame_key} in {path}")


def _rgb_from_wsg(frame: np.ndarray, image_size: int) -> np.ndarray:
    rgb = frame[:, :, :3].astype(np.float32)
    valid = rgb != -1.0
    if np.any(valid) and float(np.nanmax(rgb[valid])) > 1.5:
        rgb[valid] = rgb[valid] / 255.0
    rgb = np.where(valid, rgb, 0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    with np.load(args.dataset) as data:
        paths = data["wsv_path"].astype(str)
        frame_keys = data["frame_key"].astype(str)

    images = []
    for path, frame_key in tqdm(zip(paths, frame_keys), total=len(paths), desc=f"cache WSG full rgb {Path(args.dataset).name}"):
        images.append(_rgb_from_wsg(_load_frame(path, frame_key), args.image_size).astype(np.float16))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb_ch0_2_full=np.stack(images, axis=0),
        source_path=paths,
        frame_key=frame_keys,
        image_size=np.asarray(args.image_size, dtype=np.int32),
    )
    print(f"Saved WSG full RGB ch0-2 cache -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
