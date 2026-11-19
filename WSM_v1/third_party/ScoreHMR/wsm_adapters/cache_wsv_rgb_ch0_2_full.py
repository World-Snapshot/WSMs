from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_rgb(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    with np.load(args.dataset) as data:
        imgnames = data["imgname"].astype(str)

    images = []
    for imgname in tqdm(imgnames, desc=f"cache full rgb ch0-2 {Path(args.dataset).name}"):
        images.append(load_rgb(imgname, args.image_size).astype(np.float16))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb_ch0_2_full=np.stack(images, axis=0),
        imgname=imgnames,
        image_size=np.asarray(args.image_size, dtype=np.int32),
    )
    print(f"Saved full RGB ch0-2 cache -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
