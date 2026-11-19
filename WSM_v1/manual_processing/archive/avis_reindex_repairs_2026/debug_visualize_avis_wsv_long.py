#!/usr/bin/env python3
"""Render a long contact sheet for one AVIS WSV in current frame order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WSG_MAIN = Path("/research/cbim/vast/sf895/code/wsg/main")
if str(WSG_MAIN) not in sys.path:
    sys.path.insert(0, str(WSG_MAIN))

import pipeline30b_fix_avis_semantic_lexicographic_order as p30b  # noqa: E402


def _tile(arr: np.ndarray, label: str, width: int) -> Image.Image:
    im = Image.fromarray(arr, "RGB")
    h = max(1, int(round(width * im.height / max(im.width, 1))))
    im = im.resize((width, h), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.width, 20), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    return im


def _row(frames: list[np.ndarray], fn, name: str, width: int) -> Image.Image:
    tiles = [_tile(fn(frame), f"{name} {idx + 1:03d}", width) for idx, frame in enumerate(frames)]
    h = max(tile.height for tile in tiles) if tiles else width
    out = Image.new("RGB", (width * len(tiles), h), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        out.paste(tile, (idx * width, 0))
    return out


def render(wsv: Path, out: Path, *, tile_width: int, max_frames: int | None) -> None:
    with np.load(wsv, allow_pickle=True) as npz:
        keys = [key for key in npz.files if p30b.frame_index_from_key(key) is not None]
        keys = sorted(keys, key=lambda key: p30b.frame_index_from_key(key) or 0)
        if max_frames is not None:
            keys = keys[:max_frames]
        frames = [np.asarray(npz[key]) for key in keys]
    rows = [
        _row(frames, p30b.rgb_from_frame, "rgb", tile_width),
        _row(frames, p30b.edge_from_frame, "edge40", tile_width),
        _row(frames, p30b.normal_from_frame, "normal36-38", tile_width),
        _row(frames, p30b.semantic_from_frame, "semantic195", tile_width),
    ]
    pad = 8
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows) + pad * (len(rows) - 1)), (20, 20, 20))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + pad
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile-width", type=int, default=128)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()
    render(args.wsv, args.out, tile_width=args.tile_width, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
