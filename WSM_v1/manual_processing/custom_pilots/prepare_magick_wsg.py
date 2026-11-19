#!/usr/bin/env python3
"""End-to-end MAGICK transparency-pilot data pipeline (Head 1: RGB -> soft alpha).

One self-contained script with four subcommands (run in order):

  select    Pick a broad, reproducible MAGICK subset from the dataset index TSV
            (stratified by keyword category, NSFW filtered) -- no full download.
  download  Fetch only the selected page_ids from HuggingFace, verifying each is
            RGBA with non-constant alpha (skips/repairs bad files, retries 429s).
  prepare   Composite each RGBA foreground over a randomized background (so the
            0-2 RGB no longer carries the trivial "transparent==black" shortcut)
            and write packed WSG npz ({"data": HxWx4}) + captions/manifest.
  to-named  Rename the packed npz into self-describing named-key npz
            ({"0-2": RGB(H,W,3), "LayerDiffuse_Transparency": alpha(H,W,1)}) so it
            no longer borrows WSG numeric channel slots. Pixels are untouched;
            re-encode afterwards. (The old relighting/genlit converter branch was
            dropped -- prepare_genlit_stream.py now writes named keys directly.)

This merges the former select_magick_stratified.py + download_magick_subset.py +
prepare_magick_wsg.py + convert_to_named_keys.py into one file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS = {
    "people": r"\b(woman|man|person|people|girl|boy|child|baby|portrait|face|human|athlete|yogi|bride|groom|model|worker|chef|doctor|artist|soldier)\b",
    "animals": r"\b(cat|dog|puppy|kitten|bird|horse|cow|goat|sheep|lion|tiger|bear|elephant|monkey|panda|rabbit|hamster|fish|shark|whale|dolphin|insect|butterfly|spider|snake|frog|lizard|chicken|duck|eagle|owl)\b",
    "plants": r"\b(flower|plant|tree|leaf|leaves|moss|grass|cactus|rose|sunflower|fern|mushroom|branch|forest|garden|herb|vine|fruit)\b",
    "food": r"\b(food|cake|bread|pizza|burger|steak|sushi|soup|cheese|coffee|tea|fruit|apple|banana|orange|lemon|chocolate|ice cream|dish|plate|meal|drink|bottle|wine)\b",
    "vehicles": r"\b(car|truck|bus|train|airplane|aircraft|plane|helicopter|boat|ship|bicycle|bike|motorcycle|scooter|vehicle|rocket)\b",
    "furniture": r"\b(chair|table|sofa|couch|bed|desk|cabinet|shelf|lamp|furniture|mirror|vase|pillow|carpet)\b",
    "tools_and_objects": r"\b(tool|hammer|saw|chainsaw|knife|scissors|phone|camera|computer|laptop|watch|glasses|key|toy|ball|instrument|guitar|piano|book|umbrella|backpack|bottle|box|container)\b",
    "architecture": r"\b(building|house|home|castle|church|temple|tower|bridge|room|interior|street|city|architecture|door|window|stairs)\b",
    "fashion": r"\b(dress|shirt|coat|jacket|shoe|shoes|hat|帽|clothing|fashion|sweater|jeans|skirt|uniform|costume|outfit)\b",
    "art_and_graphics": r"\b(illustration|drawing|painting|art|cartoon|anime|logo|icon|sculpture|sticker|clipart|watercolor|vector|render)\b",
    "nature_and_landscape": r"\b(mountain|ocean|sea|beach|river|lake|waterfall|sky|cloud|sunset|sunrise|snow|ice|desert|landscape|island|rock|cave)\b",
    "materials_and_transparency": r"\b(glass|crystal|transparent|translucent|reflection|shadow|glow|smoke|mist|water|bubble|jelly|ice|liquid|metal|wood|fabric|silk|plastic)\b",
}


def classify(text: str) -> str:
    for category, pattern in CATEGORY_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return category
    return "other"


def cmd_select(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_csv(args.index, sep="\t")
    df = df[(df["nsfw"].fillna(1).astype(int) == 0)].copy()
    df["category"] = df["subject"].fillna("").map(classify)
    rng = df.sample(frac=1.0, random_state=args.seed)
    selected = []
    for category in list(CATEGORY_PATTERNS) + ["other"]:
        part = rng[rng["category"] == category].head(args.per_category)
        selected.append(part)
    out = pd.concat(selected, ignore_index=True).drop_duplicates("page_id")
    out.to_csv(args.output, sep="\t", index=False)
    print("selected", len(out), "samples")
    print(out["category"].value_counts().to_string())


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def _fetch(row: tuple[str, str], out: Path, timeout: int) -> tuple[str, bool, str]:
    import requests

    page_id, category = row
    target = out / category / f"{page_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with Image.open(target) as im:
                if im.mode == "RGBA":
                    return page_id, True, "cached"
        except Exception:
            target.unlink(missing_ok=True)
    url = f"https://huggingface.co/datasets/OneOverZero/MAGICK/resolve/main/images/{page_id[:2]}/{page_id}.png"
    last_error = "unknown error"
    for attempt in range(6):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "WSM-MAGICK-prep/1.0"})
            if response.status_code == 429:
                last_error = "429 Client Error: Too Many Requests"
                time.sleep(min(60, 3 * (2 ** attempt)))
                continue
            response.raise_for_status()
            target.write_bytes(response.content)
            with Image.open(target) as im:
                if im.mode != "RGBA":
                    raise ValueError(f"mode={im.mode}")
                if im.getchannel("A").getextrema() == (255, 255):
                    raise ValueError("alpha is constant 255")
            return page_id, True, "downloaded"
        except Exception as exc:
            last_error = str(exc)
            if attempt < 5:
                time.sleep(min(30, 2 * (attempt + 1)))
    target.unlink(missing_ok=True)
    return page_id, False, last_error


def cmd_download(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_csv(args.selection, sep="\t")
    rows = list(zip(df["page_id"].astype(str), df["category"].astype(str)))
    ok, failures = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fetch, row, args.output, args.timeout) for row in rows]
        for idx, future in enumerate(as_completed(futures), 1):
            page_id, success, detail = future.result()
            if success:
                ok += 1
            else:
                failures.append((page_id, detail))
            if idx % 100 == 0 or idx == len(rows):
                print(f"{idx}/{len(rows)} ok={ok} failed={len(failures)}", flush=True)
    if failures:
        failure_path = args.output / "download_failures.tsv"
        failure_path.write_text("page_id\terror\n" + "\n".join(f"{x}\t{y}" for x, y in failures), encoding="utf-8")
        raise SystemExit(f"{len(failures)} downloads failed; see {failure_path}")
    print(f"downloaded and verified {ok} nonconstant-alpha RGBA images")


# ---------------------------------------------------------------------------
# prepare (composite RGBA -> packed WSG npz)
# ---------------------------------------------------------------------------
def _seed_for(page_id: str) -> int:
    return int(hashlib.md5(page_id.encode("utf-8")).hexdigest()[:8], 16)


def _real_background(size: int, rng: np.random.Generator, pool: np.ndarray) -> np.ndarray:
    """Pick a real-scene background from the pool (N,H,W,3 uint8), with light aug."""
    bg = pool[int(rng.integers(0, pool.shape[0]))].astype(np.float32)
    if bg.shape[0] != size or bg.shape[1] != size:
        bg = np.asarray(
            Image.fromarray(bg.astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    if rng.random() < 0.5:  # horizontal flip
        bg = bg[:, ::-1, :].copy()
    return bg


def _make_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return an HxWx3 float32 synthetic background in [0,255] (fallback only)."""
    kind = rng.choice(3, p=[0.45, 0.30, 0.25])
    if kind == 0:  # solid random color
        color = rng.integers(0, 256, 3).astype(np.float32)
        return np.broadcast_to(color, (size, size, 3)).astype(np.float32).copy()
    if kind == 1:  # linear gradient between two random colors
        c0 = rng.integers(0, 256, 3).astype(np.float32)
        c1 = rng.integers(0, 256, 3).astype(np.float32)
        t = np.linspace(0.0, 1.0, size, dtype=np.float32)
        t = t[:, None, None] if rng.random() < 0.5 else t[None, :, None]
        return (c0 * (1.0 - t) + c1 * t).astype(np.float32)
    # kind == 2: low-frequency texture (upsampled random tiles)
    h = int(rng.integers(4, 16)); w = int(rng.integers(4, 16))
    low = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    return np.asarray(
        Image.fromarray(low).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    bg_pool = None
    if args.composite and args.background_pool is not None:
        bg_pool = np.load(args.background_pool)
        print(f"loaded {bg_pool.shape[0]} real backgrounds from {args.background_pool}")
    rows = list(csv.DictReader(args.selection.open("r", encoding="utf-8"), delimiter="\t"))
    args.output.mkdir(parents=True, exist_ok=True)
    captions = []
    manifest = []
    for row in rows:
        page_id, category = row["page_id"], row["category"]
        source = args.image_root / category / f"{page_id}.png"
        if not source.exists():
            continue
        with Image.open(source) as image:
            rgba = image.convert("RGBA").resize((args.size, args.size), Image.Resampling.LANCZOS)
            data = np.asarray(rgba, dtype=np.float32)
        if args.composite:
            rng = np.random.default_rng(_seed_for(page_id))
            alpha = (data[..., 3:4] / 255.0).astype(np.float32)
            fg = data[..., :3]
            if bg_pool is not None and rng.random() >= args.synthetic_fraction:
                bg = _real_background(args.size, rng, bg_pool)
            else:
                bg = _make_background(args.size, rng)
            rgb = fg * alpha + bg * (1.0 - alpha)
            data = np.concatenate([rgb.astype(np.float32), data[..., 3:4]], axis=-1)
        out = args.output / category / f"{page_id}.wsg.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, data=data)
        captions.append({"name": page_id, "caption": row["subject"]})
        manifest.append({"sample_id": str(out), "page_id": page_id, "category": category, "caption": row["subject"]})
    with (args.output / "captions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "caption"])
        writer.writeheader(); writer.writerows(captions)
    (args.output / "manifest.json").write_text(
        json.dumps({"task": "magick_transparency", "channels": {"0-2": "RGB", "3": "soft alpha"}, "samples": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"prepared {len(manifest)} WSG samples in {args.output}")


# ---------------------------------------------------------------------------
# to-named (packed {"data": HxWx4} -> {"0-2", "LayerDiffuse_Transparency"})
# ---------------------------------------------------------------------------
# name -> (channel_start, channel_end_exclusive); alpha kept 1ch (dataset
# replicates to 3 for the FLUX VAE).
MAGICK_NAMED_MAP = {"0-2": (0, 3), "LayerDiffuse_Transparency": (3, 4)}


def cmd_to_named(args: argparse.Namespace) -> None:
    src, dst = args.src, args.dst
    npzs = sorted(src.glob("**/*.wsg.npz"))
    print(f"[magick] {len(npzs)} npz  {src} -> {dst}", flush=True)
    done = 0
    for path in npzs:
        with np.load(path) as z:
            data = z["data"] if "data" in z else z[z.files[0]]
        arrays = {name: data[..., start:end].astype(np.float32) for name, (start, end) in MAGICK_NAMED_MAP.items()}
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(npzs)}", flush=True)
    for meta in sorted(src.glob("**/captions.csv")) + sorted(src.glob("**/manifest.json")):
        out = dst / meta.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meta, out)
    print(f"[magick] DONE {done} npz + captions/manifest -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="Stratified subset selection from the MAGICK index TSV.")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--per-category", type=int, default=320)
    s.add_argument("--seed", type=int, default=20260918)
    s.set_defaults(func=cmd_select)

    d = sub.add_parser("download", help="Download + verify the selected RGBA images.")
    d.add_argument("--selection", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--timeout", type=int, default=60)
    d.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="Composite over random backgrounds -> packed WSG npz.")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=480)
    p.add_argument("--composite", dest="composite", action="store_true", default=True,
                   help="Composite foreground over a randomized background (default).")
    p.add_argument("--no-composite", dest="composite", action="store_false",
                   help="Keep raw straight-alpha RGB (old behavior, has black-bg shortcut).")
    p.add_argument("--background-pool", type=Path, default=None,
                   help="Optional .npy (N,H,W,3 uint8) of real-scene backgrounds.")
    p.add_argument("--synthetic-fraction", type=float, default=0.15,
                   help="Fraction of samples using a synthetic background even when a pool is given.")
    p.set_defaults(func=cmd_prepare)

    n = sub.add_parser("to-named", help="Rename packed npz into named-key npz (0-2 + LayerDiffuse_Transparency).")
    n.add_argument("--src", type=Path, default=Path("data/custom_pilots/magick_transparency_v1/wsg_composited"))
    n.add_argument("--dst", type=Path, default=Path("data/custom_pilots/magick_transparency_v1/wsg_named"))
    n.set_defaults(func=cmd_to_named)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
