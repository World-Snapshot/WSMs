#!/usr/bin/env python3
"""Run the official MODUS CLI by modality pair and export canonical predictions."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


ALIASES = {
    "rgb": "rgb", "image": "rgb",
    "depth": "depth",
    "normal": "normal", "normals": "normal", "surface_normal": "normal",
    "edge": "canny", "canny": "canny",
    "seg": "seg", "segmentation": "seg", "semantic_segmentation": "seg",
}

# ``any2any_dataset.get_mask_image`` trains ``seg`` as black background plus
# one of these colorblind-friendly instance colors. Decode generated masks by
# the released target convention itself instead of tuning a threshold on GT.
SEG_FOREGROUND_PALETTE = (
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def modality(value: Any, label: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if key not in ALIASES:
        raise ValueError(
            f"Unsupported MODUS {label} modality {value!r}; supported={sorted(ALIASES)}"
        )
    return ALIASES[key]


def pair_for(sample: Mapping[str, Any]) -> tuple[str, str]:
    task = str(sample.get("task") or "")
    if task == "depth_estimation":
        return "rgb", "depth"
    if task != "any_to_any":
        raise ValueError(
            f"{sample.get('sample_id')}: MODUS adapter supports any_to_any and "
            "depth_estimation; fixed-class semantic mIoU is not compatible"
        )
    source = (
        sample.get("source_modality")
        or sample.get("condition_modality")
        or sample.get("condition_type")
    )
    target = sample.get("target_modality") or sample.get("target_type")
    return modality(source, "source"), modality(target, "target")


def input_for(sample: Mapping[str, Any], base: Path) -> Path:
    value = sample.get("condition_path") or sample.get("input_path")
    if not value:
        raise ValueError(f"{sample.get('sample_id')}: missing condition/input path")
    path = resolve(str(value), base)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def stage_modality_input(source_path: Path, staged_path: Path, source: str) -> None:
    """Encode one canonical WSM field in the image convention used by MODUS."""
    if source not in {"canny", "depth"}:
        # The released MODUS directory loader enumerates PNG inputs only.
        # Canonical manifests may point to JPEG normal/RGB visualizations, so
        # transcode the pixels instead of copying bytes under a new suffix.
        from PIL import Image

        with Image.open(source_path) as image:
            image.convert("RGB").save(staged_path)
        return
    import numpy as np
    from PIL import Image

    image = Image.open(source_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    if source == "canny":
        binary = (rgb.mean(axis=-1) > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary).convert("RGB").save(staged_path)
        return

    from matplotlib import colormaps
    spectral = np.asarray(
        colormaps["Spectral"](np.linspace(0.0, 1.0, 256))[:, :3],
        dtype=np.float32,
    )
    flat = rgb.reshape(-1, 3)
    decoded = np.empty(flat.shape[0], dtype=np.float32)
    lut_norm = np.sum(spectral * spectral, axis=1)[None, :]
    for start in range(0, flat.shape[0], 65536):
        chunk = flat[start:start + 65536]
        distances = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + lut_norm - 2.0 * chunk @ spectral.T
        )
        decoded[start:start + len(chunk)] = np.argmin(distances, axis=1) / 255.0
    scalar = decoded.reshape(rgb.shape[:2])
    encoded = np.clip(np.rint(scalar * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(encoded).convert("RGB").save(staged_path)


def canonicalize_segmentation(source_path: Path, destination: Path) -> None:
    """Convert MODUS' released RGB mask convention to canonical 0/1 IDs."""
    import numpy as np
    from PIL import Image

    pixels = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)
    candidates = np.asarray(((0, 0, 0), *SEG_FOREGROUND_PALETTE), dtype=np.float32)
    flat = pixels.reshape(-1, 3)
    labels = np.empty(flat.shape[0], dtype=np.uint8)
    for start in range(0, flat.shape[0], 65536):
        chunk = flat[start:start + 65536]
        distances = np.sum(
            np.square(chunk[:, None, :] - candidates[None, :, :]), axis=-1
        )
        labels[start:start + len(chunk)] = (np.argmin(distances, axis=1) != 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, labels.reshape(pixels.shape[:2]))


def checkpoint_from(root: Path) -> Path:
    candidates = [
        root / "EPFL-VILAB--MODUS",
        root / "modus",
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"MODUS checkpoint not found under {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--seg-category",
        default="person",
        help="Category prompt used by the official MODUS seg target.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    infer_py = repo / "infer.py"
    if not infer_py.is_file():
        raise SystemExit(f"Missing official MODUS infer.py: {infer_py}")
    sample_manifest = Path(args.samples).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    prediction_manifest = Path(args.prediction_manifest).expanduser().resolve()
    checkpoint = checkpoint_from(Path(args.model_root).expanduser().resolve())
    samples = [
        row for row in read_jsonl(sample_manifest)
        if str(row.get("task")) in {"any_to_any", "depth_estimation"}
    ]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"Invalid shard {args.shard_index}/{args.num_shards}; "
            "require num_shards >= 1 and 0 <= shard_index < num_shards"
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in samples:
        groups[pair_for(sample)].append(sample)
    predictions = []
    group_runtimes = {}
    for (source, target), group in sorted(groups.items()):
        # The released checkpoint and official Space use the semantic names
        # from instruction_16mod_stage2.yaml directly.
        cli_source = source
        cli_target = target
        group = [
            sample for index, sample in enumerate(group)
            if index % args.num_shards == args.shard_index
        ]
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("--max-samples must be positive")
            group = group[:args.max_samples]
        if not group:
            continue
        pair_root = output_root / f"{source}2{target}"
        input_dir = pair_root / "inputs"
        generated_dir = pair_root / "official_outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[dict[str, Any], str]] = []
        for index, sample in enumerate(group):
            sample_id = str(sample.get("sample_id") or "")
            stem = f"{index:04d}_{safe_name(sample_id)}"
            source_path = input_for(sample, sample_manifest.parent)
            staged_path = input_dir / f"rgb_{stem}.png"
            if args.overwrite or not staged_path.exists():
                stage_modality_input(source_path, staged_path, source)
            staged.append((sample, stem))
        command = [
            sys.executable, str(infer_py),
            "--condition", cli_source,
            "--target", cli_target,
            "--num_samples", "1",
            f"checkpoint_path={checkpoint}",
            f"input_image={input_dir}",
            f"output_dir={generated_dir}",
            f"seed={args.seed}",
            f"num_timesteps={args.steps}",
            f"image_size={args.image_size}",
            "use_instruction=true",
            "use_target_instruction=true",
            "use_condition_instruction=false",
        ]
        if target == "seg":
            command.append(f"seg_category={args.seg_category}")
        print("$ " + " ".join(str(item) for item in command), flush=True)
        elapsed = 0.0
        if args.overwrite or not generated_dir.is_dir():
            started = time.perf_counter()
            subprocess_env = os.environ.copy()
            if target == "seg":
                # The released top-level infer.py does not forward seg_category.
                # Its official inferencer reads this supported environment fallback.
                subprocess_env["HUNYUAN_SEG_CATEGORY"] = args.seg_category
            subprocess.run(command, cwd=repo, check=True, env=subprocess_env)
            elapsed = time.perf_counter() - started
        group_runtimes[f"{source}2{target}"] = {
            "samples": len(staged),
            "semantic_source": source,
            "semantic_target": target,
            "checkpoint_condition_token": cli_source,
            "checkpoint_target_token": cli_target,
            "token_route": "official_identity",
            "total_inference_seconds": elapsed,
            "inference_fps": len(staged) / elapsed if elapsed > 0 else None,
        }
        for sample, stem in staged:
            sample_dir = generated_dir / f"inputs__rgb_{stem}"
            # The official CLI writes a single input directly into
            # ``official_outputs`` but creates one named subdirectory per
            # input for directory batches. Support both layouts so smoke
            # tests and full benchmark runs share the same adapter.
            search_dir = sample_dir if sample_dir.is_dir() else generated_dir
            candidates = sorted(
                path for path in search_dir.glob(
                    f"{cli_source}2{cli_target}_*0.png"
                )
                if not any(
                    marker in path.name
                    for marker in ("_compare", "_input", "_bbox", "_intermediate")
                )
            )
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{sample.get('sample_id')}: expected exactly one MODUS output "
                    f"in {search_dir}, found {candidates}"
                )
            prediction_path = candidates[0]
            label_decode = None
            if target == "seg":
                prediction_path = pair_root / "canonical_predictions" / f"{stem}.npy"
                canonicalize_segmentation(candidates[0], prediction_path)
                label_decode = "nearest_modus_training_palette_vs_background"
            predictions.append({
                "sample_id": str(sample["sample_id"]),
                "prediction_path": str(prediction_path),
                "adapter": "modus_official_cli",
                "source_modality": source,
                "target_modality": target,
                "checkpoint_condition_token": cli_source,
                "checkpoint_target_token": cli_target,
                "token_route": "official_identity",
                "seed": args.seed,
                "steps": args.steps,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "inference_seconds": (elapsed / len(staged) if elapsed > 0 else 0.0),
                "scalar_encoding": "auto",
                "seg_category": args.seg_category if target == "seg" else None,
                "label_decode": label_decode,
            })
    atomic_jsonl(prediction_manifest, predictions)
    runtime_path = prediction_manifest.with_suffix(".runtime.json")
    runtime_path.write_text(
        json.dumps({"groups": group_runtimes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(predictions)} predictions: {prediction_manifest}")


if __name__ == "__main__":
    main()
