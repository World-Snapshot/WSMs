#!/usr/bin/env python3
"""Refresh saved WSG Reader artifacts after a normalization-contract correction.

The source tree is read-only. Refreshed normalized/denormalized WSG archives,
Reader PNGs, and a provenance manifest are written below a new output root.
This avoids repeating diffusion inference when decoded [0, 1] lanes can be
recovered from historical Reader archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.single_wsg_visualization import visualize_single_wsg


MASK_VALUE = -1.0
CATEGORICAL_MODALITIES = {"111", "195"}
SCOPED_READER_EXPORTS = {
    "normal_data": ("36-38", (36, 37, 38)),
    "edge_data": ("40", (40,)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help=(
            "Refresh a manifest-backed image field instead of WSG archives. "
            "This is intended for historical 8-bit Reader exports whose pixel "
            "values store the old normalized lane."
        ),
    )
    parser.add_argument("--manifest-task", default="conditional_rgb")
    parser.add_argument("--manifest-path-field", default="condition_path")
    parser.add_argument("--manifest-modality", default="40")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--legacy-config", type=Path, required=True)
    parser.add_argument(
        "--current-config",
        type=Path,
        default=PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--include-reader-exports",
        action="store_true",
        help=(
            "Also refresh historical modality-scoped Reader exports such as "
            "*_Normals_npz.npz and *_Edge_npz.npz."
        ),
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modality_channels(modality: str) -> list[int]:
    channels: list[int] = []
    for part in str(modality).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            channels.extend(range(int(start), int(end) + 1))
        else:
            channels.append(int(part))
    return channels


def normalization(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("normalization") or {}


def modality_mode(config: dict[str, Any], modality: str) -> str:
    contract = normalization(config)
    return str(
        (contract.get("modalities") or {}).get(
            str(modality), contract.get("default_mode", "linear")
        )
    ).lower()


def channel_range(
    config: dict[str, Any],
    channel: int,
    fallback: dict[str, Any] | None = None,
) -> tuple[float, float]:
    ranges = normalization(config).get("channel_ranges") or {}
    value = ranges.get(str(channel))
    if value is None and fallback is not None:
        value = (normalization(fallback).get("channel_ranges") or {}).get(
            str(channel)
        )
    if not isinstance(value, list) or len(value) != 2:
        raise KeyError(f"Missing portable range for channel {channel}")
    return float(value[0]), float(value[1])


def normalize_lane(
    values: np.ndarray,
    channel: int,
    mode: str,
    value_range: tuple[float, float],
) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(output) & (output != MASK_VALUE)
    if channel in (0, 1, 2):
        output[valid] = np.clip(output[valid] / 255.0, 0.0, 1.0)
        return output
    minimum, maximum = value_range
    span = max(maximum - minimum, 1e-6)
    shifted = np.clip(output[valid] - minimum, 0.0, None)
    if mode == "log":
        output[valid] = np.log1p(shifted) / max(float(np.log1p(span)), 1e-6)
    else:
        output[valid] = shifted / span
    output[valid] = np.clip(output[valid], 0.0, 1.0)
    return output


def denormalize_lane(
    values: np.ndarray,
    channel: int,
    mode: str,
    value_range: tuple[float, float],
) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(output) & (output != MASK_VALUE)
    if channel in (0, 1, 2):
        output[valid] = np.clip(output[valid], 0.0, 1.0) * 255.0
        return output
    minimum, maximum = value_range
    span = max(maximum - minimum, 1e-6)
    clipped = np.clip(output[valid], 0.0, 1.0)
    if mode == "log":
        output[valid] = np.expm1(
            clipped * max(float(np.log1p(span)), 1e-6)
        ) + minimum
    else:
        output[valid] = clipped * span + minimum
    return output


def load_archive(
    path: Path,
) -> tuple[np.ndarray, str, dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        if "wsg_data" in payload.files:
            data = np.asarray(payload["wsg_data"], dtype=np.float32)
            modality = str(payload["modality"].item())
            extras = {
                key: np.asarray(payload[key])
                for key in payload.files
                if key not in {"wsg_data", "modality", "renderer_contract"}
            }
            return data, modality, extras, "full_wsg"

        scoped_keys = [key for key in payload.files if key in SCOPED_READER_EXPORTS]
        if len(scoped_keys) != 1:
            raise KeyError(
                f"Unsupported Reader archive keys in {path}: {payload.files}"
            )
        key = scoped_keys[0]
        modality, channels = SCOPED_READER_EXPORTS[key]
        scoped = np.asarray(payload[key], dtype=np.float32)
        if scoped.ndim == 2:
            scoped = scoped[..., None]
        if scoped.ndim != 3 or scoped.shape[-1] != len(channels):
            raise ValueError(
                f"Unexpected shape {scoped.shape} for {key} in {path}"
            )
        data = np.full((*scoped.shape[:2], 256), MASK_VALUE, dtype=np.float32)
        for offset, channel in enumerate(channels):
            data[..., channel] = scoped[..., offset]
        return data, modality, {}, f"scoped_reader_export:{key}"


def artifact_paths(
    source: Path, source_root: Path, output_root: Path,
) -> tuple[Path, Path, Path, Path]:
    relative = source.relative_to(source_root)
    if not relative.name.endswith(".wsg.npz"):
        relative = relative.with_name(f"{relative.stem}.wsg.npz")
    destination = output_root / "reader_assets" / relative
    normalized_destination = destination.with_name(
        destination.name.replace(".wsg.npz", ".normalized.wsg.npz")
    )
    item_name = destination.name.removesuffix(".wsg.npz")
    raw_dir = destination.parent.parent / "raw" / item_name
    canonical = destination.parent.parent / f"{item_name}.png"
    return destination, normalized_destination, raw_dir, canonical


def source_normalized_sidecar(source: Path) -> Path | None:
    if not source.name.endswith(".wsg.npz"):
        return None
    return source.with_name(
        source.name.replace(".wsg.npz", ".normalized.wsg.npz")
    )


def primary_reader_png(root: Path, modality: str) -> Path:
    files = sorted(root.rglob("*.png"))
    if not files:
        raise RuntimeError(f"Reader produced no PNG under {root}")
    channel_tokens = tuple(
        f"channel{channel}" for channel in modality_channels(modality)
    )

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        value = 0
        if modality == "0-2" and "front_rgb" in name:
            value += 120
        if "semantic_mask" in name:
            value += 110
        if channel_tokens and any(token in name for token in channel_tokens):
            value += 80
        if name.endswith("_map.png"):
            value += 50
        if any(token in name for token in (
            "normal", "depth", "flow", "edge", "pose", "mask"
        )):
            value += 30
        if "overlay" in name:
            value -= 60
        if "summary" in name:
            value -= 100
        return value, str(path)

    return max(files, key=score)


def refresh_one(
    source: Path,
    source_root: Path,
    output_root: Path,
    legacy_config: dict[str, Any],
    current_config: dict[str, Any],
    render: bool,
) -> dict[str, Any]:
    source_data, modality, extras, source_format = load_archive(source)
    destination, normalized_destination, raw_dir, canonical_path = artifact_paths(
        source, source_root, output_root
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_source = source_normalized_sidecar(source)

    if normalized_source is not None and normalized_source.is_file():
        normalized, normalized_modality, _, _ = load_archive(normalized_source)
        if normalized_modality != modality:
            raise RuntimeError(
                f"Modality mismatch: {normalized_modality} != {modality}"
            )
        recovery = "saved_normalized_wsg"
    elif modality in CATEGORICAL_MODALITIES:
        normalized = source_data.copy()
        recovery = "categorical_identity"
    else:
        normalized = np.full_like(source_data, MASK_VALUE, dtype=np.float32)
        old_mode = modality_mode(legacy_config, modality)
        for channel in modality_channels(modality):
            normalized[..., channel] = normalize_lane(
                source_data[..., channel],
                channel,
                old_mode,
                channel_range(legacy_config, channel, current_config),
            )
        recovery = "recovered_from_legacy_denormalized_wsg"

    if modality in CATEGORICAL_MODALITIES:
        refreshed = normalized.copy()
    else:
        refreshed = np.full_like(normalized, MASK_VALUE, dtype=np.float32)
        new_mode = modality_mode(current_config, modality)
        for channel in modality_channels(modality):
            refreshed[..., channel] = denormalize_lane(
                normalized[..., channel],
                channel,
                new_mode,
                channel_range(current_config, channel),
            )

    np.savez_compressed(
        destination,
        wsg_data=refreshed,
        modality=np.asarray(modality),
        renderer_contract=np.asarray("denormalized_wsg_v2_normalization_contract"),
        **extras,
    )
    np.savez_compressed(
        normalized_destination,
        wsg_data=normalized,
        modality=np.asarray(modality),
        renderer_contract=np.asarray("normalized_wsg_v1"),
        **extras,
    )

    canonical = None
    reader_source = None
    if render:
        visualize_single_wsg(
            destination,
            raw_dir,
            declared_channels=modality_channels(modality),
        )
        selected = primary_reader_png(raw_dir, modality)
        canonical = canonical_path
        shutil.copy2(selected, canonical)
        reader_source = selected

    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_format": source_format,
        "modality": modality,
        "recovery": recovery,
        "legacy_mode": modality_mode(legacy_config, modality),
        "current_mode": modality_mode(current_config, modality),
        "normalized": str(normalized_destination),
        "denormalized": str(destination),
        "canonical_png": None if canonical is None else str(canonical),
        "reader_source": None if reader_source is None else str(reader_source),
    }


def refresh_manifest_images(
    args: argparse.Namespace,
    output_root: Path,
    current_config: dict[str, Any],
    current_path: Path,
) -> None:
    source_manifest = args.sample_manifest.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = [
        row for row in rows
        if str(row.get("task") or "") == args.manifest_task
    ]
    if not selected:
        raise ValueError(
            f"No task {args.manifest_task!r} in {source_manifest}"
        )
    channels = modality_channels(args.manifest_modality)
    if len(channels) != 1:
        raise ValueError("Manifest image refresh currently requires one channel")
    channel = channels[0]
    mode = modality_mode(current_config, args.manifest_modality)
    value_range = channel_range(current_config, channel)
    rewritten: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row.get("sample_id") or "")
        source_value = str(row.get(args.manifest_path_field) or "")
        if not sample_id or not source_value:
            raise ValueError(
                f"Missing sample_id/{args.manifest_path_field}: {row}"
            )
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = (source_manifest.parent / source).resolve()
        normalized = (
            np.asarray(Image.open(source).convert("L"), dtype=np.float32) / 255.0
        )
        physical = denormalize_lane(normalized, channel, mode, value_range)
        destination = output_root / "manifest_assets" / sample_id / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.clip(np.rint(physical), 0, 255).astype(np.uint8), mode="L"
        ).save(destination)
        tensor_path = destination.with_suffix(".normalized.npz")
        np.savez_compressed(
            tensor_path,
            normalized_data=normalized.astype(np.float32),
            modality=np.asarray(args.manifest_modality),
            renderer_contract=np.asarray("normalized_reader_png_v1"),
        )
        updated = dict(row)
        updated[args.manifest_path_field] = str(destination.resolve())
        updated["normalization_contract"] = {
            "modality": args.manifest_modality,
            "mode": mode,
            "recovery": "legacy_reader_png_normalized_lane",
            "source": str(source),
        }
        rewritten.append(updated)
        records.append({
            "sample_id": sample_id,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "normalized_tensor": str(tensor_path),
            "refreshed_image": str(destination),
            "refreshed_sha256": sha256_file(destination),
        })
    output_manifest = output_root / "samples.jsonl"
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rewritten),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output_manifest": str(output_manifest),
        "task": args.manifest_task,
        "path_field": args.manifest_path_field,
        "modality": args.manifest_modality,
        "mode": mode,
        "current_config": str(current_path),
        "current_config_sha256": sha256_file(current_path),
        "samples": len(rewritten),
        "records": records,
    }
    manifest_path = output_root / "refresh_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "samples": len(rewritten),
        "manifest": str(manifest_path),
        "sample_manifest": str(output_manifest),
    }, indent=2))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"{output_root} already exists; pass --resume to preserve it"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    legacy_path = args.legacy_config.expanduser().resolve()
    current_path = args.current_config.expanduser().resolve()
    legacy_config = json.loads(legacy_path.read_text(encoding="utf-8"))
    current_config = json.loads(current_path.read_text(encoding="utf-8"))
    if args.sample_manifest is not None:
        if args.source_root is not None:
            raise ValueError("Use either --source-root or --sample-manifest")
        refresh_manifest_images(args, output_root, current_config, current_path)
        return
    if args.source_root is None:
        raise ValueError("--source-root is required unless --sample-manifest is used")
    source_root = args.source_root.expanduser().resolve()
    candidates = sorted(
        path for path in source_root.glob("**/wsg_reader/tensors/*.wsg.npz")
        if ".normalized.wsg.npz" not in path.name
    )
    if args.include_reader_exports:
        candidates.extend(sorted(source_root.glob("**/*_npz.npz")))
        candidates = sorted(set(candidates))
    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]
    records: list[dict[str, Any]] = []
    reused = 0
    failures: list[dict[str, str]] = []
    for index, source in enumerate(candidates, 1):
        print(f"[{index}/{len(candidates)}] {source}", flush=True)
        destination, normalized_destination, raw_dir, canonical = artifact_paths(
            source, source_root, output_root
        )
        complete = (
            destination.is_file()
            and normalized_destination.is_file()
            and (args.no_render or canonical.is_file())
        )
        if args.resume and complete:
            _, modality, _, source_format = load_archive(source)
            records.append({
                "source": str(source),
                "source_sha256": sha256_file(source),
                "source_format": source_format,
                "modality": modality,
                "recovery": "reused_versioned_artifact",
                "legacy_mode": modality_mode(legacy_config, modality),
                "current_mode": modality_mode(current_config, modality),
                "normalized": str(normalized_destination),
                "denormalized": str(destination),
                "canonical_png": None if args.no_render else str(canonical),
                "reader_source": None,
            })
            reused += 1
            print("  REUSED", flush=True)
            continue
        try:
            records.append(
                refresh_one(
                    source,
                    source_root,
                    output_root,
                    legacy_config,
                    current_config,
                    not args.no_render,
                )
            )
        except Exception as exc:
            failures.append({"source": str(source), "error": repr(exc)})
            print(f"  ERROR: {exc}", flush=True)
    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "legacy_config": str(legacy_path),
        "legacy_config_sha256": sha256_file(legacy_path),
        "current_config": str(current_path),
        "current_config_sha256": sha256_file(current_path),
        "candidates": len(candidates),
        "complete": len(records),
        "reused": reused,
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    manifest_path = output_root / "refresh_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "candidates": len(candidates),
        "complete": len(records),
        "reused": reused,
        "failed": len(failures),
        "manifest": str(manifest_path),
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
