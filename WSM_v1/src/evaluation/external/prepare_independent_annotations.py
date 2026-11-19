#!/usr/bin/env python3
"""Build an annotator-independent copy of the post-release evaluation suite.

The original post-release suite stores targets in WSG channel conventions.
This tool keeps the frozen images and sample identities, but produces labels
with public third-party annotators before applying the minimal, deterministic
format adapter required by each evaluated method:

* OpenCV Canny for edges;
* Depth Anything V2 for relative depth;
* Marigold Normals for surface normals; and
* Mask2Former-Large (ADE20K) for person segmentation.

The native predictions are retained next to the adapted inputs.  WSG is not
imported and does not participate in annotation.  This distinction matters:
the files under ``wsm_inputs`` are representations of independently produced
labels, not labels produced by the WSG preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEPTH_CHANNEL_RANGE = (-3.599609375, 1.8564453125)
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DEFAULT_NORMAL_MODEL = "prs-eth/marigold-normals-v1-1"
DEFAULT_SEG_MODEL = "facebook/mask2former-swin-large-ade-semantic"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_unit_interval(values: np.ndarray) -> np.ndarray:
    """Map finite values to [0, 1] with a fixed per-image robust protocol."""
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype=np.float32)
    if not finite.any():
        raise ValueError("Annotator returned no finite values")
    low, high = np.percentile(values[finite], (2.0, 98.0))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError("Annotator returned a non-finite percentile range")
    if high <= low:
        result[finite] = 0.5
    else:
        result[finite] = np.clip((values[finite] - low) / (high - low), 0.0, 1.0)
    return result


def inverse_wsg_depth_normalization(unit_depth: np.ndarray) -> np.ndarray:
    """Represent [0,1] depth in channel 32 without changing its information."""
    low, high = DEPTH_CHANNEL_RANGE
    scale = math.log1p(high - low)
    return (low + np.expm1(np.asarray(unit_depth, dtype=np.float32) * scale)).astype(
        np.float32
    )


def save_depth_products(root: Path, sample_id: str, depth: np.ndarray) -> None:
    from matplotlib import colormaps

    native = root / "annotations/depth_anything_v2/native" / f"{sample_id}.npy"
    normalized = root / "annotations/depth_anything_v2/relative_02_98" / f"{sample_id}.npy"
    visualization = root / "annotations/depth_anything_v2/visualization" / f"{sample_id}.png"
    wsm_input = root / "wsm_inputs/relative_depth32_33" / f"{sample_id}.npz"
    for path in (native, normalized, visualization, wsm_input):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Depth Anything V2 emits inverse depth (larger means nearer).  The WSG
    # channel and paper protocol use depth (larger means farther), so apply one
    # fixed convention conversion before any model sees the annotation.
    unit = 1.0 - robust_unit_interval(depth)
    raw_channel = inverse_wsg_depth_normalization(unit)
    channels = np.stack(
        [raw_channel.astype(np.float16), np.full(unit.shape, -1.0, dtype=np.float16)],
        axis=-1,
    )
    np.save(native, np.asarray(depth, dtype=np.float32), allow_pickle=False)
    np.save(normalized, unit, allow_pickle=False)
    np.savez_compressed(wsm_input, wsg_data=channels)
    rgb = (colormaps["Spectral"](unit)[..., :3] * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb).save(visualization)


def save_normal_products(root: Path, sample_id: str, normal: np.ndarray) -> None:
    normal = np.asarray(normal, dtype=np.float32)
    if normal.ndim == 4 and normal.shape[0] == 1:
        normal = normal[0]
    if normal.ndim != 3 or normal.shape[-1] != 3:
        raise ValueError(f"Unexpected normal shape for {sample_id}: {normal.shape}")
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(length, 1e-6)

    native = root / "annotations/marigold_normals/native_xyz" / f"{sample_id}.npy"
    visualization = root / "annotations/marigold_normals/visualization" / f"{sample_id}.png"
    wsm_input = root / "wsm_inputs/normal36_38" / f"{sample_id}.npz"
    for path in (native, visualization, wsm_input):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(native, normal, allow_pickle=False)
    np.savez_compressed(wsm_input, wsg_data=normal.astype(np.float16))
    rgb = (np.clip((normal + 1.0) * 0.5, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb).save(visualization)


def save_segmentation_products(
    root: Path,
    sample_id: str,
    labels: np.ndarray,
    foreground: np.ndarray,
) -> None:
    labels = np.asarray(labels, dtype=np.uint8)
    foreground = np.asarray(foreground, dtype=np.uint8)
    native = root / "annotations/mask2former_ade20k/native_labels" / f"{sample_id}.npy"
    binary = root / "targets/mask2former_person" / f"{sample_id}.npy"
    visualization = root / "annotations/mask2former_ade20k/visualization" / f"{sample_id}.png"
    for path in (native, binary, visualization):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(native, labels, allow_pickle=False)
    np.save(binary, foreground, allow_pickle=False)
    Image.fromarray(foreground * 255).save(visualization)


def save_edge_products(root: Path, sample_id: str, image_path: Path, pool: str) -> None:
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    edge = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 100, 200)
    visualization = root / "annotations/opencv_canny_100_200" / pool / f"{sample_id}.png"
    wsm_input = root / "wsm_inputs/edge40" / f"{sample_id}.npz"
    visualization.parent.mkdir(parents=True, exist_ok=True)
    wsm_input.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(edge).save(visualization)
    np.savez_compressed(wsm_input, wsg_data=edge.astype(np.float16)[..., None])


def general_samples(source_suite: Path) -> list[dict[str, Any]]:
    return read_jsonl(source_suite / "samples/rgb_to_relative_depth.jsonl")


def human_samples(source_suite: Path) -> list[dict[str, Any]]:
    revision = source_suite.parent / "frozen_suite_v1_human_revision1"
    manifest = revision / "samples/edge_to_human_seg.jsonl"
    if manifest.is_file():
        return read_jsonl(manifest)
    return read_jsonl(source_suite / "samples/edge_to_human_seg.jsonl")


def human_rgb_path(args: argparse.Namespace, sample_id: str) -> Path:
    revision = (
        args.source_suite.parent
        / "frozen_suite_v1_human_revision1"
        / "frames"
        / f"{sample_id}.png"
    )
    if revision.is_file():
        return revision
    return args.benchmark_root / "frames/frozen_v4/human" / f"{sample_id}.png"


def annotate_depth(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_source = str(args.depth_model)
    processor = AutoImageProcessor.from_pretrained(model_source, local_files_only=args.local_files_only)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if args.depth_device.startswith("cuda") else torch.float32,
    ).to(args.depth_device).eval()
    for index, row in enumerate(rows, 1):
        sample_id = str(row["source_sample_id"])
        destination = args.output_suite / "annotations/depth_anything_v2/native" / f"{sample_id}.npy"
        if destination.is_file() and not args.overwrite:
            print(f"[depth {index:02d}/{len(rows):02d}] reuse {sample_id}", flush=True)
            continue
        image = Image.open(row["input_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(args.depth_device, dtype=model.dtype if value.is_floating_point() else None)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
        prediction = functional.interpolate(
            prediction.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze().float().cpu().numpy()
        save_depth_products(args.output_suite, sample_id, prediction)
        print(f"[depth {index:02d}/{len(rows):02d}] {sample_id}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def annotate_normals(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    from diffusers import MarigoldNormalsPipeline

    pipe = MarigoldNormalsPipeline.from_pretrained(
        str(args.normal_model),
        local_files_only=args.local_files_only,
        variant="fp16" if args.normal_device.startswith("cuda") else None,
        torch_dtype=torch.float16 if args.normal_device.startswith("cuda") else torch.float32,
    ).to(args.normal_device)
    pipe.set_progress_bar_config(disable=True)
    for index, row in enumerate(rows, 1):
        sample_id = str(row["source_sample_id"])
        destination = args.output_suite / "annotations/marigold_normals/native_xyz" / f"{sample_id}.npy"
        if destination.is_file() and not args.overwrite:
            print(f"[normal {index:02d}/{len(rows):02d}] reuse {sample_id}", flush=True)
            continue
        image = Image.open(row["input_path"]).convert("RGB")
        generator = torch.Generator(device=args.normal_device).manual_seed(args.seed)
        result = pipe(
            image,
            num_inference_steps=args.normal_steps,
            ensemble_size=1,
            processing_resolution=args.normal_resolution,
            match_input_resolution=True,
            output_type="np",
            generator=generator,
        )
        save_normal_products(args.output_suite, sample_id, result.prediction)
        print(f"[normal {index:02d}/{len(rows):02d}] {sample_id}", flush=True)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def annotate_segmentation(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    model_source = str(args.seg_model)
    processor = AutoImageProcessor.from_pretrained(
        model_source, local_files_only=args.local_files_only
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if args.seg_device.startswith("cuda") else torch.float32,
    ).to(args.seg_device).eval()
    person_ids = [
        int(index)
        for index, label in model.config.id2label.items()
        if str(label).strip().lower() == "person"
        or str(label).strip().lower().startswith("person;")
    ]
    if not person_ids:
        raise RuntimeError(f"No person class in {model.config.id2label}")
    for index, row in enumerate(rows, 1):
        sample_id = str(row["source_sample_id"])
        destination = args.output_suite / "annotations/mask2former_ade20k/native_labels" / f"{sample_id}.npy"
        if destination.is_file() and not args.overwrite:
            print(f"[seg {index:02d}/{len(rows):02d}] reuse {sample_id}", flush=True)
            continue
        image_path = human_rgb_path(args, sample_id)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(
                args.seg_device,
                dtype=model.dtype if value.is_floating_point() else None,
            )
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        labels = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0].byte().cpu().numpy()
        person = np.isin(labels, person_ids).astype(np.uint8)
        save_segmentation_products(args.output_suite, sample_id, labels, person)
        print(f"[seg {index:02d}/{len(rows):02d}] {sample_id}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_edges(args: argparse.Namespace, general: list[dict[str, Any]], human: list[dict[str, Any]]) -> None:
    for pool, rows in (("general", general), ("human", human)):
        for index, row in enumerate(rows, 1):
            sample_id = str(row["source_sample_id"])
            if pool == "general":
                image_path = Path(row["input_path"])
            else:
                image_path = human_rgb_path(args, sample_id)
            save_edge_products(args.output_suite, sample_id, image_path, pool)
            print(f"[edge {pool} {index:02d}/{len(rows):02d}] {sample_id}", flush=True)


def rebuild_manifests(args: argparse.Namespace, general: list[dict[str, Any]], human: list[dict[str, Any]]) -> None:
    source_manifests = {
        name: {
            str(row["source_sample_id"]): row
            for row in read_jsonl(args.source_suite / "samples" / f"{name}.jsonl")
        }
        for name in (
            "rgb_to_relative_depth",
            "edge_to_rgb",
            "normal_to_edge",
            "relative_depth_to_normal",
            "edge_to_human_seg",
        )
    }
    source_manifests["edge_to_human_seg"] = {
        str(row["source_sample_id"]): row for row in human
    }
    tasks: dict[str, list[dict[str, Any]]] = {name: [] for name in source_manifests}
    tasks.update({
        "relative_depth_to_edge": [],
        "edge_to_relative_depth": [],
        "edge_to_normal": [],
    })
    for base in general:
        sample_id = str(base["source_sample_id"])
        edge = args.output_suite / "annotations/opencv_canny_100_200/general" / f"{sample_id}.png"
        edge_npz = args.output_suite / "wsm_inputs/edge40" / f"{sample_id}.npz"
        depth = args.output_suite / "annotations/depth_anything_v2/relative_02_98" / f"{sample_id}.npy"
        depth_vis = args.output_suite / "annotations/depth_anything_v2/visualization" / f"{sample_id}.png"
        depth_npz = args.output_suite / "wsm_inputs/relative_depth32_33" / f"{sample_id}.npz"
        normal = args.output_suite / "annotations/marigold_normals/native_xyz" / f"{sample_id}.npy"
        normal_vis = args.output_suite / "annotations/marigold_normals/visualization" / f"{sample_id}.png"
        normal_npz = args.output_suite / "wsm_inputs/normal36_38" / f"{sample_id}.npz"

        depth_row = dict(source_manifests["rgb_to_relative_depth"][sample_id])
        depth_row.update(
            target_path=str(depth.resolve()),
            target_depth_unit="DepthAnythingV2_relative_depth_percentile_02_98",
            annotation_method="Depth Anything V2 Large (native independent output)",
            annotation_family="independent_public_annotator",
        )
        tasks["rgb_to_relative_depth"].append(depth_row)

        rgb_row = dict(source_manifests["edge_to_rgb"][sample_id])
        # Edge->RGB already used an OpenCV Canny 100/200 annotation rather
        # than a WSG-derived label. Preserve that exact condition and its WSM
        # serialization so every reused prediction is paired with the pixels
        # it actually received. Re-running Canny after crop serialization can
        # otherwise move a small number of threshold-boundary pixels.
        rgb_row.update(
            edge_method="OpenCV Canny 100/200",
            annotation_family="independent_public_annotator",
        )
        tasks["edge_to_rgb"].append(rgb_row)

        edge_row = dict(source_manifests["normal_to_edge"][sample_id])
        edge_row.update(
            condition_path=str(normal_vis.resolve()),
            wsm_condition_path=str(normal_npz.resolve()),
            target_path=str(edge.resolve()),
            normal_condition_encoding="Marigold native camera-space XYZ",
            edge_method="OpenCV Canny 100/200",
            annotation_method="Marigold Normals v1.1 + OpenCV Canny",
            annotation_family="independent_public_annotator",
        )
        tasks["normal_to_edge"].append(edge_row)

        normal_row = dict(source_manifests["relative_depth_to_normal"][sample_id])
        normal_row.update(
            condition_path=str(depth_vis.resolve()),
            wsm_condition_path=str(depth_npz.resolve()),
            target_path=str(normal.resolve()),
            target_normal_encoding="physical_xyz",
            depth_condition_encoding="DepthAnythingV2_relative_02_98",
            annotation_method="Depth Anything V2 Large + Marigold Normals v1.1",
            annotation_family="independent_public_annotator",
        )
        tasks["relative_depth_to_normal"].append(normal_row)

        source_id = str(base["sample_id"]).removesuffix("_rgb_depth")
        shared = dict(base)
        shared.update(task="any_to_any", annotation_family="independent_public_annotator")

        depth_edge_row = dict(shared)
        depth_edge_row.update(
            sample_id=f"{source_id}_depth_edge",
            source_modality="depth", condition_modality="depth", target_modality="edge",
            target_type="edge", condition_path=str(depth_vis.resolve()),
            wsm_condition_path=str(depth_npz.resolve()), target_path=str(edge.resolve()),
            wsm_head_index=121, wsm_condition_channel="32-33", wsm_target_channel="40",
            annotation_method="Depth Anything V2 Large + OpenCV Canny",
        )
        tasks["relative_depth_to_edge"].append(depth_edge_row)

        edge_depth_row = dict(shared)
        edge_depth_row.update(
            sample_id=f"{source_id}_edge_depth",
            source_modality="edge", condition_modality="edge", target_modality="depth",
            target_type="depth", condition_path=str(edge.resolve()),
            wsm_condition_path=str(edge_npz.resolve()), target_path=str(depth.resolve()),
            target_depth_unit="DepthAnythingV2_relative_depth_percentile_02_98",
            wsm_head_index=426, wsm_condition_channel="40", wsm_target_channel="32-33",
            annotation_method="OpenCV Canny + Depth Anything V2 Large",
        )
        tasks["edge_to_relative_depth"].append(edge_depth_row)

        edge_normal_row = dict(shared)
        edge_normal_row.update(
            sample_id=f"{source_id}_edge_normal",
            source_modality="edge", condition_modality="edge", target_modality="normal",
            target_type="normal", condition_path=str(edge.resolve()),
            wsm_condition_path=str(edge_npz.resolve()), target_path=str(normal.resolve()),
            target_normal_encoding="physical_xyz",
            wsm_head_index=428, wsm_condition_channel="40", wsm_target_channel="36-38",
            annotation_method="OpenCV Canny + Marigold Normals v1.1",
        )
        tasks["edge_to_normal"].append(edge_normal_row)

    for base in human:
        sample_id = str(base["source_sample_id"])
        edge = args.output_suite / "annotations/opencv_canny_100_200/human" / f"{sample_id}.png"
        edge_npz = args.output_suite / "wsm_inputs/edge40" / f"{sample_id}.npz"
        target = args.output_suite / "targets/mask2former_person" / f"{sample_id}.npy"
        row = dict(source_manifests["edge_to_human_seg"][sample_id])
        row.update(
            condition_path=str(edge.resolve()),
            wsm_condition_path=str(edge_npz.resolve()),
            target_path=str(target.resolve()),
            valid_label_ids=[0, 1],
            human_foreground_definition="Mask2Former ADE20K person class versus background",
            annotation_method="Mask2Former-Swin-Large ADE20K semantic segmentation",
            annotation_family="independent_public_annotator",
        )
        tasks["edge_to_human_seg"].append(row)

    core_names = {
        "rgb_to_relative_depth", "edge_to_rgb", "normal_to_edge",
        "relative_depth_to_normal", "edge_to_human_seg",
    }
    all_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []
    for name, rows in tasks.items():
        write_jsonl(args.output_suite / "samples" / f"{name}.jsonl", rows)
        all_rows.extend(rows)
        if name in core_names:
            core_rows.extend(rows)
    write_jsonl(args.output_suite / "samples/all_five_tasks.jsonl", core_rows)
    write_jsonl(args.output_suite / "samples/all_eight_tasks.jsonl", all_rows)
    write_jsonl(
        args.output_suite / "samples/interface_c_extended_tasks.jsonl",
        tasks["relative_depth_to_edge"]
        + tasks["edge_to_relative_depth"]
        + tasks["edge_to_normal"],
    )
    write_jsonl(
        args.output_suite / "samples/modus_independent_annotation_tasks.jsonl",
        tasks["normal_to_edge"] + tasks["relative_depth_to_normal"],
    )

    audit = {
        "schema_version": 1,
        "suite": "post-release visual data with independent public annotators",
        "source_suite": str(args.source_suite.resolve()),
        "task_counts": {name: len(rows) for name, rows in tasks.items()},
        "annotation_protocol": {
            "edge": {"implementation": "OpenCV", "operator": "Canny", "thresholds": [100, 200]},
            "relative_depth": {
                "model": DEFAULT_DEPTH_MODEL,
                "normalization": "per-image 2nd-98th percentile linear map followed by fixed inverse-depth to depth conversion (1-x)",
            },
            "surface_normal": {
                "model": DEFAULT_NORMAL_MODEL,
                "representation": "native camera-space unit XYZ",
            },
            "human_segmentation": {
                "model": DEFAULT_SEG_MODEL,
                "scoring_target": "ADE20K person class versus background",
            },
        },
        "format_adaptation": {
            "principle": "deterministic representation conversion only; no WSG annotator is invoked",
            "wsm_depth": "inverse of the published channel-32 log normalization",
            "wsm_normal": "native XYZ values stored in channels 36-38",
            "wsm_edge": "native uint8 Canny values stored in channel 40",
        },
    }
    audit_path = args.output_suite / "independent_annotation_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[3]
    benchmark = project / "src/evaluation_output/external/wsm_postrelease_video_20260905"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=benchmark)
    parser.add_argument("--source-suite", type=Path, default=benchmark / "frozen_suite_v1")
    parser.add_argument("--output-suite", type=Path, default=benchmark / "frozen_suite_v2_independent")
    parser.add_argument("--depth-model", type=Path, default=Path(DEFAULT_DEPTH_MODEL))
    parser.add_argument("--normal-model", type=Path, default=Path(DEFAULT_NORMAL_MODEL))
    parser.add_argument("--seg-model", type=Path, default=Path(DEFAULT_SEG_MODEL))
    parser.add_argument("--depth-device", default="cuda:0")
    parser.add_argument("--normal-device", default="cuda:1")
    parser.add_argument("--seg-device", default="cuda:2")
    parser.add_argument("--normal-steps", type=int, default=10)
    parser.add_argument("--normal-resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("edge", "depth", "normal", "seg", "manifests"),
        default=("edge", "depth", "normal", "seg", "manifests"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.source_suite = args.source_suite.expanduser().resolve()
    args.output_suite = args.output_suite.expanduser().resolve()
    if args.depth_model.exists():
        args.depth_model = args.depth_model.expanduser().resolve()
    if args.normal_model.exists():
        args.normal_model = args.normal_model.expanduser().resolve()
    if args.seg_model.exists():
        args.seg_model = args.seg_model.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    general = general_samples(args.source_suite)
    human = human_samples(args.source_suite)
    if "edge" in args.stages:
        build_edges(args, general, human)
    if "depth" in args.stages:
        annotate_depth(args, general)
    if "normal" in args.stages:
        annotate_normals(args, general)
    if "seg" in args.stages:
        annotate_segmentation(args, human)
    if "manifests" in args.stages:
        rebuild_manifests(args, general, human)


if __name__ == "__main__":
    main()
