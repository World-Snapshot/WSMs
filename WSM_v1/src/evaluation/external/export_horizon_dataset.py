#!/usr/bin/env python3
"""Export the paper's frozen visual data as a portable dataset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL = PROJECT_ROOT / "src/evaluation_output/external"

TASK_SLUGS = {
    "rgb_to_depth": "rgb-to-depth",
    "edge_to_rgb": "edge-reference-to-rgb",
    "normal_to_edge": "normal-to-edge",
    "edge_to_normal": "edge-to-normal",
    "relative_depth_to_normal": "relative-depth-to-normal",
    "edge_to_human_seg": "edge-to-human-segmentation",
}

TASK_CHANNELS = {
    "rgb-to-depth": ([0, 1, 2], [34]),
    "edge-reference-to-rgb": ([40], [0, 1, 2]),
    "normal-to-edge": ([36, 37, 38], [40]),
    "edge-to-normal": ([40], [36, 37, 38]),
    "relative-depth-to-normal": ([34], [36, 37, 38]),
    "edge-to-human-segmentation": ([40], [103]),
}

PATH_ROLES = {
    "input_path": "input",
    "condition_path": "condition",
    "reference_path": "reference",
    "target_path": "target",
    "mask_path": "valid_mask",
    "wsm_condition_path": "wsg_condition",
    "original_input_path": "original_input",
    "original_target_path": "original_target",
    "original_mask_path": "original_valid_mask",
    "source_rgb_path": "source_rgb",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def copy_file(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.as_posix()


def sample_root_from(record: dict[str, Any]) -> Path | None:
    explicit = record.get("wsm_reader_sample_dir")
    if explicit:
        path = Path(str(explicit))
        if path.is_dir():
            return path
    for key in ("condition_path", "input_path", "target_path"):
        value = record.get(key)
        if not value:
            continue
        for parent in Path(str(value)).parents:
            if parent.name.startswith("sample_"):
                return parent
    return None


def add_raw_wsg_npz(
    record: dict[str, Any],
    sample_dir: Path,
    output_root: Path,
) -> None:
    sample_root = sample_root_from(record)
    if sample_root is None:
        return
    reader = sample_root / "wsg_reader" if (sample_root / "wsg_reader").is_dir() else sample_root
    for source_name, output_key in (
        ("condition", "wsg_raw_condition_paths"),
        ("gt", "wsg_raw_target_paths"),
    ):
        source_root = reader / "raw" / source_name
        files = sorted(source_root.rglob("*.npz")) if source_root.is_dir() else []
        relative_paths = []
        for index, source in enumerate(files):
            destination = sample_dir / f"{output_key}_{index:02d}{source.suffix.lower()}"
            copy_file(source, destination)
            relative_paths.append(destination.relative_to(output_root).as_posix())
        if relative_paths:
            record[output_key] = relative_paths


def portable_source_id(record: dict[str, Any]) -> str:
    source_id = str(record.get("source_sample_id") or "")
    if "::" in source_id:
        path_text, frame_text = source_id.split("::", 1)
        match = re.search(r"index_\d+", frame_text)
        source_name = Path(path_text).parent.parent.name
        if match:
            return f"{source_name}::{match.group(0)}"
    video_id = str(record.get("video_id") or source_id)
    frame = record.get("frame_index")
    if frame is not None:
        return f"{Path(video_id).name}::frame_{int(frame):08d}"
    return Path(video_id).name


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def channel_indices(token: str | int | None) -> list[int]:
    """Expand a WSG channel token such as 36-38 or 40."""
    if token is None:
        return []
    values: list[int] = []
    for part in str(token).split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"unsupported WSG channel token: {token!r}")
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        values.extend(range(start, stop + 1))
    return values


def record_channels(record: dict[str, Any]) -> tuple[list[int], list[int]]:
    condition = channel_indices(
        record.get("wsm_condition_channel") or record.get("wsm_input_channel")
    )
    target = channel_indices(record.get("wsm_target_channel"))
    defaults = TASK_CHANNELS.get(str(record.get("paper_task")))
    if defaults:
        condition = condition or list(defaults[0])
        target = target or list(defaults[1])
    return condition, target


def wsv_location(source_id: str) -> tuple[Path, str]:
    if "::" not in source_id:
        raise ValueError(f"AVIS source has no WSV frame key: {source_id!r}")
    path_text, frame_text = source_id.split("::", 1)
    match = re.search(r"index_\d+", frame_text)
    if match is None:
        raise ValueError(f"AVIS source has an invalid WSV frame key: {source_id!r}")
    return Path(path_text), match.group(0)


def extract_wsv_job(job: tuple[str, str, str, list[int]]) -> tuple[str, str]:
    """Extract one modality-scoped AVIS frame in a worker process."""
    source_id, destination_text, frame_key, channels = job
    import numpy as np

    destination = Path(destination_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wsv_path, _ = wsv_location(source_id)
    with np.load(wsv_path, allow_pickle=False) as archive:
        frame = archive[frame_key]
        selected = np.ascontiguousarray(frame[..., channels])
    np.savez_compressed(
        destination,
        wsg_data=selected,
        channels=np.asarray(channels, dtype=np.int16),
        frame_key=np.asarray(frame_key),
    )
    return source_id, destination.as_posix()


def export_avis_source_wsg(
    avis_rows: dict[str, list[dict[str, Any]]],
    output_root: Path,
    workers: int,
) -> None:
    """Export each selected AVIS frame once with only task-relevant channels."""
    source_channels: dict[str, set[int]] = {}
    source_labels: dict[str, str] = {}
    for rows in avis_rows.values():
        for record in rows:
            source_id = str(record["_source_wsv_id"])
            condition, target = record_channels(record)
            source_channels.setdefault(source_id, set()).update(condition + target)
            source_labels.setdefault(source_id, str(record["source_sample_id"]))

    relative_paths: dict[str, str] = {}
    jobs: list[tuple[str, str, str, list[int]]] = []
    for source_id in sorted(source_channels):
        _, frame_key = wsv_location(source_id)
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
        destination = output_root / "data/avis-heldout/source-wsg" / f"{digest}.npz"
        relative_paths[source_id] = destination.relative_to(output_root).as_posix()
        jobs.append(
            (source_id, destination.as_posix(), frame_key, sorted(source_channels[source_id]))
        )

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(extract_wsv_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            source_id, _ = future.result()
            print(f"AVIS WSG {completed}/{len(futures)}: {source_labels[source_id]}")

    for task_slug, rows in avis_rows.items():
        for record in rows:
            source_id = str(record["_source_wsv_id"])
            condition, target = record_channels(record)
            record["source_wsg_path"] = relative_paths[source_id]
            record["source_wsg_condition_channels"] = condition
            record["source_wsg_target_channels"] = target
        write_jsonl(
            output_root / "manifests/avis-heldout" / f"{task_slug}.jsonl",
            (public_record(record) for record in rows),
        )


def export_rows(
    collection: str,
    task_key: str,
    rows: list[dict[str, Any]],
    output_root: Path,
    include_raw_wsg: bool,
) -> list[dict[str, Any]]:
    task_slug = TASK_SLUGS[task_key]
    output_rows = []
    seen = set()
    for source_record in rows:
        record = dict(source_record)
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in {collection}/{task_slug}: {sample_id}")
        seen.add(sample_id)
        sample_dir = output_root / "data" / collection / task_slug / sample_id
        for key, role in PATH_ROLES.items():
            value = record.get(key)
            if not value:
                continue
            source = Path(str(value))
            destination = sample_dir / f"{role}{source.suffix.lower()}"
            copy_file(source, destination)
            record[key] = destination.relative_to(output_root).as_posix()
        if include_raw_wsg:
            add_raw_wsg_npz(record, sample_dir, output_root)
        record.pop("wsm_reader_sample_dir", None)
        source_id = str(record.get("source_sample_id") or "")
        if Path(source_id.split("::", 1)[0]).is_absolute():
            record["_source_wsv_id"] = source_id
            record["source_sample_id"] = portable_source_id(record)
        record["collection"] = collection
        record["paper_task"] = task_slug
        output_rows.append(record)
    manifest = output_root / "manifests" / collection / f"{task_slug}.jsonl"
    write_jsonl(manifest, (public_record(record) for record in output_rows))
    return output_rows


def avis_specs(root: Path) -> dict[str, tuple[Path, str | None]]:
    return {
        "rgb_to_depth": (
            root / "crop_aligned_depth_20260901/samples.jsonl", None
        ),
        "edge_to_rgb": (root / "samples.jsonl", "conditional_rgb"),
        "normal_to_edge": (
            root / "modus_official_full24_20260901/samples/normal_to_edge.jsonl",
            None,
        ),
        "relative_depth_to_normal": (
            root
            / "modus_official_full24_20260901/samples/"
            "relative_depth_to_normal_physical_xyz.jsonl",
            None,
        ),
        "edge_to_human_seg": (
            root
            / "modus_edge_human103_full24_20260904/samples/"
            "edge_to_human_seg.jsonl",
            None,
        ),
    }


def postrelease_specs(root: Path) -> dict[str, tuple[Path, str | None]]:
    frozen = root / "frozen_suite_v1/samples"
    return {
        "rgb_to_depth": (frozen / "rgb_to_relative_depth.jsonl", None),
        "edge_to_rgb": (frozen / "edge_to_rgb.jsonl", None),
        "normal_to_edge": (frozen / "normal_to_edge.jsonl", None),
        "relative_depth_to_normal": (
            frozen / "relative_depth_to_normal.jsonl", None
        ),
        "edge_to_human_seg": (
            root
            / "frozen_suite_v1_human_revision1/samples/"
            "edge_to_human_seg.jsonl",
            None,
        ),
    }


def independent_postrelease_specs(
    root: Path,
) -> dict[str, tuple[Path, str | None]]:
    frozen = root / "frozen_suite_v2_independent/samples"
    return {
        "rgb_to_depth": (frozen / "rgb_to_relative_depth.jsonl", None),
        "edge_to_rgb": (frozen / "edge_to_rgb.jsonl", None),
        "edge_to_normal": (frozen / "edge_to_normal.jsonl", None),
        "relative_depth_to_normal": (
            frozen / "relative_depth_to_normal.jsonl", None
        ),
    }


def attach_source_rgb(
    rows: list[dict[str, Any]],
    source_rgb_by_id: dict[str, str],
) -> None:
    """Attach the exact evaluated RGB frame when it is not already an input."""
    for row in rows:
        if row.get("input_path"):
            continue
        source_id = str(row.get("source_sample_id") or "")
        source_rgb = source_rgb_by_id.get(source_id)
        if source_rgb is None:
            raise KeyError(f"missing source RGB frame for {source_id!r}")
        row["source_rgb_path"] = source_rgb


def load_spec(path: Path, task_filter: str | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if task_filter:
        rows = [row for row in rows if row.get("task") == task_filter]
    return rows


def write_sources(
    postrelease_rows: Iterable[dict[str, Any]],
    provenance_path: Path,
    output_root: Path,
) -> None:
    fields = (
        "source_sample_id", "source_title", "source_page", "source_video_url",
        "source_description", "artist", "license", "license_url", "capture_date",
        "upload_timestamp",
    )
    catalog = {}
    if provenance_path.is_file():
        catalog = {
            str(row.get("title")): row
            for row in read_jsonl(provenance_path)
            if row.get("title")
        }
    sources = {}
    for row in postrelease_rows:
        source_id = str(row.get("source_sample_id") or "")
        if source_id not in sources:
            source = {
                key: row.get(key) for key in fields if row.get(key) is not None
            }
            original = catalog.get(str(row.get("source_title")), {})
            aliases = {
                "source_title": "title",
                "source_page": "descriptionurl",
                "source_video_url": "url",
                "source_description": "description",
            }
            for key in fields:
                original_key = aliases.get(key, key)
                if source.get(key) is None and original.get(original_key) is not None:
                    source[key] = original[original_key]
            sources[source_id] = source
    write_jsonl(
        output_root / "provenance/post-release-2026-sources.jsonl",
        (sources[key] for key in sorted(sources)),
    )


def write_dataset_card(output_root: Path) -> None:
    card = """---
license: other
task_categories:
- image-to-image
- image-segmentation
tags:
- world-snapshot
- multimodal
- depth-estimation
- surface-normals
- edge-detection
- human-parsing
- zero-shot-evaluation
pretty_name: WorldSnapshot Horizon
---

# WorldSnapshot Horizon

WorldSnapshot Horizon is a compact, self-contained release of the visual
inputs and annotations used for the external comparisons in the WSM paper. It
contains three deliberately separate collections:

- **avis-heldout** reproduces the held-out AVIS examples used in the paper.
  These examples were not used to train WSM. They are duplicated here so that
  the released comparison suite does not depend on private filesystem paths or
  a second dataset download.
- **post-release-2026** contains frames from Wikimedia Commons videos captured
  or published between 2026-08-02 and 2026-09-04. They were not used in WSM
  training and were selected by date to reduce overlap with the pretraining
  data of large foundation models.
- **post-release-2026-independent-annotations** reuses the same exact RGB frames
  and replaces geometric labels with OpenCV Canny, Depth Anything V2, and
  Marigold Normals outputs used in Section 5.4.

The release covers five mappings from the original comparison: RGB to relative depth,
edge plus reference RGB to RGB, normal to edge, relative depth to normal, and
edge to human-part segmentation. The independent-annotation collection adds
edge to normal and the corresponding public-annotator versions of the reported
mappings. It contains exact evaluated RGB frames, inputs, references, targets,
validity masks, task manifests, provenance, and modality-scoped WSG arrays.
Model predictions are intentionally excluded. Full source videos are not
duplicated; each record retains the original Wikimedia URL and item-level
license needed to retrieve them.

## Contents

| Collection | RGB to depth | Edge + reference to RGB | Normal to edge | Edge to normal | Relative depth to normal | Edge to human parsing |
|---|---:|---:|---:|---:|---:|---:|
| AVIS held-out | 23 | 24 | 24 | -- | 24 | 24 |
| Post-release 2026 | 24 | 24 | 24 | -- | 24 | 24 |
| Post-release 2026, independent annotations | 24 | 24 | -- | 24 | 24 | -- |

Repeated task records may refer to the same source frame. Shared AVIS source
arrays are stored once under data/avis-heldout/source-wsg.

## Sample triplets

| Collection | Edge condition | RGB reference | RGB target |
|---|---|---|---|
| AVIS held-out | ![](./data/avis-heldout/edge-reference-to-rgb/avis_edge_rgb_000/condition.png) | ![](./data/avis-heldout/edge-reference-to-rgb/avis_edge_rgb_000/reference.png) | ![](./data/avis-heldout/edge-reference-to-rgb/avis_edge_rgb_000/target.png) |
| Post-release 2026 | ![](./data/post-release-2026/edge-reference-to-rgb/recent_000_edge_rgb/condition.png) | ![](./data/post-release-2026/edge-reference-to-rgb/recent_000_edge_rgb/reference.png) | ![](./data/post-release-2026/edge-reference-to-rgb/recent_000_edge_rgb/target.png) |

## Layout

~~~text
data/
  avis-heldout/
    source-wsg/
    <task>/<sample_id>/
  post-release-2026/
    <task>/<sample_id>/
  post-release-2026-independent-annotations/
    <task>/<sample_id>/
manifests/
  avis-heldout/<task>.jsonl
  post-release-2026/<task>.jsonl
  post-release-2026-independent-annotations/<task>.jsonl
provenance/
  post-release-2026-sources.jsonl
dataset_info.json
checksums.sha256
~~~

Each manifest is the authoritative index. Its path fields are relative to the
repository root. AVIS source NPZ files contain:

- wsg_data: H x W x C float16 values copied from the encoded WSV frame;
- channels: the corresponding WSG channel indices;
- frame_key: the original frame key.

Post-release wsg_condition files are already modality-scoped NPZ inputs made by
the same WSG annotation pipeline. The manifest records the channels and the
task-specific crop or normalization contract where applicable.

## Provenance and licenses

Every post-release source retains its Wikimedia Commons page, author,
capture date, upload timestamp, and item-level license in the provenance
manifest. Those media remain under their individual source licenses and
attribution requirements. AVIS material follows the terms of its originating
release. WSG-derived annotations and arrays are provided for research and
reproducibility; this repository does not apply one blanket license to all
underlying media.

## Reproducibility

The task manifests preserve the exact samples and modality routes used by the
paper comparison. File integrity can be checked with:

~~~bash
sha256sum -c checksums.sha256
~~~

The associated projects are
[WSMs](https://github.com/World-Snapshot/WSMs) and
[World Snapshot Graphics](https://github.com/World-Snapshot/wsg).
"""
    (output_root / "README.md").write_text(card, encoding="utf-8")


def write_verify_script(output_root: Path) -> None:
    script = """#!/usr/bin/env python3
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
failures = []
for line in (root / "checksums.sha256").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if digest != expected:
        failures.append(relative)
if failures:
    raise SystemExit("checksum mismatch: " + ", ".join(failures))
print("All checksums match.")
"""
    path = output_root / "verify.py"
    path.write_text(script, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_root: Path) -> None:
    checksum_path = output_root / "checksums.sha256"
    files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            relative = path.relative_to(output_root).as_posix()
            handle.write(f"{file_sha256(path)}  {relative}\n")


def contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith(("/", "/research/", "/common/users/", "/home/"))
    if isinstance(value, list):
        return any(contains_absolute_path(item) for item in value)
    if isinstance(value, dict):
        return any(contains_absolute_path(item) for item in value.values())
    return False


def validate_manifests(output_root: Path) -> None:
    for path in sorted((output_root / "manifests").rglob("*.jsonl")):
        for record in read_jsonl(path):
            if contains_absolute_path(record):
                raise ValueError(f"absolute path leaked into {path}: {record['sample_id']}")
            for key, value in record.items():
                if not key.endswith("_path"):
                    continue
                candidate = output_root / str(value)
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--avis-root",
        type=Path,
        default=DEFAULT_EXTERNAL / "wsm_external_heldout24",
    )
    parser.add_argument(
        "--postrelease-root",
        type=Path,
        default=DEFAULT_EXTERNAL / "wsm_postrelease_video_20260905",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wsv-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    postrelease_flat = []
    postrelease_root = args.postrelease_root.resolve()
    source_rgb_by_id = {
        str(row["source_sample_id"]): str(row["input_path"])
        for row in read_jsonl(
            postrelease_root
            / "frozen_suite_v2_independent/samples/rgb_to_relative_depth.jsonl"
        )
    }
    collections = (
        ("avis-heldout", avis_specs(args.avis_root.resolve()), True, False),
        (
            "post-release-2026",
            postrelease_specs(postrelease_root),
            False,
            False,
        ),
        (
            "post-release-2026-independent-annotations",
            independent_postrelease_specs(postrelease_root),
            False,
            True,
        ),
    )
    for collection, specs, include_raw_wsg, include_source_rgb in collections:
        all_rows[collection] = {}
        for task_key, (manifest, task_filter) in specs.items():
            source_rows = load_spec(manifest, task_filter)
            if include_source_rgb:
                attach_source_rgb(source_rows, source_rgb_by_id)
            exported = export_rows(
                collection,
                task_key,
                source_rows,
                output_root,
                include_raw_wsg=include_raw_wsg,
            )
            all_rows[collection][TASK_SLUGS[task_key]] = exported
            if collection.startswith("post-release-2026"):
                postrelease_flat.extend(exported)

    export_avis_source_wsg(
        all_rows["avis-heldout"], output_root, workers=args.wsv_workers
    )
    write_sources(
        postrelease_flat,
        postrelease_root
        / "provenance/commons_recent_captured_20260801_plus_enriched.jsonl",
        output_root,
    )
    write_dataset_card(output_root)
    write_verify_script(output_root)
    summary = {
        "format": "worldsnapshot-horizon-v2",
        "release_date": "2026-09-17",
        "purpose": (
            "Training-disjoint visual inputs, WSG annotations, and modality-scoped "
            "arrays used by the WSM paper comparisons."
        ),
        "collections": {
            collection: {
                task: len(rows) for task, rows in tasks.items()
            }
            for collection, tasks in all_rows.items()
        },
        "total_task_records": sum(
            len(rows)
            for tasks in all_rows.values()
            for rows in tasks.values()
        ),
        "predictions_included": False,
    }
    (output_root / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_manifests(output_root)
    write_checksums(output_root)
    counts = Counter(
        path.suffix.lower() or "<none>"
        for path in output_root.rglob("*")
        if path.is_file()
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("file extensions:", dict(sorted(counts.items())))
    print("output:", output_root)


if __name__ == "__main__":
    main()
