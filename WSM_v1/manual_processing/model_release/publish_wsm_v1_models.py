#!/usr/bin/env python3
"""Audit and publish the official WSM v1 Channel Head runs.

Source training directories are never modified. Text files are sanitized in
memory immediately before upload, while binary files are rejected if they
contain a known local path, user/host identifier, email, or credential.

Release selection is configuration-driven: dense runs come from the frozen
dense recommendation registry; sparse and custom runs are those carrying a
``recommended_checkpoint`` in their compact modality configuration. Reviewed
dense Heads marked as needing improvement remain included because they are the
documented deployment checkpoint for an otherwise unique task. Unselected
ablations, failed runs, and superseded runs are omitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import filecmp
import getpass
import hashlib
import io
import json
import os
import re
import socket
import struct
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


RELEASE_FORMAT = "wsm-v1-channel-head-release-v3"
# Keep every uncompressed continued-training TAR below the 15 GiB hosting
# boundary while leaving room for TAR headers and service-side bookkeeping.
ARCHIVE_TARGET_BYTES = 14 * 1024**3
DIRECT_UPLOAD_BATCH_FILES = 32
SANITIZER_VERSION = 1
DEFAULT_REPO_ID = "World-Snapshot/WSM_v1"
TEXT_SUFFIXES = {
    ".cfg", ".conf", ".csv", ".json", ".jsonl", ".log", ".md",
    ".sh", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
BINARY_SUFFIXES = {
    ".bin", ".bmp", ".ckpt", ".gif", ".jpeg", ".jpg", ".npy", ".npz",
    ".pdf", ".png", ".pt", ".pth", ".safetensors", ".tif", ".tiff",
    ".webp",
}

SKIP_NAMES = {"__pycache__", ".git", ".ipynb_checkpoints", "wandb"}
SKIP_PATTERNS = ("events.out.tfevents.", ".nfs")


@dataclasses.dataclass(frozen=True)
class SelectedRun:
    family: str
    head: str
    run_relative: str
    checkpoint_relative: str
    recommendation_status: str
    quality_status: str
    additional_relative: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ReleaseFile:
    source: Path
    path_in_repo: str
    size: int
    is_text: bool


@dataclasses.dataclass(frozen=True)
class SensitiveValue:
    label: str
    raw: bytes
    replacement: bytes


@dataclasses.dataclass(frozen=True)
class ArchivePart:
    index: int
    total: int
    items: tuple[ReleaseFile, ...]
    path_in_repo: str
    source_bytes: int
    fingerprint: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def dense_runs(project_root: Path) -> list[SelectedRun]:
    config = load_json(
        project_root / "configs" / "wsm_v1_recommended_dense_channel_heads.json"
    )
    selected: list[SelectedRun] = []
    for entry in config["entries"]:
        candidate = entry["recommended"]["candidate"]
        refiner = entry["recommended"].get("categorical_refiner_relative")
        selected.append(
            SelectedRun(
                family="train_log",
                head=f"Channel_Head_{int(entry['head_index']):06d}",
                run_relative=candidate["run_dir"],
                checkpoint_relative=candidate["checkpoint"],
                recommendation_status=entry["recommendation_status"],
                quality_status=(
                    "needs_improvement"
                    if entry_has_warning(entry)
                    else "recommended"
                ),
                additional_relative=(refiner,) if refiner else (),
            )
        )
    return selected


def entry_has_warning(entry: Mapping[str, Any]) -> bool:
    """Find the per-entry notice across historical registry schema variants."""

    queue: list[Any] = [entry]
    while queue:
        value = queue.pop()
        if isinstance(value, Mapping):
            if str(value.get("severity", "")).lower() == "warning":
                return True
            queue.extend(value.values())
        elif isinstance(value, list):
            queue.extend(value)
    return False


def compact_runs(project_root: Path, family: str, name: str) -> list[SelectedRun]:
    config = load_json(project_root / "configs" / name)
    selected: list[SelectedRun] = []
    for head, value in config["wsg_v1"].items():
        checkpoint = value.get("recommended_checkpoint")
        if not checkpoint:
            continue
        parts = PurePosixPath(checkpoint).parts
        if len(parts) < 4 or parts[0] != family or parts[1] != head:
            raise ValueError(f"Unexpected {family} checkpoint layout: {checkpoint}")
        selected.append(
            SelectedRun(
                family=family,
                head=head,
                run_relative=PurePosixPath(*parts[:3]).as_posix(),
                checkpoint_relative=checkpoint,
                recommendation_status="recommended_checkpoint",
                quality_status="recommended",
            )
        )
    return selected


def select_runs(project_root: Path) -> list[SelectedRun]:
    selected = dense_runs(project_root)
    selected += compact_runs(
        project_root, "train_log_sparse", "wsg_v1_modalities_sparse.json"
    )
    selected += compact_runs(
        project_root, "train_log_custom", "wsg_v1_modalities_custom.json"
    )
    selected.sort(key=lambda item: (item.family, item.head, item.run_relative))
    counts = Counter(item.run_relative for item in selected)
    duplicates = [path for path, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate selected runs: {duplicates[:5]}")
    return selected


def validate_runs(project_root: Path, runs: Sequence[SelectedRun]) -> None:
    errors: list[str] = []
    for run in runs:
        run_path = project_root / run.run_relative
        checkpoint = project_root / run.checkpoint_relative
        if not run_path.is_dir():
            errors.append(f"missing run: {run.run_relative}")
        if not checkpoint.is_file():
            errors.append(f"missing checkpoint: {run.checkpoint_relative}")
        try:
            checkpoint.relative_to(run_path)
        except ValueError:
            errors.append(f"checkpoint outside selected run: {run.checkpoint_relative}")
    if errors:
        raise FileNotFoundError("\n".join(errors[:50]))


def skip_reason(path: Path) -> str | None:
    if any(part in SKIP_NAMES for part in path.parts):
        return "local runtime/cache directory"
    if any(pattern in path.name for pattern in SKIP_PATTERNS):
        return "host-bearing or temporary runtime file"
    if path.is_symlink():
        return "symbolic link"
    return None


def looks_text(path: Path, size: int) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.suffix.lower() in BINARY_SUFFIXES:
        return False
    if size > 2 * 1024 * 1024:
        return False
    with path.open("rb") as handle:
        return b"\x00" not in handle.read(8192)


def iter_release_files(
    project_root: Path, runs: Sequence[SelectedRun]
) -> tuple[list[ReleaseFile], list[dict[str, str]]]:
    """Inventory selected runs while omitting non-release visualization data."""

    excluded_visualization_directories = {
        "pseudo-visualization",
        "pseudo-visualization-inference",
        "wsg-reader-visualization",
        "wsg-reader-visualization-inference",
        "visuals",
    }
    files: dict[str, ReleaseFile] = {}
    skipped: list[dict[str, str]] = []
    for run in runs:
        run_root = project_root / run.run_relative
        for base, directory_names, file_names in os.walk(run_root):
            base_path = Path(base)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                candidate = base_path / name
                relative_to_run = candidate.relative_to(run_root)
                is_sparse_step_preview = (
                    len(relative_to_run.parts) == 1
                    and re.fullmatch(r"step_\d+", name) is not None
                )
                if (
                    name in excluded_visualization_directories
                    or is_sparse_step_preview
                ):
                    skipped.append(
                        {
                            "path": candidate.relative_to(project_root).as_posix(),
                            "reason": "per-step visualization or preview directory",
                        }
                    )
                    continue
                reason = skip_reason(candidate)
                if reason:
                    skipped.append(
                        {
                            "path": candidate.relative_to(project_root).as_posix(),
                            "reason": reason,
                        }
                    )
                    continue
                kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                source = base_path / name
                remote = source.relative_to(project_root).as_posix()
                reason = skip_reason(source)
                is_step_prediction = (
                    base_path == run_root
                    and re.fullmatch(r"step_\d+_predictions\..+", name)
                    is not None
                )
                if reason or is_step_prediction:
                    skipped.append(
                        {
                            "path": remote,
                            "reason": reason or "per-step prediction preview",
                        }
                    )
                    continue
                stat = source.stat()
                files[remote] = ReleaseFile(
                    source, remote, stat.st_size, looks_text(source, stat.st_size)
                )
    for run in runs:
        latest_remote = f"{run.run_relative}/latest_trainer_state.pt"
        latest = files.get(latest_remote)
        if latest is None:
            continue
        prefix = f"{run.run_relative}/state/final_step_"
        duplicates = [
            remote for remote, item in files.items()
            if remote.startswith(prefix) and remote.endswith("/trainer_state.pt")
            and item.size == latest.size
            and filecmp.cmp(item.source, latest.source, shallow=False)
        ]
        for remote in duplicates:
            del files[remote]
            skipped.append({
                "path": remote,
                "reason": "byte-identical duplicate of latest_trainer_state.pt",
            })
    return sorted(files.values(), key=lambda item: item.path_in_repo), skipped


INFERENCE_CHECKPOINT_SIDECARS = {
    "condition_lora_adapters.pt",
    "audio_condition_projector.pt",
    "audio_modulation_projector.pt",
    "audio_modulation_gate.pt",
}


def partition_release_files(
    project_root: Path,
    runs: Sequence[SelectedRun],
    files: Sequence[ReleaseFile],
) -> tuple[list[ReleaseFile], list[ReleaseFile]]:
    """Split the audited inventory into direct inference files and training material.

    The direct tier is the smallest file closure needed by the released
    checkpoint paths: the selected checkpoint, resolved run contract, and any
    run-level inference sidecar selected by its manifest. Everything else that
    survived the exclusion policy is preserved in continued-training TARs.
    """

    inventory = {item.path_in_repo: item for item in files}
    direct_paths: set[str] = set()

    def include(path: Path, *, required: bool = False) -> None:
        remote = path.relative_to(project_root).as_posix()
        if remote in inventory:
            direct_paths.add(remote)
        elif required:
            raise FileNotFoundError(f"Required direct-release file missing: {remote}")

    def include_manifest_sidecar(directory: Path) -> None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return
        include(manifest_path, required=True)
        manifest = load_json(manifest_path)
        active = directory / str(manifest.get("active_checkpoint") or "best.pt")
        include(active, required=True)

    for run in runs:
        run_root = project_root / run.run_relative
        checkpoint = project_root / run.checkpoint_relative
        include(checkpoint, required=True)
        for name in ("resolved_config.yaml", "resolved_runtime_config.json"):
            include(run_root / name)
        for name in INFERENCE_CHECKPOINT_SIDECARS:
            include(checkpoint.parent / name)

        # Registered categorical refiners and the optional audio selector are
        # runtime dependencies, but their curves/logs remain training material.
        for relative in run.additional_relative:
            sidecar = project_root / relative
            if sidecar.is_dir():
                include_manifest_sidecar(sidecar)
            elif sidecar.is_file():
                include(sidecar, required=True)
        include_manifest_sidecar(run_root / "categorical_refiner")
        include_manifest_sidecar(run_root / "audio_region_selector")

    direct = [item for item in files if item.path_in_repo in direct_paths]
    continued_training = [
        item for item in files if item.path_in_repo not in direct_paths
    ]
    if len(direct) + len(continued_training) != len(files):
        raise RuntimeError("Release partition does not cover the complete inventory")
    return direct, continued_training


def sensitive_values(project_root: Path) -> list[SensitiveValue]:
    candidates = [
        ("project root", str(project_root), "WSM_v1"),
        ("source repository", str(project_root.parent), "."),
        ("workspace user", getpass.getuser(), "<USER>"),
        ("host", socket.gethostname(), "<HOST>"),
        ("fully-qualified host", socket.getfqdn(), "<HOST>"),
        ("home directory", str(Path.home()), "<LOCAL_HOME>"),
    ]
    values: list[SensitiveValue] = []
    seen: set[bytes] = set()
    for label, raw, replacement in candidates:
        encoded = raw.encode()
        if len(encoded) < 3 or encoded in seen:
            continue
        seen.add(encoded)
        values.append(SensitiveValue(label, encoded, replacement.encode()))
    values.sort(key=lambda item: len(item.raw), reverse=True)
    return values


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("Weights & Biases key", re.compile(rb"(?i)(?:wandb[_-]?(?:api[_-]?)?key)[=: ]+[A-Za-z0-9]{32,}")),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("email address", re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
)
LOCAL_PATH_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9._-])/(?:research|common|home|scratch|mnt|gpfs|lustre|fsx|workspace|opt)(?:/[^\s'\"<>]+)+"
)


TEXT_PRIVATE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "institutional or cluster host",
        re.compile(rb"(?i)\b(?:[a-z0-9-]+\.)+(?:edu|internal|local)\b"),
    ),
    (
        "IPv4 address",
        re.compile(rb"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])"),
    ),
    (
        "scheduler or identity environment field",
        re.compile(
            rb"(?im)\b(?:HOSTNAME|SLURM_NODELIST|MASTER_ADDR|USER|LOGNAME|HOME)"
            rb"\s*[:=]\s*[^\s,;]+"
        ),
    ),
)


def sanitize_text(data: bytes, values: Sequence[SensitiveValue]) -> tuple[bytes, Counter[str]]:
    replacements: Counter[str] = Counter()
    sanitized = data
    for value in values:
        count = sanitized.count(value.raw)
        if count:
            sanitized = sanitized.replace(value.raw, value.replacement)
            replacements[value.label] += count
    for label, pattern in SECRET_PATTERNS:
        replacement = f"<{label.upper().replace(' ', '_')}>".encode()
        sanitized, count = pattern.subn(replacement, sanitized)
        replacements[label] += count
    for label, pattern in TEXT_PRIVATE_PATTERNS:
        sanitized, count = pattern.subn(f"<{label.upper().replace(chr(32), chr(95))}>".encode(), sanitized)
        replacements[label] += count
    sanitized, count = LOCAL_PATH_PATTERN.subn(b"<LOCAL_PATH>", sanitized)
    replacements["other local absolute path"] += count
    return sanitized, replacements


def binary_metadata_chunks(path: Path) -> Sequence[bytes]:
    """Return serialization metadata without scanning numerical tensor payloads."""

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                return [raw_size]
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size > 128 * 1024 * 1024:
                raise ValueError(f"Implausible safetensors header: {path}")
            return [raw_size, handle.read(header_size)]
    if suffix in {".pt", ".pth", ".ckpt"}:
        if zipfile.is_zipfile(path):
            chunks: list[bytes] = []
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    name = info.filename.lower()
                    is_metadata = (
                        name.endswith((".pkl", ".json", ".yaml", ".yml", ".txt"))
                        or name.rsplit("/", 1)[-1] in {"version", "byteorder"}
                    )
                    if is_metadata:
                        chunks.append(archive.read(info))
            return chunks
        # Legacy torch serialization stores pickle metadata before tensor
        # storage. Local paths and host/user strings occur in this prefix.
        with path.open("rb") as handle:
            return [handle.read(32 * 1024 * 1024)]
    with path.open("rb") as handle:
        chunks = []
        while chunk := handle.read(16 * 1024 * 1024):
            chunks.append(chunk)
        return chunks


def scan_binary(path: Path, values: Sequence[SensitiveValue]) -> list[str]:
    labels: set[str] = set()
    needles = [(value.label, value.raw) for value in values]
    for data in binary_metadata_chunks(path):
        for label, needle in needles:
            if needle in data:
                labels.add(label)
        for label, pattern in SECRET_PATTERNS:
            if label != "email address" and pattern.search(data):
                labels.add(label)
    return sorted(labels)


def sanitize_python_value(value: Any, values: Sequence[SensitiveValue]) -> Any:
    """Recursively sanitize strings in a locally produced torch checkpoint."""

    if isinstance(value, str):
        sanitized, _ = sanitize_text(value.encode("utf-8"), values)
        return sanitized.decode("utf-8")
    if isinstance(value, bytes):
        sanitized, _ = sanitize_text(value, values)
        return sanitized
    if isinstance(value, dict):
        return {
            sanitize_python_value(key, values): sanitize_python_value(item, values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_python_value(item, values) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_python_value(item, values) for item in value)
    if isinstance(value, set):
        return {sanitize_python_value(item, values) for item in value}
    if hasattr(value, "__dict__") and value.__class__.__module__ in {"argparse", "types"}:
        for key, item in vars(value).items():
            setattr(value, key, sanitize_python_value(item, values))
        return value
    return value


def sanitize_torch_checkpoint(
    source: Path, destination: Path, values: Sequence[SensitiveValue]
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to sanitize trainer checkpoints") from exc
    payload = torch.load(source, map_location="cpu", weights_only=False)
    torch.save(sanitize_python_value(payload, values), destination)


def audit_release(
    files: Sequence[ReleaseFile], values: Sequence[SensitiveValue], verbose: bool
) -> tuple[Counter[str], list[dict[str, Any]], set[str]]:
    replacements: Counter[str] = Counter()
    unsafe_binary: list[dict[str, Any]] = []
    torch_transform: set[str] = set()
    torch_suffixes = {".pt", ".pth", ".ckpt"}
    for index, item in enumerate(files, 1):
        if item.is_text:
            sanitized, counts = sanitize_text(item.source.read_bytes(), values)
            replacements.update(counts)
            second, second_counts = sanitize_text(sanitized, values)
            if second != sanitized or sum(second_counts.values()):
                raise RuntimeError(f"Non-idempotent sanitization: {item.path_in_repo}")
        else:
            findings = scan_binary(item.source, values)
            if findings and item.source.suffix.lower() in torch_suffixes:
                with tempfile.NamedTemporaryFile(
                    suffix=item.source.suffix, delete=False
                ) as temporary:
                    sanitized_path = Path(temporary.name)
                try:
                    sanitize_torch_checkpoint(item.source, sanitized_path, values)
                    remaining = scan_binary(sanitized_path, values)
                    if remaining:
                        unsafe_binary.append(
                            {"path": item.path_in_repo, "findings": remaining}
                        )
                    else:
                        torch_transform.add(item.path_in_repo)
                except Exception as exc:
                    unsafe_binary.append(
                        {
                            "path": item.path_in_repo,
                            "findings": findings,
                            "sanitization_error": repr(exc),
                        }
                    )
                finally:
                    sanitized_path.unlink(missing_ok=True)
            elif findings:
                unsafe_binary.append({"path": item.path_in_repo, "findings": findings})
        if verbose and (index % 10000 == 0 or index == len(files)):
            print(f"Audited {index:,}/{len(files):,} files", flush=True)
    return replacements, unsafe_binary, torch_transform



def build_manifest(
    runs: Sequence[SelectedRun],
    files: Sequence[ReleaseFile],
    direct_files: Sequence[ReleaseFile],
    continued_training_files: Sequence[ReleaseFile],
    skipped: Sequence[Mapping[str, str]],
    replacements: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "format": RELEASE_FORMAT,
        "generated_at": utc_now(),
        "selection": {
            "dense": "configs/wsm_v1_recommended_dense_channel_heads.json",
            "sparse": "recommended_checkpoint fields in configs/wsg_v1_modalities_sparse.json",
            "custom": "recommended_checkpoint fields in configs/wsg_v1_modalities_custom.json",
            "policy": "Official deployment runs, including unique Heads with quality warnings; no unselected ablations or superseded runs.",
        },
        "counts": {
            "runs": len(runs),
            "runs_by_family": dict(sorted(Counter(item.family for item in runs).items())),
            "recommendation_status": dict(sorted(Counter(item.recommendation_status for item in runs).items())),
            "quality_status": dict(sorted(Counter(item.quality_status for item in runs).items())),
            "files": len(files),
            "bytes": sum(item.size for item in files),
            "direct_inference_files": len(direct_files),
            "direct_inference_bytes": sum(item.size for item in direct_files),
            "continued_training_files": len(continued_training_files),
            "continued_training_bytes": sum(
                item.size for item in continued_training_files
            ),
            "skipped_files": len(skipped),
        },
        "tiers": {
            "direct_inference": {
                "policy": "Selected checkpoint, resolved run contract, and active inference sidecars retain their original repository paths.",
            },
            "continued_training_materials": {
                "policy": "Trainer states, final and intermediate checkpoints, statistics, logs, curves, and other retained records are stored in uncompressed TAR parts.",
                "archive_compression": "none",
                "maximum_part_bytes": ARCHIVE_TARGET_BYTES,
            },
            "excluded": {
                "directories": [
                    "pseudo-visualization",
                    "pseudo-visualization-inference",
                    "wsg-reader-visualization",
                    "wsg-reader-visualization-inference",
                    "visuals",
                    "top-level step_* preview directories and prediction files",
                ],
                "policy": "Redundant per-step visualizations, local runtime/cache files, host-bearing TensorBoard events, and symbolic links are not released.",
            },
        },
        "omissions": {
            "exact_duplicates": "state/final_step_*/trainer_state.pt is omitted only when byte-identical to latest_trainer_state.pt",
            "runtime_metadata": "host-bearing TensorBoard event files and symbolic links are omitted",
        },
        "privacy": {
            "source_directories_modified": False,
            "sanitizer_version": SANITIZER_VERSION,
            "text_replacements": dict(sorted(replacements.items())),
            "binary_policy": "sanitize supported PyTorch states; reject any remaining sensitive binary",
        },
        "runs": [dataclasses.asdict(item) for item in runs],
        "skipped": list(skipped),
    }



def model_card(manifest: Mapping[str, Any]) -> bytes:
    counts = manifest["counts"]
    family = counts["runs_by_family"]
    parts = manifest["packaging"]["continued_training_archive_parts"]
    return f"""---
library_name: diffusers
license: other
tags:
- world-snapshot
- tunnelingnet
- flux2
- wsg-processing
- wsg-generation
---

# WSM v1 Channel Heads

This private pre-release repository stores the official WSM v1 Channel Head
training runs in three release tiers.

1. **Direct inference files.** Recommended checkpoints, resolved run contracts,
   and active inference sidecars retain their original paths. These files can be
   used without unpacking an archive.
2. **Continued-training materials.** Trainer states, final and intermediate
   checkpoints, statistics, logs, curves, and other retained training records
   are stored in {parts} deterministic, uncompressed TAR part(s), each below
   15 GiB.
3. **Excluded artifacts.** Redundant per-step visualizations and local runtime
   metadata are not released.

To download only the directly usable inference tier:

~~~bash
hf download {DEFAULT_REPO_ID} --repo-type model \
  --exclude "continued_training_materials/*" --local-dir WSM_v1
~~~

To obtain the complete release and restore the continued-training material:

~~~bash
hf download {DEFAULT_REPO_ID} --repo-type model --local-dir WSM_v1
python WSM_v1/restore_wsm_v1.py \
  --parts-dir WSM_v1/continued_training_materials --output WSM_v1
~~~

The restored training layout is unchanged:

~~~text
WSM_v1/
├── train_log/
├── train_log_sparse/
└── train_log_custom/
~~~

## Release contents

- Dense runs: {family.get('train_log', 0)}
- Sparse runs: {family.get('train_log_sparse', 0)}
- Custom runs: {family.get('train_log_custom', 0)}
- Total selected runs: {counts['runs']}
- Direct inference files: {counts['direct_inference_files']}
- Continued-training files after restoration: {counts['continued_training_files']}

Dense selection follows configs/wsm_v1_recommended_dense_channel_heads.json.
Sparse and custom selection follows the recommended_checkpoint fields in their
modality files. Reviewed dense Heads carrying a quality warning remain included
because they are the documented checkpoint for a unique task. Unselected
ablations and superseded runs are omitted.

TAR is intentionally uncompressed: it reduces repository file count without
adding expensive compression around model states, NPZ files, and images that
are already compressed. All archive members retain their original relative
paths. Per-step pseudo-visualization, wsg-reader-visualization, generic
visuals directories, and top-level sparse step previews are omitted.

Source directories are never modified. Local paths, user/host identifiers,
emails, and credential-shaped strings are removed from uploaded text and
supported PyTorch states. See RELEASE_MANIFEST.json for the exact selection,
packaging, and privacy policy.
""".encode()


def state_fingerprint(item: ReleaseFile) -> str:
    stat = item.source.stat()
    raw = f"{item.size}:{stat.st_mtime_ns}:{SANITIZER_VERSION}:{int(item.is_text)}"
    return hashlib.sha256(raw.encode()).hexdigest()



def archive_fingerprint(items: Sequence[ReleaseFile]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.path_in_repo.encode())
        digest.update(b"\0")
        digest.update(state_fingerprint(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()



def build_archive_parts(
    files: Sequence[ReleaseFile], target_bytes: int = ARCHIVE_TARGET_BYTES
) -> list[ArchivePart]:
    batches: list[list[ReleaseFile]] = []
    batch: list[ReleaseFile] = []
    batch_bytes = 0
    for item in files:
        if item.size > target_bytes:
            raise ValueError(
                f"Single continued-training file exceeds TAR target: "
                f"{item.path_in_repo} ({item.size / 2**30:.2f} GiB)"
            )
        if batch and batch_bytes + item.size > target_bytes:
            batches.append(batch)
            batch = []
            batch_bytes = 0
        batch.append(item)
        batch_bytes += item.size
    if batch:
        batches.append(batch)

    total = len(batches)
    parts: list[ArchivePart] = []
    for index, items_list in enumerate(batches, 1):
        items = tuple(items_list)
        parts.append(
            ArchivePart(
                index=index,
                total=total,
                items=items,
                path_in_repo=(
                    "continued_training_materials/"
                    f"continued_training_materials_part_{index:03d}_of_{total:03d}.tar"
                ),
                source_bytes=sum(item.size for item in items),
                fingerprint=archive_fingerprint(items),
            )
        )
    return parts


def write_archive_part(
    part: ArchivePart,
    destination: Path,
    values: Sequence[SensitiveValue],
    torch_transform: set[str],
    temporary_root: Path,
) -> None:
    def portable_header(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        return info

    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for item in part.items:
            if item.is_text:
                data, _ = sanitize_text(item.source.read_bytes(), values)
                info = portable_header(
                    archive.gettarinfo(str(item.source), arcname=item.path_in_repo)
                )
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif item.path_in_repo in torch_transform:
                with tempfile.NamedTemporaryFile(
                    suffix=item.source.suffix, dir=temporary_root, delete=False
                ) as temporary:
                    sanitized_path = Path(temporary.name)
                try:
                    sanitize_torch_checkpoint(item.source, sanitized_path, values)
                    archive.add(
                        sanitized_path, arcname=item.path_in_repo,
                        recursive=False, filter=portable_header,
                    )
                finally:
                    sanitized_path.unlink(missing_ok=True)
            else:
                archive.add(
                    item.source, arcname=item.path_in_repo,
                    recursive=False, filter=portable_header,
                )

def reuse_audit_report(
    path: Path,
    repo_id: str,
    runs: Sequence[SelectedRun],
    files: Sequence[ReleaseFile],
) -> tuple[Counter[str], list[dict[str, Any]], set[str]]:
    """Reuse a recent complete audit only when its exact source inventory is unchanged."""
    if not path.is_file():
        raise FileNotFoundError(f"No reusable audit report: {path}")
    report = load_json(path)
    if report.get("repo_id") != repo_id:
        raise ValueError(f"Audit report belongs to {report.get('repo_id')}, not {repo_id}")
    if report.get("unsafe_binary"):
        raise RuntimeError("Cannot reuse an audit report that contains unsafe binaries")
    prior = report.get("manifest") or {}
    counts = prior.get("counts") or {}
    if int(counts.get("files", -1)) != len(files):
        raise RuntimeError("Reusable audit file count no longer matches the selected release")
    if int(counts.get("bytes", -1)) != sum(item.size for item in files):
        raise RuntimeError("Reusable audit byte count no longer matches the selected release")
    expected_runs = json.loads(json.dumps([dataclasses.asdict(item) for item in runs]))
    if prior.get("runs") != expected_runs:
        raise RuntimeError("Reusable audit run selection no longer matches the release")
    privacy = prior.get("privacy") or {}
    if int(privacy.get("sanitizer_version", -1)) != SANITIZER_VERSION:
        raise RuntimeError("Reusable audit uses a different sanitizer version")
    audited_at = datetime.fromisoformat(str(report["generated_at"]))
    changed = [
        item.path_in_repo
        for item in files
        if datetime.fromtimestamp(item.source.stat().st_mtime, timezone.utc) > audited_at
    ]
    if changed:
        raise RuntimeError(
            "Selected source files changed after the reusable audit; first changes: "
            + ", ".join(changed[:10])
        )
    print(
        f"Reusing complete privacy audit from {report['generated_at']} "
        f"for {len(files):,} unchanged files",
        flush=True,
    )
    return (
        Counter(privacy.get("text_replacements") or {}),
        [],
        set(report.get("sanitized_torch_checkpoints") or []),
    )


def load_state(path: Path, repo_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"format": RELEASE_FORMAT, "repo_id": repo_id, "uploaded": {}}
    state = load_json(path)
    if state.get("repo_id") != repo_id:
        raise ValueError(f"Upload state belongs to {state.get('repo_id')}, not {repo_id}")
    return state


def ensure_private_repo(api: Any, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not bool(getattr(info, "private", False)):
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)
        info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not bool(getattr(info, "private", False)):
        raise RuntimeError(f"Refusing upload because {repo_id} is not private")


def upload_release(
    repo_id: str,
    direct_files: Sequence[ReleaseFile],
    continued_training_files: Sequence[ReleaseFile],
    values: Sequence[SensitiveValue],
    torch_transform: set[str],
    manifest: Mapping[str, Any],
    state_path: Path,
    temporary_root: Path,
    token: str | None,
    retries: int,
    recreate_repo: bool,
) -> None:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before uploading") from exc

    api = HfApi(token=token)
    temporary_root.mkdir(parents=True, exist_ok=True)
    if recreate_repo:
        api.delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)
        atomic_write_json(
            state_path,
            {"format": RELEASE_FORMAT, "repo_id": repo_id, "uploaded": {}},
        )
    ensure_private_repo(api, repo_id)
    print(f"Verified private repository: https://huggingface.co/{repo_id}")
    state = load_state(state_path, repo_id)
    uploaded = state.setdefault("uploaded", {})

    def save_progress() -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(state_path, state)

    def retry(label: str, callback: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return callback()
            except Exception as exc:
                last_error = exc
                if attempt == retries:
                    break
                delay = min(120, 10 * 2 ** (attempt - 1))
                print(
                    f"Upload failed ({attempt}/{retries}) for {label}: {exc}; "
                    f"retry in {delay}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    restore_path = Path(__file__).with_name("restore_wsm_v1_models.py")
    metadata = {
        "README.md": model_card(manifest),
        "RELEASE_MANIFEST.json": (
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        ).encode(),
        "restore_wsm_v1.py": restore_path.read_bytes(),
    }
    for remote, data in metadata.items():
        digest = hashlib.sha256(data).hexdigest()
        if uploaded.get(remote) == digest:
            continue
        retry(
            remote,
            lambda remote=remote, data=data: api.upload_file(
                path_or_fileobj=data,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update WSM v1 release metadata",
            ),
        )
        uploaded[remote] = digest
        save_progress()

    pending_direct = [
        item
        for item in direct_files
        if uploaded.get(item.path_in_repo) != state_fingerprint(item)
    ]
    print(
        f"Direct inference tier: {len(pending_direct):,}/{len(direct_files):,} "
        "files pending",
        flush=True,
    )
    for start in range(0, len(pending_direct), DIRECT_UPLOAD_BATCH_FILES):
        batch = pending_direct[start : start + DIRECT_UPLOAD_BATCH_FILES]
        temporary_paths: list[Path] = []
        in_memory_files: list[io.BytesIO] = []
        operations: list[Any] = []
        try:
            for item in batch:
                if item.is_text:
                    data, _ = sanitize_text(item.source.read_bytes(), values)
                    payload = io.BytesIO(data)
                    in_memory_files.append(payload)
                    source: Any = payload
                elif item.path_in_repo in torch_transform:
                    with tempfile.NamedTemporaryFile(
                        suffix=item.source.suffix,
                        dir=temporary_root,
                        delete=False,
                    ) as temporary:
                        sanitized_path = Path(temporary.name)
                    sanitize_torch_checkpoint(item.source, sanitized_path, values)
                    temporary_paths.append(sanitized_path)
                    source = str(sanitized_path)
                else:
                    source = str(item.source)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=item.path_in_repo,
                        path_or_fileobj=source,
                    )
                )
            ordinal = start // DIRECT_UPLOAD_BATCH_FILES + 1
            total_batches = (
                len(pending_direct) + DIRECT_UPLOAD_BATCH_FILES - 1
            ) // DIRECT_UPLOAD_BATCH_FILES
            retry(
                f"direct inference batch {ordinal}/{total_batches}",
                lambda: api.create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=operations,
                    commit_message=(
                        f"Upload direct inference files {ordinal} of {total_batches}"
                    ),
                ),
            )
            for item in batch:
                uploaded[item.path_in_repo] = state_fingerprint(item)
            save_progress()
            print(
                f"Direct inference files: {min(start + len(batch), len(pending_direct)):,}/"
                f"{len(pending_direct):,}",
                flush=True,
            )
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)
            for payload in in_memory_files:
                payload.close()

    parts = build_archive_parts(continued_training_files)
    pending = [
        part
        for part in parts
        if uploaded.get(part.path_in_repo) != part.fingerprint
    ]
    total_bytes = sum(part.source_bytes for part in pending)
    completed_bytes = 0
    started = time.monotonic()
    print(
        f"Continued-training materials: {len(pending):,}/{len(parts):,} "
        f"TAR parts pending, {total_bytes / 2**30:.2f} GiB source bytes",
        flush=True,
    )
    for ordinal, part in enumerate(pending, 1):
        archive_path = temporary_root / Path(part.path_in_repo).name
        try:
            print(
                f"Building {part.path_in_repo}: {len(part.items):,} files, "
                f"{part.source_bytes / 2**30:.2f} GiB",
                flush=True,
            )
            write_archive_part(
                part, archive_path, values, torch_transform, temporary_root
            )
            if archive_path.stat().st_size >= 15 * 1024**3:
                raise RuntimeError(
                    f"Archive exceeds 15 GiB safety boundary: {archive_path}"
                )
            retry(
                part.path_in_repo,
                lambda: api.upload_file(
                    path_or_fileobj=str(archive_path),
                    path_in_repo=part.path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=(
                        "Upload continued-training materials "
                        f"{part.index} of {part.total}"
                    ),
                ),
            )
            uploaded[part.path_in_repo] = part.fingerprint
            save_progress()
            completed_bytes += part.source_bytes
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = completed_bytes / elapsed
            eta = max(total_bytes - completed_bytes, 0) / rate if rate else 0
            print(
                f"[{ordinal:,}/{len(pending):,}] {completed_bytes / 2**30:.2f}/"
                f"{total_bytes / 2**30:.2f} GiB | {rate / 2**20:.1f} MiB/s | "
                f"ETA {eta / 3600:.1f}h",
                flush=True,
            )
        finally:
            archive_path.unlink(missing_ok=True)


def make_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "audit", "upload"))
    result.add_argument("--project-root", type=Path, default=project_root_from_script())
    result.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    result.add_argument("--token", default=None, help="normally read from the HF login cache")
    result.add_argument("--state-path", type=Path, default=None)
    result.add_argument("--temporary-root", type=Path, default=None)
    result.add_argument("--retries", type=int, default=5)
    result.add_argument(
        "--reuse-audit-report",
        action="store_true",
        help="reuse a complete audit only if its selected files are unchanged",
    )
    result.add_argument(
        "--recreate-repo",
        action="store_true",
        help="replace an incomplete legacy release with the three-tier layout",
    )
    result.add_argument("--verbose", action="store_true")
    return result


def main() -> None:
    args = make_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    state_path = (
        args.state_path
        or project_root / "train_log" / ".wsm_v1_model_release_archive_state.json"
    )
    runs = select_runs(project_root)
    validate_runs(project_root, runs)
    files, skipped = iter_release_files(project_root, runs)
    direct_files, continued_training_files = partition_release_files(
        project_root, runs, files
    )
    archive_parts = build_archive_parts(continued_training_files)

    by_family = Counter(item.family for item in runs)
    print(
        f"Selected {len(runs):,} runs: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(by_family.items())
        )
    )
    print(
        f"Tier 1 · direct inference: {len(direct_files):,} files, "
        f"{sum(item.size for item in direct_files) / 2**30:.2f} GiB"
    )
    print(
        f"Tier 2 · continued-training materials: "
        f"{len(continued_training_files):,} files, "
        f"{sum(item.size for item in continued_training_files) / 2**30:.2f} GiB "
        f"in {len(archive_parts):,} uncompressed TAR part(s)"
    )
    print(
        f"Tier 3 · excluded local/redundant artifacts: {len(skipped):,} entries"
    )
    if args.action == "plan":
        return

    values = sensitive_values(project_root)
    report_path = state_path.with_name("wsm_v1_model_release_audit.json")
    if args.reuse_audit_report:
        replacements, unsafe_binary, torch_transform = reuse_audit_report(
            report_path, args.repo_id, runs, files
        )
    else:
        replacements, unsafe_binary, torch_transform = audit_release(
            files, values, args.verbose
        )
    manifest = build_manifest(
        runs,
        files,
        direct_files,
        continued_training_files,
        skipped,
        replacements,
    )
    manifest["packaging"] = {
        "layout": "direct-inference-plus-continued-training-tar-parts",
        "reason": (
            "Keep recommended inference artifacts directly loadable while "
            "remaining below the hosting repository file-count limit."
        ),
        "direct_inference_files": len(direct_files),
        "continued_training_archive_target_bytes": ARCHIVE_TARGET_BYTES,
        "continued_training_archive_parts": len(archive_parts),
        "parts": [
            {
                "path": part.path_in_repo,
                "files": len(part.items),
                "source_bytes": part.source_bytes,
                "fingerprint": part.fingerprint,
            }
            for part in archive_parts
        ],
    }
    atomic_write_json(
        report_path,
        {
            "generated_at": utc_now(),
            "repo_id": args.repo_id,
            "repository_visibility": "private",
            "manifest": manifest,
            "unsafe_binary": unsafe_binary,
            "sanitized_torch_checkpoints": sorted(torch_transform),
        },
    )
    print(f"Text replacements: {dict(replacements)}")
    print(f"Unsafe binaries: {len(unsafe_binary)}")
    print(f"Audit report: {report_path}")
    if unsafe_binary:
        examples = "\n".join(
            f"  {item['path']}: {', '.join(item['findings'])}"
            for item in unsafe_binary[:20]
        )
        raise RuntimeError(
            "Privacy audit rejected binary artifacts; nothing was uploaded:\n"
            + examples
        )
    if args.action == "audit":
        return

    temporary_root = (
        args.temporary_root
        or project_root / "train_log" / ".wsm_v1_release_tmp"
    )
    upload_release(
        args.repo_id,
        direct_files,
        continued_training_files,
        values,
        torch_transform,
        manifest,
        state_path,
        temporary_root,
        args.token,
        args.retries,
        args.recreate_repo,
    )


if __name__ == "__main__":
    main()
