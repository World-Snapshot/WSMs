#!/usr/bin/env python3
"""Build and publish the inference-only, single-file WSM v1 checkpoint.

This publisher is deliberately separate from ``publish_wsm_v1_models.py``.
The older publisher preserves training records and continued-training material;
this publisher emits one physical ``WSM_v1.wsm`` file containing the complete
FLUX.2 Klein 4B runtime closure and every released TunnelingNet. Its terminal
index makes every component independently addressable by byte range.

No source run is modified. Upload is never implicit: ``plan``, ``audit``, and
``build`` are local operations, while ``upload`` is the only network action.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any, Mapping, Sequence


RELEASE_FORMAT = "world-snapshot-model-single-file-v1"
DEFAULT_REPO_ID = "World-Snapshot/WSM_v1"
DEFAULT_BACKBONE = Path(
    "/common/users/sf895/WSM_models/black-forest-labs/FLUX.2-klein-4B"
)
DEFAULT_FILENAME = "WSM_v1.wsm"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def repository_root(project_root: Path) -> Path:
    return project_root.parent


def load_container_module() -> Any:
    root = repository_root(project_root_from_script()) / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wsms import container

    return container


def backbone_files(backbone: Path) -> list[tuple[str, Path]]:
    root = backbone.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FLUX.2 Klein 4B backbone is missing: {root}")
    values: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        values.append((f"backbone/{path.relative_to(root).as_posix()}", path))
    required = {"backbone/model_index.json", "backbone/LICENSE.md"}
    names = {name for name, _ in values}
    missing = required - names
    if missing:
        raise FileNotFoundError(f"Backbone closure lacks: {sorted(missing)}")
    return values


def load_legacy_publisher(project_root: Path) -> Any:
    path = project_root / "manual_processing/model_release/publish_wsm_v1_models.py"
    spec = importlib.util.spec_from_file_location("wsm_v1_training_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the existing release publisher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def family_name(value: str) -> str:
    mapping = {
        "train_log": "dense",
        "train_log_sparse": "sparse",
        "train_log_custom": "custom",
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unknown WSM release family: {value}") from exc


def normalized_run_path(run: Any) -> PurePosixPath:
    source = PurePosixPath(run.run_relative)
    if len(source.parts) < 3:
        raise ValueError(f"Unexpected source run layout: {run.run_relative}")
    # Retain the run basename because it identifies the exact training
    # contract, but replace the training-log namespace with model semantics.
    return PurePosixPath(
        "tunnelingnets",
        family_name(run.family),
        run.head,
        source.parts[-1],
    )


def remap_release_files(
    items: Sequence[Any], runs: Sequence[Any]
) -> dict[str, str]:
    owners = {
        PurePosixPath(run.run_relative).as_posix(): run
        for run in runs
    }
    result: dict[str, str] = {}
    for item in items:
        source = PurePosixPath(item.path_in_repo)
        if len(source.parts) < 3:
            raise ValueError(f"Unexpected inference artifact path: {item.path_in_repo}")
        source_root = PurePosixPath(*source.parts[:3])
        run = owners.get(source_root.as_posix())
        if run is None:
            raise ValueError(
                "Inference artifact is outside every selected run and cannot be "
                f"placed safely: {item.path_in_repo}"
            )
        result[item.path_in_repo] = (
            normalized_run_path(run) / source.relative_to(source_root)
        ).as_posix()
    # Expose the learned component as a TunnelingNet. Its source filename may
    # describe the low-level optimizer implementation (for example LoRA),
    # which is not the identity of the released model component.
    for run in runs:
        if run.checkpoint_relative not in result:
            continue
        checkpoint = PurePosixPath(result[run.checkpoint_relative])
        result[run.checkpoint_relative] = checkpoint.with_name(
            f"tunnelingnet{checkpoint.suffix}"
        ).as_posix()
    return result


def remap_release_file(item: Any, runs: Sequence[Any]) -> str:
    """Compatibility helper for a single artifact."""
    return remap_release_files([item], runs)[item.path_in_repo]


def load_task_catalog(project_root: Path) -> dict[tuple[str, int], Any]:
    source_root = repository_root(project_root) / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from wsms.catalog import TaskCatalog

    return {
        (task.source, task.head_index): task
        for task in TaskCatalog.load(project_root)
    }


def collect_inference_files(
    legacy: Any,
    project_root: Path,
    runs: Sequence[Any],
) -> list[Any]:
    """Collect only the inference closure without walking training directories."""

    files: dict[str, Any] = {}

    def include(path: Path, *, required: bool = False) -> None:
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Required inference artifact is missing: {path}")
            return
        remote = path.relative_to(project_root).as_posix()
        stat = path.stat()
        files[remote] = legacy.ReleaseFile(
            path,
            remote,
            stat.st_size,
            legacy.looks_text(path, stat.st_size),
        )

    def include_manifest_sidecar(directory: Path) -> None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return
        include(manifest_path, required=True)
        manifest = legacy.load_json(manifest_path)
        active = directory / str(manifest.get("active_checkpoint") or "best.pt")
        include(active, required=True)

    for run in runs:
        run_root = project_root / run.run_relative
        checkpoint = project_root / run.checkpoint_relative
        include(checkpoint, required=True)
        for name in ("resolved_config.yaml", "resolved_runtime_config.json"):
            # Dense TunnelingNets use the canonical two-file runtime contract.
            # Some native sparse modules retain their own compact loader and
            # therefore do not carry these files at the run root.
            include(run_root / name)
        for name in legacy.INFERENCE_CHECKPOINT_SIDECARS:
            include(checkpoint.parent / name)
        for relative in run.additional_relative:
            sidecar = project_root / relative
            if sidecar.is_dir():
                include_manifest_sidecar(sidecar)
            else:
                include(sidecar, required=True)
        include_manifest_sidecar(run_root / "categorical_refiner")
        include_manifest_sidecar(run_root / "audio_region_selector")
    return [files[key] for key in sorted(files)]


def index_document(
    project_root: Path,
    runs: Sequence[Any],
    direct_files: Sequence[Any],
    backbone: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    mapped = remap_release_files(direct_files, runs)
    artifacts_by_run: dict[str, list[str]] = {}
    for destination in mapped.values():
        parts = PurePosixPath(destination).parts
        run_prefix = PurePosixPath(*parts[:4]).as_posix()
        artifacts_by_run.setdefault(run_prefix, []).append(destination)
    catalog = load_task_catalog(project_root)
    entries: list[dict[str, Any]] = []
    for run in runs:
        source = family_name(run.family)
        index = int(run.head.rsplit("_", 1)[-1])
        task = catalog.get((source, index))
        checkpoint = mapped.get(run.checkpoint_relative)
        if checkpoint is None:
            raise FileNotFoundError(
                f"Selected checkpoint is absent from inference closure: {run.checkpoint_relative}"
            )
        run_prefix = normalized_run_path(run).as_posix()
        artifacts = sorted(artifacts_by_run.get(run_prefix, []))
        entry: dict[str, Any] = {
            "task_id": task.task_id if task else f"{source}:{index:06d}",
            "tunnelingnet_id": f"{source}:{index:06d}",
            "family": source,
            "index": index,
            "inputs": list(task.inputs) if task else [],
            "output": task.output if task else "",
            "task_type": task.task_type if task else "",
            "quality_status": run.quality_status,
            "run_dir": run_prefix,
            "checkpoint": checkpoint,
            "artifacts": artifacts,
            "runtime": (
                "canonical-flux2-tunnelingnet"
                if (project_root / run.run_relative / "resolved_runtime_config.json").is_file()
                else "native-specialized"
            ),
        }
        entries.append(entry)
    entries.sort(key=lambda value: (value["family"], value["index"]))
    return {
        "format": RELEASE_FORMAT,
        "model_type": "world-snapshot-model",
        "architecture": "WSMPipeline",
        "backbone": {
            "name": "FLUX.2-klein-4B",
            "distribution": "embedded",
            "root": "backbone",
            "artifacts": [name for name, _ in backbone],
            "license": "backbone/LICENSE.md",
        },
        "activation": {
            "policy": "sparse-task-routing",
            "simultaneously_active_tunnelingnets": 1,
            "loading": "byte-range-selected-task-only",
        },
        "tunnelingnet_count": len(entries),
        "tasks": entries,
    }


def print_plan(
    runs: Sequence[Any],
    direct_files: Sequence[Any],
    backbone: Sequence[tuple[str, Path]],
    index: Mapping[str, Any],
) -> None:
    families: dict[str, int] = {}
    for run in runs:
        key = family_name(run.family)
        families[key] = families.get(key, 0) + 1
    print(f"Unified model: {len(runs):,} TunnelingNets")
    print("Families: " + ", ".join(f"{key}={value:,}" for key, value in sorted(families.items())))
    print(
        f"Inference closure: {len(direct_files):,} files, "
        f"{sum(item.size for item in direct_files) / 2**30:.2f} GiB"
    )
    print(
        f"Embedded backbone: {len(backbone):,} files, "
        f"{sum(path.stat().st_size for _, path in backbone) / 2**30:.2f} GiB"
    )
    total = sum(item.size for item in direct_files) + sum(
        path.stat().st_size for _, path in backbone
    )
    print(f"Single checkpoint payload: {total / 2**30:.2f} GiB before alignment/index")
    print("Excluded: training states, optimizer states, logs, curves, previews, and TAR archives")


def audit(
    legacy: Any,
    project_root: Path,
    direct_files: Sequence[Any],
    *,
    verbose: bool,
) -> tuple[Sequence[Any], set[str]]:
    values = legacy.sensitive_values(project_root)
    replacements, unsafe, torch_transform = legacy.audit_release(
        direct_files, values, verbose
    )
    print(f"Text replacements: {dict(replacements)}")
    if unsafe:
        examples = "\n".join(
            f"  {item['path']}: {', '.join(item['findings'])}"
            for item in unsafe[:20]
        )
        raise RuntimeError(
            "Privacy audit rejected inference artifacts; nothing was built or uploaded:\n"
            + examples
        )
    print(f"Privacy audit passed for {len(direct_files):,} inference files")
    return values, torch_transform


def prepare_single_file_members(
    legacy: Any,
    project_root: Path,
    runs: Sequence[Any],
    direct_files: Sequence[Any],
    backbone: Sequence[tuple[str, Path]],
    values: Sequence[Any],
    torch_transform: set[str],
    scratch: Path,
) -> list[tuple[str, Path]]:
    """Resolve the exact payload sources used by the monolithic checkpoint."""
    marker = scratch / ".wsm-single-file-inputs"
    if scratch.exists():
        if not marker.is_file():
            raise RuntimeError(f"Refusing to replace unmarked scratch path: {scratch}")
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    marker.write_text(RELEASE_FORMAT + "\n", encoding="utf-8")
    mapped = remap_release_files(direct_files, runs)
    members: list[tuple[str, Path]] = list(backbone)
    for ordinal, item in enumerate(direct_files, 1):
        name = mapped[item.path_in_repo]
        source = item.source
        if item.is_text:
            payload, _ = legacy.sanitize_text(source.read_bytes(), values)
            prepared = scratch / name
            prepared.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_bytes(payload)
            source = prepared
        elif item.path_in_repo in torch_transform:
            prepared = scratch / name
            prepared.parent.mkdir(parents=True, exist_ok=True)
            legacy.sanitize_torch_checkpoint(source, prepared, values)
            source = prepared
        members.append((name, source))
        if ordinal % 250 == 0:
            print(f"Prepared {ordinal:,}/{len(direct_files):,} TunnelingNet artifacts")
    license_path = repository_root(project_root) / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"WSM license is missing: {license_path}")
    members.append(("LICENSE", license_path))
    return members


def build_single_file(
    legacy: Any,
    project_root: Path,
    runs: Sequence[Any],
    direct_files: Sequence[Any],
    backbone: Sequence[tuple[str, Path]],
    index: Mapping[str, Any],
    values: Sequence[Any],
    torch_transform: set[str],
    output: Path,
) -> None:
    container = load_container_module()
    scratch = output.parent / ".wsm_v1_single_file_inputs"
    members = prepare_single_file_members(
        legacy,
        project_root,
        runs,
        direct_files,
        backbone,
        values,
        torch_transform,
        scratch,
    )
    print(f"Packing {len(members):,} addressable members into {output}")
    document = container.build_container(output, members, model_index=index)
    opened = container.WSMContainer.open(output)
    if opened.model.get("tunnelingnet_count") != len(runs):
        raise RuntimeError("Built checkpoint failed TunnelingNet-count verification")
    if len(document["members"]) != len(members):
        raise RuntimeError("Built checkpoint failed member-count verification")
    shutil.rmtree(scratch)
    print(
        f"Built and verified {output} "
        f"({output.stat().st_size / 2**30:.2f} GiB, {len(members):,} members)"
    )


def upload_single_file(
    output: Path,
    repo_id: str,
    *,
    token: str | None,
    allow_public: bool,
) -> None:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before uploading") from exc
    if not output.is_file():
        raise FileNotFoundError(output)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=False)
    if not bool(getattr(info, "private", False)) and not allow_public:
        raise RuntimeError(
            f"Refusing to upload to public repository {repo_id}; pass --allow-public-repo"
        )
    project_root = project_root_from_script()
    repository_root_path = repository_root(project_root)
    model_card = project_root / "manual_processing/model_release/README_WSM_V1.md"
    license_path = repository_root_path / "LICENSE"
    for metadata in (model_card, license_path):
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=DEFAULT_FILENAME, path_or_fileobj=str(output)),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(model_card)),
            CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=str(license_path)),
        ],
        commit_message="Publish the single-file WSM v1 checkpoint",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "audit", "build", "upload"))
    result.add_argument("--project-root", type=Path, default=project_root_from_script())
    result.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    result.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--token", default=None)
    result.add_argument("--allow-public-repo", action="store_true")
    result.add_argument("--verbose", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    legacy = load_legacy_publisher(project_root)
    runs = legacy.select_runs(project_root)
    legacy.validate_runs(project_root, runs)
    direct_files = collect_inference_files(legacy, project_root, runs)
    backbone = backbone_files(args.backbone)
    index = index_document(project_root, runs, direct_files, backbone)
    print_plan(runs, direct_files, backbone, index)
    if args.action == "plan":
        return
    values, torch_transform = audit(
        legacy, project_root, direct_files, verbose=args.verbose
    )
    if args.action == "audit":
        return
    output = (
        args.output
        or project_root / "train_log" / ".wsm_v1_single_file" / DEFAULT_FILENAME
    ).expanduser().resolve()
    build_single_file(
        legacy,
        project_root,
        runs,
        direct_files,
        backbone,
        index,
        values,
        torch_transform,
        output,
    )
    if args.action == "build":
        return
    upload_single_file(
        output,
        args.repo_id,
        token=args.token,
        allow_public=args.allow_public_repo,
    )


if __name__ == "__main__":
    main()
