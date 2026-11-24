"""Selective model resolution for WSM repositories on the Hugging Face Hub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import WSMTask, runtime_root
from .container import DEFAULT_FILENAME, WSMContainer


DEFAULT_MODEL_REPO = "World-Snapshot/WSM_v1"
DEFAULT_BASE_MODEL_REPO = "black-forest-labs/FLUX.2-klein-4B"


@dataclass(frozen=True, slots=True)
class WSMArtifacts:
    task: WSMTask
    checkpoint: Path
    base_model: Path | None
    model_root: Path
    runtime_root: Path


def _snapshot_download(
    *,
    repo_id: str,
    token: str | bool | None,
    **kwargs: Any,
) -> str:
    """Download with an actionable error for private or gated repositories."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        return snapshot_download(repo_id=repo_id, token=token, **kwargs)
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise PermissionError(
            f"Cannot access Hugging Face repository {repo_id!r}. It may be private "
            "or gated. Run 'hf auth login', set HF_TOKEN, or pass token=... to "
            "WSMPipeline.from_pretrained(), and verify that the account has access."
        ) from exc


def _remote_has_container(
    repo_id: str,
    *,
    revision: str | None,
    token: str | bool | None,
) -> bool:
    """Return whether a Hub model uses the single-file WSM layout."""
    from huggingface_hub import file_exists
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        return bool(
            file_exists(
                repo_id=repo_id,
                filename=DEFAULT_FILENAME,
                revision=revision,
                token=token,
            )
        )
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise PermissionError(
            f"Cannot access Hugging Face repository {repo_id!r}. It may be private "
            "or gated. Run 'hf auth login', set HF_TOKEN, or pass token=... to "
            "WSMPipeline.from_pretrained(), and verify that the account has access."
        ) from exc


def resolve_artifacts(
    pretrained_model_name_or_path: str | Path,
    task: WSMTask,
    *,
    base_model: str | Path | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    download_base_model: bool = True,
    runtime: str | Path | None = None,
) -> WSMArtifacts:
    """Resolve one task from a monolithic WSM checkpoint or legacy layout."""
    runtime_path = runtime_root(runtime)
    source = Path(str(pretrained_model_name_or_path)).expanduser()
    local_container = (
        source
        if source.is_file() and source.suffix == ".wsm"
        else source / DEFAULT_FILENAME
        if source.is_dir() and (source / DEFAULT_FILENAME).is_file()
        else None
    )
    if local_container is not None:
        return _resolve_container_artifacts(
            WSMContainer.open(local_container),
            task,
            runtime_path=runtime_path,
            base_model=base_model,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            download_base_model=download_base_model,
        )
    if (
        not source.exists()
        and not local_files_only
        and _remote_has_container(
            str(pretrained_model_name_or_path), revision=revision, token=token
        )
    ):
        # The public WSM v1 repository contains one range-readable checkpoint.
        # Opening it fetches only the 64-byte header and terminal JSON index.
        container = WSMContainer.open(
            str(pretrained_model_name_or_path),
            revision=revision,
            token=token,
        )
        return _resolve_container_artifacts(
            container,
            task,
            runtime_path=runtime_path,
            base_model=base_model,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            download_base_model=download_base_model,
        )
    if source.exists():
        model_root = source.resolve()
        if model_root.is_file():
            checkpoint = model_root
            model_root = _model_root_from_checkpoint(checkpoint)
        else:
            checkpoint = model_root / task.checkpoint
    else:
        model_root = Path(
            _snapshot_download(
                repo_id=str(pretrained_model_name_or_path),
                allow_patterns=[
                    f"{task.run_dir}/**",
                    "RELEASE_MANIFEST.json",
                    "README.md",
                ],
                cache_dir=str(cache_dir) if cache_dir else None,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
        )
        checkpoint = model_root / task.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"The selected tunnel checkpoint is unavailable: {checkpoint}. "
            "Verify repository access and the released model revision."
        )

    base_path = None
    if download_base_model:
        base_path = _resolve_base_model(
            base_model,
            runtime_path=runtime_path,
            cache_dir=cache_dir,
            revision=None,
            token=token,
            local_files_only=local_files_only,
        )
    return WSMArtifacts(
        task=task,
        checkpoint=checkpoint,
        base_model=base_path,
        model_root=model_root,
        runtime_root=runtime_path,
    )


def _resolve_container_artifacts(
    container: WSMContainer,
    task: WSMTask,
    *,
    runtime_path: Path,
    base_model: str | Path | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    download_base_model: bool,
) -> WSMArtifacts:
    identity = f"{task.source}:{task.head_index:06d}"
    entry = container.task_entry(identity)
    names = list(entry.get("artifacts", []))
    if not names:
        raise ValueError(f"WSM checkpoint task {identity} has no artifacts")
    embedded_backbone = list(container.model.get("backbone", {}).get("artifacts", []))
    if download_base_model and base_model is None:
        names.extend(embedded_backbone)
    root = container.materialize(names, cache_dir)
    checkpoint = root / str(entry["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Materialized WSM checkpoint is missing: {checkpoint}")
    if not download_base_model:
        base_path = None
    elif base_model is None:
        base_path = root / str(container.model["backbone"].get("root", "backbone"))
        if not base_path.is_dir():
            raise FileNotFoundError(f"Embedded WSM backbone is missing: {base_path}")
    else:
        base_path = _resolve_base_model(
            base_model,
            runtime_path=runtime_path,
            cache_dir=cache_dir,
            revision=None,
            token=token,
            local_files_only=local_files_only,
        )
    return WSMArtifacts(
        task=task,
        checkpoint=checkpoint,
        base_model=base_path,
        model_root=root,
        runtime_root=runtime_path,
    )


def _resolve_base_model(
    value: str | Path | None,
    *,
    runtime_path: Path,
    cache_dir: str | Path | None,
    revision: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    if value is not None:
        candidate = Path(str(value)).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        repo_id = str(value)
    else:
        local = runtime_path / "models" / "black-forest-labs" / "FLUX.2-klein-4B"
        if local.is_dir():
            return local.resolve()
        repo_id = DEFAULT_BASE_MODEL_REPO
    return Path(
        _snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    )


def _model_root_from_checkpoint(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        if parent.name in {"train_log", "train_log_sparse", "train_log_custom"}:
            return parent.parent
    return checkpoint.parent
