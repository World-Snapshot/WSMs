"""Discovery and unambiguous identity for released WSM task tunnels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_FAMILY_CONFIGS = {
    "dense": "wsm_v1_recommended_dense_channel_heads.json",
    "sparse": "wsg_v1_modalities_sparse.json",
    "custom": "wsg_v1_modalities_custom.json",
}
_FAMILY_LOG_ROOTS = {
    "dense": "train_log",
    "sparse": "train_log_sparse",
    "custom": "train_log_custom",
}


def runtime_root(explicit: str | Path | None = None) -> Path:
    """Locate WSM_v1 in a source checkout or in the installed wheel."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not (root / "configs").is_dir():
            raise FileNotFoundError(f"WSM runtime has no configs directory: {root}")
        return root

    source = Path(__file__).resolve().parents[2] / "WSM_v1"
    if (source / "configs").is_dir():
        return source
    bundled = Path(__file__).resolve().parent / "_bundled" / "WSM_v1"
    if (bundled / "configs").is_dir():
        return bundled
    raise FileNotFoundError(
        "The WSM v1 runtime is missing. Reinstall the official 'wsm' wheel."
    )


def _tokens(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple)) else [value]
    return tuple(str(token).strip() for token in values if str(token).strip())


@dataclass(frozen=True, slots=True)
class WSMTask:
    """One released implementation of an ordered input-to-output task tunnel."""

    source: str
    head_index: int
    task_type: str
    inputs: tuple[str, ...]
    output: str
    checkpoint: str
    run_dir: str
    condition_description: str = ""
    target_description: str = ""
    quality_status: str = "recommended"

    @property
    def head(self) -> str:
        return f"{self.source}:{self.head_index:06d}"

    @property
    def mapping(self) -> str:
        left = "+".join(self.inputs) if self.inputs else "∅"
        return f"{left}->{self.output}"

    @property
    def task_id(self) -> str:
        return f"{self.source}:{self.mapping}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "head": self.head,
            "source": self.source,
            "head_index": self.head_index,
            "task_type": self.task_type,
            "inputs": list(self.inputs),
            "output": self.output,
            "mapping": self.mapping,
            "checkpoint": self.checkpoint,
            "run_dir": self.run_dir,
            "condition_description": self.condition_description,
            "target_description": self.target_description,
            "quality_status": self.quality_status,
        }


class TaskCatalog:
    """Catalog of all recommended Dense, Sparse, and Custom task tunnels."""

    def __init__(self, tasks: Iterable[WSMTask], *, root: Path) -> None:
        self.root = Path(root)
        self._tasks = tuple(
            sorted(tasks, key=lambda task: (task.source, task.head_index))
        )
        identities = [task.head for task in self._tasks]
        if len(identities) != len(set(identities)):
            raise ValueError("The task catalog contains duplicate qualified Head IDs")

    @classmethod
    def load(cls, root: str | Path | None = None) -> "TaskCatalog":
        resolved_root = runtime_root(root)
        config_root = resolved_root / "configs"
        tasks: list[WSMTask] = []

        dense = json.loads(
            (config_root / _FAMILY_CONFIGS["dense"]).read_text(encoding="utf-8")
        )
        for entry in dense.get("entries", []):
            candidate = ((entry.get("recommended") or {}).get("candidate") or {})
            checkpoint = str(candidate.get("checkpoint") or "")
            if not checkpoint:
                continue
            tasks.append(
                WSMTask(
                    source="dense",
                    head_index=int(entry["head_index"]),
                    task_type=str(entry.get("task_type") or candidate.get("task_type") or ""),
                    inputs=_tokens(entry.get("condition") or candidate.get("condition")),
                    output=str(entry.get("target") or candidate.get("target") or ""),
                    checkpoint=checkpoint,
                    run_dir=str(candidate.get("run_dir") or Path(checkpoint).parents[2]),
                    condition_description=str(candidate.get("condition_description") or ""),
                    target_description=str(candidate.get("target_description") or ""),
                    quality_status=(
                        "needs_improvement"
                        if _contains_warning(entry)
                        else "recommended"
                    ),
                )
            )

        for source in ("sparse", "custom"):
            document = json.loads(
                (config_root / _FAMILY_CONFIGS[source]).read_text(encoding="utf-8")
            )
            for name, entry in (document.get("wsg_v1") or {}).items():
                if not str(name).startswith("Channel_Head_"):
                    continue
                checkpoint = str(entry.get("recommended_checkpoint") or "")
                if not checkpoint:
                    continue
                index = int(str(name).rsplit("_", 1)[-1])
                tasks.append(
                    WSMTask(
                        source=source,
                        head_index=index,
                        task_type=str(entry.get("task_type") or ""),
                        inputs=_tokens(entry.get("condition")),
                        output=str(entry.get("target") or ""),
                        checkpoint=checkpoint,
                        run_dir=_run_dir(checkpoint),
                        condition_description=str(entry.get("condition_description") or ""),
                        target_description=str(entry.get("target_description") or ""),
                    )
                )
        return cls(tasks, root=resolved_root)

    def list(
        self,
        *,
        source: str | None = None,
        inputs: Sequence[str] | str | None = None,
        output: str | None = None,
    ) -> list[WSMTask]:
        source_value = _normalize_source(source) if source else None
        input_values = _tokens(inputs) if inputs is not None else None
        result = []
        for task in self._tasks:
            if source_value and task.source != source_value:
                continue
            if input_values is not None and task.inputs != input_values:
                continue
            if output is not None and task.output != str(output):
                continue
            result.append(task)
        return result

    def by_head(self, head: str, *, source: str | None = None) -> WSMTask:
        source_value, index = parse_head(head, source=source)
        matches = [
            task
            for task in self._tasks
            if task.source == source_value and task.head_index == index
        ]
        if not matches:
            raise KeyError(f"No released WSM Head matches {source_value}:{index:06d}")
        return matches[0]

    def resolve(
        self,
        *,
        inputs: Sequence[str] | str | None = None,
        output: str | None = None,
        source: str | None = None,
        head: str | int | None = None,
    ) -> WSMTask:
        if head is not None:
            if inputs is not None or output is not None:
                raise ValueError("Choose either inputs/output or head, not both")
            return self.by_head(str(head), source=source)
        if inputs is None or output is None:
            raise ValueError("Task selection requires both inputs and output")

        requested = _tokens(inputs)
        matches = self.list(source=source, inputs=requested, output=output)
        if not matches:
            unordered = [
                task
                for task in self.list(source=source, output=output)
                if len(task.inputs) == len(requested)
                and set(task.inputs) == set(requested)
            ]
            if len(unordered) == 1:
                matches = unordered
            elif unordered:
                order_help = ", ".join(task.task_id for task in unordered[:8])
                raise ValueError(
                    "The input modalities exist in a different routing order. "
                    f"Use one of: {order_help}"
                )
        if not matches:
            left = "+".join(requested) if requested else "∅"
            raise KeyError(f"No released WSM tunnel matches {left}->{output}")
        if len(matches) > 1:
            choices = ", ".join(task.head for task in matches)
            raise ValueError(
                "More than one released implementation matches this mapping; "
                f"select source or qualified head: {choices}"
            )
        return matches[0]

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)


def parse_head(value: str, *, source: str | None = None) -> tuple[str, int]:
    raw = str(value).strip()
    if ":" in raw:
        family, index_text = raw.split(":", 1)
        family = _normalize_source(family)
    else:
        if source is None:
            raise ValueError(
                "Bare Head indices are ambiguous; use dense:000003, "
                "sparse:000003, or custom:000003"
            )
        family = _normalize_source(source)
        index_text = raw
    index_text = index_text.removeprefix("Channel_Head_")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"Invalid WSM Head identifier: {value!r}") from exc
    if index < 0:
        raise ValueError("Head index must be non-negative")
    return family, index


def _normalize_source(value: str) -> str:
    source = str(value).strip().lower()
    aliases = {"train_log": "dense", "train_log_sparse": "sparse", "train_log_custom": "custom"}
    source = aliases.get(source, source)
    if source not in _FAMILY_CONFIGS:
        raise ValueError("source must be dense, sparse, or custom")
    return source


def _run_dir(checkpoint: str) -> str:
    parts = Path(checkpoint).parts
    return Path(*parts[:3]).as_posix() if len(parts) >= 3 else str(Path(checkpoint).parent)


def _contains_warning(value: Any) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("severity") or "").lower() == "warning":
            return True
        return any(_contains_warning(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_warning(item) for item in value)
    return False
