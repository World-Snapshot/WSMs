"""World Snapshot Models public Python API."""

from __future__ import annotations

from .version import __version__

__all__ = [
    "TaskCatalog",
    "WSMPipeline",
    "WSMPipelineOutput",
    "WSMTask",
    "WSM_Pipeline",
    "__version__",
]


def __getattr__(name: str):
    if name in {"TaskCatalog", "WSMTask"}:
        from .catalog import TaskCatalog, WSMTask

        return {"TaskCatalog": TaskCatalog, "WSMTask": WSMTask}[name]
    if name == "WSMPipelineOutput":
        from .outputs import WSMPipelineOutput

        return WSMPipelineOutput
    if name in {"WSMPipeline", "WSM_Pipeline"}:
        from .pipeline import WSMPipeline

        return WSMPipeline
    raise AttributeError(name)
