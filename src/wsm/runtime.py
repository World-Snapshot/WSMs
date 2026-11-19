"""Bridge from the public package to the canonical WSM v1 inference runtime."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping


class RuntimeSession:
    """Thin adapter around the repository's reusable Channel Head session."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        checkpoint: Path,
        base_model: Path,
        device: str,
        categorical_refiner: str,
    ) -> None:
        self.runtime_root = Path(runtime_root).resolve()
        # Hugging Face snapshots expose files as symlinks into ``blobs/``.
        # Keep the logical snapshot path so the canonical runtime can locate
        # the resolved run configuration stored beside the checkpoint.
        self.checkpoint = Path(checkpoint).expanduser().absolute()
        self.base_model = Path(base_model).resolve()
        self.requested_device = str(device)
        self.backend = load_backend(self.runtime_root)

        os.environ["WSM_FLUX2_MODEL_PATH"] = str(self.base_model)
        previous_rank = os.environ.get("LOCAL_RANK")
        rank = _cuda_index(self.requested_device)
        if rank is not None:
            os.environ["LOCAL_RANK"] = str(rank)
        try:
            self.session = self.backend.ChannelHeadInferenceSession(
                self.checkpoint,
                categorical_refiner=categorical_refiner,
            )
        finally:
            if previous_rank is None:
                os.environ.pop("LOCAL_RANK", None)
            else:
                os.environ["LOCAL_RANK"] = previous_rank

    @property
    def device(self):
        return self.session.device

    def to(self, device: str):
        self.requested_device = str(device)
        self.session.to(device)
        return self

    def __call__(self, request: Mapping[str, Any], output_path: Path):
        return self.session(request, output_path)


def load_backend(root: Path) -> ModuleType:
    root = Path(root).resolve()
    script = root / "scripts" / "infer_TunnelingNet_Flux2_4B_wsg.py"
    if not script.is_file():
        raise FileNotFoundError(f"WSM inference runtime is missing: {script}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_name = f"wsm_v1_inference_{abs(hash(script))}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import WSM inference runtime from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _cuda_index(device: str) -> int | None:
    value = str(device).strip().lower()
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device!r}") from exc
    if value == "cpu":
        raise ValueError("WSM v1 currently requires a CUDA device for inference")
    return None
