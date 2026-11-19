"""Diffusers-compatible public pipeline for World Snapshot Models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from .catalog import TaskCatalog, WSMTask
from .hub import DEFAULT_MODEL_REPO, WSMArtifacts, resolve_artifacts
from .outputs import WSMPipelineOutput
from .runtime import RuntimeSession


_RGB_TOKENS = {
    "0-2",
    "Reference_IMG",
    "FirstFrame_RGB",
    "CurrentFrame_RGB_Reference",
    "CurrentFrame_RGB_Condition",
}


class WSMPipeline(DiffusionPipeline):
    """Load and repeatedly run one released WSM input-to-output tunnel."""

    def __init__(
        self,
        artifacts: WSMArtifacts,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        categorical_refiner: str = "auto",
        load_model: bool = True,
    ) -> None:
        super().__init__()
        self.register_to_config(
            task_id=artifacts.task.task_id,
            requested_device=str(device),
            torch_dtype=str(torch_dtype),
            categorical_refiner=str(categorical_refiner),
        )
        if torch_dtype != torch.bfloat16:
            raise ValueError("WSM v1 currently supports torch.bfloat16 inference")
        self.artifacts = artifacts
        self.task = artifacts.task
        self._requested_device = str(device)
        self._torch_dtype = torch_dtype
        self._categorical_refiner = str(categorical_refiner)
        self._session: RuntimeSession | None = None
        if load_model:
            self._ensure_session()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = DEFAULT_MODEL_REPO,
        *,
        inputs: Sequence[str] | str | None = None,
        output: str | None = None,
        source: str | None = None,
        head: str | int | None = None,
        base_model: str | Path | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        categorical_refiner: str = "auto",
        runtime_root: str | Path | None = None,
        load_model: bool = True,
        **kwargs: Any,
    ) -> "WSMPipeline":
        """Resolve a tunnel, download only its run, and load the shared backbone.

        ``inputs`` plus ``output`` are the preferred public identity. ``head`` is
        an advanced selector and must be qualified, for example ``dense:000003``.
        """
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported WSMPipeline options: {names}")
        catalog = TaskCatalog.load(runtime_root)
        task = catalog.resolve(inputs=inputs, output=output, source=source, head=head)
        artifacts = resolve_artifacts(
            pretrained_model_name_or_path,
            task,
            base_model=base_model,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            download_base_model=load_model,
            runtime=runtime_root,
        )
        return cls(
            artifacts,
            device=device,
            torch_dtype=torch_dtype,
            categorical_refiner=categorical_refiner,
            load_model=load_model,
        )

    @classmethod
    def list_tasks(
        cls,
        *,
        source: str | None = None,
        inputs: Sequence[str] | str | None = None,
        output: str | None = None,
        runtime_root: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Return every usable released mapping, optionally filtered."""
        catalog = TaskCatalog.load(runtime_root)
        return [
            task.as_dict()
            for task in catalog.list(source=source, inputs=inputs, output=output)
        ]

    @property
    def device(self) -> torch.device:
        if self._session is not None:
            return self._session.device
        return torch.device(self._requested_device)

    def to(self, device: str | torch.device, *args: Any, **kwargs: Any) -> "WSMPipeline":
        """Move the active TunnelingNet transformer to a CUDA device."""
        if args or kwargs:
            raise TypeError("WSMPipeline.to currently accepts only a device")
        self._requested_device = str(device)
        if self._session is None:
            self._ensure_session()
        else:
            self._session.to(self._requested_device)
        return self

    @torch.inference_mode()
    def __call__(
        self,
        inputs: Any = None,
        *,
        prompt: str = "",
        num_inference_steps: int = 10,
        seed: int = 20260812,
        size: int = 256,
        output_dir: str | Path | None = None,
        exact_prompt_only: bool = False,
        paper_demo: bool = False,
        return_dict: bool = True,
    ) -> WSMPipelineOutput | tuple[list[Image.Image], np.ndarray | None]:
        """Run the selected tunnel with paths, PIL images, arrays, or tensors."""
        session = self._ensure_session()
        temporary = tempfile.TemporaryDirectory(prefix="wsm-input-")
        try:
            work_root = (
                Path(output_dir).expanduser().resolve()
                if output_dir is not None
                else Path(temporary.name) / "output"
            )
            work_root.mkdir(parents=True, exist_ok=True)
            records = self._input_records(inputs, Path(temporary.name) / "inputs")
            result_path = work_root / "result.png"
            request = {
                "mode": "channel_head",
                "checkpoint": str(self.artifacts.checkpoint),
                "inputs": records,
                "prompt": str(prompt),
                "exact_prompt_only": bool(exact_prompt_only),
                "paper_demo": bool(paper_demo),
                "target": self.task.output,
                "steps": int(num_inference_steps),
                "seed": int(seed),
                "size": int(size),
                "categorical_refiner": self._categorical_refiner,
            }
            session(request, result_path)
            selected = Image.open(result_path).convert("RGB").copy()
            wsg = _load_generated_wsg(work_root / "generated.wsg.npz")
            manifest_path = work_root / "result_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {"outputs": []}
            )
            artifacts = (
                {path.name: path for path in work_root.iterdir() if path.is_file()}
                if output_dir is not None
                else {}
            )
            pipeline_output = WSMPipelineOutput(
                images=[selected],
                wsg=wsg,
                task_id=self.task.task_id,
                artifacts=artifacts,
                metadata={
                    "task": self.task.as_dict(),
                    "manifest": manifest,
                    "num_inference_steps": int(num_inference_steps),
                    "seed": int(seed),
                    "size": int(size),
                },
            )
            if return_dict:
                return pipeline_output
            return pipeline_output.images, pipeline_output.wsg
        finally:
            temporary.cleanup()

    def _ensure_session(self) -> RuntimeSession:
        if self._session is None:
            if self.artifacts.base_model is None:
                raise RuntimeError(
                    "This WSMPipeline was created with load_model=False and cannot infer"
                )
            self._session = RuntimeSession(
                runtime_root=self.artifacts.runtime_root,
                checkpoint=self.artifacts.checkpoint,
                base_model=self.artifacts.base_model,
                device=self._requested_device,
                categorical_refiner=self._categorical_refiner,
            )
        return self._session

    def _input_records(self, values: Any, temporary_root: Path) -> list[dict[str, str]]:
        expected = [token for token in self.task.inputs if token != "caption"]
        if not expected:
            if values not in (None, {}, [], ()):
                raise ValueError("This tunnel does not accept a visual or audio input")
            return []
        if isinstance(values, Mapping):
            missing = [token for token in expected if token not in values]
            extra = [str(token) for token in values if str(token) not in expected]
            if missing or extra:
                raise ValueError(
                    f"Inputs must match {expected}; missing={missing}, extra={extra}"
                )
            ordered = [values[token] for token in expected]
        elif len(expected) == 1:
            ordered = [values]
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, Path)):
            ordered = list(values)
            if len(ordered) != len(expected):
                raise ValueError(
                    f"Tunnel {self.task.task_id} requires {len(expected)} inputs in order: {expected}"
                )
        else:
            raise ValueError(
                f"Tunnel {self.task.task_id} requires an input mapping or {len(expected)} ordered values"
            )
        temporary_root.mkdir(parents=True, exist_ok=True)
        return [
            {
                "token": token,
                "path": str(_materialize_input(value, token, temporary_root, index)),
            }
            for index, (token, value) in enumerate(zip(expected, ordered))
        ]


def _materialize_input(value: Any, token: str, root: Path, index: int) -> Path:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if isinstance(value, Image.Image):
        path = root / f"{index:02d}_{_safe(token)}.png"
        value.convert("RGB").save(path)
        return path
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if token in _RGB_TOKENS and value.ndim == 3 and value.shape[-1] in {3, 4}:
            array = np.asarray(value)
            if array.dtype.kind == "f" and array.max(initial=0) <= 1.0:
                array = array * 255.0
            path = root / f"{index:02d}_{_safe(token)}.png"
            Image.fromarray(np.clip(array[..., :3], 0, 255).astype(np.uint8)).save(path)
            return path
        path = root / f"{index:02d}_{_safe(token)}.npz"
        np.savez_compressed(path, data=np.asarray(value))
        return path
    raise TypeError(
        f"Unsupported input type for modality {token!r}: {type(value).__name__}"
    )


def _load_generated_wsg(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        for key in ("wsg_data", "data", "arr_0"):
            if key in archive:
                return np.asarray(archive[key]).copy()
    return None


def _safe(value: str) -> str:
    return str(value).replace("/", "_").replace(",", "_").replace("-", "_")


WSM_Pipeline = WSMPipeline
