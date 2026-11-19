from __future__ import annotations
import base64
import io
from collections import deque
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
WEBUI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEBUI_ROOT.parent
MODALITY_CONFIGS = {
    "dense": PROJECT_ROOT / "configs/wsg_v1_modalities_dense.json",
    "sparse": PROJECT_ROOT / "configs/wsg_v1_modalities_sparse.json",
    "custom": PROJECT_ROOT / "configs/wsg_v1_modalities_custom.json",
}
RGB_INPUT_TOKENS = {
    "0-2",
    "Reference_IMG",
    "FirstFrame_RGB",
    "CurrentFrame_RGB_Reference",
    "CurrentFrame_RGB_Condition",
}
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONTROLWORLD_PLUGINS = PROJECT_ROOT.parents[1] / "ControlWorld" / "plugins"
if str(CONTROLWORLD_PLUGINS) not in sys.path:
    sys.path.append(str(CONTROLWORLD_PLUGINS))
from plugin_base import PluginBackendBase
from .catalog import build_catalog, resolve_task
from .input_validation import (
    InputValidationError,
    canonical_array_from_contract,
    inspect_tensor_input,
    validate_rgb_image,
    validate_wav_audio,
)
class PluginBackend(PluginBackendBase):
    def __init__(self, server):
        super().__init__(server)
        self.name = "wsm-v1-inference"
        self.display_name = "WSM V1"
        self._catalog: dict[str, Any] | None = None
        self._catalog_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
    def setup_gui(self):
        if self.viser is None:
            return
        self.gui_controls["status"] = self.viser.gui.add_markdown(
            "Use the **WSM V1** browser panel to select inputs, outputs, and checkpoints."
        )
    def _get_catalog(self, refresh: bool = False) -> dict[str, Any]:
        with self._catalog_lock:
            if refresh or self._catalog is None:
                self._catalog = build_catalog()
            return self._catalog
    def handle_get_catalog(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._get_catalog(refresh=bool(data.get("refresh", False)))

    @staticmethod
    def _self_preprocessor_for_token(
        catalog: dict[str, Any], token: str,
    ) -> dict[str, Any]:
        """Choose the released RGB estimator that can materialize ``token`` as WSG."""
        candidates = []
        for task in catalog.get("tasks", []):
            visual_inputs = [
                str(value) for value in task.get("inputs", [])
                if str(value) != "caption"
            ]
            if (
                task.get("supported")
                and task.get("source") == "dense"
                and str(task.get("target")) == token
                and visual_inputs == ["0-2"]
            ):
                candidates.append(task)
        if not candidates:
            raise ValueError(
                f"No released WSM RGB estimator can self-preprocess modality {token!r}"
            )
        candidates.sort(
            key=lambda task: (
                not bool(task.get("official_recommended")),
                int(task.get("index", 1_000_000)),
            )
        )
        return candidates[0]
    @staticmethod
    def _config_for_task(task: dict[str, Any]) -> Path:
        source = str(task.get("source") or "dense")
        return MODALITY_CONFIGS.get(source, MODALITY_CONFIGS["dense"])
    @staticmethod
    def _red_report(token: str, message: str) -> dict[str, Any]:
        return {
            "valid": False,
            "accepted": True,
            "runnable": False,
            "severity": "red",
            "summary": "Input retained, but it cannot be mapped safely",
            "token": token,
            "warnings": [],
            "red_reasons": [message],
            "ranges": [],
            "contract": None,
        }
    @staticmethod
    def _demo_latent_path(value: str) -> Path:
        root = (WEBUI_ROOT / "demo_assets/input_latents").resolve()
        candidate = (WEBUI_ROOT / "demo_assets" / str(value)).resolve()
        if candidate.suffix != ".pt" or not candidate.is_relative_to(root):
            raise ValueError("Paper Demo latent asset is outside the published input_latents directory")
        if not candidate.is_file():
            raise FileNotFoundError(f"Paper Demo latent asset is missing: {value}")
        return candidate

    def handle_validate_input(self, data: dict[str, Any]) -> dict[str, Any]:
        token = str(data.get("token") or "")
        try:
            catalog = self._get_catalog()
            task = resolve_task(
                catalog, str(data.get("task_id") or ""),
                str(data.get("checkpoint") or ""),
            )
            details = {
                str(item.get("token")): str(item.get("kind") or "wsg")
                for item in task.get("input_details", [])
            }
            if token not in details:
                return self._red_report(token, f"The selected task does not request input {token!r}")
            latent_asset = str(data.get("latent_asset") or "")
            if latent_asset:
                self._demo_latent_path(latent_asset)
                return {
                    "valid": True, "accepted": True, "runnable": True, "severity": "green",
                    "summary": "Exact cached Paper Demo latent; the displayed PNG is preview-only",
                    "token": token, "warnings": [], "red_reasons": [],
                    "contract": {"token": token, "kind": "paper_demo_latent"},
                }
            filename = Path(str(data.get("filename") or "input"))
            suffix = filename.suffix.lower()
            configured_kind = details[token]
            self_preprocess = bool(data.get("self_preprocess", False))
            if self_preprocess:
                if configured_kind != "wsg":
                    return self._red_report(
                        token,
                        "WSM Self-Preprocessor is only available for non-RGB WSG inputs.",
                    )
                preprocessor = self._self_preprocessor_for_token(catalog, token)
                if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                    return self._red_report(
                        token,
                        "WSM Self-Preprocessor expects an RGB PNG, JPEG, or WebP image.",
                    )
                with tempfile.TemporaryDirectory(prefix="wsm_rgb_preprocess_check_") as temp:
                    path = Path(temp) / f"input{suffix}"
                    self._decode_data_url(str(data.get("data_url") or ""), path)
                    report = dict(validate_rgb_image(path, "0-2"))
                report["summary"] = (
                    f"RGB image ready for WSM Self-Preprocessor "
                    f"{preprocessor['head']} → {token}"
                )
                report["self_preprocessor"] = {
                    "task_id": preprocessor["id"],
                    "head": preprocessor["head"],
                    "target": token,
                }
                return report
            if configured_kind == "audio":
                kind, suffix = "audio", ".wav"
            elif suffix in {".wsg", ".npz", ".npy"}:
                kind = "wsg"
            elif configured_kind == "image":
                kind = "image"
                suffix = suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
            else:
                return self._red_report(
                    token,
                    "This non-RGB modality requires raw .wsg, .npz, or .npy data. "
                    "A PNG/JPG Reader visualization is display-only and must not be used as the condition.",
                )
            with tempfile.TemporaryDirectory(prefix="wsm_input_check_") as temp:
                path = Path(temp) / f"input{suffix}"
                self._decode_data_url(
                    str(data.get("data_url") or ""), path,
                    max_bytes=512 * 1024 * 1024 if kind == "wsg" else 32 * 1024 * 1024,
                )
                if kind == "audio":
                    return validate_wav_audio(path, token)
                if kind == "image":
                    return validate_rgb_image(path, token)
                validation_token = "0-2" if token in RGB_INPUT_TOKENS else token
                report = inspect_tensor_input(
                    path, validation_token, self._config_for_task(task)
                )
                if report.get("contract") and bool(data.get("generate_preview", True)):
                    report = dict(report)
                    report["preview_data_url"] = self._tensor_preview_data_url(
                        path, report["contract"]
                    )
                    report["preview_label"] = "Quick pseudo-RGB preview · display only"
                return report
        except (InputValidationError, OSError, ValueError, TypeError) as exc:
            return self._red_report(token, str(exc))
    def _tensor_preview_data_url(self, path: Path, contract: dict[str, Any]) -> str:
        array = canonical_array_from_contract(path, contract).astype(np.float32, copy=False)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 2:
            array = np.concatenate([array, np.zeros_like(array[..., :1])], axis=-1)
        else:
            array = array[..., :3]
        preview = np.zeros_like(array, dtype=np.float32)
        valid_any = np.zeros(array.shape[:2], dtype=bool)
        for channel in range(3):
            plane = array[..., channel]
            valid = np.isfinite(plane) & (plane != -1.0)
            valid_any |= valid
            values = plane[valid]
            if values.size:
                low, high = np.percentile(values, (2.0, 98.0))
                if high <= low:
                    high = low + 1.0
                preview[..., channel] = np.clip((plane - low) / (high - low), 0.0, 1.0)
        preview[~valid_any] = 0.0
        image = Image.fromarray(np.rint(preview * 255.0).astype(np.uint8), mode="RGB")
        image.thumbnail((480, 480), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/png;base64," + encoded
    def _decode_data_url(
        self, value: str, destination: Path, max_bytes: int = 32 * 1024 * 1024
    ) -> None:
        if not value.startswith("data:") or "," not in value:
            raise ValueError("Uploaded input is not a valid data URL")
        header, encoded = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Uploaded input must use base64 encoding")
        payload = base64.b64decode(encoded, validate=True)
        if len(payload) > max_bytes:
            raise ValueError(
                f"Uploaded input exceeds the {max_bytes // (1024 * 1024)} MB limit"
            )
        destination.write_bytes(payload)
    def _append_job_log(self, job_id: str, message: str) -> None:
        lines = str(message).replace("\r", "").splitlines()
        if not lines:
            return
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for line in lines:
                if line.strip():
                    job["logs"].append(line[:4000])
            job["updated_at"] = time.time()
    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(changes)
                job["updated_at"] = time.time()
    def _job_snapshot(self, job_id: str) -> dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown inference job: {job_id}")
            snapshot = {
                "job_id": job_id,
                "status": job["status"],
                "log": "\n".join(job["logs"]),
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
            }
            if job.get("error"):
                snapshot["error_message"] = job["error"]
            if job.get("result"):
                snapshot.update(job["result"])
            return snapshot
    def _prune_finished_jobs(self) -> None:
        with self._jobs_lock:
            finished = sorted(
                (
                    (job_id, job)
                    for job_id, job in self._jobs.items()
                    if job["status"] in {"completed", "failed"}
                ),
                key=lambda item: item[1]["updated_at"],
            )
            while len(self._jobs) >= 4 and finished:
                job_id, _ = finished.pop(0)
                self._jobs.pop(job_id, None)
    def handle_start_inference(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Another WSM inference request is already running")
        self._prune_finished_jobs()
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._jobs_lock:
            self._jobs[job_id] = {
                "status": "queued",
                "logs": deque(["[ControlWorld] WSM inference job queued"], maxlen=600),
                "created_at": now,
                "updated_at": now,
                "error": "",
                "result": None,
            }
        try:
            thread = threading.Thread(
                target=self._run_inference_job,
                args=(job_id, dict(data)),
                name=f"wsm-inference-{job_id[:8]}",
                daemon=True,
            )
            thread.start()
        except Exception:
            with self._jobs_lock:
                self._jobs.pop(job_id, None)
            self._inference_lock.release()
            raise
        return self._job_snapshot(job_id)
    def handle_get_inference_status(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._job_snapshot(str(data.get("job_id") or ""))
    def _stream_worker(self, job_id: str, process: subprocess.Popen, timeout: int) -> int:
        line_queue: queue.Queue[str | None] = queue.Queue()
        def read_output() -> None:
            try:
                if process.stdout is not None:
                    for line in iter(process.stdout.readline, ""):
                        line_queue.put(line)
            finally:
                line_queue.put(None)
        reader = threading.Thread(target=read_output, name=f"wsm-log-{job_id[:8]}", daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        stream_closed = False
        no_line = object()
        try:
            while True:
                if time.monotonic() >= deadline and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise TimeoutError(f"WSM inference exceeded the {timeout}s timeout")
                try:
                    item = line_queue.get(timeout=0.25)
                except queue.Empty:
                    item = no_line
                if item is None:
                    stream_closed = True
                elif item is not no_line:
                    self._append_job_log(job_id, str(item).rstrip("\n"))
                if process.poll() is not None and stream_closed and line_queue.empty():
                    break
            return int(process.wait(timeout=2))
        finally:
            reader.join(timeout=1)
    def _execute_worker(
        self, job_id: str, request: dict[str, Any], output_path: Path, stage_label: str,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.parent / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        worker = PROJECT_ROOT / (
            "scripts/infer_sparse_structure_head.py"
            if request.get("worker") == "sparse_adapter"
            else "scripts/infer_TunnelingNet_Flux2_4B_wsg.py"
        )
        default_python = Path.home() / "miniforge3/envs/WSM/bin/python"
        python = os.environ.get(
            "WSM_WEBUI_PYTHON",
            str(default_python) if default_python.exists() else sys.executable,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        python_path = [str(PROJECT_ROOT)]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path)
        if "WSM_WEBUI_GPU" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["WSM_WEBUI_GPU"]
        else:
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        command = [python, "-u", str(worker), "--request", str(request_path), "--output", str(output_path)]
        self._append_job_log(job_id, f"[ControlWorld] {stage_label} with {python}")
        self._update_job(job_id, status="running")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        returncode = self._stream_worker(
            job_id, process, timeout=int(os.environ.get("WSM_WEBUI_TIMEOUT", "600")),
        )
        if returncode != 0:
            raise RuntimeError(f"{stage_label} exited with code {returncode}")
        if not output_path.exists():
            raise RuntimeError(f"{stage_label} produced no output image")

    def _run_inference_job(self, job_id: str, data: dict[str, Any]) -> None:
        self._update_job(job_id, status="preparing")
        self._append_job_log(job_id, "[ControlWorld] Validating task, checkpoint, and uploaded inputs")
        try:
            catalog = self._get_catalog()
            task = resolve_task(catalog, str(data.get("task_id") or ""), str(data.get("checkpoint") or ""))
            self._append_job_log(job_id, f"[ControlWorld] Selected {task['source']}:{task['head']} → {task['target']}")
            composite = bool(data.get("composite", False))
            supplied = {str(item.get("token")): item for item in data.get("inputs", [])}
            required_inputs = [
                token for token in task["inputs"]
                if token != "caption"
            ]
            if composite and (
                len(required_inputs) != 1
                or required_inputs[0] not in RGB_INPUT_TOKENS
            ):
                raise ValueError(
                    "A Chain of Visual route requires a second-stage Head with exactly one RGB input"
                )
            missing = [
                token for token in required_inputs
                if token not in supplied and not composite
            ]
            if missing:
                raise ValueError(f"Missing required inputs: {', '.join(missing)}")
            input_kinds = {
                str(item.get("token")): str(item.get("kind") or "wsg")
                for item in task.get("input_details", [])
            }
            with tempfile.TemporaryDirectory(prefix="wsm_webui_") as temp:
                temp_dir = Path(temp)
                worker_inputs = []
                for index, token in enumerate([] if composite else required_inputs):
                    supplied_input = supplied[token]
                    configured_kind = input_kinds.get(token, "wsg")
                    self_preprocess = bool(supplied_input.get("self_preprocess", False))
                    if self_preprocess:
                        if configured_kind != "wsg":
                            raise ValueError(
                                f"Input {token} is not a non-RGB WSG modality and cannot use "
                                "WSM Self-Preprocessor"
                            )
                        preprocessor = self._self_preprocessor_for_token(catalog, token)
                        filename = Path(str(supplied_input.get("filename") or "input.png"))
                        suffix = filename.suffix.lower()
                        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                            raise ValueError(
                                f"WSM Self-Preprocessor for {token} expects an RGB PNG, JPEG, or WebP image"
                            )
                        rgb_path = temp_dir / f"self_preprocessor_input_{index:02d}{suffix}"
                        self._decode_data_url(
                            str(supplied_input.get("data_url") or ""), rgb_path,
                        )
                        rgb_report = validate_rgb_image(rgb_path, "0-2")
                        if not rgb_report.get("runnable", False):
                            reasons = rgb_report.get("red_reasons") or ["invalid RGB image"]
                            raise ValueError(
                                f"Self-Preprocessor RGB input for {token} cannot run: "
                                + "; ".join(reasons)
                            )
                        self._append_job_log(
                            job_id,
                            f"[ControlWorld] Preparing RGB for WSM Self-Preprocessor "
                            f"{preprocessor['head']} → {token}",
                        )
                        worker_inputs.append({
                            "token": token,
                            "kind": "image",
                            "path": str(rgb_path),
                            "self_preprocess": True,
                        })
                        continue
                    latent_asset = str(supplied_input.get("latent_asset") or "")
                    filename = Path(str(supplied_input.get("filename") or "input"))
                    suffix = filename.suffix.lower()
                    if configured_kind == "audio":
                        kind, suffix = "audio", ".wav"
                    elif suffix in {".wsg", ".npz", ".npy"}:
                        kind = "wsg"
                    elif configured_kind == "image":
                        kind = "image"
                        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                            suffix = ".png"
                    else:
                        raise ValueError(
                            f"Input {token} is a non-RGB WSG modality; upload "
                            "a .wsg, .npz, or .npy World Snapshot tensor"
                        )
                    self._append_job_log(
                        job_id,
                        f"[ControlWorld] Preparing {'LATENT CACHE' if latent_asset else kind.upper()} input {token}",
                    )
                    path = temp_dir / f"input_{index:02d}{suffix}"
                    self._decode_data_url(
                        str(supplied_input.get("data_url") or ""),
                        path,
                        max_bytes=512 * 1024 * 1024 if kind == "wsg" else 32 * 1024 * 1024,
                    )
                    if latent_asset:
                        self._demo_latent_path(latent_asset)
                        report = {
                            "severity": "green", "runnable": True,
                            "summary": "exact cached Paper Demo latent",
                            "warnings": [], "red_reasons": [], "contract": None,
                        }
                    elif kind == "audio":
                        report = validate_wav_audio(path, token)
                    elif kind == "image":
                        report = validate_rgb_image(path, token)
                    else:
                        validation_token = "0-2" if token in RGB_INPUT_TOKENS else token
                        report = inspect_tensor_input(
                            path, validation_token, self._config_for_task(task)
                        )
                    severity = str(report.get("severity") or "red").upper()
                    self._append_job_log(job_id, f"[Input check] {severity} {token}: {report.get('summary', '')}")
                    for message in report.get("warnings", []):
                        self._append_job_log(job_id, f"[Input check] ORANGE {token}: {message}")
                    for message in report.get("red_reasons", []):
                        self._append_job_log(job_id, f"[Input check] RED {token}: {message}")
                    if not report.get("runnable", False):
                        reasons = report.get("red_reasons") or ["input cannot be mapped"]
                        raise ValueError(f"Input {token} cannot run: {'; '.join(reasons)}")
                    worker_input = {"token": token, "kind": kind, "path": str(path)}
                    if report.get("contract"):
                        worker_input["input_contract"] = report["contract"]
                    if latent_asset:
                        worker_input["latent_path"] = str(self._demo_latent_path(latent_asset))
                    worker_inputs.append(worker_input)
                request = {
                    "mode": "text_to_modality" if composite else "channel_head",
                    "checkpoint": str(data.get("checkpoint") or task["default_checkpoint"]),
                    "source": task["source"],
                    "head_index": task["index"],
                    "adapter_task": task.get("adapter_task") or "",
                    "worker": (
                        "sparse_adapter"
                        if task["source"] == "sparse"
                        and task.get("adapter_task")
                        != "rgb_to_semantic_mask_audio_region_selector"
                        else "tunnelingnet"
                    ),
                    "inputs": worker_inputs,
                    "prompt": str(data.get("prompt") or ""),
                    "exact_prompt_only": bool(data.get("exact_prompt_only", False)),
                    "paper_demo": bool(data.get("paper_demo", False)) and not composite,
                    "paper_output_view": (
                        str(data.get("paper_output_view") or "native")
                        if bool(data.get("paper_demo", False)) and not composite else ""
                    ),
                    "target": task["target"],
                    "steps": int(data.get("steps") or 10),
                    "seed": int(data.get("seed") or 20260812),
                    "size": int(data.get("size") or 256),
                }
                output_path = temp_dir / "result.png"
                self._execute_worker(
                    job_id, request, output_path,
                    "Starting Chain of Visual" if composite else "Starting WSM inference",
                )
                manifest_path = temp_dir / "result_manifest.json"
                manifest = {"outputs": []}
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                output_records = []
                temp_root = temp_dir.resolve()
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                }
                for item in manifest.get("outputs", []):
                    path = Path(str(item.get("path") or "")).resolve()
                    if not path.is_file() or not path.is_relative_to(temp_root):
                        continue
                    mime = mime_types.get(path.suffix.lower())
                    if mime is None:
                        continue
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                    output_records.append({
                        "label": str(item.get("label") or path.name),
                        "filename": str(item.get("filename") or path.name),
                        "image": f"data:{mime};base64,{encoded}",
                        "selected": bool(item.get("selected", False)),
                    })
                if not output_records:
                    encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
                    output_records = [{
                        "label": "Inference output",
                        "filename": output_path.name,
                        "image": f"data:image/png;base64,{encoded}",
                        "selected": True,
                    }]
                selected_output = next(
                    (item for item in output_records if item["selected"]),
                    output_records[0],
                )
                result = {
                    "image": selected_output["image"],
                    "outputs": output_records,
                    "task": task,
                    "checkpoint": request["checkpoint"],
                    "composite": composite,
                }
                self._append_job_log(job_id, "[ControlWorld] Inference result is ready")
                self._update_job(job_id, status="completed", result=result)
        except Exception as exc:
            self._append_job_log(job_id, f"[ControlWorld] ERROR: {type(exc).__name__}: {exc}")
            details = traceback.format_exc().strip()
            if details:
                self._append_job_log(job_id, details)
            self._update_job(job_id, status="failed", error=str(exc))
        finally:
            self._inference_lock.release()
    def handle_run_inference(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compatibility entry point for clients that still expect one blocking response."""
        started = self.handle_start_inference(data)
        job_id = started["job_id"]
        while True:
            snapshot = self._job_snapshot(job_id)
            if snapshot["status"] == "completed":
                return {
                    "image": snapshot["image"],
                    "outputs": snapshot.get("outputs", []),
                    "task": snapshot["task"],
                    "checkpoint": snapshot["checkpoint"],
                    "worker_log": snapshot["log"],
                }
            if snapshot["status"] == "failed":
                raise RuntimeError(snapshot.get("error_message") or "WSM inference failed")
            time.sleep(0.25)
