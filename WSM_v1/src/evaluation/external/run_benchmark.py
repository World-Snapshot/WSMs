#!/usr/bin/env python3
"""One-stop preparation, inference, evaluation, and reporting for WSM baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
REGISTRY_PATH = HERE / "official_registry.json"
EVALUATOR = HERE / "evaluate_predictions.py"
DIFFUSERS_ADAPTER = HERE / "adapters" / "diffusers_adapter.py"
MODUS_ADAPTER = HERE / "adapters" / "modus_adapter.py"
VGGT_ADAPTER = HERE / "adapters" / "vggt_adapter.py"
SELECTOR = HERE / "select_heldout24.py"
EXAMPLE_CONFIG = HERE / "benchmark_heldout24.example.json"
TASKS = frozenset({
    "conditional_rgb", "semantic_estimation", "any_to_any", "depth_estimation"
})
PATH_FIELDS = frozenset({
    "input_path", "condition_path", "target_path", "mask_path", "reference_path"
})


class BenchmarkError(RuntimeError):
    """Raised when continuing would make the benchmark incomplete or unfair."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BenchmarkError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def parse_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def config_template(destination: Path | None = None) -> dict[str, Any]:
    """Return the public template with paths valid at its destination."""
    payload = read_json(EXAMPLE_CONFIG)
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Expected a JSON object in {EXAMPLE_CONFIG}")
    if destination is not None:
        base = destination.parent.resolve()
        payload["workspace"] = os.path.relpath(
            PROJECT_ROOT / "src/evaluation_output/external/wsm_external_heldout24",
            base,
        )
        payload["hf_home"] = os.path.relpath(
            PROJECT_ROOT / "src/evaluation_output/external/_shared_hf_cache",
            base,
        )
    return payload


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        self.base = self.config_path.parent
        self.config = read_json(self.config_path)
        if int(self.config.get("schema_version", 0)) != 1:
            raise BenchmarkError("Experiment schema_version must be 1")
        self.registry = read_json(Path(args.registry).expanduser().resolve())
        self.workspace = resolve(self.config.get("workspace", PROJECT_ROOT / "src" / "evaluation_output" / "external" / "wsm_external_heldout24"), self.base)
        self.samples = self.workspace / "samples.jsonl"
        self.repositories = self.workspace / "repositories"
        self.environments = self.workspace / "environments"
        self.models = self.workspace / "models"
        self.outputs = self.workspace / "outputs"
        self.predictions = self.workspace / "predictions"
        self.contract = self.workspace / "evaluation_contract.json"
        self.audit_lock = self.workspace / "reproducibility_lock.json"
        self.state_path = self.workspace / "run_state.json"
        self.hf_home = resolve(
            str(self.config.get("hf_home") or self.workspace / "hf_cache"), self.base
        )
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        self.selected_methods = parse_set(args.methods)
        self.selected_tasks = parse_set(args.tasks)
        if self.selected_tasks - TASKS:
            raise BenchmarkError(
                f"Unknown tasks: {sorted(self.selected_tasks - TASKS)}"
            )
        self.accepted_licenses = set(args.accept_license)
        self.lock = (
            read_json(self.audit_lock) if self.audit_lock.is_file()
            else {"schema_version": 1, "methods": {}, "datasets": {}}
        )

    def run(self, command: Sequence[str], cwd: Path | None = None) -> None:
        print("$ " + shlex.join(str(item) for item in command), flush=True)
        if self.args.dry_run:
            return
        subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd) if cwd else None,
            check=True
        )

    def save_stage(self, name: str) -> None:
        if self.args.dry_run:
            return
        state = read_json(self.state_path) if self.state_path.is_file() else {
            "schema_version": 1, "stages": {}
        }
        state["config"] = str(self.config_path)
        state["config_sha256"] = file_sha256(self.config_path)
        state["stages"][name] = "complete"
        atomic_json(self.state_path, state)
        atomic_json(self.audit_lock, self.lock)

    def method_specs(self) -> list[dict[str, Any]]:
        output = []
        registry = self.registry["methods"]
        for raw in self.config.get("methods") or []:
            method_id = str(raw.get("id") or "")
            if method_id not in registry:
                raise BenchmarkError(f"Unknown method: {method_id}")
            spec = merge_dict(registry[method_id], raw)
            spec["id"] = method_id
            if not bool(spec.get("enabled", True)):
                continue
            if self.selected_methods and method_id not in self.selected_methods:
                continue
            method_tasks = set(spec.get("tasks") or [])
            if self.selected_tasks and not (method_tasks & self.selected_tasks):
                continue
            output.append(spec)
        return output

    def dataset_specs(self) -> list[dict[str, Any]]:
        output = []
        registry = self.registry["datasets"]
        for raw in self.config.get("datasets") or []:
            dataset_id = str(raw.get("id") or "")
            if dataset_id not in registry:
                raise BenchmarkError(f"Unknown dataset: {dataset_id}")
            spec = merge_dict(registry[dataset_id], raw)
            spec["id"] = dataset_id
            if not bool(spec.get("enabled", True)):
                continue
            dataset_tasks = set(spec.get("tasks") or [])
            if self.selected_tasks and not (dataset_tasks & self.selected_tasks):
                continue
            output.append(spec)
        return output

    def doctor(self) -> None:
        methods, datasets = self.method_specs(), self.dataset_specs()
        blockers, warnings = [], []
        if not methods:
            blockers.append("No enabled methods")
        if not datasets:
            blockers.append("No enabled datasets")
        for spec in methods:
            command = (spec.get("inference") or {}).get("command") or []
            if spec.get("automation") == "unverified":
                blockers.append(f"{spec['id']}: {spec.get('disabled_reason')}")
            elif (
                not spec.get("adapter")
                and spec.get("inference_regime") != "reported_only"
                and not command
            ):
                blockers.append(f"{spec['id']}: set a verified inference.command")
            for field in ("fairness_warning", "resource_warning"):
                if spec.get(field):
                    warnings.append(f"{spec['id']}: {spec[field]}")
        free = shutil.disk_usage(self.workspace.parent).free / 2**30
        payload = {
            "workspace": str(self.workspace),
            "samples_per_task": int(self.config.get("samples_per_task", 24)),
            "shared_samples_across_methods": True,
            "free_disk_gib": round(free, 2),
            "methods": [item["id"] for item in methods],
            "datasets": [item["id"] for item in datasets],
            "tools": {name: shutil.which(name) for name in ("git", "curl", "wget")},
            "warnings": warnings,
            "inference_blockers": blockers
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        self.save_stage("doctor")

    def require_license(self, spec: Mapping[str, Any]) -> None:
        license_id = str(spec.get("license_ack") or "")
        if not license_id:
            return
        if bool(spec.get("license_accepted")) or license_id in self.accepted_licenses:
            return
        raise BenchmarkError(
            f"{spec['id']}: review {spec.get('source', 'dataset terms')} and pass "
            f"--accept-license {license_id}, or set license_accepted=true"
        )

    def download(self, url: str, destination: Path) -> None:
        if destination.is_file():
            print(f"[reuse] {destination}")
            return
        curl = shutil.which("curl")
        if not curl:
            raise BenchmarkError("curl is required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.run([
            curl, "-L", "--fail", "--retry", "5", "--continue-at", "-",
            "--output", str(destination), url
        ])

    def prepare_data(self) -> None:
        frozen: list[dict[str, Any]] = []
        for spec in self.dataset_specs():
            self.require_license(spec)
            dataset_dir = resolve(
                spec.get("data_root", self.workspace / "datasets" / spec["id"]),
                self.base
            )
            if spec.get("builder") == "existing_manifest":
                source = resolve(str(spec.get("manifest") or ""), self.base)
                if not source.is_file() and not self.args.dry_run:
                    raise BenchmarkError(f"Missing held-out manifest: {source}")
                rows = [] if self.args.dry_run else read_jsonl(source)
                expected = int(self.config.get("samples_per_task", 24))
                wanted_tasks = self.selected_tasks or {
                    str(row.get("task") or "") for row in rows
                }
                counts = {
                    task: sum(str(row.get("task") or "") == task for row in rows)
                    for task in wanted_tasks
                }
                if bool(spec.get("auto_select")) and not self.args.dry_run:
                    selected_path = self.workspace / "selection" / "heldout24.jsonl"
                    command = [
                        sys.executable, str(SELECTOR),
                        "--candidates", str(source),
                        "--output", str(selected_path),
                        "--count-per-task", str(expected),
                        "--tasks", ",".join(sorted(wanted_tasks)),
                        "--strata", ",".join(
                            self.config.get("selection_policy", {}).get(
                                "strata_required", []
                            )
                        ),
                        "--seed", "20260829"
                    ]
                    if spec.get("created_after"):
                        command += [
                            "--min-created-date", str(spec["created_after"]),
                            "--exclude-possible-overlap"
                        ]
                    if spec.get("sealed_before_evaluation"):
                        command.append("--require-sealed")
                    self.run(command)
                    source = selected_path
                    rows = read_jsonl(source)
                for row in rows:
                    normal = dict(row)
                    for field in PATH_FIELDS:
                        if normal.get(field):
                            normal[field] = str(resolve(normal[field], source.parent))
                    frozen.append(normal)
                self.lock["datasets"][spec["id"]] = {
                    "source_manifest": str(source),
                    "source_sha256": file_sha256(source) if source.is_file() else "",
                    "created_after": spec.get("created_after", ""),
                    "sealed_before_evaluation": spec.get(
                        "sealed_before_evaluation", False
                    )
                }
                continue
            for relative in spec.get("manual_required") or []:
                required = dataset_dir / relative
                if not required.exists() and not self.args.dry_run:
                    raise BenchmarkError(f"Licensed dataset missing: {required}")
            for artifact in spec.get("artifacts") or []:
                archive = dataset_dir / artifact["filename"]
                self.download(str(artifact["url"]), archive)
                if artifact.get("extract") and not self.args.dry_run:
                    marker = dataset_dir / f".{archive.name}.extracted"
                    if not marker.is_file():
                        shutil.unpack_archive(str(archive), str(dataset_dir))
                        marker.touch()
            if spec.get("recursive_url"):
                wget = shutil.which("wget")
                if not wget:
                    raise BenchmarkError("wget is required for the official mirror")
                self.run([
                    wget, "-r", "-np", "-nH", "--cut-dirs=4", "-R",
                    "index.html*", "-P", str(dataset_dir),
                    str(spec["recursive_url"])
                ])
            raise BenchmarkError(
                f"{spec['id']}: data downloaded/verified, but a frozen 24-sample "
                "canonical manifest is still required. Use select_heldout24.py and "
                "enable external_manifest for paper evaluation."
            )
        if self.args.dry_run:
            print(f"[dry-run] would freeze manifest at {self.samples}")
            return
        expected = int(self.config.get("samples_per_task", 24))
        seen: set[str] = set()
        counts: dict[str, int] = {}
        for row in frozen:
            sample_id = str(row.get("sample_id") or "")
            task = str(row.get("task") or "")
            if not sample_id or sample_id in seen:
                raise BenchmarkError(f"Duplicate/empty sample_id: {sample_id!r}")
            if task not in TASKS:
                raise BenchmarkError(f"{sample_id}: invalid task {task!r}")
            seen.add(sample_id)
            counts[task] = counts.get(task, 0) + 1
            for required in ("target_path",):
                path = resolve(str(row.get(required) or ""), self.base)
                if not path.is_file():
                    raise BenchmarkError(f"{sample_id}: missing {required}: {path}")
        selected = self.selected_tasks or set(counts)
        wrong = {task: counts.get(task, 0) for task in selected if counts.get(task, 0) != expected}
        if wrong:
            raise BenchmarkError(
                f"Paper protocol requires exactly {expected} shared samples per task; "
                f"got {wrong}. Freeze them with select_heldout24.py."
            )
        frozen.sort(key=lambda item: (str(item["task"]), str(item["sample_id"])))
        atomic_jsonl(self.samples, frozen)
        self.lock["frozen_manifest"] = {
            "path": str(self.samples),
            "samples_per_task": expected,
            "counts": counts,
            "sha256": file_sha256(self.samples),
            "frozen_before_inference": True
        }
        print(f"Frozen {len(frozen)} samples: {self.samples}")
        self.save_stage("prepare-data")

    def method_python(self, method_id: str) -> Path:
        return self.environments / method_id / "bin" / "python"

    def prepare_methods(self) -> None:
        for spec in self.method_specs():
            method_id = spec["id"]
            if spec.get("inference_regime") == "reported_only":
                continue
            repo = self.repositories / method_id
            repo_url = str(spec.get("repo_url") or "")
            if repo_url and not repo.exists():
                self.run(["git", "clone", repo_url, str(repo)])
            if repo_url and spec.get("revision"):
                self.run(["git", "fetch", "--all", "--tags"], cwd=repo)
                self.run(["git", "checkout", str(spec["revision"])], cwd=repo)
            environment = dict(spec.get("environment") or {})
            python = (
                Path(str(spec["python"])).expanduser().resolve()
                if spec.get("python") else (
                    self.method_python(method_id)
                    if environment else Path(sys.executable)
                )
            )
            if environment and not spec.get("python") and not python.is_file():
                self.run([sys.executable, "-m", "venv", str(python.parents[1])])
            requirements = [str(item) for item in environment.get("requirements") or []]
            if requirements and not spec.get("python"):
                self.run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
                self.run([str(python), "-m", "pip", "install", *requirements])
            requirements_file = str(environment.get("requirements_file") or "")
            if requirements_file and not spec.get("python"):
                self.run([
                    str(python), "-m", "pip", "install", "-r",
                    str(repo / requirements_file)
                ])
            downloaded = {}
            for repo_id in ([] if spec.get("skip_hf_prepare") else spec.get("hf_models") or []):
                destination = self.models / method_id / str(repo_id).replace("/", "--")
                downloaded[str(repo_id)] = str(destination)
                if not destination.exists():
                    code = (
                        "from huggingface_hub import snapshot_download;"
                        "import sys;snapshot_download(repo_id=sys.argv[1],"
                        "local_dir=sys.argv[2])"
                    )
                    self.run([str(python), "-c", code, str(repo_id), str(destination)])
            commit = ""
            if repo.exists() and not self.args.dry_run:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip()
            self.lock["methods"][method_id] = {
                "repo_url": repo_url,
                "requested_revision": spec.get("revision", ""),
                "resolved_commit": commit,
                "models": downloaded,
                "inference_regime": spec.get("inference_regime", "")
            }
        self.save_stage("prepare-methods")

    def infer(self) -> None:
        if not self.samples.is_file() and not self.args.dry_run:
            raise BenchmarkError("Run prepare-data first")
        for spec in self.method_specs():
            method_id = spec["id"]
            if spec.get("inference_regime") == "reported_only":
                continue
            manifest = self.predictions / f"{method_id}.jsonl"
            output = self.outputs / method_id
            if spec.get("reuse_existing_predictions") and manifest.is_file():
                print(f"Reuse existing predictions: {manifest}", flush=True)
                continue
            method_python = (
                Path(str(spec["python"])).expanduser().resolve()
                if spec.get("python") else (
                    self.method_python(method_id)
                    if spec.get("environment") else Path(sys.executable)
                )
            )
            context = {
                "python": str(method_python),
                "project_root": str(PROJECT_ROOT),
                "repo": str(self.repositories / method_id),
                "models": str(self.models / method_id),
                "samples": str(self.samples),
                "output_dir": str(output),
                "prediction_manifest": str(manifest),
                "checkpoint": str(spec.get("checkpoint") or ""),
                "device": self.args.device
            }
            if spec.get("adapter") == "diffusers":
                command = [
                    context["python"], str(DIFFUSERS_ADAPTER),
                    "--method", str(spec["adapter_method"]),
                    "--samples", context["samples"],
                    "--output-dir", context["output_dir"],
                    "--prediction-manifest", context["prediction_manifest"],
                    "--model-root", context["models"],
                    "--device", context["device"]
                ]
            elif spec.get("adapter") == "modus":
                command = [
                    context["python"], str(MODUS_ADAPTER),
                    "--repo", context["repo"],
                    "--model-root", context["models"],
                    "--samples", context["samples"],
                    "--output-dir", context["output_dir"],
                    "--prediction-manifest", context["prediction_manifest"]
                ]
            elif spec.get("adapter") == "vggt":
                command = [
                    context["python"], str(VGGT_ADAPTER),
                    "--repo", context["repo"],
                    "--samples", context["samples"],
                    "--output-dir", context["output_dir"],
                    "--prediction-manifest", context["prediction_manifest"],
                    "--device", context["device"]
                ]
            elif spec.get("adapter") == "scaling_diffusion_perception":
                model_root = Path(context["models"])
                command = [
                    context["python"],
                    str(ADAPTERS / "scaling_diffusion_perception_adapter.py"),
                    "--repo", context["repo"],
                    "--checkpoint", str(
                        model_root / "zeeshanp--scaling_diffusion_perception"
                        / "dit_moe_generalist.pt"
                    ),
                    "--sd-root", str(
                        model_root / "LanguageMachines--stable-diffusion-2-1-base"
                    ),
                    "--samples", context["samples"],
                    "--output-dir", context["output_dir"],
                    "--prediction-manifest", context["prediction_manifest"],
                    "--device", context["device"],
                ]
            else:
                raw = (spec.get("inference") or {}).get("command") or []
                if not raw:
                    raise BenchmarkError(
                        f"{method_id}: set inference.command to an argv array that "
                        "writes the canonical prediction manifest"
                    )
                command = [str(item).format(**context) for item in raw]
            cwd = self.repositories / method_id
            self.run(command, cwd=cwd if cwd.exists() else PROJECT_ROOT)
            if not self.args.dry_run and not manifest.is_file():
                raise BenchmarkError(f"{method_id} did not write {manifest}")
        self.save_stage("infer")

    def build_contract(self) -> dict[str, Any]:
        methods = []
        for spec in self.method_specs():
            regime = str(spec.get("inference_regime") or "zero_shot_official")
            tasks = set(spec.get("tasks") or TASKS)
            if self.selected_tasks:
                tasks &= self.selected_tasks
            row = {
                "id": spec["id"],
                "display_name": spec.get("display_name", spec["id"]),
                "enabled": True,
                "tasks": sorted(tasks),
                "inference_regime": regime,
                "checkpoint": spec.get("checkpoint", ""),
                "checkpoints": spec.get("checkpoints", {}),
                "fairness_note": spec.get("fairness_note", "")
            }
            if regime == "reported_only":
                row["reported_metrics"] = spec.get("reported_metrics", {})
                row["reported_source"] = spec.get("reported_source", "")
            else:
                row["prediction_manifest"] = str(
                    self.predictions / f"{spec['id']}.jsonl"
                )
            methods.append(row)
        return {
            "schema_version": 1,
            "benchmark_name": self.config.get("benchmark_name", ""),
            "sample_manifest": str(self.samples),
            "workspace": str(self.workspace),
            "task_protocols": self.config.get("task_protocols", {}),
            "fairness_notes": self.config.get("fairness", {}),
            "methods": methods
        }

    def evaluate(self) -> None:
        contract = self.build_contract()
        if self.args.dry_run:
            print(json.dumps(contract, indent=2, ensure_ascii=False))
            return
        atomic_json(self.contract, contract)
        command = [sys.executable, str(EVALUATOR), "--config", str(self.contract)]
        if self.args.extended_evaluation:
            command.append("--extended-evaluation")
        if self.selected_methods:
            command += ["--methods", ",".join(sorted(self.selected_methods))]
        if self.selected_tasks:
            command += ["--tasks", ",".join(sorted(self.selected_tasks))]
        self.run(command)
        self.save_stage("evaluate")

    def report(self) -> None:
        tables = self.workspace / "tables"
        print(json.dumps({
            "method_metrics": str(tables / "method_metrics.csv"),
            "latex": str(tables / "method_metrics.tex"),
            "reported_only": str(tables / "reported_metrics_separate.csv"),
            "sample_metrics": str(tables / "sample_metrics.csv"),
            "audit": str(self.workspace / "audit.json"),
            "reproducibility_lock": str(self.audit_lock)
        }, indent=2, ensure_ascii=False))
        self.save_stage("report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "init", "doctor", "prepare-data", "prepare-methods",
            "infer", "evaluate", "report", "all"
        ]
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--methods", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--accept-license", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extended-evaluation",
        action="store_true",
        help=(
            "Compute the full diagnostic metric suite. The default evaluation "
            "profile reproduces only the metrics shown in paper Tables 6 and 7."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(args.config).expanduser().resolve()
    if args.stage == "init":
        if config.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite {config}; use --force")
        atomic_json(config, config_template(config))
        print(f"Wrote one-stop benchmark config: {config}")
        return
    if not config.is_file():
        raise SystemExit(f"Missing {config}; run init first")
    runner = Runner(args)
    stages = {
        "doctor": runner.doctor,
        "prepare-data": runner.prepare_data,
        "prepare-methods": runner.prepare_methods,
        "infer": runner.infer,
        "evaluate": runner.evaluate,
        "report": runner.report
    }
    if args.stage == "all":
        for name in (
            "doctor", "prepare-data", "prepare-methods", "infer", "evaluate", "report"
        ):
            print(f"\n=== {name} ===", flush=True)
            stages[name]()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
