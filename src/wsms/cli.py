"""The ``wsms`` command-line interface."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
from typing import Iterable

from .catalog import TaskCatalog, WSMTask, runtime_root
from .version import __version__


DEFAULT_MODEL_REPO = "World-Snapshot/WSM_v1"


def _add_catalog_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=("dense", "sparse", "custom"))
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="Input modality; repeat to preserve multi-input routing order",
    )
    parser.add_argument("--output", help="Output modality")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows; 0 means all")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wsms", description="World Snapshot Models")
    parser.add_argument("--version", action="version", version=f"wsms {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("list", help="List all top-level WSM capabilities")

    for name, help_text in (
        ("tunnel", "Discover task tunnels by their input-output relationship"),
        ("head", "Inspect qualified Dense, Sparse, and Custom Head implementations"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        children = command.add_subparsers(dest="action")
        listing = children.add_parser("list", help=f"List released WSM {name}s")
        _add_catalog_filters(listing)

    info = subcommands.add_parser(
        "info", help="Inspect package, runtime, model, and device information"
    )
    info.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    infer = subcommands.add_parser("infer", help="Run one released WSM task tunnel")
    infer.add_argument("--model", default=DEFAULT_MODEL_REPO, help="Model repo or local root")
    infer.add_argument("--base-model", help="FLUX.2 repo or local root")
    infer.add_argument(
        "--input",
        dest="input_files",
        action="append",
        default=[],
        metavar="TOKEN=PATH",
        help="Modality input; repeat in routing order for multi-input tunnels",
    )
    infer.add_argument("--target", help="Target modality (required unless --head is used)")
    infer.add_argument("--source", choices=("dense", "sparse", "custom"))
    infer.add_argument(
        "--head",
        help="Qualified implementation, e.g. dense:000003 (advanced)",
    )
    infer.add_argument("--prompt", default="", help="Text prompt when the tunnel uses caption")
    infer.add_argument("--steps", type=int, default=10, help="Inference steps")
    infer.add_argument("--seed", type=int, default=20260812)
    infer.add_argument("--size", type=int, default=256, help="Square output size")
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--output-dir", default="wsm-output")
    infer.add_argument("--cache-dir")
    infer.add_argument("--revision")
    infer.add_argument("--token", help="HF token; prefer 'hf auth login' or HF_TOKEN")
    infer.add_argument("--local-files-only", action="store_true")
    infer.add_argument("--exact-prompt-only", action="store_true")
    infer.add_argument("--paper-demo", action="store_true")
    infer.add_argument(
        "--categorical-refiner",
        default="auto",
        choices=("auto", "none"),
    )
    return parser


def _top_level() -> None:
    rows = (
        ("tunnel", "Discover every released ordered input -> output mapping"),
        ("head", "Inspect the qualified implementation behind each tunnel"),
        ("infer", "Run a task tunnel"),
        ("info", "Inspect package, runtime, model, and device information"),
    )
    print("WSM capabilities")
    print("COMMAND  DESCRIPTION")
    for command, description in rows:
        print(f"{command:<8} {description}")
    print()
    print("Use 'wsms tunnel list' for task mappings or 'wsms head list' for implementations.")


def _rows(tasks: Iterable[WSMTask], *, heads: bool) -> None:
    if heads:
        print(f"{'HEAD':<16} {'INPUTS':<34} {'OUTPUT':<20} {'TYPE'}")
        for task in tasks:
            inputs = ", ".join(task.inputs) if task.inputs else "∅"
            print(f"{task.head:<16} {inputs:<34} {task.output:<20} {task.task_type}")
    else:
        print(f"{'TASK ID':<58} {'INPUTS':<34} {'OUTPUT'}")
        for task in tasks:
            inputs = ", ".join(task.inputs) if task.inputs else "∅"
            print(f"{task.task_id:<58} {inputs:<34} {task.output}")


def _package_info() -> dict[str, object]:
    catalog = TaskCatalog.load()
    by_source = {
        source: len(catalog.list(source=source))
        for source in ("dense", "sparse", "custom")
    }
    packages = {}
    for distribution in (
        "torch",
        "diffusers",
        "transformers",
        "huggingface-hub",
        "peft",
    ):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        cuda = {"available": False, "error": str(exc)}
    return {
        "wsms_version": __version__,
        "runtime_root": str(runtime_root()),
        "released_tunnels": len(catalog),
        "families": by_source,
        "packages": packages,
        "cuda": cuda,
        "default_model_repo": DEFAULT_MODEL_REPO,
    }


def _show_info(as_json: bool) -> None:
    values = _package_info()
    if as_json:
        print(json.dumps(values, indent=2, ensure_ascii=False))
        return
    print(f"wsms {values['wsms_version']}")
    print(f"Runtime: {values['runtime_root']}")
    print(f"Released tunnels: {values['released_tunnels']:,}")
    families = values["families"]
    print(
        "Families: "
        + ", ".join(f"{name}={count:,}" for name, count in families.items())
    )
    print(f"Default model repository: {values['default_model_repo']}")
    packages = values["packages"]
    print("Packages: " + ", ".join(f"{name}={version}" for name, version in packages.items()))
    cuda = values["cuda"]
    print(
        f"CUDA: available={cuda.get('available')}, "
        f"devices={cuda.get('device_count', 0)}"
    )


def _parse_inputs(values: list[str]) -> tuple[list[str], dict[str, Path]]:
    tokens: list[str] = []
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --input {value!r}; expected TOKEN=PATH")
        token, path_text = value.split("=", 1)
        token = token.strip()
        if not token or not path_text.strip():
            raise ValueError(f"Invalid --input {value!r}; expected TOKEN=PATH")
        if token in paths:
            raise ValueError(f"Input modality {token!r} was provided more than once")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        tokens.append(token)
        paths[token] = path
    return tokens, paths


def _resolve_inference_task(
    catalog: TaskCatalog,
    *,
    tokens: list[str],
    target: str | None,
    prompt: str,
    source: str | None,
    head: str | None,
) -> WSMTask:
    if head:
        if target:
            raise ValueError("Use either --target or --head, not both")
        task = catalog.by_head(head, source=source)
        expected = tuple(token for token in task.inputs if token != "caption")
        if tuple(tokens) != expected:
            raise ValueError(
                f"{task.head} requires inputs in this order: {list(expected)}"
            )
        if "caption" in task.inputs and not prompt:
            raise ValueError(f"{task.head} requires --prompt")
        return task
    if not target:
        raise ValueError("--target is required unless --head is used")

    candidates = [
        task
        for task in catalog.list(source=source, output=target)
        if tuple(token for token in task.inputs if token != "caption") == tuple(tokens)
    ]
    caption_candidates = [
        task for task in candidates if ("caption" in task.inputs) == bool(prompt)
    ]
    if caption_candidates:
        candidates = caption_candidates
    if not candidates:
        route = "+".join(tokens) if tokens else "text"
        raise KeyError(f"No released WSM tunnel matches {route}->{target}")
    if len(candidates) > 1:
        choices = ", ".join(task.head for task in candidates[:12])
        raise ValueError(
            "More than one implementation matches this request; choose --source "
            f"or --head. Candidates: {choices}"
        )
    return candidates[0]


def _run_infer(args: argparse.Namespace) -> None:
    tokens, input_paths = _parse_inputs(args.input_files)
    catalog = TaskCatalog.load()
    task = _resolve_inference_task(
        catalog,
        tokens=tokens,
        target=args.target,
        prompt=args.prompt,
        source=args.source,
        head=args.head,
    )
    from .pipeline import WSMPipeline

    pipe = WSMPipeline.from_pretrained(
        args.model,
        head=task.head,
        base_model=args.base_model,
        device=args.device,
        cache_dir=args.cache_dir,
        revision=args.revision,
        token=args.token,
        local_files_only=args.local_files_only,
        categorical_refiner=args.categorical_refiner,
    )
    result = pipe(
        input_paths,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        seed=args.seed,
        size=args.size,
        output_dir=args.output_dir,
        exact_prompt_only=args.exact_prompt_only,
        paper_demo=args.paper_demo,
    )
    print(f"Task: {task.task_id}")
    for name, path in sorted(result.artifacts.items()):
        print(f"{name}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            parser.print_help()
            return 0
        if args.command == "list":
            _top_level()
            return 0
        if args.command == "info":
            _show_info(args.json)
            return 0
        if args.command == "infer":
            _run_infer(args)
            return 0
        if args.command in {"tunnel", "head"}:
            if args.action != "list":
                parser.parse_args([args.command, "--help"])
                return 0
            catalog = TaskCatalog.load()
            tasks = catalog.list(
                source=args.source,
                inputs=args.inputs,
                output=args.output,
            )
            if args.limit > 0:
                tasks = tasks[: args.limit]
            if args.json:
                print(
                    json.dumps(
                        [task.as_dict() for task in tasks],
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                _rows(tasks, heads=args.command == "head")
                print()
                print(f"{len(tasks):,} released entries")
            return 0
    except (FileNotFoundError, KeyError, PermissionError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
