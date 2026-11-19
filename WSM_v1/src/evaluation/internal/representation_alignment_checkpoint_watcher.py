#!/usr/bin/env python
"""Archive transient checkpoints from controlled representation runs.

The production trainer intentionally retains only its latest best validation
checkpoint.  A learning-curve evaluation needs the earlier model states as
well, so this evaluation-side watcher copies every completed checkpoint before
the trainer replaces it.  It never writes into the trainer's checkpoint tree
and does not interact with the running model or optimizer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time


def completed_checkpoint(path: Path) -> bool:
    weights = path / "pytorch_lora_weights.safetensors"
    if not weights.is_file() or weights.stat().st_size <= 0:
        return False
    first = weights.stat().st_size
    time.sleep(0.2)
    return weights.is_file() and weights.stat().st_size == first


def archive_checkpoint(source: Path, destination_root: Path) -> bool:
    destination = destination_root / source.name
    if destination.is_dir():
        return False
    temporary = destination_root / f".{source.name}.copying"
    if temporary.exists():
        shutil.rmtree(temporary)
    destination_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    temporary.replace(destination)
    print(f"archived {source} -> {destination}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    root = args.root.resolve()
    variants = sorted(path for path in root.iterdir() if path.is_dir())
    if not variants:
        raise SystemExit(f"No representation run directories under {root}")

    final_seen: set[Path] = set()
    while len(final_seen) < len(variants):
        for variant in variants:
            checkpoint_root = variant / "ckpt"
            archive_root = variant / "trajectory_ckpts"
            for source in sorted(checkpoint_root.glob("*")):
                if not source.is_dir() or not completed_checkpoint(source):
                    continue
                archive_checkpoint(source, archive_root)
                if source.name.startswith("final_step_"):
                    final_seen.add(variant)
        if len(final_seen) < len(variants):
            time.sleep(max(args.poll_seconds, 0.5))


if __name__ == "__main__":
    main()
