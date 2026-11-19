#!/usr/bin/env python3
"""Restore WSM v1 continued-training materials from uncompressed TAR parts."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def safe_extract(archive_path: Path, output: Path) -> int:
    output = output.resolve()
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in release archives: {member.name}")
            target = (output / member.name).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Archive path escapes output directory: {member.name}")
        archive.extractall(output, members=members)
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parts = sorted(args.parts_dir.expanduser().resolve().rglob("*.tar"))
    if not parts:
        raise FileNotFoundError(f"No TAR parts found under {args.parts_dir}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    restored = 0
    for index, part in enumerate(parts, 1):
        count = safe_extract(part, output)
        restored += count
        print(f"[{index}/{len(parts)}] {part.name}: {count:,} files", flush=True)
    print(f"Restored {restored:,} files into {output}", flush=True)


if __name__ == "__main__":
    main()
