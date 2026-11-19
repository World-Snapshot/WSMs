#!/usr/bin/env python3
"""Periodically record GPU resource usage for benchmark provenance."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path


FIELDS = [
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--duration-hours", type=float, default=8.0)
    return parser.parse_args()


def query() -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    return [[item.strip() for item in line.split(",")] for line in output.splitlines()]


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration_hours * 3600.0
    new_file = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["timestamp_utc", *FIELDS])
        while time.monotonic() < deadline:
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                for row in query():
                    writer.writerow([timestamp, *row])
                handle.flush()
            except (OSError, subprocess.SubprocessError) as error:
                writer.writerow([timestamp, "query_error", str(error)])
                handle.flush()
            time.sleep(max(0.25, args.interval_seconds))


if __name__ == "__main__":
    main()
