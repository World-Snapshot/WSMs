import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def latest_log_line(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()[-8192:].decode("utf-8", errors="ignore")
    lines = []
    for chunk in data.replace("\r", "\n").splitlines():
        line = ANSI_RE.sub("", chunk).strip()
        if line:
            lines.append(line)
    return lines[-1] if lines else ""


def has_running_encoder(output_dir: Path) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "precompute_wsg_vae_latents.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    output_args = {
        f"--output-dir {output_dir.as_posix()}",
        f"--output-dir {output_dir.resolve().as_posix()}",
    }
    return any(any(arg in line for arg in output_args) for line in result.stdout.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="wsg")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"

    while has_running_encoder(output_dir):
        print(
            f"[VAE cache][{args.split_name}] "
            f"{datetime.now().strftime('%F %T')} progress",
            flush=True,
        )
        for shard_id in range(args.num_shards):
            line = latest_log_line(logs_dir / f"shard_{shard_id}.log")
            if line:
                print(f"  shard_{shard_id}: {line}", flush=True)
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
