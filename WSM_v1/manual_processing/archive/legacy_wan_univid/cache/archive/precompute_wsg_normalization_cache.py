import argparse
import json
import random
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.wsg_dataset import MASK_VALUE, parse_channel_key

NORMALIZATION_VERSION = "sampled_percentile_v2"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def write_status_file(path_value: Optional[str], payload: dict) -> None:
    if not path_value:
        return
    status_path = Path(path_value)
    if not status_path.is_absolute():
        status_path = PROJECT_ROOT / status_path
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(status_path)


def parse_channels(value: str) -> list[int]:
    channels: list[int] = []
    for item in value.split():
        channels.extend(parse_channel_key(item))
    return sorted(set(ch for ch in channels if ch not in (0, 1, 2)))


def iter_wsg_items(root: Path, max_arrays: Optional[int], seed: int) -> list[tuple[str, str, Optional[str]]]:
    candidates = sorted(list(root.glob("**/*.wsg")) + list(root.glob("**/*.wsg.npz")))
    wsv_candidates = sorted(list(root.glob("**/*.wsv")) + list(root.glob("**/*.wsv.npz")))
    items: list[tuple[str, str, Optional[str]]] = [("wsg", str(path), None) for path in candidates]
    for wsv_path in wsv_candidates:
        try:
            with np.load(wsv_path) as data:
                for key in data.files:
                    items.append(("wsv", str(wsv_path), key))
        except Exception:
            continue
    if max_arrays is not None and len(items) > max_arrays:
        items = random.Random(seed).sample(items, max_arrays)
    return items


def empty_stats(channels: list[int]) -> dict[int, dict]:
    return {
        ch: {
            "values_seen": 0,
            "min": None,
            "max": None,
            "is_integer_like": True,
            "unique": set(),
            "unique_overflow": False,
            "samples": [],
        }
        for ch in channels
    }


def update_stats(stats: dict[int, dict], ch: int, values: np.ndarray, unique_limit: int) -> None:
    if values.size == 0:
        return
    st = stats[ch]
    st["values_seen"] += int(values.size)
    cur_min = float(np.min(values))
    cur_max = float(np.max(values))
    st["min"] = cur_min if st["min"] is None else min(st["min"], cur_min)
    st["max"] = cur_max if st["max"] is None else max(st["max"], cur_max)
    st["samples"].append(values.astype(np.float32, copy=True))
    if st["is_integer_like"] and not np.allclose(values, np.round(values)):
        st["is_integer_like"] = False
    if not st["unique_overflow"]:
        unique_values = np.unique(values)
        if unique_values.size > unique_limit * 2:
            st["unique_overflow"] = True
            st["unique"].clear()
            return
        st["unique"].update(float(value) for value in unique_values.tolist())
        if len(st["unique"]) > unique_limit * 2:
            st["unique_overflow"] = True
            st["unique"].clear()


def merge_stats(left: dict[int, dict], right: dict[int, dict], unique_limit: int) -> dict[int, dict]:
    for ch, src in right.items():
        dst = left[ch]
        dst["values_seen"] += int(src["values_seen"])
        if src["min"] is not None:
            dst["min"] = src["min"] if dst["min"] is None else min(dst["min"], src["min"])
        if src["max"] is not None:
            dst["max"] = src["max"] if dst["max"] is None else max(dst["max"], src["max"])
        dst["is_integer_like"] = bool(dst["is_integer_like"] and src["is_integer_like"])
        if dst["unique_overflow"] or src["unique_overflow"]:
            dst["unique_overflow"] = True
            dst["unique"].clear()
        else:
            dst["unique"].update(src["unique"])
            if len(dst["unique"]) > unique_limit * 2:
                dst["unique_overflow"] = True
                dst["unique"].clear()
        dst["samples"].extend(src["samples"])
    return left


def load_item(item: tuple[str, str, Optional[str]]) -> Optional[np.ndarray]:
    storage, file_path, frame_key = item
    try:
        with np.load(file_path) as data:
            if storage == "wsv":
                return data[frame_key]
            if "data" in data:
                return data["data"]
            if "wsg_data" in data:
                return data["wsg_data"]
    except Exception:
        return None
    return None


def _float_matches(left: object, right: float, eps: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= eps
    except (TypeError, ValueError):
        return False


def normalization_matches(
    channel_entry: dict,
    sampled_arrays: int,
    max_values_per_array: int,
    unique_limit: int,
    seed: int,
    clip_enabled: bool,
    percentile_low: float,
    percentile_high: float,
) -> bool:
    norm = channel_entry.get("normalization")
    if not isinstance(norm, dict):
        return False
    if norm.get("normalization_version") != NORMALIZATION_VERSION:
        return False
    expected = {
        "sampled_arrays": sampled_arrays,
        "sampled_values_per_array": max_values_per_array,
        "normalization_unique_limit": unique_limit,
        "normalization_seed": seed,
        "continuous_percentile_clip_enabled": clip_enabled,
    }
    for key, value in expected.items():
        if norm.get(key) != value:
            return False
    if clip_enabled:
        return (
            _float_matches(norm.get("continuous_percentile_low"), percentile_low)
            and _float_matches(norm.get("continuous_percentile_high"), percentile_high)
        )
    return True


def worker_scan(payload: tuple[list[tuple[str, str, Optional[str]]], list[int], int, int, int]) -> dict[int, dict]:
    items, channels, max_values_per_array, unique_limit, seed = payload
    stats = empty_stats(channels)
    rng = np.random.default_rng(seed)
    for item in items:
        array = load_item(item)
        if array is None:
            continue
        channel_count = array.shape[-1]
        for ch in channels:
            if ch >= channel_count:
                continue
            values = np.asarray(array[..., ch]).reshape(-1)
            values = values[values != MASK_VALUE]
            if values.size == 0:
                continue
            values = values[np.isfinite(values)].astype(np.float32, copy=False)
            if values.size == 0:
                continue
            if values.size > max_values_per_array:
                indices = rng.choice(values.size, max_values_per_array, replace=False)
                values = values[indices]
            update_stats(stats, ch, values, unique_limit)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", action="append", required=True)
    parser.add_argument("--cache", default="configs/wsg_dataset_info.json")
    parser.add_argument("--channels", required=True, help="Space/comma/range channel list, e.g. '72 73 74-76'.")
    parser.add_argument("--max-arrays", type=int, default=600)
    parser.add_argument("--max-values-per-array", type=int, default=1000)
    parser.add_argument("--unique-limit", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--continuous-percentile-low", type=float, default=2.5)
    parser.add_argument("--continuous-percentile-high", type=float, default=97.5)
    parser.add_argument("--disable-continuous-percentile-clip", action="store_true")
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args()
    if not 0.0 <= args.continuous_percentile_low < args.continuous_percentile_high <= 100.0:
        raise ValueError(
            "Expected --continuous-percentile-low/high to satisfy "
            "0 <= low < high <= 100."
        )

    channels = parse_channels(args.channels)
    print(f"[normalization] channels={channels}", flush=True)
    if not channels:
        return

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    now = datetime.now().isoformat()
    status = {
        "changed": False,
        "datasets": [],
        "normalization_version": NORMALIZATION_VERSION,
        "stage": "normalization",
        "state": "running",
        "started_at": now,
        "total_datasets": len(args.dataset_path),
    }
    run_start = time.monotonic()
    write_status_file(args.status_file, status)
    for dataset_index, dataset_path in enumerate(args.dataset_path, start=1):
        root = Path(dataset_path).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        items = iter_wsg_items(root, args.max_arrays, int(args.seed))
        workers = max(1, min(int(args.workers), len(items) or 1))
        key = str(root)
        entry = cache.setdefault(key, {
            "dataset_name": root.name,
            "dataset_path": str(root),
            "channels": {},
        })
        entry.setdefault("channels", {})
        clip_enabled = not bool(args.disable_continuous_percentile_clip)
        channels_to_scan = [
            ch for ch in channels
            if not normalization_matches(
                entry["channels"].get(str(ch), {}),
                sampled_arrays=len(items),
                max_values_per_array=int(args.max_values_per_array),
                unique_limit=int(args.unique_limit),
                seed=int(args.seed),
                clip_enabled=clip_enabled,
                percentile_low=float(args.continuous_percentile_low),
                percentile_high=float(args.continuous_percentile_high),
            )
        ]
        skipped = len(channels) - len(channels_to_scan)
        if not channels_to_scan:
            print(
                f"[normalization] dataset={root} skip all {len(channels)} channels; "
                f"cached stats match sampled_arrays={len(items)} max_values_per_array={args.max_values_per_array} "
                f"seed={args.seed} continuous_clip={'off' if not clip_enabled else f'{args.continuous_percentile_low}-{args.continuous_percentile_high}'}",
                flush=True,
            )
            status["datasets"].append({
                "dataset_path": str(root),
                "changed": False,
                "scanned_channels": [],
                "skipped_channels": channels,
            })
            status.update({
                "current_dataset": str(root),
                "dataset_index": dataset_index,
                "dataset_state": "skipped_cached",
                "completed_datasets": dataset_index,
                "elapsed_seconds": int(time.monotonic() - run_start),
            })
            write_status_file(args.status_file, status)
            continue
        status["changed"] = True
        dataset_status = {
            "dataset_path": str(root),
            "changed": True,
            "scanned_channels": channels_to_scan,
            "skipped_channels": [ch for ch in channels if ch not in set(channels_to_scan)],
            "shards_total": workers,
            "shards_done": 0,
        }
        status["datasets"].append(dataset_status)
        dataset_start = time.monotonic()
        status.update({
            "current_dataset": str(root),
            "dataset_index": dataset_index,
            "dataset_state": "scanning",
            "completed_datasets": dataset_index - 1,
            "current_dataset_shards_done": 0,
            "current_dataset_shards_total": workers,
            "elapsed_seconds": int(time.monotonic() - run_start),
        })
        write_status_file(args.status_file, status)
        print(
            f"[normalization] dataset_index={dataset_index}/{len(args.dataset_path)} dataset={root} "
            f"sampled_arrays={len(items)} workers={workers} "
            f"max_values_per_array={args.max_values_per_array} seed={args.seed} "
            f"continuous_clip={'off' if not clip_enabled else f'{args.continuous_percentile_low}-{args.continuous_percentile_high}'} "
            f"scan_channels={len(channels_to_scan)} skipped_cached={skipped}",
            flush=True,
        )
        chunks = [items[index::workers] for index in range(workers)]
        payloads = [
            (chunk, channels_to_scan, int(args.max_values_per_array), int(args.unique_limit), int(args.seed) + index)
            for index, chunk in enumerate(chunks)
        ]
        merged = empty_stats(channels_to_scan)
        if workers == 1:
            iterator = (worker_scan(payloads[0]),)
        else:
            with Pool(processes=workers) as pool:
                iterator = pool.imap_unordered(worker_scan, payloads)
                iterator = tqdm(iterator, total=len(payloads), desc="normalization shards", unit="shard", dynamic_ncols=True)
                for shard_done, partial in enumerate(iterator, start=1):
                    merged = merge_stats(merged, partial, int(args.unique_limit))
                    elapsed = time.monotonic() - dataset_start
                    per_shard = elapsed / max(1, shard_done)
                    eta = per_shard * max(0, workers - shard_done)
                    dataset_status["shards_done"] = shard_done
                    dataset_status["elapsed_seconds"] = int(elapsed)
                    dataset_status["eta_seconds"] = int(eta)
                    status.update({
                        "dataset_state": "scanning",
                        "current_dataset_shards_done": shard_done,
                        "current_dataset_shards_total": workers,
                        "current_dataset_elapsed_seconds": int(elapsed),
                        "current_dataset_eta_seconds": int(eta),
                        "elapsed_seconds": int(time.monotonic() - run_start),
                    })
                    write_status_file(args.status_file, status)
                    print(
                        f"[normalization] progress dataset_index={dataset_index}/{len(args.dataset_path)} "
                        f"dataset={root.name} shards={shard_done}/{workers} "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                        flush=True,
                    )
                iterator = ()
        for shard_done, partial in enumerate(iterator, start=1):
            merged = merge_stats(merged, partial, int(args.unique_limit))
            elapsed = time.monotonic() - dataset_start
            eta = 0.0
            dataset_status["shards_done"] = shard_done
            dataset_status["elapsed_seconds"] = int(elapsed)
            dataset_status["eta_seconds"] = int(eta)
            status.update({
                "current_dataset_shards_done": shard_done,
                "current_dataset_shards_total": workers,
                "current_dataset_elapsed_seconds": int(elapsed),
                "current_dataset_eta_seconds": int(eta),
                "elapsed_seconds": int(time.monotonic() - run_start),
            })
            write_status_file(args.status_file, status)

        entry["last_updated"] = now
        for ch in channels_to_scan:
            st = merged[ch]
            channel_entry = entry["channels"].setdefault(str(ch), {})
            if st["min"] is None or st["max"] is None:
                channel_entry["valid_ratio"] = 0.0
                channel_entry["has_data"] = False
                channel_entry["normalization"] = {
                    "normalization_version": NORMALIZATION_VERSION,
                    "computed_at": now,
                    "source": "WSG automatic sampled normalization precompute",
                    "sampled_values": int(st["values_seen"]),
                    "sampled_arrays": len(items),
                    "sampled_values_per_array": int(args.max_values_per_array),
                    "normalization_workers": workers,
                    "normalization_seed": int(args.seed),
                    "normalization_unique_limit": int(args.unique_limit),
                    "continuous_percentile_clip_enabled": clip_enabled,
                    "continuous_percentile_low": float(args.continuous_percentile_low),
                    "continuous_percentile_high": float(args.continuous_percentile_high),
                }
                print(f"[normalization] ch{ch}: no valid data", flush=True)
                continue
            unique_values = sorted(st["unique"])
            discrete = (
                st["is_integer_like"]
                and not st["unique_overflow"]
                and len(unique_values) <= int(args.unique_limit)
            )
            min_val = float(st["min"])
            max_val = float(st["max"])
            percentile_low = 0.0
            percentile_high = 100.0
            if not discrete and not args.disable_continuous_percentile_clip:
                sampled_values = np.concatenate(st["samples"]) if st["samples"] else np.array([], dtype=np.float32)
                if sampled_values.size > 0:
                    min_val = float(np.percentile(sampled_values, float(args.continuous_percentile_low)))
                    max_val = float(np.percentile(sampled_values, float(args.continuous_percentile_high)))
                    percentile_low = float(args.continuous_percentile_low)
                    percentile_high = float(args.continuous_percentile_high)
                    if min_val == max_val:
                        min_val = float(st["min"])
                        max_val = float(st["max"])
                        percentile_low = 0.0
                        percentile_high = 100.0
            channel_entry["valid_ratio"] = 1.0
            channel_entry["has_data"] = True
            channel_entry["normalization"] = {
                "min_val": min_val,
                "max_val": max_val,
                "raw_min_val": st["min"],
                "raw_max_val": st["max"],
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
                "normalization_version": NORMALIZATION_VERSION,
                "computed_at": now,
                "source": "WSG automatic sampled normalization precompute",
                "sampled_values": int(st["values_seen"]),
                "sampled_arrays": len(items),
                "sampled_values_per_array": int(args.max_values_per_array),
                "normalization_workers": workers,
                "normalization_seed": int(args.seed),
                "normalization_unique_limit": int(args.unique_limit),
                "continuous_percentile_clip_enabled": clip_enabled,
                "continuous_percentile_low": float(args.continuous_percentile_low),
                "continuous_percentile_high": float(args.continuous_percentile_high),
                "is_integer_like": bool(st["is_integer_like"]),
                "unique_count": len(unique_values) if unique_values else None,
                "unique_values": unique_values if discrete else None,
                "unique_full_scan": False,
                "discrete_unique_protected": bool(discrete),
                "continuous_percentile_clipped": bool(not discrete and percentile_low != 0.0 and percentile_high != 100.0),
            }
            print(
                f"[normalization] ch{ch}: range=({min_val}, {max_val}) "
                f"raw=({st['min']}, {st['max']}) "
                f"values={st['values_seen']} unique={len(unique_values) if unique_values else 'n/a'} "
                f"discrete={discrete}",
                flush=True,
            )
        dataset_elapsed = time.monotonic() - dataset_start
        dataset_status["completed"] = True
        dataset_status["elapsed_seconds"] = int(dataset_elapsed)
        dataset_status["eta_seconds"] = 0
        status.update({
            "dataset_state": "completed",
            "completed_datasets": dataset_index,
            "current_dataset_elapsed_seconds": int(dataset_elapsed),
            "current_dataset_eta_seconds": 0,
            "elapsed_seconds": int(time.monotonic() - run_start),
        })
        write_status_file(args.status_file, status)
        print(
            f"[normalization] completed dataset_index={dataset_index}/{len(args.dataset_path)} "
            f"dataset={root.name} elapsed={format_duration(dataset_elapsed)}",
            flush=True,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp_path.replace(cache_path)
    print(f"[normalization] updated {cache_path}", flush=True)
    status.update({
        "state": "completed",
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": int(time.monotonic() - run_start),
    })
    write_status_file(args.status_file, status)
    if args.status_file:
        status_path = Path(args.status_file)
        if not status_path.is_absolute():
            status_path = PROJECT_ROOT / status_path
        print(f"[normalization] wrote status {status_path}", flush=True)


if __name__ == "__main__":
    main()
