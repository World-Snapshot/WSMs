#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def _convert_floating_tensors_inplace(value, dtype: torch.dtype):
    if torch.is_tensor(value):
        if value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _convert_floating_tensors_inplace(item, dtype)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _convert_floating_tensors_inplace(item, dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_convert_floating_tensors_inplace(item, dtype) for item in value)
    return value


def _tensor_dtypes(value, out: set[str]) -> None:
    if torch.is_tensor(value) and value.is_floating_point():
        out.add(str(value.dtype))
    elif isinstance(value, dict):
        for item in value.values():
            _tensor_dtypes(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _tensor_dtypes(item, out)


def _convert_file(path: Path, dtype: torch.dtype) -> tuple[int, int, set[str], set[str]]:
    before_size = path.stat().st_size
    payload = torch.load(path, map_location="cpu", weights_only=False)
    before_dtypes: set[str] = set()
    _tensor_dtypes(payload, before_dtypes)
    _convert_floating_tensors_inplace(payload, dtype)
    after_dtypes: set[str] = set()
    _tensor_dtypes(payload, after_dtypes)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    after_size = path.stat().st_size
    return before_size, after_size, before_dtypes, after_dtypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--glob", default="parts/**/*.pt")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dtype = _dtype_from_name(args.dtype)
    paths = sorted(args.cache_dir.glob(args.glob))
    if args.limit is not None:
        paths = paths[: int(args.limit)]
    if not paths:
        print(f"[convert-cache-dtype] no files matched {args.cache_dir / args.glob}")
        return

    total_before = 0
    total_after = 0
    for index, path in enumerate(paths, start=1):
        before, after, before_dtypes, after_dtypes = _convert_file(path, dtype)
        total_before += before
        total_after += after
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(
                f"[convert-cache-dtype] {index}/{len(paths)} "
                f"last={path} {before / 1024**2:.2f}MiB->{after / 1024**2:.2f}MiB "
                f"dtypes={sorted(before_dtypes)}->{sorted(after_dtypes)} "
                f"total={total_before / 1024**3:.2f}GiB->{total_after / 1024**3:.2f}GiB",
                flush=True,
            )

    print(
        f"[convert-cache-dtype] done files={len(paths)} "
        f"total={total_before / 1024**3:.2f}GiB->{total_after / 1024**3:.2f}GiB "
        f"saved={(total_before - total_after) / 1024**3:.2f}GiB"
    )


if __name__ == "__main__":
    main()
