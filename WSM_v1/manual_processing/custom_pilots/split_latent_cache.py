#!/usr/bin/env python3
"""Split a single-file FLUX cache into the sharded layout used by training."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--modalities", nargs="+", required=True)
    p.add_argument("--num-shards", type=int, default=8)
    args = p.parse_args()
    for modality in args.modalities:
        src = args.root / f"{modality.replace('-', '_')}.pt"
        payload = torch.load(src, map_location="cpu", weights_only=False)
        out_dir = args.root / "shards" / modality.replace("-", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        n = int(payload["latents"].shape[0])
        for shard_id in range(args.num_shards):
            indices = list(range(shard_id, n, args.num_shards))
            part = dict(payload)
            for key in ("sample_ids", "sample_sources", "video_ids", "frame_keys", "frame_indices", "crop_infos", "source_shard_indices", "shard_indices"):
                if key in part:
                    part[key] = [part[key][i] for i in indices]
            part["latents"] = payload["latents"][indices]
            part["num_samples"] = len(indices)
            part["num_shards"] = args.num_shards
            part["shard_id"] = shard_id
            torch.save(part, out_dir / f"shard_{shard_id:05d}_of_{args.num_shards:05d}.pt")
        print(f"split {modality}: {n} rows into {args.num_shards} shards")

if __name__ == "__main__":
    main()
