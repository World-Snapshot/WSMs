#!/usr/bin/env python
"""Precompute FLUX.2 text embeddings for WSG training.

The default production mode is a global caption cache keyed by sample_id. That
keeps captions non-duplicated across Channel_Head tasks: every dataset sample is present
once per shard, and training can recover its text embedding by sample_id.

The older per-Channel_Head prompt mode is still available for experiments that encode
caption + role tags as one complete prompt, but it intentionally duplicates
samples across Channel_Head entries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from diffusers import Flux2KleinPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_TunnelingNet_Flux2_4B_wsg import build_prompts, resolve_path, safe_modality_name  # noqa: E402

SPECIAL_CONDITIONS = {"caption", "Reference_IMG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--modality-config", default="configs/wsg_v1_modalities_dense.json")
    parser.add_argument("--latent-cache-dir", action="append", dest="latent_cache_dirs", default=None)
    parser.add_argument("--dense-latent-cache-dir", default="data/wsg_flux2_frame_latents_dense_multidataset")
    parser.add_argument("--xpose-latent-cache-dir", default="data/xpose_flux2_frame_latents_sample10pct_2frames")
    parser.add_argument("--output-dir", default="data/wsg_flux2_text_embeddings_dense_single_condition")
    parser.add_argument("--global-captions", action="store_true", help="Encode dataset captions once per sample_id instead of per-Channel_Head prompts.")
    parser.add_argument("--channel-head-index", type=int, default=270)
    parser.add_argument("--all-channel-heads", action="store_true")
    parser.add_argument("--channel-head-indices", default=None, help="Comma-separated Channel_Head indices; overrides --all-channel-heads/--channel-head-index.")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional per-Channel_Head/per-shard sample cap for smoke tests.")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_entries(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    raw_entries = config.get("wsg_v1", config)
    entries: dict[int, dict[str, Any]] = {}
    for key, value in raw_entries.items():
        if not key.startswith("Channel_Head_"):
            continue
        entries[int(key.split("_")[-1])] = dict(value)
    return dict(sorted(entries.items()))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def modality_prompt(entry: dict[str, Any], target: str, conditions: list[str]) -> str:
    tags: list[str] = []
    if "caption" in conditions:
        tags.append("caption text")
    if "Reference_IMG" in conditions:
        tags.append("reference image")
    for condition in conditions:
        if condition not in SPECIAL_CONDITIONS:
            tags.append(f"condition modality {condition}")
    tags.append(f"target modality {target}")
    return ". ".join(tags) + "."


def prompt_for_source(source: dict[str, Any], marker: str) -> str:
    caption = build_prompts([source])[0]
    suffix = "Preserve the reference appearance and follow the supplied condition modality." if "reference image" in marker else "Follow the supplied condition modality."
    return " ".join(part.strip() for part in (caption, marker, suffix) if part and part.strip())


def shard_path(root: Path, modality: str, shard_id: int, num_shards: int) -> Path:
    return root / "shards" / safe_modality_name(modality) / f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"


def load_payload(root: Path, modality: str, shard_id: int, num_shards: int) -> dict[str, Any] | None:
    path = shard_path(root, modality, shard_id, num_shards)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def iter_payload_paths(root: Path, shard_id: int, num_shards: int) -> list[Path]:
    shard_name = f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
    shard_root = root / "shards"
    if not shard_root.exists():
        return []
    return sorted(shard_root.glob(f"*/{shard_name}"))


def load_prompt_overrides(root: Path) -> dict[str, str]:
    prompts_path = root / "prompts.pt"
    if not prompts_path.is_file():
        return {}
    payload = torch.load(prompts_path, map_location="cpu", weights_only=False)
    sample_ids = list(payload.get("sample_ids") or [])
    prompts = list(payload.get("prompts") or [])
    if len(sample_ids) != len(prompts):
        raise ValueError(
            f"{prompts_path} contains {len(sample_ids)} sample_ids but {len(prompts)} prompts"
        )
    return {str(sample_id): str(prompt) for sample_id, prompt in zip(sample_ids, prompts)}


def collect_global_caption_records(cache_roots: list[Path], shard_id: int, num_shards: int, max_samples: int | None) -> list[dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    for root in cache_roots:
        prompt_overrides = load_prompt_overrides(root)
        for path in iter_payload_paths(root, shard_id, num_shards):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            sample_ids = [str(item) for item in payload.get("sample_ids", [])]
            sources = payload.get("sample_sources") or [{} for _ in sample_ids]
            for sample_id, source in zip(sample_ids, sources):
                if sample_id in records_by_id:
                    continue
                records_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "sample_source": dict(source or {}),
                    "cache_root": str(root),
                    "source_shard": str(path),
                    "prompt": prompt_overrides.get(sample_id),
                }
            del payload
    records = [records_by_id[item] for item in sorted(records_by_id)]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def records_for_root(root: Path, target: str, latent_conditions: list[str], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    target_payload = load_payload(root, target, shard_id, num_shards)
    if target_payload is None:
        return []
    condition_payloads = []
    for condition in latent_conditions:
        payload = load_payload(root, condition, shard_id, num_shards)
        if payload is None:
            return []
        condition_payloads.append(payload)
    target_ids = [str(item) for item in target_payload.get("sample_ids", [])]
    available = set(target_ids)
    for payload in condition_payloads:
        available &= {str(item) for item in payload.get("sample_ids", [])}
    sources = target_payload.get("sample_sources") or [{} for _ in target_ids]
    records = []
    for index, sample_id in enumerate(target_ids):
        if sample_id in available:
            records.append({"sample_id": sample_id, "sample_source": sources[index], "cache_root": str(root)})
    return records


def collect_records(cache_roots: list[Path], entry: dict[str, Any], shard_id: int, num_shards: int, max_samples: int | None) -> tuple[str, list[str], str, list[dict[str, Any]]]:
    target = str(entry["target"])
    conditions = as_list(entry.get("condition"))
    latent_conditions = [condition for condition in conditions if condition not in SPECIAL_CONDITIONS]
    marker = modality_prompt(entry, target, conditions)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in cache_roots:
        for record in records_for_root(root, target, latent_conditions, shard_id, num_shards):
            if record["sample_id"] in seen:
                continue
            seen.add(record["sample_id"])
            records.append(record)
    records.sort(key=lambda item: item["sample_id"])
    if max_samples is not None:
        records = records[:max_samples]
    return target, conditions, marker, records


def selected_channel_heads(args: argparse.Namespace, entries: dict[int, dict[str, Any]]) -> list[int]:
    if args.channel_head_indices:
        return [int(item) for item in args.channel_head_indices.split(",") if item.strip()]
    if args.all_channel_heads:
        return list(entries)
    return [args.channel_head_index]


def encode_channel_head_shard(args: argparse.Namespace, pipe: Flux2KleinPipeline | None, cache_roots: list[Path], channel_head_index: int, entry: dict[str, Any], shard_id: int) -> Path | None:
    target, conditions, marker, records = collect_records(cache_roots, entry, shard_id, args.num_shards, args.max_samples)
    channel_head_dir = resolve_path(args.output_dir) / f"Channel_Head_{channel_head_index:06d}"
    out_path = channel_head_dir / f"shard_{shard_id:05d}_of_{args.num_shards:05d}.pt"
    print(f"[flux2-text] Channel_Head={channel_head_index:06d} shard={shard_id}/{args.num_shards} target={target} conditions={conditions} records={len(records)}", flush=True)
    if args.dry_run:
        return out_path
    if not records:
        return None
    if out_path.exists() and not args.overwrite:
        print(f"[flux2-text] skip existing {out_path}", flush=True)
        return out_path
    if pipe is None:
        raise RuntimeError("pipe is required unless --dry-run is used")
    channel_head_dir.mkdir(parents=True, exist_ok=True)
    sample_ids: list[str] = []
    descriptions: list[str] = []
    cache_roots_for_samples: list[str] = []
    unique_prompts: list[str] = []
    prompt_to_index: dict[str, int] = {}
    sample_text_indices: list[int] = []
    for record in records:
        prompt = prompt_for_source(record["sample_source"], marker)
        text_index = prompt_to_index.get(prompt)
        if text_index is None:
            text_index = len(unique_prompts)
            prompt_to_index[prompt] = text_index
            unique_prompts.append(prompt)
        sample_ids.append(record["sample_id"])
        descriptions.append(prompt)
        sample_text_indices.append(text_index)
        cache_roots_for_samples.append(record["cache_root"])

    prompt_chunks: list[torch.Tensor] = []
    text_id_chunks: list[torch.Tensor] = []
    start = time.time()
    for start_index in tqdm(range(0, len(unique_prompts), args.batch_size), desc=f"text Channel_Head {channel_head_index:06d} shard {shard_id}"):
        prompts = unique_prompts[start_index:start_index + args.batch_size]
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(prompts, device=args.device, max_sequence_length=args.max_sequence_length)
        prompt_embeds = prompt_embeds.detach().to("cpu", dtype=torch.bfloat16)
        text_ids = text_ids.detach().to("cpu")
        if text_ids.ndim == 2:
            text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1).clone()
        prompt_chunks.append(prompt_embeds)
        text_id_chunks.append(text_ids)
    temporary = out_path.with_suffix(".tmp")
    torch.save({
        "channel_head_index": channel_head_index,
        "target_modality": target,
        "condition_modalities": conditions,
        "latent_condition_modalities": [condition for condition in conditions if condition not in SPECIAL_CONDITIONS],
        "modality_type_prompt": marker,
        "sample_ids": sample_ids,
        "cache_roots": cache_roots_for_samples,
        "descriptions": descriptions,
        "unique_prompts": unique_prompts,
        "sample_text_indices": sample_text_indices,
        "prompt_embeds": torch.cat(prompt_chunks, dim=0),
        "text_ids": torch.cat(text_id_chunks, dim=0),
        "dtype": "bfloat16",
        "max_sequence_length": args.max_sequence_length,
        "created_unix_time": time.time(),
    }, temporary)
    temporary.replace(out_path)
    print(f"[flux2-text] saved {out_path} samples={len(sample_ids)} unique_prompts={len(unique_prompts)} elapsed_min={(time.time() - start) / 60.0:.2f}", flush=True)
    return out_path



def encode_global_caption_shard(args: argparse.Namespace, pipe: Flux2KleinPipeline | None, cache_roots: list[Path], shard_id: int) -> Path | None:
    records = collect_global_caption_records(cache_roots, shard_id, args.num_shards, args.max_samples)
    output_dir = resolve_path(args.output_dir)
    out_path = output_dir / f"shard_{shard_id:05d}_of_{args.num_shards:05d}.pt"
    print(f"[flux2-text] global captions shard={shard_id}/{args.num_shards} records={len(records)}", flush=True)
    if args.dry_run:
        return out_path
    if not records:
        return None
    if out_path.exists() and not args.overwrite:
        print(f"[flux2-text] skip existing {out_path}", flush=True)
        return out_path
    if pipe is None:
        raise RuntimeError("pipe is required unless --dry-run is used")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids: list[str] = []
    cache_roots_for_samples: list[str] = []
    source_shards: list[str] = []
    sample_sources: list[dict[str, Any]] = []
    unique_prompts: list[str] = []
    prompt_to_index: dict[str, int] = {}
    sample_text_indices: list[int] = []

    fallback_captions = build_prompts([record["sample_source"] for record in records])
    captions = [
        str(record.get("prompt") or fallback)
        for record, fallback in zip(records, fallback_captions)
    ]
    for record, caption in zip(records, captions):
        text_index = prompt_to_index.get(caption)
        if text_index is None:
            text_index = len(unique_prompts)
            prompt_to_index[caption] = text_index
            unique_prompts.append(caption)
        sample_ids.append(record["sample_id"])
        cache_roots_for_samples.append(record["cache_root"])
        source_shards.append(record["source_shard"])
        sample_sources.append(record["sample_source"])
        sample_text_indices.append(text_index)

    prompt_chunks: list[torch.Tensor] = []
    text_id_chunks: list[torch.Tensor] = []
    start = time.time()
    for start_index in tqdm(range(0, len(unique_prompts), args.batch_size), desc=f"caption shard {shard_id}"):
        prompts = unique_prompts[start_index:start_index + args.batch_size]
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(prompts, device=args.device, max_sequence_length=args.max_sequence_length)
        prompt_embeds = prompt_embeds.detach().to("cpu", dtype=torch.bfloat16)
        text_ids = text_ids.detach().to("cpu")
        if text_ids.ndim == 2:
            text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1).clone()
        prompt_chunks.append(prompt_embeds)
        text_id_chunks.append(text_ids)

    temporary = out_path.with_suffix(".tmp")
    torch.save({
        "cache_type": "global_caption",
        "sample_ids": sample_ids,
        "cache_roots": cache_roots_for_samples,
        "source_shards": source_shards,
        "sample_sources": sample_sources,
        "descriptions": [unique_prompts[index] for index in sample_text_indices],
        "unique_prompts": unique_prompts,
        "sample_text_indices": sample_text_indices,
        "prompt_embeds": torch.cat(prompt_chunks, dim=0),
        "text_ids": torch.cat(text_id_chunks, dim=0),
        "dtype": "bfloat16",
        "max_sequence_length": args.max_sequence_length,
        "created_unix_time": time.time(),
    }, temporary)
    temporary.replace(out_path)
    print(f"[flux2-text] saved global {out_path} samples={len(sample_ids)} unique_prompts={len(unique_prompts)} elapsed_min={(time.time() - start) / 60.0:.2f}", flush=True)
    return out_path

def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    entries = load_entries(resolve_path(args.modality_config)) if not args.global_captions else {}
    channel_head_indices = selected_channel_heads(args, entries) if not args.global_captions else []
    cache_dirs = args.latent_cache_dirs or [args.dense_latent_cache_dir, args.xpose_latent_cache_dir]
    cache_roots = [resolve_path(item) for item in cache_dirs if item and resolve_path(item).exists()]
    if not cache_roots:
        raise RuntimeError("No latent cache roots exist")
    shard_ids = [args.shard_id] if args.shard_id is not None else list(range(args.num_shards))
    print(json.dumps({"mode": "global_captions" if args.global_captions else "per_channel_head_prompt", "channel_head_count": len(channel_head_indices), "shards": shard_ids, "cache_roots": [str(item) for item in cache_roots], "dry_run": args.dry_run}, indent=2), flush=True)
    pipe = None
    if not args.dry_run:
        pipe = Flux2KleinPipeline.from_pretrained(str(resolve_path(args.model_path)), torch_dtype=torch.bfloat16).to(args.device)
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer.to("cpu")
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to("cpu")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    for shard_id in shard_ids:
        if args.global_captions:
            encode_global_caption_shard(args, pipe, cache_roots, int(shard_id))
            continue
        for channel_head_index in channel_head_indices:
            if channel_head_index not in entries:
                raise KeyError(f"Channel_Head_{channel_head_index:06d} missing from config")
            encode_channel_head_shard(args, pipe, cache_roots, channel_head_index, entries[channel_head_index], int(shard_id))


if __name__ == "__main__":
    main()
