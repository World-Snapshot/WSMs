#!/usr/bin/env python3
"""Generate sample-aligned captions for the MP-100 Universal Pose cache.

Each worker writes one resumable JSONL shard. Merge mode validates complete
sample-ID coverage, preserves the category templates, and installs generated
captions as prompts.pt beside the encoded image latents.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]

CATEGORY_LABEL_OVERRIDES = {
    "amur_tiger": "Amur tiger",
    "arcticwolf": "Arctic wolf",
    "bighornsheep": "bighorn sheep",
    "californiansealion": "California sea lion",
    "capebuffalo": "Cape buffalo",
    "commonwarthog": "common warthog",
    "fallowdeer": "fallow deer",
    "fennecfox": "fennec fox",
    "gentoopenguin": "gentoo penguin",
    "germanshepherddog": "German shepherd dog",
    "goldenretriever": "golden retriever",
    "greyseal": "grey seal",
    "grizzlybear": "grizzly bear",
    "olivebaboon": "olive baboon",
    "proboscismonkey": "proboscis monkey",
    "przewalskihorse": "Przewalski's horse",
    "short_sleeved": "short-sleeved",
    "long_sleeved": "long-sleeved",
    "spider_monkey": "spider monkey",
    "swivelchair": "swivel chair",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("../../wsg/output/PoseAnything_MP100"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/mp100_universal_pose_flux2_frame_latents"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../../wsg/main/checkpoints/blip2-opt-2.7b"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--install-prompts", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def readable_category(category_name: str) -> str:
    base = re.sub(r"_(body|face)$", "", str(category_name))
    for source, target in sorted(CATEGORY_LABEL_OVERRIDES.items(), key=lambda item: -len(item[0])):
        base = base.replace(source, target)
    return " ".join(base.replace("_", " ").split())


def normalize_generated_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip()).strip(" \t\r\n\"'")
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def compose_caption(category_name: str, generated: str) -> str:
    label = readable_category(category_name)
    anchor = f"A photo of {label}."
    detail = normalize_generated_caption(generated)
    if not detail:
        return anchor
    if detail.lower() in {f"a photo of {label.lower()}.", f"an image of {label.lower()}."}:
        return anchor
    return f"{anchor} {detail}"


def load_records(cache_dir: Path) -> list[dict[str, Any]]:
    index_path = cache_dir / "dataset_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = list(payload["records"])
    if len(records) != int(payload["num_samples"]):
        raise ValueError(f"{index_path} record count does not match num_samples")
    return records


def shard_path(cache_dir: Path, shard_id: int, num_shards: int) -> Path:
    return cache_dir / "caption_shards" / f"shard_{shard_id:05d}_of_{num_shards:05d}.jsonl"


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(str(json.loads(line)["sample_id"]))
    return completed


def generate(args: argparse.Namespace) -> None:
    dataset_root = resolve_project_path(args.dataset_root)
    cache_dir = resolve_project_path(args.cache_dir)
    model_path = resolve_project_path(args.model_path)
    records = load_records(cache_dir)
    selected = [
        record for index, record in enumerate(records)
        if index % int(args.num_shards) == int(args.shard_id)
    ]
    if args.max_samples is not None:
        selected = selected[: int(args.max_samples)]

    output_path = shard_path(cache_dir, args.shard_id, args.num_shards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    selected = [record for record in selected if record["sample_id"] not in completed]
    if not selected:
        print(f"{output_path}: complete ({len(completed)} captions)")
        return

    dtype = dtype_from_name(args.dtype)
    processor = Blip2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=dtype, local_files_only=True
    ).to(args.device).eval()

    started = time.time()
    with output_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(selected), int(args.batch_size)):
            batch = selected[offset : offset + int(args.batch_size)]
            images = []
            for record in batch:
                image_path = dataset_root / "images" / record["relative_path"]
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(args.device, dtype)
            with torch.inference_mode():
                token_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    num_beams=int(args.num_beams),
                )
            generated = processor.batch_decode(token_ids, skip_special_tokens=True)
            for record, raw_caption in zip(batch, generated):
                result = {
                    "sample_id": record["sample_id"],
                    "relative_path": record["relative_path"],
                    "split": record["split"],
                    "category_name": record["category_name"],
                    "category_label": readable_category(record["category_name"]),
                    "num_instances": int(record.get("num_instances") or 1),
                    "caption": compose_caption(record["category_name"], raw_caption),
                    "generated_detail": normalize_generated_caption(raw_caption),
                    "captioner": "Salesforce/blip2-opt-2.7b",
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            done = offset + len(batch)
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"shard {args.shard_id}/{args.num_shards}: "
                f"{done}/{len(selected)} new, {done / elapsed:.2f} images/s",
                flush=True,
            )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def merge(args: argparse.Namespace) -> None:
    cache_dir = resolve_project_path(args.cache_dir)
    records = load_records(cache_dir)
    expected_ids = [str(record["sample_id"]) for record in records]
    result_by_id: dict[str, dict[str, Any]] = {}
    for sid in range(int(args.num_shards)):
        path = shard_path(cache_dir, sid, args.num_shards)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_id = str(item["sample_id"])
                if sample_id in result_by_id:
                    raise ValueError(f"Duplicate caption for {sample_id}")
                result_by_id[sample_id] = item
    missing = [sample_id for sample_id in expected_ids if sample_id not in result_by_id]
    extra = sorted(set(result_by_id).difference(expected_ids))
    if missing or extra:
        raise ValueError(f"Caption coverage mismatch: missing={missing[:10]}, extra={extra[:10]}")

    ordered = [result_by_id[sample_id] for sample_id in expected_ids]
    jsonl_path = cache_dir / "captions.jsonl"
    temporary_jsonl = jsonl_path.with_name(f".{jsonl_path.name}.{os.getpid()}.tmp")
    with temporary_jsonl.open("w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary_jsonl.replace(jsonl_path)

    csv_path = cache_dir / "captions.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fields = [
        "sample_id", "relative_path", "split", "category_name", "category_label",
        "num_instances", "caption", "generated_detail", "captioner",
    ]
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    temporary_csv.replace(csv_path)

    if args.install_prompts:
        prompts_path = cache_dir / "prompts.pt"
        backup_path = cache_dir / "prompts_category_templates.pt"
        if prompts_path.is_file() and not backup_path.exists():
            prompts_path.replace(backup_path)
        atomic_torch_save(
            {
                "dataset": "PoseAnything_MP100",
                "captioner": "Salesforce/blip2-opt-2.7b+known-category-anchor",
                "num_samples": len(ordered),
                "sample_ids": expected_ids,
                "prompts": [item["caption"] for item in ordered],
                "splits": [item["split"] for item in ordered],
            },
            prompts_path,
        )
    print(f"Merged {len(ordered)} captions into {jsonl_path} and {csv_path}")


def main() -> None:
    args = parse_args()
    merge(args) if args.merge else generate(args)


if __name__ == "__main__":
    main()
