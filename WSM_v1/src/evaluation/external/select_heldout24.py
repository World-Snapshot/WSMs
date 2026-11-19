#!/usr/bin/env python3
"""Freeze a deterministic, contamination-aware, stratified 24-sample task suite."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


TASKS = frozenset({
    "conditional_rgb", "semantic_estimation", "any_to_any", "depth_estimation"
})
PATH_FIELDS = frozenset({
    "input_path", "condition_path", "target_path", "mask_path", "reference_path"
})


class SelectionError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SelectionError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        raise SelectionError(f"Invalid ISO capture/creation date: {text!r}") from exc


def perceptual_hash(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(gray.getdata())
    except (OSError, ValueError):
        return None
    value = 0
    for row in range(8):
        for column in range(8):
            value <<= 1
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value |= int(left > right)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def normalise(
    rows: list[dict[str, Any]], base: Path, min_date: dt.date | None,
    exclude_overlap: bool, require_sealed: bool
) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    output = []
    rejected: collections.Counter[str] = collections.Counter()
    seen_ids: set[str] = set()
    seen_target_hashes: set[str] = set()
    image_hashes: list[int] = []
    for original in rows:
        row = dict(original)
        sample_id = str(row.get("sample_id") or "")
        task = str(row.get("task") or "")
        if not sample_id or sample_id in seen_ids:
            rejected["duplicate_or_empty_id"] += 1
            continue
        if task not in TASKS:
            rejected["invalid_task"] += 1
            continue
        created = parse_date(
            row.get("capture_date") or row.get("creation_date") or row.get("published_date")
        )
        if min_date and (created is None or created < min_date):
            rejected["too_old_or_unknown_date"] += 1
            continue
        overlap = str(row.get("training_overlap") or "unknown").lower()
        if exclude_overlap and overlap not in {"no", "false", "none", "sealed"}:
            rejected["possible_training_overlap"] += 1
            continue
        if require_sealed and not bool(row.get("sealed_before_evaluation")):
            rejected["not_sealed"] += 1
            continue
        missing = False
        for field in PATH_FIELDS:
            if row.get(field):
                path = Path(str(row[field])).expanduser()
                path = path if path.is_absolute() else (base / path).resolve()
                row[field] = str(path)
                if not path.is_file():
                    missing = True
        target = Path(str(row.get("target_path") or ""))
        if missing or not target.is_file():
            rejected["missing_file"] += 1
            continue
        target_hash = sha256(target)
        if target_hash in seen_target_hashes:
            rejected["exact_duplicate_target"] += 1
            continue
        phash = perceptual_hash(target)
        if phash is not None and any(hamming(phash, old) <= 4 for old in image_hashes):
            rejected["near_duplicate_target"] += 1
            continue
        seen_ids.add(sample_id)
        seen_target_hashes.add(target_hash)
        if phash is not None:
            image_hashes.append(phash)
        row["target_sha256"] = target_hash
        row["selection_creation_date"] = created.isoformat() if created else ""
        output.append(row)
    return output, rejected


def values_for(row: Mapping[str, Any], fields: list[str]) -> tuple[str, ...]:
    metadata = row.get("strata") if isinstance(row.get("strata"), Mapping) else {}
    values = []
    for field in fields:
        value = row.get(field)
        if value in (None, ""):
            value = metadata.get(field, "unknown")
        values.append(str(value))
    return tuple(values)


def select_task(
    candidates: list[dict[str, Any]], count: int, fields: list[str], seed: int
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise SelectionError(f"Need {count} candidates, only {len(candidates)} remain")
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    global_frequency = collections.Counter(
        (field, value)
        for row in shuffled
        for field, value in zip(fields, values_for(row, fields))
    )
    selected = []
    selected_frequency: collections.Counter[tuple[str, str]] = collections.Counter()
    while len(selected) < count:
        best_index, best_score = -1, None
        for index, row in enumerate(shuffled):
            values = values_for(row, fields)
            diversity = 0.0
            rarity = 0.0
            for field, value in zip(fields, values):
                key = (field, value)
                diversity += 1.0 / (1.0 + selected_frequency[key])
                rarity += 1.0 / max(1, global_frequency[key])
            private_bonus = 1.0 if str(
                row.get("provenance_visibility") or ""
            ).lower() in {"private", "sealed", "not_public"} else 0.0
            date_value = parse_date(row.get("selection_creation_date"))
            recency = date_value.toordinal() / 1_000_000.0 if date_value else 0.0
            score = diversity * 100.0 + rarity * 10.0 + private_bonus + recency
            if best_score is None or score > best_score:
                best_index, best_score = index, score
        chosen = shuffled.pop(best_index)
        selected.append(chosen)
        for field, value in zip(fields, values_for(chosen, fields)):
            selected_frequency[(field, value)] += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default="")
    parser.add_argument("--count-per-task", type=int, default=24)
    parser.add_argument("--tasks", default=",".join(sorted(TASKS)))
    parser.add_argument(
        "--strata",
        default="indoor_outdoor,human_nonhuman,texture_complexity,condition_sparsity"
    )
    parser.add_argument("--min-created-date", default="")
    parser.add_argument("--exclude-possible-overlap", action="store_true")
    parser.add_argument("--require-sealed", action="store_true")
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    audit_path = (
        Path(args.audit).expanduser().resolve()
        if args.audit else output_path.with_suffix(".selection_audit.json")
    )
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    unknown = tasks - TASKS
    if unknown:
        raise SelectionError(f"Unknown tasks: {sorted(unknown)}")
    fields = [item.strip() for item in args.strata.split(",") if item.strip()]
    minimum = parse_date(args.min_created_date)
    candidates, rejected = normalise(
        read_jsonl(candidate_path), candidate_path.parent, minimum,
        args.exclude_possible_overlap, args.require_sealed
    )
    selected = []
    task_audit = {}
    for task in sorted(tasks):
        pool = [row for row in candidates if str(row["task"]) == task]
        chosen = select_task(
            pool, args.count_per_task, fields,
            args.seed + int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "big")
        )
        selected.extend(chosen)
        task_audit[task] = {
            "eligible": len(pool),
            "selected": len(chosen),
            "strata": {
                field: dict(collections.Counter(
                    values_for(row, fields)[index] for row in chosen
                ))
                for index, field in enumerate(fields)
            }
        }
    selected.sort(key=lambda row: (str(row["task"]), str(row["sample_id"])))
    atomic_jsonl(output_path, selected)
    audit = {
        "schema_version": 1,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "frozen_manifest": str(output_path),
        "frozen_manifest_sha256": sha256(output_path),
        "selection_happened_before_inference": True,
        "same_samples_for_every_method_within_task": True,
        "count_per_task": args.count_per_task,
        "tasks": task_audit,
        "strata_fields": fields,
        "minimum_creation_date": args.min_created_date,
        "exclude_possible_overlap": args.exclude_possible_overlap,
        "require_sealed": args.require_sealed,
        "seed": args.seed,
        "rejected": dict(rejected)
    }
    atomic_text(audit_path, json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(f"Frozen {len(selected)} samples: {output_path}")
    print(f"Selection audit: {audit_path}")


if __name__ == "__main__":
    main()
