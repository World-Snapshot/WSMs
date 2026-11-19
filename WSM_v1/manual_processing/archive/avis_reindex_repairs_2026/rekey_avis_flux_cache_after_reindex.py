#!/usr/bin/env python3
"""Rekey AVIS Flux/text cache metadata after WSV frame indices are reindexed.

This tool changes only metadata fields such as sample_ids, sample_sources,
frame_keys, and frame_indices. Latent/text tensors are intentionally left in
place. It is intended for the legacy AVIS lexicographic crop_frame bug:
old current index -> true AVIS JSON/natural index.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

DONE_MARKER = ".avis_frame_rekey_after_reindex_done.json"

SAMPLE_RE = re.compile(r"(?P<prefix>.*?/AVIS_(?P<split>train|val|test)_results/(?P<video>.*?)/wsv/merge\.wsv)::index_(?P<a>\d+)-index_(?P<b>\d+)$")
KEY_RE = re.compile(r"index_(\d+)$")


def load_split_mapping(json_path: Path) -> Dict[str, Dict[int, int]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    result: Dict[str, Dict[int, int]] = {}
    for video in payload.get("videos", []):
        file_names = video.get("file_names", [])
        if not file_names:
            continue
        video_name = file_names[0].split("/", 1)[0]
        original_index = {name: idx + 1 for idx, name in enumerate(file_names)}
        lexicographic = sorted(file_names)
        result[video_name] = {old_idx + 1: original_index[name] for old_idx, name in enumerate(lexicographic)}
    return result


def load_mappings(base_output_dir: Path) -> Dict[str, Dict[str, Dict[int, int]]]:
    return {
        "train": load_split_mapping(base_output_dir / "AVIS_train_results" / "train.json"),
        "val": load_split_mapping(base_output_dir / "AVIS_val_results" / "val.json"),
        "test": load_split_mapping(base_output_dir / "AVIS_test_results" / "test.json"),
    }


def map_index(split: str, video: str, old_index: int, mappings: Dict[str, Dict[str, Dict[int, int]]]) -> int:
    return int(mappings.get(split, {}).get(video, {}).get(old_index, old_index))


def rekey_frame_key(key: Any, split: Optional[str], video: Optional[str], mappings: Dict[str, Dict[str, Dict[int, int]]]) -> Any:
    if not isinstance(key, str) or split is None or video is None:
        return key
    m = KEY_RE.fullmatch(key)
    if not m:
        return key
    new_idx = map_index(split, video, int(m.group(1)), mappings)
    return f"index_{new_idx:08d}"


def parse_sample_id(sample_id: Any) -> Optional[Tuple[str, str, str, int, int]]:
    if not isinstance(sample_id, str):
        return None
    m = SAMPLE_RE.fullmatch(sample_id)
    if not m:
        return None
    return m.group("prefix"), m.group("split"), m.group("video"), int(m.group("a")), int(m.group("b"))


def rekey_sample_id(sample_id: Any, mappings: Dict[str, Dict[str, Dict[int, int]]]) -> Tuple[Any, bool, Optional[str], Optional[str]]:
    parsed = parse_sample_id(sample_id)
    if parsed is None:
        return sample_id, False, None, None
    prefix, split, video, a, b = parsed
    new_a = map_index(split, video, a, mappings)
    new_b = map_index(split, video, b, mappings)
    new_id = f"{prefix}::index_{new_a:08d}-index_{new_b:08d}"
    return new_id, new_id != sample_id, split, video


def rekey_payload(payload: Dict[str, Any], mappings: Dict[str, Dict[str, Dict[int, int]]]) -> Dict[str, int]:
    changed = 0
    seen_context: list[Tuple[Optional[str], Optional[str]]] = []

    if isinstance(payload.get("sample_ids"), list):
        new_ids = []
        for sample_id in payload["sample_ids"]:
            new_id, did, split, video = rekey_sample_id(sample_id, mappings)
            new_ids.append(new_id)
            seen_context.append((split, video))
            changed += int(did)
        payload["sample_ids"] = new_ids

    if isinstance(payload.get("sample_sources"), list):
        for i, source in enumerate(payload["sample_sources"]):
            if not isinstance(source, dict):
                continue
            split = video = None
            if isinstance(source.get("sample_id"), str):
                new_id, did, split, video = rekey_sample_id(source["sample_id"], mappings)
                source["sample_id"] = new_id
                changed += int(did)
            if split is None or video is None:
                if i < len(seen_context):
                    split, video = seen_context[i]
            if isinstance(source.get("frame_keys"), list):
                new_keys = [rekey_frame_key(k, split, video, mappings) for k in source["frame_keys"]]
                changed += int(new_keys != source["frame_keys"])
                source["frame_keys"] = new_keys

    # Dense latent shard metadata.
    if isinstance(payload.get("frame_keys"), list):
        new_keys = []
        for i, key in enumerate(payload["frame_keys"]):
            split = video = None
            if i < len(seen_context):
                split, video = seen_context[i]
            new_keys.append(rekey_frame_key(key, split, video, mappings))
        changed += int(new_keys != payload["frame_keys"])
        payload["frame_keys"] = new_keys

    if isinstance(payload.get("frame_indices"), list):
        new_indices = []
        for i, idx in enumerate(payload["frame_indices"]):
            split = video = None
            if i < len(seen_context):
                split, video = seen_context[i]
            if split is not None and video is not None:
                try:
                    new_indices.append(map_index(split, video, int(idx), mappings))
                except Exception:
                    new_indices.append(idx)
            else:
                new_indices.append(idx)
        changed += int(new_indices != payload["frame_indices"])
        payload["frame_indices"] = new_indices

    return {"changed_fields_or_items": changed}


def rekey_file(path_s: str, base_output_dir_s: str, dry_run: bool) -> Dict[str, Any]:
    path = Path(path_s)
    mappings = load_mappings(Path(base_output_dir_s))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {"path": str(path), "changed": 0, "skipped": "not_dict"}
    stats = rekey_payload(payload, mappings)
    changed = int(stats["changed_fields_or_items"])
    if changed and not dry_run:
        tmp = path.with_name(path.name + ".rekey.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
    return {"path": str(path), "changed": changed, "dry_run": dry_run}


def discover_pt_files(roots: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".pt":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.pt"))
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, action="append", required=True)
    ap.add_argument("--base-output-dir", type=Path, default=Path("/common/users/sf895/wave2pixel"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Allow running when a cache-root completion marker exists.")
    args = ap.parse_args()

    markers = [root / DONE_MARKER for root in args.cache_root if root.is_dir()]
    existing_markers = [marker for marker in markers if marker.exists()]
    if existing_markers and not args.force:
        raise SystemExit(
            "Refusing to run because AVIS frame rekey completion marker exists. "
            f"Use --force only if you intentionally want to reprocess: {existing_markers}"
        )

    files = discover_pt_files(args.cache_root)
    print(json.dumps({"files": len(files), "dry_run": args.dry_run, "workers": args.workers}, indent=2), flush=True)
    totals = {"files": 0, "changed": 0}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(rekey_file, str(path), str(args.base_output_dir), bool(args.dry_run)) for path in files]
        for fut in as_completed(futs):
            result = fut.result()
            totals["files"] += 1
            totals["changed"] += int(result.get("changed", 0))
            print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps(totals, indent=2, sort_keys=True), flush=True)
    if not args.dry_run:
        for marker in markers:
            marker.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
