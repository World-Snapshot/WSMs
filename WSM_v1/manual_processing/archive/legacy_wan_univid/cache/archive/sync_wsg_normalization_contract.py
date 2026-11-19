#!/usr/bin/env python3
"""Sync portable channel ranges from wsg_dataset_info.json into a modality config."""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality-config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", action="append", default=[])
    return parser.parse_args()


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_channels(modality):
    channels = []
    for part in str(modality).split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            channels.extend(range(int(start), int(end) + 1))
        else:
            channels.append(int(part))
    return channels


NON_LATENT_CONDITION_TOKENS = {"caption", "Reference_IMG"}


def _condition_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def collect_modalities(payload):
    modalities = set()
    for key, group in payload["wsg_v1"].items():
        if not isinstance(group, dict):
            modalities.add(str(key))
            continue
        if "target" in group and "condition" in group:
            modalities.add(str(group["target"]))
            modalities.update(
                item for item in _condition_list(group.get("condition"))
                if item not in NON_LATENT_CONDITION_TOKENS
            )
            continue
        if "conditions" in group or "description" in group:
            modalities.add(str(key))
            modalities.update(str(item) for item in (group.get("conditions") or {}))
            continue
        for target, spec in group.items():
            if str(target).startswith("__"):
                continue
            modalities.add(str(target))
            if isinstance(spec, dict):
                modalities.update(str(item) for item in (spec.get("conditions") or {}))
    return sorted(modalities)


def aggregate_ranges(cache, dataset_paths, channels):
    ranges = {"0": [0.0, 255.0], "1": [0.0, 255.0], "2": [0.0, 255.0]}
    for channel in channels:
        if channel in (0, 1, 2):
            continue
        minimum = None
        maximum = None
        for dataset_path in dataset_paths:
            entry = cache.get(str(resolve(dataset_path)), cache.get(str(dataset_path)))
            normalization = (
                (entry or {}).get("channels", {}).get(str(channel), {}).get("normalization")
            )
            if (
                not isinstance(normalization, dict)
                or "min_val" not in normalization
                or "max_val" not in normalization
            ):
                continue
            current_min = float(normalization["min_val"])
            current_max = float(normalization["max_val"])
            minimum = current_min if minimum is None else min(minimum, current_min)
            maximum = current_max if maximum is None else max(maximum, current_max)
        if minimum is not None and maximum is not None:
            ranges[str(channel)] = [minimum, maximum]
    return ranges


def find_object_span(text, key):
    marker = f'"{key}"'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find top-level {marker}")
    brace = text.find("{", start + len(marker))
    depth = 0
    in_string = False
    escaped = False
    for index in range(brace, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return brace, index + 1
    raise ValueError(f"Unclosed object for {marker}")


def main():
    args = parse_args()
    config_path = resolve(args.modality_config)
    cache_path = resolve(args.cache)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"{config_path} is missing top-level normalization contract")
    channels = sorted(
        {
            channel
            for modality in collect_modalities(payload)
            for channel in parse_channels(modality)
        }
    )
    existing_ranges = normalization.get("channel_ranges") or {}
    aggregated_ranges = aggregate_ranges(cache, args.dataset_path, channels)
    for channel in channels:
        key = str(channel)
        if key not in aggregated_ranges and key in existing_ranges:
            aggregated_ranges[key] = existing_ranges[key]
    normalization["channel_ranges"] = aggregated_ranges

    text = config_path.read_text(encoding="utf-8")
    start, end = find_object_span(text, "normalization")
    replacement_lines = json.dumps(normalization, indent=4, ensure_ascii=False).splitlines()
    replacement_lines[-1] = "  " + replacement_lines[-1]
    replacement = "\n".join(replacement_lines)
    updated = text[:start] + replacement + text[end:]
    json.loads(updated)
    config_path.write_text(updated, encoding="utf-8")
    print(f"updated {config_path}: {len(normalization['channel_ranges'])} channel ranges")


if __name__ == "__main__":
    main()
