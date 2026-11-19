#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


CHANNEL_RE = re.compile(r"^\s*-\s*(\d+)(?:\s*-\s*(\d+))?\s*:\s*(.*)$")


def compact_channels(channels):
    ordered = sorted(set(channels))
    if not ordered:
        return None

    ranges = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))

    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return ",".join(parts)


def parse_modalities(markdown_text):
    modalities = []
    current_channels = None
    current_descriptions = None

    def flush_current():
        nonlocal current_channels, current_descriptions
        key = compact_channels(current_channels or [])
        if key is not None:
            modalities.append(
                {
                    "key": key,
                    "description": " | ".join(current_descriptions or []),
                }
            )
        current_channels = None
        current_descriptions = None

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("### "):
            flush_current()
            if line.strip() == "### wsg vae":
                current_channels = []
                current_descriptions = []
            continue

        match = CHANNEL_RE.match(line)
        if current_channels is not None and match:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            description = match.group(3).strip()
            if end < start:
                raise ValueError(f"Invalid channel range: {line}")
            current_channels.extend(range(start, end + 1))
            if description:
                if start == end:
                    current_descriptions.append(f"{start}: {description}")
                else:
                    current_descriptions.append(f"{start}-{end}: {description}")

    flush_current()
    return modalities


def build_graph(modalities):
    keys = [modality["key"] for modality in modalities]
    return {
        "wsg_v1": {
            modality["key"]: {
                "description": modality["description"],
                "conditions": {
                    condition: {}
                    for condition in keys
                    if condition != modality["key"]
                },
            }
            for modality in modalities
        }
    }


def dump_compact_conditions(graph):
    lines = ["{", '  "wsg_v1": {']
    items = list(graph["wsg_v1"].items())
    for idx, (modality, spec) in enumerate(items):
        modality_comma = "," if idx + 1 < len(items) else ""
        conditions = json.dumps(spec["conditions"], ensure_ascii=False, separators=(", ", ": "))
        description = json.dumps(spec["description"], ensure_ascii=False)
        lines.extend(
            [
                f'    "{modality}": {{',
                f'      "description": {description},',
                f'      "conditions": {conditions}',
                f"    }}{modality_comma}",
            ]
        )
    lines.extend(["  }", "}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate WSG v1 modality-condition JSON from channel_wsg_distribution.md."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("channel_wsg_distribution.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("wsg_v1_modalities.json"),
    )
    args = parser.parse_args()

    markdown_text = args.input.read_text(encoding="utf-8")
    modalities = parse_modalities(markdown_text)
    graph = build_graph(modalities)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump_compact_conditions(graph), encoding="utf-8")
    print(f"Wrote {args.output} with {len(modalities)} modalities.")


if __name__ == "__main__":
    main()
