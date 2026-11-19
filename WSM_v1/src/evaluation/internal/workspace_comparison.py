#!/usr/bin/env python
"""Compare two WSM evaluation workspaces on matched Head contracts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = PROJECT_ROOT / "src/evaluation_output/wsm-v18-categorical-eval"
DEFAULT_CANDIDATE = PROJECT_ROOT / "src/evaluation_output/wsm-v18-channel131-local-refresh"
CATEGORICAL_TARGETS = {"111", "131", "135", "136", "195"}
METRICS = {
    "decoded_mse_mean": "lower",
    "decoded_mae_mean": "lower",
    "decoded_psnr_db_mean": "higher",
    "decoded_ssim_mean": "higher",
    "categorical_strict_pixel_accuracy_micro": "higher",
    "categorical_tolerant_pixel_accuracy_micro": "higher",
    "categorical_balanced_strict_accuracy_macro": "higher",
    "categorical_mean_iou": "higher",
    "categorical_strict_pixel_accuracy_unrefined_micro": "higher",
    "categorical_mean_iou_unrefined": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw in ("", None):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline).expanduser().resolve()
    candidate_root = Path(args.candidate).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else candidate_root / "comparison_to_imported_fallback"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv(baseline_root / "tables/head_metrics.csv")
    candidate_rows = read_csv(candidate_root / "tables/head_metrics.csv")
    baseline = {int(row["head_index"]): row for row in baseline_rows}
    candidate = {int(row["head_index"]): row for row in candidate_rows}
    if not candidate:
        raise RuntimeError("Candidate workspace has no completed Head metrics")

    baseline_samples = read_csv(baseline_root / "tables/sample_metrics.csv")
    candidate_samples = read_csv(candidate_root / "tables/sample_metrics.csv")
    sample_ids: dict[str, dict[int, set[str]]] = {
        "baseline": defaultdict(set),
        "candidate": defaultdict(set),
    }
    for label, rows in (
        ("baseline", baseline_samples),
        ("candidate", candidate_samples),
    ):
        for row in rows:
            sample_ids[label][int(row["head_index"])].add(row["sample_id"])

    compared = []
    for head_index, new in sorted(candidate.items()):
        old = baseline.get(head_index)
        if old is None:
            raise RuntimeError(f"Head {head_index:06d} missing from baseline")
        contract = ("task_type", "condition", "target")
        mismatch = [key for key in contract if old[key] != new[key]]
        if mismatch:
            raise RuntimeError(
                f"Head {head_index:06d} contract mismatch: {mismatch}"
            )
        if old["evaluation_contract_id"] != new["evaluation_contract_id"]:
            raise RuntimeError(
                f"Head {head_index:06d} evaluation-contract mismatch"
            )

        target = new["target"]
        categorical = target in CATEGORICAL_TARGETS
        primary_metric = (
            "categorical_mean_iou" if categorical else "decoded_mse_mean"
        )
        primary_direction = METRICS[primary_metric]
        old_primary = number(old, primary_metric)
        new_primary = number(new, primary_metric)
        if old_primary is None or new_primary is None:
            primary_improvement = None
        elif primary_direction == "higher":
            primary_improvement = new_primary - old_primary
        else:
            primary_improvement = old_primary - new_primary
        relative_primary_improvement = None
        if (
            primary_improvement is not None
            and old_primary is not None
            and abs(old_primary) > 1e-12
        ):
            relative_primary_improvement = (
                primary_improvement / abs(old_primary)
            )
        conditions = [item for item in new["condition"].split(",") if item]
        direction = (
            "output_131" if target == "131"
            else "input_131" if "131" in conditions
            else "other"
        )
        record: dict[str, Any] = {
            "head_index": head_index,
            "task_type": new["task_type"],
            "condition": new["condition"],
            "target": target,
            "direction": direction,
            "baseline_run": old["run"],
            "candidate_run": new["run"],
            "baseline_checkpoint_step": old["checkpoint_step"],
            "candidate_checkpoint_step": new["checkpoint_step"],
            "baseline_samples": int(old["samples"]),
            "candidate_samples": int(new["samples"]),
            "sample_ids_equal": (
                sample_ids["baseline"][head_index]
                == sample_ids["candidate"][head_index]
            ),
            "primary_metric": primary_metric,
            "primary_baseline": old_primary,
            "primary_candidate": new_primary,
            "primary_improvement": primary_improvement,
            "primary_relative_improvement": relative_primary_improvement,
        }
        for metric, direction_name in METRICS.items():
            old_value = number(old, metric)
            new_value = number(new, metric)
            delta = (
                None if old_value is None or new_value is None
                else new_value - old_value
            )
            improvement = (
                None if delta is None else
                delta if direction_name == "higher" else -delta
            )
            record[f"{metric}_baseline"] = old_value
            record[f"{metric}_candidate"] = new_value
            record[f"{metric}_delta"] = delta
            record[f"{metric}_improvement"] = improvement
        compared.append(record)

    fields = list(compared[0])
    with (output_dir / "per_head_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compared)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        improvements = [
            float(row["primary_improvement"])
            for row in rows if row["primary_improvement"] is not None
        ]
        relative = [
            float(row["primary_relative_improvement"])
            for row in rows
            if row["primary_relative_improvement"] is not None
        ]
        return {
            "heads": len(rows),
            "sample_ids_equal_heads": sum(
                bool(row["sample_ids_equal"]) for row in rows
            ),
            "improved_heads": sum(value > 1e-12 for value in improvements),
            "tied_heads": sum(abs(value) <= 1e-12 for value in improvements),
            "worsened_heads": sum(value < -1e-12 for value in improvements),
            "mean_primary_improvement": mean(improvements),
            "median_primary_improvement": (
                statistics.median(improvements) if improvements else None
            ),
            "median_relative_primary_improvement": (
                statistics.median(relative) if relative else None
            ),
        }

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in compared:
        by_target[row["target"]].append(row)
        by_direction[row["direction"]].append(row)

    target_summary = {}
    for target, rows in sorted(by_target.items()):
        metric = rows[0]["primary_metric"]
        payload = summarize(rows)
        payload.update({
            "primary_metric": metric,
            "baseline_mean": mean([
                float(row["primary_baseline"]) for row in rows
                if row["primary_baseline"] is not None
            ]),
            "candidate_mean": mean([
                float(row["primary_candidate"]) for row in rows
                if row["primary_candidate"] is not None
            ]),
        })
        target_summary[target] = payload

    summary = {
        "schema_version": 1,
        "baseline": str(baseline_root),
        "candidate": str(candidate_root),
        "evaluation_contract_id": candidate_rows[0]["evaluation_contract_id"],
        "overall": summarize(compared),
        "by_direction": {
            key: summarize(rows) for key, rows in sorted(by_direction.items())
        },
        "by_target": target_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Local channel-131 refresh versus imported fallback",
        "",
        f"- Matched Heads: {summary['overall']['heads']}",
        f"- Identical held-out sample sets: "
        f"{summary['overall']['sample_ids_equal_heads']}/"
        f"{summary['overall']['heads']}",
        f"- Primary-metric wins / ties / losses: "
        f"{summary['overall']['improved_heads']} / "
        f"{summary['overall']['tied_heads']} / "
        f"{summary['overall']['worsened_heads']}",
        "",
        "| Target | Heads | Primary metric | Fallback | Local refresh | Wins | Losses |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for target, payload in target_summary.items():
        lines.append(
            f"| {target} | {payload['heads']} | {payload['primary_metric']} | "
            f"{fmt(payload['baseline_mean'])} | "
            f"{fmt(payload['candidate_mean'])} | "
            f"{payload['improved_heads']} | {payload['worsened_heads']} |"
        )
    lines.extend((
        "",
        "Continuous-target means are reported only within each target. "
        "The overall win/loss count combines per-Head direction-correct "
        "improvements and does not average MSE across modalities.",
        "",
    ))
    (output_dir / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
