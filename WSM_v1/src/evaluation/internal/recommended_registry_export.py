#!/usr/bin/env python
"""Freeze evaluated dense Channel Heads into a portable recommendation table."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.internal.channel_head_evaluator import (  # noqa: E402
    HeadCandidate,
    discover_candidates,
    load_recommended_candidates,
)


DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "src/evaluation_output/wsm-v18-categorical-eval/catalog.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "configs/wsm_v1_recommended_dense_channel_heads.json"
)

# These categorical input--output relations were withheld from the compact
# paper matrices under a conservative end-to-end review. The official checkpoint
# remains deployable and recommended; the notice marks an upgrade priority
# without silently substituting a run.
PHYSICAL_INPUT_MODALITIES = frozenset({
    "111", "126", "127", "128", "129", "130", "131", "132", "133",
    "134", "135", "136", "143", "144-146", "147-149", "150",
    "151-153", "154",
})
VISUAL_INPUT_MODALITIES = frozenset({
    "103", "195", "32-33", "34-35", "39", "40", "41-42", "67-69",
    "90-92", "99-101",
})
BRIDGE_INPUT_MODALITIES = frozenset({"3", "36-38"})
PHYSICAL_TARGETS_NEEDING_IMPROVEMENT = frozenset({"111", "131", "136"})
VISUAL_TARGETS_NEEDING_IMPROVEMENT = frozenset()
LOW_RELEVANCE_RELATIONS = frozenset({
    ("conditional_control", "151-153", "135"),
    ("conditional_control", "154", "135"),
})
QUALITY_REVIEW_TASK_TYPES = frozenset({
    "conditional_control", "inverse_rendering",
})
QUALITY_REVIEW_DATE = "2026-08-20"
PRESERVED_MANUAL_NOTICE_CODES = frozenset({"invalid_condition_dataset_routing"})


def quality_notice_for(candidate: HeadCandidate) -> dict[str, Any] | None:
    """Return the reviewed deployment notice for one exact Head contract."""
    if candidate.task_type not in QUALITY_REVIEW_TASK_TYPES:
        return None
    source = str(candidate.condition[0]) if candidate.condition else "caption"
    target = str(candidate.target)
    physical_relation = (
        source in PHYSICAL_INPUT_MODALITIES
        and target in PHYSICAL_TARGETS_NEEDING_IMPROVEMENT
    )
    visual_relation = (
        source in VISUAL_INPUT_MODALITIES
        and target in VISUAL_TARGETS_NEEDING_IMPROVEMENT
    )
    bridge_relation = (
        source in BRIDGE_INPUT_MODALITIES
        and target in (
            PHYSICAL_TARGETS_NEEDING_IMPROVEMENT
            | VISUAL_TARGETS_NEEDING_IMPROVEMENT
        )
    )
    low_relevance_relation = (
        candidate.task_type, source, target
    ) in LOW_RELEVANCE_RELATIONS
    if not (physical_relation or visual_relation or bridge_relation or low_relevance_relation):
        return None
    return {
        "status": "needs_improvement",
        "severity": "warning",
        "code": "reviewed_variable_end_to_end_categorical_quality",
        "title": "Further improvement recommended",
        "message": (
            "Current end-to-end evaluation indicates that this input-output "
            "relation has not yet reached uniformly strong and stable "
            "performance across the benchmark. Some cases are already useful; "
            "low modality relevance and/or task-specific difficulty may reduce "
            "consistency. Review the result before deployment."
        ),
        "reason_codes": [
            "low_modality_relevance_or_unresolved_task_difficulty",
            "variable_post_refiner_categorical_quality",
        ],
        "review_source": "WSM v1 quantitative review",
        "reviewed_at": QUALITY_REVIEW_DATE,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--log-root", default=str(PROJECT_ROOT / "train_log"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_from_channel_head(path: str) -> Path:
    parts = Path(path).parts
    try:
        index = next(
            i for i, part in enumerate(parts)
            if part.startswith("Channel_Head_")
        )
    except StopIteration as exc:
        raise ValueError(f"No Channel_Head component in {path}") from exc
    return Path(*parts[index:])


def relative_candidate(candidate: HeadCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    for key in ("checkpoint", "run_dir", "runtime_config", "log_root"):
        payload[key] = str(Path(payload[key]).resolve().relative_to(PROJECT_ROOT))
    return payload


def localize_evaluated_candidate(
    candidate_payload: dict[str, Any],
    log_root: Path,
) -> HeadCandidate | None:
    raw = dict(candidate_payload)
    checkpoint_tail = path_from_channel_head(str(raw["checkpoint"]))
    run_tail = Path(*checkpoint_tail.parts[:2])
    checkpoint = log_root / checkpoint_tail
    run_dir = log_root / run_tail
    runtime_config = run_dir / "resolved_runtime_config.json"
    if not checkpoint.is_file() or not runtime_config.is_file():
        return None
    raw.update({
        "checkpoint": str(checkpoint.resolve()),
        "run_dir": str(run_dir.resolve()),
        "runtime_config": str(runtime_config.resolve()),
        "log_root": str(log_root.resolve()),
        "condition": tuple(str(item) for item in raw.get("condition") or []),
        "target_dataset_whitelist": tuple(
            str(item) for item in raw.get("target_dataset_whitelist") or []
        ),
        "default_mismatches": tuple(
            str(item) for item in raw.get("default_mismatches") or []
        ),
    })
    return HeadCandidate(**raw)


def contract_key(candidate: HeadCandidate) -> tuple[str, int]:
    return candidate.task_type, int(candidate.head_index)


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    preserved_manual_notices = {}
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        preserved_manual_notices = {
            (str(item.get("task_type") or ""), int(item.get("head_index") or -1)): dict(item["quality_notice"])
            for item in previous.get("entries") or []
            if str((item.get("quality_notice") or {}).get("code") or "")
            in PRESERVED_MANUAL_NOTICE_CODES
        }
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_entries = catalog.get("entries") or []

    discovered, discovery_audit = discover_candidates(
        [log_root],
        selection_policy="canonical-coverage",
    )
    discovered_by_key = {contract_key(item): item for item in discovered}

    entries = []
    problems = []
    for record in catalog_entries:
        evaluated_raw = dict(record.get("candidate") or {})
        evaluated_head = int(evaluated_raw["head_index"])
        key = (str(evaluated_raw["task_type"]), evaluated_head)
        exact = localize_evaluated_candidate(evaluated_raw, log_root)
        if exact is not None:
            recommended = exact
            status = "exact_evaluated_run"
        else:
            recommended = discovered_by_key.get(key)
            status = "contract_compatible_local_recommendation"
        if recommended is None:
            problems.append(f"Head {evaluated_head:06d}: no local recommendation")
            continue
        evaluated_condition = tuple(
            str(item) for item in evaluated_raw.get("condition") or []
        )
        mismatches = []
        if recommended.target != str(evaluated_raw["target"]):
            mismatches.append("target")
        if tuple(recommended.condition) != evaluated_condition:
            mismatches.append("condition")
        if recommended.task_type != str(evaluated_raw["task_type"]):
            mismatches.append("task_type")
        if mismatches:
            problems.append(
                f"Head {evaluated_head:06d}: contract mismatch {mismatches}"
            )
            continue

        evaluated_checkpoint = path_from_channel_head(
            str(evaluated_raw["checkpoint"])
        )
        evaluated_run = Path(*evaluated_checkpoint.parts[:2])
        run_dir = Path(recommended.run_dir)
        sidecar_dir = run_dir / "categorical_refiner"
        sidecar_best = sidecar_dir / "best.pt"
        sidecar_manifest = sidecar_dir / "manifest.json"
        entry_payload = {
            "head_source": recommended.head_source,
            "head_index": evaluated_head,
            "task_type": recommended.task_type,
            "condition": list(recommended.condition),
            "target": recommended.target,
            "recommendation_status": status,
            "evaluated": {
                "candidate_cache_key": record.get("candidate_cache_key"),
                "run_relative": str(Path("train_log") / evaluated_run),
                "checkpoint_relative": str(
                    Path("train_log") / evaluated_checkpoint
                ),
                "available_locally": exact is not None,
            },
            "recommended": {
                "candidate": relative_candidate(recommended),
                "categorical_refiner_relative": (
                    str(sidecar_dir.resolve().relative_to(PROJECT_ROOT))
                    if sidecar_dir.is_dir() else None
                ),
                "categorical_refiner_best_available": sidecar_best.is_file(),
                "categorical_refiner_manifest_available": sidecar_manifest.is_file(),
            },
        }
        quality_notice = (
            quality_notice_for(recommended)
            or preserved_manual_notices.get(key)
        )
        if quality_notice is not None:
            entry_payload["quality_notice"] = quality_notice
        entries.append(entry_payload)

    if problems:
        raise RuntimeError(
            f"Could not freeze {len(problems)} entries:\n"
            + "\n".join(problems[:50])
        )
    entries.sort(key=lambda item: (item["task_type"], item["head_index"]))
    if len(entries) != len(catalog_entries):
        raise RuntimeError(
            f"Registry coverage mismatch: {len(entries)} != {len(catalog_entries)}"
        )
    status_counts = Counter(item["recommendation_status"] for item in entries)
    task_counts = Counter(item["task_type"] for item in entries)
    target_counts = Counter(item["target"] for item in entries)
    quality_entries = [item for item in entries if item.get("quality_notice")]
    quality_target_counts = Counter(item["target"] for item in quality_entries)
    quality_task_counts = Counter(item["task_type"] for item in quality_entries)
    contract_id = str(catalog.get("evaluation_contract_id") or "")
    payload = {
        "schema_version": 3,
        "name": "WSM v1 Augustus official recommended dense Channel Heads",
        "head_source_default": "dense",
        "method_identity": (
            "WSM Channel Heads are routed task modules within TunnelingNet. "
            "PEFT-compatible checkpoint filenames describe the storage backend, "
            "not the identity or capability of the method."
        ),
        "generated_at": datetime.now().astimezone().isoformat(),
        "project_root_contract": "All model paths are relative to WSM_v1.",
        "source_evaluation": str(catalog_path.relative_to(PROJECT_ROOT)),
        "source_catalog_sha256": sha256_file(catalog_path),
        "evaluation_contract_id": contract_id,
        "policy": {
            "primary": (
                "Freeze the exact dense runs used by the complete "
                "1,148-Head evaluation. Sparse and custom recommendations live "
                "directly in their compact modality configuration files."
            ),
            "missing_imported_run": (
                "For imported evaluation runs absent locally, freeze one "
                "contract-compatible canonical-coverage run now; future "
                "evaluation must not rediscover it silently."
            ),
            "update": (
                "Edit or regenerate this reviewed file explicitly when an "
                "official Head changes."
            ),
        },
        "quality_review": {
            "default_status": "recommended",
            "marked_status": "needs_improvement",
            "marked_entries": len(quality_entries),
            "by_target": dict(quality_target_counts),
            "by_task_type": dict(quality_task_counts),
            "criteria": (
                "Reviewed categorical input-output relations withheld from the "
                "compact paper matrices because their end-to-end performance "
                "is not yet uniformly strong and stable."
            ),
            "review_source": (
                "WSM v1 quantitative review"
            ),
            "reviewed_at": QUALITY_REVIEW_DATE,
        },
        "counts": {
            "entries": len(entries),
            "by_head_source": dict(Counter(item.get("head_source", "dense") for item in entries)),
            "by_recommendation_status": dict(status_counts),
            "by_task_type": dict(task_counts),
            "by_target": dict(target_counts),
            "quality_needs_improvement": len(quality_entries),
            "categorical_refiners_available": sum(
                bool(
                    (item["recommended"] or {}).get(
                        "categorical_refiner_best_available"
                    )
                )
                for item in entries
            ),
        },
        "discovery_audit_at_freeze": discovery_audit,
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)

    loaded, audit = load_recommended_candidates(output_path)
    if len(loaded) != len(entries):
        raise RuntimeError("Registry round-trip validation failed")
    print(json.dumps({
        "output": str(output_path),
        "entries": len(entries),
        "status_counts": dict(status_counts),
        "task_counts": dict(task_counts),
        "categorical_refiners_available": payload["counts"][
            "categorical_refiners_available"
        ],
        "loader_audit": audit,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
