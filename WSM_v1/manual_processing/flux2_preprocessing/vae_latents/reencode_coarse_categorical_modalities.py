#!/usr/bin/env python3
"""Maintain the coarse categorical outputs for WSG modalities 111, 131, and 136.

Current-release scope
---------------------
The experiments associated with this script do *not* re-encode the underlying
VAE latent shards and do not retrain the base Channel Heads. The released
patch folds the existing fine-grained categorical supervision sidecars into a
smaller coarse label space; the associated training job then retrains only the
small categorical refiners. Consequently, the original Heads still generate
the mask regions, while the final output classifier chooses among fewer,
broader categories.

The more complete--but substantially more expensive--alternative is to rebuild
the source modality data under the coarse ontology, regenerate its VAE
latents, and retrain every affected base Head. That path is intentionally not
claimed by the current results. It requires ``--from-raw --include-latents``
followed by base-Head retraining; the default path below modifies categorical
sidecars only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENCODER = Path(__file__).with_name("precompute_wsg_flux_frame_latents.py")
ALLOWED_MODALITIES = ("111", "131", "136")
EXPECTED_CLASS_COUNTS = {"111": 10, "131": 6, "136": 10}
EXPECTED_ONTOLOGY_REVISIONS = {
    "111": "wsg_v1_object_asset_coarse10_v1",
    "131": "wsg_v1_material_coarse6_v1",
    "136": "wsg_v1_thing_physical_object_coarse10_v1",
}
ONTOLOGY_METADATA = {
    "111": {
        "semantic_name": "Object/Asset Category",
        "ontology_revision": EXPECTED_ONTOLOGY_REVISIONS["111"],
        "class_count": 10,
    },
    "131": {
        "semantic_name": "Material Class",
        "ontology_revision": EXPECTED_ONTOLOGY_REVISIONS["131"],
        "class_count": 6,
    },
    "136": {
        "semantic_name": "Thing/Physical Object Category",
        "ontology_revision": EXPECTED_ONTOLOGY_REVISIONS["136"],
        "class_count": 10,
    },
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wsg_reader.categorical_visualization import (  # noqa: E402
    CHANNEL_111_ID_TO_CATEGORY,
    COARSE_MATERIAL_CATEGORY_COLORS,
    COARSE_OBJECT_CATEGORY_COLORS,
    coarse_material_category_id,
    coarse_object_category_id,
    coarse_source_object_category_id,
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.coarse-remap.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source_path(payload: dict, row_index: int) -> str:
    sources = payload.get("sample_sources") or []
    if row_index < len(sources):
        source = sources[row_index]
        if isinstance(source, dict):
            for key in ("source_path", "wsg_path", "sample_id"):
                if source.get(key):
                    return str(source[key])
        elif source:
            return str(source)
    return str(payload["sample_ids"][row_index])


def _fine_to_coarse_lookup(
    modality: str,
    fine_values: torch.Tensor,
    source_path: str,
) -> torch.Tensor:
    if modality == "111":
        coarse = [
            coarse_object_category_id(CHANNEL_111_ID_TO_CATEGORY.get(int(value), "unknown"))
            for value in fine_values.tolist()
        ]
    elif modality == "131":
        coarse = [coarse_material_category_id(int(value)) for value in fine_values.tolist()]
    elif modality == "136":
        coarse = [
            coarse_source_object_category_id(int(value), source_path)
            for value in fine_values.tolist()
        ]
    else:
        raise ValueError(modality)
    return torch.as_tensor(coarse, dtype=torch.long)


def _repair_empty_row(
    modality: str,
    latent_payload: dict,
    row_index: int,
    spatial_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-read only a legacy row whose old category sidecar had no region."""
    from manual_processing.flux2_preprocessing.vae_latents.precompute_wsg_flux_frame_latents import (
        _categorical_contract_arrays,
        _load_cached_source_frame,
        _raw_categorical_map,
    )

    frame, source_path = _load_cached_source_frame(latent_payload, row_index)
    crop_infos = latent_payload.get("crop_infos") or []
    crop = crop_infos[row_index] if row_index < len(crop_infos) else None
    semantic = _raw_categorical_map(modality, frame, source_path, crop)
    return _categorical_contract_arrays(modality, semantic, spatial_size)


def remap_existing_sidecars(
    output_dir: Path,
    num_shards: int,
    spatial_size: int = 60,
) -> None:
    """Fold the existing fine labels into coarse classes without touching VAE latents."""
    for modality in ALLOWED_MODALITIES:
        class_count = EXPECTED_CLASS_COUNTS[modality]
        palette = (
            COARSE_MATERIAL_CATEGORY_COLORS
            if modality == "131"
            else COARSE_OBJECT_CATEGORY_COLORS
        )
        prototypes = torch.as_tensor(
            [palette[index] for index in range(class_count)], dtype=torch.uint8
        )
        for shard_id in range(num_shards):
            filename = f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
            latent_path = output_dir / "shards" / modality / filename
            sidecar_path = output_dir / "categorical_targets" / modality / filename
            latent = torch.load(latent_path, map_location="cpu", mmap=True, weights_only=False)
            sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            if list(sidecar["sample_ids"]) != list(latent["sample_ids"]):
                raise RuntimeError(f"Sample alignment mismatch: {sidecar_path}")
            revision = (sidecar.get("contract") or {}).get("ontology_revision")
            if revision == EXPECTED_ONTOLOGY_REVISIONS[modality]:
                print(f"[coarse-remap] already current: {sidecar_path}", flush=True)
                continue

            fine_values = torch.as_tensor(sidecar["category_values"], dtype=torch.long)
            old_spatial = torch.as_tensor(sidecar["category_spatial_index_map"], dtype=torch.long)
            if tuple(old_spatial.shape[1:]) != (spatial_size, spatial_size):
                raise RuntimeError(f"Unexpected spatial shape in {sidecar_path}: {old_spatial.shape}")
            row_count = len(sidecar["sample_ids"])
            new_spatial = torch.zeros((row_count, spatial_size, spatial_size), dtype=torch.uint8)
            new_multihot = torch.zeros((row_count, class_count), dtype=torch.uint8)
            new_counts = torch.zeros((row_count, class_count), dtype=torch.int32)
            repaired = []
            failures = []
            lookup_cache: dict[str, torch.Tensor] = {}
            for row_index in range(row_count):
                source_path = _source_path(latent, row_index)
                source_family = (
                    "ShapeNetSem" if "ShapeNetSem" in source_path
                    else "Objaverse" if "Objaverse" in source_path or "Articulation" in source_path
                    else source_path
                )
                lookup = lookup_cache.get(source_family)
                if lookup is None:
                    lookup = _fine_to_coarse_lookup(modality, fine_values, source_path)
                    lookup_cache[source_family] = lookup
                old_row = old_spatial[row_index]
                valid = (old_row > 0) & (old_row <= len(fine_values))
                if bool(valid.any()):
                    new_spatial[row_index][valid] = (
                        lookup[old_row[valid] - 1] + 1
                    ).to(torch.uint8)
                else:
                    try:
                        multihot, counts, spatial = _repair_empty_row(
                            modality, latent, row_index, spatial_size
                        )
                        new_multihot[row_index] = multihot
                        new_counts[row_index] = counts
                        new_spatial[row_index] = spatial.to(torch.uint8)
                        repaired.append(str(sidecar["sample_ids"][row_index]))
                        continue
                    except Exception as exc:
                        failures.append(
                            {
                                "sample_id": str(sidecar["sample_ids"][row_index]),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                counts = torch.bincount(
                    new_spatial[row_index].reshape(-1).long(), minlength=class_count + 1
                )[1 : class_count + 1]
                new_counts[row_index] = counts.to(torch.int32)
                new_multihot[row_index] = (counts > 0).to(torch.uint8)

            contract = {
                "version": 3,
                "kind": "semantic_categorical_target",
                "alignment": "latent_shard_sample_ids",
                "source": "fine_sidecar_to_coarse_ontology_remap",
                "runtime_requires_source_annotations": False,
                "spatial_encoding": "zero_background_one_based_category_index",
                "spatial_alignment": "latent_grid",
                "spatial_shape": [spatial_size, spatial_size],
                "complete": len(failures) == 0,
                **ONTOLOGY_METADATA[modality],
            }
            replacement = {
                "contract": contract,
                "modality": modality,
                "sample_ids": list(sidecar["sample_ids"]),
                "category_values": torch.arange(class_count, dtype=torch.long),
                "category_multihot": new_multihot,
                "category_pixel_counts": new_counts,
                "category_spatial_index_map": new_spatial,
                "category_prototypes_rgb": prototypes,
                "background_rgb": torch.zeros(3, dtype=torch.uint8),
                "failed_sample_ids": [item["sample_id"] for item in failures],
            }
            _atomic_torch_save(replacement, sidecar_path)
            print(
                f"[coarse-remap] modality={modality} shard={shard_id}/{num_shards} "
                f"rows={row_count} repaired={len(repaired)} failures={len(failures)}",
                flush=True,
            )
            if failures:
                failure_path = sidecar_path.with_suffix(".failures.yaml")
                failure_path.write_text(
                    yaml.safe_dump({"failures": failures}, sort_keys=False), encoding="utf-8"
                )


def validate_sidecars(output_dir: Path, num_shards: int) -> None:
    """Validate that every rebuilt sidecar remains aligned to its latent shard."""
    summary: dict[str, dict[str, int]] = {}
    for modality in ALLOWED_MODALITIES:
        rows = 0
        failures = 0
        class_count = EXPECTED_CLASS_COUNTS[modality]
        for shard_id in range(num_shards):
            filename = f"shard_{shard_id:05d}_of_{num_shards:05d}.pt"
            latent_path = output_dir / "shards" / modality / filename
            sidecar_path = output_dir / "categorical_targets" / modality / filename
            latent = torch.load(latent_path, map_location="cpu", mmap=True, weights_only=False)
            sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            sample_ids = list(latent["sample_ids"])
            if list(sidecar["sample_ids"]) != sample_ids:
                raise RuntimeError(f"Sample alignment mismatch: {sidecar_path}")
            values = torch.as_tensor(sidecar["category_values"], dtype=torch.long)
            if values.tolist() != list(range(class_count)):
                raise RuntimeError(
                    f"Unexpected category values for {modality}: {values.tolist()}"
                )
            expected_shapes = {
                "category_multihot": (len(sample_ids), class_count),
                "category_pixel_counts": (len(sample_ids), class_count),
                "category_spatial_index_map": (len(sample_ids), 60, 60),
                "category_prototypes_rgb": (class_count, 3),
            }
            for key, shape in expected_shapes.items():
                if tuple(torch.as_tensor(sidecar[key]).shape) != shape:
                    raise RuntimeError(
                        f"Unexpected {key} shape in {sidecar_path}: "
                        f"{tuple(torch.as_tensor(sidecar[key]).shape)} != {shape}"
                    )
            spatial = torch.as_tensor(sidecar["category_spatial_index_map"])
            if spatial.numel() and (int(spatial.min()) < 0 or int(spatial.max()) > class_count):
                raise RuntimeError(f"Invalid spatial category index in {sidecar_path}")
            contract = sidecar.get("contract") or {}
            if contract.get("ontology_revision") != EXPECTED_ONTOLOGY_REVISIONS[modality]:
                raise RuntimeError(f"Unexpected ontology revision in {sidecar_path}")
            rows += len(sample_ids)
            failures += len(sidecar.get("failed_sample_ids") or [])
        summary[modality] = {
            "classes": class_count,
            "rows": rows,
            "failures": failures,
        }
    print(yaml.safe_dump({"validated_categorical_sidecars": summary}, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="data/wsg_flux2_frame_latents_dense_multidataset/precompute_flux_frame_multidataset.yaml",
    )
    parser.add_argument(
        "--output-dir", default="data/wsg_flux2_frame_latents_dense_multidataset"
    )
    parser.add_argument("--devices", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing 111/131/136 sidecars without rebuilding them.",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "Re-read every source WSG instead of directly folding the existing "
            "fine-category sidecars. Intended only for repairing the source extraction."
        ),
    )
    parser.add_argument(
        "--include-latents",
        action="store_true",
        help="Also replace VAE latent shards; default only rebuilds category sidecars.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if args.validate_only:
        validate_sidecars(output_dir, args.num_shards)
        return
    if not args.from_raw:
        if not args.execute:
            print(
                "Dry run only: directly remap existing fine-category sidecars for "
                "111/131/136 and selectively repair empty rows. Add --execute to run."
            )
            return
        remap_existing_sidecars(output_dir, args.num_shards)
        validate_sidecars(output_dir, args.num_shards)
        print("Completed direct coarse-category sidecar remap.")
        return
    log_dir = output_dir / "logs" / "coarse_categorical_reencode"
    source_config = Path(args.config)
    if not source_config.is_absolute():
        source_config = PROJECT_ROOT / source_config
    effective_config = source_config
    if args.execute:
        log_dir.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        params = config["dataset"]["params"]
        relevant_families = {"Articulation_results", "ShapeNetSem_results"}
        params["dataset_paths"] = [
            path
            for path in params["dataset_paths"]
            if Path(path).name in relevant_families
        ]
        if not params["dataset_paths"]:
            raise ValueError("No Articulation or ShapeNetSem dataset roots remain")
        effective_config = log_dir / "coarse_categorical_reencode.yaml"
        effective_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    commands: list[tuple[list[str], Path, dict[str, str]]] = []
    for shard_id in range(args.num_shards):
        device = args.devices[shard_id % len(args.devices)]
        command = [
            sys.executable,
            str(ENCODER),
            "--config",
            str(effective_config),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda:0",
            "--dtype",
            "bf16",
            "--cache-dtype",
            "bf16",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--num-shards",
            str(args.num_shards),
            "--shard-id",
            str(shard_id),
            "--crop-mode",
            "center_short_side_square",
            "--filter-invalid-modalities",
            "--modalities",
            *ALLOWED_MODALITIES,
        ]
        if not args.include_latents:
            command.append("--categorical-sidecars-only")
        if args.max_samples is not None:
            command[2:2] = ["--max-samples", str(args.max_samples)]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = device
        commands.append((command, log_dir / f"shard_{shard_id}.log", environment))

    if not args.execute:
        for command, log_path, environment in commands:
            print(f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']} {' '.join(command)}")
            print(f"  log: {log_path}")
        mode = "latents and sidecars" if args.include_latents else "category sidecars only"
        print(f"Dry run only ({mode}). Add --execute to run modalities 111, 131, and 136.")
        return

    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []
    for command, log_path, environment in commands:
        handle = log_path.open("wb")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, log_path))
        print(f"started pid={process.pid} log={log_path}", flush=True)

    failures = []
    for process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((return_code, log_path))
    if failures:
        raise SystemExit(f"Coarse categorical re-encoding failed: {failures}")
    validate_sidecars(output_dir, args.num_shards)
    print("Completed modalities 111, 131, and 136.")


if __name__ == "__main__":
    main()
