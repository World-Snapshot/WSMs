#!/usr/bin/env bash
set -euo pipefail

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_small.json}"

required_modalities() {
  python - "${MODALITY_CONFIG}" <<'PY'
import json
import os
import sys
from pathlib import Path
from src.dataset.wsg_dataset import flatten_wsg_modality_graph

with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    raw_graph = json.load(f)["wsg_v1"]
flat_graph = flatten_wsg_modality_graph(raw_graph)
targets = (os.environ.get("TARGETS") or os.environ.get("WSG_DEFAULT_TARGETS") or "").split()
if not targets:
    targets = list(raw_graph.keys()) if all(str(key).startswith("lora_and_muti-lanes_index_") for key in raw_graph) else list(flat_graph.keys())

modalities = []
for target in targets:
    if target in raw_graph and isinstance(raw_graph[target], dict) and str(target).startswith("lora_and_muti-lanes_index_"):
        for lane, spec in raw_graph[target].items():
            if str(lane).startswith("__"):
                continue
            modalities.append(str(lane))
            modalities.extend(str(item) for item in (spec.get("conditions") or {}).keys())
    elif target in flat_graph:
        modalities.append(str(target))
        modalities.extend(str(item) for item in (flat_graph.get(target, {}).get("conditions") or {}).keys())
    else:
        raise SystemExit(f"Unknown WSG cache target: {target}")
for modality in dict.fromkeys(modalities):
    print(modality)
PY
}

required_modality_cache_names() {
  required_modalities | python -c '
import sys
from src.dataset.wsg_dataset import safe_modality_name
for line in sys.stdin:
    modality = line.strip()
    if modality:
        print(safe_modality_name(modality))
'
}

required_sparse_modalities() {
  python - "${MODALITY_CONFIG}" <<'PY'
import json
import os
import sys
from pathlib import Path
from src.dataset.wsg_dataset import LORA_AND_MULTI_LANES_PREFIX
from src.pipelines.wsg_sparse_codecs import build_sparse_codec, sparse_modalities_from_config

with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    raw_graph = json.load(f)["wsg_v1"]
selected = (os.environ.get("TARGETS") or os.environ.get("WSG_DEFAULT_TARGETS") or "").split()
if not selected:
    selected = list(raw_graph.keys()) if all(str(key).startswith(LORA_AND_MULTI_LANES_PREFIX) for key in raw_graph) else list(raw_graph.keys())
required = set()
for target in selected:
    spec = raw_graph.get(target)
    if isinstance(spec, dict) and str(target).startswith(LORA_AND_MULTI_LANES_PREFIX):
        for lane, lane_spec in spec.items():
            if str(lane).startswith("__"):
                continue
            required.add(str(lane))
            required.update(str(item) for item in (lane_spec.get("conditions") or {}).keys())
    elif isinstance(spec, dict):
        required.add(str(target))
        required.update(str(item) for item in (spec.get("conditions") or {}).keys())
sparse = sparse_modalities_from_config(sys.argv[1])
mode = os.environ.get("WSG_SPARSE_CODEC_MODE", "disabled")
for modality in sorted(required & sparse):
    if build_sparse_codec(
        modality,
        mode=mode,
        sigma=float(os.environ.get("WSG_SPARSE_HEATMAP_SIGMA", "3.0")),
        radius=int(os.environ.get("WSG_SPARSE_HEATMAP_RADIUS", "8")),
        peak_threshold=float(os.environ.get("WSG_SPARSE_HEATMAP_PEAK_THRESHOLD", "0.10")),
        nms_radius=int(os.environ.get("WSG_SPARSE_HEATMAP_NMS_RADIUS", "6")),
    ) is None:
        continue
    print(modality)
PY
}

print_modality_cache_summary() {
  python - "${MODALITY_CONFIG}" <<'PY'
import json
import os
import sys
from pathlib import Path
from src.dataset.wsg_dataset import flatten_wsg_modality_graph, parse_channel_key

with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    raw_graph = json.load(f)["wsg_v1"]
flat_graph = flatten_wsg_modality_graph(raw_graph)
selected = (os.environ.get("TARGETS") or os.environ.get("WSG_DEFAULT_TARGETS") or "").split()
if not selected:
    selected = list(raw_graph.keys()) if all(str(key).startswith("lora_and_muti-lanes_index_") for key in raw_graph) else list(flat_graph.keys())
modalities = []
roles = {}
for target in selected:
    if target in raw_graph and isinstance(raw_graph[target], dict) and str(target).startswith("lora_and_muti-lanes_index_"):
        for lane, spec in raw_graph[target].items():
            lane = str(lane)
            if lane.startswith("__"):
                continue
            modalities.append(lane)
            roles[lane] = "target"
            for condition in (spec.get("conditions") or {}).keys():
                condition = str(condition)
                modalities.append(condition)
                roles.setdefault(condition, "condition")
    elif target in flat_graph:
        modalities.append(str(target))
        roles[str(target)] = "target"
        for condition in (flat_graph.get(target, {}).get("conditions") or {}).keys():
            condition = str(condition)
            modalities.append(condition)
            roles.setdefault(condition, "condition")
    else:
        raise SystemExit(f"Unknown WSG cache target: {target}")
modalities = list(dict.fromkeys(modalities))
widths = {modality: len(parse_channel_key(modality)) for modality in modalities}
targets = [modality for modality in modalities if roles.get(modality) == "target"]
conditions = [modality for modality in modalities if roles.get(modality) != "target"]
total_width = sum(widths.values())
print(
    "[VAE cache] required modalities: "
    f"{len(modalities)} total ({len(targets)} targets, {len(conditions)} conditions); "
    f"{total_width} raw channels; avg {total_width / max(1, len(modalities)):.2f} channels/modality",
    flush=True,
)
print(
    "[VAE cache] target modalities: "
    + ", ".join(f"{modality}({widths[modality]})" for modality in targets),
    flush=True,
)
print(
    "[VAE cache] condition modalities: "
    + ", ".join(f"{modality}({widths[modality]})" for modality in conditions),
    flush=True,
)
PY
}

has_modality_cache() {
  local cache_dir="$1"
  local modality="$2"
  [[ -f "${cache_dir}/${modality}.pt" ]] || compgen -G "${cache_dir}/shards/${modality}/shard_*_of_*.pt" >/dev/null
}

needs_vae_cache() {
  local cache_dir="$1"
  local modality
  while IFS= read -r modality; do
    if ! has_modality_cache "${cache_dir}" "${modality}"; then
      echo "Missing VAE latent cache for ${modality} in ${cache_dir}" >&2
      return 0
    fi
  done < <(required_modality_cache_names)
  return 1
}

vae_cache_crop_mode_mismatch() {
  local cache_dir="$1"
  local expected_crop_mode="${WSG_VAE_CROP_MODE:-}"
  local modality cache_path
  if [[ -z "${expected_crop_mode}" ]]; then
    return 1
  fi
  while IFS= read -r modality; do
    cache_path=""
    if [[ -f "${cache_dir}/${modality}.pt" ]]; then
      cache_path="${cache_dir}/${modality}.pt"
    else
      cache_path="$(find "${cache_dir}/shards/${modality}" -maxdepth 1 -type f -name 'shard_*_of_*.pt' 2>/dev/null | sort | head -n 1 || true)"
    fi
    if [[ -z "${cache_path}" ]]; then
      continue
    fi
    if python - "${cache_path}" "${expected_crop_mode}" "${cache_dir}" "${modality}" <<'PY'
import sys
import torch

path = sys.argv[1]
expected = sys.argv[2]
cache_dir = sys.argv[3]
modality = sys.argv[4]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
except TypeError:
    payload = torch.load(path, map_location="cpu", weights_only=False)
crop_mode = payload.get("crop_mode")
crop_infos = payload.get("crop_infos") or []
if crop_mode is None and crop_infos:
    crop_mode = crop_infos[0].get("crop_mode")
if crop_mode != expected:
    print(
        f"VAE latent cache crop_mode mismatch for {modality} in {cache_dir}: "
        f"found {crop_mode!r}, expected {expected!r}",
        file=sys.stderr,
    )
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
  done < <(required_modality_cache_names)
  return 1
}

vae_cache_sparse_codec_mismatch() {
  local cache_dir="$1"
  local expected_mode="${WSG_SPARSE_CODEC_MODE:-disabled}"
  if [[ "${expected_mode}" == "0" || "${expected_mode}" == "false" || "${expected_mode}" == "False" ]]; then
    expected_mode="disabled"
  fi
  if [[ "${expected_mode}" == "disabled" ]]; then
    return 1
  fi
  local modality safe_name cache_path
  while IFS= read -r modality; do
    safe_name="$(python - "${modality}" <<'PY'
import sys
from src.dataset.wsg_dataset import safe_modality_name
print(safe_modality_name(sys.argv[1]))
PY
)"
    cache_path=""
    if [[ -f "${cache_dir}/${safe_name}.pt" ]]; then
      cache_path="${cache_dir}/${safe_name}.pt"
    else
      cache_path="$(find "${cache_dir}/shards/${safe_name}" -maxdepth 1 -type f -name 'shard_*_of_*.pt' 2>/dev/null | sort | head -n 1 || true)"
    fi
    if [[ -z "${cache_path}" ]]; then
      continue
    fi
    if python - "${cache_path}" "${expected_mode}" "${cache_dir}" "${modality}" "${WSG_COMPACT_SPARSE_LANES:-1}" <<'PY'
import sys
import torch
from src.pipelines.wsg_sparse_codecs import build_sparse_codec

path, expected, cache_dir, modality, expected_compact = sys.argv[1:]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
except TypeError:
    payload = torch.load(path, map_location="cpu", weights_only=False)
mode = str(payload.get("sparse_codec_mode", "disabled")).lower()
modalities = set(str(item) for item in payload.get("sparse_codec_modalities", []))
if mode not in {expected.lower(), "big_token_heatmap" if expected.lower() == "heatmap" else expected.lower()} or modality not in modalities:
    print(
        f"VAE latent cache sparse codec mismatch for {modality} in {cache_dir}: "
        f"found mode={mode!r}, modalities={sorted(modalities)}, expected={expected!r}",
        file=sys.stderr,
    )
    raise SystemExit(0)
codec = build_sparse_codec(
    modality,
    mode=expected,
    sigma=float(payload.get("sparse_heatmap_sigma", 3.0)),
    radius=int(payload.get("sparse_heatmap_radius", 8)),
    peak_threshold=float(payload.get("sparse_heatmap_peak_threshold", 0.10)),
    nms_radius=int(payload.get("sparse_heatmap_nms_radius", 6)),
)
compact_requested = str(expected_compact).lower() not in {"0", "false", "no"}
if compact_requested and codec is not None and int(getattr(codec, "latent_lane_count", 1)) > 1:
    if payload.get("sparse_cache_format") != "compact_lanes_v1":
        print(
            f"VAE latent cache sparse compact-lane mismatch for {modality} in {cache_dir}: "
            f"found format={payload.get('sparse_cache_format')!r}, expected='compact_lanes_v1'",
            file=sys.stderr,
        )
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
  done < <(required_sparse_modalities)
  return 1
}

needs_text_cache() {
  local cache_dir="$1"
  [[ ! -f "${cache_dir}/prompts.pt" || ! -f "${cache_dir}/text_embeddings.pt" ]]
}

prepare_text_cache() {
  local config_path="$1"
  local cache_dir="$2"
  if needs_text_cache "${cache_dir}"; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python manual_processing/text_embeddings/precompute_wsg_text_embeddings.py \
      --config "${config_path}" \
      --latent-cache-dir "${cache_dir}" \
      --dtype bf16 \
      --batch-size 8 \
      --skip-existing
  fi
}

required_channels() {
  required_modalities | python -c '
import sys
from src.dataset.wsg_dataset import parse_channel_key
channels = []
for line in sys.stdin:
    modality = line.strip()
    if modality:
        channels.extend(parse_channel_key(modality))
channels = sorted(set(ch for ch in channels if ch not in (0, 1, 2)))
print(" ".join(str(ch) for ch in channels))
'
}

precompute_normalization_cache() {
  unset WSG_VAE_FORCE_REENCODE
  local channels
  channels="$(required_channels)"
  if [[ -z "${channels}" ]]; then
    return
  fi
  echo "[normalization] precomputing sampled channel stats before VAE cache: ${channels}"
  local status_file="${TMPDIR:-/tmp}/wsg_normalization_status_$$.json"
  local percentile_args=()
  if [[ "${WSG_NORMALIZATION_PERCENTILE_CLIP:-1}" == "0" || "${WSG_NORMALIZATION_PERCENTILE_CLIP:-1}" == "false" ]]; then
    percentile_args+=(--disable-continuous-percentile-clip)
  else
    percentile_args+=(
      --continuous-percentile-low "${WSG_NORMALIZATION_PERCENTILE_LOW:-2.5}"
      --continuous-percentile-high "${WSG_NORMALIZATION_PERCENTILE_HIGH:-97.5}"
    )
  fi
  python scripts/cache/precompute_wsg_normalization_cache.py \
    --dataset-path "../../wsg/output/AVIS_train_results" \
    --cache "configs/wsg_dataset_info.json" \
    --channels "${channels}" \
    --max-arrays "${WSG_NORMALIZATION_MAX_ARRAYS:-600}" \
    --max-values-per-array "${WSG_NORMALIZATION_MAX_VALUES_PER_ARRAY:-1000}" \
    --workers "${WSG_NORMALIZATION_WORKERS:-8}" \
    --seed "${WSG_NORMALIZATION_SEED:-20260516}" \
    --status-file "${status_file}" \
    "${percentile_args[@]}"
  if python - "${status_file}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
if not status_path.exists():
    raise SystemExit(1)
with status_path.open("r", encoding="utf-8") as f:
    status = json.load(f)
raise SystemExit(0 if status.get("changed") else 1)
PY
  then
    export WSG_VAE_FORCE_REENCODE=1
    echo "[normalization] stats changed; VAE cache will overwrite affected run modalities"
  else
    unset WSG_VAE_FORCE_REENCODE
    echo "[normalization] stats unchanged; VAE cache can reuse existing latents"
  fi
}

print_modality_cache_summary
precompute_normalization_cache

train_crop_mismatch=0
if vae_cache_crop_mode_mismatch "data/wsg_vae_latents_avis_train"; then
  train_crop_mismatch=1
fi
sparse_codec_mismatch=0
if vae_cache_sparse_codec_mismatch "data/wsg_vae_latents_avis_train"; then
  sparse_codec_mismatch=1
fi

if [[ "${WSG_VAE_FORCE_REENCODE:-0}" == "1" ]] || needs_vae_cache "data/wsg_vae_latents_avis_train" || [[ "${train_crop_mismatch}" == "1" ]] || [[ "${sparse_codec_mismatch}" == "1" ]]; then
  export WSG_CACHE_MODALITIES
  WSG_CACHE_MODALITIES="$(required_modalities | xargs)"
  if [[ "${train_crop_mismatch}" == "1" || "${sparse_codec_mismatch}" == "1" ]]; then
    export WSG_VAE_FORCE_REENCODE=1
  fi
  bash manual_processing/vae_latents/launch_wsg_vae_latents_stride16_fp32.sh
fi
prepare_text_cache "manual_processing/configs/wsm_v1_augustus_precompute_avis_train.yaml" "data/wsg_vae_latents_avis_train"

val_crop_mismatch=0
test_crop_mismatch=0
if vae_cache_crop_mode_mismatch "data/wsg_vae_latents_avis_val"; then
  val_crop_mismatch=1
fi
if vae_cache_crop_mode_mismatch "data/wsg_vae_latents_avis_test"; then
  test_crop_mismatch=1
fi
sparse_codec_mismatch_val=0
sparse_codec_mismatch_test=0
if vae_cache_sparse_codec_mismatch "data/wsg_vae_latents_avis_val"; then
  sparse_codec_mismatch_val=1
fi
if vae_cache_sparse_codec_mismatch "data/wsg_vae_latents_avis_test"; then
  sparse_codec_mismatch_test=1
fi

if [[ "${WSG_VAE_FORCE_REENCODE:-0}" == "1" ]] || needs_vae_cache "data/wsg_vae_latents_avis_val" || needs_vae_cache "data/wsg_vae_latents_avis_test" || [[ "${val_crop_mismatch}" == "1" ]] || [[ "${test_crop_mismatch}" == "1" ]] || [[ "${sparse_codec_mismatch_val}" == "1" ]] || [[ "${sparse_codec_mismatch_test}" == "1" ]]; then
  export WSG_CACHE_MODALITIES
  WSG_CACHE_MODALITIES="$(required_modalities | xargs)"
  if [[ "${val_crop_mismatch}" == "1" ]] || [[ "${test_crop_mismatch}" == "1" ]] || [[ "${sparse_codec_mismatch_val}" == "1" ]] || [[ "${sparse_codec_mismatch_test}" == "1" ]]; then
    export WSG_VAE_FORCE_REENCODE=1
  fi
  bash manual_processing/vae_latents/launch_wsg_vae_latents_avis_val_test_stride16_fp32.sh
fi
prepare_text_cache "manual_processing/configs/wsm_v1_augustus_precompute_avis_val.yaml" "data/wsg_vae_latents_avis_val"
prepare_text_cache "manual_processing/configs/wsm_v1_augustus_precompute_avis_test.yaml" "data/wsg_vae_latents_avis_test"
