#!/usr/bin/env bash
set -euo pipefail

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
BASE_CONFIG="${BASE_CONFIG:-manual_processing/flux2_preprocessing/configs/flux2_precompute_avis_train.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/wsg_flux2_frame_latents_dense_multidataset}"
NORMALIZATION_CACHE="${NORMALIZATION_CACHE:-configs/wsg_dataset_info.json}"
FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-models/black-forest-labs/FLUX.2-klein-4B}"
WSG_VAE_CROP_MODE="${WSG_VAE_CROP_MODE:-center_short_side_square}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DTYPE="${DTYPE:-bf16}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
SAMPLE_MAJOR_MERGE_EVERY="${SAMPLE_MAJOR_MERGE_EVERY:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_NORMALIZATION="${RUN_NORMALIZATION:-0}"
# Reuse existing sampled normalization ranges by default; set RUN_NORMALIZATION=1 to rescan.
# FLUX frame cache uses log normalization by default. Keep this in sync with modality config contracts.
NORMALIZATION_MODE="${NORMALIZATION_MODE:-log}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
WSG_MIN_FREE_GPU_MB="${WSG_MIN_FREE_GPU_MB:-6000}"
WSG_CACHE_GPU_USAGE_LOG="${WSG_CACHE_GPU_USAGE_LOG:-0}"

DATASET_PATHS_DEFAULT="/common/users/sf895/wave2pixel/AVIS_train_results /common/users/sf895/wave2pixel/Articulation_results /common/users/sf895/wave2pixel/ShapeNetSem_results /common/users/sf895/wave2pixel/MatSynth_results /common/users/sf895/wave2pixel/AVIS_val_results /common/users/sf895/wave2pixel/AVIS_test_results /research/cbim/vast/sf895/code/wsg/output/XPose_results /research/cbim/vast/sf895/code/wsg/output/XPose_WSV_results"
read -r -a DATASET_PATH_CANDIDATES <<< "${DATASET_PATHS:-${DATASET_PATHS_DEFAULT}}"
DATASET_PATHS=()
for dataset_path in "${DATASET_PATH_CANDIDATES[@]}"; do
  if [[ -d "${dataset_path}" ]]; then
    DATASET_PATHS+=("${dataset_path}")
  else
    echo "[flux-frame-cache] skip missing dataset=${dataset_path}"
  fi
done
if [[ "${#DATASET_PATHS[@]}" -eq 0 ]]; then
  echo "[flux-frame-cache] ERROR: no dataset directories found" >&2
  exit 1
fi
read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "[flux-frame-cache] ERROR: GPU_IDS is empty" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
if [[ "${WSG_CACHE_TEE_MAIN_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "${OUTPUT_DIR}/logs/main.log") 2>&1
fi

SOURCE_MODALITY_CONFIG="${MODALITY_CONFIG}"
EFFECTIVE_MODALITY_CONFIG="${OUTPUT_DIR}/wsg_v1_modalities_flux_frame_${NORMALIZATION_MODE}.json"
python - "${SOURCE_MODALITY_CONFIG}" "${EFFECTIVE_MODALITY_CONFIG}" "${NORMALIZATION_MODE}" <<'PYMODCFG'
import json
import sys
from pathlib import Path

NON_VAE = {"caption", "Reference_IMG"}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def collect_modalities(wsg_v1):
    modalities = []
    if isinstance(wsg_v1, dict) and all(str(key).startswith("Channel_Head_") or str(key).startswith("Channel_Head_and_muti-lanes_index_") for key in wsg_v1):
        for entry in wsg_v1.values():
            if not isinstance(entry, dict):
                continue
            if "target" in entry and "condition" in entry:
                modalities.append(str(entry["target"]))
                modalities.extend(as_list(entry.get("condition")))
                continue
            for target, spec in entry.items():
                if str(target).startswith("__"):
                    continue
                modalities.append(str(target))
                if isinstance(spec, dict):
                    modalities.extend(str(condition) for condition in (spec.get("conditions") or {}))
        return list(dict.fromkeys(modalities))
    for target, spec in wsg_v1.items():
        modalities.append(str(target))
        if isinstance(spec, dict):
            modalities.extend(str(condition) for condition in (spec.get("conditions") or {}))
    return list(dict.fromkeys(modalities))

source_path, output_path, normalization_mode = sys.argv[1:]
with Path(source_path).open("r", encoding="utf-8") as f:
    source = json.load(f)
modalities = [item for item in collect_modalities(source["wsg_v1"]) if item not in NON_VAE]
normalization = source.setdefault("normalization", {}).copy()
normalization["default_mode"] = str(normalization_mode)
normalization.setdefault("modalities", {})
channel_ranges = normalization.setdefault("channel_ranges", {})
# Avoid per-shard lazy range scans for channels intentionally absent from the older cache.
for channel in (33, 35):
    channel_ranges.setdefault(str(channel), [0.0, 1.0])
for channel in (122, 123, 124):
    channel_ranges.setdefault(str(channel), [0.0, 255.0])
for channel in (155, 156, 157):
    channel_ranges.setdefault(str(channel), [0.0, 1.0])
cache_payload = {
    key: value for key, value in source.items()
    if key not in {"wsg_v1", "__task_type_notes__"}
}
cache_payload["normalization"] = normalization
cache_payload["wsg_v1"] = {
    modality: {"description": str(modality), "conditions": {}}
    for modality in modalities
}
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with Path(output_path).open("w", encoding="utf-8") as f:
    json.dump(cache_payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PYMODCFG
MODALITY_CONFIG="${EFFECTIVE_MODALITY_CONFIG}"

mapfile -t CACHE_INFO < <(python - "${MODALITY_CONFIG}" <<'PYMOD'
import json
import sys
from pathlib import Path

from src.dataset.wsg_dataset import flatten_wsg_modality_graph, parse_channel_key

NON_VAE = {"caption", "Reference_IMG"}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def collect_modalities(wsg_v1):
    modalities = []
    if isinstance(wsg_v1, dict) and all(str(key).startswith("Channel_Head_") or str(key).startswith("Channel_Head_and_muti-lanes_index_") for key in wsg_v1):
        for entry in wsg_v1.values():
            if not isinstance(entry, dict):
                continue
            if "target" in entry and "condition" in entry:
                modalities.append(str(entry["target"]))
                modalities.extend(as_list(entry.get("condition")))
                continue
            for target, spec in entry.items():
                if str(target).startswith("__"):
                    continue
                modalities.append(str(target))
                if isinstance(spec, dict):
                    modalities.extend(str(condition) for condition in (spec.get("conditions") or {}))
        return list(dict.fromkeys(modalities))
    graph = flatten_wsg_modality_graph(wsg_v1)
    for target, spec in graph.items():
        modalities.append(str(target))
        modalities.extend(str(condition) for condition in (spec.get("conditions") or {}))
    return list(dict.fromkeys(modalities))

with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    payload = json.load(f)
modalities = collect_modalities(payload["wsg_v1"])
channels = []
for modality in modalities:
    if modality in NON_VAE:
        continue
    channels.extend(parse_channel_key(modality))
channels = sorted(set(channel for channel in channels if channel not in (0, 1, 2)))
print(" ".join(modalities))
print(" ".join(str(channel) for channel in channels))
PYMOD
)
MODALITIES="${CACHE_INFO[0]}"
CHANNELS="${CACHE_INFO[1]}"
if [[ -n "${MODALITIES_OVERRIDE:-}" ]]; then
  MODALITIES="${MODALITIES_OVERRIDE}"
  CHANNELS="$(python - "${MODALITIES}" <<'PYMOD'
import sys
from src.dataset.wsg_dataset import parse_channel_key

channels = []
for modality in sys.argv[1].split():
    if modality in {"caption", "Reference_IMG"}:
        continue
    channels.extend(parse_channel_key(modality))
print(" ".join(str(channel) for channel in sorted(set(channels))))
PYMOD
)"
fi
read -r -a MODALITY_LIST <<< "${MODALITIES}"

# Text/Reference_IMG are not VAE image modalities.
VAE_MODALITY_LIST=()
for modality in "${MODALITY_LIST[@]}"; do
  if [[ "${modality}" == "caption" || "${modality}" == "Reference_IMG" ]]; then
    continue
  fi
  VAE_MODALITY_LIST+=("${modality}")
done

cat <<EOF
[flux-frame-cache] source_modality_config=${SOURCE_MODALITY_CONFIG}
[flux-frame-cache] modality_config=${MODALITY_CONFIG}
[flux-frame-cache] output_dir=${OUTPUT_DIR}
[flux-frame-cache] flux_model_path=${FLUX_MODEL_PATH}
[flux-frame-cache] crop_mode=${WSG_VAE_CROP_MODE}
[flux-frame-cache] datasets=${DATASET_PATHS[*]}
[flux-frame-cache] gpu_ids=${GPU_ID_LIST[*]} min_free_gpu_mb=${WSG_MIN_FREE_GPU_MB}
[flux-frame-cache] dtype=${DTYPE} cache_dtype=${CACHE_DTYPE} batch_size=${BATCH_SIZE} max_samples=${MAX_SAMPLES:-all}
[flux-frame-cache] normalization_mode=${NORMALIZATION_MODE} run_normalization=${RUN_NORMALIZATION}
[flux-frame-cache] modalities=${VAE_MODALITY_LIST[*]}
[flux-frame-cache] channels=${CHANNELS}
EOF

if [[ -n "${CHANNELS}" && "${RUN_NORMALIZATION}" == "1" ]]; then
  NORMALIZATION_ARGS=()
  for dataset_path in "${DATASET_PATHS[@]}"; do
    NORMALIZATION_ARGS+=(--dataset-path "${dataset_path}")
  done
  python scripts/cache/precompute_wsg_normalization_cache.py \
    "${NORMALIZATION_ARGS[@]}" \
    --cache "${NORMALIZATION_CACHE}" \
    --channels "${CHANNELS}" \
    --max-arrays "${WSG_NORMALIZATION_MAX_ARRAYS:-600}" \
    --max-values-per-array "${WSG_NORMALIZATION_MAX_VALUES_PER_ARRAY:-1000}" \
    --workers "${WSG_NORMALIZATION_WORKERS:-8}" \
    --seed "${WSG_NORMALIZATION_SEED:-20260516}" \
    --status-file "${OUTPUT_DIR}/logs/normalization_status.json"
else
  echo "[flux-frame-cache] skip normalization RUN_NORMALIZATION=${RUN_NORMALIZATION} channels=${CHANNELS}"
fi

NORMALIZATION_SYNC_ARGS=()
for dataset_path in "${DATASET_PATHS[@]}"; do
  NORMALIZATION_SYNC_ARGS+=(--dataset-path "${dataset_path}")
done
python scripts/cache/sync_wsg_normalization_contract.py \
  --modality-config "${MODALITY_CONFIG}" \
  --cache "${NORMALIZATION_CACHE}" \
  "${NORMALIZATION_SYNC_ARGS[@]}"

CONFIG_PATH="${OUTPUT_DIR}/precompute_flux_frame_multidataset.yaml"
python - "${BASE_CONFIG}" "${CONFIG_PATH}" "${MODALITY_CONFIG}" "${NORMALIZATION_CACHE}" "${NORMALIZATION_MODE}" "${FLUX_MODEL_PATH}" "${DATASET_PATHS[@]}" <<'PYCFG'
import sys
from pathlib import Path

import yaml

base_config, output_config, modality_config, normalization_cache, normalization_mode, flux_model_path, *dataset_paths = sys.argv[1:]
with Path(base_config).open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
params = cfg.setdefault("dataset", {}).setdefault("params", {})
params["dataset_paths"] = dataset_paths
params.pop("normalization_dataset_paths", None)
params["modality_config_path"] = modality_config
params["normalization_cache_path"] = normalization_cache
params["normalization_mode"] = normalization_mode
params["data_type"] = "auto"
params["patch_hw"] = 480
params["num_frames"] = 1
params["video_window_stride"] = 1
params["include_conditions"] = True
params["augment"] = False
cfg.setdefault("model", {}).setdefault("params", {})["flux_model_path"] = flux_model_path
Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with Path(output_config).open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PYCFG

pids=()
shard_ids=()
gpu_monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d '[:space:]')"
    if [[ -z "${free_mb}" ]]; then
      echo "[flux-frame-cache] ERROR: could not query free memory for GPU ${gpu_id}" >&2
      exit 1
    fi
    echo "[flux-frame-cache] gpu=${gpu_id} free_mb=${free_mb}"
    if (( free_mb < WSG_MIN_FREE_GPU_MB )); then
      echo "[flux-frame-cache] ERROR: GPU ${gpu_id} has only ${free_mb} MiB free; need at least ${WSG_MIN_FREE_GPU_MB} MiB." >&2
      exit 1
    fi
  done
fi

monitor_gpu_usage() {
  local interval="${GPU_USAGE_LOG_INTERVAL:-10}"
  local log_file="${OUTPUT_DIR}/logs/gpu_usage.log"
  echo "[flux-frame-cache] gpu_usage_log=${log_file}"
  while true; do
    {
      date '+%F %T'
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    } >> "${log_file}" 2>/dev/null || true
    sleep "${interval}"
  done
}

if [[ "${WSG_CACHE_GPU_USAGE_LOG}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  : > "${OUTPUT_DIR}/logs/gpu_usage.log"
  monitor_gpu_usage &
  gpu_monitor_pid="$!"
fi

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  device_id="${GPU_ID_LIST[$((shard_id % ${#GPU_ID_LIST[@]}))]}"
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES="${device_id}" python manual_processing/flux2_preprocessing/vae_latents/precompute_wsg_flux_frame_latents.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    "${MAX_SAMPLE_ARGS[@]}" \
    --flux-model-path "${FLUX_MODEL_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --skip-existing \
    --dtype "${DTYPE}" \
    --cache-dtype "${CACHE_DTYPE}" \
    --sample-major-merge-every "${SAMPLE_MAJOR_MERGE_EVERY}" \
    --crop-mode "${WSG_VAE_CROP_MODE}" \
    --filter-invalid-modalities \
    --modalities "${VAE_MODALITY_LIST[@]}" \
    > "${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
  shard_ids+=("${shard_id}")
done

monitor_shards() {
  local interval="${VAE_CACHE_PROGRESS_INTERVAL:-60}"
  while true; do
    local running=0
    local pid
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
    if [[ "${running}" -eq 0 ]]; then
      break
    fi
    echo "[flux-frame-cache] $(date '+%F %T') running_shards=${running}/${#pids[@]}"
    local shard_id log_file latest
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
      log_file="${OUTPUT_DIR}/logs/shard_${shard_id}.log"
      if [[ -f "${log_file}" ]]; then
        latest="$(tail -c 4096 "${log_file}" | tr '\r' '\n' | sed -E 's/\x1B\[[0-9;?]*[A-Za-z]//g' | sed '/^[[:space:]]*$/d' | tail -n 1 || true)"
        [[ -n "${latest}" ]] && echo "  shard_${shard_id}: ${latest}"
      fi
    done
    sleep "${interval}"
  done
}

monitor_shards &
monitor_pid="$!"
status=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  shard_id="${shard_ids[$index]}"
  set +e
  wait "${pid}"
  rc="$?"
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[flux-frame-cache] ERROR: shard_${shard_id} pid=${pid} exited rc=${rc}" >&2
    status="${rc}"
  fi
done
wait "${monitor_pid}" 2>/dev/null || true
if [[ -n "${gpu_monitor_pid}" ]]; then
  kill "${gpu_monitor_pid}" 2>/dev/null || true
fi
exit "${status}"
