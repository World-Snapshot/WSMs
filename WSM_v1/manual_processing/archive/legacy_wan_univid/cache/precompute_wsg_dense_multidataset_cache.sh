#!/usr/bin/env bash
set -euo pipefail

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
BASE_CONFIG="${BASE_CONFIG:-manual_processing/configs/wsm_v1_augustus_precompute_avis_train.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/wsg_vae_latents_dense_multidataset}"
NORMALIZATION_CACHE="${NORMALIZATION_CACHE:-configs/wsg_dataset_info.json}"
WSG_VAE_CROP_MODE="${WSG_VAE_CROP_MODE:-center_short_side_square}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DTYPE="${DTYPE:-bf16}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
SAMPLE_MAJOR_MERGE_EVERY="${SAMPLE_MAJOR_MERGE_EVERY:-100}"
RUN_TEXT_EMBEDDINGS="${RUN_TEXT_EMBEDDINGS:-1}"
RUN_NORMALIZATION="${RUN_NORMALIZATION:-1}"
# Per-modality modes come from MODALITY_CONFIG. This is only the fallback/default.
NORMALIZATION_MODE="${NORMALIZATION_MODE:-linear}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
WSG_MIN_FREE_GPU_MB="${WSG_MIN_FREE_GPU_MB:-6000}"
WSG_CACHE_GPU_USAGE_LOG="${WSG_CACHE_GPU_USAGE_LOG:-0}"

DATASET_PATHS_DEFAULT="/common/users/sf895/wave2pixel/AVIS_train_results /common/users/sf895/wave2pixel/Articulation_results /common/users/sf895/wave2pixel/ShapeNetSem_results /common/users/sf895/wave2pixel/MatSynth_results /common/users/sf895/wave2pixel/AVIS_val_results /common/users/sf895/wave2pixel/AVIS_test_results"
read -r -a DATASET_PATHS <<< "${DATASET_PATHS:-${DATASET_PATHS_DEFAULT}}"
read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "[dense-cache] ERROR: GPU_IDS is empty" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
if [[ "${WSG_CACHE_TEE_MAIN_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "${OUTPUT_DIR}/logs/main.log") 2>&1
fi

mapfile -t CACHE_INFO < <(python - "${MODALITY_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

from src.dataset.wsg_dataset import flatten_wsg_modality_graph, parse_channel_key

with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    graph = flatten_wsg_modality_graph(json.load(f)["wsg_v1"])

modalities = []
for target, spec in graph.items():
    modalities.append(str(target))
    modalities.extend(str(condition) for condition in (spec.get("conditions") or {}))
modalities = list(dict.fromkeys(modalities))
channels = []
for modality in modalities:
    channels.extend(parse_channel_key(modality))
channels = sorted(set(channel for channel in channels if channel not in (0, 1, 2)))
print(" ".join(modalities))
print(" ".join(str(channel) for channel in channels))
PY
)

MODALITIES="${CACHE_INFO[0]}"
CHANNELS="${CACHE_INFO[1]}"
if [[ -n "${MODALITIES_OVERRIDE:-}" ]]; then
  MODALITIES="${MODALITIES_OVERRIDE}"
  CHANNELS="$(python - "${MODALITIES}" <<'PY'
import sys
from src.dataset.wsg_dataset import parse_channel_key

channels = []
for modality in sys.argv[1].split():
    channels.extend(parse_channel_key(modality))
print(" ".join(str(channel) for channel in sorted(set(channels))))
PY
)"
fi
read -r -a MODALITY_LIST <<< "${MODALITIES}"

echo "[dense-cache] modality_config=${MODALITY_CONFIG}"
echo "[dense-cache] output_dir=${OUTPUT_DIR}"
echo "[dense-cache] crop_mode=${WSG_VAE_CROP_MODE}"
echo "[dense-cache] datasets=${DATASET_PATHS[*]}"
echo "[dense-cache] gpu_ids=${GPU_ID_LIST[*]} min_free_gpu_mb=${WSG_MIN_FREE_GPU_MB}"
echo "[dense-cache] dtype=${DTYPE} cache_dtype=${CACHE_DTYPE} sample_major_merge_every=${SAMPLE_MAJOR_MERGE_EVERY}"
echo "[dense-cache] normalization_mode=${NORMALIZATION_MODE} run_normalization=${RUN_NORMALIZATION}"
echo "[dense-cache] modalities=${MODALITIES}"
echo "[dense-cache] channels=${CHANNELS}"

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
  echo "[dense-cache] skip normalization RUN_NORMALIZATION=${RUN_NORMALIZATION} channels=${CHANNELS}"
fi

NORMALIZATION_SYNC_ARGS=()
for dataset_path in "${DATASET_PATHS[@]}"; do
  NORMALIZATION_SYNC_ARGS+=(--dataset-path "${dataset_path}")
done
python scripts/cache/sync_wsg_normalization_contract.py \
  --modality-config "${MODALITY_CONFIG}" \
  --cache "${NORMALIZATION_CACHE}" \
  "${NORMALIZATION_SYNC_ARGS[@]}"

CONFIG_PATH="${OUTPUT_DIR}/precompute_dense_multidataset.yaml"
python - "${BASE_CONFIG}" "${CONFIG_PATH}" "${MODALITY_CONFIG}" "${NORMALIZATION_CACHE}" "${NORMALIZATION_MODE}" "${DATASET_PATHS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, output_config, modality_config, normalization_cache, normalization_mode, *dataset_paths = sys.argv[1:]
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
params["num_frames"] = 21
params["video_window_stride"] = 16
params["include_conditions"] = True
params["augment"] = False
Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with Path(output_config).open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

pids=()
shard_ids=()
gpu_monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d '[:space:]')"
    if [[ -z "${free_mb}" ]]; then
      echo "[dense-cache] ERROR: could not query free memory for GPU ${gpu_id}" >&2
      exit 1
    fi
    echo "[dense-cache] gpu=${gpu_id} free_mb=${free_mb}"
    if (( free_mb < WSG_MIN_FREE_GPU_MB )); then
      echo "[dense-cache] ERROR: GPU ${gpu_id} has only ${free_mb} MiB free; need at least ${WSG_MIN_FREE_GPU_MB} MiB. Set GPU_IDS to free GPUs or lower WSG_MIN_FREE_GPU_MB if intentional." >&2
      exit 1
    fi
  done
fi

monitor_gpu_usage() {
  local interval="${GPU_USAGE_LOG_INTERVAL:-10}"
  local log_file="${OUTPUT_DIR}/logs/gpu_usage.log"
  echo "[dense-cache] gpu_usage_log=${log_file}"
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

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  device_id="${GPU_ID_LIST[$((shard_id % ${#GPU_ID_LIST[@]}))]}"
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES="${device_id}" python manual_processing/vae_latents/precompute_wsg_vae_latents.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --sample-major \
    --skip-existing \
    --dtype "${DTYPE}" \
    --cache-dtype "${CACHE_DTYPE}" \
    --sample-major-merge-every "${SAMPLE_MAJOR_MERGE_EVERY}" \
    --crop-mode "${WSG_VAE_CROP_MODE}" \
    --filter-invalid-modalities \
    --modalities "${MODALITY_LIST[@]}" \
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
    echo "[dense-cache] $(date '+%F %T') running_shards=${running}/${#pids[@]}"
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
    echo "[dense-cache] ERROR: shard_${shard_id} pid=${pid} exited rc=${rc}"
    status=1
  else
    echo "[dense-cache] shard_${shard_id} pid=${pid} completed"
  fi
done
wait "${monitor_pid}" 2>/dev/null || true
if [[ -n "${gpu_monitor_pid}" ]]; then
  kill "${gpu_monitor_pid}" 2>/dev/null || true
  wait "${gpu_monitor_pid}" 2>/dev/null || true
fi
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

if [[ "${RUN_TEXT_EMBEDDINGS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python manual_processing/text_embeddings/precompute_wsg_text_embeddings.py \
    --config "${CONFIG_PATH}" \
    --latent-cache-dir "${OUTPUT_DIR}" \
    --dtype bf16 \
    --batch-size 8

  for prompt_style in numeric semantic; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python manual_processing/text_embeddings/precompute_wsg_modality_prompt_embeddings.py \
      --config configs/wsm_v1_augustus_train_latent_1.3b.yaml \
      --latent-cache-dir "${OUTPUT_DIR}" \
      --modality-config "${MODALITY_CONFIG}" \
      --prompt-style "${prompt_style}" \
      --dtype bf16 \
      --batch-size 8
  done
fi

python - "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path
import torch

out = Path(sys.argv[1])
for path in sorted((out / "shards").glob("*")):
    shard = next(path.glob("shard_*.pt"), None)
    if shard is None:
        continue
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    print(f"[dense-cache] {path.name}: first_shard_samples={len(payload.get('sample_ids', []))} filtered={payload.get('filtered_sample_ids')}")
PY
