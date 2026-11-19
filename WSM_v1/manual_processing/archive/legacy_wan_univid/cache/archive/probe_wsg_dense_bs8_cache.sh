#!/usr/bin/env bash
set -euo pipefail

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
CONFIG_PATH="${CONFIG_PATH:-data/wsg_vae_latents_dense_multidataset/precompute_dense_multidataset.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/wsg_vae_latents_dense_multidataset_bs8_probe_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DTYPE="${DTYPE:-fp32}"

read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
mkdir -p "${OUTPUT_DIR}/logs"

mapfile -t MODALITIES < <(python - "${MODALITY_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

from src.dataset.wsg_dataset import flatten_wsg_modality_graph

with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
    graph = flatten_wsg_modality_graph(json.load(f)["wsg_v1"])
modalities = []
for target, spec in graph.items():
    modalities.append(str(target))
    modalities.extend(str(condition) for condition in (spec.get("conditions") or {}))
print("\n".join(dict.fromkeys(modalities)))
PY
)

{
  echo "[bs8-probe] output_dir=${OUTPUT_DIR}"
  echo "[bs8-probe] gpu_ids=${GPU_IDS} num_shards=${NUM_SHARDS} batch_size=${BATCH_SIZE} max_samples=${MAX_SAMPLES}"
  echo "[bs8-probe] modalities=${MODALITIES[*]}"
} | tee "${OUTPUT_DIR}/logs/main.log"

(
  while true; do
    date "+%F %T"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    sleep 5
  done
) > "${OUTPUT_DIR}/logs/gpu_usage.log" 2>&1 &
monitor_pid="$!"

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  device_id="${GPU_ID_LIST[$((shard_id % ${#GPU_ID_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="${device_id}" python manual_processing/vae_latents/precompute_wsg_vae_latents.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --max-samples "${MAX_SAMPLES}" \
    --sample-major \
    --skip-existing \
    --dtype "${DTYPE}" \
    --crop-mode center_short_side_square \
    --filter-invalid-modalities \
    --modalities "${MODALITIES[@]}" \
    > "${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
while true; do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  echo "[bs8-probe] $(date '+%F %T') running_shards=${running}/${#pids[@]}" | tee -a "${OUTPUT_DIR}/logs/main.log"
  [[ "${running}" -eq 0 ]] && break
  sleep 30
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
kill "${monitor_pid}" 2>/dev/null || true

echo "[bs8-probe] done status=${status}" | tee -a "${OUTPUT_DIR}/logs/main.log"
exit "${status}"
