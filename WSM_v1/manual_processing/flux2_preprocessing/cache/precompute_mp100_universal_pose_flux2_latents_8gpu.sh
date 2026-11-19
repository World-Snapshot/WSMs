#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-mp100_universal_pose_instance_crop_flux2_cache}"
PROJECT_ROOT="${PROJECT_ROOT:-/research/cbim/vast/sf895/code/WSMs/WSM_v1}"
PYTHON_BIN="${PYTHON_BIN:-/research/cbim/vast/sf895/miniforge3/envs/WaveGen/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/research/cbim/vast/sf895/code/wsg/output/PoseAnything_MP100}"
OUTPUT_DIR="${OUTPUT_DIR:-data/mp100_universal_pose_flux2_instance_crop_latents}"
FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-models/black-forest-labs/FLUX.2-klein-4B}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATCH_HW="${PATCH_HW:-480}"
DTYPE="${DTYPE:-bf16}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
MIN_FREE_GPU_MB="${WSG_MIN_FREE_GPU_MB:-6000}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

SCRIPT="manual_processing/flux2_preprocessing/cache/precompute_mp100_universal_pose_flux2_latents.py"

if [[ "${RUN_IN_TMUX:-1}" == "1" && -z "${TMUX:-}" ]]; then
  cd "${PROJECT_ROOT}"
  mkdir -p "${OUTPUT_DIR}/logs"
  env -u LD_LIBRARY_PATH tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${PROJECT_ROOT} && FORCE_REBUILD=${FORCE_REBUILD} RUN_IN_TMUX=0 bash ${BASH_SOURCE[0]}"
  echo "[mp100-flux] started tmux session ${SESSION_NAME}"
  echo "[mp100-flux] monitor: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${OUTPUT_DIR}/logs/main.log") 2>&1
read -r -a GPU_ID_LIST <<< "${GPU_IDS}"

if (( ${#GPU_ID_LIST[@]} < NUM_SHARDS )); then
  echo "Need at least NUM_SHARDS GPU IDs; got ${#GPU_ID_LIST[@]} for ${NUM_SHARDS}" >&2
  exit 1
fi

echo "[mp100-flux] dataset=${DATASET_ROOT} output=${OUTPUT_DIR}"
echo "[mp100-flux] transform=instance_crop_then_letterbox_square patch=${PATCH_HW} shards=${NUM_SHARDS} gpus=${GPU_IDS}"

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
  echo "[mp100-flux] smoke/sample limit=${MAX_SAMPLES}"
fi

SKIP_EXISTING_ARGS=(--skip-existing)
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  SKIP_EXISTING_ARGS=()
  echo "[mp100-flux] force rebuild enabled"
fi

# Strict pair/category audit and shared index generation; this returns before
# loading the VAE and is safe to run on CPU.
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${SCRIPT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --flux-model-path "${FLUX_MODEL_PATH}" \
  --patch-hw "${PATCH_HW}" \
  --num-shards "${NUM_SHARDS}" \
  "${MAX_SAMPLE_ARGS[@]}" \
  --index-only

while true; do
  ready=1
  for gpu in "${GPU_ID_LIST[@]:0:${NUM_SHARDS}}"; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d '[:space:]')"
    if (( free_mb < MIN_FREE_GPU_MB )); then ready=0; fi
  done
  if (( ready == 1 )); then break; fi
  if [[ "${WAIT_FOR_GPUS}" != "1" ]]; then
    echo "[mp100-flux] GPUs busy" >&2
    exit 1
  fi
  echo "[mp100-flux] $(date '+%F %T') waiting for ${MIN_FREE_GPU_MB} MiB free per GPU"
  sleep 60
done

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ID_LIST[$shard_id]}"
  echo "[mp100-flux] launch shard=${shard_id}/${NUM_SHARDS} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --flux-model-path "${FLUX_MODEL_PATH}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --patch-hw "${PATCH_HW}" \
    --dtype "${DTYPE}" \
    --cache-dtype "${CACHE_DTYPE}" \
    "${MAX_SAMPLE_ARGS[@]}" \
    "${SKIP_EXISTING_ARGS[@]}" \
    > "${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
while true; do
  running=0
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && running=$((running + 1))
  done
  echo "[mp100-flux] $(date '+%F %T') running_shards=${running}/${#pids[@]}"
  for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    log="${OUTPUT_DIR}/logs/shard_${shard_id}.log"
    [[ -f "${log}" ]] && tail -c 4096 "${log}" | tr '\r' '\n' | sed '/^[[:space:]]*$/d' | tail -n 1 | sed "s/^/  shard_${shard_id}: /"
  done
  (( running == 0 )) && break
  sleep 30
done

for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "[mp100-flux] at least one shard failed; inspect ${OUTPUT_DIR}/logs/shard_*.log" >&2
  exit "${status}"
fi

echo "[mp100-flux] complete: ${OUTPUT_DIR}"
