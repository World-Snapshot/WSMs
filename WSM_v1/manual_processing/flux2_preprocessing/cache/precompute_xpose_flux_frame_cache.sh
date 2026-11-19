#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-xpose_flux_frame_cache}"
PROJECT_ROOT=/research/cbim/vast/sf895/code/WSMs/WSM_v1
DATASET_DIR="${DATASET_DIR:-/common/users/sf895/wsg/output/XPose_results}"
OUTPUT_DIR="${OUTPUT_DIR:-data/xpose_flux2_frame_latents_sample10pct_2frames}"
FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-models/black-forest-labs/FLUX.2-klein-4B}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_FRAMES="${NUM_FRAMES:-2}"
VIDEO_SAMPLE_FRACTION="${VIDEO_SAMPLE_FRACTION:-0.1}"
SAMPLE_SEED="${SAMPLE_SEED:-20260621}"
PATCH_HW="${PATCH_HW:-480}"
DTYPE="${DTYPE:-bf16}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
MIN_FREE_GPU_MB="${WSG_MIN_FREE_GPU_MB:-6000}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"

if [[ "${RUN_IN_TMUX:-1}" == "1" && -z "${TMUX:-}" ]]; then
  cd "${PROJECT_ROOT}"
  mkdir -p "${OUTPUT_DIR}/logs"
  env -u LD_LIBRARY_PATH tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${PROJECT_ROOT} && RUN_IN_TMUX=0 bash manual_processing/flux2_preprocessing/cache/precompute_xpose_flux_frame_cache.sh"
  echo "[xpose-flux] started tmux session ${SESSION_NAME}"
  exit 0
fi

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/stop"
exec > >(tee -a "${OUTPUT_DIR}/logs/main.log") 2>&1
read -r -a GPU_ID_LIST <<< "${GPU_IDS}"

echo "[xpose-flux] dataset=${DATASET_DIR} output=${OUTPUT_DIR}"
echo "[xpose-flux] source_videos=51797 video_sample_fraction=${VIDEO_SAMPLE_FRACTION} random_frames_per_video=${NUM_FRAMES} sample_seed=${SAMPLE_SEED}"
echo "[xpose-flux] dtype=${DTYPE} cache_dtype=${CACHE_DTYPE} batch_videos=${BATCH_SIZE}"

while true; do
  ready=1
  for gpu in "${GPU_ID_LIST[@]}"; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d '[:space:]')"
    if (( free_mb < MIN_FREE_GPU_MB )); then ready=0; fi
  done
  if (( ready == 1 )); then break; fi
  if [[ "${WAIT_FOR_GPUS}" != "1" ]]; then echo "[xpose-flux] GPUs busy"; exit 1; fi
  echo "[xpose-flux] $(date '+%F %T') waiting for all GPUs to have ${MIN_FREE_GPU_MB} MiB free"
  sleep 60
done

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ID_LIST[$((shard_id % ${#GPU_ID_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python manual_processing/flux2_preprocessing/cache/precompute_xpose_flux_frame_latents.py \
    --dataset-dir "${DATASET_DIR}" --output-dir "${OUTPUT_DIR}" \
    --flux-model-path "${FLUX_MODEL_PATH}" --num-shards "${NUM_SHARDS}" --shard-id "${shard_id}" \
    --batch-size "${BATCH_SIZE}" --num-frames "${NUM_FRAMES}" --one-window-per-sample \
    --video-sample-fraction "${VIDEO_SAMPLE_FRACTION}" --sample-seed "${SAMPLE_SEED}" \
    --patch-hw "${PATCH_HW}" --dtype "${DTYPE}" --cache-dtype "${CACHE_DTYPE}" --skip-existing \
    --stop-file "${OUTPUT_DIR}/stop/shard_${shard_id}.stop" \
    > "${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
while true; do
  running=0
  for pid in "${pids[@]}"; do kill -0 "${pid}" 2>/dev/null && running=$((running + 1)); done
  echo "[xpose-flux] $(date '+%F %T') running_shards=${running}/${#pids[@]}"
  for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    log="${OUTPUT_DIR}/logs/shard_${shard_id}.log"
    [[ -f "${log}" ]] && tail -c 4096 "${log}" | tr '\r' '\n' | sed '/^[[:space:]]*$/d' | tail -n 1 | sed "s/^/  shard_${shard_id}: /"
  done
  (( running == 0 )) && break
  sleep 60
done
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
