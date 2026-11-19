#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-xpose_pose_vae_cache}"

if [[ "${RUN_IN_TMUX:-1}" == "1" && -z "${TMUX:-}" ]]; then
  cd /research/cbim/vast/sf895/code/WSMs/WSM_v1
  mkdir -p "${OUTPUT_DIR:-data/xpose_vae_latents}/logs"
  env -u LD_LIBRARY_PATH tmux new-session -d -s "${SESSION_NAME}" \
    "cd /research/cbim/vast/sf895/code/WSMs/WSM_v1 && RUN_IN_TMUX=0 bash manual_processing/cache/precompute_xpose_pose_vae_cache.sh"
  echo "[xpose-cache] started tmux session ${SESSION_NAME}"
  echo "[xpose-cache] attach: env -u LD_LIBRARY_PATH tmux attach -t ${SESSION_NAME}"
  exit 0
fi

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

DATASET_DIR="${DATASET_DIR:-/common/users/sf895/wsg/output/XPose_results}"
CONFIG_PATH="${CONFIG_PATH:-manual_processing/configs/wsm_v1_augustus_precompute_avis_train.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/xpose_vae_latents}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_FRAMES="${NUM_FRAMES:-21}"
VIDEO_WINDOW_STRIDE="${VIDEO_WINDOW_STRIDE:-16}"
PATCH_HW="${PATCH_HW:-480}"
DTYPE="${DTYPE:-bf16}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
ONE_WINDOW_PER_SAMPLE="${ONE_WINDOW_PER_SAMPLE:-1}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_TEXT_EMBEDDINGS="${RUN_TEXT_EMBEDDINGS:-1}"
TEXT_EMBEDDING_GPU="${TEXT_EMBEDDING_GPU:-${GPU_IDS%% *}}"
TEXT_EMBEDDING_GPU_IDS="${TEXT_EMBEDDING_GPU_IDS:-${GPU_IDS}}"
TEXT_EMBEDDING_BATCH_SIZE="${TEXT_EMBEDDING_BATCH_SIZE:-8}"
WSG_MIN_FREE_GPU_MB="${WSG_MIN_FREE_GPU_MB:-6000}"
VAE_CACHE_PROGRESS_INTERVAL="${VAE_CACHE_PROGRESS_INTERVAL:-60}"

read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
read -r -a TEXT_EMBEDDING_GPU_ID_LIST <<< "${TEXT_EMBEDDING_GPU_IDS}"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "[xpose-cache] ERROR: GPU_IDS is empty" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
STOP_DIR="${OUTPUT_DIR}/stop"
mkdir -p "${STOP_DIR}"
rm -f "${STOP_DIR}"/shard_*.stop 2>/dev/null || true
if [[ "${XPOSE_CACHE_TEE_MAIN_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "${OUTPUT_DIR}/logs/main.log") 2>&1
fi

echo "[xpose-cache] dataset_dir=${DATASET_DIR}"
echo "[xpose-cache] output_dir=${OUTPUT_DIR}"
echo "[xpose-cache] num_shards=${NUM_SHARDS} batch_size=${BATCH_SIZE}"
echo "[xpose-cache] num_frames=${NUM_FRAMES} stride=${VIDEO_WINDOW_STRIDE} patch_hw=${PATCH_HW}"
echo "[xpose-cache] one_window_per_sample=${ONE_WINDOW_PER_SAMPLE}"
echo "[xpose-cache] run_text_embeddings=${RUN_TEXT_EMBEDDINGS} text_embedding_gpu=${TEXT_EMBEDDING_GPU}"
echo "[xpose-cache] text_embedding_gpu_ids=${TEXT_EMBEDDING_GPU_ID_LIST[*]}"
echo "[xpose-cache] gpu_ids=${GPU_ID_LIST[*]} min_free_gpu_mb=${WSG_MIN_FREE_GPU_MB}"
echo "[xpose-cache] dtype=${DTYPE} cache_dtype=${CACHE_DTYPE}"

request_stop() {
  echo "[xpose-cache] $(date '+%F %T') stop requested; shards will merge completed batches after their current batch"
  mkdir -p "${STOP_DIR}"
  local sid
  for sid in $(seq 0 $((NUM_SHARDS - 1))); do
    : > "${STOP_DIR}/shard_${sid}.stop"
  done
}

trap request_stop INT TERM HUP

if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d '[:space:]')"
    if [[ -z "${free_mb}" ]]; then
      echo "[xpose-cache] ERROR: could not query free memory for GPU ${gpu_id}" >&2
      exit 1
    fi
    echo "[xpose-cache] gpu=${gpu_id} free_mb=${free_mb}"
    if (( free_mb < WSG_MIN_FREE_GPU_MB )); then
      echo "[xpose-cache] ERROR: GPU ${gpu_id} has only ${free_mb} MiB free; need ${WSG_MIN_FREE_GPU_MB} MiB." >&2
      exit 1
    fi
  done
fi

pids=()
shard_ids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  device_id="${GPU_ID_LIST[$((shard_id % ${#GPU_ID_LIST[@]}))]}"
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES="${device_id}" \
    python manual_processing/cache/precompute_xpose_pose_vae_latents.py \
      --dataset-dir "${DATASET_DIR}" \
      --config "${CONFIG_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-id "${shard_id}" \
      --num-frames "${NUM_FRAMES}" \
      --video-window-stride "${VIDEO_WINDOW_STRIDE}" \
      --patch-hw "${PATCH_HW}" \
      --dtype "${DTYPE}" \
      --cache-dtype "${CACHE_DTYPE}" \
      --skip-existing \
      --stop-file "${STOP_DIR}/shard_${shard_id}.stop" \
      $([[ "${ONE_WINDOW_PER_SAMPLE}" == "1" ]] && echo "--one-window-per-sample") \
      > "${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
  shard_ids+=("${shard_id}")
done

monitor_shards() {
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
    echo "[xpose-cache] $(date '+%F %T') running_shards=${running}/${#pids[@]}"
    local shard_id log_file latest
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
      log_file="${OUTPUT_DIR}/logs/shard_${shard_id}.log"
      if [[ -f "${log_file}" ]]; then
        latest="$(tail -c 4096 "${log_file}" | tr '\r' '\n' | sed -E 's/\x1B\[[0-9;?]*[A-Za-z]//g' | sed '/^[[:space:]]*$/d' | tail -n 1 || true)"
        [[ -n "${latest}" ]] && echo "  shard_${shard_id}: ${latest}"
      fi
    done
    sleep "${VAE_CACHE_PROGRESS_INTERVAL}"
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
    echo "[xpose-cache] ERROR: shard_${shard_id} pid=${pid} exited rc=${rc}"
    status=1
  else
    echo "[xpose-cache] shard_${shard_id} pid=${pid} completed"
  fi
done

wait "${monitor_pid}" 2>/dev/null || true
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

if [[ "${RUN_TEXT_EMBEDDINGS}" == "1" ]]; then
  echo "[xpose-cache] encoding XPose caption text embeddings on ${#TEXT_EMBEDDING_GPU_ID_LIST[@]} GPUs"
  text_pids=()
  text_status=0
  text_num_shards="${#TEXT_EMBEDDING_GPU_ID_LIST[@]}"
  for text_shard_id in $(seq 0 $((text_num_shards - 1))); do
    text_device_id="${TEXT_EMBEDDING_GPU_ID_LIST[$text_shard_id]}"
    CUDA_VISIBLE_DEVICES="${text_device_id}" python manual_processing/text_embeddings/precompute_wsg_text_embeddings.py \
      --config "${CONFIG_PATH}" \
      --latent-cache-dir "${OUTPUT_DIR}" \
      --dtype bf16 \
      --batch-size "${TEXT_EMBEDDING_BATCH_SIZE}" \
      --num-shards "${text_num_shards}" \
      --shard-id "${text_shard_id}" \
      --filter-to-latent-samples \
      --skip-existing \
      > "${OUTPUT_DIR}/logs/text_shard_${text_shard_id}.log" 2>&1 &
    text_pids+=("$!")
  done
  for text_pid in "${text_pids[@]}"; do
    if ! wait "${text_pid}"; then
      text_status=1
    fi
  done
  if [[ "${text_status}" -ne 0 ]]; then
    echo "[xpose-cache] ERROR: one or more text embedding shards failed" >&2
    exit "${text_status}"
  fi

  echo "[xpose-cache] encoding XPose numeric and semantic modality prompt embeddings"
  for prompt_style in numeric semantic; do
    CUDA_VISIBLE_DEVICES="${TEXT_EMBEDDING_GPU}" python manual_processing/text_embeddings/precompute_wsg_modality_prompt_embeddings.py \
      --config "${CONFIG_PATH}" \
      --latent-cache-dir "${OUTPUT_DIR}" \
      --modality-config configs/wsg_v1_modalities_dense.json \
      --prompt-style "${prompt_style}" \
      --dtype bf16 \
      --batch-size "${TEXT_EMBEDDING_BATCH_SIZE}" \
      --skip-existing
  done
fi
