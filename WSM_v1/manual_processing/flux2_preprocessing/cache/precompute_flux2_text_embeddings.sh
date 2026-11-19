#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/research/cbim/vast/sf895/code/WSMs/WSM_v1}"
PYTHON_BIN="${PYTHON_BIN:-/research/cbim/vast/sf895/miniforge3/envs/WaveGen/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/black-forest-labs/FLUX.2-klein-4B}"
MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
DENSE_LATENT_CACHE_DIR="${DENSE_LATENT_CACHE_DIR:-data/wsg_flux2_frame_latents_dense_multidataset}"
XPOSE_LATENT_CACHE_DIR="${XPOSE_LATENT_CACHE_DIR:-data/xpose_flux2_frame_latents_sample10pct_2frames}"
LATENT_CACHE_DIRS="${LATENT_CACHE_DIRS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data/wsg_flux2_caption_embeddings_all_datasets}"
CHANNEL_HEAD_INDEX="${CHANNEL_HEAD_INDEX:-270}"
CHANNEL_HEAD_INDICES="${CHANNEL_HEAD_INDICES:-}"
ALL_CHANNEL_HEADS="${ALL_CHANNEL_HEADS:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"
TMUX_SESSION="${TMUX_SESSION:-flux2_text_all_channel_heads}"
GLOBAL_CAPTIONS="${GLOBAL_CAPTIONS:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/data/wsg_flux2_caption_embeddings_all_datasets/logs/global_captions}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --modality-config "${MODALITY_CONFIG}"
  --dense-latent-cache-dir "${DENSE_LATENT_CACHE_DIR}"
  --xpose-latent-cache-dir "${XPOSE_LATENT_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --num-shards "${NUM_SHARDS}"
  --batch-size "${BATCH_SIZE}"
)
if [[ -n "${LATENT_CACHE_DIRS}" ]]; then
  IFS=, read -r -a LATENT_CACHE_ARRAY <<< "${LATENT_CACHE_DIRS}"
  for LATENT_CACHE_DIR in "${LATENT_CACHE_ARRAY[@]}"; do
    COMMON_ARGS+=(--latent-cache-dir "${LATENT_CACHE_DIR}")
  done
fi
if [[ "${GLOBAL_CAPTIONS}" == "1" ]]; then
  COMMON_ARGS+=(--global-captions)
elif [[ -n "${CHANNEL_HEAD_INDICES}" ]]; then
  COMMON_ARGS+=(--channel-head-indices "${CHANNEL_HEAD_INDICES}")
elif [[ "${ALL_CHANNEL_HEADS}" == "1" ]]; then
  COMMON_ARGS+=(--all-channel-heads)
else
  COMMON_ARGS+=(--channel-head-index "${CHANNEL_HEAD_INDEX}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

IFS=, read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${RUN_IN_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  tmux new-session -d -s "${TMUX_SESSION}" "cd ${PROJECT_ROOT} && RUN_IN_TMUX=0 GPU_IDS=${GPU_IDS} GLOBAL_CAPTIONS=${GLOBAL_CAPTIONS} ALL_CHANNEL_HEADS=${ALL_CHANNEL_HEADS} CHANNEL_HEAD_INDEX=${CHANNEL_HEAD_INDEX} CHANNEL_HEAD_INDICES=${CHANNEL_HEAD_INDICES} OUTPUT_DIR=${OUTPUT_DIR} DRY_RUN=${DRY_RUN} OVERWRITE=${OVERWRITE} bash manual_processing/flux2_preprocessing/cache/precompute_flux2_text_embeddings.sh"
  echo "[flux2-text] started tmux session ${TMUX_SESSION}"
  echo "[flux2-text] logs=${LOG_DIR}"
  exit 0
fi

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  gpu="${GPU_ARRAY[$((shard % ${#GPU_ARRAY[@]}))]}"
  log="${LOG_DIR}/shard_${shard}.log"
  echo "[flux2-text] shard=${shard}/${NUM_SHARDS} gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" manual_processing/flux2_preprocessing/cache/precompute_flux2_text_embeddings.py \
    "${COMMON_ARGS[@]}" \
    --shard-id "${shard}" \
    --device cuda > "${log}" 2>&1 &
done
wait
echo "[flux2-text] all shards complete"
