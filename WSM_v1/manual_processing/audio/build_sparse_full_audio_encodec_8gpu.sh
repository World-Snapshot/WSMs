#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/research/cbim/vast/sf895/code/WSMs/WSM_v1}"
PYTHON_BIN="${PYTHON_BIN:-/research/cbim/vast/sf895/miniforge3/envs/WaveGen/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-/common/users/sf895/wave2pixel/AVIS_train_results}"
VAL_ROOT="${VAL_ROOT:-/common/users/sf895/wave2pixel/AVIS_val_results}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/sparse_structure/datasets}"
LOG_DIR="${LOG_DIR:-${ROOT}/train_log_sparse/launch_logs}"
SHARD_DIR="${SHARD_DIR:-${OUT_DIR}/encodec_full_audio_shards}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FRAME_STRIDE="${FRAME_STRIDE:-8}"
AUDIO_CODEC_SECONDS="${AUDIO_CODEC_SECONDS:-10}"
AUDIO_CODEC_BANDWIDTH="${AUDIO_CODEC_BANDWIDTH:-6.0}"
AUDIO_MODEL_NAME="${AUDIO_MODEL_NAME:-facebook/encodec_24khz}"
TRAIN_NAME="${TRAIN_NAME:-wsv_structured_scorehmr_full_audio_sem195_encodec_10s_bw6.0_train.npz}"
VAL_NAME="${VAL_NAME:-wsv_structured_scorehmr_full_audio_sem195_encodec_10s_bw6.0_val.npz}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "Need at least NUM_SHARDS GPU ids, got ${#GPU_ARRAY[@]} for ${NUM_SHARDS}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SHARD_DIR}"
cd "${ROOT}"

echo "[encodec-8gpu] root=${ROOT}"
echo "[encodec-8gpu] train_root=${TRAIN_ROOT} val_root=${VAL_ROOT}"
echo "[encodec-8gpu] shards=${NUM_SHARDS} gpus=${GPU_IDS} out_dir=${OUT_DIR}"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ARRAY[$shard]}"
  shard_train="encodec_full_audio_train_shard$(printf '%02d' "${shard}")of$(printf '%02d' "${NUM_SHARDS}").npz"
  shard_val="encodec_full_audio_val_shard$(printf '%02d' "${shard}")of$(printf '%02d' "${NUM_SHARDS}").npz"
  log_path="${LOG_DIR}/encodec_full_audio_${STAMP}_shard$(printf '%02d' "${shard}").log"
  echo "[encodec-8gpu] launch shard=${shard}/${NUM_SHARDS} gpu=${gpu} log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" third_party/ScoreHMR/wsm_adapters/build_wsv_audio_dataset.py \
      --train-root "${TRAIN_ROOT}" \
      --val-root "${VAL_ROOT}" \
      --out-dir "${SHARD_DIR}" \
      --train-name "${shard_train}" \
      --val-name "${shard_val}" \
      --frame-stride "${FRAME_STRIDE}" \
      --audio-encoder encodec \
      --audio-model-name "${AUDIO_MODEL_NAME}" \
      --audio-source full \
      --audio-codec-seconds "${AUDIO_CODEC_SECONDS}" \
      --audio-codec-bandwidth "${AUDIO_CODEC_BANDWIDTH}" \
      --audio-encoder-device cuda \
      --audio-encoder-batch-size 1 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}"
  ) > "${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[encodec-8gpu] at least one shard failed; inspect ${LOG_DIR}/encodec_full_audio_${STAMP}_shard*.log" >&2
  exit 1
fi

merge_train_args=()
merge_val_args=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  merge_train_args+=(--merge-input "${SHARD_DIR}/encodec_full_audio_train_shard$(printf '%02d' "${shard}")of$(printf '%02d' "${NUM_SHARDS}").npz")
  merge_val_args+=(--merge-input "${SHARD_DIR}/encodec_full_audio_val_shard$(printf '%02d' "${shard}")of$(printf '%02d' "${NUM_SHARDS}").npz")
done

"${PYTHON_BIN}" third_party/ScoreHMR/wsm_adapters/build_wsv_audio_dataset.py \
  "${merge_train_args[@]}" \
  --merge-output "${OUT_DIR}/${TRAIN_NAME}"
"${PYTHON_BIN}" third_party/ScoreHMR/wsm_adapters/build_wsv_audio_dataset.py \
  "${merge_val_args[@]}" \
  --merge-output "${OUT_DIR}/${VAL_NAME}"

echo "[encodec-8gpu] done train=${OUT_DIR}/${TRAIN_NAME} val=${OUT_DIR}/${VAL_NAME}"
