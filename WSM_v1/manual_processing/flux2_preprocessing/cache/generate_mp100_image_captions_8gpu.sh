#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
LOG_DIR="${PROJECT_ROOT}/data/mp100_universal_pose_flux2_frame_latents/caption_logs"

mkdir -p "${LOG_DIR}"
PIDS=()
for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
  CUDA_VISIBLE_DEVICES="${RANK}" \
  HF_HUB_DISABLE_PROGRESS_BARS=1 \
  TRANSFORMERS_VERBOSITY=error \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mp100_image_captions.py" \
    --device cuda \
    --dtype fp16 \
    --batch-size 8 \
    --num-beams 4 \
    --num-shards "${NUM_GPUS}" \
    --shard-id "${RANK}" \
    >"${LOG_DIR}/caption_rank_${RANK}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more caption workers failed. Inspect ${LOG_DIR}." >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mp100_image_captions.py" \
  --num-shards "${NUM_GPUS}" \
  --merge \
  --install-prompts
