#!/usr/bin/env bash
set -euo pipefail

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

OUTPUT_DIR="data/wsg_vae_latents_avis_train"
mkdir -p "${OUTPUT_DIR}/logs"

MODALITY_ARGS=()
if [[ -n "${WSG_CACHE_MODALITIES:-}" ]]; then
  read -r -a MODALITY_LIST <<< "${WSG_CACHE_MODALITIES}"
  MODALITY_ARGS=(--modalities "${MODALITY_LIST[@]}")
  echo "[VAE cache][avis_train] restricted modalities: ${WSG_CACHE_MODALITIES}"
fi
SKIP_EXISTING_ARGS=(--skip-existing)
if [[ "${WSG_VAE_FORCE_REENCODE:-0}" == "1" ]]; then
  SKIP_EXISTING_ARGS=()
  echo "[VAE cache][avis_train] normalization changed; overwriting restricted modality latents"
fi
CROP_MODE_ARGS=()
if [[ -n "${WSG_VAE_CROP_MODE:-}" ]]; then
  CROP_MODE_ARGS=(--crop-mode "${WSG_VAE_CROP_MODE}")
  echo "[VAE cache][avis_train] crop_mode=${WSG_VAE_CROP_MODE}"
fi
SPARSE_CODEC_ARGS=(
  --sparse-codec-mode "${WSG_SPARSE_CODEC_MODE:-disabled}"
  --sparse-heatmap-sigma "${WSG_SPARSE_HEATMAP_SIGMA:-3.0}"
  --sparse-heatmap-radius "${WSG_SPARSE_HEATMAP_RADIUS:-8}"
  --sparse-heatmap-peak-threshold "${WSG_SPARSE_HEATMAP_PEAK_THRESHOLD:-0.10}"
  --sparse-heatmap-nms-radius "${WSG_SPARSE_HEATMAP_NMS_RADIUS:-6}"
)
if [[ "${WSG_COMPACT_SPARSE_LANES:-1}" == "0" || "${WSG_COMPACT_SPARSE_LANES:-1}" == "false" || "${WSG_COMPACT_SPARSE_LANES:-1}" == "False" ]]; then
  SPARSE_CODEC_ARGS+=(--no-compact-sparse-lanes)
else
  SPARSE_CODEC_ARGS+=(--compact-sparse-lanes)
fi
if [[ "${WSG_SPARSE_CODEC_MODE:-disabled}" != "disabled" ]]; then
  echo "[VAE cache][avis_train] sparse_codec=${WSG_SPARSE_CODEC_MODE:-disabled} sigma=${WSG_SPARSE_HEATMAP_SIGMA:-3.0} radius=${WSG_SPARSE_HEATMAP_RADIUS:-8} compact_lanes=${WSG_COMPACT_SPARSE_LANES:-1}"
fi

pids=()
for shard_id in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${shard_id}" python manual_processing/vae_latents/precompute_wsg_vae_latents.py \
    --config manual_processing/configs/wsm_v1_augustus_precompute_avis_train.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 2 \
    --num-workers 4 \
    --num-shards 8 \
    --shard-id "${shard_id}" \
    --sample-major \
    --dtype fp32 \
    "${SKIP_EXISTING_ARGS[@]}" \
    "${CROP_MODE_ARGS[@]}" \
    "${SPARSE_CODEC_ARGS[@]}" \
    "${MODALITY_ARGS[@]}" \
    > "${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
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
    echo "[VAE cache][avis_train] $(date '+%F %T') running_shards=${running}/${#pids[@]}"
    local shard_id log_file latest
    for shard_id in 0 1 2 3 4 5 6 7; do
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
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
wait "${monitor_pid}" 2>/dev/null || true
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python manual_processing/text_embeddings/precompute_wsg_text_embeddings.py \
  --config manual_processing/configs/wsm_v1_augustus_precompute_avis_train.yaml \
  --latent-cache-dir "${OUTPUT_DIR}" \
  --dtype bf16 \
  --batch-size 8 \
  --skip-existing
