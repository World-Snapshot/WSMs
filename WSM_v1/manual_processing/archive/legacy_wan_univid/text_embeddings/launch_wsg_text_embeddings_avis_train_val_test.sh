#!/usr/bin/env bash
set -euo pipefail

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WaveGen

cd /research/cbim/vast/sf895/code/WSMs/WSM_v1

run_split() {
  local config="$1"
  local latent_cache_dir="$2"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python manual_processing/text_embeddings/precompute_wsg_text_embeddings.py \
    --config "${config}" \
    --latent-cache-dir "${latent_cache_dir}" \
    --dtype bf16 \
    --batch-size 8 \
    --skip-existing
}

run_split "manual_processing/configs/wsm_v1_augustus_precompute_avis_train.yaml" "data/wsg_vae_latents_avis_train"
run_split "manual_processing/configs/wsm_v1_augustus_precompute_avis_val.yaml" "data/wsg_vae_latents_avis_val"
run_split "manual_processing/configs/wsm_v1_augustus_precompute_avis_test.yaml" "data/wsg_vae_latents_avis_test"
