#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/research/cbim/vast/sf895/code/WSMs/WSM_v1}"
RUN_TAG="${2:-head0_step0940_euler10}"
PYTHON_BIN="/research/cbim/vast/sf895/miniforge3/envs/WaveGen/bin/python"
CHECKPOINT="${ROOT}/train_log_custom/Channel_Head_000000/20260828_024500_mp100_instance_crop_gradient_v1/ckpt/0000940/pytorch_lora_weights.safetensors"
OUTPUT_ROOT="${ROOT}/data/wsg_flux2_frame_latents_dense_multidataset/annotations/Universal_Pose/${RUN_TAG}"

mkdir -p "${OUTPUT_ROOT}"

for shard in 0 1 2 3 4 5 6 7; do
    shard_name="$(printf '%05d' "${shard}")"
    session="avis_pose_s${shard}"
    output_dir="${OUTPUT_ROOT}/shard_${shard_name}_of_00008"
    mkdir -p "${output_dir}"
    env -u LD_LIBRARY_PATH tmux kill-session -t "${session}" 2>/dev/null || true
    env -u LD_LIBRARY_PATH tmux new-session -d -s "${session}" \
        "cd '${ROOT}' && PYTHONPATH='${ROOT}' CUDA_VISIBLE_DEVICES='${shard}' '${PYTHON_BIN}' scripts/train_TunnelingNet_Flux2_4B_wsg.py \
        --model-path models/black-forest-labs/FLUX.2-klein-4B \
        --modality-config configs/wsg_v1_modalities_custom.json \
        --dense-latent-cache-dir data/wsg_flux2_frame_latents_dense_multidataset \
        --text-embedding-cache-dir data/wsg_flux2_caption_embeddings_all_datasets \
        --channel-head-index 0 --target-modality Universal_Pose --condition-modalities 0-2 \
        --channel-head-rank 8 --channel-head-alpha 274 \
        --max-steps 1 --batch-size 1 --num-workers 0 --val-num-workers 0 --val-loss-batches 0 \
        --seed 20260622 --disable-preflight --output-dir '${output_dir}' \
        --inference-only --inference-checkpoint '${CHECKPOINT}' --categorical-refiner off \
        --inference-label annotations --inference-steps 10 \
        --inference-shard-id '${shard}' --inference-num-shards 8 \
        --inference-save-raw-latent --inference-cache-only \
        --preview-split all --preview-num-samples 100000 --preview-random \
        --trainable-target on --trainable-condition off \
        > '${output_dir}/launch.log' 2>&1"
    printf 'launched %s on GPU %s -> %s\n' "${session}" "${shard}" "${output_dir}"
done

