#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCOREHMR_ROOT="${WSM_ROOT}/third_party/ScoreHMR"
SPARSE_DATA_ROOT="${SPARSE_DATA_ROOT:-${WSM_ROOT}/data/sparse_structure}"
SPARSE_CACHE_ROOT="${SPARSE_CACHE_ROOT:-${SPARSE_DATA_ROOT}/cache}"
SPARSE_LOG_ROOT="${SPARSE_LOG_ROOT:-${WSM_ROOT}/train_log_sparse}"
SPARSE_CLEAN_LOG_LAYOUT="${SPARSE_CLEAN_LOG_LAYOUT:-1}"
SPARSE_DATASET_ROOT="${SPARSE_DATASET_ROOT:-${SPARSE_DATA_ROOT}/datasets}"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# Preferred path: choose sparse Channel Heads from the sparse config, then
# dispatch each entry by its adapter_task. TASKS is kept as a legacy override
# for direct adapter experiments.
SPARSE_CONFIG_PATH="${SPARSE_CONFIG_PATH:-${WSM_ROOT}/configs/wsg_v1_modalities_sparse.json}"
case "${SPARSE_CONFIG_PATH}" in
  *wsg_v1_modalities_sparse_single_condition_channel_head_condition_first.json)
    SPARSE_CONFIG_PATH="${WSM_ROOT}/configs/wsg_v1_modalities_sparse.json" ;;
esac
# Heads 008/009 remain opt-in audio-mask generation experiments. Head 010 is
# the validated RGB-mask + audio region-selector task and is part of the
# official sparse default set.
CHANNEL_HEADS="${CHANNEL_HEADS:-0,1,2,3,5,6,10}"
TASKS="${TASKS:-}"
SKIP_UNIMPLEMENTED="${SKIP_UNIMPLEMENTED:-1}"
# Max number of sparse Channel Heads to run concurrently. 0 means strictly
# serial. The default sparse set maps seven heads to seven GPUs. Channel_Head_000007
# (visual-to-audio generation) is kept opt-in because its current quality is not
# reliable enough for default sparse training.
PARALLEL_TASKS="${PARALLEL_TASKS:-7}"

RUN_NAME="${RUN_NAME:-wsv_structured_scorehmr}"
DATASET_PREFIX="${DATASET_PREFIX:-wsv_structured_scorehmr}"
SESSION_NAME="${SESSION_NAME:-scorehmr_${RUN_NAME}}"

TRAIN_ROOT="${TRAIN_ROOT:-/common/users/sf895/wave2pixel/AVIS_train_results}"
VAL_ROOT="${VAL_ROOT:-/common/users/sf895/wave2pixel/AVIS_val_results}"
OBJ_TRAIN_ROOTS="${OBJ_TRAIN_ROOTS:-/research/cbim/vast/sf895/code/wsg/output/Articulation_results,/research/cbim/vast/sf895/code/wsg/output/ShapeNetSem_results}"
OBJ_VAL_ROOTS="${OBJ_VAL_ROOTS:-}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_VIDEO="${MAX_FRAMES_PER_VIDEO:-}"

TOTAL_STEPS="${TOTAL_STEPS:-10000}"
TOTAL_STEPS_BBOX="${TOTAL_STEPS_BBOX:-3000}"
TOTAL_STEPS_POSE="${TOTAL_STEPS_POSE:-3000}"
BBOX_LR="${BBOX_LR:-3e-4}"
BBOX_LOSS_MODE="${BBOX_LOSS_MODE:-pretrained_center_gaussian}"
POSE_LR="${POSE_LR:-3e-4}"
POSE_MODE="${POSE_MODE:-kinematic_rotations}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-500}"
VALIDATION_FREQ="${VALIDATION_FREQ:-100}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-500}"
TOTAL_STEPS_AUDIO_CATEGORY="${TOTAL_STEPS_AUDIO_CATEGORY:-100000}"
TOTAL_STEPS_AUDIO_GENERATION="${TOTAL_STEPS_AUDIO_GENERATION:-100000}"
TOTAL_STEPS_AUDIO_DENSE_MASK="${TOTAL_STEPS_AUDIO_DENSE_MASK:-1000}"
USE_BETAS="${USE_BETAS:-1}"
BOX_SOURCE="${BOX_SOURCE:-smpl72}"
SMPL_BOX_SOURCE="${SMPL_BOX_SOURCE:-smplerx_region}"

MAX_BBOX_SLOTS="${MAX_BBOX_SLOTS:-10}"
BBOX_MIN_VISIBLE_POINTS="${BBOX_MIN_VISIBLE_POINTS:-8}"
BBOX_VISIBILITY_THRESHOLD="${BBOX_VISIBILITY_THRESHOLD:-0.5}"
BBOX_MAX_BOX_AREA_FRACTION="${BBOX_MAX_BOX_AREA_FRACTION:-0.8}"
BBOX_MARGIN_MODE="${BBOX_MARGIN_MODE:-relative}"
BBOX_MARGIN_PIXELS="${BBOX_MARGIN_PIXELS:-20}"
BBOX_MARGIN_FRACTION="${BBOX_MARGIN_FRACTION:-0.2}"
BBOX_REQUIRE_DWPOSE="${BBOX_REQUIRE_DWPOSE:-1}"
BBOX_DWPOSE_MIN_POINTS="${BBOX_DWPOSE_MIN_POINTS:-8}"
BBOX_DWPOSE_SCORE_THRESHOLD="${BBOX_DWPOSE_SCORE_THRESHOLD:-0.5}"
MAX_OBJ_SLOTS="${MAX_OBJ_SLOTS:-128}"
OBJ_MODE="${OBJ_MODE:-growing_tree_ce_base}"
OBJ_LR="${OBJ_LR:-3e-4}"
AUDIO_LR="${AUDIO_LR:-1e-3}"
AUDIO_ENCODER="${AUDIO_ENCODER:-ast}"
AUDIO_MODEL_NAME="${AUDIO_MODEL_NAME:-MIT/ast-finetuned-audioset-10-10-0.4593}"
AUDIO_GENERATION_ENCODER="${AUDIO_GENERATION_ENCODER:-encodec}"
AUDIO_GENERATION_MODEL_NAME="${AUDIO_GENERATION_MODEL_NAME:-facebook/encodec_24khz}"
AUDIO_ENCODER_DEVICE="${AUDIO_ENCODER_DEVICE:-cuda}"
AUDIO_GENERATION_ENCODER_DEVICE="${AUDIO_GENERATION_ENCODER_DEVICE:-cpu}"
AUDIO_DIM="${AUDIO_DIM:-1024}"
AUDIO_ENCODER_BATCH_SIZE="${AUDIO_ENCODER_BATCH_SIZE:-16}"
AUDIO_FRAME_SAMPLES_PER_VIDEO="${AUDIO_FRAME_SAMPLES_PER_VIDEO:-all}"
# Best validated sparse-audio recipes.  These are deliberately head-specific:
# Head 005 predicts the sound-source box from RGB + audio, while Head 006
# classifies the source category from audio alone.  Every value remains
# overridable for ablations without changing the reproducible defaults.
HEAD005_RECIPE="${HEAD005_RECIPE:-clip_frozen_mask_fused}"
HEAD005_INIT_CHECKPOINT="${HEAD005_INIT_CHECKPOINT:-${SPARSE_LOG_ROOT}/Channel_Head_000005/20260811_183000_clip_frozen_mask_fused_final/ckpt/model-best.pt}"
HEAD005_TOTAL_STEPS="${HEAD005_TOTAL_STEPS:-2000}"
HEAD005_LR="${HEAD005_LR:-1e-3}"
HEAD005_FUSION_CROSS_WEIGHT="${HEAD005_FUSION_CROSS_WEIGHT:-0.875}"
HEAD005_MASK_THRESHOLD="${HEAD005_MASK_THRESHOLD:-0.57}"
HEAD005_LOG_FREQ="${HEAD005_LOG_FREQ:-50}"
HEAD005_VALIDATION_FREQ="${HEAD005_VALIDATION_FREQ:-50}"
HEAD005_CHECKPOINT_FREQ="${HEAD005_CHECKPOINT_FREQ:-250}"
HEAD006_RECIPE="${HEAD006_RECIPE:-prototype_temporal5_video_balanced}"
HEAD006_TOTAL_STEPS="${HEAD006_TOTAL_STEPS:-3000}"
HEAD006_LR="${HEAD006_LR:-1e-3}"
HEAD006_TEMPORAL_RADIUS="${HEAD006_TEMPORAL_RADIUS:-2}"
HEAD006_LOG_FREQ="${HEAD006_LOG_FREQ:-50}"
HEAD006_VALIDATION_FREQ="${HEAD006_VALIDATION_FREQ:-50}"
HEAD006_CHECKPOINT_FREQ="${HEAD006_CHECKPOINT_FREQ:-250}"
HEAD010_BASE_RUN="${HEAD010_BASE_RUN:-${WSM_ROOT}/train_log/Channel_Head_000038/20260810_181544_q_20260810-103532}"
HEAD010_BASE_CHECKPOINT="${HEAD010_BASE_CHECKPOINT:-${HEAD010_BASE_RUN}/ckpt/final_step_0001000/pytorch_lora_weights.safetensors}"
HEAD010_TOTAL_STEPS="${HEAD010_TOTAL_STEPS:-3000}"
HEAD010_LR="${HEAD010_LR:-3e-4}"
HEAD010_BATCH_SIZE="${HEAD010_BATCH_SIZE:-8}"
HEAD010_EVAL_BATCH_SIZE="${HEAD010_EVAL_BATCH_SIZE:-16}"
HEAD010_EVAL_FREQ="${HEAD010_EVAL_FREQ:-50}"
HEAD010_MAX_CANDIDATES="${HEAD010_MAX_CANDIDATES:-24}"
# Sparse structure tasks now default to full videos and full frame coverage.
# Audio tasks inherit those full-data defaults unless explicitly overridden.
AUDIO_MAX_VIDEOS="${AUDIO_MAX_VIDEOS:-${MAX_VIDEOS}}"
AUDIO_FRAME_STRIDE="${AUDIO_FRAME_STRIDE:-${FRAME_STRIDE}}"
AUDIO_CONDITION_SOURCE="${AUDIO_CONDITION_SOURCE:-frame}"
AUDIO_MIN_RMS_DBFS="${AUDIO_MIN_RMS_DBFS:--30}"
AUDIO_DENSE_MASK_CONDITIONING_MODE="${AUDIO_DENSE_MASK_CONDITIONING_MODE:-attention}"
AUDIO_DENSE_MASK_GLOBAL_CONDITION="${AUDIO_DENSE_MASK_GLOBAL_CONDITION:-1}"
AUDIO_DENSE_MASK_SEPARATE_CONDITION_ADAPTERS="${AUDIO_DENSE_MASK_SEPARATE_CONDITION_ADAPTERS:-1}"
AUDIO_DENSE_MASK_PREVIEW_NUM_SAMPLES="${AUDIO_DENSE_MASK_PREVIEW_NUM_SAMPLES:-3}"
AUDIO_DENSE_MASK_VALIDATION_FREQ="${AUDIO_DENSE_MASK_VALIDATION_FREQ:-50}"
AUDIO_DENSE_MASK_VAL_LOSS_BATCHES="${AUDIO_DENSE_MASK_VAL_LOSS_BATCHES:-24}"
AUDIO_DENSE_MASK_CHECKPOINT_FREQ="${AUDIO_DENSE_MASK_CHECKPOINT_FREQ:-50}"
AUDIO_DENSE_MASK_VISUALIZATION_FREQ="${AUDIO_DENSE_MASK_VISUALIZATION_FREQ:-50}"
AUDIO_DENSE_MASK_PROTOTYPE_CE_WEIGHT="${AUDIO_DENSE_MASK_PROTOTYPE_CE_WEIGHT:-0.5}"
AUDIO_DENSE_MASK_PROTOTYPE_CE_TEMPERATURE="${AUDIO_DENSE_MASK_PROTOTYPE_CE_TEMPERATURE:-0.25}"
AUDIO_DENSE_MASK_PROTOTYPE_CE_NUM_CLUSTERS="${AUDIO_DENSE_MASK_PROTOTYPE_CE_NUM_CLUSTERS:-8}"
AUDIO_DENSE_MASK_PROTOTYPE_CE_KMEANS_ITERATIONS="${AUDIO_DENSE_MASK_PROTOTYPE_CE_KMEANS_ITERATIONS:-4}"
AUDIO_DENSE_MASK_TRAINABLE_TARGET="${AUDIO_DENSE_MASK_TRAINABLE_TARGET:-auto}"
AUDIO_DENSE_MASK_TRAINABLE_CONDITION="${AUDIO_DENSE_MASK_TRAINABLE_CONDITION:-auto}"
AUDIO_DENSE_MASK_OVERLAY_TARGET="${AUDIO_DENSE_MASK_OVERLAY_TARGET:-1}"
AUDIO_DENSE_MASK_OVERLAY_CACHE_DIR="${AUDIO_DENSE_MASK_OVERLAY_CACHE_DIR:-${SPARSE_DATA_ROOT}/audio_semantic_overlay_flux2_latents}"
AUDIO_DENSE_MASK_OVERLAY_MODALITY="${AUDIO_DENSE_MASK_OVERLAY_MODALITY:-195_overlay_rgb}"
AUDIO_DENSE_MASK_EDIT_LOSS_WEIGHT="${AUDIO_DENSE_MASK_EDIT_LOSS_WEIGHT:-0}"
AUDIO_DENSE_MASK_EDIT_LOSS_POWER="${AUDIO_DENSE_MASK_EDIT_LOSS_POWER:-1}"
AUDIO_CODEC_SECONDS="${AUDIO_CODEC_SECONDS:-10}"
AUDIO_CODEC_BANDWIDTH="${AUDIO_CODEC_BANDWIDTH:-6.0}"
# 007 visual-to-audio trainer. baseline keeps the lightweight EnCodec-code
# regression head; musicgen_lora uses MusicGen-small as an audio prior and
# trains image-prefix conditioning plus parameter-efficient MusicGen adapters.
AUDIO_GENERATION_TRAINER="${AUDIO_GENERATION_TRAINER:-musicgen_lora}"
MUSICGEN_MODEL_NAME="${MUSICGEN_MODEL_NAME:-facebook/musicgen-small}"
MUSICGEN_LR="${MUSICGEN_LR:-1e-4}"
MUSICGEN_TRAIN_MODE="${MUSICGEN_TRAIN_MODE:-full}"
MUSICGEN_CHECKPOINT_FREQ="${MUSICGEN_CHECKPOINT_FREQ:-5000}"
MUSICGEN_BATCH_SIZE="${MUSICGEN_BATCH_SIZE:-2}"
MUSICGEN_BATCH_SIZE_VAL="${MUSICGEN_BATCH_SIZE_VAL:-2}"
MUSICGEN_PREFIX_LENGTH="${MUSICGEN_PREFIX_LENGTH:-16}"
MUSICGEN_LORA_RANK="${MUSICGEN_LORA_RANK:-32}"
MUSICGEN_LORA_ALPHA="${MUSICGEN_LORA_ALPHA:-16}"
MUSICGEN_LORA_DROPOUT="${MUSICGEN_LORA_DROPOUT:-0.05}"
MUSICGEN_AUDIO_SECONDS="${MUSICGEN_AUDIO_SECONDS:-1}"
MUSICGEN_MAX_VAL_BATCHES="${MUSICGEN_MAX_VAL_BATCHES:-8}"
MUSICGEN_EVAL_NEW_TOKENS="${MUSICGEN_EVAL_NEW_TOKENS:-50}"
MUSICGEN_BF16="${MUSICGEN_BF16:-1}"
MAX_KEYPOINTS="${MAX_KEYPOINTS:-160}"
FULL_RGB_IMAGE_SIZE="${FULL_RGB_IMAGE_SIZE:-224}"
MAX_OBJ_POINTCLOUD_POINTS="${MAX_OBJ_POINTCLOUD_POINTS:-1024}"
SMPL_CUDA_VISIBLE_DEVICES="${SMPL_CUDA_VISIBLE_DEVICES:-0}"
BBOX_CUDA_VISIBLE_DEVICES="${BBOX_CUDA_VISIBLE_DEVICES:-1}"
DWPOSE_CUDA_VISIBLE_DEVICES="${DWPOSE_CUDA_VISIBLE_DEVICES:-2}"
MEDIAPIPE_CUDA_VISIBLE_DEVICES="${MEDIAPIPE_CUDA_VISIBLE_DEVICES:-3}"
OBJ_CUDA_VISIBLE_DEVICES="${OBJ_CUDA_VISIBLE_DEVICES:-4}"
AUDIO_CUDA_VISIBLE_DEVICES="${AUDIO_CUDA_VISIBLE_DEVICES:-5}"
# Per-Channel-Head GPU map. The default sparse set is 0,1,2,3,5,6,10.
# Channel_Head_000004 / OBJ, Channel_Head_000007 / visual-to-audio, and the
# audio-mask experiments 000008/000009 remain opt-in.
# Override with e.g.
# CHANNEL_HEAD_GPU_MAP="5:5,6:6,7:7". Keys may be raw indices (5),
# zero-padded indices (000005), or full names (Channel_Head_000005).
CHANNEL_HEAD_GPU_MAP="${CHANNEL_HEAD_GPU_MAP:-0:0,1:1,2:2,3:3,5:4,6:5,10:6}"

FORCE_REBUILD="${FORCE_REBUILD:-0}"
FORCE_RECACHE="${FORCE_RECACHE:-0}"
RESET_RUN="${RESET_RUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
DETACH="${DETACH:-1}"
INSIDE_TMUX="${INSIDE_TMUX:-0}"

CONDA_SH="${CONDA_SH:-/research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-WSM}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" ]]; then
  LOG_DIR="${SPARSE_LOG_ROOT}/queue_logs"
else
  LOG_DIR="${SPARSE_LOG_ROOT}/launch_logs"
fi
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log}"

quote() {
  printf "%q" "$1"
}

if [[ "${DETACH}" == "1" && "${INSIDE_TMUX}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  if env -u LD_LIBRARY_PATH tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}"
    echo "Attach with: env -u LD_LIBRARY_PATH tmux attach -t ${SESSION_NAME}"
    exit 1
  fi

  CMD="cd $(quote "${WSM_ROOT}") && \
INSIDE_TMUX=1 DETACH=0 \
SPARSE_CONFIG_PATH=$(quote "${SPARSE_CONFIG_PATH}") CHANNEL_HEADS=$(quote "${CHANNEL_HEADS}") WSG_CHANNEL_HEAD_RANK=$(quote "${WSG_CHANNEL_HEAD_RANK:-8}") WSG_CHANNEL_HEAD_ALPHA=$(quote "${WSG_CHANNEL_HEAD_ALPHA:-}") \
SPARSE_DATA_ROOT=$(quote "${SPARSE_DATA_ROOT}") SPARSE_CACHE_ROOT=$(quote "${SPARSE_CACHE_ROOT}") SPARSE_LOG_ROOT=$(quote "${SPARSE_LOG_ROOT}") SPARSE_CLEAN_LOG_LAYOUT=$(quote "${SPARSE_CLEAN_LOG_LAYOUT}") SPARSE_DATASET_ROOT=$(quote "${SPARSE_DATASET_ROOT}") \
TASKS=$(quote "${TASKS}") SKIP_UNIMPLEMENTED=$(quote "${SKIP_UNIMPLEMENTED}") PARALLEL_TASKS=$(quote "${PARALLEL_TASKS}") \
RUN_NAME=$(quote "${RUN_NAME}") DATASET_PREFIX=$(quote "${DATASET_PREFIX}") SESSION_NAME=$(quote "${SESSION_NAME}") \
TRAIN_ROOT=$(quote "${TRAIN_ROOT}") VAL_ROOT=$(quote "${VAL_ROOT}") \
OBJ_TRAIN_ROOTS=$(quote "${OBJ_TRAIN_ROOTS}") OBJ_VAL_ROOTS=$(quote "${OBJ_VAL_ROOTS}") \
MAX_VIDEOS=$(quote "${MAX_VIDEOS}") FRAME_STRIDE=$(quote "${FRAME_STRIDE}") MAX_FRAMES_PER_VIDEO=$(quote "${MAX_FRAMES_PER_VIDEO}") \
TOTAL_STEPS=$(quote "${TOTAL_STEPS}") TOTAL_STEPS_BBOX=$(quote "${TOTAL_STEPS_BBOX}") TOTAL_STEPS_POSE=$(quote "${TOTAL_STEPS_POSE}") BBOX_LR=$(quote "${BBOX_LR}") BBOX_LOSS_MODE=$(quote "${BBOX_LOSS_MODE}") POSE_LR=$(quote "${POSE_LR}") POSE_MODE=$(quote "${POSE_MODE}") BATCH_SIZE=$(quote "${BATCH_SIZE}") BATCH_SIZE_VAL=$(quote "${BATCH_SIZE_VAL}") \
TOTAL_STEPS_AUDIO_DENSE_MASK=$(quote "${TOTAL_STEPS_AUDIO_DENSE_MASK}") AUDIO_DENSE_MASK_VALIDATION_FREQ=$(quote "${AUDIO_DENSE_MASK_VALIDATION_FREQ}") AUDIO_DENSE_MASK_VAL_LOSS_BATCHES=$(quote "${AUDIO_DENSE_MASK_VAL_LOSS_BATCHES}") AUDIO_DENSE_MASK_CHECKPOINT_FREQ=$(quote "${AUDIO_DENSE_MASK_CHECKPOINT_FREQ}") AUDIO_DENSE_MASK_VISUALIZATION_FREQ=$(quote "${AUDIO_DENSE_MASK_VISUALIZATION_FREQ}") \
AUDIO_DENSE_MASK_PROTOTYPE_CE_WEIGHT=$(quote "${AUDIO_DENSE_MASK_PROTOTYPE_CE_WEIGHT}") AUDIO_DENSE_MASK_PROTOTYPE_CE_TEMPERATURE=$(quote "${AUDIO_DENSE_MASK_PROTOTYPE_CE_TEMPERATURE}") AUDIO_DENSE_MASK_PROTOTYPE_CE_NUM_CLUSTERS=$(quote "${AUDIO_DENSE_MASK_PROTOTYPE_CE_NUM_CLUSTERS}") AUDIO_DENSE_MASK_PROTOTYPE_CE_KMEANS_ITERATIONS=$(quote "${AUDIO_DENSE_MASK_PROTOTYPE_CE_KMEANS_ITERATIONS}") \
NUM_WORKERS=$(quote "${NUM_WORKERS}") LOG_FREQ=$(quote "${LOG_FREQ}") VALIDATION_FREQ=$(quote "${VALIDATION_FREQ}") CHECKPOINT_FREQ=$(quote "${CHECKPOINT_FREQ}") \
USE_BETAS=$(quote "${USE_BETAS}") MAX_BBOX_SLOTS=$(quote "${MAX_BBOX_SLOTS}") BBOX_MIN_VISIBLE_POINTS=$(quote "${BBOX_MIN_VISIBLE_POINTS}") BBOX_VISIBILITY_THRESHOLD=$(quote "${BBOX_VISIBILITY_THRESHOLD}") BBOX_MAX_BOX_AREA_FRACTION=$(quote "${BBOX_MAX_BOX_AREA_FRACTION}") BBOX_REQUIRE_DWPOSE=$(quote "${BBOX_REQUIRE_DWPOSE}") BBOX_DWPOSE_MIN_POINTS=$(quote "${BBOX_DWPOSE_MIN_POINTS}") BBOX_DWPOSE_SCORE_THRESHOLD=$(quote "${BBOX_DWPOSE_SCORE_THRESHOLD}") BBOX_MARGIN_MODE=$(quote "${BBOX_MARGIN_MODE}") BBOX_MARGIN_PIXELS=$(quote "${BBOX_MARGIN_PIXELS}") BBOX_MARGIN_FRACTION=$(quote "${BBOX_MARGIN_FRACTION}") MAX_OBJ_SLOTS=$(quote "${MAX_OBJ_SLOTS}") OBJ_MODE=$(quote "${OBJ_MODE}") OBJ_LR=$(quote "${OBJ_LR}") AUDIO_LR=$(quote "${AUDIO_LR}") AUDIO_ENCODER=$(quote "${AUDIO_ENCODER}") AUDIO_MODEL_NAME=$(quote "${AUDIO_MODEL_NAME}") AUDIO_GENERATION_ENCODER=$(quote "${AUDIO_GENERATION_ENCODER}") AUDIO_GENERATION_MODEL_NAME=$(quote "${AUDIO_GENERATION_MODEL_NAME}") AUDIO_ENCODER_DEVICE=$(quote "${AUDIO_ENCODER_DEVICE}") AUDIO_GENERATION_ENCODER_DEVICE=$(quote "${AUDIO_GENERATION_ENCODER_DEVICE}") AUDIO_DIM=$(quote "${AUDIO_DIM}") AUDIO_ENCODER_BATCH_SIZE=$(quote "${AUDIO_ENCODER_BATCH_SIZE}") AUDIO_FRAME_SAMPLES_PER_VIDEO=$(quote "${AUDIO_FRAME_SAMPLES_PER_VIDEO}") AUDIO_MAX_VIDEOS=$(quote "${AUDIO_MAX_VIDEOS}") AUDIO_FRAME_STRIDE=$(quote "${AUDIO_FRAME_STRIDE}") AUDIO_CONDITION_SOURCE=$(quote "${AUDIO_CONDITION_SOURCE}") AUDIO_MIN_RMS_DBFS=$(quote "${AUDIO_MIN_RMS_DBFS}") AUDIO_DENSE_MASK_CONDITIONING_MODE=$(quote "${AUDIO_DENSE_MASK_CONDITIONING_MODE}") AUDIO_DENSE_MASK_GLOBAL_CONDITION=$(quote "${AUDIO_DENSE_MASK_GLOBAL_CONDITION}") AUDIO_DENSE_MASK_SEPARATE_CONDITION_ADAPTERS=$(quote "${AUDIO_DENSE_MASK_SEPARATE_CONDITION_ADAPTERS}") AUDIO_DENSE_MASK_PREVIEW_NUM_SAMPLES=$(quote "${AUDIO_DENSE_MASK_PREVIEW_NUM_SAMPLES}") AUDIO_DENSE_MASK_TRAINABLE_TARGET=$(quote "${AUDIO_DENSE_MASK_TRAINABLE_TARGET}") AUDIO_DENSE_MASK_TRAINABLE_CONDITION=$(quote "${AUDIO_DENSE_MASK_TRAINABLE_CONDITION}") AUDIO_DENSE_MASK_OVERLAY_TARGET=$(quote "${AUDIO_DENSE_MASK_OVERLAY_TARGET}") AUDIO_DENSE_MASK_OVERLAY_CACHE_DIR=$(quote "${AUDIO_DENSE_MASK_OVERLAY_CACHE_DIR}") AUDIO_DENSE_MASK_OVERLAY_MODALITY=$(quote "${AUDIO_DENSE_MASK_OVERLAY_MODALITY}") AUDIO_DENSE_MASK_EDIT_LOSS_WEIGHT=$(quote "${AUDIO_DENSE_MASK_EDIT_LOSS_WEIGHT}") AUDIO_DENSE_MASK_EDIT_LOSS_POWER=$(quote "${AUDIO_DENSE_MASK_EDIT_LOSS_POWER}") AUDIO_CODEC_SECONDS=$(quote "${AUDIO_CODEC_SECONDS}") AUDIO_CODEC_BANDWIDTH=$(quote "${AUDIO_CODEC_BANDWIDTH}") AUDIO_GENERATION_TRAINER=$(quote "${AUDIO_GENERATION_TRAINER}") MUSICGEN_MODEL_NAME=$(quote "${MUSICGEN_MODEL_NAME}") MUSICGEN_LR=$(quote "${MUSICGEN_LR}") MUSICGEN_TRAIN_MODE=$(quote "${MUSICGEN_TRAIN_MODE}") MUSICGEN_CHECKPOINT_FREQ=$(quote "${MUSICGEN_CHECKPOINT_FREQ}") MUSICGEN_BATCH_SIZE=$(quote "${MUSICGEN_BATCH_SIZE}") MUSICGEN_BATCH_SIZE_VAL=$(quote "${MUSICGEN_BATCH_SIZE_VAL}") MUSICGEN_PREFIX_LENGTH=$(quote "${MUSICGEN_PREFIX_LENGTH}") MUSICGEN_LORA_RANK=$(quote "${MUSICGEN_LORA_RANK}") MUSICGEN_LORA_ALPHA=$(quote "${MUSICGEN_LORA_ALPHA}") MUSICGEN_LORA_DROPOUT=$(quote "${MUSICGEN_LORA_DROPOUT}") MUSICGEN_AUDIO_SECONDS=$(quote "${MUSICGEN_AUDIO_SECONDS}") MUSICGEN_MAX_VAL_BATCHES=$(quote "${MUSICGEN_MAX_VAL_BATCHES}") MUSICGEN_EVAL_NEW_TOKENS=$(quote "${MUSICGEN_EVAL_NEW_TOKENS}") MUSICGEN_BF16=$(quote "${MUSICGEN_BF16}") MAX_KEYPOINTS=$(quote "${MAX_KEYPOINTS}") FULL_RGB_IMAGE_SIZE=$(quote "${FULL_RGB_IMAGE_SIZE}") MAX_OBJ_POINTCLOUD_POINTS=$(quote "${MAX_OBJ_POINTCLOUD_POINTS}") \
HEAD005_RECIPE=$(quote "${HEAD005_RECIPE}") HEAD005_INIT_CHECKPOINT=$(quote "${HEAD005_INIT_CHECKPOINT}") HEAD005_TOTAL_STEPS=$(quote "${HEAD005_TOTAL_STEPS}") HEAD005_LR=$(quote "${HEAD005_LR}") HEAD005_FUSION_CROSS_WEIGHT=$(quote "${HEAD005_FUSION_CROSS_WEIGHT}") HEAD005_MASK_THRESHOLD=$(quote "${HEAD005_MASK_THRESHOLD}") HEAD005_LOG_FREQ=$(quote "${HEAD005_LOG_FREQ}") HEAD005_VALIDATION_FREQ=$(quote "${HEAD005_VALIDATION_FREQ}") HEAD005_CHECKPOINT_FREQ=$(quote "${HEAD005_CHECKPOINT_FREQ}") \
HEAD006_RECIPE=$(quote "${HEAD006_RECIPE}") HEAD006_TOTAL_STEPS=$(quote "${HEAD006_TOTAL_STEPS}") HEAD006_LR=$(quote "${HEAD006_LR}") HEAD006_TEMPORAL_RADIUS=$(quote "${HEAD006_TEMPORAL_RADIUS}") HEAD006_LOG_FREQ=$(quote "${HEAD006_LOG_FREQ}") HEAD006_VALIDATION_FREQ=$(quote "${HEAD006_VALIDATION_FREQ}") HEAD006_CHECKPOINT_FREQ=$(quote "${HEAD006_CHECKPOINT_FREQ}") \
HEAD010_BASE_RUN=$(quote "${HEAD010_BASE_RUN}") HEAD010_BASE_CHECKPOINT=$(quote "${HEAD010_BASE_CHECKPOINT}") HEAD010_TOTAL_STEPS=$(quote "${HEAD010_TOTAL_STEPS}") HEAD010_LR=$(quote "${HEAD010_LR}") HEAD010_BATCH_SIZE=$(quote "${HEAD010_BATCH_SIZE}") HEAD010_EVAL_BATCH_SIZE=$(quote "${HEAD010_EVAL_BATCH_SIZE}") HEAD010_EVAL_FREQ=$(quote "${HEAD010_EVAL_FREQ}") HEAD010_MAX_CANDIDATES=$(quote "${HEAD010_MAX_CANDIDATES}") \
BOX_SOURCE=$(quote "${BOX_SOURCE}") SMPL_BOX_SOURCE=$(quote "${SMPL_BOX_SOURCE}") \
SMPL_CUDA_VISIBLE_DEVICES=$(quote "${SMPL_CUDA_VISIBLE_DEVICES}") BBOX_CUDA_VISIBLE_DEVICES=$(quote "${BBOX_CUDA_VISIBLE_DEVICES}") \
DWPOSE_CUDA_VISIBLE_DEVICES=$(quote "${DWPOSE_CUDA_VISIBLE_DEVICES}") MEDIAPIPE_CUDA_VISIBLE_DEVICES=$(quote "${MEDIAPIPE_CUDA_VISIBLE_DEVICES}") OBJ_CUDA_VISIBLE_DEVICES=$(quote "${OBJ_CUDA_VISIBLE_DEVICES}") AUDIO_CUDA_VISIBLE_DEVICES=$(quote "${AUDIO_CUDA_VISIBLE_DEVICES}") CHANNEL_HEAD_GPU_MAP=$(quote "${CHANNEL_HEAD_GPU_MAP}") \
FORCE_REBUILD=$(quote "${FORCE_REBUILD}") FORCE_RECACHE=$(quote "${FORCE_RECACHE}") RESET_RUN=$(quote "${RESET_RUN}") DRY_RUN=$(quote "${DRY_RUN}") \
CONDA_SH=$(quote "${CONDA_SH}") CONDA_ENV=$(quote "${CONDA_ENV}") TIMESTAMP=$(quote "${TIMESTAMP}") LOG_FILE=$(quote "${LOG_FILE}") \
bash $(quote "${SCRIPT_PATH}")"

  env -u LD_LIBRARY_PATH tmux new-session -d -s "${SESSION_NAME}" "${CMD}"
  echo "Started tmux session: ${SESSION_NAME}"
  echo "Log file: ${LOG_FILE}"
  echo "Attach with: env -u LD_LIBRARY_PATH tmux attach -t ${SESSION_NAME}"
  exit 0
fi

mkdir -p "${LOG_DIR}" "${SPARSE_DATASET_ROOT}" "${SPARSE_CACHE_ROOT}"
cd "${SCOREHMR_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if ! [[ "${PARALLEL_TASKS}" =~ ^[0-9]+$ ]]; then
  echo "PARALLEL_TASKS must be a non-negative integer, got: ${PARALLEL_TASKS}" >&2
  exit 2
fi
PARALLEL_TASKS_INT="$((10#${PARALLEL_TASKS}))"

resolve_channel_heads() {
  python - "${SPARSE_CONFIG_PATH}" "${CHANNEL_HEADS}" <<'INNERPY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
raw_heads = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
with config_path.open("r") as f:
    graph = json.load(f)["wsg_v1"]

for raw in raw_heads:
    key = raw if raw.startswith("Channel_Head_") else f"Channel_Head_{int(raw):06d}"
    if key not in graph:
        raise SystemExit(f"Missing {key} in {config_path}")
    entry = graph[key]
    print("\t".join([
        key,
        str(entry.get("adapter_task", "")),
        str(entry.get("task_type", "")),
        json.dumps(entry.get("condition", ""), ensure_ascii=False),
        str(entry.get("target", "")),
        str(entry.get("condition_preprocess", "")),
        "1" if bool(entry.get("pair_exists", False)) else "0",
    ]))
INNERPY
}

build_common_args() {
  local train_name="$1"
  local val_name="$2"
  local args=(
    --train-root "${TRAIN_ROOT}"
    --val-root "${VAL_ROOT}"
    --frame-stride "${FRAME_STRIDE}"
    --out-dir "${SPARSE_DATASET_ROOT}"
    --train-name "${train_name}"
    --val-name "${val_name}"
  )
  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi
  if [[ -n "${MAX_FRAMES_PER_VIDEO}" ]]; then
    args+=(--max-frames-per-video "${MAX_FRAMES_PER_VIDEO}")
  fi
  printf '%s\n' "${args[@]}"
}

build_audio_args() {
  local train_name="$1"
  local val_name="$2"
  local args=(
    --train-root "${TRAIN_ROOT}"
    --val-root "${VAL_ROOT}"
    --frame-stride "${AUDIO_FRAME_STRIDE}"
    --out-dir "${SPARSE_DATASET_ROOT}"
    --train-name "${train_name}"
    --val-name "${val_name}"
  )
  if [[ -n "${AUDIO_MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${AUDIO_MAX_VIDEOS}")
  fi
  if [[ -n "${MAX_FRAMES_PER_VIDEO}" ]]; then
    args+=(--max-frames-per-video "${MAX_FRAMES_PER_VIDEO}")
  fi
  printf '%s\n' "${args[@]}"
}

build_obj_args() {
  local train_name="$1"
  local val_name="$2"
  local args=()
  IFS=',' read -r -a OBJ_TRAIN_ARRAY <<< "${OBJ_TRAIN_ROOTS}"
  for ROOT in "${OBJ_TRAIN_ARRAY[@]}"; do
    ROOT="$(echo "${ROOT}" | xargs)"
    [[ -n "${ROOT}" ]] && args+=(--train-root "${ROOT}")
  done
  IFS=',' read -r -a OBJ_VAL_ARRAY <<< "${OBJ_VAL_ROOTS}"
  for ROOT in "${OBJ_VAL_ARRAY[@]}"; do
    ROOT="$(echo "${ROOT}" | xargs)"
    [[ -n "${ROOT}" ]] && args+=(--val-root "${ROOT}")
  done
  args+=(--out-dir "${SPARSE_DATASET_ROOT}" --frame-stride "${FRAME_STRIDE}" --train-name "${train_name}" --val-name "${val_name}")
  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi
  if [[ -n "${MAX_FRAMES_PER_VIDEO}" ]]; then
    args+=(--max-frames-per-video "${MAX_FRAMES_PER_VIDEO}")
  fi
  printf '%s\n' "${args[@]}"
}

maybe_reset_run() {
  local name="$1"
  if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" ]]; then
    return 0
  fi
  if [[ "${RESET_RUN}" == "1" ]]; then
    echo "[sparse-structure] reset outputs for ${name}"
    rm -rf "${SPARSE_LOG_ROOT}/tensorboard/${name}" "${SPARSE_LOG_ROOT}/checkpoints/${name}" "${SPARSE_LOG_ROOT}/visuals/${name}"
  fi
}

prepare_sparse_run_dir() {
  local key="$1"
  local adapter_task="$2"
  local task_type="$3"
  local condition="$4"
  local target="$5"
  local condition_preprocess="$6"
  local pair_exists="$7"
  local run_dir
  if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" ]]; then
    if [[ "${key}" == Channel_Head_* ]]; then
      run_dir="${SPARSE_LOG_ROOT}/${key}/${TIMESTAMP}_${RUN_NAME}"
    else
      run_dir="${SPARSE_LOG_ROOT}/legacy_${key}/${TIMESTAMP}_${RUN_NAME}"
    fi
    if [[ "${RESET_RUN}" == "1" ]]; then
      rm -rf "${run_dir}"
    fi
    mkdir -p "${run_dir}/state"
    cat >"${run_dir}/state/resolved_config.yaml" <<EOF
channel_head: "${key}"
adapter_task: "${adapter_task}"
task_type: "${task_type}"
condition: '${condition}'
target: "${target}"
condition_preprocess: "${condition_preprocess}"
pair_exists: ${pair_exists}
run_name: "${RUN_NAME}"
dataset_prefix: "${DATASET_PREFIX}"
total_steps: ${TOTAL_STEPS}
total_steps_audio_category: ${TOTAL_STEPS_AUDIO_CATEGORY}
total_steps_audio_generation: ${TOTAL_STEPS_AUDIO_GENERATION}
total_steps_audio_dense_mask: ${TOTAL_STEPS_AUDIO_DENSE_MASK}
batch_size: ${BATCH_SIZE}
batch_size_val: ${BATCH_SIZE_VAL}
log_freq: ${LOG_FREQ}
validation_freq: ${VALIDATION_FREQ}
checkpoint_freq: ${CHECKPOINT_FREQ}
audio_dense_mask_validation_freq: ${AUDIO_DENSE_MASK_VALIDATION_FREQ}
audio_dense_mask_val_loss_batches: ${AUDIO_DENSE_MASK_VAL_LOSS_BATCHES}
audio_dense_mask_checkpoint_freq: ${AUDIO_DENSE_MASK_CHECKPOINT_FREQ}
audio_dense_mask_visualization_freq: ${AUDIO_DENSE_MASK_VISUALIZATION_FREQ}
audio_dense_mask_prototype_ce_weight: ${AUDIO_DENSE_MASK_PROTOTYPE_CE_WEIGHT}
audio_dense_mask_prototype_ce_temperature: ${AUDIO_DENSE_MASK_PROTOTYPE_CE_TEMPERATURE}
audio_dense_mask_prototype_ce_num_clusters: ${AUDIO_DENSE_MASK_PROTOTYPE_CE_NUM_CLUSTERS}
audio_dense_mask_prototype_ce_kmeans_iterations: ${AUDIO_DENSE_MASK_PROTOTYPE_CE_KMEANS_ITERATIONS}
head005_recipe: "${HEAD005_RECIPE}"
head005_init_checkpoint: "${HEAD005_INIT_CHECKPOINT}"
head005_total_steps: ${HEAD005_TOTAL_STEPS}
head005_lr: ${HEAD005_LR}
head005_fusion_cross_weight: ${HEAD005_FUSION_CROSS_WEIGHT}
head005_mask_threshold: ${HEAD005_MASK_THRESHOLD}
head005_log_freq: ${HEAD005_LOG_FREQ}
head005_validation_freq: ${HEAD005_VALIDATION_FREQ}
head005_checkpoint_freq: ${HEAD005_CHECKPOINT_FREQ}
head006_recipe: "${HEAD006_RECIPE}"
head006_total_steps: ${HEAD006_TOTAL_STEPS}
head006_lr: ${HEAD006_LR}
head006_temporal_radius: ${HEAD006_TEMPORAL_RADIUS}
head006_log_freq: ${HEAD006_LOG_FREQ}
head006_validation_freq: ${HEAD006_VALIDATION_FREQ}
head006_checkpoint_freq: ${HEAD006_CHECKPOINT_FREQ}
head010_base_checkpoint: "${HEAD010_BASE_CHECKPOINT}"
head010_base_run: "${HEAD010_BASE_RUN}"
head010_total_steps: ${HEAD010_TOTAL_STEPS}
head010_lr: ${HEAD010_LR}
head010_batch_size: ${HEAD010_BATCH_SIZE}
head010_eval_batch_size: ${HEAD010_EVAL_BATCH_SIZE}
head010_eval_freq: ${HEAD010_EVAL_FREQ}
head010_max_candidates: ${HEAD010_MAX_CANDIDATES}
box_source: "${BOX_SOURCE}"
smpl_box_source: "${SMPL_BOX_SOURCE}"
audio_encoder: "${AUDIO_ENCODER}"
audio_generation_encoder: "${AUDIO_GENERATION_ENCODER}"
audio_generation_trainer: "${AUDIO_GENERATION_TRAINER}"
musicgen_model_name: "${MUSICGEN_MODEL_NAME}"
musicgen_train_mode: "${MUSICGEN_TRAIN_MODE}"
musicgen_checkpoint_freq: ${MUSICGEN_CHECKPOINT_FREQ}
musicgen_lora_rank: ${MUSICGEN_LORA_RANK}
musicgen_audio_seconds: ${MUSICGEN_AUDIO_SECONDS}
train_root: "${TRAIN_ROOT}"
val_root: "${VAL_ROOT}"
EOF
  else
    run_dir="${SPARSE_LOG_ROOT}"
  fi
  printf '%s\n' "${run_dir}"
}

clean_layout_args() {
  if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" ]]; then
    printf '%s\n' "--clean-log-layout"
  fi
}

active_output_dir() {
  if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" && -n "${SPARSE_ACTIVE_RUN_DIR:-}" ]]; then
    printf '%s\n' "${SPARSE_ACTIVE_RUN_DIR}"
  else
    printf '%s\n' "${SPARSE_LOG_ROOT}"
  fi
}

run_smpl() {
  local task_run="${RUN_NAME}_cond0_2_crop_to_72_87_90_93"
  local task_prefix="${DATASET_PREFIX}_smpl_${SMPL_BOX_SOURCE}"
  local train_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_val.npz"
  local train_pare="${SPARSE_CACHE_ROOT}/rgb_ch0_2_crop_pare/${task_prefix}_train.npz"
  local val_pare="${SPARSE_CACHE_ROOT}/rgb_ch0_2_crop_pare/${task_prefix}_val.npz"

  echo "[smpl] condition=RGB channels 0-2 crop PARE features"
  echo "[smpl] crop/box source=${SMPL_BOX_SOURCE}"
  echo "[smpl] target=SMPL/SMPLer-X channels 72-87 and 90-93"
  echo "[smpl] dataset=${train_dataset}"
  echo "[smpl] cache=${train_pare}"

  local build_args=()
  mapfile -t build_args < <(build_common_args "$(basename "${train_dataset}")" "$(basename "${val_dataset}")")
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    echo "[smpl] building WSV SMPL dataset"
    python wsm_adapters/build_wsv_smpl_dataset.py "${build_args[@]}" --box-source "${SMPL_BOX_SOURCE}"
  else
    echo "[smpl] skip dataset build; found ${train_dataset} and ${val_dataset}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${train_pare}" ]]; then
    echo "[smpl] caching train RGB ch0-2 crop PARE features"
    python wsm_adapters/cache_wsv_pare_feats.py --dataset "${train_dataset}" --out "${train_pare}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}"
  else
    echo "[smpl] skip train crop cache; found ${train_pare}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${val_pare}" ]]; then
    echo "[smpl] caching val RGB ch0-2 crop PARE features"
    python wsm_adapters/cache_wsv_pare_feats.py --dataset "${val_dataset}" --out "${val_pare}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}"
  else
    echo "[smpl] skip val crop cache; found ${val_pare}"
  fi

  maybe_reset_run "${task_run}"

  local output_dir
  output_dir="$(active_output_dir)"
  local train_args=(--name "${task_run}" --train-dataset "${train_dataset}" --val-dataset "${val_dataset}" --train-pare "${train_pare}" --val-pare "${val_pare}" --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" --batch-size-val "${BATCH_SIZE_VAL}" --num-workers "${NUM_WORKERS}" --log-freq "${LOG_FREQ}" --val-freq "${VALIDATION_FREQ}" --checkpoint-freq "${CHECKPOINT_FREQ}" --output-dir "${output_dir}" --no-use-bbox)
  mapfile -t CLEAN_ARGS < <(clean_layout_args)
  train_args+=("${CLEAN_ARGS[@]}")
  if [[ "${USE_BETAS}" == "1" ]]; then
    train_args+=(--use-betas)
  else
    train_args+=(--no-use-betas)
  fi

  echo "[smpl] start training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  python wsm_adapters/train_wsv_smpl.py "${train_args[@]}"
}

run_bbox() {
  local task_run="${RUN_NAME}_cond0_2_full_to_88_89"
  local task_prefix="${DATASET_PREFIX}_bbox_${BOX_SOURCE}"
  local train_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_val.npz"
  local rgb_cache_dir="${SPARSE_CACHE_ROOT}/rgb_ch0_2_full_${FULL_RGB_IMAGE_SIZE}"
  local train_rgb="${rgb_cache_dir}/${task_prefix}_train.npz"
  local val_rgb="${rgb_cache_dir}/${task_prefix}_val.npz"

  echo "[bbox] condition=full RGB channels 0-2 resized to ${FULL_RGB_IMAGE_SIZE}"
  echo "[bbox] target=multi-person boxes derived from ${BOX_SOURCE}"
  echo "[bbox] pose filter: visibility>=${BBOX_VISIBILITY_THRESHOLD} min_visible_points=${BBOX_MIN_VISIBLE_POINTS} max_area_fraction=${BBOX_MAX_BOX_AREA_FRACTION}"
  echo "[bbox] margin: mode=${BBOX_MARGIN_MODE} pixels=${BBOX_MARGIN_PIXELS} fraction=${BBOX_MARGIN_FRACTION}"
  echo "[bbox] human evidence filter: require_dwpose=${BBOX_REQUIRE_DWPOSE} dwpose_min_points=${BBOX_DWPOSE_MIN_POINTS} dwpose_score_threshold=${BBOX_DWPOSE_SCORE_THRESHOLD:-none}"
  echo "[bbox] dataset=${train_dataset}"
  echo "[bbox] cache=${train_rgb}"

  local build_args=()
  mapfile -t build_args < <(build_common_args "$(basename "${train_dataset}")" "$(basename "${val_dataset}")")
  build_args+=(--max-slots "${MAX_BBOX_SLOTS}" --box-source "${BOX_SOURCE}" --min-visible-points "${BBOX_MIN_VISIBLE_POINTS}" --visibility-threshold "${BBOX_VISIBILITY_THRESHOLD}" --max-box-area-fraction "${BBOX_MAX_BOX_AREA_FRACTION}" --bbox-margin-mode "${BBOX_MARGIN_MODE}" --bbox-margin-pixels "${BBOX_MARGIN_PIXELS}" --bbox-margin-fraction "${BBOX_MARGIN_FRACTION}" --dwpose-min-points "${BBOX_DWPOSE_MIN_POINTS}")
  if [[ "${BBOX_REQUIRE_DWPOSE}" == "1" ]]; then
    build_args+=(--require-dwpose)
  else
    build_args+=(--no-require-dwpose)
  fi
  if [[ -n "${BBOX_DWPOSE_SCORE_THRESHOLD}" ]]; then
    build_args+=(--dwpose-score-threshold "${BBOX_DWPOSE_SCORE_THRESHOLD}")
  fi
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    echo "[bbox] building WSV bbox dataset"
    python wsm_adapters/build_wsv_bbox_dataset.py "${build_args[@]}"
  else
    echo "[bbox] skip dataset build; found ${train_dataset} and ${val_dataset}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${train_rgb}" ]]; then
    echo "[bbox] caching train full RGB ch0-2 tensors"
    python wsm_adapters/cache_wsv_rgb_ch0_2_full.py --dataset "${train_dataset}" --out "${train_rgb}" --image-size "${FULL_RGB_IMAGE_SIZE}"
  else
    echo "[bbox] skip train full RGB cache; found ${train_rgb}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${val_rgb}" ]]; then
    echo "[bbox] caching val full RGB ch0-2 tensors"
    python wsm_adapters/cache_wsv_rgb_ch0_2_full.py --dataset "${val_dataset}" --out "${val_rgb}" --image-size "${FULL_RGB_IMAGE_SIZE}"
  else
    echo "[bbox] skip val full RGB cache; found ${val_rgb}"
  fi

  maybe_reset_run "${task_run}"

  echo "[bbox] start training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  local output_dir
  output_dir="$(active_output_dir)"
  mapfile -t CLEAN_ARGS < <(clean_layout_args)
  python wsm_adapters/train_wsv_bbox.py --name "${task_run}" --train-dataset "${train_dataset}" --val-dataset "${val_dataset}" --train-rgb-cache "${train_rgb}" --val-rgb-cache "${val_rgb}" --total-steps "${TOTAL_STEPS_BBOX}" --batch-size "${BATCH_SIZE}" --batch-size-val "${BATCH_SIZE_VAL}" --num-workers "${NUM_WORKERS}" --log-freq "${LOG_FREQ}" --val-freq "${VALIDATION_FREQ}" --checkpoint-freq "${CHECKPOINT_FREQ}" --output-dir "${output_dir}" --lr "${BBOX_LR}" --loss-mode "${BBOX_LOSS_MODE}" "${CLEAN_ARGS[@]}"
}

run_crop_structure() {
  local task="$1"
  local target_label="$2"
  local task_run="${RUN_NAME}_cond0_2_crop_to_${target_label}"
  local task_prefix="${DATASET_PREFIX}_${task}_crop"
  local train_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_val.npz"
  local train_pare="${SPARSE_CACHE_ROOT}/rgb_ch0_2_crop_pare/${task_prefix}_train.npz"
  local val_pare="${SPARSE_CACHE_ROOT}/rgb_ch0_2_crop_pare/${task_prefix}_val.npz"

  echo "[${task}] condition=RGB channels 0-2 crop PARE features"
  echo "[${task}] target=${target_label}"
  echo "[${task}] dataset=${train_dataset}"
  echo "[${task}] cache=${train_pare}"

  local build_args=()
  mapfile -t build_args < <(build_common_args "$(basename "${train_dataset}")" "$(basename "${val_dataset}")")
  build_args+=(--task "${task}" --max-keypoints "${MAX_KEYPOINTS}")
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    echo "[${task}] building WSV structure dataset"
    python wsm_adapters/build_wsv_structure_dataset.py "${build_args[@]}"
  else
    echo "[${task}] skip dataset build; found ${train_dataset} and ${val_dataset}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${train_pare}" ]]; then
    echo "[${task}] caching train RGB ch0-2 crop PARE features"
    python wsm_adapters/cache_wsv_pare_feats.py --dataset "${train_dataset}" --out "${train_pare}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}"
  else
    echo "[${task}] skip train crop cache; found ${train_pare}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${val_pare}" ]]; then
    echo "[${task}] caching val RGB ch0-2 crop PARE features"
    python wsm_adapters/cache_wsv_pare_feats.py --dataset "${val_dataset}" --out "${val_pare}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}"
  else
    echo "[${task}] skip val crop cache; found ${val_pare}"
  fi

  maybe_reset_run "${task_run}"

  echo "[${task}] start training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  local output_dir
  output_dir="$(active_output_dir)"
  mapfile -t CLEAN_ARGS < <(clean_layout_args)
  python wsm_adapters/train_wsv_structure.py --name "${task_run}" --task "${task}" --condition crop_pare --train-dataset "${train_dataset}" --val-dataset "${val_dataset}" --train-cache "${train_pare}" --val-cache "${val_pare}" --total-steps "${TOTAL_STEPS_POSE}" --batch-size "${BATCH_SIZE}" --batch-size-val "${BATCH_SIZE_VAL}" --num-workers "${NUM_WORKERS}" --log-freq "${LOG_FREQ}" --val-freq "${VALIDATION_FREQ}" --checkpoint-freq "${CHECKPOINT_FREQ}" --output-dir "${output_dir}" --lr "${POSE_LR}" --pose-mode "${POSE_MODE}" "${CLEAN_ARGS[@]}"
}

run_obj() {
  local task="obj"
  local target_label="bfs_delta_tree"
  local task_run="${RUN_NAME}_cond_obj_pointcloud_to_obj_${target_label}"
  if [[ "${OBJ_MODE}" != "dense" ]]; then
    task_run="${task_run}_${OBJ_MODE}"
  fi
  local task_prefix="${DATASET_PREFIX}_${task}_crop_${target_label}"
  local train_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_val.npz"
  local pointcloud_cache_dir="${SPARSE_CACHE_ROOT}/obj_pointcloud_norm_${MAX_OBJ_POINTCLOUD_POINTS}"
  local train_pointcloud="${pointcloud_cache_dir}/${task_prefix}_train.npz"
  local val_pointcloud="${pointcloud_cache_dir}/${task_prefix}_val.npz"

  echo "[obj] condition=object/group normalized vertex point cloud (${MAX_OBJ_POINTCLOUD_POINTS} points)"
  echo "[obj] target=BFS skeleton tree slots: parent slot, delta xyz, absolute xyz; built from WSG class=3 joints and class=4 bones"
  echo "[obj] obj_mode=${OBJ_MODE}"
  echo "[obj] dataset=${train_dataset}"
  echo "[obj] cache=${train_pointcloud}"

  local build_args=()
  mapfile -t build_args < <(build_obj_args "$(basename "${train_dataset}")" "$(basename "${val_dataset}")")
  build_args+=(--task obj --max-slots "${MAX_OBJ_SLOTS}")
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    echo "[obj] building WSV structure dataset"
    python wsm_adapters/build_wsv_structure_dataset.py "${build_args[@]}"
  else
    echo "[obj] skip dataset build; found ${train_dataset} and ${val_dataset}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${train_pointcloud}" ]]; then
    echo "[obj] caching train normalized OBJ point clouds"
    python wsm_adapters/cache_wsg_obj_pointcloud.py --dataset "${train_dataset}" --out "${train_pointcloud}" --max-points "${MAX_OBJ_POINTCLOUD_POINTS}"
  else
    echo "[obj] skip train pointcloud cache; found ${train_pointcloud}"
  fi

  if [[ "${FORCE_RECACHE}" == "1" || ! -f "${val_pointcloud}" ]]; then
    echo "[obj] caching val normalized OBJ point clouds"
    python wsm_adapters/cache_wsg_obj_pointcloud.py --dataset "${val_dataset}" --out "${val_pointcloud}" --max-points "${MAX_OBJ_POINTCLOUD_POINTS}"
  else
    echo "[obj] skip val pointcloud cache; found ${val_pointcloud}"
  fi

  maybe_reset_run "${task_run}"

  echo "[obj] start training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  local output_dir
  output_dir="$(active_output_dir)"
  mapfile -t CLEAN_ARGS < <(clean_layout_args)
  python wsm_adapters/train_wsv_structure.py --name "${task_run}" --task obj --condition pointcloud --obj-mode "${OBJ_MODE}" --train-dataset "${train_dataset}" --val-dataset "${val_dataset}" --train-cache "${train_pointcloud}" --val-cache "${val_pointcloud}" --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" --batch-size-val "${BATCH_SIZE_VAL}" --num-workers "${NUM_WORKERS}" --log-freq "${LOG_FREQ}" --val-freq "${VALIDATION_FREQ}" --checkpoint-freq "${CHECKPOINT_FREQ}" --output-dir "${output_dir}" --lr "${OBJ_LR}" "${CLEAN_ARGS[@]}"
}



run_audio_dense_mask_adapter() {
  local head="$1"
  local training_recipe="${2:-overlay_baseline}"
  local overlay_target="${AUDIO_DENSE_MASK_OVERLAY_TARGET}"
  local categorical_loss_mode="off"
  local categorical_loss_weight="0.5"
  local task_run="${RUN_NAME}_${head}_scene_audio_to_semantic_mask_dense_${training_recipe}"
  case "${training_recipe}" in
    overlay_baseline)
      ;;
    rgb_mask_prototype_ce)
      # Head 009 follows dense RGB-to-195 estimation: predict the original 195
      # mask, while borrowing only the Head 008 single-token audio wrapper.
      overlay_target=0
      categorical_loss_mode="prototype_ce"
      categorical_loss_weight="${AUDIO_DENSE_MASK_PROTOTYPE_CE_WEIGHT}"
      ;;
    *)
      echo "[audio-dense-mask:${head}] unknown training_recipe=${training_recipe}" >&2
      exit 2
      ;;
  esac
  local audio_source="frame"
  local selected_audio_encoder="${AUDIO_ENCODER}"
  local selected_audio_model_name="${AUDIO_MODEL_NAME}"
  local selected_audio_device="${AUDIO_ENCODER_DEVICE}"
  local frame_samples_per_video="${AUDIO_FRAME_SAMPLES_PER_VIDEO}"
  local frame_samples_label="${frame_samples_per_video}"
  if [[ -z "${frame_samples_per_video}" || "${frame_samples_per_video}" == "all" || "${frame_samples_per_video}" == "none" || "${frame_samples_per_video}" == "0" ]]; then
    frame_samples_per_video=""
    frame_samples_label="all"
  fi
  local task_prefix="${DATASET_PREFIX}_scene_audio_175_184_sem195_${audio_source}_video${frame_samples_label}_${selected_audio_encoder}"
  local train_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_val.npz"

  echo "[audio-dense-mask:${head}] recipe=${training_recipe} condition=current RGB + single-token AST frame audio"
  echo "[audio-dense-mask:${head}] target=dense channel 195 latent via TunnelingNet categorical_loss=${categorical_loss_mode} weight=${categorical_loss_weight}"
  echo "[audio-dense-mask:${head}] dataset=${train_dataset}"
  echo "[audio-dense-mask:${head}] audio_encoder=${selected_audio_encoder} model=${selected_audio_model_name} device=${selected_audio_device}"
  echo "[audio-dense-mask:${head}] audio_build max_videos=${AUDIO_MAX_VIDEOS:-all} frame_stride=${AUDIO_FRAME_STRIDE}"

  local build_args=()
  mapfile -t build_args < <(build_audio_args "$(basename "${train_dataset}")" "$(basename "${val_dataset}")")
  build_args+=(--audio-encoder "${selected_audio_encoder}" --audio-model-name "${selected_audio_model_name}" --audio-dim "${AUDIO_DIM}" --audio-encoder-batch-size "${AUDIO_ENCODER_BATCH_SIZE}" --audio-encoder-device "${selected_audio_device}" --audio-source "${audio_source}" --audio-codec-seconds "${AUDIO_CODEC_SECONDS}" --audio-codec-bandwidth "${AUDIO_CODEC_BANDWIDTH}")
  if [[ -n "${frame_samples_per_video}" ]]; then
    build_args+=(--frame-samples-per-video "${frame_samples_per_video}")
  fi

  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    local lock_dir="${train_dataset}.build_lock"
    if mkdir "${lock_dir}" 2>/dev/null; then
      trap 'rm -rf "${lock_dir}"' RETURN
      if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
        echo "[audio-dense-mask:${head}] building WSV frame-audio AST dataset"
        python wsm_adapters/build_wsv_audio_dataset.py "${build_args[@]}"
      else
        echo "[audio-dense-mask:${head}] skip dataset build after lock; found ${train_dataset} and ${val_dataset}"
      fi
      rm -rf "${lock_dir}"
      trap - RETURN
    else
      echo "[audio-dense-mask:${head}] waiting for dataset build lock: ${lock_dir}"
      while [[ -d "${lock_dir}" ]]; do
        sleep 10
      done
      if [[ ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
        echo "[audio-dense-mask:${head}] dataset lock released but dataset is missing: ${train_dataset}" >&2
        exit 2
      fi
      echo "[audio-dense-mask:${head}] dataset available after wait"
    fi
  else
    echo "[audio-dense-mask:${head}] skip dataset build; found ${train_dataset} and ${val_dataset}"
  fi

  maybe_reset_run "${task_run}"
  local output_dir
  output_dir="$(active_output_dir)"
  local audio_condition_dim
  audio_condition_dim="$(/research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python - "${train_dataset}" <<'PYDIM'
import numpy as np
import sys
with np.load(sys.argv[1], allow_pickle=False) as data:
    print(int(data['audio_feat'].shape[1]))
PYDIM
)"
  echo "[audio-dense-mask:${head}] audio_condition_dim=${audio_condition_dim}"
  echo "[audio-dense-mask:${head}] audio_conditioning_mode=${AUDIO_DENSE_MASK_CONDITIONING_MODE}; separate_condition_adapters=${AUDIO_DENSE_MASK_SEPARATE_CONDITION_ADAPTERS} audio_global_condition=${AUDIO_DENSE_MASK_GLOBAL_CONDITION} preview_num_samples=${AUDIO_DENSE_MASK_PREVIEW_NUM_SAMPLES} trainable_target=${AUDIO_DENSE_MASK_TRAINABLE_TARGET} trainable_condition=${AUDIO_DENSE_MASK_TRAINABLE_CONDITION}"
  echo "[audio-dense-mask:${head}] max_steps=${TOTAL_STEPS_AUDIO_DENSE_MASK} validation_freq=${AUDIO_DENSE_MASK_VALIDATION_FREQ} val_loss_batches=${AUDIO_DENSE_MASK_VAL_LOSS_BATCHES} checkpoint_freq=${AUDIO_DENSE_MASK_CHECKPOINT_FREQ} visualization_freq=${AUDIO_DENSE_MASK_VISUALIZATION_FREQ}"
  echo "[audio-dense-mask:${head}] overlay_target=${overlay_target} overlay_cache=${AUDIO_DENSE_MASK_OVERLAY_CACHE_DIR} overlay_modality=${AUDIO_DENSE_MASK_OVERLAY_MODALITY}"
  echo "[audio-dense-mask:${head}] edit_loss_weight=${AUDIO_DENSE_MASK_EDIT_LOSS_WEIGHT} edit_loss_power=${AUDIO_DENSE_MASK_EDIT_LOSS_POWER}"
  local dense_target_cache_dir=""
  local dense_target_modality=""
  local target_visual_modality=""
  if [[ "${overlay_target}" == "1" ]]; then
    dense_target_cache_dir="${AUDIO_DENSE_MASK_OVERLAY_CACHE_DIR}"
    dense_target_modality="${AUDIO_DENSE_MASK_OVERLAY_MODALITY}"
    target_visual_modality="0-2"
    local overlay_complete=1
    local overlay_shard
    for overlay_shard in $(seq 0 7); do
      if [[ ! -f "${dense_target_cache_dir}/shards/${dense_target_modality}/shard_$(printf %05d "${overlay_shard}")_of_00008.pt" ]]; then
        overlay_complete=0
        break
      fi
    done
    if [[ "${overlay_complete}" != "1" || "${FORCE_REBUILD}" == "1" ]]; then
      echo "[audio-dense-mask:${head}] building opaque RGB+semantic-mask overlay target cache"
      /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python         "${WSM_ROOT}/manual_processing/flux2_preprocessing/cache/build_audio_semantic_overlay_flux2_latents.py"         --model-path "${FLUX2_MODEL_PATH:-${WSM_ROOT}/models/black-forest-labs/FLUX.2-klein-4B}"         --source-cache-dir "${WSM_ROOT}/data/wsg_flux2_frame_latents_dense_multidataset"         --output-cache-dir "${dense_target_cache_dir}"         --output-modality "${dense_target_modality}"         --audio-npz "${train_dataset}"         --audio-npz "${val_dataset}"         --device "cuda"
    else
      echo "[audio-dense-mask:${head}] skip overlay target cache build; found all 8 shards under ${dense_target_cache_dir}/shards/${dense_target_modality}"
    fi
  fi
  echo "[audio-dense-mask:${head}] trainable_condition=${AUDIO_DENSE_MASK_TRAINABLE_CONDITION}"
  echo "[audio-dense-mask:${head}] start dense TunnelingNet training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

  (
    cd "${WSM_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}" \
    NUM_PROCESSES=1 \
    RUN_IN_TMUX=0 \
    WSG_CHANNEL_HEAD_INDEX="$((10#${head#Channel_Head_}))" \
    WSG_CHANNEL_HEAD_RANK="${WSG_CHANNEL_HEAD_RANK:-8}" \
    WSG_CHANNEL_HEAD_ALPHA="${WSG_CHANNEL_HEAD_ALPHA:-}" \
    MODALITY_CONFIG="${SPARSE_CONFIG_PATH}" \
    WSG_AUDIO_DENSE_CONDITION_TRAIN="${train_dataset}" \
    WSG_AUDIO_DENSE_CONDITION_VAL="${val_dataset}" \
    WSG_AUDIO_CONDITION_DIM="${audio_condition_dim}" \
    WSG_AUDIO_DENSE_TARGET_CACHE_DIR="${dense_target_cache_dir}" \
    WSG_AUDIO_DENSE_TARGET_MODALITY="${dense_target_modality}" \
    WSG_TARGET_VISUAL_MODALITY="${target_visual_modality}" \
    WSG_AUDIO_CONDITIONING_MODE="${AUDIO_DENSE_MASK_CONDITIONING_MODE}" \
    WSG_SEPARATE_CONDITION_ADAPTERS="${AUDIO_DENSE_MASK_SEPARATE_CONDITION_ADAPTERS}" \
    WSG_AUDIO_GLOBAL_CONDITION="${AUDIO_DENSE_MASK_GLOBAL_CONDITION}" \
    WSG_PREVIEW_NUM_SAMPLES="${AUDIO_DENSE_MASK_PREVIEW_NUM_SAMPLES}" \
    WSG_TRAINABLE_TARGET="${AUDIO_DENSE_MASK_TRAINABLE_TARGET}" \
    WSG_TRAINABLE_CONDITION="${AUDIO_DENSE_MASK_TRAINABLE_CONDITION}" \
    WSG_EDIT_LOSS_WEIGHT="${AUDIO_DENSE_MASK_EDIT_LOSS_WEIGHT}" \
    WSG_EDIT_LOSS_POWER="${AUDIO_DENSE_MASK_EDIT_LOSS_POWER}" \
    WSG_CATEGORICAL_LOSS_MODE="${categorical_loss_mode}" \
    WSG_CATEGORICAL_LOSS_WEIGHT="${categorical_loss_weight}" \
    WSG_CATEGORICAL_TEMPERATURE="${AUDIO_DENSE_MASK_PROTOTYPE_CE_TEMPERATURE}" \
    WSG_CATEGORICAL_NUM_CLUSTERS="${AUDIO_DENSE_MASK_PROTOTYPE_CE_NUM_CLUSTERS}" \
    WSG_CATEGORICAL_KMEANS_ITERATIONS="${AUDIO_DENSE_MASK_PROTOTYPE_CE_KMEANS_ITERATIONS}" \
    WSG_USE_XPOSE_CACHE=0 \
    WSG_TARGET_DATASET_WHITELIST="AVIS_train_results,AVIS_val_results,AVIS_test_results" \
    MAX_STEPS="${TOTAL_STEPS_AUDIO_DENSE_MASK}" \
    EVAL_EVERY_STEPS="${AUDIO_DENSE_MASK_VALIDATION_FREQ}" \
    VAL_LOSS_BATCHES="${AUDIO_DENSE_MASK_VAL_LOSS_BATCHES}" \
    SAVE_EVERY_STEPS="${AUDIO_DENSE_MASK_CHECKPOINT_FREQ}" \
    VIZ_EVERY_STEPS="${AUDIO_DENSE_MASK_VISUALIZATION_FREQ}" \
    MAX_LOCAL_SAMPLES=0 \
    TRAIN_BATCH_SIZE=1 \
    RUN_ROOT="${output_dir}" \
    LAUNCH_LOG="${output_dir}/dense_launch.log" \
    RUN_ID="$(basename "${output_dir}")" \
    bash scripts/launch_scripts/train_dense_structure_models.sh
  )
}

run_audio_adapter() {
  local audio_task="$1"
  local task_label="$2"
  local task_run="${RUN_NAME}_${task_label}"
  local audio_source="${AUDIO_CONDITION_SOURCE}"
  local selected_audio_encoder="${AUDIO_ENCODER}"
  local selected_audio_model_name="${AUDIO_MODEL_NAME}"
  local selected_audio_device="${AUDIO_ENCODER_DEVICE}"
  local frame_samples_per_video="${AUDIO_FRAME_SAMPLES_PER_VIDEO}"
  local frame_samples_label="${frame_samples_per_video}"
  if [[ -z "${frame_samples_per_video}" || "${frame_samples_per_video}" == "all" || "${frame_samples_per_video}" == "none" || "${frame_samples_per_video}" == "0" ]]; then
    frame_samples_per_video=""
    frame_samples_label="all"
  fi
  local task_prefix="${DATASET_PREFIX}_scene_audio_175_184_sem195_${audio_source}_video${frame_samples_label}_${selected_audio_encoder}"
  if [[ "${audio_source}" == "full" ]]; then
    frame_samples_per_video=""
    task_prefix="${DATASET_PREFIX}_full_condition_audio_sem195_${selected_audio_encoder}"
  elif [[ "${audio_source}" != "frame" ]]; then
    echo "[audio:${audio_task}] unsupported AUDIO_CONDITION_SOURCE=${audio_source}; expected frame or full" >&2
    exit 2
  fi
  if [[ "${audio_task}" == "visual_to_audio" ]]; then
    if [[ "${AUDIO_GENERATION_TRAINER}" == "musicgen_lora" ]]; then
      # Root fix for 007: condition crop and target audio must be frame-aligned.
      # Use WSG audio bytes from channels 175-184 instead of unrelated full WAV.
      audio_source="frame"
      selected_audio_encoder="musicgen"
      selected_audio_model_name="${MUSICGEN_MODEL_NAME}"
      selected_audio_device="${AUDIO_GENERATION_ENCODER_DEVICE}"
      task_prefix="${DATASET_PREFIX}_scene_audio_175_184_sem195_frame_video${frame_samples_label}_musicgen_codes_${MUSICGEN_AUDIO_SECONDS}s"
    elif [[ "${AUDIO_GENERATION_TRAINER}" == "baseline" ]]; then
      audio_source="full"
      frame_samples_per_video=""
      selected_audio_encoder="${AUDIO_GENERATION_ENCODER}"
      selected_audio_model_name="${AUDIO_GENERATION_MODEL_NAME}"
      selected_audio_device="${AUDIO_GENERATION_ENCODER_DEVICE}"
      task_prefix="${DATASET_PREFIX}_full_audio_sem195_${selected_audio_encoder}_${AUDIO_CODEC_SECONDS}s_bw${AUDIO_CODEC_BANDWIDTH}"
    else
      echo "[audio:${audio_task}] unsupported AUDIO_GENERATION_TRAINER=${AUDIO_GENERATION_TRAINER}; expected musicgen_lora or baseline" >&2
      exit 2
    fi
  fi
  local train_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${task_prefix}_val.npz"

  echo "[audio:${audio_task}] source=${audio_source}; frame source uses WSG channels 175-184, full source uses WAVAudios/*.wav"
  echo "[audio:${audio_task}] semantic supervision comes from channel 195"
  echo "[audio:${audio_task}] dataset=${train_dataset}"
  echo "[audio:${audio_task}] audio_encoder=${selected_audio_encoder} model=${selected_audio_model_name} device=${selected_audio_device}"
  echo "[audio:${audio_task}] audio_dim=${AUDIO_DIM} image_size=${FULL_RGB_IMAGE_SIZE}"
  echo "[audio:${audio_task}] min_audio_rms_dbfs=${AUDIO_MIN_RMS_DBFS}"
  echo "[audio:${audio_task}] audio_build max_videos=${AUDIO_MAX_VIDEOS:-all} frame_stride=${AUDIO_FRAME_STRIDE}"
  if [[ -n "${frame_samples_per_video}" ]]; then
    echo "[audio:${audio_task}] random frame/audio samples per video=${frame_samples_per_video}"
  elif [[ "${audio_source}" == "frame" ]]; then
    echo "[audio:${audio_task}] using all usable frame/audio samples per video"
  fi

  local build_args=()
  mapfile -t build_args < <(build_audio_args "$(basename "${train_dataset}")" "$(basename "${val_dataset}")")
  local selected_codec_seconds="${AUDIO_CODEC_SECONDS}"
  if [[ "${audio_task}" == "visual_to_audio" && "${AUDIO_GENERATION_TRAINER}" == "musicgen_lora" ]]; then
    selected_codec_seconds="${MUSICGEN_AUDIO_SECONDS}"
  fi
  build_args+=(--audio-encoder "${selected_audio_encoder}" --audio-model-name "${selected_audio_model_name}" --audio-dim "${AUDIO_DIM}" --audio-encoder-batch-size "${AUDIO_ENCODER_BATCH_SIZE}" --audio-encoder-device "${selected_audio_device}" --audio-source "${audio_source}" --audio-codec-seconds "${selected_codec_seconds}" --audio-codec-bandwidth "${AUDIO_CODEC_BANDWIDTH}")
  if [[ -n "${frame_samples_per_video}" ]]; then
    build_args+=(--frame-samples-per-video "${frame_samples_per_video}")
  fi

  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    local lock_dir="${train_dataset}.build_lock"
    if mkdir "${lock_dir}" 2>/dev/null; then
      trap 'rm -rf "${lock_dir}"' RETURN
      if [[ "${FORCE_REBUILD}" == "1" || ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
        echo "[audio:${audio_task}] building WSV ${audio_source}-audio dataset"
        python wsm_adapters/build_wsv_audio_dataset.py "${build_args[@]}"
      else
        echo "[audio:${audio_task}] skip dataset build after lock; found ${train_dataset} and ${val_dataset}"
      fi
      rm -rf "${lock_dir}"
      trap - RETURN
    else
      echo "[audio:${audio_task}] waiting for dataset build lock: ${lock_dir}"
      while [[ -d "${lock_dir}" ]]; do
        sleep 10
      done
      if [[ ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
        echo "[audio:${audio_task}] dataset lock released but dataset is missing: ${train_dataset}" >&2
        exit 2
      fi
      echo "[audio:${audio_task}] dataset available after wait"
    fi
  else
    echo "[audio:${audio_task}] skip dataset build; found ${train_dataset} and ${val_dataset}"
  fi

  maybe_reset_run "${task_run}"

  echo "[audio:${audio_task}] start training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  local output_dir
  output_dir="$(active_output_dir)"
  mapfile -t CLEAN_ARGS < <(clean_layout_args)
  if [[ "${audio_task}" == "visual_to_audio" && "${AUDIO_GENERATION_TRAINER}" == "musicgen_lora" ]]; then
    local musicgen_args=()
    if [[ "${MUSICGEN_BF16}" == "1" ]]; then
      musicgen_args+=(--bf16)
    fi
    python wsm_adapters/train_wsv_musicgen_lora.py --name "${task_run}" --train-dataset "${train_dataset}" --val-dataset "${val_dataset}" --model-name "${MUSICGEN_MODEL_NAME}" --total-steps "${TOTAL_STEPS_AUDIO_GENERATION}" --batch-size "${MUSICGEN_BATCH_SIZE}" --batch-size-val "${MUSICGEN_BATCH_SIZE_VAL}" --num-workers "${NUM_WORKERS}" --log-freq "${LOG_FREQ}" --checkpoint-freq "${MUSICGEN_CHECKPOINT_FREQ}" --output-dir "${output_dir}" --lr "${MUSICGEN_LR}" --train-mode "${MUSICGEN_TRAIN_MODE}" --image-size "${FULL_RGB_IMAGE_SIZE}" --audio-seconds "${MUSICGEN_AUDIO_SECONDS}" --min-audio-rms-dbfs "${AUDIO_MIN_RMS_DBFS}" --prefix-length "${MUSICGEN_PREFIX_LENGTH}" --lora-rank "${MUSICGEN_LORA_RANK}" --lora-alpha "${MUSICGEN_LORA_ALPHA}" --lora-dropout "${MUSICGEN_LORA_DROPOUT}" --max-val-batches "${MUSICGEN_MAX_VAL_BATCHES}" --eval-new-tokens "${MUSICGEN_EVAL_NEW_TOKENS}" "${musicgen_args[@]}" "${CLEAN_ARGS[@]}"
  else
    local audio_total_steps="${TOTAL_STEPS}"
    local audio_lr="${AUDIO_LR}"
    local audio_log_freq="${LOG_FREQ}"
    local audio_validation_freq="${VALIDATION_FREQ}"
    local audio_checkpoint_freq="${CHECKPOINT_FREQ}"
    local audio_recipe_args=()
    if [[ "${audio_task}" == "localization" ]]; then
      case "${HEAD005_RECIPE}" in
        clip_frozen_mask_fused)
          if [[ ! -f "${HEAD005_INIT_CHECKPOINT}" ]]; then
            echo "[audio:localization] Head005 default checkpoint is missing: ${HEAD005_INIT_CHECKPOINT}" >&2
            echo "Set HEAD005_INIT_CHECKPOINT to the self-contained fused model-best.pt." >&2
            exit 2
          fi
          audio_total_steps="${HEAD005_TOTAL_STEPS}"
          audio_lr="${HEAD005_LR}"
          audio_log_freq="${HEAD005_LOG_FREQ}"
          audio_validation_freq="${HEAD005_VALIDATION_FREQ}"
          audio_checkpoint_freq="${HEAD005_CHECKPOINT_FREQ}"
          audio_recipe_args+=(
            --localization-mode clip_resnet_joint_mask
            --adapter-init-checkpoint "${HEAD005_INIT_CHECKPOINT}"
            --fusion-cross-weight "${HEAD005_FUSION_CROSS_WEIGHT}"
            --mask-threshold "${HEAD005_MASK_THRESHOLD}"
            --freeze-clip-base
            --validate-before-training
          )
          echo "[audio:localization] Head005 recipe=${HEAD005_RECIPE} init=${HEAD005_INIT_CHECKPOINT} fusion_cross_weight=${HEAD005_FUSION_CROSS_WEIGHT} mask_threshold=${HEAD005_MASK_THRESHOLD}"
          ;;
        legacy)
          echo "[audio:localization] Head005 recipe=legacy"
          ;;
        *)
          echo "[audio:localization] unsupported HEAD005_RECIPE=${HEAD005_RECIPE}" >&2
          exit 2
          ;;
      esac
    elif [[ "${audio_task}" == "category" ]]; then
      case "${HEAD006_RECIPE}" in
        prototype_temporal5_video_balanced)
          audio_total_steps="${HEAD006_TOTAL_STEPS}"
          audio_lr="${HEAD006_LR}"
          audio_log_freq="${HEAD006_LOG_FREQ}"
          audio_validation_freq="${HEAD006_VALIDATION_FREQ}"
          audio_checkpoint_freq="${HEAD006_CHECKPOINT_FREQ}"
          audio_recipe_args+=(
            --category-mode prototype_balanced
            --category-balanced-sampling
            --video-balanced-sampling
            --temporal-radius "${HEAD006_TEMPORAL_RADIUS}"
          )
          echo "[audio:category] Head006 recipe=${HEAD006_RECIPE} temporal_window=$((2 * HEAD006_TEMPORAL_RADIUS + 1)) category_balanced=1 video_balanced=1"
          ;;
        legacy)
          audio_total_steps="${TOTAL_STEPS_AUDIO_CATEGORY}"
          echo "[audio:category] Head006 recipe=legacy"
          ;;
        *)
          echo "[audio:category] unsupported HEAD006_RECIPE=${HEAD006_RECIPE}" >&2
          exit 2
          ;;
      esac
    fi
    python wsm_adapters/train_wsv_audio.py --name "${task_run}" --task "${audio_task}" --train-dataset "${train_dataset}" --val-dataset "${val_dataset}" --total-steps "${audio_total_steps}" --batch-size "${BATCH_SIZE}" --batch-size-val "${BATCH_SIZE_VAL}" --num-workers "${NUM_WORKERS}" --log-freq "${audio_log_freq}" --val-freq "${audio_validation_freq}" --checkpoint-freq "${audio_checkpoint_freq}" --output-dir "${output_dir}" --lr "${audio_lr}" --image-size "${FULL_RGB_IMAGE_SIZE}" --min-audio-rms-dbfs "${AUDIO_MIN_RMS_DBFS}" "${audio_recipe_args[@]}" "${CLEAN_ARGS[@]}"
  fi
}

run_audio_region_selector() {
  local output_dir
  output_dir="$(active_output_dir)"
  local train_dataset="${SPARSE_DATASET_ROOT}/${DATASET_PREFIX}_scene_audio_175_184_sem195_frame_videoall_${AUDIO_ENCODER}_train.npz"
  local val_dataset="${SPARSE_DATASET_ROOT}/${DATASET_PREFIX}_scene_audio_175_184_sem195_frame_videoall_${AUDIO_ENCODER}_val.npz"
  if [[ ! -d "${HEAD010_BASE_RUN}" || ! -f "${HEAD010_BASE_RUN}/categorical_refiner/best.pt" ]]; then
    echo "[audio-region-selector] missing initialized RGB->195 run: ${HEAD010_BASE_RUN}" >&2
    exit 2
  fi
  if [[ ! -f "${HEAD010_BASE_CHECKPOINT}" ]]; then
    echo "[audio-region-selector] missing RGB->195 base checkpoint: ${HEAD010_BASE_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ ! -f "${train_dataset}" || ! -f "${val_dataset}" ]]; then
    echo "[audio-region-selector] missing frame-audio dataset; build Head 005/006 data first" >&2
    echo "train=${train_dataset} val=${val_dataset}" >&2
    exit 2
  fi
  local checkpoint_relative
  checkpoint_relative="${HEAD010_BASE_CHECKPOINT#"${HEAD010_BASE_RUN}/"}"
  local deployed_checkpoint="${output_dir}/${checkpoint_relative}"
  # Clone the trained RGB->195 run. Its generator and RGB category head stay frozen.
  cp -a "${HEAD010_BASE_RUN}/." "${output_dir}/"
  mkdir -p "${output_dir}/state/audio_region_selector_training"

  /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python - \
    "${HEAD010_BASE_CHECKPOINT}" "${SPARSE_CONFIG_PATH}" "${output_dir}" \
    "${train_dataset}" "${val_dataset}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

checkpoint = Path(sys.argv[1]).resolve()
sparse_config = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()
audio_train = Path(sys.argv[4]).resolve()
audio_val = Path(sys.argv[5]).resolve()
base_run = checkpoint.parents[2]
resolved = yaml.safe_load((base_run / "resolved_config.yaml").read_text()) or {}
runtime = json.loads((base_run / "resolved_runtime_config.json").read_text())
entry = json.loads(sparse_config.read_text())["wsg_v1"]["Channel_Head_000010"]
resolved.update({
    "channel_head_index": 10,
    "modality_config": str(sparse_config),
    "condition_modalities": "0-2",
    "target_modality": "195",
    "categorical_refiner_audio_train": str(audio_train),
    "categorical_refiner_audio_val": str(audio_val),
})
runtime.setdefault("wsg", {})["channel_head_index"] = 10
runtime["wsg"]["entry"] = entry
runtime.setdefault("model", {})["audio_region_selector"] = True
runtime.setdefault("train", {})["categorical_refiner_audio_train"] = str(audio_train)
runtime["train"]["categorical_refiner_audio_val"] = str(audio_val)
(output_dir / "resolved_config.yaml").write_text(
    yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
)
(output_dir / "resolved_runtime_config.json").write_text(
    json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

  echo "[audio-region-selector] initialized run=${HEAD010_BASE_RUN}"
  echo "[audio-region-selector] frozen RGB generator=${HEAD010_BASE_CHECKPOINT}"
  echo "[audio-region-selector] frozen RGB categorical refiner=${HEAD010_BASE_RUN}/categorical_refiner/best.pt"
  echo "[audio-region-selector] candidate supervision=${train_dataset}"
  echo "[audio-region-selector] start multi-region selector on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  (
    cd "${WSM_ROOT}"
    /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python \
      src/pipelines/categorical_refiner.py export-sidecar \
      --base-checkpoint "${deployed_checkpoint}" \
      --refiner-checkpoint "${HEAD010_BASE_RUN}/categorical_refiner/best.pt" \
      --output-dir "${output_dir}/categorical_refiner" \
      --head-index-override 10 --overwrite
    /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python \
      src/pipelines/categorical_refiner.py train-audio-selector \
      --audio-train "${train_dataset}" \
      --audio-val "${val_dataset}" \
      --target-modality 195 \
      --head-index 10 \
      --output-dir "${output_dir}/state/audio_region_selector_training" \
      --device cuda:0 \
      --batch-size "${HEAD010_BATCH_SIZE}" \
      --eval-batch-size "${HEAD010_EVAL_BATCH_SIZE}" \
      --max-steps "${HEAD010_TOTAL_STEPS}" \
      --eval-every "${HEAD010_EVAL_FREQ}" \
      --learning-rate "${HEAD010_LR}" \
      --maximum-candidates "${HEAD010_MAX_CANDIDATES}"
    /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python \
      src/pipelines/categorical_refiner.py export-audio-selector \
      --base-checkpoint "${deployed_checkpoint}" \
      --selector-checkpoint "${output_dir}/state/audio_region_selector_training/best.pt" \
      --output-dir "${output_dir}/audio_region_selector" \
      --overwrite
  )
}


cuda_for_channel_head() {
  local head="$1"
  local default_cuda="$2"
  local head_suffix head_num raw key value key_suffix key_num
  if [[ -z "${CHANNEL_HEAD_GPU_MAP}" || -z "${head}" ]]; then
    printf '%s\n' "${default_cuda}"
    return 0
  fi
  head_suffix="${head#Channel_Head_}"
  if [[ "${head_suffix}" =~ ^[0-9]+$ ]]; then
    head_num="$((10#${head_suffix}))"
  else
    head_num="${head_suffix}"
  fi
  IFS=',' read -r -a GPU_MAP_ITEMS <<< "${CHANNEL_HEAD_GPU_MAP}"
  for raw in "${GPU_MAP_ITEMS[@]}"; do
    raw="$(echo "${raw}" | xargs)"
    [[ -z "${raw}" ]] && continue
    if [[ "${raw}" == *"="* ]]; then
      key="${raw%%=*}"
      value="${raw#*=}"
    elif [[ "${raw}" == *":"* ]]; then
      key="${raw%%:*}"
      value="${raw#*:}"
    else
      continue
    fi
    key="$(echo "${key}" | xargs)"
    value="$(echo "${value}" | xargs)"
    key_suffix="${key#Channel_Head_}"
    key_suffix="${key_suffix#channel_head_}"
    if [[ "${key}" == "${head}" || "${key_suffix}" == "${head_suffix}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
    if [[ "${key_suffix}" =~ ^[0-9]+$ ]]; then
      key_num="$((10#${key_suffix}))"
      if [[ "${key_num}" == "${head_num}" ]]; then
        printf '%s\n' "${value}"
        return 0
      fi
    fi
  done
  printf '%s\n' "${default_cuda}"
}

default_cuda_for_task() {
  local task="$1"
  case "${task}" in
    smpl) printf '%s\n' "${SMPL_CUDA_VISIBLE_DEVICES}" ;;
    bbox) printf '%s\n' "${BBOX_CUDA_VISIBLE_DEVICES}" ;;
    dwpose) printf '%s\n' "${DWPOSE_CUDA_VISIBLE_DEVICES}" ;;
    mediapipe) printf '%s\n' "${MEDIAPIPE_CUDA_VISIBLE_DEVICES}" ;;
    obj) printf '%s\n' "${OBJ_CUDA_VISIBLE_DEVICES}" ;;
    *) printf '%s\n' "${AUDIO_CUDA_VISIBLE_DEVICES}" ;;
  esac
}

run_task() {
  local task="$1"
  local default_cuda
  local selected_cuda
  default_cuda="$(default_cuda_for_task "${task}")"
  selected_cuda="$(cuda_for_channel_head "${CURRENT_CHANNEL_HEAD:-}" "${default_cuda}")"
  case "${task}" in
    smpl)
      CUDA_VISIBLE_DEVICES="${selected_cuda}" run_smpl
      ;;
    bbox)
      CUDA_VISIBLE_DEVICES="${selected_cuda}" run_bbox
      ;;
    dwpose)
      CUDA_VISIBLE_DEVICES="${selected_cuda}" run_crop_structure dwpose "64_66"
      ;;
    mediapipe)
      CUDA_VISIBLE_DEVICES="${selected_cuda}" run_crop_structure mediapipe "95_98"
      ;;
    obj)
      CUDA_VISIBLE_DEVICES="${selected_cuda}" run_obj
      ;;
    *)
      echo "[${task}] unknown task."
      [[ "${SKIP_UNIMPLEMENTED}" == "1" ]] || exit 2
      ;;
  esac
}

run_channel_head() {
  local head="$1"
  local adapter_task="$2"
  local task_type="$3"
  local condition="$4"
  local target="$5"
  local condition_preprocess="$6"
  local pair_exists="$7"

  echo "[sparse-structure] channel_head=${head} adapter_task=${adapter_task} task_type=${task_type} condition=${condition} target=${target} condition_preprocess=${condition_preprocess} pair_exists=${pair_exists}"

  if [[ "${pair_exists}" != "1" ]]; then
    echo "[${head}] skip: pair_exists=false"
    return 0
  fi

  CURRENT_CHANNEL_HEAD="${head}"
  case "${adapter_task}" in
    smpl|bbox|dwpose|mediapipe|obj)
      run_task "${adapter_task}"
      ;;
    scene_audio_rgb_to_semantic_instance_localization)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_adapter localization "${head}_scene_audio_rgb_to_semantic_box"
      ;;
    scene_audio_rgb_to_semantic_instance_segmentation)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_adapter segmentation "${head}_scene_audio_rgb_to_semantic_mask"
      ;;
    scene_audio_to_semantic_mask_dense)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_dense_mask_adapter "${head}" overlay_baseline
      ;;
    rgb_audio_to_semantic_mask_dense_prototype_ce)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_dense_mask_adapter "${head}" rgb_mask_prototype_ce
      ;;
    rgb_to_semantic_mask_audio_region_selector)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_region_selector
      ;;
    scene_audio_to_semantic_category)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_adapter category "${head}_scene_audio_to_semantic_category"
      ;;
    crop_rgb_to_scene_audio_waveform)
      CUDA_VISIBLE_DEVICES="$(cuda_for_channel_head "${head}" "${AUDIO_CUDA_VISIBLE_DEVICES}")" run_audio_adapter visual_to_audio "${head}_crop_rgb_to_scene_audio"
      ;;
    audio_to_semantic_instance|caption_audio_to_semantic_instance|crop_rgb_to_audio_waveform|scene_audio_to_semantic_instance_localization)
      echo "[${head}] adapter_task=${adapter_task} is registered in config but no training adapter is implemented yet."
      [[ "${SKIP_UNIMPLEMENTED}" == "1" ]] || exit 2
      ;;
    *)
      echo "[${head}] unknown adapter_task=${adapter_task}"
      [[ "${SKIP_UNIMPLEMENTED}" == "1" ]] || exit 2
      ;;
  esac
}

throttle_parallel_tasks() {
  if (( PARALLEL_TASKS_INT <= 0 )); then
    return 0
  fi
  while (( ${#PIDS[@]} >= PARALLEL_TASKS_INT )); do
    local pid="${PIDS[0]}"
    wait "${pid}"
    PIDS=("${PIDS[@]:1}")
  done
}

wait_for_parallel_tasks() {
  local pid
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
}

{
  echo "[sparse-structure] start $(date)"
  echo "[sparse-structure] root=${SCOREHMR_ROOT}"
  echo "[sparse-structure] sparse_data_root=${SPARSE_DATA_ROOT}"
  echo "[sparse-structure] sparse_cache_root=${SPARSE_CACHE_ROOT}"
  echo "[sparse-structure] sparse_log_root=${SPARSE_LOG_ROOT}"
  echo "[sparse-structure] clean_log_layout=${SPARSE_CLEAN_LOG_LAYOUT}"
  echo "[sparse-structure] run=${RUN_NAME}"
  echo "[sparse-structure] dataset_prefix=${DATASET_PREFIX}"
  echo "[sparse-structure] config=${SPARSE_CONFIG_PATH}"
  echo "[sparse-structure] channel_heads=${CHANNEL_HEADS}"
  echo "[sparse-structure] channel_head_gpu_map=${CHANNEL_HEAD_GPU_MAP:-<default>}"
  echo "[sparse-structure] legacy_tasks=${TASKS:-<none>} parallel=${PARALLEL_TASKS}"
  echo "[sparse-structure] train_root=${TRAIN_ROOT}"
  echo "[sparse-structure] val_root=${VAL_ROOT}"
  echo "[sparse-structure] obj_train_roots=${OBJ_TRAIN_ROOTS}"
  echo "[sparse-structure] obj_val_roots=${OBJ_VAL_ROOTS:-split-from-train}"
  echo "[sparse-structure] max_videos=${MAX_VIDEOS} frame_stride=${FRAME_STRIDE} max_frames_per_video=${MAX_FRAMES_PER_VIDEO:-none}"
  echo "[sparse-structure] total_steps=${TOTAL_STEPS} audio_category_steps=${TOTAL_STEPS_AUDIO_CATEGORY} audio_generation_steps=${TOTAL_STEPS_AUDIO_GENERATION} batch=${BATCH_SIZE} val_batch=${BATCH_SIZE_VAL}"
  echo "[sparse-structure] obj_mode=${OBJ_MODE}"
  echo "[sparse-structure] obj_lr=${OBJ_LR}"
  echo "[sparse-structure] audio_lr=${AUDIO_LR} audio_encoder=${AUDIO_ENCODER} audio_model=${AUDIO_MODEL_NAME} audio_dim=${AUDIO_DIM}"
  echo "[sparse-structure] audio_generation_encoder=${AUDIO_GENERATION_ENCODER} audio_generation_model=${AUDIO_GENERATION_MODEL_NAME} audio_generation_device=${AUDIO_GENERATION_ENCODER_DEVICE} codec_seconds=${AUDIO_CODEC_SECONDS} codec_bandwidth=${AUDIO_CODEC_BANDWIDTH}"
  echo "[sparse-structure] audio_generation_trainer=${AUDIO_GENERATION_TRAINER} musicgen_model=${MUSICGEN_MODEL_NAME} musicgen_train_mode=${MUSICGEN_TRAIN_MODE} musicgen_lora_rank=${MUSICGEN_LORA_RANK}"
  echo "[sparse-structure] audio_frame_samples_per_video=${AUDIO_FRAME_SAMPLES_PER_VIDEO}"
  echo "[sparse-structure] audio_max_videos=${AUDIO_MAX_VIDEOS:-all} audio_frame_stride=${AUDIO_FRAME_STRIDE}"
  echo "[sparse-structure] audio_condition_source=${AUDIO_CONDITION_SOURCE}"
  echo "[sparse-structure] audio_min_rms_dbfs=${AUDIO_MIN_RMS_DBFS}"
  echo "[sparse-structure] head005_recipe=${HEAD005_RECIPE} steps=${HEAD005_TOTAL_STEPS} lr=${HEAD005_LR} fusion=${HEAD005_FUSION_CROSS_WEIGHT} mask_threshold=${HEAD005_MASK_THRESHOLD}"
  echo "[sparse-structure] head006_recipe=${HEAD006_RECIPE} steps=${HEAD006_TOTAL_STEPS} lr=${HEAD006_LR} temporal_radius=${HEAD006_TEMPORAL_RADIUS}"
  echo "[sparse-structure] max_obj_pointcloud_points=${MAX_OBJ_POINTCLOUD_POINTS}"
  echo "[sparse-structure] log_freq=${LOG_FREQ} validation_freq=${VALIDATION_FREQ} checkpoint_freq=${CHECKPOINT_FREQ}"
  echo "[sparse-structure] bbox_steps=${TOTAL_STEPS_BBOX} bbox_lr=${BBOX_LR} bbox_loss_mode=${BBOX_LOSS_MODE}"
  echo "[sparse-structure] pose_steps=${TOTAL_STEPS_POSE} pose_lr=${POSE_LR} pose_mode=${POSE_MODE}"
  echo "[sparse-structure] box_source=${BOX_SOURCE} smpl_box_source=${SMPL_BOX_SOURCE}"
  echo "[sparse-structure] force_rebuild=${FORCE_REBUILD} force_recache=${FORCE_RECACHE} reset_run=${RESET_RUN} dry_run=${DRY_RUN}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[sparse-structure] dry run: resolved tasks only; no dataset build or training will start"
    resolve_channel_heads
    exit 0
  fi

  declare -a PIDS=()
  if [[ -n "${TASKS}" ]]; then
    IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
    for RAW_TASK in "${TASK_ARRAY[@]}"; do
      TASK="$(echo "${RAW_TASK}" | xargs)"
      [[ -z "${TASK}" ]] && continue
      if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" ]]; then
        TASK_RUN_DIR="$(prepare_sparse_run_dir "${TASK}" "${TASK}" "legacy" "" "" "" "1")"
        TASK_LOG="${TASK_RUN_DIR}/train.log"
      else
        TASK_RUN_DIR="${SPARSE_LOG_ROOT}"
        TASK_LOG="${LOG_DIR}/${RUN_NAME}_${TASK}_${TIMESTAMP}.log"
      fi
      echo "[sparse-structure] ===== legacy task ${TASK}; log=${TASK_LOG} ====="
      if (( PARALLEL_TASKS_INT > 0 )); then
        throttle_parallel_tasks
        (
          SPARSE_ACTIVE_RUN_DIR="${TASK_RUN_DIR}" run_task "${TASK}"
        ) >"${TASK_LOG}" 2>&1 &
        PIDS+=("$!")
      else
        SPARSE_ACTIVE_RUN_DIR="${TASK_RUN_DIR}" run_task "${TASK}" >"${TASK_LOG}" 2>&1
      fi
    done
  else
    while IFS=$'\t' read -r HEAD ADAPTER_TASK TASK_TYPE CONDITION TARGET CONDITION_PREPROCESS PAIR_EXISTS; do
      [[ -z "${HEAD}" ]] && continue
      if [[ "${SPARSE_CLEAN_LOG_LAYOUT}" == "1" ]]; then
        TASK_RUN_DIR="$(prepare_sparse_run_dir "${HEAD}" "${ADAPTER_TASK}" "${TASK_TYPE}" "${CONDITION}" "${TARGET}" "${CONDITION_PREPROCESS}" "${PAIR_EXISTS}")"
        TASK_LOG="${TASK_RUN_DIR}/train.log"
      else
        TASK_RUN_DIR="${SPARSE_LOG_ROOT}"
        TASK_LOG="${LOG_DIR}/${RUN_NAME}_${HEAD}_${ADAPTER_TASK}_${TIMESTAMP}.log"
      fi
      echo "[sparse-structure] ===== ${HEAD} ${ADAPTER_TASK}; log=${TASK_LOG} ====="
      if (( PARALLEL_TASKS_INT > 0 )); then
        throttle_parallel_tasks
        (
          SPARSE_ACTIVE_RUN_DIR="${TASK_RUN_DIR}" run_channel_head "${HEAD}" "${ADAPTER_TASK}" "${TASK_TYPE}" "${CONDITION}" "${TARGET}" "${CONDITION_PREPROCESS}" "${PAIR_EXISTS}"
        ) >"${TASK_LOG}" 2>&1 &
        PIDS+=("$!")
      else
        SPARSE_ACTIVE_RUN_DIR="${TASK_RUN_DIR}" run_channel_head "${HEAD}" "${ADAPTER_TASK}" "${TASK_TYPE}" "${CONDITION}" "${TARGET}" "${CONDITION_PREPROCESS}" "${PAIR_EXISTS}" >"${TASK_LOG}" 2>&1
      fi
    done < <(resolve_channel_heads)
  fi

  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "[sparse-structure] waiting for ${#PIDS[@]} background task(s)"
    wait_for_parallel_tasks
  fi

  echo "[sparse-structure] finished $(date)"
} 2>&1 | tee "${LOG_FILE}"
