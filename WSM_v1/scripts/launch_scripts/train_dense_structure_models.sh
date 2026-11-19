#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/research/cbim/vast/sf895/code/WSMs/WSM_v1"
cd "${PROJECT_ROOT}"

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WSM

# TunnelingNet FLUX.2 dense structure model training.
# The historical "1.3b" component is retained in the filename to make the
# migration from the Wan launcher explicit; this launcher uses FLUX.2 Klein 4B.

# Multi-Channel_Head parallel training support
# Usage: WSG_CHANNEL_HEAD_LIST="3,270,275,313,9999" CUSTOM_CHANNEL_HEAD_JSON='{"Channel_Head_009999":{...}}' bash train_...sh
WSG_CHANNEL_HEAD_LIST="${WSG_CHANNEL_HEAD_LIST:-}"
CUSTOM_CHANNEL_HEAD_JSON="${CUSTOM_CHANNEL_HEAD_JSON:-}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INFERENCE_ONLY="${INFERENCE_ONLY:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
INFERENCE_CHECKPOINT="${INFERENCE_CHECKPOINT:-}"
INFERENCE_LABEL="${INFERENCE_LABEL:-}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-}"
MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
case "${MODALITY_CONFIG}" in
  *wsg_v1_modalities_dense_single_condition_channel_head_condition_first.json)
    MODALITY_CONFIG="configs/wsg_v1_modalities_dense.json" ;;
  *wsg_v1_modalities_sparse_single_condition_channel_head_condition_first.json)
    MODALITY_CONFIG="configs/wsg_v1_modalities_sparse.json" ;;
  *wsg_v1_modalities_custom_single_condition_channel_head_condition_first.json)
    MODALITY_CONFIG="configs/wsg_v1_modalities_custom.json" ;;
esac
if [[ -z "${TRAIN_LOG_ROOT}" ]]; then
  case "$(basename "${MODALITY_CONFIG}")" in
    wsg_v1_modalities_sparse.json) TRAIN_LOG_ROOT="train_log_sparse" ;;
    wsg_v1_modalities_custom.json) TRAIN_LOG_ROOT="train_log_custom" ;;
    *) TRAIN_LOG_ROOT="train_log" ;;
  esac
fi

# Handle multi-Channel_Head mode
if [[ -n "${WSG_CHANNEL_HEAD_LIST}" && -z "${_MULTI_CHANNEL_HEAD_CHILD:-}" ]]; then
  echo "[WSM_v1-multi] Multi-Channel_Head parallel training mode"
  echo "[WSM_v1-multi] Channel_Head list: ${WSG_CHANNEL_HEAD_LIST}"

  # Prepare custom config if provided
  ACTUAL_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
  if [[ -n "${CUSTOM_CHANNEL_HEAD_JSON}" ]]; then
    MULTI_RUN_ID="${MULTI_RUN_ID:-multi_$(date +%Y%m%d_%H%M%S)}"
    TEMP_CONFIG_DIR="${TRAIN_LOG_ROOT}/temp_configs/${MULTI_RUN_ID}"
    mkdir -p "${TEMP_CONFIG_DIR}"
    TEMP_CONFIG="${TEMP_CONFIG_DIR}/config_with_custom.json"

    /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python - \
      "${ACTUAL_CONFIG}" "${CUSTOM_CHANNEL_HEAD_JSON}" "${TEMP_CONFIG}" <<'PY'
import json
import sys

base_path, custom_json, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(base_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Merge custom Channel_Head configs
if custom_json.startswith('{'):
    custom = json.loads(custom_json)
else:
    with open(custom_json, 'r', encoding='utf-8') as f:
        custom = json.load(f)

config['wsg_v1'].update(custom)
print(f"[WSM_v1-multi] Merged {len(custom)} custom Channel_Head configs", flush=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
PY
    ACTUAL_CONFIG="${TEMP_CONFIG}"
    echo "[WSM_v1-multi] Created temporary config: ${ACTUAL_CONFIG}"
  fi

  # Cache model to fast local storage (/dev/shm RAM disk) to avoid NFS contention.
  # 4 parallel processes each reading 15GB from NFS is the startup bottleneck.
  # Copy once here, then all processes load from RAM speed.
  SRC_MODEL="${FLUX2_MODEL_PATH:-models/black-forest-labs/FLUX.2-klein-4B}"
  # Resolve symlinks: the local model path is often a symlink into NFS.
  REAL_SRC_MODEL="$(readlink -f "${SRC_MODEL}" 2>/dev/null || echo "${SRC_MODEL}")"
  if [[ "${MODEL_CACHE:-1}" == "1" && -d "${REAL_SRC_MODEL}" ]]; then
    CACHE_ROOT="${MODEL_CACHE_DIR:-/dev/shm/flux2_model_cache}"
    CACHED_MODEL="${CACHE_ROOT}/$(basename "${REAL_SRC_MODEL}")"
    # Compare real-data sizes (dereferenced). A bare symlink at dest counts as not-cached.
    src_size=$(du -sbL "${REAL_SRC_MODEL}" 2>/dev/null | cut -f1)
    if [[ -L "${CACHED_MODEL}" ]]; then dst_size=0; else dst_size=$(du -sb "${CACHED_MODEL}" 2>/dev/null | cut -f1 || echo 0); fi
    if [[ "${src_size}" != "${dst_size}" ]]; then
      echo "[WSM_v1-multi] Caching model to ${CACHED_MODEL} (one-time copy ~15GB, dereferenced)..."
      mkdir -p "${CACHE_ROOT}"
      rm -rf "${CACHED_MODEL}"
      cp -rL "${REAL_SRC_MODEL}" "${CACHED_MODEL}"
      echo "[WSM_v1-multi] Model cached ($(du -sh "${CACHED_MODEL}" 2>/dev/null | cut -f1)). All processes will load from RAM."
    else
      echo "[WSM_v1-multi] Model already cached at ${CACHED_MODEL}"
    fi
    export FLUX2_MODEL_PATH="${CACHED_MODEL}"
  fi

  # Parse Channel_Head list and launch each
  IFS=',' read -ra CHANNEL_HEAD_ARRAY <<< "${WSG_CHANNEL_HEAD_LIST}"
  IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS:-0,1,2,3,4,5,6,7}"
  if (( ${#GPU_ARRAY[@]} < ${#CHANNEL_HEAD_ARRAY[@]} )); then
    echo "[WSM_v1-multi] Need at least ${#CHANNEL_HEAD_ARRAY[@]} GPU IDs, got ${#GPU_ARRAY[@]}" >&2
    exit 1
  fi

  for i in "${!CHANNEL_HEAD_ARRAY[@]}"; do
    channel_head_idx="${CHANNEL_HEAD_ARRAY[$i]}"
    gpu_id="${GPU_ARRAY[$i]}"
    port=$((29500 + channel_head_idx))
    session_name="flux2_channel_head${channel_head_idx}_${MULTI_RUN_ID:-parallel}"

    # Auto-detect latest checkpoint for this Channel_Head
    head_dir="${TRAIN_LOG_ROOT}/Channel_Head_$(printf '%06d' ${channel_head_idx})"
    latest_checkpoint=""
    if [[ -d "${head_dir}" ]]; then
      latest_checkpoint=$(find "${head_dir}" -path "*/state/step_*" -type d 2>/dev/null | sort -V | tail -1 || true)
    fi
    resume_flag=""
    if [[ -n "${latest_checkpoint}" && -n "${RESUME_FROM_LATEST}" ]]; then
      resume_flag="RESUME_FROM_CHECKPOINT=${latest_checkpoint}"
      echo "[WSM_v1-multi]   Resume from: ${latest_checkpoint}"
    fi

    echo "[WSM_v1-multi] Launching Channel_Head ${channel_head_idx} on GPU ${gpu_id} port ${port}"

    env -u LD_LIBRARY_PATH tmux new-session -d -s "${session_name}" \
      "cd ${PROJECT_ROOT} && \
       _MULTI_CHANNEL_HEAD_CHILD=1 \
       ${resume_flag} \
       FLUX2_MODEL_PATH=${FLUX2_MODEL_PATH} \
       TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT} \
       MODALITY_CONFIG=${ACTUAL_CONFIG} \
       WSG_DENSE_LATENT_CACHE_DIR=${WSG_DENSE_LATENT_CACHE_DIR:-data/wsg_flux2_frame_latents_dense_multidataset} \
       WSG_XPOSE_LATENT_CACHE_DIR=${WSG_XPOSE_LATENT_CACHE_DIR:-data/xpose_flux2_frame_latents_sample10pct_2frames} \
       WSG_USE_XPOSE_CACHE=${WSG_USE_XPOSE_CACHE:-1} \
       WSG_TEXT_EMBED_CACHE_DIR=${WSG_TEXT_EMBED_CACHE_DIR:-data/wsg_flux2_caption_embeddings_all_datasets} \
       WSG_AUDIO_DENSE_CONDITION_TRAIN=${WSG_AUDIO_DENSE_CONDITION_TRAIN:-} \
       WSG_AUDIO_DENSE_CONDITION_VAL=${WSG_AUDIO_DENSE_CONDITION_VAL:-} \
       WSG_AUDIO_CONDITION_DIM=${WSG_AUDIO_CONDITION_DIM:-0} \
       WSG_AUDIO_DENSE_TARGET_CACHE_DIR=${WSG_AUDIO_DENSE_TARGET_CACHE_DIR:-} \
       WSG_AUDIO_DENSE_TARGET_MODALITY=${WSG_AUDIO_DENSE_TARGET_MODALITY:-} \
       WSG_TARGET_VISUAL_MODALITY=${WSG_TARGET_VISUAL_MODALITY:-} \
       WSG_AUDIO_CONDITIONING_MODE=${WSG_AUDIO_CONDITIONING_MODE:-attention} \
       WSG_CHANNEL_HEAD_INDEX=${channel_head_idx} \
       GPU_IDS=${gpu_id} \
       NUM_PROCESSES=1 \
       MAIN_PROCESS_PORT=${port} \
       RUN_IN_TMUX=0 \
       TMUX_SESSION=${session_name} \
       MAX_STEPS=${MAX_STEPS:-1000} \
       EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-20} \
       VIZ_EVERY_STEPS=${VIZ_EVERY_STEPS:-50} \
       SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-50} \
       TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1} \
       GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1} \
       LEARNING_RATE=${LEARNING_RATE:-1e-4} \
       NUM_WORKERS=${NUM_WORKERS:-4} \
       VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-0} \
       RESUME_FROM_LATEST=${RESUME_FROM_LATEST:-} \
       WSG_FLUX_CONDITION_ATTENTION_MODE=${WSG_FLUX_CONDITION_ATTENTION_MODE:-auto} \
       WSG_ATTENTION_ESTIMATION=${WSG_ATTENTION_ESTIMATION:-auto} \
       WSG_ATTENTION_CONTROL=${WSG_ATTENTION_CONTROL:-auto} \
       WSG_ATTENTION_INVERSE_RENDERING=${WSG_ATTENTION_INVERSE_RENDERING:-auto} \
       WSG_ATTENTION_NARROWGEN=${WSG_ATTENTION_NARROWGEN:-auto} \
       WSG_TRAINABLE_TARGET=${WSG_TRAINABLE_TARGET:-auto} \
       WSG_TRAINABLE_CONDITION=${WSG_TRAINABLE_CONDITION:-auto} \
       WSG_DENSE_ENHANCE=${WSG_DENSE_ENHANCE:-auto} \
       WSG_DE_TIMESTEP=${WSG_DE_TIMESTEP:-auto} \
       WSG_DE_MINSNR=${WSG_DE_MINSNR:-auto} \
       WSG_DE_HUBER=${WSG_DE_HUBER:-auto} \
       WSG_SINGLE_STEP_X0=${WSG_SINGLE_STEP_X0:-0} \
       WSG_GRADIENT_DOMAIN_LOSS_WEIGHT=${WSG_GRADIENT_DOMAIN_LOSS_WEIGHT:-0} \
       WSG_FOCAL_REGRESSION_GAMMA=${WSG_FOCAL_REGRESSION_GAMMA:-0} \
       WSG_FOREGROUND_IOU_LOSS_WEIGHT=${WSG_FOREGROUND_IOU_LOSS_WEIGHT:-0} \
       WSG_CATEGORICAL_LOSS_MODE=${WSG_CATEGORICAL_LOSS_MODE:-off} \
       WSG_CATEGORICAL_LOSS_WEIGHT=${WSG_CATEGORICAL_LOSS_WEIGHT:-0.5} \
       WSG_CATEGORICAL_TEMPERATURE=${WSG_CATEGORICAL_TEMPERATURE:-0.25} \
       WSG_CATEGORICAL_MARGIN=${WSG_CATEGORICAL_MARGIN:-0.25} \
       WSG_CATEGORICAL_NUM_CLUSTERS=${WSG_CATEGORICAL_NUM_CLUSTERS:-8} \
       WSG_CATEGORICAL_KMEANS_ITERATIONS=${WSG_CATEGORICAL_KMEANS_ITERATIONS:-4} \
       WSG_HIGH_NOISE_ANCHOR=${WSG_HIGH_NOISE_ANCHOR:-auto} \
       WSG_MULTIRES_NOISE=${WSG_MULTIRES_NOISE:-0} \
       WSG_HISTORICAL_REFERENCE_POLICY=${WSG_HISTORICAL_REFERENCE_POLICY:-tolerant} \
       RUN_ID_SUFFIX=${RUN_ID_SUFFIX:-} \
       bash scripts/launch_scripts/train_dense_structure_models.sh"

    echo "[WSM_v1-multi]   Session: ${session_name}"
  done

  echo "[WSM_v1-multi] All ${#CHANNEL_HEAD_ARRAY[@]} Channel_Head training sessions started"
  echo "[WSM_v1-multi] List sessions: tmux ls"
  echo "[WSM_v1-multi] Attach: tmux attach -t flux2_channel_head<INDEX>_..."
  exit 0
fi

FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-models/black-forest-labs/FLUX.2-klein-4B}"
TRAIN_ENTRY="${TRAIN_ENTRY:-scripts/train_TunnelingNet_Flux2_4B_wsg.py}"
MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
WSG_DENSE_LATENT_CACHE_DIR="${WSG_DENSE_LATENT_CACHE_DIR:-data/wsg_flux2_frame_latents_dense_multidataset}"
WSG_XPOSE_LATENT_CACHE_DIR="${WSG_XPOSE_LATENT_CACHE_DIR:-data/xpose_flux2_frame_latents_sample10pct_2frames}"
WSG_USE_XPOSE_CACHE="${WSG_USE_XPOSE_CACHE:-1}"

WSG_CHANNEL_HEAD_INDEX="${WSG_CHANNEL_HEAD_INDEX:-270}"
if [[ "${WSG_CHANNEL_HEAD_INDEX}" =~ ^[0-9]+$ ]]; then
  WSG_CHANNEL_HEAD_INDEX="$((10#${WSG_CHANNEL_HEAD_INDEX}))"
fi
printf -v WSG_CHANNEL_HEAD_NAME "Channel_Head_%06d" "${WSG_CHANNEL_HEAD_INDEX}"
WSG_TEXT_EMBED_CACHE_DIR="${WSG_TEXT_EMBED_CACHE_DIR:-data/wsg_flux2_caption_embeddings_all_datasets}"
# Optional frame-level audio condition NPZs for audio-conditioned dense tasks
# such as sparse Channel_Head_000008. Empty keeps the standard visual-latent path.
WSG_AUDIO_DENSE_CONDITION_TRAIN="${WSG_AUDIO_DENSE_CONDITION_TRAIN:-}"
WSG_AUDIO_DENSE_CONDITION_VAL="${WSG_AUDIO_DENSE_CONDITION_VAL:-}"
WSG_AUDIO_CONDITION_DIM="${WSG_AUDIO_CONDITION_DIM:-0}"
WSG_AUDIO_DENSE_TARGET_CACHE_DIR="${WSG_AUDIO_DENSE_TARGET_CACHE_DIR:-}"
WSG_AUDIO_DENSE_TARGET_MODALITY="${WSG_AUDIO_DENSE_TARGET_MODALITY:-}"
WSG_TARGET_VISUAL_MODALITY="${WSG_TARGET_VISUAL_MODALITY:-}"
# Audio condition injection mode. attention preserves the old behavior; modulation
# uses FiLM-style target modulation and does not append audio as an attention lane.
WSG_AUDIO_CONDITIONING_MODE="${WSG_AUDIO_CONDITIONING_MODE:-attention}"
# Audio condition option. Defaults are off to preserve prior runs.
# WSG_AUDIO_GLOBAL_CONDITION=1 keeps audio_condition_feature as one global token
# instead of expanding it spatially.
WSG_AUDIO_GLOBAL_CONDITION="${WSG_AUDIO_GLOBAL_CONDITION:-0}"
WSG_EDIT_LOSS_WEIGHT="${WSG_EDIT_LOSS_WEIGHT:-0}"
WSG_EDIT_LOSS_POWER="${WSG_EDIT_LOSS_POWER:-1}"
# Optional comma-separated dataset_name whitelist for the target modality. Empty
# means use target_dataset_whitelist_by_modality from the modality config.
WSG_TARGET_DATASET_WHITELIST="${WSG_TARGET_DATASET_WHITELIST:-}"
WSG_CHANNEL_HEAD_RANK="${WSG_CHANNEL_HEAD_RANK:-8}"
WSG_CHANNEL_HEAD_ALPHA="${WSG_CHANNEL_HEAD_ALPHA:-274}"
MAX_STEPS="${MAX_STEPS:-1000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
TRAINING_SEED="${TRAINING_SEED:-20260622}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-20}"
VIZ_EVERY_STEPS="${VIZ_EVERY_STEPS:-50}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-5}"
DISABLE_PREFLIGHT="${DISABLE_PREFLIGHT:-0}"
SAVE_OPTIMIZER_STATE_8BIT="${SAVE_OPTIMIZER_STATE_8BIT:-1}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.05}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
MAX_LOCAL_SAMPLES="${MAX_LOCAL_SAMPLES:-0}"
VAL_LOSS_BATCHES="${VAL_LOSS_BATCHES:-24}"
# Fixed timestep grid tiled across val batches (deterministic, low-variance,
# comparable across runs). Empty => fixed random timestep per batch.
VAL_TIMESTEP_GRID="${VAL_TIMESTEP_GRID:-0.1,0.25,0.5,0.75,0.9}"
# Checkpoint only on validation-loss improvement; stop if no improvement within
# this many seconds (default 7200 = 2h).
EARLY_STOP_SECONDS="${EARLY_STOP_SECONDS:-7200}"
# Preview frame selection. Leaving SAMPLE_ID/VIDEO_ID/FRAME_INDEX empty makes
# the trainer pick a deterministic sample from the current task's held-out split.
# Set VIDEO_ID for video-specific inspection, or SAMPLE_ID to pin an exact frame.
WSG_PREVIEW_SAMPLE_ID="${WSG_PREVIEW_SAMPLE_ID:-}"
WSG_PREVIEW_VIDEO_ID="${WSG_PREVIEW_VIDEO_ID:-}"
WSG_PREVIEW_FRAME_INDEX="${WSG_PREVIEW_FRAME_INDEX:-}"
WSG_PREVIEW_FRAME_FRACTION="${WSG_PREVIEW_FRAME_FRACTION:-0.75}"
WSG_PREVIEW_SPLIT="${WSG_PREVIEW_SPLIT:-test}"
WSG_PREVIEW_NUM_SAMPLES="${WSG_PREVIEW_NUM_SAMPLES:-3}"
# Condition attention topology, resolved PER TASK_TYPE.
# "modality_tunneling" = Modality Tunneling Attention: target self-attention +
# additive target->condition/reference attention. "native_joint" = Native Joint
# Attention: original FLUX/Omini-style single-softmax. Per-task defaults:
# estimation/conditional_control/inverse_rendering => modality_tunneling,
# narrow_generation => native_joint. Legacy aliases "dual"/"joint" are accepted.
WSG_FLUX_CONDITION_ATTENTION_MODE="${WSG_FLUX_CONDITION_ATTENTION_MODE:-auto}"
WSG_ATTENTION_ESTIMATION="${WSG_ATTENTION_ESTIMATION:-auto}"
WSG_ATTENTION_CONTROL="${WSG_ATTENTION_CONTROL:-auto}"
WSG_ATTENTION_INVERSE_RENDERING="${WSG_ATTENTION_INVERSE_RENDERING:-auto}"
WSG_ATTENTION_NARROWGEN="${WSG_ATTENTION_NARROWGEN:-auto}"
# Target lane trainability. "auto" (default) uses the per-task recipe
# (estimation/control/inverse_rendering => on, narrow_generation => off). "on"/"off"
# force all tasks. "off" reproduces the original frozen-target recipe.
WSG_TRAINABLE_TARGET="${WSG_TRAINABLE_TARGET:-auto}"
WSG_TRAINABLE_CONDITION="${WSG_TRAINABLE_CONDITION:-auto}"
# Universal dense-prediction loss enhancement (timestep low-noise shift + Min-SNR-gamma
# weighting + Huber loss). "auto" (default) uses the per-task recipe
# (estimation/control/inverse_rendering => on, narrow_generation => off). "on"/"off"
# force all tasks. Hyperparameters are tunable via WSG_DENSE_ENHANCE_{SHIFT,SCALE,GAMMA,HUBER_DELTA}.
WSG_DENSE_ENHANCE="${WSG_DENSE_ENHANCE:-auto}"
# Per-lever ablation switches. "auto" follows WSG_DENSE_ENHANCE; "on"/"off" override
# a single lever so each of A (timestep shift) / B (Min-SNR) / C (Huber) can be
# trained in isolation.
WSG_DE_TIMESTEP="${WSG_DE_TIMESTEP:-auto}"
WSG_DE_MINSNR="${WSG_DE_MINSNR:-auto}"
WSG_DE_HUBER="${WSG_DE_HUBER:-auto}"
WSG_SINGLE_STEP_X0="${WSG_SINGLE_STEP_X0:-0}"
WSG_GRADIENT_DOMAIN_LOSS_WEIGHT="${WSG_GRADIENT_DOMAIN_LOSS_WEIGHT:-0}"
WSG_FOCAL_REGRESSION_GAMMA="${WSG_FOCAL_REGRESSION_GAMMA:-0}"
WSG_FOREGROUND_IOU_LOSS_WEIGHT="${WSG_FOREGROUND_IOU_LOSS_WEIGHT:-0}"
WSG_CATEGORICAL_LOSS_MODE="${WSG_CATEGORICAL_LOSS_MODE:-off}"
WSG_CATEGORICAL_LOSS_WEIGHT="${WSG_CATEGORICAL_LOSS_WEIGHT:-0.5}"
WSG_CATEGORICAL_TEMPERATURE="${WSG_CATEGORICAL_TEMPERATURE:-0.25}"
WSG_CATEGORICAL_MARGIN="${WSG_CATEGORICAL_MARGIN:-0.25}"
WSG_CATEGORICAL_NUM_CLUSTERS="${WSG_CATEGORICAL_NUM_CLUSTERS:-8}"
WSG_CATEGORICAL_KMEANS_ITERATIONS="${WSG_CATEGORICAL_KMEANS_ITERATIONS:-4}"
WSG_HIGH_NOISE_ANCHOR="${WSG_HIGH_NOISE_ANCHOR:-auto}"
WSG_MULTIRES_NOISE="${WSG_MULTIRES_NOISE:-0}"
WSG_SEPARATE_CONDITION_ADAPTERS="${WSG_SEPARATE_CONDITION_ADAPTERS:-0}"
# Historical RGB routing for inverse rendering and narrow generation.
# strict drops samples without an earlier frame; tolerant uses a task-safe fallback.
WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY:-tolerant}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29620}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"
TMUX_SESSION="${TMUX_SESSION:-flux2_omini_channel_head_${WSG_CHANNEL_HEAD_INDEX}}"
# RUN_ID_SUFFIX (optional) tags the run dir so different configs are tellable
# apart, e.g. 20260624_161052_ablA_timestep. It also avoids same-second timestamp
# collisions when several runs of the same Channel_Head start together.
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)${RUN_ID_SUFFIX:+_${RUN_ID_SUFFIX}}}"
RUN_ROOT="${RUN_ROOT:-${TRAIN_LOG_ROOT}/Channel_Head_$(printf '%06d' "${WSG_CHANNEL_HEAD_INDEX}")/${RUN_ID}}"
LAUNCH_LOG="${LAUNCH_LOG:-${RUN_ROOT}/launch.log}"
RESOLVED_CONFIG="${RESOLVED_CONFIG:-${RUN_ROOT}/resolved_config.yaml}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${INFERENCE_ONLY}" == "1" ]]; then
  if [[ -z "${INFERENCE_CHECKPOINT}" ]]; then
    INFERENCE_CHECKPOINT="$(find "${TRAIN_LOG_ROOT}/${WSG_CHANNEL_HEAD_NAME}" -path "*/ckpt/*/pytorch_lora_weights.safetensors" -type f 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${INFERENCE_CHECKPOINT}" ]]; then
    echo "[WSM_v1] ERROR: no checkpoint found for ${WSG_CHANNEL_HEAD_NAME}" >&2
    exit 1
  fi
  RUN_ROOT="$(dirname "$(dirname "$(dirname "${INFERENCE_CHECKPOINT}")")")"
  checkpoint_dir="$(basename "$(dirname "${INFERENCE_CHECKPOINT}")")"
  if [[ -z "${INFERENCE_LABEL}" ]]; then
    INFERENCE_LABEL="${checkpoint_dir#final_}"
    [[ "${INFERENCE_LABEL}" == step_* ]] || INFERENCE_LABEL="step_${INFERENCE_LABEL}"
  fi
  LAUNCH_LOG="${RUN_ROOT}/reinference_${INFERENCE_LABEL}.log"
  RESOLVED_CONFIG="${RUN_ROOT}/resolved_inference_config.yaml"
fi

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[WSM_v1] ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${FLUX2_MODEL_PATH}" "FLUX.2 model"
require_path "${MODALITY_CONFIG}" "modality config"
require_path "${WSG_DENSE_LATENT_CACHE_DIR}/shards" "dense latent shards"
if [[ "${WSG_USE_XPOSE_CACHE}" == "1" ]]; then
  require_path "${WSG_XPOSE_LATENT_CACHE_DIR}/shards" "XPose latent shards"
fi

mkdir -p "${RUN_ROOT}"

# Auto-detect latest checkpoint for single-Channel_Head resume.
if [[ -z "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_LATEST}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  head_dir="${TRAIN_LOG_ROOT}/Channel_Head_$(printf '%06d' ${WSG_CHANNEL_HEAD_INDEX})"
  if [[ -d "${head_dir}" ]]; then
    latest_checkpoint=$(find "${head_dir}" -path "*/state/step_*" -type d 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "${latest_checkpoint}" ]]; then
      RESUME_FROM_CHECKPOINT="${latest_checkpoint}"
      echo "[WSM_v1] Resume from latest checkpoint: ${latest_checkpoint}"
    fi
  fi
fi

if [[ -n "${CUSTOM_CHANNEL_HEAD_JSON}" ]]; then
  TEMP_CONFIG_DIR="${TRAIN_LOG_ROOT}/temp_configs/${RUN_ID}"
  mkdir -p "${TEMP_CONFIG_DIR}"
  TEMP_CONFIG="${TEMP_CONFIG_DIR}/config_with_custom.json"
  /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python - \
    "${MODALITY_CONFIG}" "${CUSTOM_CHANNEL_HEAD_JSON}" "${TEMP_CONFIG}" <<'PYMERGE'
import json
import sys

base_path, custom_json, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(base_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)
if custom_json.startswith("{"):
    custom = json.loads(custom_json)
else:
    with open(custom_json, "r", encoding="utf-8") as handle:
        custom = json.load(handle)
config["wsg_v1"].update(custom)
print(f"[WSM_v1] merged {len(custom)} launcher Channel_Head override(s): {output_path}", flush=True)
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=False)
PYMERGE
  MODALITY_CONFIG="${TEMP_CONFIG}"
fi

TARGET_AND_CONDITION="$(
  /research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python - \
    "${MODALITY_CONFIG}" "${WSG_CHANNEL_HEAD_INDEX}" <<'PY'
import json
import os
import sys

path, index = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as handle:
    config = json.load(handle)
entries = config.get("wsg_v1", config)
key = f"Channel_Head_{index:06d}"
if key not in entries:
    raise SystemExit(f"Missing {key} in {path}")
entry = entries[key]
target = entry.get("target")
condition = entry.get("condition")
if target is None or condition is None:
    raise SystemExit(f"{key} must define target and condition")
if isinstance(condition, list):
    condition = ",".join(str(item) for item in condition)
task_type = str(entry.get("task_type") or "")
recipes = {
    "estimation_or_reconstruction": {"attention": "modality_tunneling", "trainable_target": True, "trainable_condition": False, "dense_enhance": False, "high_noise_anchor": True},
    "conditional_control": {"attention": "modality_tunneling", "trainable_target": True, "trainable_condition": False, "dense_enhance": False, "high_noise_anchor": True},
    "inverse_rendering": {"attention": "modality_tunneling", "trainable_target": True, "trainable_condition": False, "dense_enhance": False, "high_noise_anchor": True},
    "conditional_control+narrow_generation": {"attention": "native_joint", "trainable_target": True, "trainable_condition": False, "dense_enhance": False, "high_noise_anchor": True},
}
default = {"attention": "modality_tunneling", "trainable_target": True, "trainable_condition": True, "dense_enhance": False, "high_noise_anchor": False}
recipe = recipes.get(task_type, default)

def env(name, default_value):
    return os.environ.get(name, default_value)

def onoff_auto_bool(raw, recipe_value):
    raw = str(raw).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(recipe_value)

def normalize_attention(value):
    value = str(value).strip().lower()
    aliases = {
        "joint": "native_joint",
        "native_joint": "native_joint",
        "native-joint": "native_joint",
        "native_joint_attention": "native_joint",
        "dual": "modality_tunneling",
        "modality_tunneling": "modality_tunneling",
        "modality-tunneling": "modality_tunneling",
        "modality_tunneling_attention": "modality_tunneling",
    }
    return aliases.get(value)

def resolve_attention():
    global_mode = normalize_attention(env("WSG_FLUX_CONDITION_ATTENTION_MODE", "auto"))
    if global_mode:
        return global_mode
    per_task_env = {
        "estimation_or_reconstruction": "WSG_ATTENTION_ESTIMATION",
        "conditional_control": "WSG_ATTENTION_CONTROL",
        "inverse_rendering": "WSG_ATTENTION_INVERSE_RENDERING",
        "conditional_control+narrow_generation": "WSG_ATTENTION_NARROWGEN",
    }.get(task_type)
    if per_task_env:
        value = normalize_attention(env(per_task_env, "auto"))
        if value:
            return value
    return recipe["attention"]

def resolve_onoff(name, recipe_key):
    raw = env(name, "auto")
    return onoff_auto_bool(raw, recipe[recipe_key])

resolved_attention = resolve_attention()
resolved_trainable_target = resolve_onoff("WSG_TRAINABLE_TARGET", "trainable_target")
resolved_trainable_condition = resolve_onoff("WSG_TRAINABLE_CONDITION", "trainable_condition")
resolved_dense_enhance = resolve_onoff("WSG_DENSE_ENHANCE", "dense_enhance")

def lever(name):
    raw = env(name, "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return resolved_dense_enhance

resolved_de_timestep = lever("WSG_DE_TIMESTEP")
resolved_de_minsnr = lever("WSG_DE_MINSNR")
resolved_de_huber = lever("WSG_DE_HUBER")
resolved_high_noise_anchor = resolve_onoff("WSG_HIGH_NOISE_ANCHOR", "high_noise_anchor")
resolved_single_step_x0 = env("WSG_SINGLE_STEP_X0", "0").strip().lower() in {"1", "true", "yes", "on"}
resolved_multires_noise = env("WSG_MULTIRES_NOISE", "0").strip().lower() in {"1", "true", "yes", "on"}
fields = [
    target, condition, task_type, resolved_attention, str(resolved_trainable_target).lower(),
    str(resolved_trainable_condition).lower(), str(resolved_dense_enhance).lower(),
    str(resolved_de_timestep).lower(), str(resolved_de_minsnr).lower(), str(resolved_de_huber).lower(),
    str(resolved_single_step_x0).lower(), str(resolved_high_noise_anchor).lower(), str(resolved_multires_noise).lower(),
]
print("\t".join(fields))
PY
)"
IFS=$'\t' read -r TARGET_MODALITY CONDITION_MODALITIES TASK_TYPE RESOLVED_ATTENTION RESOLVED_TRAINABLE_TARGET RESOLVED_TRAINABLE_CONDITION RESOLVED_DENSE_ENHANCE RESOLVED_DE_TIMESTEP RESOLVED_DE_MINSNR RESOLVED_DE_HUBER RESOLVED_SINGLE_STEP_X0 RESOLVED_HIGH_NOISE_ANCHOR RESOLVED_MULTIRES_NOISE <<< "${TARGET_AND_CONDITION}"

cat > "${RESOLVED_CONFIG}" <<EOF
run_id: "${RUN_ID}"
launcher: "scripts/launch_scripts/train_dense_structure_models.sh"
training_entry: "${TRAIN_ENTRY}"
base_model: "${FLUX2_MODEL_PATH}"
modality_config: "${MODALITY_CONFIG}"
dense_latent_cache: "${WSG_DENSE_LATENT_CACHE_DIR}"
xpose_latent_cache: "${WSG_XPOSE_LATENT_CACHE_DIR}"
text_embedding_cache: "${WSG_TEXT_EMBED_CACHE_DIR}"
audio_dense_condition_train: "${WSG_AUDIO_DENSE_CONDITION_TRAIN}"
audio_dense_condition_val: "${WSG_AUDIO_DENSE_CONDITION_VAL}"
audio_condition_dim: ${WSG_AUDIO_CONDITION_DIM}
audio_dense_target_cache_dir: "${WSG_AUDIO_DENSE_TARGET_CACHE_DIR}"
audio_dense_target_modality: "${WSG_AUDIO_DENSE_TARGET_MODALITY}"
target_visual_modality: "${WSG_TARGET_VISUAL_MODALITY}"
audio_conditioning_mode: "${WSG_AUDIO_CONDITIONING_MODE}"
audio_global_condition: ${WSG_AUDIO_GLOBAL_CONDITION}
edit_loss_weight: ${WSG_EDIT_LOSS_WEIGHT}
edit_loss_power: ${WSG_EDIT_LOSS_POWER}
target_dataset_whitelist: "${WSG_TARGET_DATASET_WHITELIST}"
use_xpose_cache: ${WSG_USE_XPOSE_CACHE}
channel_head_index: ${WSG_CHANNEL_HEAD_INDEX}
target_modality: "${TARGET_MODALITY}"
condition_modalities: "${CONDITION_MODALITIES}"
channel_head_rank: ${WSG_CHANNEL_HEAD_RANK}
channel_head_alpha: "${WSG_CHANNEL_HEAD_ALPHA:-auto}"
condition_attention_mode: "${WSG_FLUX_CONDITION_ATTENTION_MODE}"
attention_estimation: "${WSG_ATTENTION_ESTIMATION}"
attention_conditional_control: "${WSG_ATTENTION_CONTROL}"
attention_inverse_rendering: "${WSG_ATTENTION_INVERSE_RENDERING}"
attention_narrowgen: "${WSG_ATTENTION_NARROWGEN}"
trainable_target: "${WSG_TRAINABLE_TARGET}"
trainable_condition: "${WSG_TRAINABLE_CONDITION}"
dense_enhance: "${WSG_DENSE_ENHANCE}"
de_timestep: "${WSG_DE_TIMESTEP}"
de_minsnr: "${WSG_DE_MINSNR}"
de_huber: "${WSG_DE_HUBER}"
single_step_x0: ${WSG_SINGLE_STEP_X0}
gradient_domain_loss_weight: ${WSG_GRADIENT_DOMAIN_LOSS_WEIGHT}
focal_regression_gamma: ${WSG_FOCAL_REGRESSION_GAMMA}
foreground_iou_loss_weight: ${WSG_FOREGROUND_IOU_LOSS_WEIGHT}
categorical_loss_mode: "${WSG_CATEGORICAL_LOSS_MODE}"
categorical_loss_weight: ${WSG_CATEGORICAL_LOSS_WEIGHT}
categorical_temperature: ${WSG_CATEGORICAL_TEMPERATURE}
categorical_margin: ${WSG_CATEGORICAL_MARGIN}
categorical_num_clusters: ${WSG_CATEGORICAL_NUM_CLUSTERS}
categorical_kmeans_iterations: ${WSG_CATEGORICAL_KMEANS_ITERATIONS}
high_noise_anchor: "${WSG_HIGH_NOISE_ANCHOR}"
multires_noise: ${WSG_MULTIRES_NOISE}
historical_reference_policy: "${WSG_HISTORICAL_REFERENCE_POLICY}"
resolved_task_type: "${TASK_TYPE}"
resolved_condition_attention_mode: "${RESOLVED_ATTENTION}"
resolved_trainable_target: ${RESOLVED_TRAINABLE_TARGET}
resolved_trainable_condition: ${RESOLVED_TRAINABLE_CONDITION}
resolved_dense_enhance: ${RESOLVED_DENSE_ENHANCE}
resolved_de_timestep: ${RESOLVED_DE_TIMESTEP}
resolved_de_minsnr: ${RESOLVED_DE_MINSNR}
resolved_de_huber: ${RESOLVED_DE_HUBER}
resolved_single_step_x0: ${RESOLVED_SINGLE_STEP_X0}
resolved_high_noise_anchor: ${RESOLVED_HIGH_NOISE_ANCHOR}
resolved_multires_noise: ${RESOLVED_MULTIRES_NOISE}
max_steps: ${MAX_STEPS}
batch_size_per_process: ${TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
learning_rate: ${LEARNING_RATE}
num_workers: ${NUM_WORKERS}
val_num_workers: ${VAL_NUM_WORKERS}
mixed_precision: "${MIXED_PRECISION}"
gradient_checkpointing: ${GRADIENT_CHECKPOINTING}
seed: ${TRAINING_SEED}
save_every_steps: ${SAVE_EVERY_STEPS}
eval_every_steps: ${EVAL_EVERY_STEPS}
log_every_steps: ${LOG_EVERY_STEPS}
disable_preflight: ${DISABLE_PREFLIGHT}
save_optimizer_state_8bit: ${SAVE_OPTIMIZER_STATE_8BIT}
val_split_ratio: ${VAL_SPLIT_RATIO}
max_val_samples: ${MAX_VAL_SAMPLES}
max_local_samples: ${MAX_LOCAL_SAMPLES}
val_loss_batches: ${VAL_LOSS_BATCHES}
val_timestep_grid: "${VAL_TIMESTEP_GRID}"
early_stop_seconds: ${EARLY_STOP_SECONDS}
preview_sample_id: "${WSG_PREVIEW_SAMPLE_ID}"
preview_video_id: "${WSG_PREVIEW_VIDEO_ID}"
preview_frame_index: "${WSG_PREVIEW_FRAME_INDEX}"
preview_frame_fraction: "${WSG_PREVIEW_FRAME_FRACTION}"
preview_num_samples: ${WSG_PREVIEW_NUM_SAMPLES}
gpu_ids: "${GPU_IDS}"
num_processes: ${NUM_PROCESSES}
main_process_port: ${MAIN_PROCESS_PORT}
run_root: "${RUN_ROOT}"
EOF

COMMAND=(
  accelerate launch
)
if (( NUM_PROCESSES > 1 )); then
  COMMAND+=(--multi_gpu)
fi
COMMAND+=(
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  --mixed_precision "${MIXED_PRECISION}"
  "${TRAIN_ENTRY}"
  --model-path "${FLUX2_MODEL_PATH}"
  --modality-config "${MODALITY_CONFIG}"
  --dense-latent-cache-dir "${WSG_DENSE_LATENT_CACHE_DIR}"
  --text-embedding-cache-dir "${WSG_TEXT_EMBED_CACHE_DIR}"
  --audio-dense-condition-train "${WSG_AUDIO_DENSE_CONDITION_TRAIN}"
  --audio-dense-condition-val "${WSG_AUDIO_DENSE_CONDITION_VAL}"
  --audio-condition-dim "${WSG_AUDIO_CONDITION_DIM}"
  --audio-dense-target-cache-dir "${WSG_AUDIO_DENSE_TARGET_CACHE_DIR}"
  --audio-dense-target-modality "${WSG_AUDIO_DENSE_TARGET_MODALITY}"
  --target-visual-modality "${WSG_TARGET_VISUAL_MODALITY}"
  --audio-conditioning-mode "${WSG_AUDIO_CONDITIONING_MODE}"
  --target-dataset-whitelist "${WSG_TARGET_DATASET_WHITELIST}"
  --channel-head-index "${WSG_CHANNEL_HEAD_INDEX}"
  --target-modality "${TARGET_MODALITY}"
  --condition-modalities "${CONDITION_MODALITIES}"
  --channel-head-rank "${WSG_CHANNEL_HEAD_RANK}"
  --max-steps "${MAX_STEPS}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --num-workers "${NUM_WORKERS}"
  --val-num-workers "${VAL_NUM_WORKERS}"
  --seed "${TRAINING_SEED}"
  --save-every-steps "${SAVE_EVERY_STEPS}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
  --viz-every-steps "${VIZ_EVERY_STEPS}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --output-dir "${RUN_ROOT}"
  --val-split-ratio "${VAL_SPLIT_RATIO}"
  --max-val-samples "${MAX_VAL_SAMPLES}"
  --max-local-samples "${MAX_LOCAL_SAMPLES}"
  --val-loss-batches "${VAL_LOSS_BATCHES}"
  --val-timestep-grid "${VAL_TIMESTEP_GRID}"
  --early-stop-seconds "${EARLY_STOP_SECONDS}"
  --condition-attention-mode "${WSG_FLUX_CONDITION_ATTENTION_MODE}"
  --attention-estimation "${WSG_ATTENTION_ESTIMATION}"
  --attention-conditional-control "${WSG_ATTENTION_CONTROL}"
  --attention-inverse-rendering "${WSG_ATTENTION_INVERSE_RENDERING}"
  --attention-narrowgen "${WSG_ATTENTION_NARROWGEN}"
  --historical-reference-policy "${WSG_HISTORICAL_REFERENCE_POLICY}"
)

COMMAND+=(--trainable-target "${WSG_TRAINABLE_TARGET}")
COMMAND+=(--trainable-condition "${WSG_TRAINABLE_CONDITION}")
COMMAND+=(--dense-enhance "${WSG_DENSE_ENHANCE}")
COMMAND+=(--de-timestep "${WSG_DE_TIMESTEP}")
COMMAND+=(--de-minsnr "${WSG_DE_MINSNR}")
COMMAND+=(--de-huber "${WSG_DE_HUBER}")
if [[ "${WSG_SINGLE_STEP_X0}" == "1" ]]; then COMMAND+=(--single-step-x0); fi
COMMAND+=(--gradient-domain-loss-weight "${WSG_GRADIENT_DOMAIN_LOSS_WEIGHT}")
COMMAND+=(--focal-regression-gamma "${WSG_FOCAL_REGRESSION_GAMMA}")
COMMAND+=(--foreground-iou-loss-weight "${WSG_FOREGROUND_IOU_LOSS_WEIGHT}")
COMMAND+=(--categorical-loss-mode "${WSG_CATEGORICAL_LOSS_MODE}")
COMMAND+=(--categorical-loss-weight "${WSG_CATEGORICAL_LOSS_WEIGHT}")
COMMAND+=(--categorical-temperature "${WSG_CATEGORICAL_TEMPERATURE}")
COMMAND+=(--categorical-margin "${WSG_CATEGORICAL_MARGIN}")
COMMAND+=(--categorical-num-clusters "${WSG_CATEGORICAL_NUM_CLUSTERS}")
COMMAND+=(--categorical-kmeans-iterations "${WSG_CATEGORICAL_KMEANS_ITERATIONS}")
COMMAND+=(--high-noise-anchor "${WSG_HIGH_NOISE_ANCHOR}")
if [[ "${WSG_MULTIRES_NOISE}" == "1" ]]; then COMMAND+=(--multires-noise); fi
if [[ "${WSG_SEPARATE_CONDITION_ADAPTERS}" == "1" ]]; then COMMAND+=(--separate-condition-adapters); fi
if [[ "${WSG_AUDIO_GLOBAL_CONDITION}" == "1" ]]; then COMMAND+=(--audio-global-condition); fi
COMMAND+=(--edit-loss-weight "${WSG_EDIT_LOSS_WEIGHT}")
COMMAND+=(--edit-loss-power "${WSG_EDIT_LOSS_POWER}")
if [[ -n "${WSG_CHANNEL_HEAD_ALPHA}" ]]; then
  COMMAND+=(--channel-head-alpha "${WSG_CHANNEL_HEAD_ALPHA}")
fi
if [[ -n "${WSG_PREVIEW_SAMPLE_ID}" ]]; then
  COMMAND+=(--preview-sample-id "${WSG_PREVIEW_SAMPLE_ID}")
fi
if [[ -n "${WSG_PREVIEW_VIDEO_ID}" ]]; then
  COMMAND+=(--preview-video-id "${WSG_PREVIEW_VIDEO_ID}")
fi
if [[ -n "${WSG_PREVIEW_FRAME_INDEX}" ]]; then
  COMMAND+=(--preview-frame-index "${WSG_PREVIEW_FRAME_INDEX}")
fi
if [[ -n "${WSG_PREVIEW_FRAME_FRACTION}" ]]; then
  COMMAND+=(--preview-frame-fraction "${WSG_PREVIEW_FRAME_FRACTION}")
fi
COMMAND+=(--preview-split "${WSG_PREVIEW_SPLIT}")
COMMAND+=(--preview-num-samples "${WSG_PREVIEW_NUM_SAMPLES}")
if [[ "${WSG_USE_XPOSE_CACHE}" == "1" ]]; then
  COMMAND+=(--xpose-latent-cache-dir "${WSG_XPOSE_LATENT_CACHE_DIR}")
fi
if [[ "${SAVE_OPTIMIZER_STATE_8BIT}" == "1" ]]; then
  COMMAND+=(--save-optimizer-state-8bit)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  COMMAND+=(--gradient-checkpointing)
fi
if [[ "${DISABLE_PREFLIGHT}" == "1" ]]; then
  COMMAND+=(--disable-preflight)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  COMMAND+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${INFERENCE_ONLY}" == "1" ]]; then
  COMMAND+=(
    --inference-only
    --inference-checkpoint "${INFERENCE_CHECKPOINT}"
    --inference-label "${INFERENCE_LABEL}"
    --inference-steps "${INFERENCE_STEPS}"
  )
fi

printf -v COMMAND_STRING '%q ' "${COMMAND[@]}"
echo "[WSM_v1] Channel_Head=${WSG_CHANNEL_HEAD_INDEX} condition=${CONDITION_MODALITIES} target=${TARGET_MODALITY}"
echo "[WSM_v1] resolved_config=${RESOLVED_CONFIG}"
echo "[WSM_v1] command=${COMMAND_STRING}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

require_path "${TRAIN_ENTRY}" "TunnelingNet FLUX.2 training entry"

if [[ "${RUN_IN_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  # Pass every per-run variable explicitly into the tmux command string. tmux
  # sessions can share a server whose environment was inherited from whichever
  # run created the server first; relying on inheritance leaks one run's GPU_IDS
  # / Channel_Head index / alpha into later runs and collapses parallel runs onto a
  # single GPU. CUDA_VISIBLE_DEVICES is reasserted from GPU_IDS inside the child.
  env -u LD_LIBRARY_PATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd ${PROJECT_ROOT} && \
     CUDA_VISIBLE_DEVICES=${GPU_IDS} \
     GPU_IDS=${GPU_IDS} \
     NUM_PROCESSES=${NUM_PROCESSES} \
     MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT} \
     WSG_CHANNEL_HEAD_INDEX=${WSG_CHANNEL_HEAD_INDEX} \
     WSG_CHANNEL_HEAD_RANK=${WSG_CHANNEL_HEAD_RANK} \
     WSG_CHANNEL_HEAD_ALPHA=${WSG_CHANNEL_HEAD_ALPHA:-} \
     WSG_DENSE_LATENT_CACHE_DIR=${WSG_DENSE_LATENT_CACHE_DIR} \
     WSG_XPOSE_LATENT_CACHE_DIR=${WSG_XPOSE_LATENT_CACHE_DIR} \
     WSG_USE_XPOSE_CACHE=${WSG_USE_XPOSE_CACHE} \
     WSG_TEXT_EMBED_CACHE_DIR=${WSG_TEXT_EMBED_CACHE_DIR} \
     WSG_TARGET_DATASET_WHITELIST=${WSG_TARGET_DATASET_WHITELIST:-} \
     WSG_AUDIO_DENSE_CONDITION_TRAIN=${WSG_AUDIO_DENSE_CONDITION_TRAIN:-} \
     WSG_AUDIO_DENSE_CONDITION_VAL=${WSG_AUDIO_DENSE_CONDITION_VAL:-} \
     WSG_AUDIO_CONDITION_DIM=${WSG_AUDIO_CONDITION_DIM:-0} \
     WSG_AUDIO_CONDITIONING_MODE=${WSG_AUDIO_CONDITIONING_MODE:-attention} \
     MAX_STEPS=${MAX_STEPS} \
     DISABLE_PREFLIGHT=${DISABLE_PREFLIGHT} \
     TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
     WSG_PREVIEW_SAMPLE_ID=${WSG_PREVIEW_SAMPLE_ID:-} \
     WSG_PREVIEW_VIDEO_ID=${WSG_PREVIEW_VIDEO_ID:-} \
     WSG_PREVIEW_FRAME_INDEX=${WSG_PREVIEW_FRAME_INDEX:-} \
     WSG_PREVIEW_FRAME_FRACTION=${WSG_PREVIEW_FRAME_FRACTION:-0.75} \
     WSG_PREVIEW_NUM_SAMPLES=${WSG_PREVIEW_NUM_SAMPLES:-3} \
     WSG_FLUX_CONDITION_ATTENTION_MODE=${WSG_FLUX_CONDITION_ATTENTION_MODE:-auto} \
     WSG_ATTENTION_ESTIMATION=${WSG_ATTENTION_ESTIMATION:-auto} \
     WSG_ATTENTION_CONTROL=${WSG_ATTENTION_CONTROL:-auto} \
     WSG_ATTENTION_INVERSE_RENDERING=${WSG_ATTENTION_INVERSE_RENDERING:-auto} \
     WSG_ATTENTION_NARROWGEN=${WSG_ATTENTION_NARROWGEN:-auto} \
     WSG_TRAINABLE_TARGET=${WSG_TRAINABLE_TARGET:-auto} \
     WSG_TRAINABLE_CONDITION=${WSG_TRAINABLE_CONDITION:-auto} \
     WSG_DENSE_ENHANCE=${WSG_DENSE_ENHANCE:-auto} \
     WSG_DE_TIMESTEP=${WSG_DE_TIMESTEP:-auto} \
     WSG_DE_MINSNR=${WSG_DE_MINSNR:-auto} \
     WSG_DE_HUBER=${WSG_DE_HUBER:-auto} \
     WSG_SINGLE_STEP_X0=${WSG_SINGLE_STEP_X0:-0} \
     WSG_GRADIENT_DOMAIN_LOSS_WEIGHT=${WSG_GRADIENT_DOMAIN_LOSS_WEIGHT:-0} \
     WSG_FOCAL_REGRESSION_GAMMA=${WSG_FOCAL_REGRESSION_GAMMA:-0} \
     WSG_FOREGROUND_IOU_LOSS_WEIGHT=${WSG_FOREGROUND_IOU_LOSS_WEIGHT:-0} \
     WSG_CATEGORICAL_LOSS_MODE=${WSG_CATEGORICAL_LOSS_MODE:-off} \
     WSG_CATEGORICAL_LOSS_WEIGHT=${WSG_CATEGORICAL_LOSS_WEIGHT:-0.5} \
     WSG_CATEGORICAL_TEMPERATURE=${WSG_CATEGORICAL_TEMPERATURE:-0.25} \
     WSG_CATEGORICAL_MARGIN=${WSG_CATEGORICAL_MARGIN:-0.25} \
     WSG_CATEGORICAL_NUM_CLUSTERS=${WSG_CATEGORICAL_NUM_CLUSTERS:-8} \
     WSG_CATEGORICAL_KMEANS_ITERATIONS=${WSG_CATEGORICAL_KMEANS_ITERATIONS:-4} \
     WSG_HIGH_NOISE_ANCHOR=${WSG_HIGH_NOISE_ANCHOR:-auto} \
     WSG_MULTIRES_NOISE=${WSG_MULTIRES_NOISE:-0} \
     WSG_HISTORICAL_REFERENCE_POLICY=${WSG_HISTORICAL_REFERENCE_POLICY:-tolerant} \
     MODALITY_CONFIG=${MODALITY_CONFIG} \
     FLUX2_MODEL_PATH=${FLUX2_MODEL_PATH} \
     TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT} \
     RUN_IN_TMUX=0 \
     RUN_ID=${RUN_ID} \
     RUN_ROOT=${RUN_ROOT} \
     LAUNCH_LOG=${LAUNCH_LOG} \
     INFERENCE_ONLY=${INFERENCE_ONLY} \
     INFERENCE_STEPS=${INFERENCE_STEPS} \
     INFERENCE_CHECKPOINT=${INFERENCE_CHECKPOINT:-} \
     INFERENCE_LABEL=${INFERENCE_LABEL:-} \
     RESUME_FROM_LATEST=${RESUME_FROM_LATEST:-} \
     RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-} \
     bash scripts/launch_scripts/train_dense_structure_models.sh"
  echo "[WSM_v1] started tmux session ${TMUX_SESSION}"
  echo "[WSM_v1] log=${LAUNCH_LOG}"
  exit 0
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
exec "${COMMAND[@]}"
