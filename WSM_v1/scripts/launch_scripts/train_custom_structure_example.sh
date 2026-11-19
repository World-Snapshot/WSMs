#!/usr/bin/env bash
set -euo pipefail

# Train the custom-pilot tasks declared in configs/wsg_v1_modalities_custom.json.
#
# Head 0  (estimation)                : MP-100 RGB -> Universal_Pose
# Head 1  (estimation)                : MAGICK  RGB (0-2) -> Transparency/alpha (3)
# Head 2  (conditional_control+narrow): IC-Light illumination (3-5) + caption
#                                       + Reference_IMG (source 0-2) -> relit RGB (6-8)
#
# Usage from WSM_v1:
#   bash scripts/launch_scripts/train_custom_structure_example.sh            # head0 (default)
#   bash scripts/launch_scripts/train_custom_structure_example.sh head1      # transparency (1 GPU)
#   bash scripts/launch_scripts/train_custom_structure_example.sh head2      # relighting  (1 GPU)
#   bash scripts/launch_scripts/train_custom_structure_example.sh all        # head1 then head2, sequential
#
# GPU selection (single GPU per head by default; one card is enough to train a head):
#   head0 -> GPU 0 (GPU_ID, default 0).
#   head1 -> GPU 1, head2 -> GPU 2 (GPU_IDS, default; NUM_PROCESSES derived from it).
#   Pass GPU_IDS=<id> to override (e.g. resume head2 on a free card while GPU 2 is busy).
#   A multi-GPU list (GPU_IDS=0,1,2,3) still works and runs DDP, but is optional, not required.
# The launcher reads each head's condition/target modalities straight from the JSON by
# WSG_CHANNEL_HEAD_INDEX, so caption / Reference_IMG routing follows the config automatically.

SCRIPT_PATH="$(readlink -f "$0")"
PROJECT_ROOT="${PROJECT_ROOT:-/research/cbim/vast/sf895/code/WSMs/WSM_v1}"
MODE="${1:-head0}"
COMMON_CONFIG="configs/wsg_v1_modalities_custom.json"
TRAIN_LAUNCHER="scripts/launch_scripts/train_dense_structure_models.sh"
cd "${PROJECT_ROOT}"

# Count GPUs in a comma-separated GPU_IDS list -> NUM_PROCESSES.
count_gpus() { local ids="$1"; IFS=',' read -ra a <<< "${ids}"; echo "${#a[@]}"; }

launch_head0() {
  local gpu_id="${1:-0}"
  echo "[custom-example] Head 0: MP-100 RGB -> Universal_Pose on GPU ${gpu_id}"
  TRAIN_LOG_ROOT=train_log_custom \
  MODALITY_CONFIG="${COMMON_CONFIG}" \
  WSG_CHANNEL_HEAD_INDEX=0 \
  WSG_DENSE_LATENT_CACHE_DIR=data/mp100_universal_pose_flux2_instance_crop_latents \
  WSG_TEXT_EMBED_CACHE_DIR=data/mp100_universal_pose_flux2_text_embeddings \
  WSG_USE_XPOSE_CACHE=0 \
  GPU_IDS="${gpu_id}" NUM_PROCESSES=1 \
  WSG_PREVIEW_NUM_SAMPLES="${WSG_PREVIEW_NUM_SAMPLES:-7}" \
  MAX_STEPS="${HEAD0_MAX_STEPS:-2000}" \
  WSG_GRADIENT_DOMAIN_LOSS_WEIGHT="${HEAD0_GRADIENT_DOMAIN_LOSS_WEIGHT:-0.1}" \
  WSG_FOCAL_REGRESSION_GAMMA=0 WSG_CATEGORICAL_LOSS_MODE=off \
  RUN_ID_SUFFIX="${HEAD0_RUN_ID_SUFFIX:-mp100_universal_pose_gradient}" \
  RUN_IN_TMUX=1 TMUX_SESSION="${HEAD0_TMUX_SESSION:-custom_head000000_universal_pose}" \
  bash "${TRAIN_LAUNCHER}"
}

# Head 1 -- MAGICK transparency estimator: RGB (0-2) -> LayerDiffuse_Transparency.
# Estimation task: no caption, no Reference_IMG (launcher reads this from the JSON head).
# Modalities are self-describing named keys, so no numeric-channel collision with WSG.
launch_head1() {
  local gpu_ids="${1:-1}"
  local nproc; nproc="$(count_gpus "${gpu_ids}")"
  echo "[custom-example] Head 1: MAGICK RGB(0-2) -> LayerDiffuse_Transparency on GPUs ${gpu_ids} (nproc=${nproc})"
  TRAIN_LOG_ROOT=train_log_custom \
  MODALITY_CONFIG="${COMMON_CONFIG}" \
  WSG_CHANNEL_HEAD_INDEX=1 \
  WSG_DENSE_LATENT_CACHE_DIR="${HEAD1_LATENT_CACHE:-data/custom_pilots/magick_transparency_v1/cache_named}" \
  WSG_USE_XPOSE_CACHE=0 \
  GPU_IDS="${gpu_ids}" NUM_PROCESSES="${nproc}" \
  WSG_PREVIEW_NUM_SAMPLES="${WSG_PREVIEW_NUM_SAMPLES:-7}" \
  MAX_STEPS="${HEAD1_MAX_STEPS:-2000}" \
  WSG_CATEGORICAL_LOSS_MODE=off \
  RUN_ID_SUFFIX="${HEAD1_RUN_ID_SUFFIX:-magick_transparency}" \
  RUN_IN_TMUX="${RUN_IN_TMUX:-1}" TMUX_SESSION="${HEAD1_TMUX_SESSION:-custom_head000001_transparency}" \
  bash "${TRAIN_LAUNCHER}"
}

# Head 2 -- IC-Light relighting: ICLight_Lighting + source RGB (0-2) -> ICLight_Relighted_RGB.
# inverse_rendering + tolerant policy: the target is not 0-2, so the tolerant current-frame
# fallback feeds each sample's own source RGB (0-2) as a leakage-free per-sample reference --
# the same mechanism the MatSynth material-rerendering heads use. No AVIS first-frame cache needed.
# NO caption: matches the official inverse_rendering heads (cond=['3','0-2']). The IBL probe carries
# all per-sample lighting and the 0-2 reference carries object identity, so no text embed is needed.
launch_head2() {
  local gpu_ids="${1:-2}"
  local nproc; nproc="$(count_gpus "${gpu_ids}")"
  echo "[custom-example] Head 2: relighting (ICLight_Lighting + 0-2) -> ICLight_Relighted_RGB on GPUs ${gpu_ids} (nproc=${nproc})"
  TRAIN_LOG_ROOT=train_log_custom \
  MODALITY_CONFIG="${COMMON_CONFIG}" \
  WSG_CHANNEL_HEAD_INDEX=2 \
  WSG_DENSE_LATENT_CACHE_DIR="${HEAD2_LATENT_CACHE:-data/custom_pilots/iclight_relighting_v3/cache_named_v2}" \
  WSG_USE_XPOSE_CACHE=0 \
  WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY:-tolerant}" \
  GPU_IDS="${gpu_ids}" NUM_PROCESSES="${nproc}" \
  WSG_PREVIEW_NUM_SAMPLES="${WSG_PREVIEW_NUM_SAMPLES:-7}" \
  VIZ_EVERY_STEPS="${VIZ_EVERY_STEPS:-2000}" \
  EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}" \
  SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}" \
  MAX_STEPS="${HEAD2_MAX_STEPS:-3000}" \
  WSG_CATEGORICAL_LOSS_MODE=off \
  RUN_ID_SUFFIX="${HEAD2_RUN_ID_SUFFIX:-iclight_relighting}" \
  RUN_IN_TMUX="${RUN_IN_TMUX:-1}" TMUX_SESSION="${HEAD2_TMUX_SESSION:-custom_head000002_relighting}" \
  bash "${TRAIN_LAUNCHER}"
}

case "${MODE}" in
  head0) launch_head0 "${GPU_ID:-0}" ;;
  head1) launch_head1 "${GPU_IDS:-1}" ;;
  head2) launch_head2 "${GPU_IDS:-2}" ;;
  all)
    # Sequential, one GPU per head: transparency on GPU 1, then relighting on GPU 2.
    RUN_IN_TMUX=0 launch_head1 "${HEAD1_GPU_IDS:-1}"
    RUN_IN_TMUX=0 launch_head2 "${HEAD2_GPU_IDS:-2}"
    ;;
  help|-h|--help) sed -n '3,26p' "${SCRIPT_PATH}" ;;
  *) echo "Unknown mode: ${MODE}; expected head0 | head1 | head2 | all" >&2; exit 2 ;;
esac
