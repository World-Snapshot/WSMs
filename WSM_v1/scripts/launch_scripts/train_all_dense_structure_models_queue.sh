#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/research/cbim/vast/sf895/code/WSMs/WSM_v1}"
cd "${PROJECT_ROOT}"

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate WSM

# Queue launcher for large-scale dense Channel Head training.
#
# This script does not implement training itself. It reads the Channel Head JSON,
# filters trainable entries, and schedules one single-GPU invocation of
# train_dense_structure_models.sh per Channel Head.
#
# Defaults are intentionally conservative:
# - only JSON entries with pair_exists=true are launched;
# - existing trained Channel Heads are skipped if any ckpt/*/pytorch_lora_weights.safetensors
#   already exists under train_log/Channel_Head_xxxxxx;
# - concurrency equals the number of GPUs in GPU_IDS.
#
# To retrain a completed Channel Head, delete its train_log/Channel_Head_xxxxxx
# directory or set WSG_QUEUE_SKIP_EXISTING=0.

MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-scripts/launch_scripts/train_dense_structure_models.sh}"
PYTHON_BIN="${PYTHON_BIN:-/research/cbim/vast/sf895/miniforge3/envs/WSM/bin/python}"

GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "[tunnel-queue] ERROR: GPU_IDS is empty" >&2
  exit 1
fi
MAX_ACTIVE="${MAX_ACTIVE:-${#GPU_ARRAY[@]}}"
if (( MAX_ACTIVE < 1 || MAX_ACTIVE > ${#GPU_ARRAY[@]} )); then
  echo "[tunnel-queue] ERROR: MAX_ACTIVE=${MAX_ACTIVE} must be in [1, ${#GPU_ARRAY[@]}]" >&2
  exit 1
fi

QUEUE_RUN_ID="${QUEUE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-train_log/queue_logs}"
QUEUE_LOG="${QUEUE_LOG:-${QUEUE_LOG_DIR}/dense_structure_queue_${QUEUE_RUN_ID}.log}"
mkdir -p "${QUEUE_LOG_DIR}"

PAIR_EXISTS_ONLY="${PAIR_EXISTS_ONLY:-1}"
TASK_TYPES="${TASK_TYPES:-}"
# Task types listed here are launched before the rest, preserving Channel Head
# index order inside each priority bucket. Empty disables priority ordering.
PRIORITY_TASK_TYPES="${PRIORITY_TASK_TYPES:-estimation_or_reconstruction,conditional_control+narrow_generation}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"
CHANNEL_HEAD_LIST="${CHANNEL_HEAD_LIST:-${WSG_CHANNEL_HEAD_LIST:-}}"
WSG_QUEUE_SKIP_EXISTING="${WSG_QUEUE_SKIP_EXISTING:-1}"
WSG_QUEUE_SKIP_MODE="${WSG_QUEUE_SKIP_MODE:-ckpt}"  # ckpt | dir | none
WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY:-tolerant}"
WSG_QUEUE_DRY_RUN="${WSG_QUEUE_DRY_RUN:-0}"
WSG_QUEUE_MODE="${WSG_QUEUE_MODE:-run}"  # run | plan
WSG_QUEUE_PLAN_OUTPUT="${WSG_QUEUE_PLAN_OUTPUT:-}"
WSG_QUEUE_RECHECK_NO_PAIR_MARKERS="${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS:-1}"
WSG_QUEUE_SELF_HEAL_ONLY="${WSG_QUEUE_SELF_HEAL_ONLY:-0}"
WSG_PAIR_AUDIT_SCRIPT="${WSG_PAIR_AUDIT_SCRIPT:-manual_processing/flux2_preprocessing/audit_dense_channel_head_pairs.py}"
WSG_PAIR_AUDIT_CACHE_DIR="${WSG_PAIR_AUDIT_CACHE_DIR:-data/wsg_flux2_frame_latents_dense_multidataset}"
WSG_PAIR_AUDIT_XPOSE_CACHE_DIR="${WSG_PAIR_AUDIT_XPOSE_CACHE_DIR:-data/xpose_flux2_frame_latents_sample10pct_2frames}"
WSG_PAIR_AUDIT_REPORT="${WSG_PAIR_AUDIT_REPORT:-${QUEUE_LOG_DIR}/dense_pair_recheck_${QUEUE_RUN_ID}.json}"
POLL_SECONDS="${POLL_SECONDS:-30}"

# Per-task training length overrides. Entries not listed here use MAX_STEPS
# from the environment, or the single-task launcher default of 1000.
#
# Example:
#   TASK_TYPE_MAX_STEPS_JSON='{"inverse_rendering":700,"conditional_control":700}' bash scripts/launch_scripts/train_all_dense_structure_models_queue.sh
TASK_TYPE_MAX_STEPS_JSON="${TASK_TYPE_MAX_STEPS_JSON:-{\"inverse_rendering\":700,\"conditional_control\":700\}}"
DEFAULT_MAX_STEPS="${MAX_STEPS:-1000}"

# Target-specific recipe overrides are applied after the four task-type
# defaults above. Disable this layer to recover the legacy queue behavior.
WSG_QUEUE_TARGET_RECIPE_OVERRIDES="${WSG_QUEUE_TARGET_RECIPE_OVERRIDES:-1}"
WSG_QUEUE_PROTOTYPE_CE_WEIGHT="${WSG_QUEUE_PROTOTYPE_CE_WEIGHT:-0.5}"
WSG_QUEUE_GRADIENT_DOMAIN_LOSS_WEIGHT="${WSG_QUEUE_GRADIENT_DOMAIN_LOSS_WEIGHT:-0.1}"

case "${WSG_QUEUE_MODE}" in
  run) ;;
  plan)
    # Planning must never remove stale marker directories or launch workers.
    WSG_QUEUE_DRY_RUN=1
    ;;
  *)
    echo "[tunnel-queue] ERROR: unknown WSG_QUEUE_MODE=${WSG_QUEUE_MODE}" >&2
    exit 2
    ;;
esac

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[tunnel-queue] ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_path "${MODALITY_CONFIG}" "modality config"
require_path "${TRAIN_LAUNCHER}" "single Channel Head launcher"

mapfile -t CANDIDATE_HEADS < <(
  "${PYTHON_BIN}" - "${MODALITY_CONFIG}" "${PAIR_EXISTS_ONLY}" "${TASK_TYPES}" "${PRIORITY_TASK_TYPES}" "${START_INDEX}" "${END_INDEX}" "${CHANNEL_HEAD_LIST}" "${TASK_TYPE_MAX_STEPS_JSON}" "${DEFAULT_MAX_STEPS}" "${WSG_QUEUE_TARGET_RECIPE_OVERRIDES}" "${WSG_QUEUE_PROTOTYPE_CE_WEIGHT}" "${WSG_QUEUE_GRADIENT_DOMAIN_LOSS_WEIGHT}" <<'PY'
import json
import sys

(
    path, pair_only_raw, task_types_raw, priority_task_types_raw, start_raw, end_raw,
    explicit_raw, steps_json_raw, default_steps_raw, target_overrides_raw,
    prototype_ce_weight_raw, gradient_weight_raw,
) = sys.argv[1:]
pair_only = str(pair_only_raw).strip().lower() in {"1", "true", "yes", "on"}
target_overrides = str(target_overrides_raw).strip().lower() in {"1", "true", "yes", "on"}
task_types = {item.strip() for item in task_types_raw.split(",") if item.strip()}
priority_task_types = [item.strip() for item in priority_task_types_raw.split(",") if item.strip()]
priority_rank = {task_type: rank for rank, task_type in enumerate(priority_task_types)}
start = int(start_raw) if str(start_raw).strip() else None
end = int(end_raw) if str(end_raw).strip() else None
default_steps = int(default_steps_raw)
explicit = None
if str(explicit_raw).strip():
    explicit = {int(item.strip()) for item in explicit_raw.split(",") if item.strip()}
try:
    task_step_overrides = json.loads(steps_json_raw) if str(steps_json_raw).strip() else {}
except json.JSONDecodeError as exc:
    raise SystemExit(f"Invalid TASK_TYPE_MAX_STEPS_JSON: {exc}") from exc
task_step_overrides = {str(k): int(v) for k, v in task_step_overrides.items()}
prototype_ce_weight = float(prototype_ce_weight_raw)
gradient_weight = float(gradient_weight_raw)

pose_map_targets = {"67-69", "90-92", "99-101", "Universal_Pose"}
spatial_targets = {"32-33", "34-35", "36-38", "39", "40", "41-42", "103"}

with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
entries = payload.get("wsg_v1", payload)

records = []
for key, entry in entries.items():
    if not key.startswith("Channel_Head_"):
        continue
    try:
        index = int(key.rsplit("_", 1)[1])
    except ValueError:
        continue
    if explicit is not None and index not in explicit:
        continue
    if start is not None and index < start:
        continue
    if end is not None and index > end:
        continue
    if pair_only and not bool(entry.get("pair_exists")):
        continue
    task_type = str(entry.get("task_type") or "")
    if task_types and task_type not in task_types:
        continue
    max_steps = task_step_overrides.get(task_type, default_steps)
    target = str(entry.get("target") or "")
    recipe = "task-default"
    categorical_mode = "inherit"
    categorical_weight = "inherit"
    gradient_loss_weight = "inherit"
    if target_overrides:
        if target == "195":
            max_steps = 1000
            recipe = "target-195-prototype-ce"
            categorical_mode = "prototype_ce"
            categorical_weight = f"{prototype_ce_weight:g}"
            gradient_loss_weight = "0"
        elif target in pose_map_targets:
            max_steps = 2000
            recipe = "pose-map-gradient"
            categorical_mode = "off"
            gradient_loss_weight = f"{gradient_weight:g}"
        elif target in spatial_targets:
            max_steps = 2000
            recipe = "spatial-target-2000"
            categorical_mode = "off"
            gradient_loss_weight = "0"
    records.append((index, task_type, max_steps, recipe, categorical_mode, categorical_weight, gradient_loss_weight))

def sort_key(record):
    index, task_type = record[:2]
    return (priority_rank.get(task_type, len(priority_rank)), index)

for record in sorted(records, key=sort_key):
    print("\t".join(str(value) for value in record))
PY
)

has_checkpoint_artifact() {
  local head_dir="$1"
  [[ -d "${head_dir}" ]] || return 1
  [[ -n "$(find "${head_dir}" -path "*/ckpt/*/pytorch_lora_weights.safetensors" -type f -print -quit 2>/dev/null)" ]]
}

has_no_pair_marker() {
  local head_dir="$1"
  [[ -d "${head_dir}" ]] || return 1
  [[ -n "$(find "${head_dir}" -name "NO_TRAINING_NO_PAIRS.txt" -type f -print -quit 2>/dev/null)" ]]
}

declare -A CANDIDATE_HEAD_SET=()
declare -A SELF_HEALED_HEAD_SET=()
for record in "${CANDIDATE_HEADS[@]}"; do
  IFS=$'\t' read -r index _task_type _max_steps <<< "${record}"
  CANDIDATE_HEAD_SET["${index}"]=1
done

self_heal_no_pair_markers() {
  [[ "${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS}" == "1" ]] || return 0

  local marker_count=0 index head_dir
  for index in "${!CANDIDATE_HEAD_SET[@]}"; do
    printf -v head_dir "train_log/Channel_Head_%06d" "${index}"
    if has_no_pair_marker "${head_dir}" && ! has_checkpoint_artifact "${head_dir}"; then
      marker_count=$((marker_count + 1))
    fi
  done
  (( marker_count > 0 )) || return 0

  if [[ ! -f "${WSG_PAIR_AUDIT_SCRIPT}" ]]; then
    echo "[tunnel-queue] WARNING: no-pair self-heal unavailable; missing audit script ${WSG_PAIR_AUDIT_SCRIPT}; using legacy skip logic" >&2
    return 0
  fi

  local marker_file
  if ! marker_file="$(mktemp "${QUEUE_LOG_DIR}/no_pair_markers.XXXXXX")"; then
    echo "[tunnel-queue] WARNING: cannot create marker audit list; using legacy skip logic" >&2
    return 0
  fi
  for index in "${!CANDIDATE_HEAD_SET[@]}"; do
    printf -v head_dir "train_log/Channel_Head_%06d" "${index}"
    if has_no_pair_marker "${head_dir}" && ! has_checkpoint_artifact "${head_dir}"; then
      printf 'channel_head_index=%s\n' "${index}" >> "${marker_file}"
    fi
  done

  echo "[tunnel-queue] rechecking ${marker_count} no-pair marker(s) with current cache/config"
  if ! "${PYTHON_BIN}" "${WSG_PAIR_AUDIT_SCRIPT}" \
    --config "${MODALITY_CONFIG}" \
    --cache-dir "${WSG_PAIR_AUDIT_CACHE_DIR}" \
    --xpose-cache-dir "${WSG_PAIR_AUDIT_XPOSE_CACHE_DIR}" \
    --historical-reference-policy "${WSG_HISTORICAL_REFERENCE_POLICY}" \
    --head-indices-file "${marker_file}" \
    --output "${WSG_PAIR_AUDIT_REPORT}"; then
    echo "[tunnel-queue] WARNING: no-pair audit failed; preserving markers and using legacy skip logic" >&2
    rm -f -- "${marker_file}" || true
    return 0
  fi

  local recoverable_file
  if ! recoverable_file="$(mktemp "${QUEUE_LOG_DIR}/recoverable_heads.XXXXXX")"; then
    echo "[tunnel-queue] WARNING: cannot create self-heal scratch file; using legacy skip logic" >&2
    rm -f -- "${marker_file}" || true
    return 0
  fi
  if ! "${PYTHON_BIN}" - "${WSG_PAIR_AUDIT_REPORT}" > "${recoverable_file}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    report = json.load(handle)
for row in report.get("rows", []):
    if row.get("pair_exists") and int((row.get("runtime_splits") or {}).get("train", 0)) > 0:
        print(int(row["index"]))
PY
  then
    echo "[tunnel-queue] WARNING: cannot parse no-pair audit; preserving markers and using legacy skip logic" >&2
    rm -f -- "${recoverable_file}" "${marker_file}" || true
    return 0
  fi

  local recovered=0
  while IFS= read -r index; do
    [[ -n "${index}" && -n "${CANDIDATE_HEAD_SET[${index}]:-}" ]] || continue
    printf -v head_dir "train_log/Channel_Head_%06d" "${index}"
    has_no_pair_marker "${head_dir}" || continue
    has_checkpoint_artifact "${head_dir}" && continue
    if [[ "${WSG_QUEUE_DRY_RUN}" == "1" ]]; then
      SELF_HEALED_HEAD_SET["${index}"]=1
      printf '[tunnel-queue] would self-heal Channel_Head_%06d: current runtime train pairs exist\n' "${index}"
      recovered=$((recovered + 1))
    elif rm -rf -- "${head_dir}"; then
      SELF_HEALED_HEAD_SET["${index}"]=1
      printf '[tunnel-queue] self-healed Channel_Head_%06d: removed stale no-pair directory\n' "${index}"
      recovered=$((recovered + 1))
    else
      echo "[tunnel-queue] WARNING: failed to remove ${head_dir}; preserving legacy skip behavior" >&2
    fi
  done < "${recoverable_file}"
  rm -f -- "${recoverable_file}" "${marker_file}" || true
  echo "[tunnel-queue] no-pair recheck markers=${marker_count} recovered=${recovered} report=${WSG_PAIR_AUDIT_REPORT}"
  return 0
}

if ! self_heal_no_pair_markers; then
  echo "[tunnel-queue] WARNING: no-pair self-heal failed unexpectedly; using legacy skip logic" >&2
fi
if [[ "${WSG_QUEUE_SELF_HEAL_ONLY}" == "1" ]]; then
  exit 0
fi

already_trained() {
  local index="$1"
  local head_dir
  printf -v head_dir "train_log/Channel_Head_%06d" "${index}"
  if [[ "${WSG_QUEUE_SKIP_EXISTING}" != "1" || "${WSG_QUEUE_SKIP_MODE}" == "none" ]]; then
    return 1
  fi
  if [[ "${WSG_QUEUE_SKIP_MODE}" == "dir" ]]; then
    [[ -d "${head_dir}" ]]
    return
  fi
  if has_checkpoint_artifact "${head_dir}"; then
    return 0
  fi
  [[ -n "${SELF_HEALED_HEAD_SET[${index}]:-}" ]] && return 1
  has_no_pair_marker "${head_dir}"
}

session_alive() {
  local session="$1"
  tmux has-session -t "${session}" 2>/dev/null
}

declare -a TODO_HEADS=()
declare -a TODO_TASK_TYPES=()
declare -a TODO_MAX_STEPS=()
declare -a TODO_RECIPES=()
declare -a TODO_CATEGORICAL_MODES=()
declare -a TODO_CATEGORICAL_WEIGHTS=()
declare -a TODO_GRADIENT_WEIGHTS=()
skipped=0
for record in "${CANDIDATE_HEADS[@]}"; do
  IFS=$'	' read -r index task_type max_steps recipe categorical_mode categorical_weight gradient_weight <<< "${record}"
  if already_trained "${index}"; then
    printf '[tunnel-queue] skip Channel_Head_%06d: existing training artifact found\n' "${index}"
    skipped=$((skipped + 1))
  else
    TODO_HEADS+=("${index}")
    TODO_TASK_TYPES+=("${task_type}")
    TODO_MAX_STEPS+=("${max_steps}")
    TODO_RECIPES+=("${recipe}")
    TODO_CATEGORICAL_MODES+=("${categorical_mode}")
    TODO_CATEGORICAL_WEIGHTS+=("${categorical_weight}")
    TODO_GRADIENT_WEIGHTS+=("${gradient_weight}")
  fi
done

{
  echo "[tunnel-queue] run_id=${QUEUE_RUN_ID}"
  echo "[tunnel-queue] modality_config=${MODALITY_CONFIG}"
  echo "[tunnel-queue] launcher=${TRAIN_LAUNCHER}"
  echo "[tunnel-queue] gpu_ids=${GPU_IDS}"
  echo "[tunnel-queue] max_active=${MAX_ACTIVE}"
  echo "[tunnel-queue] pair_exists_only=${PAIR_EXISTS_ONLY}"
  echo "[tunnel-queue] task_types=${TASK_TYPES:-<all>}"
  echo "[tunnel-queue] priority_task_types=${PRIORITY_TASK_TYPES:-<none>}"
  echo "[tunnel-queue] default_max_steps=${DEFAULT_MAX_STEPS}"
  echo "[tunnel-queue] task_type_max_steps_json=${TASK_TYPE_MAX_STEPS_JSON}"
  echo "[tunnel-queue] target_recipe_overrides=${WSG_QUEUE_TARGET_RECIPE_OVERRIDES}"
  echo "[tunnel-queue] prototype_ce_weight=${WSG_QUEUE_PROTOTYPE_CE_WEIGHT}"
  echo "[tunnel-queue] gradient_domain_loss_weight=${WSG_QUEUE_GRADIENT_DOMAIN_LOSS_WEIGHT}"
  echo "[tunnel-queue] start_index=${START_INDEX:-<none>} end_index=${END_INDEX:-<none>}"
  echo "[tunnel-queue] candidates=${#CANDIDATE_HEADS[@]} skipped=${skipped} queued=${#TODO_HEADS[@]}"
  echo "[tunnel-queue] skip_mode=${WSG_QUEUE_SKIP_MODE} skip_existing=${WSG_QUEUE_SKIP_EXISTING}"
  echo "[tunnel-queue] recheck_no_pair_markers=${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS}"
  echo "[tunnel-queue] historical_reference_policy=${WSG_HISTORICAL_REFERENCE_POLICY}"
  echo "[tunnel-queue] mode=${WSG_QUEUE_MODE}"
  echo "[tunnel-queue] log=${QUEUE_LOG}"
} | tee -a "${QUEUE_LOG}"

emit_plan() {
  local i
  if [[ -n "${WSG_QUEUE_PLAN_OUTPUT}" ]]; then
    local output_dir temp_output
    output_dir="$(dirname "${WSG_QUEUE_PLAN_OUTPUT}")"
    mkdir -p "${output_dir}"
    temp_output="${WSG_QUEUE_PLAN_OUTPUT}.tmp.$$"
    : > "${temp_output}"
    for i in "${!TODO_HEADS[@]}"; do
      printf '%s\t%s\t%s\n' "${TODO_HEADS[$i]}" "${TODO_TASK_TYPES[$i]}" "${TODO_MAX_STEPS[$i]}" >> "${temp_output}"
    done
    mv "${temp_output}" "${WSG_QUEUE_PLAN_OUTPUT}"
    echo "[tunnel-queue] wrote plan=${WSG_QUEUE_PLAN_OUTPUT} entries=${#TODO_HEADS[@]}" | tee -a "${QUEUE_LOG}"
  else
    for i in "${!TODO_HEADS[@]}"; do
      printf '%s\t%s\t%s\n' "${TODO_HEADS[$i]}" "${TODO_TASK_TYPES[$i]}" "${TODO_MAX_STEPS[$i]}"
    done
  fi
}

if [[ "${WSG_QUEUE_MODE}" == "plan" ]]; then
  emit_plan
  exit 0
fi

if [[ "${WSG_QUEUE_DRY_RUN}" == "1" ]]; then
  for i in "${!TODO_HEADS[@]}"; do
    printf '[tunnel-queue] would launch Channel_Head_%06d task_type=%s max_steps=%s recipe=%s categorical=%s gradient=%s\n' \
      "${TODO_HEADS[$i]}" "${TODO_TASK_TYPES[$i]}" "${TODO_MAX_STEPS[$i]}" "${TODO_RECIPES[$i]}" \
      "${TODO_CATEGORICAL_MODES[$i]}" "${TODO_GRADIENT_WEIGHTS[$i]}"
  done | tee -a "${QUEUE_LOG}"
  exit 0
fi

declare -a ACTIVE_SESSIONS=()
declare -a ACTIVE_HEADS=()
declare -a ACTIVE_SLOTS=()
next_idx=0
launched=0
finished=0

compact_finished() {
  local new_sessions=()
  local new_heads=()
  local new_slots=()
  for i in "${!ACTIVE_SESSIONS[@]}"; do
    local session="${ACTIVE_SESSIONS[$i]}"
    local head="${ACTIVE_HEADS[$i]}"
    local slot="${ACTIVE_SLOTS[$i]}"
    if session_alive "${session}"; then
      new_sessions+=("${session}")
      new_heads+=("${head}")
      new_slots+=("${slot}")
    else
      printf '[tunnel-queue] finished Channel_Head_%06d gpu=%s session=%s\n' "${head}" "${GPU_ARRAY[$slot]}" "${session}" | tee -a "${QUEUE_LOG}"
      finished=$((finished + 1))
    fi
  done
  ACTIVE_SESSIONS=("${new_sessions[@]}")
  ACTIVE_HEADS=("${new_heads[@]}")
  ACTIVE_SLOTS=("${new_slots[@]}")
}

next_free_slot() {
  local slot
  local active_slot
  local used
  for (( slot=0; slot<MAX_ACTIVE; slot++ )); do
    used=0
    for active_slot in "${ACTIVE_SLOTS[@]}"; do
      if [[ "${active_slot}" == "${slot}" ]]; then
        used=1
        break
      fi
    done
    if (( used == 0 )); then
      echo "${slot}"
      return 0
    fi
  done
  return 1
}

launch_one() {
  local index="$1"
  local slot="$2"
  local task_type="$3"
  local max_steps="$4"
  local recipe="$5"
  local categorical_mode="$6"
  local categorical_weight="$7"
  local gradient_weight="$8"
  local gpu="${GPU_ARRAY[$slot]}"
  local session
  printf -v session "tunnel_ch%06d_%s" "${index}" "${QUEUE_RUN_ID}"
  local port=$((29500 + (index % 1000) + slot * 1000))
  [[ "${categorical_mode}" == "inherit" ]] && categorical_mode="${WSG_CATEGORICAL_LOSS_MODE:-off}"
  [[ "${categorical_weight}" == "inherit" ]] && categorical_weight="${WSG_CATEGORICAL_LOSS_WEIGHT:-0.5}"
  [[ "${gradient_weight}" == "inherit" ]] && gradient_weight="${WSG_GRADIENT_DOMAIN_LOSS_WEIGHT:-0}"
  printf '[tunnel-queue] launch Channel_Head_%06d task_type=%s max_steps=%s recipe=%s categorical=%s gradient=%s gpu=%s port=%d session=%s\n' \
    "${index}" "${task_type}" "${max_steps}" "${recipe}" "${categorical_mode}" "${gradient_weight}" "${gpu}" "${port}" "${session}" | tee -a "${QUEUE_LOG}"
  GPU_IDS="${gpu}" \
  NUM_PROCESSES=1 \
  MAIN_PROCESS_PORT="${port}" \
  WSG_CHANNEL_HEAD_INDEX="${index}" \
  MAX_STEPS="${max_steps}" \
  WSG_CATEGORICAL_LOSS_MODE="${categorical_mode}" \
  WSG_CATEGORICAL_LOSS_WEIGHT="${categorical_weight}" \
  WSG_GRADIENT_DOMAIN_LOSS_WEIGHT="${gradient_weight}" \
  RUN_IN_TMUX=1 \
  TMUX= \
  TMUX_SESSION="${session}" \
  MODALITY_CONFIG="${MODALITY_CONFIG}" \
  WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY}" \
  RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-queue_${QUEUE_RUN_ID}}" \
  bash "${TRAIN_LAUNCHER}" | tee -a "${QUEUE_LOG}"
  ACTIVE_SESSIONS+=("${session}")
  ACTIVE_HEADS+=("${index}")
  ACTIVE_SLOTS+=("${slot}")
  launched=$((launched + 1))
}

while (( next_idx < ${#TODO_HEADS[@]} || ${#ACTIVE_SESSIONS[@]} > 0 )); do
  compact_finished
  while (( next_idx < ${#TODO_HEADS[@]} && ${#ACTIVE_SESSIONS[@]} < MAX_ACTIVE )); do
    slot="$(next_free_slot)"
    launch_one "${TODO_HEADS[$next_idx]}" "${slot}" "${TODO_TASK_TYPES[$next_idx]}" "${TODO_MAX_STEPS[$next_idx]}" \
      "${TODO_RECIPES[$next_idx]}" "${TODO_CATEGORICAL_MODES[$next_idx]}" \
      "${TODO_CATEGORICAL_WEIGHTS[$next_idx]}" "${TODO_GRADIENT_WEIGHTS[$next_idx]}"
    next_idx=$((next_idx + 1))
  done
  if (( next_idx < ${#TODO_HEADS[@]} || ${#ACTIVE_SESSIONS[@]} > 0 )); then
    active_desc=""
    for i in "${!ACTIVE_SESSIONS[@]}"; do
      active_desc+=$(printf ' Channel_Head_%06d:gpu%s:%s' "${ACTIVE_HEADS[$i]}" "${GPU_ARRAY[${ACTIVE_SLOTS[$i]}]}" "${ACTIVE_SESSIONS[$i]}")
    done
    echo "[tunnel-queue] active=${#ACTIVE_SESSIONS[@]} launched=${launched}/${#TODO_HEADS[@]} finished=${finished}${active_desc}" | tee -a "${QUEUE_LOG}"
    sleep "${POLL_SECONDS}"
  fi
done

echo "[tunnel-queue] done: candidates=${#CANDIDATE_HEADS[@]} skipped=${skipped} launched=${launched} finished=${finished}" | tee -a "${QUEUE_LOG}"
