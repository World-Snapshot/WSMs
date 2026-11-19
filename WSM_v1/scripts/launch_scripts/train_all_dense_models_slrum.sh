#!/usr/bin/env bash
set -euo pipefail

# Slurm keepalive/controller for dense Channel Head training.
#
# Default behavior:
# - controller submits up to 16 one-GPU worker jobs;
# - first 4 workers are submitted immediately;
# - remaining workers are spread over 5 minutes;
# - controller resubmits itself before the 8-hour cluster walltime;
# - workers call train_all_dense_structure_models_queue.sh for one Channel Head.
#
# The filename intentionally keeps the requested "slrum" spelling.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

QUEUE_SCRIPT="${QUEUE_SCRIPT:-${SCRIPT_DIR}/train_all_dense_structure_models_queue.sh}"
MODALITY_CONFIG="${MODALITY_CONFIG:-configs/wsg_v1_modalities_dense.json}"

SLRUM_MODE="${SLRUM_MODE:-controller}"  # controller | worker
SLRUM_RUN_ID="${SLRUM_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SLRUM_STATE_DIR="${SLRUM_STATE_DIR:-train_log/slrum_dense_queue}"
SLRUM_LOG_DIR="${SLRUM_LOG_DIR:-${SLRUM_STATE_DIR}/logs}"
SLRUM_STATE_FILE="${SLRUM_STATE_FILE:-${SLRUM_STATE_DIR}/state.tsv}"
SLRUM_LOCK_DIR="${SLRUM_LOCK_DIR:-${SLRUM_STATE_DIR}/lock}"
SLRUM_PLAN_FILE="${SLRUM_PLAN_FILE:-${SLRUM_STATE_DIR}/plan_${SLRUM_RUN_ID}.tsv}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-600}"
CLAIM_STALE_SECONDS="${CLAIM_STALE_SECONDS:-600}"

WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY:-tolerant}"
WSG_QUEUE_RECHECK_NO_PAIR_MARKERS="${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS:-1}"

MIN_GPU_MEMORY_MB="${MIN_GPU_MEMORY_MB:-24576}"
MAX_GPU_USED_MEMORY_MB="${MAX_GPU_USED_MEMORY_MB:-1024}"
MAX_ACTIVE_WORKERS="${MAX_ACTIVE_WORKERS:-16}"
MAX_TOTAL_SUBMISSIONS="${MAX_TOTAL_SUBMISSIONS:-0}"
INITIAL_SUBMISSIONS="${INITIAL_SUBMISSIONS:-4}"
RAMP_SUBMISSIONS="${RAMP_SUBMISSIONS:-12}"
RAMP_SECONDS="${RAMP_SECONDS:-300}"
CONTROLLER_POLL_SECONDS="${CONTROLLER_POLL_SECONDS:-30}"
RESTART_AFTER_SECONDS="${RESTART_AFTER_SECONDS:-25200}"  # 7 hours
SELF_RESUBMIT="${SELF_RESUBMIT:-1}"
SUBMIT_WORKERS="${SUBMIT_WORKERS:-1}"

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-07:55:00}"
SLURM_CONTROLLER_TIME="${SLURM_CONTROLLER_TIME:-07:50:00}"
SLURM_GPUS_PER_WORKER="${SLURM_GPUS_PER_WORKER:-1}"
SLURM_CPUS_PER_WORKER="${SLURM_CPUS_PER_WORKER:-8}"
SLURM_MEM_PER_WORKER="${SLURM_MEM_PER_WORKER:-64G}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"
SLURM_EXTRA_ARGS="${SLURM_EXTRA_ARGS:-}"

mkdir -p "${SLRUM_STATE_DIR}" "${SLRUM_LOG_DIR}"

log() {
  printf '[slrum-dense] %s\n' "$*" | tee -a "${SLRUM_LOG_DIR}/controller_${SLRUM_RUN_ID}.log"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[slrum-dense] ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

with_lock() {
  local waited=0
  local now mtime age
  until mkdir "${SLRUM_LOCK_DIR}" 2>/dev/null; do
    if [[ -d "${SLRUM_LOCK_DIR}" ]]; then
      now="$(date +%s)"
      mtime="$(stat -c %Y "${SLRUM_LOCK_DIR}" 2>/dev/null || echo "${now}")"
      age=$((now - mtime))
      if (( age > LOCK_STALE_SECONDS )); then
        echo "[slrum-dense] removing stale state lock age=${age}s path=${SLRUM_LOCK_DIR}" >&2
        rmdir "${SLRUM_LOCK_DIR}" 2>/dev/null || true
        continue
      fi
    fi
    sleep 1
    waited=$((waited + 1))
    if (( waited > 120 )); then
      echo "[slrum-dense] ERROR: timed out waiting for state lock ${SLRUM_LOCK_DIR}" >&2
      exit 1
    fi
  done
  local rc=0
  "$@" || rc=$?
  rmdir "${SLRUM_LOCK_DIR}" 2>/dev/null || true
  return "${rc}"
}

ensure_state_file() {
  if [[ ! -f "${SLRUM_STATE_FILE}" ]]; then
    printf 'timestamp\thead\tstatus\tjob_id\trun_id\ttask_type\tmax_steps\tnote\n' > "${SLRUM_STATE_FILE}"
  fi
}

append_state_unlocked() {
  local head="$1"
  local status="$2"
  local job_id="$3"
  local task_type="$4"
  local max_steps="$5"
  local note="${6:-}"
  ensure_state_file
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${head}" "${status}" "${job_id}" "${SLRUM_RUN_ID}" "${task_type}" "${max_steps}" "${note}" >> "${SLRUM_STATE_FILE}"
}

append_state() {
  with_lock append_state_unlocked "$@"
}


refresh_queue_plan() {
  WSG_QUEUE_MODE=plan \
  WSG_QUEUE_PLAN_OUTPUT="${SLRUM_PLAN_FILE}" \
  WSG_QUEUE_RECHECK_NO_PAIR_MARKERS=0 \
  QUEUE_RUN_ID="slrum_plan_${SLRUM_RUN_ID}" \
  QUEUE_LOG_DIR="${SLRUM_LOG_DIR}/queue_logs" \
  MODALITY_CONFIG="${MODALITY_CONFIG}" \
  WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY}" \
  bash "${QUEUE_SCRIPT}"
  log "refreshed queue plan path=${SLRUM_PLAN_FILE} entries=$(wc -l < "${SLRUM_PLAN_FILE}")"
}

generate_candidates() {
  [[ -f "${SLRUM_PLAN_FILE}" ]] || refresh_queue_plan >&2
  cat "${SLRUM_PLAN_FILE}"
}

head_has_checkpoint_artifact() {
  local head="$1"
  local head_dir
  printf -v head_dir "train_log/Channel_Head_%06d" "${head}"
  [[ -n "$(find "${head_dir}" -path "*/ckpt/*/pytorch_lora_weights.safetensors" -type f -print -quit 2>/dev/null)" ]]
}

latest_state_for_head() {
  local head="$1"
  [[ -f "${SLRUM_STATE_FILE}" ]] || return 1
  awk -F'\t' -v h="${head}" 'NR > 1 && $2 == h {line=$0} END {if (line) print line}' "${SLRUM_STATE_FILE}"
}

job_active() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "-" ]] || return 1
  command -v squeue >/dev/null 2>&1 || return 1
  [[ -n "$(squeue -h -j "${job_id}" -o '%i' 2>/dev/null)" ]]
}

refresh_state() {
  ensure_state_file
  local tmp
  tmp="$(mktemp "${SLRUM_STATE_DIR}/state_refresh.XXXXXX")"
  cp "${SLRUM_STATE_FILE}" "${tmp}"
  while IFS=$'\t' read -r _ts head status job_id _run_id task_type max_steps _note; do
    [[ "${head}" == "head" || -z "${head}" ]] && continue
    case "${status}" in
      claimed)
        local ts_epoch now_epoch claim_age
        ts_epoch="$(date -d "${_ts}" +%s 2>/dev/null || date +%s)"
        now_epoch="$(date +%s)"
        claim_age=$((now_epoch - ts_epoch))
        if (( claim_age > CLAIM_STALE_SECONDS )); then
          printf '%s\t%s\tclaim-expired\t%s\t%s\t%s\t%s\tclaim-age-%ss\n' "$(date -Is)" "${head}" "${job_id}" "${SLRUM_RUN_ID}" "${task_type}" "${max_steps}" "${claim_age}" >> "${tmp}"
        fi
        ;;
      submitted|running)
        if ! job_active "${job_id}"; then
          printf '%s\t%s\tstale\t%s\t%s\t%s\t%s\tjob-not-active\n' "$(date -Is)" "${head}" "${job_id}" "${SLRUM_RUN_ID}" "${task_type}" "${max_steps}" >> "${tmp}"
        fi
        ;;
    esac
  done < <(awk -F'\t' 'NR > 1 {state[$2]=$0} END {for (h in state) print state[h]}' "${SLRUM_STATE_FILE}")
  mv "${tmp}" "${SLRUM_STATE_FILE}"
}

head_is_blocked() {
  local head="$1"
  local line status
  line="$(latest_state_for_head "${head}" || true)"
  [[ -n "${line}" ]] || return 1
  IFS=$'\t' read -r _ts _head status _job_id _run_id _task_type _max_steps _note <<< "${line}"
  case "${status}" in
    completed|skipped)
      [[ "${_run_id}" == "${SLRUM_RUN_ID}" ]] && return 0
      return 1
      ;;
    claimed|submitted|running|stale|failed|gpu-rejected|finished-no-artifact)
      return 0
      ;;
  esac
  return 1
}

submitted_count() {
  [[ -f "${SLRUM_STATE_FILE}" ]] || { echo 0; return; }
  awk -F'\t' 'NR > 1 {state[$2]=$0} END {for (h in state) {split(state[h], f, "\t"); if (f[3] == "claimed" || f[3] == "submitted" || f[3] == "running" || f[3] == "stale" || f[3] == "failed" || f[3] == "gpu-rejected" || f[3] == "finished-no-artifact") count += 1} print count + 0}' "${SLRUM_STATE_FILE}"
}

active_count() {
  [[ -f "${SLRUM_STATE_FILE}" ]] || { echo 0; return; }
  local count=0
  while IFS=$'\t' read -r _ts _head status job_id _run_id _task_type _max_steps _note; do
    case "${status}" in
      submitted|running)
        if job_active "${job_id}"; then
          count=$((count + 1))
        fi
        ;;
    esac
  done < <(awk -F'\t' 'NR > 1 {state[$2]=$0} END {for (h in state) print state[h]}' "${SLRUM_STATE_FILE}")
  echo "${count}"
}

sbatch_common_args() {
  local args=()
  [[ -n "${SLURM_PARTITION}" ]] && args+=(--partition="${SLURM_PARTITION}")
  [[ -n "${SLURM_ACCOUNT}" ]] && args+=(--account="${SLURM_ACCOUNT}")
  [[ -n "${SLURM_QOS}" ]] && args+=(--qos="${SLURM_QOS}")
  [[ -n "${SLURM_CONSTRAINT}" ]] && args+=(--constraint="${SLURM_CONSTRAINT}")
  if [[ -n "${SLURM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    args+=(${SLURM_EXTRA_ARGS})
  fi
  printf '%s\n' "${args[@]}"
}

submit_worker() {
  local head="$1"
  local task_type="$2"
  local max_steps="$3"
  local job_id
  local output="${SLRUM_LOG_DIR}/worker_ch${head}_%j.log"
  local export_vars
  export_vars="ALL,SLRUM_MODE=worker,SLRUM_RUN_ID=${SLRUM_RUN_ID},SLRUM_STATE_DIR=${SLRUM_STATE_DIR},SLRUM_STATE_FILE=${SLRUM_STATE_FILE},SLRUM_LOG_DIR=${SLRUM_LOG_DIR},WSG_CHANNEL_HEAD_INDEX=${head},MAX_STEPS=${max_steps},TASK_TYPE=${task_type},PROJECT_ROOT=${PROJECT_ROOT},QUEUE_SCRIPT=${QUEUE_SCRIPT},MODALITY_CONFIG=${MODALITY_CONFIG},WSG_HISTORICAL_REFERENCE_POLICY=${WSG_HISTORICAL_REFERENCE_POLICY},WSG_QUEUE_RECHECK_NO_PAIR_MARKERS=${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS},MIN_GPU_MEMORY_MB=${MIN_GPU_MEMORY_MB},MAX_GPU_USED_MEMORY_MB=${MAX_GPU_USED_MEMORY_MB}"
  mapfile -t common_args < <(sbatch_common_args)
  local sbatch_rc=0
  set +e
  job_id="$(sbatch --parsable \
    --job-name="$(printf 'dch_%06d' "${head}")" \
    --output="${output}" \
    --time="${SLURM_TIME}" \
    --gres="gpu:${SLURM_GPUS_PER_WORKER}" \
    --cpus-per-task="${SLURM_CPUS_PER_WORKER}" \
    --mem="${SLURM_MEM_PER_WORKER}" \
    --export="${export_vars}" \
    "${common_args[@]}" \
    "${SCRIPT_PATH}")"
  sbatch_rc=$?
  set -e
  if (( sbatch_rc != 0 )); then
    append_state "${head}" "failed" "-" "${task_type}" "${max_steps}" "sbatch-exit-${sbatch_rc}"
    return "${sbatch_rc}"
  fi
  append_state "${head}" "submitted" "${job_id}" "${task_type}" "${max_steps}" "sbatch"
  log "submitted Channel_Head_$(printf '%06d' "${head}") job=${job_id} task_type=${task_type} max_steps=${max_steps}"
}

submit_controller_restart() {
  local output="${SLRUM_LOG_DIR}/controller_restart_%j.log"
  local export_vars
  export_vars="ALL,SLRUM_MODE=controller,SLRUM_RUN_ID=${SLRUM_RUN_ID},SLRUM_STATE_DIR=${SLRUM_STATE_DIR},SLRUM_STATE_FILE=${SLRUM_STATE_FILE},SLRUM_LOG_DIR=${SLRUM_LOG_DIR},PROJECT_ROOT=${PROJECT_ROOT},QUEUE_SCRIPT=${QUEUE_SCRIPT},MODALITY_CONFIG=${MODALITY_CONFIG},WSG_HISTORICAL_REFERENCE_POLICY=${WSG_HISTORICAL_REFERENCE_POLICY},WSG_QUEUE_RECHECK_NO_PAIR_MARKERS=${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS},SELF_RESUBMIT=${SELF_RESUBMIT}"
  mapfile -t common_args < <(sbatch_common_args)
  local job_id
  job_id="$(sbatch --parsable \
    --job-name="dch_controller" \
    --output="${output}" \
    --time="${SLURM_CONTROLLER_TIME}" \
    --export="${export_vars}" \
    "${common_args[@]}" \
    "${SCRIPT_PATH}")"
  log "resubmitted controller job=${job_id}"
}

claim_next_candidate_unlocked() {
  local head task_type max_steps
  refresh_state
  while IFS=$'\t' read -r head task_type max_steps; do
    [[ -n "${head}" ]] || continue
    if head_is_blocked "${head}"; then
      continue
    fi
    append_state_unlocked "${head}" "claimed" "-" "${task_type}" "${max_steps}" "controller-claim"
    printf '%s\t%s\t%s\n' "${head}" "${task_type}" "${max_steps}"
    return 0
  done < <(generate_candidates)
  return 1
}

claim_next_candidate() {
  with_lock claim_next_candidate_unlocked
}
self_heal_no_pair_markers() {
  [[ "${WSG_QUEUE_RECHECK_NO_PAIR_MARKERS}" == "1" ]] || return 0
  if ! WSG_QUEUE_SELF_HEAL_ONLY=1 \
    WSG_QUEUE_RECHECK_NO_PAIR_MARKERS=1 \
    WSG_QUEUE_DRY_RUN=0 \
    MODALITY_CONFIG="${MODALITY_CONFIG}" \
    WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY}" \
    bash "${QUEUE_SCRIPT}"; then
    log "WARNING: no-pair self-heal failed; continuing with legacy skip logic"
  fi
  return 0
}

run_controller() {
  require_path "${MODALITY_CONFIG}" "modality config"
  require_path "${QUEUE_SCRIPT}" "dense queue script"
  self_heal_no_pair_markers
  refresh_queue_plan
  ensure_state_file
  local started_at now elapsed submitted active target_active ramp_extra ramp_delay record head task_type max_steps
  started_at="$(date +%s)"
  ramp_delay=0
  if (( RAMP_SUBMISSIONS > 0 )); then
    ramp_delay=$((RAMP_SECONDS / RAMP_SUBMISSIONS))
    (( ramp_delay < 1 )) && ramp_delay=1
  fi

  log "controller start run_id=${SLRUM_RUN_ID} state=${SLRUM_STATE_FILE}"
  log "limits max_active=${MAX_ACTIVE_WORKERS} max_total=${MAX_TOTAL_SUBMISSIONS:-0} initial=${INITIAL_SUBMISSIONS} ramp=${RAMP_SUBMISSIONS}/${RAMP_SECONDS}s restart_after=${RESTART_AFTER_SECONDS}s"

  while true; do
    now="$(date +%s)"
    elapsed=$((now - started_at))
    if (( elapsed >= RESTART_AFTER_SECONDS )); then
      log "restart threshold reached elapsed=${elapsed}s"
      if [[ "${SELF_RESUBMIT}" == "1" && "${SUBMIT_WORKERS}" == "1" ]]; then
        submit_controller_restart
      fi
      exit 0
    fi

    with_lock refresh_state
    submitted="$(submitted_count)"
    active="$(active_count)"

    target_active="${INITIAL_SUBMISSIONS}"
    if (( RAMP_SUBMISSIONS > 0 && RAMP_SECONDS > 0 )); then
      ramp_extra=$((elapsed * RAMP_SUBMISSIONS / RAMP_SECONDS))
      (( ramp_extra > RAMP_SUBMISSIONS )) && ramp_extra="${RAMP_SUBMISSIONS}"
      target_active=$((INITIAL_SUBMISSIONS + ramp_extra))
    else
      target_active="${MAX_ACTIVE_WORKERS}"
    fi
    (( target_active > MAX_ACTIVE_WORKERS )) && target_active="${MAX_ACTIVE_WORKERS}"
    (( target_active < 1 )) && target_active=1

    log "status submitted_attempts=${submitted}/${MAX_TOTAL_SUBMISSIONS:-0} active_jobs=${active}/${target_active} max_active=${MAX_ACTIVE_WORKERS}"

    if (( MAX_TOTAL_SUBMISSIONS > 0 && submitted >= MAX_TOTAL_SUBMISSIONS )); then
      log "submission cap reached; controller will only keep state fresh"
      sleep "${CONTROLLER_POLL_SECONDS}"
      continue
    fi

    while (( active < target_active )); do
      if (( MAX_TOTAL_SUBMISSIONS > 0 && submitted >= MAX_TOTAL_SUBMISSIONS )); then
        break
      fi
      record="$(claim_next_candidate || true)"
      if [[ -z "${record}" ]]; then
        log "no eligible Channel Heads remain"
        break
      fi
      IFS=$'\t' read -r head task_type max_steps <<< "${record}"
      if [[ "${SUBMIT_WORKERS}" == "1" ]]; then
        submit_worker "${head}" "${task_type}" "${max_steps}"
      else
        log "dry controller would submit Channel_Head_$(printf '%06d' "${head}") task_type=${task_type} max_steps=${max_steps}"
        append_state "${head}" "submitted" "dry-run" "${task_type}" "${max_steps}" "dry-run"
      fi
      submitted=$((submitted + 1))
      active=$((active + 1))
    done

    if (( active < target_active )); then
      sleep "${CONTROLLER_POLL_SECONDS}"
    elif (( ramp_delay > 0 && target_active < MAX_ACTIVE_WORKERS )); then
      sleep "${ramp_delay}"
    else
      sleep "${CONTROLLER_POLL_SECONDS}"
    fi
  done
}

visible_gpu_id() {
  local cvd="${CUDA_VISIBLE_DEVICES:-0}"
  cvd="${cvd%%,*}"
  [[ -n "${cvd}" ]] || cvd="0"
  echo "${cvd}"
}

gpu_is_acceptable() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[slrum-dense] ERROR: nvidia-smi not found" >&2
    return 1
  fi
  local query_output gpu total used visible_gpu
  visible_gpu="$(visible_gpu_id)"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    query_output="$(nvidia-smi -i "${visible_gpu}" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)"
  else
    query_output="$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)"
  fi
  while IFS=',' read -r gpu total used; do
    gpu="${gpu//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    if (( total >= MIN_GPU_MEMORY_MB && used <= MAX_GPU_USED_MEMORY_MB )); then
      echo "${gpu}"
      return 0
    fi
  done <<< "${query_output}"
  echo "[slrum-dense] no visible idle GPU satisfies total>=${MIN_GPU_MEMORY_MB}MB used<=${MAX_GPU_USED_MEMORY_MB}MB" >&2
  nvidia-smi >&2 || true
  return 1
}

run_worker() {
  local head="${WSG_CHANNEL_HEAD_INDEX:?WSG_CHANNEL_HEAD_INDEX is required in worker mode}"
  local task_type="${TASK_TYPE:-unknown}"
  local max_steps="${MAX_STEPS:?MAX_STEPS is required in worker mode}"
  require_path "${QUEUE_SCRIPT}" "dense queue script"
  append_state "${head}" "running" "${SLURM_JOB_ID:-manual}" "${task_type}" "${max_steps}" "worker-start"

  local gpu
  if ! gpu="$(gpu_is_acceptable)"; then
    append_state "${head}" "gpu-rejected" "${SLURM_JOB_ID:-manual}" "${task_type}" "${max_steps}" "no-idle-24gb-gpu"
    exit 75
  fi

  local queue_run_id="slrum_${SLRUM_RUN_ID}_ch${head}"
  echo "[slrum-dense] worker head=${head} job=${SLURM_JOB_ID:-manual} gpu=${gpu} queue_run_id=${queue_run_id}"
  set +e
  GPU_IDS="${gpu}" \
  MAX_ACTIVE=1 \
  CHANNEL_HEAD_LIST="${head}" \
  WSG_CHANNEL_HEAD_LIST="${head}" \
  MAX_STEPS="${max_steps}" \
  QUEUE_RUN_ID="${queue_run_id}" \
  QUEUE_LOG_DIR="${SLRUM_LOG_DIR}/queue_logs" \
  MODALITY_CONFIG="${MODALITY_CONFIG}" \
  WSG_HISTORICAL_REFERENCE_POLICY="${WSG_HISTORICAL_REFERENCE_POLICY}" \
  bash "${QUEUE_SCRIPT}"
  local rc=$?
  set -e

  if (( rc == 0 )) && head_has_checkpoint_artifact "${head}"; then
    append_state "${head}" "completed" "${SLURM_JOB_ID:-manual}" "${task_type}" "${max_steps}" "queue-exited-zero"
  elif (( rc == 0 )); then
    append_state "${head}" "finished-no-artifact" "${SLURM_JOB_ID:-manual}" "${task_type}" "${max_steps}" "queue-exited-zero"
  else
    append_state "${head}" "failed" "${SLURM_JOB_ID:-manual}" "${task_type}" "${max_steps}" "queue-exit-${rc}"
  fi
  exit "${rc}"
}

case "${SLRUM_MODE}" in
  controller)
    run_controller
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "[slrum-dense] ERROR: unknown SLRUM_MODE=${SLRUM_MODE}" >&2
    exit 2
    ;;
esac
