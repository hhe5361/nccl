#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

TOPOLOGY_FILE="${SCRIPT_DIR}/network_topology_internal_ips.txt"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT}}"
STATUS_ROOT="${STATUS_ROOT:-/mnt/nfs_share/cts_experiments/.ddp_status}"
OUTPUT_ROOT=""
RUNNER_TEMPLATE=""
VALIDATION_TEMPLATE="${VALIDATION_TEMPLATE:-}"
SWITCH_START_TEMPLATE="${SWITCH_START_TEMPLATE:-}"
SWITCH_STOP_TEMPLATE="${SWITCH_STOP_TEMPLATE:-}"
PORT_CLEANUP_CHECK_TEMPLATE="${PORT_CLEANUP_CHECK_TEMPLATE:-}"
SWITCH_ENABLE="${SWITCH_ENABLE:-0}"
SWITCH_LOGGER_DIR="${SWITCH_LOGGER_DIR:-}"
SWITCH_INTERVAL_SEC="${SWITCH_INTERVAL_SEC:-1}"
SWITCH_DPU_HOST="${SWITCH_DPU_HOST:-172.16.0.100}"
SWITCH_DPU_USER="${SWITCH_DPU_USER:-ubuntu}"
SWITCH_DPU_PORT="${SWITCH_DPU_PORT:-22}"
SWITCH_DPU_PASSWORD="${SWITCH_DPU_PASSWORD:-}"
NETWORK_NODE_PASSWORD="${NETWORK_NODE_PASSWORD:-}"
SWITCH_PASSWORD="${SWITCH_PASSWORD:-}"
WORKERS_CSV="${WORKERS_CSV:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}"
MASTER_WORKER="${MASTER_WORKER:-worker01}"
MASTER_PORT="${MASTER_PORT:-29500}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
USE_CONTAINER="${USE_CONTAINER:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-nccl-cu121-dev}"
CONTAINER_CLEAN_START="${CONTAINER_CLEAN_START:-1}"
RESTART_CONTAINER_ON_FAILURE="${RESTART_CONTAINER_ON_FAILURE:-1}"
REPEATS="${REPEATS:-1}"
MODES_CSV="${MODES_CSV:-STOCK}"
EXPERIMENTS_CSV="${EXPERIMENTS_CSV:-allreduce_ring,allreduce_tree,alltoall}"
WORLD_SIZE=0
RUN_ID="${RUN_ID:-$(date +%m%d%H%M)}"
SWITCH_RUN_ID=""
SWITCH_LOG_DIR=""
SWITCH_PID_FILE=""
SWITCH_MARKERS_JSONL=""
SWITCH_STARTED=0
SWITCH_STOPPED=0
TOTAL_RUNS=0
CURRENT_RUN_INDEX=0
FAILURE_COUNT=0
WAIT_FAILURE_REASON=""
WAIT_FAILURE_DETAIL=""
WAIT_FAILED_WORKERS=""

usage() {
  cat <<'EOF'
Usage:
  run_dev_matrix_master.sh --output-root DIR --runner-template TEMPLATE [options]

Required:
  --output-root DIR
  --runner-template TEMPLATE

Optional:
  --repeats N
  --modes CSV
  --experiments CSV
  --master-port PORT
  --timeout-sec SEC
  --use-container 0|1
  --container-name NAME
  --remote-repo-root PATH
  --status-root PATH
  --run-id ID
  --validation-template TEMPLATE
  --switch-start-template TEMPLATE
  --switch-stop-template TEMPLATE
  --port-cleanup-check-template TEMPLATE
  --switch-enable 0|1
  --switch-logger-dir PATH
  --switch-interval-sec SEC

Runner template placeholders:
  {WORKER} {WORKER_RANK} {WORLD_SIZE} {MASTER_ADDR} {MASTER_PORT}
  {MODE} {EXPERIMENT} {REPEAT} {RUN_ID} {OUTPUT_DIR} {WORKER_OUTPUT_DIR}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --runner-template) RUNNER_TEMPLATE="${2:-}"; shift 2 ;;
    --repeats) REPEATS="${2:-}"; shift 2 ;;
    --modes) MODES_CSV="${2:-}"; shift 2 ;;
    --experiments) EXPERIMENTS_CSV="${2:-}"; shift 2 ;;
    --master-port) MASTER_PORT="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --use-container) USE_CONTAINER="${2:-}"; shift 2 ;;
    --container-name) CONTAINER_NAME="${2:-}"; shift 2 ;;
    --remote-repo-root) REMOTE_REPO_ROOT="${2:-}"; shift 2 ;;
    --status-root) STATUS_ROOT="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --validation-template) VALIDATION_TEMPLATE="${2:-}"; shift 2 ;;
    --switch-start-template) SWITCH_START_TEMPLATE="${2:-}"; shift 2 ;;
    --switch-stop-template) SWITCH_STOP_TEMPLATE="${2:-}"; shift 2 ;;
    --port-cleanup-check-template) PORT_CLEANUP_CHECK_TEMPLATE="${2:-}"; shift 2 ;;
    --switch-enable) SWITCH_ENABLE="${2:-}"; shift 2 ;;
    --switch-logger-dir) SWITCH_LOGGER_DIR="${2:-}"; shift 2 ;;
    --switch-interval-sec) SWITCH_INTERVAL_SEC="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) deploy_die "Unknown argument: $1" ;;
  esac
done

[[ -n "${OUTPUT_ROOT}" ]] || deploy_die "--output-root is required"
[[ -n "${RUNNER_TEMPLATE}" ]] || deploy_die "--runner-template is required"
[[ -f "${TOPOLOGY_FILE}" ]] || deploy_die "Topology file not found: ${TOPOLOGY_FILE}"

IFS=',' read -r -a WORKERS <<< "${WORKERS_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
IFS=',' read -r -a EXPERIMENTS <<< "${EXPERIMENTS_CSV}"
WORLD_SIZE="${#WORKERS[@]}"
TOTAL_RUNS=$(( ${#EXPERIMENTS[@]} * ${#MODES[@]} * REPEATS ))

master_addr="$(deploy_lookup_worker_ip "${TOPOLOGY_FILE}" "${MASTER_WORKER}")" || deploy_die "Cannot resolve master worker IP for ${MASTER_WORKER}"

ensure_local_dir() {
  mkdir -p "$1"
}

write_run_status() {
  local run_output_dir="$1"
  local status="$2"
  local reason="$3"
  local detail="$4"
  local failed_workers="$5"
  python3 - "$run_output_dir" "$status" "$reason" "$detail" "$failed_workers" <<'PY'
import json, sys
from pathlib import Path
run_output_dir = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "reason": sys.argv[3],
    "detail": sys.argv[4],
    "failed_workers": [w for w in sys.argv[5].split(",") if w],
}
(run_output_dir / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

switch_logger_enabled() {
  [[ "${SWITCH_ENABLE}" == "1" ]]
}

switch_logger_meta_file() {
  printf '%s' "${OUTPUT_ROOT}/${RUN_ID}/switch_logger_meta.env"
}

run_level_meta_file() {
  local run_output_dir="$1"
  printf '%s' "${run_output_dir}/switch_logger_meta.env"
}

switch_logger_write_run_meta() {
  local output_file="$1"
  [[ -n "${RUN_ID}" ]] || return 0
  deploy_write_kv_file "${output_file}" \
    RUN_ID "${RUN_ID}" \
    SWITCH_RUN_ID "${SWITCH_RUN_ID}" \
    SWITCH_LOG_DIR "${SWITCH_LOG_DIR}" \
    PID_FILE "${SWITCH_PID_FILE}" \
    MARKERS_JSONL "${SWITCH_MARKERS_JSONL}"
}

switch_logger_start() {
  switch_logger_enabled || return 0
  [[ -n "${SWITCH_LOGGER_DIR}" ]] || deploy_die "SWITCH_ENABLE=1 requires SWITCH_LOGGER_DIR"
  [[ -n "${SWITCH_DPU_PASSWORD}" ]] || deploy_die "SWITCH_ENABLE=1 requires SWITCH_DPU_PASSWORD"
  [[ -n "${NETWORK_NODE_PASSWORD}" ]] || deploy_die "SWITCH_ENABLE=1 requires NETWORK_NODE_PASSWORD"
  [[ -n "${SWITCH_PASSWORD}" ]] || deploy_die "SWITCH_ENABLE=1 requires SWITCH_PASSWORD"

  deploy_log "INFO" "Starting switch logger on dpu_host=${SWITCH_DPU_HOST} interval_sec=${SWITCH_INTERVAL_SEC}"
  local output key value
  output="$(
    deploy_ssh_target_cmd \
      "${SWITCH_DPU_HOST}" \
      "${SWITCH_DPU_USER}" \
      "${SWITCH_DPU_PORT}" \
      "${SWITCH_DPU_PASSWORD}" \
      "cd '${SWITCH_LOGGER_DIR}' && bash ./start_switch_congestion_loggers.sh --network-node-password '${NETWORK_NODE_PASSWORD}' --switch-password '${SWITCH_PASSWORD}' --interval-sec '${SWITCH_INTERVAL_SEC}'"
  )"

  while IFS='=' read -r key value; do
    [[ -n "${key}" ]] || continue
    case "${key}" in
      RUN_ID) SWITCH_RUN_ID="${value}" ;;
      LOG_DIR) SWITCH_LOG_DIR="${value}" ;;
      PID_FILE) SWITCH_PID_FILE="${value}" ;;
      MARKERS_JSONL) SWITCH_MARKERS_JSONL="${value}" ;;
    esac
  done <<< "${output}"

  [[ -n "${SWITCH_RUN_ID}" ]] || deploy_die "Switch logger start did not return RUN_ID"
  [[ -n "${SWITCH_PID_FILE}" || -n "${SWITCH_LOG_DIR}" ]] || deploy_die "Switch logger start did not return PID_FILE or LOG_DIR"

  SWITCH_STARTED=1
  SWITCH_STOPPED=0
  switch_logger_write_run_meta "$(switch_logger_meta_file)"
  deploy_log "INFO" "Switch logger started run_id=${SWITCH_RUN_ID} log_dir=${SWITCH_LOG_DIR}"
}

switch_logger_stop() {
  switch_logger_enabled || return 0
  [[ "${SWITCH_STARTED}" == "1" ]] || return 0
  [[ "${SWITCH_STOPPED}" == "0" ]] || return 0
  [[ -n "${SWITCH_LOGGER_DIR}" && -n "${SWITCH_DPU_HOST}" ]] || return 0

  if [[ -n "${SWITCH_PID_FILE}" ]]; then
    deploy_log "INFO" "Stopping switch logger on dpu_host=${SWITCH_DPU_HOST} pid_file=${SWITCH_PID_FILE}"
    deploy_ssh_target_cmd \
      "${SWITCH_DPU_HOST}" \
      "${SWITCH_DPU_USER}" \
      "${SWITCH_DPU_PORT}" \
      "${SWITCH_DPU_PASSWORD}" \
      "cd '${SWITCH_LOGGER_DIR}' && bash ./stop_switch_congestion_loggers.sh --pid-file '${SWITCH_PID_FILE}'"
  elif [[ -n "${SWITCH_LOG_DIR}" ]]; then
    deploy_log "INFO" "Stopping switch logger on dpu_host=${SWITCH_DPU_HOST} log_dir=${SWITCH_LOG_DIR}"
    deploy_ssh_target_cmd \
      "${SWITCH_DPU_HOST}" \
      "${SWITCH_DPU_USER}" \
      "${SWITCH_DPU_PORT}" \
      "${SWITCH_DPU_PASSWORD}" \
      "cd '${SWITCH_LOGGER_DIR}' && bash ./stop_switch_congestion_loggers.sh --log-dir '${SWITCH_LOG_DIR}'"
  fi

  SWITCH_STOPPED=1
}

switch_logger_marker() {
  local marker="$1"
  local message="$2"
  switch_logger_enabled || return 0
  [[ "${SWITCH_STARTED}" == "1" ]] || return 0
  [[ -n "${SWITCH_RUN_ID}" ]] || return 0
  deploy_ssh_target_cmd \
    "${SWITCH_DPU_HOST}" \
    "${SWITCH_DPU_USER}" \
    "${SWITCH_DPU_PORT}" \
    "${SWITCH_DPU_PASSWORD}" \
    "cd '${SWITCH_LOGGER_DIR}' && bash ./log_run_marker.sh --run-id '${SWITCH_RUN_ID}' --marker '${marker}' --source 'run_dev_matrix_master.sh' --message '${message}'"
}

trap 'switch_logger_stop' EXIT

ensure_container_ready_all() {
  [[ "${USE_CONTAINER}" == "1" ]] || return 0
  local reset_flag=""
  [[ "${CONTAINER_CLEAN_START}" == "1" ]] && reset_flag="--reset"
  deploy_log "INFO" "Ensuring persistent containers across workers reset=${CONTAINER_CLEAN_START}"
  for worker in "${WORKERS[@]}"; do
    local host_ip remote_repo_root
    host_ip="$(deploy_lookup_worker_ip "${TOPOLOGY_FILE}" "${worker}")" || deploy_die "Cannot resolve IP for ${worker}"
    remote_repo_root="$(deploy_worker_repo_root "${worker}" "${REMOTE_REPO_ROOT}")"
    local remote_cmd="cd '${remote_repo_root}' && CONTAINER_NAME='${CONTAINER_NAME}' bash research/code/deploy/run_dev_container.sh ${reset_flag} --start-only"
    deploy_log "INFO" "container-prepare worker=${worker} host_ip=${host_ip} repo_root=${remote_repo_root}"
    if [[ "${worker}" == "${MASTER_WORKER}" ]]; then
      if ! bash -lc "${remote_cmd}"; then
        deploy_die "container-prepare failed worker=${worker} host_ip=${host_ip} repo_root=${remote_repo_root}"
      fi
    else
      if ! deploy_ssh_cmd "${worker}" "${host_ip}" "${remote_cmd}"; then
        deploy_die "container-prepare failed worker=${worker} host_ip=${host_ip} repo_root=${remote_repo_root}"
      fi
    fi
  done
}

maybe_restart_containers_after_failure() {
  [[ "${USE_CONTAINER}" == "1" && "${RESTART_CONTAINER_ON_FAILURE}" == "1" ]] || return 0
  deploy_log "WARN" "Failure detected. Restarting persistent containers before next run."
  CONTAINER_CLEAN_START=1 ensure_container_ready_all
}

run_hook_if_set() {
  local hook_template="$1"
  local run_id="$2"
  local experiment="$3"
  local mode="$4"
  local repeat="$5"
  local run_output_dir="$6"
  [[ -n "${hook_template}" ]] || return 0
  local cmd
  cmd="$(deploy_replace_tokens "${hook_template}" \
    RUN_ID "${run_id}" \
    EXPERIMENT "${experiment}" \
    MODE "${mode}" \
    REPEAT "${repeat}" \
    OUTPUT_DIR "${run_output_dir}" \
    MASTER_ADDR "${master_addr}" \
    MASTER_PORT "${MASTER_PORT}" \
    SWITCH_RUN_ID "${SWITCH_RUN_ID}" \
    SWITCH_LOG_DIR "${SWITCH_LOG_DIR}" \
    SWITCH_PID_FILE "${SWITCH_PID_FILE}")"
  deploy_log "INFO" "Running hook: ${cmd}"
  bash -lc "${cmd}"
}

launch_worker_once() {
  local worker="$1"
  local worker_rank="$2"
  local experiment="$3"
  local mode="$4"
  local repeat="$5"
  local run_id="$6"
  local run_output_dir="$7"
  local status_dir="$8"

  local host_ip status_file worker_output_dir runner_cmd runner_cmd_b64 remote_cmd remote_repo_root worker_log_file
  host_ip="$(deploy_lookup_worker_ip "${TOPOLOGY_FILE}" "${worker}")" || deploy_die "Cannot resolve IP for ${worker}"
  status_file="${status_dir}/${worker}.status"
  worker_output_dir="${run_output_dir}/${worker}"
  ensure_local_dir "${worker_output_dir}"
  remote_repo_root="$(deploy_worker_repo_root "${worker}" "${REMOTE_REPO_ROOT}")"
  worker_log_file="${worker_output_dir}/worker_launcher.log"

  runner_cmd="$(deploy_replace_tokens "${RUNNER_TEMPLATE}" \
    WORKER "${worker}" \
    WORKER_RANK "${worker_rank}" \
    WORLD_SIZE "${WORLD_SIZE}" \
    MASTER_ADDR "${master_addr}" \
    MASTER_PORT "${MASTER_PORT}" \
    MODE "${mode}" \
    EXPERIMENT "${experiment}" \
    REPEAT "${repeat}" \
    RUN_ID "${run_id}" \
    OUTPUT_DIR "${run_output_dir}" \
    WORKER_OUTPUT_DIR "${worker_output_dir}")"
  runner_cmd_b64="$(deploy_encode_b64 "${runner_cmd}")"

  remote_cmd="cd '${remote_repo_root}' && bash research/code/deploy/run_dev_worker_once.sh \
    --status-file '${status_file}' \
    --worker '${worker}' \
    --experiment '${experiment}' \
    --mode '${mode}' \
    --repeat '${repeat}' \
    --command-b64 '${runner_cmd_b64}' \
    --timeout-sec '${TIMEOUT_SEC}' \
    --use-container '${USE_CONTAINER}' \
    --container-name '${CONTAINER_NAME}'"

  deploy_log "INFO" "launch worker=${worker} host_ip=${host_ip} repo_root=${remote_repo_root} log_file=${worker_log_file}"
  if [[ "${worker}" == "${MASTER_WORKER}" ]]; then
    bash -lc "${remote_cmd}" >"${worker_log_file}" 2>&1 &
  else
    deploy_ssh_cmd "${worker}" "${host_ip}" "${remote_cmd}" >"${worker_log_file}" 2>&1 &
  fi
}

wait_for_status_barrier() {
  local status_dir="$1"
  local run_id="$2"
  local experiment="$3"
  local mode="$4"
  local repeat="$5"
  local start_ts now_ts elapsed
  start_ts="$(date +%s)"
  WAIT_FAILURE_REASON=""
  WAIT_FAILURE_DETAIL=""
  WAIT_FAILED_WORKERS=""

  while true; do
    local all_done=1
    local done_count=0
    local pending_count=0
    local status_parts=()
    for worker in "${WORKERS[@]}"; do
      local status_file="${status_dir}/${worker}.status"
      if [[ ! -f "${status_file}" ]]; then
        all_done=0
        pending_count=$(( pending_count + 1 ))
        status_parts+=("${worker}:missing")
        continue
      fi
      local state stage
      state="$(deploy_read_status_field "${status_file}" state)"
      stage="$(deploy_read_status_field "${status_file}" stage)"
      if [[ "${state}" == "-1" ]]; then
        deploy_log "ERROR" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} worker=${worker} failed stage=${stage}"
        WAIT_FAILURE_REASON="worker_failed"
        WAIT_FAILURE_DETAIL="worker=${worker} stage=${stage}"
        WAIT_FAILED_WORKERS="${worker}"
        return 1
      fi
      if [[ "${state}" == "0" ]]; then
        done_count=$(( done_count + 1 ))
      else
        all_done=0
        pending_count=$(( pending_count + 1 ))
      fi
      status_parts+=("${worker}:${state}/${stage}")
    done

    if [[ "${all_done}" == "1" ]]; then
      deploy_log "INFO" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} all workers reached STATUS_DDP=0"
      return 0
    fi

    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))
    deploy_log "INFO" "waiting run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} done=${done_count}/${WORLD_SIZE} pending=${pending_count} states=$(IFS=,; echo "${status_parts[*]}")"
    if (( elapsed > TIMEOUT_SEC )); then
      deploy_log "ERROR" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} timeout after ${TIMEOUT_SEC}s"
      WAIT_FAILURE_REASON="timeout"
      WAIT_FAILURE_DETAIL="timeout after ${TIMEOUT_SEC}s"
      WAIT_FAILED_WORKERS="$(printf '%s\n' "${status_parts[@]}" | awk -F: '$2 !~ /^0\\// {print $1}' | paste -sd, -)"
      return 1
    fi

    sleep 5
  done
}

run_validation_if_set() {
  local run_id="$1"
  local experiment="$2"
  local mode="$3"
  local repeat="$4"
  local run_output_dir="$5"
  [[ -n "${VALIDATION_TEMPLATE}" ]] || return 0
  local cmd
  cmd="$(deploy_replace_tokens "${VALIDATION_TEMPLATE}" \
    RUN_ID "${run_id}" \
    EXPERIMENT "${experiment}" \
    MODE "${mode}" \
    REPEAT "${repeat}" \
    OUTPUT_DIR "${run_output_dir}" \
    SWITCH_RUN_ID "${SWITCH_RUN_ID}" \
    SWITCH_LOG_DIR "${SWITCH_LOG_DIR}" \
    SWITCH_PID_FILE "${SWITCH_PID_FILE}")"
  deploy_log "INFO" "Running correctness validation: ${cmd}"
  bash -lc "${cmd}"
}

run_cleanup_check_if_set() {
  local run_id="$1"
  local experiment="$2"
  local mode="$3"
  local repeat="$4"
  local run_output_dir="$5"
  [[ -n "${PORT_CLEANUP_CHECK_TEMPLATE}" ]] || return 0
  local cmd
  cmd="$(deploy_replace_tokens "${PORT_CLEANUP_CHECK_TEMPLATE}" \
    RUN_ID "${run_id}" \
    EXPERIMENT "${experiment}" \
    MODE "${mode}" \
    REPEAT "${repeat}" \
    OUTPUT_DIR "${run_output_dir}" \
    MASTER_PORT "${MASTER_PORT}" \
    SWITCH_RUN_ID "${SWITCH_RUN_ID}" \
    SWITCH_LOG_DIR "${SWITCH_LOG_DIR}" \
    SWITCH_PID_FILE "${SWITCH_PID_FILE}")"
  deploy_log "INFO" "Running port cleanup check: ${cmd}"
  bash -lc "${cmd}"
}

run_one() {
  local experiment="$1"
  local mode="$2"
  local repeat="$3"
  local run_id="${RUN_ID}"
  local run_output_dir="${OUTPUT_ROOT}/${RUN_ID}/${mode}/${experiment}/repeat_$(printf '%02d' "${repeat}")"
  local status_dir="${STATUS_ROOT}/${RUN_ID}/${mode}/${experiment}/repeat_$(printf '%02d' "${repeat}")"
  CURRENT_RUN_INDEX=$(( CURRENT_RUN_INDEX + 1 ))

  ensure_local_dir "${run_output_dir}"
  ensure_local_dir "${status_dir}"
  find "${status_dir}" -type f -name '*.status' -delete 2>/dev/null || true
  switch_logger_write_run_meta "$(run_level_meta_file "${run_output_dir}")"
  write_run_status "${run_output_dir}" "running" "-" "launched" ""

  deploy_log "INFO" "progress=${CURRENT_RUN_INDEX}/${TOTAL_RUNS} start run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} master=${MASTER_WORKER} world_size=${WORLD_SIZE} port=${MASTER_PORT} output_dir=${run_output_dir}"
  switch_logger_marker "mode_start" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} output_dir=${run_output_dir}"
  run_hook_if_set "${SWITCH_START_TEMPLATE}" "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"

  local rank=0
  switch_logger_marker "ddp_start" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} before dispatch"
  for worker in "${WORKERS[@]}"; do
    deploy_log "INFO" "dispatch run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} worker=${worker} rank=${rank}"
    launch_worker_once "${worker}" "${rank}" "${experiment}" "${mode}" "${repeat}" "${run_id}" "${run_output_dir}" "${status_dir}"
    rank=$(( rank + 1 ))
  done

  if ! wait_for_status_barrier "${status_dir}" "${run_id}" "${experiment}" "${mode}" "${repeat}"; then
    switch_logger_marker "ddp_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} failed barrier"
    switch_logger_marker "mode_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} failed"
    write_run_status "${run_output_dir}" "failed" "${WAIT_FAILURE_REASON:-barrier_failed}" "${WAIT_FAILURE_DETAIL:-barrier failed}" "${WAIT_FAILED_WORKERS:-}"
    run_hook_if_set "${SWITCH_STOP_TEMPLATE}" "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}" || true
    maybe_restart_containers_after_failure
    return 1
  fi

  switch_logger_marker "ddp_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} barrier complete"
  run_hook_if_set "${SWITCH_STOP_TEMPLATE}" "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"
  if ! run_validation_if_set "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"; then
    write_run_status "${run_output_dir}" "failed" "validation_failed" "correctness validation failed" ""
    maybe_restart_containers_after_failure
    return 1
  fi
  if ! run_cleanup_check_if_set "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"; then
    write_run_status "${run_output_dir}" "failed" "cleanup_check_failed" "port cleanup check failed" ""
    maybe_restart_containers_after_failure
    return 1
  fi
  switch_logger_marker "mode_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} done"
  write_run_status "${run_output_dir}" "done" "-" "completed" ""

  deploy_log "INFO" "done run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat}"
}

ensure_local_dir "${OUTPUT_ROOT}/${RUN_ID}"
ensure_local_dir "${STATUS_ROOT}"
ensure_container_ready_all
switch_logger_start
switch_logger_marker "matrix_start" "run_id=${RUN_ID} output_root=${OUTPUT_ROOT}/${RUN_ID} world_size=${WORLD_SIZE} modes=${MODES_CSV} experiments=${EXPERIMENTS_CSV} repeats=${REPEATS}"

for experiment in "${EXPERIMENTS[@]}"; do
  switch_logger_marker "exp_start" "experiment=${experiment}"
  for mode in "${MODES[@]}"; do
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
      if ! run_one "${experiment}" "${mode}" "${repeat}"; then
        FAILURE_COUNT=$(( FAILURE_COUNT + 1 ))
        deploy_log "ERROR" "Experiment failed: experiment=${experiment} mode=${mode} repeat=${repeat}"
        continue
      fi
    done
  done
  switch_logger_marker "exp_end" "experiment=${experiment}"
done

switch_logger_marker "matrix_end" "run_id=${RUN_ID} output_root=${OUTPUT_ROOT}/${RUN_ID}"
switch_logger_stop
if (( FAILURE_COUNT > 0 )); then
  deploy_log "WARN" "Matrix completed with failures count=${FAILURE_COUNT} run_id=${RUN_ID} output_root=${OUTPUT_ROOT}/${RUN_ID}"
  exit 1
fi
deploy_log "INFO" "All experiments completed run_id=${RUN_ID} output_root=${OUTPUT_ROOT}/${RUN_ID}"
