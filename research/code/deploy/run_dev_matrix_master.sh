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

master_addr="$(deploy_lookup_worker_ip "${TOPOLOGY_FILE}" "${MASTER_WORKER}")" || deploy_die "Cannot resolve master worker IP for ${MASTER_WORKER}"

ensure_local_dir() {
  mkdir -p "$1"
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
  [[ -n "${NETWORK_NODE_PASSWORD}" ]] || deploy_die "SWITCH_ENABLE=1 requires NETWORK_NODE_PASSWORD"
  [[ -n "${SWITCH_PASSWORD}" ]] || deploy_die "SWITCH_ENABLE=1 requires SWITCH_PASSWORD"
  [[ -x "${SWITCH_LOGGER_DIR}/start_switch_congestion_loggers.sh" || -f "${SWITCH_LOGGER_DIR}/start_switch_congestion_loggers.sh" ]] || deploy_die "start_switch_congestion_loggers.sh not found under ${SWITCH_LOGGER_DIR}"

  deploy_log "INFO" "Starting switch logger interval_sec=${SWITCH_INTERVAL_SEC}"
  local output key value
  output="$(
    bash "${SWITCH_LOGGER_DIR}/start_switch_congestion_loggers.sh" \
      --network-node-password "${NETWORK_NODE_PASSWORD}" \
      --switch-password "${SWITCH_PASSWORD}" \
      --interval-sec "${SWITCH_INTERVAL_SEC}"
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
  [[ -n "${SWITCH_LOGGER_DIR}" ]] || return 0

  if [[ -n "${SWITCH_PID_FILE}" ]]; then
    deploy_log "INFO" "Stopping switch logger pid_file=${SWITCH_PID_FILE}"
    bash "${SWITCH_LOGGER_DIR}/stop_switch_congestion_loggers.sh" --pid-file "${SWITCH_PID_FILE}"
  elif [[ -n "${SWITCH_LOG_DIR}" ]]; then
    deploy_log "INFO" "Stopping switch logger log_dir=${SWITCH_LOG_DIR}"
    bash "${SWITCH_LOGGER_DIR}/stop_switch_congestion_loggers.sh" --log-dir "${SWITCH_LOG_DIR}"
  fi

  SWITCH_STOPPED=1
}

switch_logger_marker() {
  local marker="$1"
  local message="$2"
  switch_logger_enabled || return 0
  [[ "${SWITCH_STARTED}" == "1" ]] || return 0
  [[ -n "${SWITCH_RUN_ID}" ]] || return 0
  bash "${SWITCH_LOGGER_DIR}/log_run_marker.sh" \
    --run-id "${SWITCH_RUN_ID}" \
    --marker "${marker}" \
    --source "run_dev_matrix_master.sh" \
    --message "${message}"
}

trap 'switch_logger_stop' EXIT

ensure_container_ready_all() {
  [[ "${USE_CONTAINER}" == "1" ]] || return 0
  local reset_flag=""
  [[ "${CONTAINER_CLEAN_START}" == "1" ]] && reset_flag="--reset"
  deploy_log "INFO" "Ensuring persistent containers across workers reset=${CONTAINER_CLEAN_START}"
  for worker in "${WORKERS[@]}"; do
    local host_ip
    host_ip="$(deploy_lookup_worker_ip "${TOPOLOGY_FILE}" "${worker}")" || deploy_die "Cannot resolve IP for ${worker}"
    local remote_cmd="cd '${REMOTE_REPO_ROOT}' && CONTAINER_NAME='${CONTAINER_NAME}' bash research/code/deploy/run_dev_container.sh ${reset_flag} --start-only"
    if [[ "${worker}" == "${MASTER_WORKER}" ]]; then
      bash -lc "${remote_cmd}"
    else
      deploy_ssh_cmd "${worker}" "${host_ip}" "${remote_cmd}"
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

  local host_ip status_file worker_output_dir runner_cmd runner_cmd_b64 remote_cmd
  host_ip="$(deploy_lookup_worker_ip "${TOPOLOGY_FILE}" "${worker}")" || deploy_die "Cannot resolve IP for ${worker}"
  status_file="${status_dir}/${worker}.status"
  worker_output_dir="${run_output_dir}/${worker}"
  ensure_local_dir "${worker_output_dir}"

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

  remote_cmd="cd '${REMOTE_REPO_ROOT}' && bash research/code/deploy/run_dev_worker_once.sh \
    --status-file '${status_file}' \
    --worker '${worker}' \
    --experiment '${experiment}' \
    --mode '${mode}' \
    --repeat '${repeat}' \
    --command-b64 '${runner_cmd_b64}' \
    --timeout-sec '${TIMEOUT_SEC}' \
    --use-container '${USE_CONTAINER}' \
    --container-name '${CONTAINER_NAME}'"

  if [[ "${worker}" == "${MASTER_WORKER}" ]]; then
    bash -lc "${remote_cmd}" &
  else
    deploy_ssh_cmd "${worker}" "${host_ip}" "${remote_cmd}" >/dev/null 2>&1 &
  fi
}

wait_for_status_barrier() {
  local status_dir="$1"
  local experiment="$2"
  local mode="$3"
  local repeat="$4"
  local start_ts now_ts elapsed
  start_ts="$(date +%s)"

  while true; do
    local all_done=1
    for worker in "${WORKERS[@]}"; do
      local status_file="${status_dir}/${worker}.status"
      if [[ ! -f "${status_file}" ]]; then
        all_done=0
        continue
      fi
      local state stage
      state="$(deploy_read_status_field "${status_file}" state)"
      stage="$(deploy_read_status_field "${status_file}" stage)"
      if [[ "${state}" == "-1" ]]; then
        deploy_log "ERROR" "experiment=${experiment} mode=${mode} repeat=${repeat} worker=${worker} failed stage=${stage}"
        return 1
      fi
      if [[ "${state}" != "0" ]]; then
        all_done=0
      fi
    done

    if [[ "${all_done}" == "1" ]]; then
      deploy_log "INFO" "experiment=${experiment} mode=${mode} repeat=${repeat} all workers reached STATUS_DDP=0"
      return 0
    fi

    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))
    if (( elapsed > TIMEOUT_SEC )); then
      deploy_log "ERROR" "experiment=${experiment} mode=${mode} repeat=${repeat} timeout after ${TIMEOUT_SEC}s"
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

  ensure_local_dir "${run_output_dir}"
  ensure_local_dir "${status_dir}"
  find "${status_dir}" -type f -name '*.status' -delete 2>/dev/null || true
  switch_logger_write_run_meta "$(run_level_meta_file "${run_output_dir}")"

  deploy_log "INFO" "start run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} master=${MASTER_WORKER} world_size=${WORLD_SIZE} port=${MASTER_PORT}"
  switch_logger_marker "mode_start" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} output_dir=${run_output_dir}"
  run_hook_if_set "${SWITCH_START_TEMPLATE}" "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"

  local rank=0
  switch_logger_marker "ddp_start" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} before dispatch"
  for worker in "${WORKERS[@]}"; do
    deploy_log "INFO" "dispatch run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} worker=${worker} rank=${rank}"
    launch_worker_once "${worker}" "${rank}" "${experiment}" "${mode}" "${repeat}" "${run_id}" "${run_output_dir}" "${status_dir}"
    rank=$(( rank + 1 ))
  done

  if ! wait_for_status_barrier "${status_dir}" "${experiment}" "${mode}" "${repeat}"; then
    switch_logger_marker "ddp_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} failed barrier"
    switch_logger_marker "mode_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} failed"
    run_hook_if_set "${SWITCH_STOP_TEMPLATE}" "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}" || true
    maybe_restart_containers_after_failure
    return 1
  fi

  switch_logger_marker "ddp_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} barrier complete"
  run_hook_if_set "${SWITCH_STOP_TEMPLATE}" "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"
  run_validation_if_set "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"
  run_cleanup_check_if_set "${run_id}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"
  switch_logger_marker "mode_end" "run_id=${run_id} experiment=${experiment} mode=${mode} repeat=${repeat} done"

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
        deploy_die "Experiment failed: experiment=${experiment} mode=${mode} repeat=${repeat}"
      fi
    done
  done
  switch_logger_marker "exp_end" "experiment=${experiment}"
done

switch_logger_marker "matrix_end" "run_id=${RUN_ID} output_root=${OUTPUT_ROOT}/${RUN_ID}"
switch_logger_stop
deploy_log "INFO" "All experiments completed run_id=${RUN_ID} output_root=${OUTPUT_ROOT}/${RUN_ID}"
