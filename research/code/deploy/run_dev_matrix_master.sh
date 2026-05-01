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
  --validation-template TEMPLATE
  --switch-start-template TEMPLATE
  --switch-stop-template TEMPLATE
  --port-cleanup-check-template TEMPLATE

Runner template placeholders:
  {WORKER} {WORKER_RANK} {WORLD_SIZE} {MASTER_ADDR} {MASTER_PORT}
  {MODE} {EXPERIMENT} {REPEAT} {OUTPUT_DIR} {WORKER_OUTPUT_DIR}
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
    --validation-template) VALIDATION_TEMPLATE="${2:-}"; shift 2 ;;
    --switch-start-template) SWITCH_START_TEMPLATE="${2:-}"; shift 2 ;;
    --switch-stop-template) SWITCH_STOP_TEMPLATE="${2:-}"; shift 2 ;;
    --port-cleanup-check-template) PORT_CLEANUP_CHECK_TEMPLATE="${2:-}"; shift 2 ;;
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
  local experiment="$2"
  local mode="$3"
  local repeat="$4"
  local run_output_dir="$5"
  [[ -n "${hook_template}" ]] || return 0
  local cmd
  cmd="$(deploy_replace_tokens "${hook_template}" \
    EXPERIMENT "${experiment}" \
    MODE "${mode}" \
    REPEAT "${repeat}" \
    OUTPUT_DIR "${run_output_dir}" \
    MASTER_ADDR "${master_addr}" \
    MASTER_PORT "${MASTER_PORT}")"
  deploy_log "INFO" "Running hook: ${cmd}"
  bash -lc "${cmd}"
}

launch_worker_once() {
  local worker="$1"
  local worker_rank="$2"
  local experiment="$3"
  local mode="$4"
  local repeat="$5"
  local run_output_dir="$6"
  local status_dir="$7"

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
  local experiment="$1"
  local mode="$2"
  local repeat="$3"
  local run_output_dir="$4"
  [[ -n "${VALIDATION_TEMPLATE}" ]] || return 0
  local cmd
  cmd="$(deploy_replace_tokens "${VALIDATION_TEMPLATE}" \
    EXPERIMENT "${experiment}" \
    MODE "${mode}" \
    REPEAT "${repeat}" \
    OUTPUT_DIR "${run_output_dir}")"
  deploy_log "INFO" "Running correctness validation: ${cmd}"
  bash -lc "${cmd}"
}

run_cleanup_check_if_set() {
  local experiment="$1"
  local mode="$2"
  local repeat="$3"
  local run_output_dir="$4"
  [[ -n "${PORT_CLEANUP_CHECK_TEMPLATE}" ]] || return 0
  local cmd
  cmd="$(deploy_replace_tokens "${PORT_CLEANUP_CHECK_TEMPLATE}" \
    EXPERIMENT "${experiment}" \
    MODE "${mode}" \
    REPEAT "${repeat}" \
    OUTPUT_DIR "${run_output_dir}" \
    MASTER_PORT "${MASTER_PORT}")"
  deploy_log "INFO" "Running port cleanup check: ${cmd}"
  bash -lc "${cmd}"
}

run_one() {
  local experiment="$1"
  local mode="$2"
  local repeat="$3"
  local run_id="${experiment}/${mode}/repeat_$(printf '%02d' "${repeat}")"
  local run_output_dir="${OUTPUT_ROOT}/${run_id}"
  local status_dir="${STATUS_ROOT}/${experiment}/${mode}/repeat_$(printf '%02d' "${repeat}")"

  ensure_local_dir "${run_output_dir}"
  ensure_local_dir "${status_dir}"
  find "${status_dir}" -type f -name '*.status' -delete 2>/dev/null || true

  deploy_log "INFO" "start experiment=${experiment} mode=${mode} repeat=${repeat} master=${MASTER_WORKER} world_size=${WORLD_SIZE} port=${MASTER_PORT}"
  run_hook_if_set "${SWITCH_START_TEMPLATE}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"

  local rank=0
  for worker in "${WORKERS[@]}"; do
    deploy_log "INFO" "dispatch experiment=${experiment} mode=${mode} repeat=${repeat} worker=${worker} rank=${rank}"
    launch_worker_once "${worker}" "${rank}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}" "${status_dir}"
    rank=$(( rank + 1 ))
  done

  if ! wait_for_status_barrier "${status_dir}" "${experiment}" "${mode}" "${repeat}"; then
    run_hook_if_set "${SWITCH_STOP_TEMPLATE}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}" || true
    maybe_restart_containers_after_failure
    return 1
  fi

  run_hook_if_set "${SWITCH_STOP_TEMPLATE}" "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"
  run_validation_if_set "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"
  run_cleanup_check_if_set "${experiment}" "${mode}" "${repeat}" "${run_output_dir}"

  deploy_log "INFO" "done experiment=${experiment} mode=${mode} repeat=${repeat}"
}

ensure_local_dir "${OUTPUT_ROOT}"
ensure_local_dir "${STATUS_ROOT}"
ensure_container_ready_all

for experiment in "${EXPERIMENTS[@]}"; do
  for mode in "${MODES[@]}"; do
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
      if ! run_one "${experiment}" "${mode}" "${repeat}"; then
        deploy_die "Experiment failed: experiment=${experiment} mode=${mode} repeat=${repeat}"
      fi
    done
  done
done

deploy_log "INFO" "All experiments completed output_root=${OUTPUT_ROOT}"
