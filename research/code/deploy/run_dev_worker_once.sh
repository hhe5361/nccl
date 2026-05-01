#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/common.sh"

STATUS_FILE=""
WORKER_NAME=""
EXPERIMENT_NAME=""
MODE_NAME=""
REPEAT_INDEX=""
COMMAND_B64=""
TIMEOUT_SEC=600
USE_CONTAINER=1
CONTAINER_NAME="${CONTAINER_NAME:-nccl-cu121-dev}"

usage() {
  cat <<'EOF'
Usage:
  run_dev_worker_once.sh
    --status-file PATH
    --worker NAME
    --experiment NAME
    --mode NAME
    --repeat INDEX
    --command-b64 BASE64
    [--timeout-sec 600]
    [--use-container 1]
    [--container-name NAME]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status-file) STATUS_FILE="${2:-}"; shift 2 ;;
    --worker) WORKER_NAME="${2:-}"; shift 2 ;;
    --experiment) EXPERIMENT_NAME="${2:-}"; shift 2 ;;
    --mode) MODE_NAME="${2:-}"; shift 2 ;;
    --repeat) REPEAT_INDEX="${2:-}"; shift 2 ;;
    --command-b64) COMMAND_B64="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --use-container) USE_CONTAINER="${2:-}"; shift 2 ;;
    --container-name) CONTAINER_NAME="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) deploy_die "Unknown argument: $1" ;;
  esac
done

[[ -n "${STATUS_FILE}" ]] || deploy_die "--status-file is required"
[[ -n "${WORKER_NAME}" ]] || deploy_die "--worker is required"
[[ -n "${EXPERIMENT_NAME}" ]] || deploy_die "--experiment is required"
[[ -n "${MODE_NAME}" ]] || deploy_die "--mode is required"
[[ -n "${REPEAT_INDEX}" ]] || deploy_die "--repeat is required"
[[ -n "${COMMAND_B64}" ]] || deploy_die "--command-b64 is required"

COMMAND="$(python3 -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode())' "${COMMAND_B64}")"

export STATUS_DDP=1
deploy_write_status "${STATUS_FILE}" "1" "${WORKER_NAME}" "${EXPERIMENT_NAME}" "${MODE_NAME}" "${REPEAT_INDEX}" "launch" "worker started"

on_error() {
  local exit_code=$?
  export STATUS_DDP=-1
  deploy_write_status "${STATUS_FILE}" "-1" "${WORKER_NAME}" "${EXPERIMENT_NAME}" "${MODE_NAME}" "${REPEAT_INDEX}" "error" "exit_code=${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

run_direct() {
  timeout "${TIMEOUT_SEC}" bash -lc "${COMMAND}"
}

run_in_container() {
  timeout "${TIMEOUT_SEC}" env CONTAINER_NAME="${CONTAINER_NAME}" bash "${REPO_ROOT}/research/code/deploy/run_dev_container.sh" --start-only -- >/dev/null
  timeout "${TIMEOUT_SEC}" env CONTAINER_NAME="${CONTAINER_NAME}" bash "${REPO_ROOT}/research/code/deploy/run_dev_container.sh" -- bash -lc "${COMMAND}"
}

deploy_log "INFO" "worker=${WORKER_NAME} experiment=${EXPERIMENT_NAME} mode=${MODE_NAME} repeat=${REPEAT_INDEX} status=running use_container=${USE_CONTAINER}"
deploy_write_status "${STATUS_FILE}" "1" "${WORKER_NAME}" "${EXPERIMENT_NAME}" "${MODE_NAME}" "${REPEAT_INDEX}" "running" "command dispatched"

if [[ "${USE_CONTAINER}" == "1" ]]; then
  run_in_container
else
  run_direct
fi

export STATUS_DDP=0
deploy_write_status "${STATUS_FILE}" "0" "${WORKER_NAME}" "${EXPERIMENT_NAME}" "${MODE_NAME}" "${REPEAT_INDEX}" "done" "worker completed"
deploy_log "INFO" "worker=${WORKER_NAME} experiment=${EXPERIMENT_NAME} mode=${MODE_NAME} repeat=${REPEAT_INDEX} status=done"
