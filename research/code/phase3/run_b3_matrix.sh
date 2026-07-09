#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase3_b3_matrix_$(date +%y%m%d_%H%M%S)}
MASTER_SERVER=${MASTER_SERVER:-worker01}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT=${MASTER_PORT:-${MASTER_PORT_BASE:-40000}}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-/mnt/nfs_share/cts_experiments}
MATRIX_ROOT=${MATRIX_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
ALL_WORKERS=${ALL_WORKERS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
PAYLOAD_MB=${PAYLOAD_MB:-128}
RUN_MODES=${RUN_MODES:-stock,b3}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
SLEEP_MS=${SLEEP_MS:-0}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-600}
PORT_READY_TIMEOUT_SEC=${PORT_READY_TIMEOUT_SEC:-60}
STATUS_POLL_SEC=${STATUS_POLL_SEC:-5}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu132-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase2/collective_b2.py}
WORKER_SCRIPT=${WORKER_SCRIPT:-research/code/phase3/run_b3_mode_worker.sh}
COMPARE_SCRIPT=${COMPARE_SCRIPT:-research/code/phase2/compare_b2_vs_stock.py}

HOST_REPO_ROOT=${HOST_REPO_ROOT:-${REPO_ROOT}}
REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-${HOST_REPO_ROOT}}
REMOTE_REPO_ROOT_MAP=${REMOTE_REPO_ROOT_MAP:-}
CONTAINER_REPO_ROOT=${CONTAINER_REPO_ROOT:-/workspace/$(basename "${REPO_ROOT}")}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu132-dev}
CONTAINER_RESET_AT_START=${CONTAINER_RESET_AT_START:-1}
RESTART_CONTAINERS_ON_FAILURE=${RESTART_CONTAINERS_ON_FAILURE:-1}
WORKER_SSH_USER=${WORKER_SSH_USER:-}
WORKER_SSH_USER_MAP=${WORKER_SSH_USER_MAP:-}
WORKER_SSH_PORT=${WORKER_SSH_PORT:-22}
WORKER_SSH_PORT_MAP=${WORKER_SSH_PORT_MAP:-}
WORKER_SSH_PASSWORD=${WORKER_SSH_PASSWORD:-}
SSH_CONNECT_TIMEOUT_SEC=${SSH_CONNECT_TIMEOUT_SEC:-10}
NETWORK_TOPOLOGY_FILE=${NETWORK_TOPOLOGY_FILE:-${REPO_ROOT}/research/env/network_topology_internal_ips.txt}
STOP_ON_FAILURE=${STOP_ON_FAILURE:-0}

NCCL_PHASE3_WARMUP_INTERVALS=${NCCL_PHASE3_WARMUP_INTERVALS:-4}
NCCL_PHASE3_HI_INTERVALS=${NCCL_PHASE3_HI_INTERVALS:-2}
NCCL_PHASE3_LO_INTERVALS=${NCCL_PHASE3_LO_INTERVALS:-8}
NCCL_PHASE3_OCC_RATIO_HIGH_PCT=${NCCL_PHASE3_OCC_RATIO_HIGH_PCT:-90}
NCCL_PHASE3_LAG_RATIO_HIGH_PCT=${NCCL_PHASE3_LAG_RATIO_HIGH_PCT:-100}
NCCL_PHASE3_DELAY_RATIO_HIGH_PCT=${NCCL_PHASE3_DELAY_RATIO_HIGH_PCT:-125}

SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-1}
DPU_NODE_HOST=${DPU_NODE_HOST:-172.16.0.100}
DPU_NODE_USER=${DPU_NODE_USER:-ubuntu}
DPU_NODE_PORT=${DPU_NODE_PORT:-22}
NETWORK_NODE_PORT=${NETWORK_NODE_PORT:-}
SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT:-/home/ubuntu/hyoeun/switch_setup_task/switch_congestion_logger}
SWITCH_LOG_INTERVAL_SEC=${SWITCH_LOG_INTERVAL_SEC:-1}
SWITCH_LOG_SHARED_ROOT=${SWITCH_LOG_SHARED_ROOT:-/mnt/nfs/cts_experiments/switch_log}
SWITCH_METADATA_FILE=${SWITCH_METADATA_FILE:-${MATRIX_ROOT}/switch_logger.env}

mkdir -p "${MATRIX_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[phase3-matrix] docker is required on the master host." >&2
  exit 1
fi

if [[ "$(hostname -s)" != "${MASTER_SERVER}" ]]; then
  echo "[phase3-matrix] run this script on ${MASTER_SERVER}. current host=$(hostname -s)" >&2
  exit 1
fi

IFS=',' read -r -a ALL_WORKER_ARRAY <<< "${ALL_WORKERS}"
IFS=',' read -r -a MODE_VALUES <<< "${RUN_MODES}"
NNODES=${#ALL_WORKER_ARRAY[@]}

if (( NNODES == 0 )); then
  echo "[phase3-matrix] ALL_WORKERS is empty." >&2
  exit 1
fi

if [[ "${ALL_WORKER_ARRAY[0]}" != "${MASTER_SERVER}" ]]; then
  echo "[phase3-matrix] ALL_WORKERS must start with MASTER_SERVER=${MASTER_SERVER}." >&2
  exit 1
fi

require_sshpass() {
  if [[ -n "${WORKER_SSH_PASSWORD}" || "${SWITCH_LOG_ENABLE}" == "1" ]] && ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase3-matrix] sshpass is required when WORKER_SSH_PASSWORD is set or switch logging is enabled." >&2
    exit 1
  fi
  if [[ "${SWITCH_LOG_ENABLE}" == "1" && ! -n "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase3-matrix] DPU_NODE_PWD must be set when SWITCH_LOG_ENABLE=1." >&2
    exit 1
  fi
}

resolve_worker_ssh_user() {
  local worker=$1
  local mapping entry map_worker map_user
  if [[ -n "${WORKER_SSH_USER_MAP}" ]]; then
    IFS=',' read -r -a mapping <<< "${WORKER_SSH_USER_MAP}"
    for entry in "${mapping[@]}"; do
      map_worker=${entry%%=*}
      map_user=${entry#*=}
      if [[ "${map_worker}" == "${worker}" && -n "${map_user}" ]]; then
        echo "${map_user}"
        return 0
      fi
    done
  fi
  if [[ -n "${WORKER_SSH_USER}" ]]; then
    echo "${WORKER_SSH_USER}"
    return 0
  fi
  echo "${worker}"
}

resolve_worker_ssh_host() {
  local worker=$1
  local ssh_host
  if [[ ! -f "${NETWORK_TOPOLOGY_FILE}" ]]; then
    echo "[phase3-matrix] NETWORK_TOPOLOGY_FILE not found: ${NETWORK_TOPOLOGY_FILE}" >&2
    return 1
  fi
  ssh_host=$(awk -v target="${worker}" '
    /^\[Workers\]/ { in_workers=1; next }
    /^\[/ && $0 !~ /^\[Workers\]/ { in_workers=0 }
    in_workers && $1 == "-" {
      gsub(":", "", $2)
      if ($2 == target) {
        print $3
        exit
      }
    }
  ' "${NETWORK_TOPOLOGY_FILE}")
  if [[ -z "${ssh_host}" ]]; then
    echo "[phase3-matrix] failed to resolve SSH host for worker=${worker} from ${NETWORK_TOPOLOGY_FILE}" >&2
    return 1
  fi
  echo "${ssh_host}"
}

resolve_remote_repo_root() {
  local worker=$1
  local ssh_user root mapping entry map_worker map_root path_tail
  ssh_user=$(resolve_worker_ssh_user "${worker}")
  if [[ -n "${REMOTE_REPO_ROOT_MAP}" ]]; then
    IFS=',' read -r -a mapping <<< "${REMOTE_REPO_ROOT_MAP}"
    for entry in "${mapping[@]}"; do
      map_worker=${entry%%=*}
      map_root=${entry#*=}
      if [[ "${map_worker}" == "${worker}" && -n "${map_root}" ]]; then
        echo "${map_root}"
        return 0
      fi
    done
  fi
  root="${REMOTE_REPO_ROOT}"
  root="${root//\{worker\}/${worker}}"
  root="${root//\{user\}/${ssh_user}}"
  if [[ "${root}" == "${REMOTE_REPO_ROOT}" && "${root}" == /home/*/* ]]; then
    path_tail=${root#/home/*/}
    root="/home/${ssh_user}/${path_tail}"
  fi
  echo "${root}"
}

resolve_worker_ssh_port() {
  local worker=$1
  local mapping entry map_worker map_port
  if [[ -n "${WORKER_SSH_PORT_MAP}" ]]; then
    IFS=',' read -r -a mapping <<< "${WORKER_SSH_PORT_MAP}"
    for entry in "${mapping[@]}"; do
      map_worker=${entry%%=*}
      map_port=${entry#*=}
      if [[ "${map_worker}" == "${worker}" && -n "${map_port}" ]]; then
        echo "${map_port}"
        return 0
      fi
    done
  fi
  echo "${WORKER_SSH_PORT}"
}

remote_worker_bash() {
  local worker=$1
  local cmd=$2
  local ssh_user
  local ssh_host
  local ssh_port
  if [[ "${worker}" == "${MASTER_SERVER}" ]]; then
    bash -lc "${cmd}"
    return
  fi

  require_sshpass
  ssh_user=$(resolve_worker_ssh_user "${worker}")
  ssh_host=$(resolve_worker_ssh_host "${worker}")
  ssh_port=$(resolve_worker_ssh_port "${worker}")
  if [[ -n "${WORKER_SSH_PASSWORD}" ]]; then
    sshpass -p "${WORKER_SSH_PASSWORD}" \
      ssh -p "${ssh_port}" -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${ssh_user}@${ssh_host}" \
      "bash -lc $(printf '%q' "${cmd}")"
  else
    ssh -p "${ssh_port}" -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${ssh_user}@${ssh_host}" \
      "bash -lc $(printf '%q' "${cmd}")"
  fi
}

remote_dpu_bash() {
  local cmd=$1
  require_sshpass
  sshpass -p "${DPU_NODE_PWD}" \
    ssh -p "${DPU_NODE_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${DPU_NODE_USER}@${DPU_NODE_HOST}" \
    "bash -lc $(printf '%q' "${cmd}")"
}

SWITCH_LOG_STARTED=0
SWITCH_LOG_RUN_ID=
SWITCH_LOG_DIR=
SWITCH_LOG_LOCAL_DIR=
SWITCH_LOG_PID_FILE=
SWITCH_LOG_MARKERS_JSONL=
CLEANUP_DONE=0

write_switch_metadata() {
  cat > "${SWITCH_METADATA_FILE}" <<EOF
SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE}
SWITCH_LOG_RUN_ID=${SWITCH_LOG_RUN_ID}
SWITCH_LOG_DIR=${SWITCH_LOG_DIR}
SWITCH_LOG_LOCAL_DIR=${SWITCH_LOG_LOCAL_DIR}
SWITCH_LOG_PID_FILE=${SWITCH_LOG_PID_FILE}
SWITCH_LOG_MARKERS_JSONL=${SWITCH_LOG_MARKERS_JSONL}
EOF
}

start_switch_logger() {
  if [[ -z "${NETWORK_NODE_PASSWORD:-}" || -z "${SWITCH_PASSWORD:-}" ]]; then
    echo "[phase3-matrix] NETWORK_NODE_PASSWORD and SWITCH_PASSWORD must be set when SWITCH_LOG_ENABLE=1." >&2
    exit 1
  fi
  local cmd
  local output
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && "
  if [[ -n "${NETWORK_NODE_PORT}" ]]; then
    cmd+="export NETWORK_NODE_PORT=$(printf '%q' "${NETWORK_NODE_PORT}") && "
  fi
  cmd+="./start_switch_congestion_loggers.sh --interval-sec $(printf '%q' "${SWITCH_LOG_INTERVAL_SEC}") --network-node-password $(printf '%q' "${NETWORK_NODE_PASSWORD}") --switch-password $(printf '%q' "${SWITCH_PASSWORD}")"
  output=$(remote_dpu_bash "${cmd}")
  while IFS='=' read -r key value; do
    case "${key}" in
      RUN_ID) SWITCH_LOG_RUN_ID=${value} ;;
      LOG_DIR) SWITCH_LOG_DIR=${value} ;;
      PID_FILE) SWITCH_LOG_PID_FILE=${value} ;;
      MARKERS_JSONL) SWITCH_LOG_MARKERS_JSONL=${value} ;;
    esac
  done <<< "${output}"
  if [[ -z "${SWITCH_LOG_RUN_ID}" || -z "${SWITCH_LOG_PID_FILE}" ]]; then
    echo "[phase3-matrix] failed to parse switch logger metadata" >&2
    echo "${output}" >&2
    exit 1
  fi
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
  write_switch_metadata
  SWITCH_LOG_STARTED=1
  echo "[phase3-matrix] switch logger started run_id=${SWITCH_LOG_RUN_ID} local_dir=${SWITCH_LOG_LOCAL_DIR}"
}

emit_switch_marker() {
  local marker=$1
  local message=${2:-}
  local source_tag=${3:-phase3_matrix}
  [[ "${SWITCH_LOG_ENABLE}" == "1" ]] || return 0
  [[ -n "${SWITCH_LOG_RUN_ID}" ]] || return 0
  local cmd
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --marker $(printf '%q' "${marker}") --source $(printf '%q' "${source_tag}")"
  if [[ -n "${message}" ]]; then
    cmd+=" --message $(printf '%q' "${message}")"
  fi
  remote_dpu_bash "${cmd}" >/dev/null
}

emit_switch_marker_best_effort() {
  emit_switch_marker "$@" || true
}

stop_switch_logger() {
  [[ "${SWITCH_LOG_ENABLE}" == "1" ]] || return 0
  [[ -n "${SWITCH_LOG_PID_FILE}" ]] || return 0
  local cmd
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./stop_switch_congestion_loggers.sh --pid-file $(printf '%q' "${SWITCH_LOG_PID_FILE}")"
  remote_dpu_bash "${cmd}" >/dev/null || true
  echo "[phase3-matrix] switch logger stopped run_id=${SWITCH_LOG_RUN_ID}"
}

cleanup() {
  if (( CLEANUP_DONE == 1 )); then
    return 0
  fi
  CLEANUP_DONE=1
  if (( SWITCH_LOG_STARTED == 1 )); then
    emit_switch_marker_best_effort "matrix_end" "run_id=${RUN_ID}" "phase3_matrix"
    stop_switch_logger
  fi
}
trap cleanup EXIT INT TERM

prepare_worker_container() {
  local worker=$1
  local remote_repo_root
  local cmd
  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  if [[ "${CONTAINER_RESET_AT_START}" == "1" ]]; then
    cmd="docker rm -f $(printf '%q' "${CONTAINER_NAME}") >/dev/null 2>&1 || true; cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh true >/dev/null"
  else
    cmd="cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh true >/dev/null"
  fi
  echo "[phase3-matrix] prepare container worker=${worker} reset=${CONTAINER_RESET_AT_START}"
  remote_worker_bash "${worker}" "${cmd}"
}

report_master_port() {
  echo "[phase3-matrix] master port inspection port=${MASTER_PORT}"
  ss -ltnp 2>/dev/null | awk -v port=":${MASTER_PORT}" 'NR == 1 || index($4, port) { print }' || true
  echo "[phase3-matrix] candidate local torchrun/python processes:"
  ps -ef | grep -E 'torchrun|torch\.distributed\.run|collective_b2\.py|collective_b3\.py' | grep -v grep || true
}

wait_for_master_port_free() {
  local timeout_sec=${1:-30}
  local waited=0
  while (( waited < timeout_sec )); do
    if ! ss -ltn 2>/dev/null | awk -v port=":${MASTER_PORT}" 'index($4, port) { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[phase3-matrix] master port is free port=${MASTER_PORT}"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[phase3-matrix] ERROR: master port remained busy port=${MASTER_PORT}" >&2
  report_master_port
  return 1
}

wait_for_master_port_listen() {
  local rank0_pid=$1
  local waited=0
  while (( waited < PORT_READY_TIMEOUT_SEC )); do
    if ss -ltn 2>/dev/null | awk -v port=":${MASTER_PORT}" 'index($4, port) { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[phase3-matrix] rank0 port is listening port=${MASTER_PORT}"
      return 0
    fi
    if ! kill -0 "${rank0_pid}" >/dev/null 2>&1; then
      echo "[phase3-matrix] ERROR: rank0 launcher exited before opening port=${MASTER_PORT}" >&2
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[phase3-matrix] ERROR: timed out waiting for rank0 port listen port=${MASTER_PORT}" >&2
  report_master_port
  return 1
}

write_pending_status() {
  local status_file=$1
  local worker=$2
  local rank=$3
  local mode=$4
  mkdir -p "$(dirname "${status_file}")"
  cat > "${status_file}" <<EOF
status=9
state=pending
worker=${worker}
node_rank=${rank}
mode=${mode}
experiment=${CURRENT_EXPERIMENT_LABEL}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
rc=
pid=
updated_at=$(date +%s)
message=pending
EOF
}

read_status_field() {
  local file=$1
  local key=$2
  awk -F= -v k="${key}" '$1 == k { print substr($0, length(k) + 2); exit }' "${file}" 2>/dev/null || true
}

cleanup_worker_processes() {
  local worker=$1
  local remote_repo_root
  local cmd
  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  cmd="cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh bash -lc $(printf '%q' "pkill -f 'torchrun|torch\\.distributed\\.run|collective_b2\\.py|collective_b3\\.py' >/dev/null 2>&1 || true; sleep 1; ps -ef | grep -E 'torchrun|torch\\.distributed\\.run|collective_b2\\.py|collective_b3\\.py' | grep -v grep || true")"
  remote_worker_bash "${worker}" "${cmd}" >/dev/null 2>&1 || true
}

cleanup_all_workers() {
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    cleanup_worker_processes "${worker}"
  done
}

restart_all_containers() {
  local worker
  local original_reset="${CONTAINER_RESET_AT_START}"
  CONTAINER_RESET_AT_START=1
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    prepare_worker_container "${worker}"
  done
  CONTAINER_RESET_AT_START="${original_reset}"
}

mode_upper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

validate_mode() {
  case "$1" in
    stock|b3) ;;
    *)
      echo "[phase3-matrix] unsupported mode=$1. RUN_MODES must contain only stock,b3." >&2
      return 1
      ;;
  esac
}

build_worker_host_command() {
  local worker=$1
  local rank=$2
  local mode=$3
  local run_root=$4
  local status_file=$5
  local remote_repo_root

  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  local inner
  inner=$(cat <<EOF
cd $(printf '%q' "${CONTAINER_REPO_ROOT}") && \
RUN_ID=$(printf '%q' "${CURRENT_EXPERIMENT_ID}") \
EXPERIMENT_LABEL=$(printf '%q' "${CURRENT_EXPERIMENT_LABEL}") \
MODE=$(printf '%q' "${mode}") \
MASTER_ADDR=$(printf '%q' "${MASTER_ADDR}") \
MASTER_PORT=$(printf '%q' "${MASTER_PORT}") \
NNODES=$(printf '%q' "${NNODES}") \
NPROC_PER_NODE=1 \
NODE_RANK=$(printf '%q' "${rank}") \
WORKER_NAME=$(printf '%q' "${worker}") \
MASTER_SERVER=$(printf '%q' "${MASTER_SERVER}") \
TORCH_ENV=$(printf '%q' "${TORCH_ENV}") \
TARGET_SCRIPT=$(printf '%q' "${TARGET_SCRIPT}") \
LOG_ROOT=$(printf '%q' "${CURRENT_EXP_ROOT}") \
RUN_ROOT=$(printf '%q' "${run_root}") \
STATUS_FILE=$(printf '%q' "${status_file}") \
STATUS_DIR=$(printf '%q' "$(dirname "${status_file}")") \
COLLECTIVE=$(printf '%q' "${CURRENT_COLLECTIVE}") \
STEPS=$(printf '%q' "${STEPS}") \
WARMUP_STEPS=$(printf '%q' "${WARMUP_STEPS}") \
PAYLOAD_MB=$(printf '%q' "${PAYLOAD_MB}") \
DTYPE=$(printf '%q' "${DTYPE}") \
SLEEP_MS=$(printf '%q' "${SLEEP_MS}") \
MODE_TIMEOUT_SEC=$(printf '%q' "${MODE_TIMEOUT_SEC}") \
NCCL_ALGO=$(printf '%q' "${CURRENT_ALGO}") \
NCCL_PROTO=auto \
NCCL_PHASE3_WARMUP_INTERVALS=$(printf '%q' "${NCCL_PHASE3_WARMUP_INTERVALS}") \
NCCL_PHASE3_HI_INTERVALS=$(printf '%q' "${NCCL_PHASE3_HI_INTERVALS}") \
NCCL_PHASE3_LO_INTERVALS=$(printf '%q' "${NCCL_PHASE3_LO_INTERVALS}") \
NCCL_PHASE3_OCC_RATIO_HIGH_PCT=$(printf '%q' "${NCCL_PHASE3_OCC_RATIO_HIGH_PCT}") \
NCCL_PHASE3_LAG_RATIO_HIGH_PCT=$(printf '%q' "${NCCL_PHASE3_LAG_RATIO_HIGH_PCT}") \
NCCL_PHASE3_DELAY_RATIO_HIGH_PCT=$(printf '%q' "${NCCL_PHASE3_DELAY_RATIO_HIGH_PCT}") \
PHASE3_MODE=$(printf '%q' "${mode}") \
bash $(printf '%q' "${WORKER_SCRIPT}")
EOF
)

  cat <<EOF
cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh bash -lc $(printf '%q' "${inner}")
EOF
}

declare -A LAUNCH_PIDS=()
declare -A LAUNCH_LOGS=()

launch_worker_mode() {
  local worker=$1
  local rank=$2
  local mode=$3
  local run_root=$4
  local status_file=$5
  local mode_upper_value=$6
  local worker_log_root="${run_root}/${worker}"
  local launcher_log="${worker_log_root}/launcher.log"
  mkdir -p "${worker_log_root}"
  write_pending_status "${status_file}" "${worker}" "${rank}" "${mode_upper_value}"
  local host_cmd
  host_cmd=$(build_worker_host_command "${worker}" "${rank}" "${mode}" "${run_root}" "${status_file}")

  echo "[phase3-matrix] launch worker=${worker} rank=${rank} mode=${mode_upper_value}"
  if [[ "${worker}" == "${MASTER_SERVER}" ]]; then
    bash -lc "${host_cmd}" >"${launcher_log}" 2>&1 &
  else
    remote_worker_bash "${worker}" "${host_cmd}" >"${launcher_log}" 2>&1 &
  fi
  LAUNCH_PIDS["${worker}"]=$!
  LAUNCH_LOGS["${worker}"]="${launcher_log}"
}

status_snapshot() {
  local status_dir=$1
  local snapshot=""
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    local status_file="${status_dir}/${worker}.status"
    local status="missing"
    if [[ -f "${status_file}" ]]; then
      status=$(read_status_field "${status_file}" "status")
    fi
    snapshot+="${worker}:${status} "
  done
  echo "${snapshot% }"
}

wait_for_mode_completion() {
  local status_dir=$1
  local mode_upper_value=$2
  local start_ts=$3
  while true; do
    local success_count=0
    local running_count=0
    local failed_count=0
    local pending_count=0
    local worker
    for worker in "${ALL_WORKER_ARRAY[@]}"; do
      local status_file="${status_dir}/${worker}.status"
      local status="missing"
      if [[ -f "${status_file}" ]]; then
        status=$(read_status_field "${status_file}" "status")
      fi
      case "${status}" in
        0) success_count=$((success_count + 1)) ;;
        1) running_count=$((running_count + 1)) ;;
        -1) failed_count=$((failed_count + 1)) ;;
        *) pending_count=$((pending_count + 1)) ;;
      esac
    done

    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[phase3-matrix] mode=${mode_upper_value} elapsed=${elapsed}s success=${success_count}/${NNODES} running=${running_count} pending=${pending_count} failed=${failed_count} statuses=$(status_snapshot "${status_dir}")"

    if (( failed_count > 0 )); then
      return 1
    fi
    if (( success_count == NNODES )); then
      return 0
    fi
    if (( elapsed >= MODE_TIMEOUT_SEC )); then
      echo "[phase3-matrix] ERROR: mode timeout reached mode=${mode_upper_value} timeout=${MODE_TIMEOUT_SEC}s" >&2
      return 2
    fi
    sleep "${STATUS_POLL_SEC}"
  done
}

wait_for_launchers() {
  local rc=0
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    local pid="${LAUNCH_PIDS[${worker}]:-}"
    if [[ -n "${pid}" ]]; then
      wait "${pid}" || rc=1
    fi
  done
  return "${rc}"
}

print_failed_worker_logs() {
  local status_dir=$1
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    local status_file="${status_dir}/${worker}.status"
    if [[ -f "${status_file}" ]] && [[ "$(read_status_field "${status_file}" "status")" == "-1" ]]; then
      local launcher_log="${LAUNCH_LOGS[${worker}]:-}"
      echo "[phase3-matrix] failed worker=${worker} launcher_log=${launcher_log}"
      if [[ -n "${launcher_log}" && -f "${launcher_log}" ]]; then
        tail -n 80 "${launcher_log}" || true
      fi
    fi
  done
}

write_env_setup() {
  local exp_root=$1
  local switch_json=""
  if [[ "${SWITCH_LOG_ENABLE}" == "1" && -n "${SWITCH_LOG_RUN_ID:-}" ]]; then
    switch_json=$(cat <<EOF
,
  "switch_log_enable": 1,
  "switch_log_run_id": "${SWITCH_LOG_RUN_ID}",
  "switch_log_dir": "${SWITCH_LOG_DIR:-}",
  "switch_log_local_dir": "${SWITCH_LOG_LOCAL_DIR:-}",
  "switch_log_markers_jsonl": "${SWITCH_LOG_MARKERS_JSONL:-}",
  "dpu_node_host": "${DPU_NODE_HOST}",
  "master_server": "${MASTER_SERVER}"
EOF
)
  fi

  cat > "${exp_root}/env_setup.json" <<EOF
{
  "phase": "B3",
  "run_id": "${CURRENT_EXPERIMENT_ID}",
  "experiment_label": "${CURRENT_EXPERIMENT_LABEL}",
  "collective": "${CURRENT_COLLECTIVE}",
  "run_modes": "${RUN_MODES}",
  "master_addr": "${MASTER_ADDR}",
  "master_port": ${MASTER_PORT},
  "nnodes": ${NNODES},
  "steps": ${STEPS},
  "warmup_steps": ${WARMUP_STEPS},
  "payload_mb": ${PAYLOAD_MB},
  "dtype": "${DTYPE}",
  "nccl_algo": "${CURRENT_ALGO}",
  "nccl_proto": "auto",
  "policy_name": "runtime_receiver_window_b3",
  "policy_formula": "W_eff = clamp(W_min, W_base, min(W_sem, W_fb)); topology prior excluded",
  "mode_contract": "STOCK: phase3 disabled; B3: phase3 runtime feedback enabled",
  "phase1_static_w": 0,
  "phase2_b2_enable": 0,
  "phase3_b3_enable_for_b3_mode": 1,
  "phase3_warmup_intervals": ${NCCL_PHASE3_WARMUP_INTERVALS},
  "phase3_hi_intervals": ${NCCL_PHASE3_HI_INTERVALS},
  "phase3_lo_intervals": ${NCCL_PHASE3_LO_INTERVALS},
  "phase3_occ_ratio_high_pct": ${NCCL_PHASE3_OCC_RATIO_HIGH_PCT},
  "phase3_lag_ratio_high_pct": ${NCCL_PHASE3_LAG_RATIO_HIGH_PCT},
  "phase3_delay_ratio_high_pct": ${NCCL_PHASE3_DELAY_RATIO_HIGH_PCT},
  "execution_model": "master_orchestrated_single_port",
  "worker_pool": "${ALL_WORKERS}",
  "network_topology_file": "${NETWORK_TOPOLOGY_FILE}",
  "remote_repo_root": "${REMOTE_REPO_ROOT}",
  "remote_repo_root_map": "${REMOTE_REPO_ROOT_MAP}",
  "worker_ssh_port": "${WORKER_SSH_PORT}",
  "worker_ssh_port_map": "${WORKER_SSH_PORT_MAP}",
  "dpu_node_port": "${DPU_NODE_PORT}",
  "network_node_port": "${NETWORK_NODE_PORT}",
  "status_ddp_contract": "1=running,0=success,-1=failed",
  "mode_timeout_sec": ${MODE_TIMEOUT_SEC},
  "container_name": "${CONTAINER_NAME}",
  "container_reset_at_start": 0${switch_json}
}
EOF
}

compare_outputs() {
  local exp_root=$1
  local validation_json="${exp_root}/final_output_validation.json"
  if [[ ! -f "${REPO_ROOT}/${COMPARE_SCRIPT}" ]]; then
    echo "[phase3-matrix] compare script missing path=${COMPARE_SCRIPT}"
    return 0
  fi
  echo "[phase3-matrix] validating final outputs against STOCK"
  python3 "${REPO_ROOT}/${COMPARE_SCRIPT}" \
    --experiment-root "${exp_root}" \
    --output-json "${validation_json}" || true
}

run_mode() {
  local mode=$1
  local mode_upper_value
  local run_root
  local status_dir
  local rank0_pid
  local mode_start_ts
  local mode_rc=0
  mode_upper_value=$(mode_upper "${mode}")
  run_root="${CURRENT_EXP_ROOT}/${mode_upper_value}"
  status_dir="${CURRENT_EXP_ROOT}/.ddp_status/${mode_upper_value}"
  mkdir -p "${run_root}" "${status_dir}"

  echo "[phase3-matrix] ------------------------------------------------------------"
  echo "[phase3-matrix] start mode=${mode_upper_value} port=${MASTER_PORT}"

  cleanup_all_workers
  wait_for_master_port_free 10
  emit_switch_marker "mode_start" "run_id=${RUN_ID} experiment=${CURRENT_EXPERIMENT_ID} mode=${mode_upper_value} collective=${CURRENT_COLLECTIVE} algo=${CURRENT_ALGO} payload_mb=${PAYLOAD_MB} workers=${ALL_WORKERS} master_port=${MASTER_PORT}" "phase3_collective"

  unset LAUNCH_PIDS
  unset LAUNCH_LOGS
  declare -g -A LAUNCH_PIDS=()
  declare -g -A LAUNCH_LOGS=()

  launch_worker_mode "${ALL_WORKER_ARRAY[0]}" 0 "${mode}" "${run_root}" "${status_dir}/${ALL_WORKER_ARRAY[0]}.status" "${mode_upper_value}"

  rank0_pid="${LAUNCH_PIDS[${ALL_WORKER_ARRAY[0]}]}"
  if ! wait_for_master_port_listen "${rank0_pid}"; then
    mode_rc=1
    print_failed_worker_logs "${status_dir}"
  else
    for idx in "${!ALL_WORKER_ARRAY[@]}"; do
      if (( idx == 0 )); then
        continue
      fi
      worker="${ALL_WORKER_ARRAY[$idx]}"
      launch_worker_mode "${worker}" "${idx}" "${mode}" "${run_root}" "${status_dir}/${worker}.status" "${mode_upper_value}"
    done

    mode_start_ts=$(date +%s)
    if ! wait_for_mode_completion "${status_dir}" "${mode_upper_value}" "${mode_start_ts}"; then
      mode_rc=1
      print_failed_worker_logs "${status_dir}"
    fi
  fi

  wait_for_launchers || mode_rc=1
  emit_switch_marker_best_effort "mode_end" "run_id=${RUN_ID} experiment=${CURRENT_EXPERIMENT_ID} mode=${mode_upper_value} collective=${CURRENT_COLLECTIVE} algo=${CURRENT_ALGO} payload_mb=${PAYLOAD_MB} rc=${mode_rc} master_port=${MASTER_PORT}" "phase3_collective"

  cleanup_all_workers
  wait_for_master_port_free 15 || mode_rc=1

  if (( mode_rc != 0 )) && [[ "${RESTART_CONTAINERS_ON_FAILURE}" == "1" ]]; then
    echo "[phase3-matrix] restarting containers after failed mode=${mode_upper_value}"
    restart_all_containers
  fi

  if (( mode_rc == 0 )); then
    echo "[phase3-matrix] complete mode=${mode_upper_value}"
  else
    echo "[phase3-matrix] failed mode=${mode_upper_value}" >&2
  fi
  return "${mode_rc}"
}

EXPERIMENT_IDS=()
EXPERIMENT_COLLS=()
EXPERIMENT_ALGOS=()

add_experiment() {
  EXPERIMENT_IDS+=("$1")
  EXPERIMENT_COLLS+=("$2")
  EXPERIMENT_ALGOS+=("$3")
}

add_experiment "allreduce_ring_${PAYLOAD_MB}mb" "allreduce" "Ring"
add_experiment "allreduce_tree_${PAYLOAD_MB}mb" "allreduce" "Tree"
add_experiment "allgather_auto_${PAYLOAD_MB}mb" "allgather" "auto"
add_experiment "reducescatter_auto_${PAYLOAD_MB}mb" "reducescatter" "auto"
add_experiment "alltoall_auto_${PAYLOAD_MB}mb" "alltoall" "auto"

for mode in "${MODE_VALUES[@]}"; do
  validate_mode "${mode}"
done

MANIFEST_JSON="${MATRIX_ROOT}/matrix_manifest.json"

echo "[phase3-matrix] RUN_ID=${RUN_ID}"
echo "[phase3-matrix] MASTER_SERVER=${MASTER_SERVER} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[phase3-matrix] WORKERS=${ALL_WORKERS}"
echo "[phase3-matrix] MODES=${RUN_MODES}"
echo "[phase3-matrix] TOTAL_EXPERIMENTS=${#EXPERIMENT_IDS[@]}"
echo "[phase3-matrix] MATRIX_ROOT=${MATRIX_ROOT}"

if [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
  start_switch_logger
  emit_switch_marker "matrix_start" "run_id=${RUN_ID} modes=${RUN_MODES} workers=${ALL_WORKERS} payload_mb=${PAYLOAD_MB}" "phase3_matrix"
fi

for worker in "${ALL_WORKER_ARRAY[@]}"; do
  prepare_worker_container "${worker}"
done

{
  echo "{"
  echo "  \"run_id\": \"${RUN_ID}\","
  echo "  \"master_server\": \"${MASTER_SERVER}\","
  echo "  \"master_addr\": \"${MASTER_ADDR}\","
  echo "  \"master_port\": ${MASTER_PORT},"
  echo "  \"run_modes\": \"${RUN_MODES}\","
  echo "  \"dtype\": \"${DTYPE}\","
  echo "  \"payload_mb\": ${PAYLOAD_MB},"
  echo "  \"steps\": ${STEPS},"
  echo "  \"warmup_steps\": ${WARMUP_STEPS},"
  echo "  \"mode_timeout_sec\": ${MODE_TIMEOUT_SEC},"
  echo "  \"worker_pool\": \"${ALL_WORKERS}\","
  echo "  \"execution_model\": \"master_orchestrated_single_port\","
  echo "  \"target_script\": \"${TARGET_SCRIPT}\","
  echo "  \"worker_script\": \"${WORKER_SCRIPT}\","
  echo "  \"network_topology_file\": \"${NETWORK_TOPOLOGY_FILE}\","
  echo "  \"remote_repo_root\": \"${REMOTE_REPO_ROOT}\","
  echo "  \"remote_repo_root_map\": \"${REMOTE_REPO_ROOT_MAP}\","
  echo "  \"worker_ssh_port\": \"${WORKER_SSH_PORT}\","
  echo "  \"worker_ssh_port_map\": \"${WORKER_SSH_PORT_MAP}\","
  echo "  \"dpu_node_port\": \"${DPU_NODE_PORT}\","
  echo "  \"network_node_port\": \"${NETWORK_NODE_PORT}\","
  echo "  \"container_name\": \"${CONTAINER_NAME}\","
  echo "  \"container_reset_at_start\": ${CONTAINER_RESET_AT_START},"
  echo "  \"switch_log_enable\": ${SWITCH_LOG_ENABLE},"
  if [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
    echo "  \"switch_log_run_id\": \"${SWITCH_LOG_RUN_ID}\","
    echo "  \"switch_log_dir\": \"${SWITCH_LOG_DIR}\","
    echo "  \"switch_log_local_dir\": \"${SWITCH_LOG_LOCAL_DIR}\","
  fi
  echo "  \"experiments\": ["
  for idx in "${!EXPERIMENT_IDS[@]}"; do
    comma=","
    if (( idx == ${#EXPERIMENT_IDS[@]} - 1 )); then
      comma=""
    fi
    cat <<EOF
    {
      "index": ${idx},
      "id": "${EXPERIMENT_IDS[$idx]}",
      "collective": "${EXPERIMENT_COLLS[$idx]}",
      "algo": "${EXPERIMENT_ALGOS[$idx]}",
      "payload_mb": ${PAYLOAD_MB}
    }${comma}
EOF
  done
  echo "  ]"
  echo "}"
} > "${MANIFEST_JSON}"

overall_rc=0
for idx in "${!EXPERIMENT_IDS[@]}"; do
  CURRENT_EXPERIMENT_ID=${EXPERIMENT_IDS[$idx]}
  CURRENT_EXPERIMENT_LABEL=${CURRENT_EXPERIMENT_ID}
  CURRENT_COLLECTIVE=${EXPERIMENT_COLLS[$idx]}
  CURRENT_ALGO=${EXPERIMENT_ALGOS[$idx]}
  CURRENT_EXP_ROOT="${MATRIX_ROOT}/$(printf "%02d_%s" "$((idx+1))" "${CURRENT_EXPERIMENT_ID}")"
  status_dir="${MATRIX_ROOT}/.matrix_status/$(printf "%02d_%s" "$((idx+1))" "${CURRENT_EXPERIMENT_ID}")"
  mkdir -p "${CURRENT_EXP_ROOT}" "${status_dir}"

  echo "[phase3-matrix] ============================================================"
  echo "[phase3-matrix] start idx=${idx} id=${CURRENT_EXPERIMENT_ID} coll=${CURRENT_COLLECTIVE} algo=${CURRENT_ALGO} payload=${PAYLOAD_MB}MB"
  emit_switch_marker_best_effort "exp_start" "run_id=${RUN_ID} experiment=${CURRENT_EXPERIMENT_ID} collective=${CURRENT_COLLECTIVE} algo=${CURRENT_ALGO} payload_mb=${PAYLOAD_MB} workers=${ALL_WORKERS}" "phase3_matrix"

  write_env_setup "${CURRENT_EXP_ROOT}"

  exp_rc=0
  for mode in "${MODE_VALUES[@]}"; do
    if ! run_mode "${mode}"; then
      exp_rc=1
      overall_rc=1
      if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
        break
      fi
    fi
  done

  compare_outputs "${CURRENT_EXP_ROOT}"

  {
    echo "rc=${exp_rc}"
    echo "experiment=${CURRENT_EXPERIMENT_ID}"
    echo "updated_at=$(date +%s)"
  } > "${status_dir}/summary.status"

  if (( exp_rc != 0 )); then
    echo "[phase3-matrix] experiment failed idx=${idx} id=${CURRENT_EXPERIMENT_ID}"
  else
    echo "[phase3-matrix] experiment complete idx=${idx} id=${CURRENT_EXPERIMENT_ID}"
  fi

  emit_switch_marker_best_effort "exp_end" "run_id=${RUN_ID} experiment=${CURRENT_EXPERIMENT_ID} collective=${CURRENT_COLLECTIVE} algo=${CURRENT_ALGO} payload_mb=${PAYLOAD_MB} rc=${exp_rc}" "phase3_matrix"

  if (( exp_rc != 0 )) && [[ "${STOP_ON_FAILURE}" == "1" ]]; then
    break
  fi
done

exit "${overall_rc}"
