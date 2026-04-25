#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase2_b2_collective_$(date +%y%m%d_%H%M%S)}
EXPERIMENT_LABEL=${EXPERIMENT_LABEL:-${RUN_ID}}
MASTER_SERVER=${MASTER_SERVER:-worker01}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT=${MASTER_PORT:-${MASTER_PORT_BASE:-40000}}
ALL_WORKERS=${ALL_WORKERS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
RUN_MODES=${RUN_MODES:-stock,b2_w2,b2_w4,b2_w5,b2_w6,b2_w7,b2_w8}
COLLECTIVE=${COLLECTIVE:-allreduce}
PAYLOAD_MB=${PAYLOAD_MB:-128}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
SLEEP_MS=${SLEEP_MS:-0}
ALGO_SETTING=${NCCL_ALGO:-auto}
PROTO_SETTING=${NCCL_PROTO:-auto}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-600}
PORT_READY_TIMEOUT_SEC=${PORT_READY_TIMEOUT_SEC:-60}
STATUS_POLL_SEC=${STATUS_POLL_SEC:-5}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase2/collective_b2.py}
WORKER_SCRIPT=${WORKER_SCRIPT:-research/code/phase2/run_b2_mode_worker.sh}
COMPARE_SCRIPT=${COMPARE_SCRIPT:-research/code/phase2/compare_b2_vs_stock.py}

HOST_REPO_ROOT=${HOST_REPO_ROOT:-${REPO_ROOT}}
REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-${HOST_REPO_ROOT}}
CONTAINER_REPO_ROOT=${CONTAINER_REPO_ROOT:-/workspace/$(basename "${REPO_ROOT}")}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu121-dev}
CONTAINER_RESET_AT_START=${CONTAINER_RESET_AT_START:-0}
RESTART_CONTAINERS_ON_FAILURE=${RESTART_CONTAINERS_ON_FAILURE:-1}
WORKER_SSH_USER=${WORKER_SSH_USER:-}
WORKER_SSH_USER_MAP=${WORKER_SSH_USER_MAP:-}
WORKER_SSH_PASSWORD=${WORKER_SSH_PASSWORD:-}
SSH_CONNECT_TIMEOUT_SEC=${SSH_CONNECT_TIMEOUT_SEC:-10}
NETWORK_TOPOLOGY_FILE=${NETWORK_TOPOLOGY_FILE:-${REPO_ROOT}/research/env/network_topology_internal_ips.txt}

SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-0}
SWITCH_METADATA_FILE=${SWITCH_METADATA_FILE:-}
DPU_NODE_HOST=${DPU_NODE_HOST:-172.16.0.100}
DPU_NODE_USER=${DPU_NODE_USER:-ubuntu}
SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT:-/home/ubuntu/hyoeun/switch_setup_task/switch_congestion_logger}
SWITCH_LOG_SHARED_ROOT=${SWITCH_LOG_SHARED_ROOT:-/mnt/nfs/cts_experiments/switch_log}

mkdir -p "${LOG_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[phase2-master] docker is required on the master host for container orchestration." >&2
  exit 1
fi

if [[ "$(hostname -s)" != "${MASTER_SERVER}" ]]; then
  echo "[phase2-master] run this script on ${MASTER_SERVER}. current host=$(hostname -s)" >&2
  exit 1
fi

IFS=',' read -r -a ALL_WORKER_ARRAY <<< "${ALL_WORKERS}"
IFS=',' read -r -a MODE_VALUES <<< "${RUN_MODES}"

if (( ${#ALL_WORKER_ARRAY[@]} == 0 )); then
  echo "[phase2-master] ALL_WORKERS is empty." >&2
  exit 1
fi

if [[ "${ALL_WORKER_ARRAY[0]}" != "${MASTER_SERVER}" ]]; then
  echo "[phase2-master] ALL_WORKERS must start with MASTER_SERVER=${MASTER_SERVER}." >&2
  exit 1
fi

count_in_list() {
  local list=$1
  local count=0
  local item
  IFS=',' read -r -a _items <<< "${list}"
  for item in "${_items[@]}"; do
    [[ -n "${item}" ]] && count=$((count + 1))
  done
  echo "${count}"
}

NNODES=$(count_in_list "${ALL_WORKERS}")

require_sshpass() {
  if [[ -n "${WORKER_SSH_PASSWORD}" ]] && ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase2-master] sshpass is required when WORKER_SSH_PASSWORD is set." >&2
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
    echo "[phase2-master] NETWORK_TOPOLOGY_FILE not found: ${NETWORK_TOPOLOGY_FILE}" >&2
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
    echo "[phase2-master] failed to resolve SSH host for worker=${worker} from ${NETWORK_TOPOLOGY_FILE}" >&2
    return 1
  fi
  echo "${ssh_host}"
}

remote_worker_bash() {
  local worker=$1
  local cmd=$2
  local ssh_user
  local ssh_host
  if [[ "${worker}" == "${MASTER_SERVER}" ]]; then
    bash -lc "${cmd}"
    return
  fi

  require_sshpass
  ssh_user=$(resolve_worker_ssh_user "${worker}")
  ssh_host=$(resolve_worker_ssh_host "${worker}")
  if [[ -n "${WORKER_SSH_PASSWORD}" ]]; then
    sshpass -p "${WORKER_SSH_PASSWORD}" \
      ssh -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${ssh_user}@${ssh_host}" \
      "bash -lc $(printf '%q' "${cmd}")"
  else
    ssh -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${ssh_user}@${ssh_host}" \
      "bash -lc $(printf '%q' "${cmd}")"
  fi
}

emit_switch_marker() {
  local marker=$1
  local message=${2:-}
  local source_tag=${3:-phase2_collective}
  [[ "${SWITCH_LOG_ENABLE}" == "1" ]] || return 0
  [[ -n "${SWITCH_LOG_RUN_ID:-}" ]] || return 0
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase2-master] sshpass is required when switch markers are enabled." >&2
    exit 1
  fi
  local cmd
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --marker $(printf '%q' "${marker}") --source $(printf '%q' "${source_tag}")"
  if [[ -n "${message}" ]]; then
    cmd+=" --message $(printf '%q' "${message}")"
  fi
  if [[ -z "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase2-master] DPU_NODE_PWD must be set when switch markers are enabled." >&2
    exit 1
  fi
  sshpass -p "${DPU_NODE_PWD}" \
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${DPU_NODE_USER}@${DPU_NODE_HOST}" \
    "bash -lc $(printf '%q' "${cmd}")" >/dev/null
}

if [[ -n "${SWITCH_METADATA_FILE}" && -f "${SWITCH_METADATA_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SWITCH_METADATA_FILE}"
fi
if [[ -z "${SWITCH_LOG_LOCAL_DIR:-}" && -n "${SWITCH_LOG_RUN_ID:-}" ]]; then
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
fi

mode_upper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

mode_static_w() {
  local mode=$1
  case "${mode}" in
    stock) echo 0 ;;
    b2_w*) echo "${mode#b2_w}" ;;
    *) echo "[phase2-master] unsupported mode=${mode}" >&2; return 1 ;;
  esac
}

report_master_port() {
  echo "[phase2-master] master port inspection port=${MASTER_PORT}"
  ss -ltnp 2>/dev/null | awk -v port=":${MASTER_PORT}" 'NR == 1 || index($4, port) { print }' || true
  echo "[phase2-master] candidate local torchrun/python processes:"
  ps -ef | grep -E 'torchrun|torch\.distributed\.run|collective_b2\.py' | grep -v grep || true
}

wait_for_master_port_free() {
  local timeout_sec=${1:-30}
  local waited=0
  while (( waited < timeout_sec )); do
    if ! ss -ltn 2>/dev/null | awk -v port=":${MASTER_PORT}" 'index($4, port) { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[phase2-master] master port is free port=${MASTER_PORT}"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[phase2-master] ERROR: master port remained busy port=${MASTER_PORT}" >&2
  report_master_port
  return 1
}

wait_for_master_port_listen() {
  local rank0_pid=$1
  local waited=0
  while (( waited < PORT_READY_TIMEOUT_SEC )); do
    if ss -ltn 2>/dev/null | awk -v port=":${MASTER_PORT}" 'index($4, port) { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[phase2-master] rank0 port is listening port=${MASTER_PORT}"
      return 0
    fi
    if ! kill -0 "${rank0_pid}" >/dev/null 2>&1; then
      echo "[phase2-master] ERROR: rank0 launcher exited before opening port=${MASTER_PORT}" >&2
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[phase2-master] ERROR: timed out waiting for rank0 port listen port=${MASTER_PORT}" >&2
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
experiment=${EXPERIMENT_LABEL}
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

container_bootstrap_cmd() {
  cat <<EOF
docker rm -f $(printf '%q' "${CONTAINER_NAME}") >/dev/null 2>&1 || true
cd $(printf '%q' "${REMOTE_REPO_ROOT}") && ./research/code/deploy/run_dev_container.sh true >/dev/null
EOF
}

ensure_container_ready() {
  local worker=$1
  local cmd="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && ./research/code/deploy/run_dev_container.sh true >/dev/null"
  if [[ "${CONTAINER_RESET_AT_START}" == "1" ]]; then
    cmd=$(container_bootstrap_cmd)
  fi
  echo "[phase2-master] ensuring container on ${worker}"
  remote_worker_bash "${worker}" "${cmd}"
}

cleanup_worker_processes() {
  local worker=$1
  local cmd
  cmd="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && ./research/code/deploy/run_dev_container.sh bash -lc $(printf '%q' "pkill -f 'torchrun|torch\\.distributed\\.run|collective_b2\\.py' >/dev/null 2>&1 || true; sleep 1; ps -ef | grep -E 'torchrun|torch\\.distributed\\.run|collective_b2\\.py' | grep -v grep || true")"
  remote_worker_bash "${worker}" "${cmd}" >/dev/null 2>&1 || true
}

build_worker_host_command() {
  local worker=$1
  local rank=$2
  local mode=$3
  local run_root=$4
  local status_file=$5
  local mode_upper_value=$6
  local static_w=$7

  local inner
  inner=$(cat <<EOF
cd $(printf '%q' "${CONTAINER_REPO_ROOT}") && \
RUN_ID=$(printf '%q' "${RUN_ID}") \
EXPERIMENT_LABEL=$(printf '%q' "${EXPERIMENT_LABEL}") \
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
LOG_ROOT=$(printf '%q' "${LOG_ROOT}") \
RUN_ROOT=$(printf '%q' "${run_root}") \
STATUS_FILE=$(printf '%q' "${status_file}") \
STATUS_DIR=$(printf '%q' "$(dirname "${status_file}")") \
COLLECTIVE=$(printf '%q' "${COLLECTIVE}") \
STEPS=$(printf '%q' "${STEPS}") \
WARMUP_STEPS=$(printf '%q' "${WARMUP_STEPS}") \
PAYLOAD_MB=$(printf '%q' "${PAYLOAD_MB}") \
DTYPE=$(printf '%q' "${DTYPE}") \
SLEEP_MS=$(printf '%q' "${SLEEP_MS}") \
MODE_TIMEOUT_SEC=$(printf '%q' "${MODE_TIMEOUT_SEC}") \
NCCL_ALGO=$(printf '%q' "${ALGO_SETTING}") \
NCCL_PROTO=$(printf '%q' "${PROTO_SETTING}") \
PHASE2_MODE=$(printf '%q' "${mode}") \
PHASE2_STATIC_W=$(printf '%q' "${static_w}") \
bash $(printf '%q' "${WORKER_SCRIPT}")
EOF
)

  cat <<EOF
cd $(printf '%q' "${REMOTE_REPO_ROOT}") && ./research/code/deploy/run_dev_container.sh bash -lc $(printf '%q' "${inner}")
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
  local static_w=$7
  local worker_log_root="${run_root}/${worker}"
  local launcher_log="${worker_log_root}/launcher.log"
  mkdir -p "${worker_log_root}"
  write_pending_status "${status_file}" "${worker}" "${rank}" "${mode_upper_value}"
  local host_cmd
  host_cmd=$(build_worker_host_command "${worker}" "${rank}" "${mode}" "${run_root}" "${status_file}" "${mode_upper_value}" "${static_w}")

  echo "[phase2-master] launch worker=${worker} rank=${rank} mode=${mode_upper_value} static_w=${static_w}"
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
    echo "[phase2-master] mode=${mode_upper_value} elapsed=${elapsed}s success=${success_count}/${NNODES} running=${running_count} pending=${pending_count} failed=${failed_count} statuses=$(status_snapshot "${status_dir}")"

    if (( failed_count > 0 )); then
      return 1
    fi
    if (( success_count == NNODES )); then
      return 0
    fi
    if (( elapsed >= MODE_TIMEOUT_SEC )); then
      echo "[phase2-master] ERROR: mode timeout reached mode=${mode_upper_value} timeout=${MODE_TIMEOUT_SEC}s" >&2
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
      echo "[phase2-master] failed worker=${worker} launcher_log=${launcher_log}"
      if [[ -n "${launcher_log}" && -f "${launcher_log}" ]]; then
        tail -n 80 "${launcher_log}" || true
      fi
    fi
  done
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
    ensure_container_ready "${worker}"
  done
  CONTAINER_RESET_AT_START="${original_reset}"
}

VALIDATION_JSON="${LOG_ROOT}/final_output_validation.json"

write_env_setup() {
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

  cat > "${LOG_ROOT}/env_setup.json" <<EOF
{
  "phase": "B2",
  "run_id": "${RUN_ID}",
  "experiment_label": "${EXPERIMENT_LABEL}",
  "collective": "${COLLECTIVE}",
  "run_modes": "${RUN_MODES}",
  "master_addr": "${MASTER_ADDR}",
  "master_port": ${MASTER_PORT},
  "nnodes": ${NNODES},
  "steps": ${STEPS},
  "warmup_steps": ${WARMUP_STEPS},
  "payload_mb": ${PAYLOAD_MB},
  "dtype": "${DTYPE}",
  "nccl_algo": "${ALGO_SETTING}",
  "nccl_proto": "${PROTO_SETTING}",
  "policy_name": "static_receiver_window_sweep",
  "policy_formula": "W_eff = min(W_base, W_cfg) with PHASE1_STATIC_W override",
  "phase2_b2_enable": 0,
  "phase3_b3_enable": 0,
  "b3_disabled_by_runner": true,
  "execution_model": "master_orchestrated_single_port",
  "worker_pool": "${ALL_WORKERS}",
  "network_topology_file": "${NETWORK_TOPOLOGY_FILE}",
  "ssh_host_resolution": "resolve worker internal IP from [Workers] section in network topology file",
  "status_ddp_contract": "1=running,0=success,-1=failed",
  "mode_timeout_sec": ${MODE_TIMEOUT_SEC},
  "container_name": "${CONTAINER_NAME}",
  "container_reset_at_start": ${CONTAINER_RESET_AT_START}${switch_json}
}
EOF
}

compare_outputs() {
  if [[ ! -f "${REPO_ROOT}/${COMPARE_SCRIPT}" ]]; then
    echo "[phase2-master] compare script missing path=${COMPARE_SCRIPT}"
    return 0
  fi
  echo "[phase2-master] validating final outputs against STOCK"
  python3 "${REPO_ROOT}/${COMPARE_SCRIPT}" \
    --experiment-root "${LOG_ROOT}" \
    --output-json "${VALIDATION_JSON}" || true
}

case "${COLLECTIVE}" in
  allreduce|allgather|reducescatter|alltoall) ;;
  *)
    echo "[phase2-master] unsupported COLLECTIVE=${COLLECTIVE}" >&2
    exit 1
    ;;
esac

echo "[phase2-master] RUN_ID=${RUN_ID}"
echo "[phase2-master] EXPERIMENT_LABEL=${EXPERIMENT_LABEL}"
echo "[phase2-master] WORKERS=${ALL_WORKERS}"
echo "[phase2-master] MODES=${RUN_MODES}"
echo "[phase2-master] COLLECTIVE=${COLLECTIVE} PAYLOAD_MB=${PAYLOAD_MB} NCCL_ALGO=${ALGO_SETTING} NCCL_PROTO=${PROTO_SETTING}"
echo "[phase2-master] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} TIMEOUT=${MODE_TIMEOUT_SEC}s"
echo "[phase2-master] LOG_ROOT=${LOG_ROOT}"

write_env_setup

wait_for_master_port_free 5

if [[ "${CONTAINER_RESET_AT_START}" == "1" ]]; then
  echo "[phase2-master] resetting worker containers before experiment start"
fi
for worker in "${ALL_WORKER_ARRAY[@]}"; do
  ensure_container_ready "${worker}"
done
cleanup_all_workers
wait_for_master_port_free 10

emit_switch_marker "exp_start" "experiment=${EXPERIMENT_LABEL} collective=${COLLECTIVE} algo=${ALGO_SETTING} payload_mb=${PAYLOAD_MB}" "phase2_collective"

experiment_rc=0
for mode in "${MODE_VALUES[@]}"; do
  mode_upper_value=$(mode_upper "${mode}")
  static_w=$(mode_static_w "${mode}")
  run_root="${LOG_ROOT}/${mode_upper_value}"
  status_dir="${LOG_ROOT}/.ddp_status/${mode_upper_value}"
  mkdir -p "${run_root}" "${status_dir}"

  echo "[phase2-master] ------------------------------------------------------------"
  echo "[phase2-master] start mode=${mode_upper_value} static_w=${static_w} port=${MASTER_PORT}"

  cleanup_all_workers
  wait_for_master_port_free 10
  emit_switch_marker "mode_start" "run_id=${RUN_ID} mode=${mode_upper_value} collective=${COLLECTIVE} algo=${ALGO_SETTING} payload_mb=${PAYLOAD_MB} static_w=${static_w}" "phase2_collective"

  unset LAUNCH_PIDS
  unset LAUNCH_LOGS
  declare -A LAUNCH_PIDS=()
  declare -A LAUNCH_LOGS=()

  launch_worker_mode "${ALL_WORKER_ARRAY[0]}" 0 "${mode}" "${run_root}" "${status_dir}/${ALL_WORKER_ARRAY[0]}.status" "${mode_upper_value}" "${static_w}"

  rank0_pid="${LAUNCH_PIDS[${ALL_WORKER_ARRAY[0]}]}"
  if ! wait_for_master_port_listen "${rank0_pid}"; then
    experiment_rc=1
    print_failed_worker_logs "${status_dir}"
    break
  fi

  for idx in "${!ALL_WORKER_ARRAY[@]}"; do
    if (( idx == 0 )); then
      continue
    fi
    worker="${ALL_WORKER_ARRAY[$idx]}"
    launch_worker_mode "${worker}" "${idx}" "${mode}" "${run_root}" "${status_dir}/${worker}.status" "${mode_upper_value}" "${static_w}"
  done

  mode_start_ts=$(date +%s)
  if ! wait_for_mode_completion "${status_dir}" "${mode_upper_value}" "${mode_start_ts}"; then
    experiment_rc=1
    print_failed_worker_logs "${status_dir}"
    wait_for_launchers || true
    cleanup_all_workers
    wait_for_master_port_free 15 || true
    if [[ "${RESTART_CONTAINERS_ON_FAILURE}" == "1" ]]; then
      echo "[phase2-master] restarting containers after failed mode=${mode_upper_value}"
      restart_all_containers
    fi
    break
  fi

  wait_for_launchers || experiment_rc=1
  cleanup_all_workers
  if ! wait_for_master_port_free 15; then
    experiment_rc=1
    break
  fi

  emit_switch_marker "mode_end" "run_id=${RUN_ID} mode=${mode_upper_value} collective=${COLLECTIVE} algo=${ALGO_SETTING} payload_mb=${PAYLOAD_MB} static_w=${static_w}" "phase2_collective"
  echo "[phase2-master] complete mode=${mode_upper_value}"
done

emit_switch_marker "exp_end" "experiment=${EXPERIMENT_LABEL} rc=${experiment_rc}" "phase2_collective" || true
compare_outputs

if (( experiment_rc != 0 )); then
  echo "[phase2-master] experiment failed label=${EXPERIMENT_LABEL}" >&2
  exit "${experiment_rc}"
fi

echo "[phase2-master] experiment complete label=${EXPERIMENT_LABEL}"
