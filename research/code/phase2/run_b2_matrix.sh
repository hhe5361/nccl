#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase2_b2_matrix_$(date +%y%m%d_%H%M%S)}
MASTER_SERVER=${MASTER_SERVER:-worker01}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT=${MASTER_PORT:-${MASTER_PORT_BASE:-40000}}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-/mnt/nfs_share/cts_experiments}
MATRIX_ROOT=${MATRIX_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
ALL_WORKERS=${ALL_WORKERS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
PAYLOAD_MB=${PAYLOAD_MB:-128}
RUN_MODES=${RUN_MODES:-stock,b2_w2,b2_w4,b2_w5,b2_w6,b2_w7,b2_w8}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
SLEEP_MS=${SLEEP_MS:-0}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-600}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
INNER_SCRIPT=${INNER_SCRIPT:-research/code/phase2/run_b2_collective.sh}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-${REPO_ROOT}}
REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-${HOST_REPO_ROOT}}
CONTAINER_REPO_ROOT=${CONTAINER_REPO_ROOT:-/workspace/$(basename "${REPO_ROOT}")}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu121-dev}
WORKER_SSH_USER=${WORKER_SSH_USER:-}
WORKER_SSH_USER_MAP=${WORKER_SSH_USER_MAP:-}
WORKER_SSH_PORT=${WORKER_SSH_PORT:-22}
WORKER_SSH_PORT_MAP=${WORKER_SSH_PORT_MAP:-}
WORKER_SSH_PASSWORD=${WORKER_SSH_PASSWORD:-}
SSH_CONNECT_TIMEOUT_SEC=${SSH_CONNECT_TIMEOUT_SEC:-10}
NETWORK_TOPOLOGY_FILE=${NETWORK_TOPOLOGY_FILE:-${REPO_ROOT}/research/env/network_topology_internal_ips.txt}
CONTAINER_RESET_AT_START=${CONTAINER_RESET_AT_START:-1}
RESTART_CONTAINERS_ON_FAILURE=${RESTART_CONTAINERS_ON_FAILURE:-1}
STOP_ON_FAILURE=${STOP_ON_FAILURE:-0}

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
  echo "[phase2-matrix] docker is required on the master host." >&2
  exit 1
fi

if [[ "$(hostname -s)" != "${MASTER_SERVER}" ]]; then
  echo "[phase2-matrix] run this script on ${MASTER_SERVER}. current host=$(hostname -s)" >&2
  exit 1
fi

IFS=',' read -r -a ALL_WORKER_ARRAY <<< "${ALL_WORKERS}"

if (( ${#ALL_WORKER_ARRAY[@]} == 0 )); then
  echo "[phase2-matrix] ALL_WORKERS is empty." >&2
  exit 1
fi

if [[ "${ALL_WORKER_ARRAY[0]}" != "${MASTER_SERVER}" ]]; then
  echo "[phase2-matrix] ALL_WORKERS must start with MASTER_SERVER=${MASTER_SERVER}." >&2
  exit 1
fi

require_sshpass() {
  if [[ -n "${WORKER_SSH_PASSWORD}" || "${SWITCH_LOG_ENABLE}" == "1" ]] && ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase2-matrix] sshpass is required when WORKER_SSH_PASSWORD is set or switch logging is enabled." >&2
    exit 1
  fi
  if [[ "${SWITCH_LOG_ENABLE}" == "1" && ! -n "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase2-matrix] DPU_NODE_PWD must be set when SWITCH_LOG_ENABLE=1." >&2
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
    echo "[phase2-matrix] NETWORK_TOPOLOGY_FILE not found: ${NETWORK_TOPOLOGY_FILE}" >&2
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
    echo "[phase2-matrix] failed to resolve SSH host for worker=${worker} from ${NETWORK_TOPOLOGY_FILE}" >&2
    return 1
  fi
  echo "${ssh_host}"
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
    echo "[phase2-matrix] NETWORK_NODE_PASSWORD and SWITCH_PASSWORD must be set when SWITCH_LOG_ENABLE=1." >&2
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
    echo "[phase2-matrix] failed to parse switch logger metadata" >&2
    echo "${output}" >&2
    exit 1
  fi
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
  write_switch_metadata
  SWITCH_LOG_STARTED=1
  echo "[phase2-matrix] switch logger started run_id=${SWITCH_LOG_RUN_ID} local_dir=${SWITCH_LOG_LOCAL_DIR}"
}

emit_switch_marker() {
  local marker=$1
  local message=${2:-}
  local source_tag=${3:-phase2_matrix}
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
  echo "[phase2-matrix] switch logger stopped run_id=${SWITCH_LOG_RUN_ID}"
}

prepare_worker_container() {
  local worker=$1
  local cmd
  if [[ "${CONTAINER_RESET_AT_START}" == "1" ]]; then
    cmd="docker rm -f $(printf '%q' "${CONTAINER_NAME}") >/dev/null 2>&1 || true; cd $(printf '%q' "${REMOTE_REPO_ROOT}") && ./research/code/deploy/run_dev_container.sh true >/dev/null"
  else
    cmd="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && ./research/code/deploy/run_dev_container.sh true >/dev/null"
  fi
  echo "[phase2-matrix] prepare container worker=${worker} reset=${CONTAINER_RESET_AT_START}"
  remote_worker_bash "${worker}" "${cmd}"
}

cleanup() {
  if (( CLEANUP_DONE == 1 )); then
    return 0
  fi
  CLEANUP_DONE=1
  if (( SWITCH_LOG_STARTED == 1 )); then
    emit_switch_marker_best_effort "matrix_end" "run_id=${RUN_ID}" "phase2_matrix"
    stop_switch_logger
  fi
}
trap cleanup EXIT INT TERM

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

MANIFEST_JSON="${MATRIX_ROOT}/matrix_manifest.json"

echo "[phase2-matrix] RUN_ID=${RUN_ID}"
echo "[phase2-matrix] MASTER_SERVER=${MASTER_SERVER} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[phase2-matrix] WORKERS=${ALL_WORKERS}"
echo "[phase2-matrix] TOTAL_EXPERIMENTS=${#EXPERIMENT_IDS[@]}"
echo "[phase2-matrix] MATRIX_ROOT=${MATRIX_ROOT}"

if [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
  start_switch_logger
  emit_switch_marker "matrix_start" "run_id=${RUN_ID}" "phase2_matrix"
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
  echo "  \"network_topology_file\": \"${NETWORK_TOPOLOGY_FILE}\","
  echo "  \"ssh_host_resolution\": \"resolve worker internal IP from [Workers] section in network topology file\","
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
  exp_id=${EXPERIMENT_IDS[$idx]}
  exp_coll=${EXPERIMENT_COLLS[$idx]}
  exp_algo=${EXPERIMENT_ALGOS[$idx]}
  exp_root="${MATRIX_ROOT}/$(printf "%02d_%s" "$((idx+1))" "${exp_id}")"
  status_dir="${MATRIX_ROOT}/.matrix_status/$(printf "%02d_%s" "$((idx+1))" "${exp_id}")"
  mkdir -p "${status_dir}"

  echo "[phase2-matrix] ============================================================"
  echo "[phase2-matrix] start idx=${idx} id=${exp_id} coll=${exp_coll} algo=${exp_algo} payload=${PAYLOAD_MB}MB"
  emit_switch_marker_best_effort "exp_start" "experiment=${exp_id} collective=${exp_coll} algo=${exp_algo} payload_mb=${PAYLOAD_MB}" "phase2_matrix"

  set +e
  RUN_ID="${exp_id}" \
  EXPERIMENT_LABEL="${exp_id}" \
  LOG_ROOT="${exp_root}" \
  MASTER_SERVER="${MASTER_SERVER}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  ALL_WORKERS="${ALL_WORKERS}" \
  RUN_MODES="${RUN_MODES}" \
  COLLECTIVE="${exp_coll}" \
  PAYLOAD_MB="${PAYLOAD_MB}" \
  DTYPE="${DTYPE}" \
  STEPS="${STEPS}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  SLEEP_MS="${SLEEP_MS}" \
  MODE_TIMEOUT_SEC="${MODE_TIMEOUT_SEC}" \
  NCCL_ALGO="${exp_algo}" \
  NCCL_PROTO="auto" \
  TORCH_ENV="${TORCH_ENV}" \
  HOST_REPO_ROOT="${HOST_REPO_ROOT}" \
  REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT}" \
  CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  CONTAINER_RESET_AT_START=0 \
  RESTART_CONTAINERS_ON_FAILURE="${RESTART_CONTAINERS_ON_FAILURE}" \
  WORKER_SSH_USER="${WORKER_SSH_USER}" \
  WORKER_SSH_USER_MAP="${WORKER_SSH_USER_MAP}" \
  WORKER_SSH_PASSWORD="${WORKER_SSH_PASSWORD}" \
  SSH_CONNECT_TIMEOUT_SEC="${SSH_CONNECT_TIMEOUT_SEC}" \
  SWITCH_LOG_ENABLE="${SWITCH_LOG_ENABLE}" \
  SWITCH_METADATA_FILE="${SWITCH_METADATA_FILE}" \
  DPU_NODE_HOST="${DPU_NODE_HOST}" \
  DPU_NODE_USER="${DPU_NODE_USER}" \
  DPU_NODE_PWD="${DPU_NODE_PWD:-}" \
  SWITCH_LOGGER_ROOT="${SWITCH_LOGGER_ROOT}" \
  SWITCH_LOG_SHARED_ROOT="${SWITCH_LOG_SHARED_ROOT}" \
  bash "${REPO_ROOT}/${INNER_SCRIPT}"
  run_rc=$?
  set -e

  {
    echo "rc=${run_rc}"
    echo "experiment=${exp_id}"
    echo "updated_at=$(date +%s)"
  } > "${status_dir}/summary.status"

  if (( run_rc != 0 )); then
    overall_rc=1
    echo "[phase2-matrix] experiment failed idx=${idx} id=${exp_id}"
  else
    echo "[phase2-matrix] experiment complete idx=${idx} id=${exp_id}"
  fi

  emit_switch_marker_best_effort "exp_end" "experiment=${exp_id} rc=${run_rc}" "phase2_matrix"

  if (( run_rc != 0 )) && [[ "${STOP_ON_FAILURE}" == "1" ]]; then
    break
  fi
done

exit "${overall_rc}"
