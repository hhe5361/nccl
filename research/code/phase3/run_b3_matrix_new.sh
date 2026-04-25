#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase3_b3_matrix_$(date +%y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-32500}
PORT_STRIDE=${PORT_STRIDE:-10}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-/mnt/nfs_share/cts_experiments}
MATRIX_ROOT=${MATRIX_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
ALL_WORKERS=${ALL_WORKERS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
INTRA_RACK_HOSTS=${INTRA_RACK_HOSTS:-worker01,worker02,worker03,worker04}
INTER_RACK_HOSTS=${INTER_RACK_HOSTS:-worker01,worker02,worker05,worker06}
MIXED_8_HOSTS=${MIXED_8_HOSTS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
PAYLOAD_MB=${PAYLOAD_MB:-32}
RUN_MODES=${RUN_MODES:-stock,b2,b3}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
SLEEP_MS=${SLEEP_MS:-0}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
INNER_SCRIPT=${INNER_SCRIPT:-research/code/phase3/run_b3_collective.sh}
RACK_MAP_FILE=${NCCL_RACK_MAP_FILE:-${REPO_ROOT}/research/code/phase3/rack_map.txt}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
MASTER_SERVER=${MASTER_SERVER:-}

SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-1}
DPU_NODE_HOST=${DPU_NODE_HOST:-172.16.0.100}
DPU_NODE_USER=${DPU_NODE_USER:-ubuntu}
SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT:-/home/ubuntu/hyoeun/switch_setup_task/switch_congestion_logger}
SWITCH_LOG_INTERVAL_SEC=${SWITCH_LOG_INTERVAL_SEC:-1}
SWITCH_LOG_SHARED_ROOT=${SWITCH_LOG_SHARED_ROOT:-/mnt/nfs/cts_experiments/switch_log}
SWITCH_METADATA_FILE=${SWITCH_METADATA_FILE:-${MATRIX_ROOT}/switch_logger.env}

mkdir -p "${MATRIX_ROOT}"

IFS=',' read -r -a ALL_WORKER_ARRAY <<< "${ALL_WORKERS}"
if [[ -z "${MASTER_SERVER}" ]]; then
  MASTER_SERVER=${ALL_WORKER_ARRAY[0]}
fi

contains_worker() {
  local worker=$1
  local list=$2
  local item
  IFS=',' read -r -a _items <<< "${list}"
  for item in "${_items[@]}"; do
    if [[ "${item}" == "${worker}" ]]; then
      return 0
    fi
  done
  return 1
}

index_in_list() {
  local worker=$1
  local list=$2
  local idx=0
  local item
  IFS=',' read -r -a _items <<< "${list}"
  for item in "${_items[@]}"; do
    if [[ "${item}" == "${worker}" ]]; then
      echo "${idx}"
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

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

is_master_server() {
  [[ "${WORKER_NAME}" == "${MASTER_SERVER}" || "$(hostname -f 2>/dev/null || true)" == "${MASTER_SERVER}" || "$(hostname -s 2>/dev/null || true)" == "${MASTER_SERVER}" ]]
}

require_sshpass() {
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase3-matrix-new] sshpass is required for switch logger integration." >&2
    exit 1
  fi
}

remote_dpu_bash() {
  local cmd=$1
  require_sshpass
  if [[ -z "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase3-matrix-new] DPU_NODE_PWD must be set when SWITCH_LOG_ENABLE=1 on MASTER_SERVER." >&2
    exit 1
  fi
  sshpass -p "${DPU_NODE_PWD}" \
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
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
  cat > "${SWITCH_METADATA_FILE}" <<EOF2
SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE}
SWITCH_LOG_RUN_ID=$(printf '%q' "${SWITCH_LOG_RUN_ID}")
SWITCH_LOG_DIR=$(printf '%q' "${SWITCH_LOG_DIR}")
SWITCH_LOG_LOCAL_DIR=$(printf '%q' "${SWITCH_LOG_LOCAL_DIR}")
SWITCH_LOG_PID_FILE=$(printf '%q' "${SWITCH_LOG_PID_FILE}")
SWITCH_LOG_MARKERS_JSONL=$(printf '%q' "${SWITCH_LOG_MARKERS_JSONL}")
DPU_NODE_HOST=$(printf '%q' "${DPU_NODE_HOST}")
DPU_NODE_USER=$(printf '%q' "${DPU_NODE_USER}")
SWITCH_LOGGER_ROOT=$(printf '%q' "${SWITCH_LOGGER_ROOT}")
SWITCH_LOG_SHARED_ROOT=$(printf '%q' "${SWITCH_LOG_SHARED_ROOT}")
MASTER_SERVER=$(printf '%q' "${MASTER_SERVER}")
EOF2
}

wait_for_switch_metadata() {
  local timeout_sec=${1:-120}
  local waited=0
  while [[ ! -f "${SWITCH_METADATA_FILE}" ]]; do
    if (( waited >= timeout_sec )); then
      echo "[phase3-matrix-new] timed out waiting for switch metadata file ${SWITCH_METADATA_FILE}" >&2
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  # shellcheck disable=SC1090
  source "${SWITCH_METADATA_FILE}"
}

start_switch_logger() {
  local cmd
  local output
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./start_switch_congestion_loggers.sh --interval-sec $(printf '%q' "${SWITCH_LOG_INTERVAL_SEC}")"
  if [[ -n "${NETWORK_NODE_PASSWORD:-}" ]]; then
    cmd+=" --network-node-password $(printf '%q' "${NETWORK_NODE_PASSWORD}")"
  fi
  if [[ -n "${SWITCH_PASSWORD:-}" ]]; then
    cmd+=" --switch-password $(printf '%q' "${SWITCH_PASSWORD}")"
  fi
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
    echo "[phase3-matrix-new] failed to parse switch logger metadata" >&2
    echo "${output}" >&2
    exit 1
  fi
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
  write_switch_metadata
  SWITCH_LOG_STARTED=1
  echo "[phase3-matrix-new] switch logger started run_id=${SWITCH_LOG_RUN_ID} log_dir=${SWITCH_LOG_DIR} local_dir=${SWITCH_LOG_LOCAL_DIR}"
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
  echo "[phase3-matrix-new] switch logger stopped run_id=${SWITCH_LOG_RUN_ID}"
}

cleanup() {
  if (( CLEANUP_DONE == 1 )); then
    return 0
  fi
  CLEANUP_DONE=1
  if (( SWITCH_LOG_STARTED == 1 )) && is_master_server; then
    emit_switch_marker_best_effort "matrix_end" "run_id=${RUN_ID}" "phase3_matrix"
    stop_switch_logger
  fi
}
trap cleanup EXIT INT TERM

if [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
  if is_master_server; then
    start_switch_logger
    emit_switch_marker "matrix_start" "run_id=${RUN_ID}" "phase3_matrix"
  else
    wait_for_switch_metadata
  fi
fi

EXPERIMENT_IDS=()
EXPERIMENT_HOSTS=()
EXPERIMENT_PLACEMENTS=()
EXPERIMENT_COLLS=()
EXPERIMENT_ALGOS=()

add_experiment() {
  EXPERIMENT_IDS+=("$1")
  EXPERIMENT_HOSTS+=("$2")
  EXPERIMENT_PLACEMENTS+=("$3")
  EXPERIMENT_COLLS+=("$4")
  EXPERIMENT_ALGOS+=("$5")
}

for placement_name in intra inter mixed8; do
  case "${placement_name}" in
    intra)
      placement_label="intra-rack"
      placement_hosts="${INTRA_RACK_HOSTS}"
      ;;
    inter)
      placement_label="inter-rack"
      placement_hosts="${INTER_RACK_HOSTS}"
      ;;
    mixed8)
      placement_label="mixed-8"
      placement_hosts="${MIXED_8_HOSTS}"
      ;;
  esac
  add_experiment "${placement_name}_allreduce_ring_${PAYLOAD_MB}mb" "${placement_hosts}" "${placement_label}" "allreduce" "Ring"
  add_experiment "${placement_name}_allreduce_tree_${PAYLOAD_MB}mb" "${placement_hosts}" "${placement_label}" "allreduce" "Tree"
  add_experiment "${placement_name}_alltoall_${PAYLOAD_MB}mb" "${placement_hosts}" "${placement_label}" "alltoall" "auto"
done

MANIFEST_JSON="${MATRIX_ROOT}/matrix_manifest.json"
if [[ "${WORKER_NAME}" == "${ALL_WORKER_ARRAY[0]}" ]]; then
  {
    echo "{"
    echo "  \"run_id\": \"${RUN_ID}\"," 
    echo "  \"master_addr\": \"${MASTER_ADDR}\"," 
    echo "  \"master_port_base\": ${MASTER_PORT_BASE},"
    echo "  \"run_modes\": \"${RUN_MODES}\"," 
    echo "  \"dtype\": \"${DTYPE}\"," 
    echo "  \"payload_mb\": ${PAYLOAD_MB},"
    echo "  \"steps\": ${STEPS},"
    echo "  \"warmup_steps\": ${WARMUP_STEPS},"
    echo "  \"rack_map_file\": \"${RACK_MAP_FILE}\"," 
    echo "  \"worker_pool\": \"${ALL_WORKERS}\"," 
    echo "  \"intra_rack_hosts\": \"${INTRA_RACK_HOSTS}\"," 
    echo "  \"inter_rack_hosts\": \"${INTER_RACK_HOSTS}\"," 
    echo "  \"mixed_8_hosts\": \"${MIXED_8_HOSTS}\"," 
    echo "  \"master_server\": \"${MASTER_SERVER}\"," 
    if [[ "${SWITCH_LOG_ENABLE}" == "1" && -f "${SWITCH_METADATA_FILE}" ]]; then
      # shellcheck disable=SC1090
      source "${SWITCH_METADATA_FILE}"
      echo "  \"switch_log_enable\": 1,"
      echo "  \"switch_log_run_id\": \"${SWITCH_LOG_RUN_ID}\"," 
      echo "  \"switch_log_dir\": \"${SWITCH_LOG_DIR}\"," 
      echo "  \"switch_log_local_dir\": \"${SWITCH_LOG_LOCAL_DIR}\"," 
      echo "  \"switch_log_markers_jsonl\": \"${SWITCH_LOG_MARKERS_JSONL}\"," 
      echo "  \"dpu_node_host\": \"${DPU_NODE_HOST}\"," 
    else
      echo "  \"switch_log_enable\": 0,"
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
      "hosts": "${EXPERIMENT_HOSTS[$idx]}",
      "placement": "${EXPERIMENT_PLACEMENTS[$idx]}",
      "collective": "${EXPERIMENT_COLLS[$idx]}",
      "algo": "${EXPERIMENT_ALGOS[$idx]}",
      "payload_mb": ${PAYLOAD_MB}
    }${comma}
EOF
    done
    echo "  ]"
    echo "}"
  } > "${MANIFEST_JSON}"
fi

echo "[phase3-matrix-new] RUN_ID=${RUN_ID}"
echo "[phase3-matrix-new] WORKER_NAME=${WORKER_NAME}"
echo "[phase3-matrix-new] MASTER_SERVER=${MASTER_SERVER}"
echo "[phase3-matrix-new] MATRIX_ROOT=${MATRIX_ROOT}"
echo "[phase3-matrix-new] TOTAL_EXPERIMENTS=${#EXPERIMENT_IDS[@]}"
echo "[phase3-matrix-new] PORT_STRIDE=${PORT_STRIDE}"
if [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
  echo "[phase3-matrix-new] SWITCH_METADATA_FILE=${SWITCH_METADATA_FILE}"
fi

for idx in "${!EXPERIMENT_IDS[@]}"; do
  exp_id=${EXPERIMENT_IDS[$idx]}
  exp_hosts=${EXPERIMENT_HOSTS[$idx]}
  exp_placement=${EXPERIMENT_PLACEMENTS[$idx]}
  exp_coll=${EXPERIMENT_COLLS[$idx]}
  exp_algo=${EXPERIMENT_ALGOS[$idx]}
  exp_nodes=$(count_in_list "${exp_hosts}")
  exp_root="${MATRIX_ROOT}/$(printf "%02d_%s" "$((idx+1))" "${exp_id}")"
  status_dir="${MATRIX_ROOT}/.matrix_status/$(printf "%02d_%s" "$((idx+1))" "${exp_id}")"
  status_file="${status_dir}/${WORKER_NAME}.status"
  port_base=$((MASTER_PORT_BASE + idx * PORT_STRIDE))

  mkdir -p "${status_dir}"
  rm -f "${status_file}"

  if is_master_server && [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
    emit_switch_marker_best_effort "exp_start" "experiment=${exp_id} placement=${exp_placement} collective=${exp_coll} algo=${exp_algo} payload_mb=${PAYLOAD_MB}" "phase3_matrix"
  fi

  run_rc=0
  participated=0
  if contains_worker "${WORKER_NAME}" "${exp_hosts}"; then
    participated=1
    exp_rank=$(index_in_list "${WORKER_NAME}" "${exp_hosts}")
    echo "[phase3-matrix-new] start idx=${idx} id=${exp_id} rank=${exp_rank}/${exp_nodes} placement=${exp_placement} coll=${exp_coll} algo=${exp_algo} payload=${PAYLOAD_MB}MB"
    set +e
    RUN_ID="${exp_id}" \
    LOG_ROOT="${exp_root}" \
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT_BASE="${port_base}" \
    NNODES="${exp_nodes}" \
    NPROC_PER_NODE=1 \
    NODE_RANK="${exp_rank}" \
    WORKER_NAME="${WORKER_NAME}" \
    MASTER_SERVER="${MASTER_SERVER}" \
    SWITCH_LOG_ENABLE="${SWITCH_LOG_ENABLE}" \
    SWITCH_METADATA_FILE="${SWITCH_METADATA_FILE}" \
    DPU_NODE_HOST="${DPU_NODE_HOST}" \
    DPU_NODE_USER="${DPU_NODE_USER}" \
    SWITCH_LOGGER_ROOT="${SWITCH_LOGGER_ROOT}" \
    SWITCH_LOG_SHARED_ROOT="${SWITCH_LOG_SHARED_ROOT}" \
    SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_LOCAL_DIR}" \
    PLACEMENT="${exp_placement}" \
    COLLECTIVE="${exp_coll}" \
    RUN_MODES="${RUN_MODES}" \
    PAYLOAD_MB="${PAYLOAD_MB}" \
    DTYPE="${DTYPE}" \
    STEPS="${STEPS}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    SLEEP_MS="${SLEEP_MS}" \
    NCCL_ALGO="${exp_algo}" \
    NCCL_PROTO="auto" \
    NCCL_RACK_MAP_FILE="${RACK_MAP_FILE}" \
    TORCH_ENV="${TORCH_ENV}" \
    bash "${REPO_ROOT}/${INNER_SCRIPT}"
    run_rc=$?
    set -e
  else
    echo "[phase3-matrix-new] skip idx=${idx} id=${exp_id} worker=${WORKER_NAME}"
  fi

  {
    echo "participated=${participated}"
    echo "rc=${run_rc}"
  } > "${status_file}"

  while true; do
    ready=1
    for worker in "${ALL_WORKER_ARRAY[@]}"; do
      if [[ ! -f "${status_dir}/${worker}.status" ]]; then
        ready=0
        break
      fi
    done
    if (( ready == 1 )); then
      break
    fi
    sleep 2
  done

  failed=0
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    rc=$(awk -F= '/^rc=/{print $2}' "${status_dir}/${worker}.status")
    if [[ "${rc}" != "0" ]]; then
      failed=1
      break
    fi
  done

  if is_master_server && [[ "${SWITCH_LOG_ENABLE}" == "1" ]]; then
    emit_switch_marker_best_effort "exp_end" "experiment=${exp_id} placement=${exp_placement} collective=${exp_coll} algo=${exp_algo} payload_mb=${PAYLOAD_MB} rc=${failed}" "phase3_matrix"
  fi

  if (( failed == 1 )); then
    echo "[phase3-matrix-new] experiment failed idx=${idx} id=${exp_id}" >&2
    exit 1
  fi

  echo "[phase3-matrix-new] complete idx=${idx} id=${exp_id}"
done

echo "[phase3-matrix-new] all experiments completed"
