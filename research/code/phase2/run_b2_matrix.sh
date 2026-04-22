#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase2_b2_matrix_$(date +%y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-31500}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-/mnt/nfs_share/cts_experiments}
MATRIX_ROOT=${MATRIX_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
ALL_WORKERS=${ALL_WORKERS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
INTRA_RACK_HOSTS=${INTRA_RACK_HOSTS:-worker01,worker02,worker03,worker04}
INTER_RACK_HOSTS=${INTER_RACK_HOSTS:-worker01,worker02,worker05,worker06}
PAYLOAD_MB=${PAYLOAD_MB:-32}
RUN_MODES=${RUN_MODES:-stock,b2}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
SLEEP_MS=${SLEEP_MS:-0}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
INNER_SCRIPT=${INNER_SCRIPT:-research/code/phase2/run_b2_collective.sh}
RACK_MAP_FILE=${NCCL_RACK_MAP_FILE:-${REPO_ROOT}/research/code/phase2/rack_map.txt}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}

mkdir -p "${MATRIX_ROOT}"

IFS=',' read -r -a ALL_WORKER_ARRAY <<< "${ALL_WORKERS}"

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

EXPERIMENT_IDS=()
EXPERIMENT_HOSTS=()
EXPERIMENT_COLLS=()
EXPERIMENT_PAYLOADS=()

add_experiment() {
  EXPERIMENT_IDS+=("$1")
  EXPERIMENT_HOSTS+=("$2")
  EXPERIMENT_COLLS+=("$3")
  EXPERIMENT_PAYLOADS+=("$4")
}

add_experiment "intra_allreduce_${PAYLOAD_MB}mb" "${INTRA_RACK_HOSTS}" "allreduce" "${PAYLOAD_MB}"
add_experiment "inter_allreduce_${PAYLOAD_MB}mb" "${INTER_RACK_HOSTS}" "allreduce" "${PAYLOAD_MB}"
add_experiment "intra_allgather_${PAYLOAD_MB}mb" "${INTRA_RACK_HOSTS}" "allgather" "${PAYLOAD_MB}"
add_experiment "inter_allgather_${PAYLOAD_MB}mb" "${INTER_RACK_HOSTS}" "allgather" "${PAYLOAD_MB}"
add_experiment "intra_reducescatter_${PAYLOAD_MB}mb" "${INTRA_RACK_HOSTS}" "reducescatter" "${PAYLOAD_MB}"
add_experiment "inter_reducescatter_${PAYLOAD_MB}mb" "${INTER_RACK_HOSTS}" "reducescatter" "${PAYLOAD_MB}"
add_experiment "intra_alltoall_${PAYLOAD_MB}mb" "${INTRA_RACK_HOSTS}" "alltoall" "${PAYLOAD_MB}"
add_experiment "inter_alltoall_${PAYLOAD_MB}mb" "${INTER_RACK_HOSTS}" "alltoall" "${PAYLOAD_MB}"

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
      "collective": "${EXPERIMENT_COLLS[$idx]}",
      "payload_mb": ${EXPERIMENT_PAYLOADS[$idx]}
    }${comma}
EOF
    done
    echo "  ]"
    echo "}"
  } > "${MANIFEST_JSON}"
fi

echo "[phase2-matrix] RUN_ID=${RUN_ID}"
echo "[phase2-matrix] WORKER_NAME=${WORKER_NAME}"
echo "[phase2-matrix] MATRIX_ROOT=${MATRIX_ROOT}"
echo "[phase2-matrix] TOTAL_EXPERIMENTS=${#EXPERIMENT_IDS[@]}"

for idx in "${!EXPERIMENT_IDS[@]}"; do
  exp_id=${EXPERIMENT_IDS[$idx]}
  exp_hosts=${EXPERIMENT_HOSTS[$idx]}
  exp_coll=${EXPERIMENT_COLLS[$idx]}
  exp_payload=${EXPERIMENT_PAYLOADS[$idx]}
  exp_nodes=$(count_in_list "${exp_hosts}")
  exp_root="${MATRIX_ROOT}/$(printf "%02d_%s" "$((idx+1))" "${exp_id}")"
  status_dir="${MATRIX_ROOT}/.matrix_status/$(printf "%02d_%s" "$((idx+1))" "${exp_id}")"
  status_file="${status_dir}/${WORKER_NAME}.status"
  port_base=$((MASTER_PORT_BASE + idx * 10))

  mkdir -p "${status_dir}"
  rm -f "${status_file}"

  run_rc=0
  participated=0
  if contains_worker "${WORKER_NAME}" "${exp_hosts}"; then
    participated=1
    exp_rank=$(index_in_list "${WORKER_NAME}" "${exp_hosts}")
    echo "[phase2-matrix] start idx=${idx} id=${exp_id} rank=${exp_rank}/${exp_nodes} coll=${exp_coll} payload=${exp_payload}MB"
    set +e
    RUN_ID="${exp_id}" \
    LOG_ROOT="${exp_root}" \
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT_BASE="${port_base}" \
    NNODES="${exp_nodes}" \
    NPROC_PER_NODE=1 \
    NODE_RANK="${exp_rank}" \
    WORKER_NAME="${WORKER_NAME}" \
    COLLECTIVE="${exp_coll}" \
    RUN_MODES="${RUN_MODES}" \
    PAYLOAD_MB="${exp_payload}" \
    DTYPE="${DTYPE}" \
    STEPS="${STEPS}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    SLEEP_MS="${SLEEP_MS}" \
    NCCL_ALGO="auto" \
    NCCL_PROTO="auto" \
    NCCL_RACK_MAP_FILE="${RACK_MAP_FILE}" \
    TORCH_ENV="${TORCH_ENV}" \
    bash "${REPO_ROOT}/${INNER_SCRIPT}"
    run_rc=$?
    set -e
  else
    echo "[phase2-matrix] skip idx=${idx} id=${exp_id} worker=${WORKER_NAME}"
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

  if (( failed == 1 )); then
    echo "[phase2-matrix] experiment failed idx=${idx} id=${exp_id}" >&2
    exit 1
  fi

  echo "[phase2-matrix] complete idx=${idx} id=${exp_id}"
done

echo "[phase2-matrix] all experiments completed"
