#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-34500}
ALL_WORKERS=${ALL_WORKERS:-worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08}
INTRA_HOSTS=${INTRA_HOSTS:-worker01,worker02,worker03,worker04}
INTER_HOSTS=${INTER_HOSTS:-worker01,worker02,worker05,worker06}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu132-custom/bin/activate}
RUN_ID=${RUN_ID:-appendix1_equiv_matrix_$(date +%y%m%d_%H%M%S)}
MATRIX_ROOT=${MATRIX_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
PAYLOAD_SMALL_MB=${PAYLOAD_SMALL_MB:-4}
PAYLOAD_LARGE_MB=${PAYLOAD_LARGE_MB:-32}
STEPS=${STEPS:-20}
WARMUP_STEPS=${WARMUP_STEPS:-2}
RUN_MODES=${RUN_MODES:-stock,b3}

contains_worker() {
  local worker=$1
  local csv=$2
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [[ "${worker}" == "${item}" ]]; then
      return 0
    fi
  done
  return 1
}

index_in_list() {
  local worker=$1
  local csv=$2
  IFS=',' read -r -a items <<< "${csv}"
  for idx in "${!items[@]}"; do
    if [[ "${worker}" == "${items[$idx]}" ]]; then
      echo "${idx}"
      return 0
    fi
  done
  return 1
}

declare -a CASES=(
  "01_intra_allreduce_ring_${PAYLOAD_SMALL_MB}mb|${INTRA_HOSTS}|intra-rack|allreduce|Ring|${PAYLOAD_SMALL_MB}"
  "02_inter_allreduce_ring_${PAYLOAD_LARGE_MB}mb|${INTER_HOSTS}|inter-rack|allreduce|Ring|${PAYLOAD_LARGE_MB}"
  "03_intra_alltoall_${PAYLOAD_SMALL_MB}mb|${INTRA_HOSTS}|intra-rack|alltoall|auto|${PAYLOAD_SMALL_MB}"
  "04_inter_alltoall_${PAYLOAD_LARGE_MB}mb|${INTER_HOSTS}|inter-rack|alltoall|auto|${PAYLOAD_LARGE_MB}"
)

mkdir -p "${MATRIX_ROOT}"

echo "[appendix1-matrix] RUN_ID=${RUN_ID}"
echo "[appendix1-matrix] MATRIX_ROOT=${MATRIX_ROOT}"

for idx in "${!CASES[@]}"; do
  IFS='|' read -r exp_id exp_hosts exp_placement exp_collective exp_algo exp_payload <<< "${CASES[$idx]}"
  port_base=$((MASTER_PORT_BASE + idx * 10))
  exp_root="${MATRIX_ROOT}/${exp_id}"

  if contains_worker "${WORKER_NAME}" "${exp_hosts}"; then
    exp_rank=$(index_in_list "${WORKER_NAME}" "${exp_hosts}")
    exp_nodes=$(awk -F',' '{print NF}' <<< "${exp_hosts}")
    echo "[appendix1-matrix] start idx=${idx} id=${exp_id} rank=${exp_rank}/${exp_nodes}"
    RUN_ID="${exp_id}" \
    LOG_ROOT="${exp_root}" \
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT_BASE="${port_base}" \
    NNODES="${exp_nodes}" \
    NODE_RANK="${exp_rank}" \
    WORKER_NAME="${WORKER_NAME}" \
    TORCH_ENV="${TORCH_ENV}" \
    PLACEMENT="${exp_placement}" \
    COLLECTIVE="${exp_collective}" \
    RUN_MODES="${RUN_MODES}" \
    PAYLOAD_MB="${exp_payload}" \
    NCCL_ALGO="${exp_algo}" \
    STEPS="${STEPS}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    bash "${REPO_ROOT}/research/code/appendix1/run_equiv_collective.sh"
  else
    echo "[appendix1-matrix] skip idx=${idx} id=${exp_id} worker=${WORKER_NAME}"
  fi
done
