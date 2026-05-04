#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-appendix2_matrix_$(date +%y%m%d_%H%M%S)}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-/mnt/nfs_share/cts_experiments}
MATRIX_ROOT=${MATRIX_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
PAYLOAD_MB=${PAYLOAD_MB:-128}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-2}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-300}
RUN_MODES=${RUN_MODES:-stock,stock_2}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}

mkdir -p "${MATRIX_ROOT}"

echo "[appendix2-matrix] RUN_ID=${RUN_ID}"
echo "[appendix2-matrix] MATRIX_ROOT=${MATRIX_ROOT}"
echo "[appendix2-matrix] RUN_MODES=${RUN_MODES} STEPS=${STEPS} WARMUP_STEPS=${WARMUP_STEPS}"

run_one() {
  local exp_id=$1
  local collective=$2
  local algo=$3
  local root="${MATRIX_ROOT}/${exp_id}"

  echo "[appendix2-matrix] start exp_id=${exp_id} collective=${collective} algo=${algo}"
  RUN_ID="${exp_id}" \
  EXPERIMENT_LABEL="${exp_id}" \
  LOG_ROOT="${root}" \
  RUN_MODES="${RUN_MODES}" \
  COLLECTIVE="${collective}" \
  PAYLOAD_MB="${PAYLOAD_MB}" \
  DTYPE="${DTYPE}" \
  STEPS="${STEPS}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  MODE_TIMEOUT_SEC="${MODE_TIMEOUT_SEC}" \
  TORCH_ENV="${TORCH_ENV}" \
  NCCL_ALGO="${algo}" \
  NCCL_PROTO="auto" \
  SWITCH_LOG_ENABLE=0 \
  bash "${REPO_ROOT}/research/code/appendix2/run_stock2_collective.sh"
}

run_one "allreduce_ring_10step" "allreduce" "Ring"
run_one "alltoall_10step" "alltoall" "auto"

echo "[appendix2-matrix] complete"
