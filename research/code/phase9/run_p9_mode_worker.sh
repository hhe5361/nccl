#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:?RUN_ID is required}
EXPERIMENT_LABEL=${EXPERIMENT_LABEL:-${RUN_ID}}
MODE=${MODE:?MODE is required}
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR is required}
MASTER_PORT=${MASTER_PORT:?MASTER_PORT is required}
NNODES=${NNODES:?NNODES is required}
NODE_RANK=${NODE_RANK:?NODE_RANK is required}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase4/ddp_b4.py}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
STATUS_FILE=${STATUS_FILE:?STATUS_FILE is required}
STATUS_DIR=${STATUS_DIR:-$(dirname "${STATUS_FILE}")}
STEPS=${STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-20}
DTYPE=${DTYPE:-float32}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-600}
ALGO_SETTING=${NCCL_ALGO:-auto}
PROTO_SETTING=${NCCL_PROTO:-auto}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_LAYERS=${NUM_LAYERS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
BUCKET_CAP_MB=${BUCKET_CAP_MB:-1}
LR=${LR:-0.01}
MODEL_SEED=${MODEL_SEED:-20260504}
REPEAT_LABEL=${REPEAT_LABEL:-repeat_01}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/${WORKER_NAME}" "${STATUS_DIR}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_PRELOAD="/workspace/nccl/build/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-0}
export NCCL_PHASE2_B2_ENABLE=0
export NCCL_PHASE2_LOG=0
export NCCL_PHASE3_B3_ENABLE=0
export NCCL_PHASE3_LOG=0
export NCCL_PHASE4_ENABLE=0
export NCCL_PHASE4_LOG=0
export NCCL_PHASE5_LOG=0
export NCCL_PHASE6_ENABLE=0
export NCCL_PHASE6_LOG=0
export NCCL_PHASE7_ENABLE=0
export NCCL_PHASE7_LOG=0
export NCCL_PHASE9_LOG=${NCCL_PHASE9_LOG:-1}
export NCCL_APPENDIX2_GROUP_LOG=0
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_PHASE1_STATIC_W=0

case "${MODE}" in
  stock)
    ;;
  *)
    echo "[phase9-worker] unsupported MODE=${MODE}" >&2
    exit 1
    ;;
esac

if [[ "${ALGO_SETTING}" == "auto" ]]; then
  unset NCCL_ALGO
else
  export NCCL_ALGO="${ALGO_SETTING}"
fi

if [[ "${PROTO_SETTING}" == "auto" ]]; then
  unset NCCL_PROTO
else
  export NCCL_PROTO="${PROTO_SETTING}"
fi

MODE_UPPER=$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')
export PHASE4_MODE="${MODE}"
export PHASE4_OUTPUT_DIR="${RUN_ROOT}"
export PHASE4_REPEAT_LABEL="${REPEAT_LABEL}"
export NCCL_DEBUG_FILE="${RUN_ROOT}/${WORKER_NAME}/nccl.%h.%p.log"
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE="${NNODES}"
export RANK="${NODE_RANK}"
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export GROUP_RANK="${NODE_RANK}"

STATUS_DDP=-1
RUN_RC=1
RUN_STATE=failed
STATUS_MESSAGE=init
STARTED_AT=$(date +%s)

write_status() {
  cat > "${STATUS_FILE}" <<EOF
status=${STATUS_DDP}
state=${RUN_STATE}
worker=${WORKER_NAME}
node_rank=${NODE_RANK}
mode=${MODE_UPPER}
experiment=${EXPERIMENT_LABEL}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
rc=${RUN_RC}
pid=$$
updated_at=$(date +%s)
started_at=${STARTED_AT}
message=${STATUS_MESSAGE}
EOF
}

mark_running() {
  STATUS_DDP=1
  RUN_STATE=running
  RUN_RC=
  STATUS_MESSAGE=running
  write_status
}

mark_success() {
  STATUS_DDP=0
  RUN_STATE=success
  RUN_RC=0
  STATUS_MESSAGE=success
  write_status
}

mark_failure() {
  local rc=${1:-1}
  local message=${2:-failed}
  STATUS_DDP=-1
  RUN_STATE=failed
  RUN_RC=${rc}
  STATUS_MESSAGE=${message}
  write_status
}

cleanup() {
  if [[ "${RUN_STATE}" != "success" ]]; then
    mark_failure "${RUN_RC:-1}" "${STATUS_MESSAGE:-failed}"
  fi
}
trap cleanup EXIT INT TERM

echo "[phase9-worker] RUN_ID=${RUN_ID} EXPERIMENT=${EXPERIMENT_LABEL}"
echo "[phase9-worker] REPEAT_LABEL=${REPEAT_LABEL}"
echo "[phase9-worker] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES} MODE=${MODE_UPPER}"
echo "[phase9-worker] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[phase9-worker] PHASE9_LOG=${NCCL_PHASE9_LOG}"
echo "[phase9-worker] HIDDEN_DIM=${HIDDEN_DIM} NUM_LAYERS=${NUM_LAYERS} BATCH_SIZE=${BATCH_SIZE} BUCKET_CAP_MB=${BUCKET_CAP_MB} LR=${LR}"
echo "[phase9-worker] MODEL_SEED=${MODEL_SEED}"
echo "[phase9-worker] NCCL_ALGO=${ALGO_SETTING} NCCL_PROTO=${PROTO_SETTING} STATUS_FILE=${STATUS_FILE}"
echo "[phase9-worker] NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"

mark_running

set +e
timeout --signal=TERM --kill-after=30 "${MODE_TIMEOUT_SEC}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/${TARGET_SCRIPT}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --dtype "${DTYPE}" \
    --output-dir "${RUN_ROOT}" \
    --tag "${MODE_UPPER}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --batch-size "${BATCH_SIZE}" \
    --bucket-cap-mb "${BUCKET_CAP_MB}" \
    --lr "${LR}" \
    --model-seed "${MODEL_SEED}"
RUN_RC=$?
set -e

if (( RUN_RC == 0 )); then
  mark_success
  echo "[phase9-worker] complete worker=${WORKER_NAME} mode=${MODE_UPPER}"
  exit 0
fi

if (( RUN_RC == 124 )); then
  STATUS_MESSAGE=timeout
else
  STATUS_MESSAGE=launcher_failed
fi
echo "[phase9-worker] failure worker=${WORKER_NAME} mode=${MODE_UPPER} rc=${RUN_RC} message=${STATUS_MESSAGE}" >&2
exit "${RUN_RC}"
