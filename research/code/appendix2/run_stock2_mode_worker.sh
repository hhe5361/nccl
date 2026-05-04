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
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:?NODE_RANK is required}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
MASTER_SERVER=${MASTER_SERVER:-worker01}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase2/collective_b2.py}
LOG_ROOT=${LOG_ROOT:?LOG_ROOT is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
STATUS_FILE=${STATUS_FILE:?STATUS_FILE is required}
STATUS_DIR=${STATUS_DIR:-$(dirname "${STATUS_FILE}")}
COLLECTIVE=${COLLECTIVE:?COLLECTIVE is required}
STEPS=${STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-2}
PAYLOAD_MB=${PAYLOAD_MB:-128}
DTYPE=${DTYPE:-float32}
SLEEP_MS=${SLEEP_MS:-0}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-300}
ALGO_SETTING=${NCCL_ALGO:-auto}
PROTO_SETTING=${NCCL_PROTO:-auto}

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/${WORKER_NAME}" "${STATUS_DIR}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_PRELOAD="/workspace/nccl/build/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-0}
export NCCL_PHASE2_B2_ENABLE=0
export NCCL_PHASE2_LOG=${NCCL_PHASE2_LOG:-0}
export NCCL_PHASE3_B3_ENABLE=0
export NCCL_PHASE3_LOG=${NCCL_PHASE3_LOG:-0}
export NCCL_PHASE4_ENABLE=0
export NCCL_PHASE4_LOG=${NCCL_PHASE4_LOG:-0}
export NCCL_APPENDIX2_DISABLE_WSTALL_LOG=${NCCL_APPENDIX2_DISABLE_WSTALL_LOG:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_PHASE1_STATIC_W=0

case "${MODE}" in
  stock)
    export NCCL_APPENDIX2_GROUP_LOG=0
    ;;
  stock_2)
    export NCCL_APPENDIX2_GROUP_LOG=1
    ;;
  *)
    echo "[appendix2-worker] unsupported MODE=${MODE}" >&2
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

if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun)
elif python - <<'PYTORCHCHECK' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('torch.distributed.run') else 1)
PYTORCHCHECK
then
  LAUNCHER=(python -m torch.distributed.run)
else
  echo "[appendix2-worker] PyTorch launcher not found in current environment." >&2
  exit 1
fi

MODE_UPPER=$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')
export PHASE2_MODE="${MODE}"
export PHASE2_OUTPUT_DIR="${RUN_ROOT}"
export NCCL_DEBUG_FILE="${RUN_ROOT}/${WORKER_NAME}/nccl.%h.%p.log"

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
  export STATUS_DDP=1
  RUN_STATE=running
  RUN_RC=
  STATUS_MESSAGE=running
  write_status
}

mark_success() {
  export STATUS_DDP=0
  RUN_STATE=success
  RUN_RC=0
  STATUS_MESSAGE=success
  write_status
}

mark_failure() {
  local rc=${1:-1}
  local message=${2:-failed}
  export STATUS_DDP=-1
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

echo "[appendix2-worker] RUN_ID=${RUN_ID} EXPERIMENT=${EXPERIMENT_LABEL}"
echo "[appendix2-worker] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES} MODE=${MODE_UPPER}"
echo "[appendix2-worker] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[appendix2-worker] COLLECTIVE=${COLLECTIVE} PAYLOAD_MB=${PAYLOAD_MB} NCCL_ALGO=${ALGO_SETTING} NCCL_PROTO=${PROTO_SETTING}"
echo "[appendix2-worker] APPENDIX2_GROUP_LOG=${NCCL_APPENDIX2_GROUP_LOG} STATUS_FILE=${STATUS_FILE}"
echo "[appendix2-worker] NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"

mark_running

set +e
timeout --signal=TERM --kill-after=30 "${MODE_TIMEOUT_SEC}" \
  "${LAUNCHER[@]}" \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/${TARGET_SCRIPT}" \
    --collective "${COLLECTIVE}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --payload-mb "${PAYLOAD_MB}" \
    --dtype "${DTYPE}" \
    --sleep-ms "${SLEEP_MS}" \
    --output-dir "${RUN_ROOT}" \
    --tag "${MODE_UPPER}"
RUN_RC=$?
set -e

if (( RUN_RC == 0 )); then
  mark_success
  echo "[appendix2-worker] complete worker=${WORKER_NAME} mode=${MODE_UPPER}"
  exit 0
fi

if (( RUN_RC == 124 )); then
  STATUS_MESSAGE=timeout
else
  STATUS_MESSAGE=launcher_failed
fi
echo "[appendix2-worker] failure worker=${WORKER_NAME} mode=${MODE_UPPER} rc=${RUN_RC} message=${STATUS_MESSAGE}" >&2
exit "${RUN_RC}"
