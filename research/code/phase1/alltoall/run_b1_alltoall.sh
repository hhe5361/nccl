#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

infer_node_rank() {
  local worker_name=$1
  if [[ "${worker_name}" =~ ^worker([0-9]+)$ ]]; then
    local idx=${BASH_REMATCH[1]}
    if (( 1 <= 10#${idx} && 10#${idx} <= 8 )); then
      echo $((10#${idx} - 1))
      return 0
    fi
  fi
  echo "cannot infer NODE_RANK from WORKER_NAME='${worker_name}'" >&2
  return 1
}

RUN_ID=${RUN_ID:-phase1_b1_alltoall_manual}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30500}
NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
NODE_RANK=${NODE_RANK:-$(infer_node_rank "${WORKER_NAME}")}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu132-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase1/alltoall/alltoall_b1.py}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
W_LIST=${W_LIST:-1,2,4,8}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
PAYLOAD_MB=${PAYLOAD_MB:-64}
DTYPE=${DTYPE:-float32}
SLEEP_MS=${SLEEP_MS:-0}

mkdir -p "${LOG_ROOT}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_PRELOAD="/workspace/nccl/build/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

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
  echo "[phase1-alltoall] PyTorch launcher not found." >&2
  exit 1
fi

IFS=',' read -r -a W_VALUES <<< "${W_LIST}"

echo "[phase1-alltoall] RUN_ID=${RUN_ID}"
echo "[phase1-alltoall] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES}"
echo "[phase1-alltoall] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT_BASE=${MASTER_PORT_BASE}"
echo "[phase1-alltoall] LOG_ROOT=${LOG_ROOT}"
echo "[phase1-alltoall] TARGET_SCRIPT=${TARGET_SCRIPT}"
echo "[phase1-alltoall] W_LIST=${W_LIST}"
echo "[phase1-alltoall] PAYLOAD_MB=${PAYLOAD_MB} DTYPE=${DTYPE} NCCL_PROTO=${NCCL_PROTO}"

for idx in "${!W_VALUES[@]}"; do
  W="${W_VALUES[$idx]}"
  RUN_ROOT="${LOG_ROOT}/W${W}"
  WORKER_LOG_ROOT="${RUN_ROOT}/${WORKER_NAME}"
  MASTER_PORT=$((MASTER_PORT_BASE + idx))

  mkdir -p "${RUN_ROOT}" "${WORKER_LOG_ROOT}"

  export NCCL_PHASE1_STATIC_W="${W}"
  export NCCL_DEBUG_FILE="${WORKER_LOG_ROOT}/nccl.%h.%p.log"
  export PHASE1_OUTPUT_DIR="${RUN_ROOT}"

  echo "[phase1-alltoall] starting W=${W} MASTER_PORT=${MASTER_PORT}"
  echo "[phase1-alltoall] NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"

  "${LAUNCHER[@]}" \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/${TARGET_SCRIPT}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --payload-mb "${PAYLOAD_MB}" \
    --dtype "${DTYPE}" \
    --sleep-ms "${SLEEP_MS}" \
    --output-dir "${RUN_ROOT}" \
    --tag "W${W}" \
    "$@"

  sleep 2
done
