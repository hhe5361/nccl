#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

infer_node_rank() {
  local worker_name=$1
  if [[ "${worker_name}" =~ ^worker([0-9]+)$ ]]; then
    local idx=${BASH_REMATCH[1]}
    echo $((10#${idx} - 1))
    return 0
  fi
  echo "cannot infer NODE_RANK from WORKER_NAME='${worker_name}'" >&2
  return 1
}

RUN_ID=${RUN_ID:-phase0_$(date +%y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
NODE_RANK=${NODE_RANK:-$(infer_node_rank "${WORKER_NAME}")}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}/${WORKER_NAME}}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase0/ring_allreduce_loop.py}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}

mkdir -p "${LOG_ROOT}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_LIBRARY_PATH=/workspace/nccl/build/lib:${LD_LIBRARY_PATH:-}
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-${LOG_ROOT}/nccl-phase0.%h.%p.log}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

echo "[phase0] RUN_ID=${RUN_ID}"
echo "[phase0] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[phase0] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[phase0] LOG_ROOT=${LOG_ROOT}"
echo "[phase0] TARGET_SCRIPT=${TARGET_SCRIPT}"
echo "[phase0] NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
echo "[phase0] NCCL_ALGO=${NCCL_ALGO} NCCL_PROTO=${NCCL_PROTO} NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"

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
  echo "[phase0] PyTorch launcher not found in current environment." >&2
  echo "[phase0] Verify the venv first: python -c 'import torch; print(torch.__version__)'" >&2
  exit 1
fi

"${LAUNCHER[@]}" \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/${TARGET_SCRIPT}" "$@"
