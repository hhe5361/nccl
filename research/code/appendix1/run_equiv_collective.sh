#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

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

RUN_ID=${RUN_ID:-appendix1_equiv_$(date +%y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-34500}
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
NODE_RANK=${NODE_RANK:-$(infer_node_rank "${WORKER_NAME}")}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu132-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/appendix1/collective_equiv_check.py}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
RUN_MODES=${RUN_MODES:-stock,b3}
COLLECTIVE=${COLLECTIVE:-allreduce}
PLACEMENT=${PLACEMENT:-inter-rack}
STEPS=${STEPS:-20}
WARMUP_STEPS=${WARMUP_STEPS:-2}
PAYLOAD_MB=${PAYLOAD_MB:-32}
DTYPE=${DTYPE:-int32}
SLEEP_MS=${SLEEP_MS:-0}
PROTO_SETTING=${NCCL_PROTO:-auto}

if [[ "${COLLECTIVE}" == "allreduce" ]]; then
  ALGO_SETTING=${NCCL_ALGO:-Ring}
else
  ALGO_SETTING=${NCCL_ALGO:-auto}
fi

mkdir -p "${LOG_ROOT}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_PRELOAD="/workspace/nccl/build/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-0}
export NCCL_PHASE1_STATIC_W=0
export NCCL_PHASE2_B2_ENABLE=0
export NCCL_PHASE2_LOG=${NCCL_PHASE2_LOG:-0}
export NCCL_PHASE3_LOG=${NCCL_PHASE3_LOG:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

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
  echo "[appendix1] PyTorch launcher not found in current environment." >&2
  exit 1
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  cat > "${LOG_ROOT}/env_setup.json" <<EOF
{
  "appendix": "A1",
  "run_id": "${RUN_ID}",
  "placement": "${PLACEMENT}",
  "collective": "${COLLECTIVE}",
  "run_modes": "${RUN_MODES}",
  "master_addr": "${MASTER_ADDR}",
  "master_port_base": ${MASTER_PORT_BASE},
  "nnodes": ${NNODES},
  "nproc_per_node": ${NPROC_PER_NODE},
  "steps": ${STEPS},
  "warmup_steps": ${WARMUP_STEPS},
  "payload_mb": ${PAYLOAD_MB},
  "dtype": "${DTYPE}",
  "nccl_algo": "${ALGO_SETTING}",
  "nccl_proto": "${PROTO_SETTING}",
  "goal": "semantic_equivalence_stock_vs_b3"
}
EOF
fi

echo "[appendix1] RUN_ID=${RUN_ID}"
echo "[appendix1] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES}"
echo "[appendix1] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT_BASE=${MASTER_PORT_BASE}"
echo "[appendix1] COLLECTIVE=${COLLECTIVE} RUN_MODES=${RUN_MODES}"
echo "[appendix1] PAYLOAD_MB=${PAYLOAD_MB} DTYPE=${DTYPE} NCCL_ALGO=${ALGO_SETTING} NCCL_PROTO=${PROTO_SETTING}"

IFS=',' read -r -a MODE_VALUES <<< "${RUN_MODES}"
for idx in "${!MODE_VALUES[@]}"; do
  MODE="${MODE_VALUES[$idx]}"
  MODE_UPPER=$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')
  RUN_ROOT="${LOG_ROOT}/${MODE_UPPER}"
  WORKER_LOG_ROOT="${RUN_ROOT}/${WORKER_NAME}"
  MASTER_PORT=$((MASTER_PORT_BASE + idx))

  mkdir -p "${RUN_ROOT}" "${WORKER_LOG_ROOT}"

  case "${MODE}" in
    stock)
      export NCCL_PHASE3_B3_ENABLE=0
      ;;
    b3)
      export NCCL_PHASE3_B3_ENABLE=1
      ;;
    *)
      echo "[appendix1] unsupported RUN_MODES entry=${MODE}" >&2
      exit 1
      ;;
  esac

  export APPENDIX1_MODE="${MODE_UPPER}"
  export APPENDIX1_OUTPUT_DIR="${RUN_ROOT}"
  export NCCL_DEBUG_FILE="${WORKER_LOG_ROOT}/nccl.%h.%p.log"

  echo "[appendix1] starting MODE=${MODE} MASTER_PORT=${MASTER_PORT}"
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
    --tag "${MODE_UPPER}" \
    "$@"
done
