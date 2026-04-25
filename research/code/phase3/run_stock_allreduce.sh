#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export RUN_ID=${RUN_ID:-phase3_stock_allreduce_$(date +%y%m%d_%H%M%S)}
export RUN_MODES=${RUN_MODES:-stock}
export COLLECTIVE=${COLLECTIVE:-allreduce}
export PLACEMENT=${PLACEMENT:-stock-allreduce}
export NNODES=${NNODES:-8}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-32500}
export PAYLOAD_MB=${PAYLOAD_MB:-32}
export DTYPE=${DTYPE:-float32}
export STEPS=${STEPS:-40}
export WARMUP_STEPS=${WARMUP_STEPS:-5}
export SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-0}
export NCCL_ALGO=${NCCL_ALGO:-auto}
export NCCL_PROTO=${NCCL_PROTO:-auto}

exec "${SCRIPT_DIR}/run_b3_collective.sh" "$@"
