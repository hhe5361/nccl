#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-appendix2_collective_$(date +%y%m%d_%H%M%S)}
EXPERIMENT_LABEL=${EXPERIMENT_LABEL:-${RUN_ID}}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
RUN_MODES=${RUN_MODES:-stock,stock_2,phase4}
COLLECTIVE=${COLLECTIVE:-allreduce}
PAYLOAD_MB=${PAYLOAD_MB:-128}
DTYPE=${DTYPE:-float32}
STEPS=${STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-2}
SLEEP_MS=${SLEEP_MS:-0}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-300}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase2/collective_b2.py}
WORKER_SCRIPT=${WORKER_SCRIPT:-research/code/appendix2/run_stock2_mode_worker.sh}
COMPARE_SCRIPT=${COMPARE_SCRIPT:-research/code/phase2/compare_b2_vs_stock.py}
SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-0}

exec env \
  RUN_ID="${RUN_ID}" \
  EXPERIMENT_LABEL="${EXPERIMENT_LABEL}" \
  LOG_ROOT="${LOG_ROOT}" \
  RUN_MODES="${RUN_MODES}" \
  COLLECTIVE="${COLLECTIVE}" \
  PAYLOAD_MB="${PAYLOAD_MB}" \
  DTYPE="${DTYPE}" \
  STEPS="${STEPS}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  SLEEP_MS="${SLEEP_MS}" \
  MODE_TIMEOUT_SEC="${MODE_TIMEOUT_SEC}" \
  TORCH_ENV="${TORCH_ENV}" \
  TARGET_SCRIPT="${TARGET_SCRIPT}" \
  WORKER_SCRIPT="${WORKER_SCRIPT}" \
  COMPARE_SCRIPT="${COMPARE_SCRIPT}" \
  SWITCH_LOG_ENABLE="${SWITCH_LOG_ENABLE}" \
  bash "${REPO_ROOT}/research/code/phase2/run_b2_collective.sh"
