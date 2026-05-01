#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

OUTPUT_ROOT=""
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
PAYLOAD_MB="${PAYLOAD_MB:-128}"
REPEATS="${REPEATS:-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/torch-cu121-custom}"
BOOTSTRAP_IF_MISSING="${BOOTSTRAP_IF_MISSING:-1}"
BOOTSTRAP_TARGET="${BOOTSTRAP_TARGET:-all}"
SWITCH_ENABLE="${SWITCH_ENABLE:-0}"
SWITCH_LOGGER_DIR="${SWITCH_LOGGER_DIR:-}"
SWITCH_INTERVAL_SEC="${SWITCH_INTERVAL_SEC:-1}"

usage() {
  cat <<'EOF'
Usage:
  run_pr_phase1_deploy_matrix.sh --output-root DIR [options]

Options:
  --output-root DIR
  --steps 50
  --warmup-steps 5
  --payload-mb 128
  --repeats 5
  --nproc-per-node 1
  --venv-dir /workspace/venvs/torch-cu121-custom
  --bootstrap-if-missing 0|1
  --bootstrap-target nccl|tests|pytorch|all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="${2:-}"; shift 2 ;;
    --payload-mb) PAYLOAD_MB="${2:-}"; shift 2 ;;
    --repeats) REPEATS="${2:-}"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="${2:-}"; shift 2 ;;
    --venv-dir) VENV_DIR="${2:-}"; shift 2 ;;
    --bootstrap-if-missing) BOOTSTRAP_IF_MISSING="${2:-}"; shift 2 ;;
    --bootstrap-target) BOOTSTRAP_TARGET="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${OUTPUT_ROOT}" ]] || { echo "--output-root is required" >&2; exit 1; }

mode_labels=()
for whole in 1 2 3 4 5 6 7 8; do
  mode_labels+=("W${whole}_0")
  if [[ "${whole}" != "8" ]]; then
    mode_labels+=("W${whole}_5")
  fi
done

modes_csv="$(IFS=,; echo "${mode_labels[*]}")"
experiments_csv="allreduce_ring,allreduce_tree,alltoall"

runner_template="bash research/code/deploy/run_pytorch_ddp_once.sh \
  --experiment {EXPERIMENT} \
  --mode {MODE} \
  --output-dir {WORKER_OUTPUT_DIR} \
  --master-addr {MASTER_ADDR} \
  --master-port {MASTER_PORT} \
  --world-size {WORLD_SIZE} \
  --node-rank {WORKER_RANK} \
  --steps ${STEPS} \
  --warmup-steps ${WARMUP_STEPS} \
  --payload-mb ${PAYLOAD_MB} \
  --nproc-per-node ${NPROC_PER_NODE} \
  --venv-dir '${VENV_DIR}' \
  --python-script 'research/code/pr_phase1/ddp_collective_runner.py' \
  --bootstrap-if-missing ${BOOTSTRAP_IF_MISSING} \
  --bootstrap-target ${BOOTSTRAP_TARGET}"

master_args=(
  --output-root "${OUTPUT_ROOT}"
  --runner-template "${runner_template}"
  --repeats "${REPEATS}"
  --modes "${modes_csv}"
  --experiments "${experiments_csv}"
  --switch-enable "${SWITCH_ENABLE}"
  --switch-interval-sec "${SWITCH_INTERVAL_SEC}"
)

if [[ -n "${SWITCH_LOGGER_DIR}" ]]; then
  master_args+=(--switch-logger-dir "${SWITCH_LOGGER_DIR}")
fi

env \
  SWITCH_ENABLE="${SWITCH_ENABLE}" \
  SWITCH_LOGGER_DIR="${SWITCH_LOGGER_DIR}" \
  SWITCH_INTERVAL_SEC="${SWITCH_INTERVAL_SEC}" \
  REPEATS="${REPEATS}" \
  MODES_CSV="${modes_csv}" \
  EXPERIMENTS_CSV="${experiments_csv}" \
  bash "${REPO_ROOT}/research/code/deploy/run_dev_matrix_master.sh" \
    "${master_args[@]}"
