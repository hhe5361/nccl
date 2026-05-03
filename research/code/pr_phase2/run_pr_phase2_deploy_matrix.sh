#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

OUTPUT_ROOT=""
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
PAYLOAD_MB="${PAYLOAD_MB:-128}"
REPEATS="${REPEATS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/torch-cu121-custom}"
BOOTSTRAP_IF_MISSING="${BOOTSTRAP_IF_MISSING:-0}"
BOOTSTRAP_TARGET="${BOOTSTRAP_TARGET:-all}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"

usage() {
  cat <<'EOF'
Usage:
  run_pr_phase2_deploy_matrix.sh --output-root DIR [options]

Options:
  --output-root DIR
  --steps 50
  --warmup-steps 5
  --payload-mb 128
  --repeats 1
  --nproc-per-node 1
  --venv-dir /workspace/venvs/torch-cu121-custom
  --bootstrap-if-missing 0|1
  --bootstrap-target nccl|tests|pytorch|all
  --timeout-sec 300
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
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${OUTPUT_ROOT}" ]] || { echo "--output-root is required" >&2; exit 1; }

mode_labels=(
  "W2_0"
  "W3_0"
  "W4_0"
  "W5_0"
  "W6_0"
  "W7_0"
  "W8_0"
)

modes_csv="$(IFS=,; echo "${mode_labels[*]}")"
experiments_csv="allreduce_ring"

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
  --switch-enable "0"
  --timeout-sec "${TIMEOUT_SEC}"
)

env \
  SWITCH_ENABLE="0" \
  REPEATS="${REPEATS}" \
  MODES_CSV="${modes_csv}" \
  EXPERIMENTS_CSV="${experiments_csv}" \
  TIMEOUT_SEC="${TIMEOUT_SEC}" \
  bash "${REPO_ROOT}/research/code/deploy/run_dev_matrix_master.sh" \
    "${master_args[@]}"
