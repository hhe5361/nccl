#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}

EXPERIMENT_NAME=""
MODE_NAME=""
OUTPUT_DIR=""
MASTER_ADDR=""
MASTER_PORT=""
WORLD_SIZE=""
NODE_RANK=""
STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
PAYLOAD_MB="${PAYLOAD_MB:-128}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VENV_DIR="${VENV_DIR:-${WORKSPACE_ROOT}/venvs/torch-cu121-custom}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-research/code/pr_phase1/ddp_collective_runner.py}"
BOOTSTRAP_IF_MISSING="${BOOTSTRAP_IF_MISSING:-0}"
BOOTSTRAP_TARGET="${BOOTSTRAP_TARGET:-all}"

usage() {
  cat <<'EOF'
Usage:
  run_pytorch_ddp_once.sh
    --experiment NAME
    --mode NAME
    --output-dir DIR
    --master-addr IP
    --master-port PORT
    --world-size N
    --node-rank R
    [--steps 50]
    [--warmup-steps 5]
    [--payload-mb 128]
    [--nproc-per-node 1]
    [--venv-dir DIR]
    [--python-script PATH]
    [--bootstrap-if-missing 0|1]
    [--bootstrap-target nccl|tests|pytorch|all]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT_NAME="${2:-}"; shift 2 ;;
    --mode) MODE_NAME="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --master-addr) MASTER_ADDR="${2:-}"; shift 2 ;;
    --master-port) MASTER_PORT="${2:-}"; shift 2 ;;
    --world-size) WORLD_SIZE="${2:-}"; shift 2 ;;
    --node-rank) NODE_RANK="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="${2:-}"; shift 2 ;;
    --payload-mb) PAYLOAD_MB="${2:-}"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="${2:-}"; shift 2 ;;
    --venv-dir) VENV_DIR="${2:-}"; shift 2 ;;
    --python-script) PYTHON_SCRIPT="${2:-}"; shift 2 ;;
    --bootstrap-if-missing) BOOTSTRAP_IF_MISSING="${2:-}"; shift 2 ;;
    --bootstrap-target) BOOTSTRAP_TARGET="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${EXPERIMENT_NAME}" ]] || { echo "--experiment is required" >&2; exit 1; }
[[ -n "${MODE_NAME}" ]] || { echo "--mode is required" >&2; exit 1; }
[[ -n "${OUTPUT_DIR}" ]] || { echo "--output-dir is required" >&2; exit 1; }
[[ -n "${MASTER_ADDR}" ]] || { echo "--master-addr is required" >&2; exit 1; }
[[ -n "${MASTER_PORT}" ]] || { echo "--master-port is required" >&2; exit 1; }
[[ -n "${WORLD_SIZE}" ]] || { echo "--world-size is required" >&2; exit 1; }
[[ -n "${NODE_RANK}" ]] || { echo "--node-rank is required" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"

collective=""
algorithm=""
case "${EXPERIMENT_NAME}" in
  allreduce_ring)
    collective="allreduce"
    algorithm="ring"
    export NCCL_ALGO="Ring"
    ;;
  allreduce_tree|treeallreduce)
    collective="allreduce"
    algorithm="tree"
    export NCCL_ALGO="Tree"
    ;;
  alltoall|alltoall_auto)
    collective="alltoall"
    algorithm="auto"
    unset NCCL_ALGO || true
    ;;
  *)
    echo "Unsupported experiment: ${EXPERIMENT_NAME}" >&2
    exit 1
    ;;
esac

mode_to_w() {
  local mode="$1"
  case "${mode}" in
    W*)
      printf '%s' "${mode#W}" | tr '_' '.'
      ;;
    *)
      return 1
      ;;
  esac
}

if w_value="$(mode_to_w "${MODE_NAME}")"; then
  export NCCL_PHASE1_INFLIGHT_W="${w_value}"
  export NCCL_PHASE1_INFLIGHT_LOG="${NCCL_PHASE1_INFLIGHT_LOG:-1}"
else
  unset NCCL_PHASE1_INFLIGHT_W || true
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-NET}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if [[ "${BOOTSTRAP_IF_MISSING}" == "1" ]]; then
    echo "[ddp-runner] venv missing, bootstrapping target=${BOOTSTRAP_TARGET} at ${VENV_DIR}"
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" VENV_DIR="${VENV_DIR}" \
      bash "${REPO_ROOT}/research/code/deploy/bootstrap_pytorch_env.sh" "${BOOTSTRAP_TARGET}"
  else
    echo "[ddp-runner] venv not found: ${VENV_DIR}" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

runner_log="${OUTPUT_DIR}/launcher_env.txt"
cat > "${runner_log}" <<EOF
experiment=${EXPERIMENT_NAME}
mode=${MODE_NAME}
collective=${collective}
algorithm=${algorithm}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
world_size=${WORLD_SIZE}
node_rank=${NODE_RANK}
steps=${STEPS}
warmup_steps=${WARMUP_STEPS}
payload_mb=${PAYLOAD_MB}
nproc_per_node=${NPROC_PER_NODE}
venv_dir=${VENV_DIR}
python_script=${PYTHON_SCRIPT}
nccl_phase1_inflight_w=${NCCL_PHASE1_INFLIGHT_W:-}
nccl_algo=${NCCL_ALGO:-auto}
EOF

torchrun \
  --nnodes "${WORLD_SIZE}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${PYTHON_SCRIPT}" \
  --collective "${collective}" \
  --algorithm "${algorithm}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --payload-mb "${PAYLOAD_MB}" \
  --output-dir "${OUTPUT_DIR}"
