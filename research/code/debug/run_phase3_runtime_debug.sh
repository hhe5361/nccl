#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase3_runtime_debug_$(date +%y%m%d_%H%M%S)}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
DEBUG_SHARED_ROOT=${DEBUG_SHARED_ROOT:-/mnt/nfs_share/cts_experiments/debug}
DEBUG_RUN_ROOT="${DEBUG_SHARED_ROOT}/${RUN_ID}"
DEBUG_WORKER_ROOT="${DEBUG_RUN_ROOT}/${WORKER_NAME}"
mkdir -p "${DEBUG_WORKER_ROOT}"

export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-1}
export NCCL_PHASE2_LOG=${NCCL_PHASE2_LOG:-1}
export NCCL_PHASE3_LOG=${NCCL_PHASE3_LOG:-1}
export TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/debug/collective_b3_runtime_debug.py}
export LOG_ROOT
export RUN_ID
export PHASE3_RUNTIME_DEBUG_ROOT="${DEBUG_WORKER_ROOT}"

snapshot_file="${DEBUG_WORKER_ROOT}/launcher_snapshot.txt"
stdout_file="${DEBUG_WORKER_ROOT}/launcher.stdout.log"
stderr_file="${DEBUG_WORKER_ROOT}/launcher.stderr.log"
env_file="${DEBUG_WORKER_ROOT}/debug_env.json"

python3 - <<'PY' > "${env_file}"
import json
import os

keys = [
    "RUN_ID",
    "WORKER_NAME",
    "MASTER_ADDR",
    "MASTER_PORT_BASE",
    "MASTER_SERVER",
    "NNODES",
    "RUN_MODES",
    "COLLECTIVE",
    "PLACEMENT",
    "PAYLOAD_MB",
    "DTYPE",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_PHASE0_LOG",
    "NCCL_PHASE2_LOG",
    "NCCL_PHASE3_LOG",
    "NCCL_RACK_MAP_FILE",
    "TORCH_DISTRIBUTED_DEBUG",
    "TORCH_SHOW_CPP_STACKTRACES",
    "PYTHONFAULTHANDLER",
    "TARGET_SCRIPT",
    "LOG_ROOT",
    "PHASE3_RUNTIME_DEBUG_ROOT",
]
print(json.dumps({k: os.environ.get(k, "") for k in keys}, indent=2, sort_keys=True))
PY

capture_snapshot() {
  {
    echo "==== snapshot_ts ===="
    date --iso-8601=ns 2>/dev/null || date
    echo
    echo "==== ps -ef ===="
    ps -ef || true
    echo
    echo "==== ss -ltnp ===="
    ss -ltnp || true
    echo
    echo "==== nvidia-smi ===="
    nvidia-smi || true
    echo
    echo "==== relevant env ===="
    env | grep -E '^(RUN_ID|WORKER_NAME|MASTER_|NNODES|RUN_MODES|COLLECTIVE|PLACEMENT|PAYLOAD_MB|DTYPE|NCCL_|TORCH_|PYTHONFAULTHANDLER|TARGET_SCRIPT|LOG_ROOT)=' | sort || true
  } > "${snapshot_file}" 2>&1 || true
}

preflight_validate() {
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "[runtime-debug] MASTER_ADDR must be set" >&2
    exit 2
  fi
  if [[ -z "${MASTER_PORT_BASE:-}" ]]; then
    echo "[runtime-debug] MASTER_PORT_BASE must be set" >&2
    exit 2
  fi
  if [[ -z "${NNODES:-}" ]]; then
    echo "[runtime-debug] NNODES must be set" >&2
    exit 2
  fi
  if ! [[ "${NNODES}" =~ ^[0-9]+$ ]]; then
    echo "[runtime-debug] NNODES must be numeric: '${NNODES}'" >&2
    exit 2
  fi
  if ! [[ "${MASTER_PORT_BASE}" =~ ^[0-9]+$ ]]; then
    echo "[runtime-debug] MASTER_PORT_BASE must be numeric: '${MASTER_PORT_BASE}'" >&2
    exit 2
  fi
  if (( NNODES < 1 )); then
    echo "[runtime-debug] NNODES must be >= 1" >&2
    exit 2
  fi
  if ! [[ "${WORKER_NAME}" =~ ^worker([0-9]+)$ ]]; then
    echo "[runtime-debug] WORKER_NAME must look like worker01..worker08: '${WORKER_NAME}'" >&2
    exit 2
  fi
  local idx=${BASH_REMATCH[1]}
  local node_rank=$((10#${idx} - 1))
  if (( node_rank < 0 || node_rank >= NNODES )); then
    echo "[runtime-debug] inferred node rank ${node_rank} is outside NNODES=${NNODES}" >&2
    exit 2
  fi
}

is_master_node() {
  [[ "${WORKER_NAME}" == "worker01" || "${WORKER_NAME}" == "${MASTER_SERVER:-}" ]]
}

preflight_port_check() {
  if ! is_master_node; then
    return 0
  fi
  local port=${MASTER_PORT_BASE}
  if ss -ltnp 2>/dev/null | grep -q "[.:]${port}[[:space:]]"; then
    echo "[runtime-debug] rendezvous port ${port} is already in use on ${WORKER_NAME}" >&2
    ss -ltnp 2>/dev/null | grep "[.:]${port}[[:space:]]" >&2 || true
    exit 3
  fi
}

capture_nccl_tails() {
  local out="${DEBUG_WORKER_ROOT}/nccl_log_tails.txt"
  {
    echo "==== nccl log tail capture ===="
    date --iso-8601=ns 2>/dev/null || date
    shopt -s nullglob
    for log in "${LOG_ROOT}"/*/"${WORKER_NAME}"/nccl.*.log; do
      echo
      echo "==== ${log} ===="
      tail -n 120 "${log}" || true
    done
  } > "${out}" 2>&1 || true
}

preflight_validate
preflight_port_check
capture_snapshot

set +e
bash "${REPO_ROOT}/research/code/phase3/run_b3_collective.sh" "$@" \
  > >(tee "${stdout_file}") \
  2> >(tee "${stderr_file}" >&2)
rc=$?
set -e

capture_snapshot
capture_nccl_tails

if (( rc != 0 )); then
  echo "[runtime-debug] phase3 collective run failed rc=${rc}" >&2
  exit "${rc}"
fi

echo "[runtime-debug] phase3 collective run completed"
