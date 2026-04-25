#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:-phase3_runtime_debug_$(date +%y%m%d_%H%M%S)}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
DEBUG_ROOT="${LOG_ROOT}/_runtime_debug/${WORKER_NAME}"
mkdir -p "${DEBUG_ROOT}"

export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-1}
export NCCL_PHASE2_LOG=${NCCL_PHASE2_LOG:-1}
export NCCL_PHASE3_LOG=${NCCL_PHASE3_LOG:-1}
export TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/debug/collective_b3_runtime_debug.py}
export LOG_ROOT
export RUN_ID

snapshot_file="${DEBUG_ROOT}/launcher_snapshot.txt"
stdout_file="${DEBUG_ROOT}/launcher.stdout.log"
stderr_file="${DEBUG_ROOT}/launcher.stderr.log"
env_file="${DEBUG_ROOT}/debug_env.json"

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

capture_nccl_tails() {
  local out="${DEBUG_ROOT}/nccl_log_tails.txt"
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
