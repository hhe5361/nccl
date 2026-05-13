#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

export NCCL_PHASE5_LOG=${NCCL_PHASE5_LOG:-1}

exec bash "${REPO_ROOT}/research/code/phase4/run_b4_matrix.sh" \
  --config "${SCRIPT_DIR}/phase5_b4_post_receive_ddp.json" \
  "$@"
