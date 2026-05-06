#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_PATH=${SCRIPT_DIR}/phase9_post_rate_probe.json
if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH=${2:?config path required}
elif [[ -n "${1:-}" ]]; then
  CONFIG_PATH=${1}
fi

exec bash "${REPO_ROOT}/research/code/phase4/run_b4_matrix.sh" --config "${CONFIG_PATH}"
