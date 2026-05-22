#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_PATH=${1:-"${SCRIPT_DIR}/phase7_ddp_timeline.json"}

exec "${SCRIPT_DIR}/../phase4/run_b4_matrix.sh" --config "${CONFIG_PATH}"
