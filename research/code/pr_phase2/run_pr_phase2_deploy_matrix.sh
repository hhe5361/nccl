#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/pr_phase2_config.json}"
MODES_CONFIG_PATH="${MODES_CONFIG_PATH:-${SCRIPT_DIR}/pr_phase2_modes.json}"
OUTPUT_ROOT_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage:
  run_pr_phase2_deploy_matrix.sh [options]

Options:
  --config FILE
  --modes-config FILE
  --output-root DIR      # optional override for config output_root
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="${2:-}"; shift 2 ;;
    --modes-config) MODES_CONFIG_PATH="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT_OVERRIDE="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -f "${CONFIG_PATH}" ]] || { echo "Config file not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${MODES_CONFIG_PATH}" ]] || { echo "Modes config file not found: ${MODES_CONFIG_PATH}" >&2; exit 1; }

read_config_value() {
  local key="$1"
  python3 - "${CONFIG_PATH}" "${key}" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
value = data.get(key, "")
if isinstance(value, bool):
    print("1" if value else "0")
elif isinstance(value, list):
    print(",".join(str(x) for x in value))
else:
    print(value)
PY
}

read_modes_csv() {
  python3 - "${MODES_CONFIG_PATH}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
modes = data.get("modes", [])
print(",".join(str(x) for x in modes))
PY
}

OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-$(read_config_value output_root)}"
STEPS="$(read_config_value steps)"
WARMUP_STEPS="$(read_config_value warmup_steps)"
PAYLOAD_MB="$(read_config_value payload_mb)"
REPEATS="$(read_config_value repeats)"
NPROC_PER_NODE="$(read_config_value nproc_per_node)"
VENV_DIR="$(read_config_value venv_dir)"
BOOTSTRAP_IF_MISSING="$(read_config_value bootstrap_if_missing)"
BOOTSTRAP_TARGET="$(read_config_value bootstrap_target)"
TIMEOUT_SEC="$(read_config_value timeout_sec)"
EXPERIMENTS_CSV="$(read_config_value experiments)"
HIDDEN_DIM="$(read_config_value hidden_dim)"
NUM_LAYERS="$(read_config_value num_layers)"
BATCH_SIZE="$(read_config_value batch_size)"
BUCKET_CAP_MB="$(read_config_value bucket_cap_mb)"
LR="$(read_config_value lr)"
MODES_CSV="$(read_modes_csv)"

[[ -n "${OUTPUT_ROOT}" ]] || { echo "output_root is missing in config and not overridden" >&2; exit 1; }
[[ -n "${MODES_CSV}" ]] || { echo "No modes found in ${MODES_CONFIG_PATH}" >&2; exit 1; }
[[ -n "${EXPERIMENTS_CSV}" ]] || { echo "No experiments found in ${CONFIG_PATH}" >&2; exit 1; }

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
  --python-script 'research/code/pr_phase2/ddp_model_runner.py' \
  --bootstrap-if-missing ${BOOTSTRAP_IF_MISSING} \
  --bootstrap-target ${BOOTSTRAP_TARGET} \
  --runner-extra-args '--hidden-dim ${HIDDEN_DIM} --num-layers ${NUM_LAYERS} --batch-size ${BATCH_SIZE} --bucket-cap-mb ${BUCKET_CAP_MB} --lr ${LR}'"

master_args=(
  --output-root "${OUTPUT_ROOT}"
  --runner-template "${runner_template}"
  --repeats "${REPEATS}"
  --modes "${MODES_CSV}"
  --experiments "${EXPERIMENTS_CSV}"
  --switch-enable "0"
  --timeout-sec "${TIMEOUT_SEC}"
)

env \
  SWITCH_ENABLE="0" \
  REPEATS="${REPEATS}" \
  MODES_CSV="${MODES_CSV}" \
  EXPERIMENTS_CSV="${EXPERIMENTS_CSV}" \
  TIMEOUT_SEC="${TIMEOUT_SEC}" \
  bash "${REPO_ROOT}/research/code/deploy/run_dev_matrix_master.sh" \
    "${master_args[@]}"
