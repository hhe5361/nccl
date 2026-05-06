#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_PATH=${SCRIPT_DIR}/phase10_phase4_congestion.json
if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH=${2:?config path required}
elif [[ -n "${1:-}" ]]; then
  CONFIG_PATH=${1}
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[phase10-matrix] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

eval "$(
python3 - "${CONFIG_PATH}" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
run = cfg.get("run", {})
paths = cfg.get("paths", {})
switch = cfg.get("switch_logging", {})
pairs = {
    "CFG_RUN_ID_PREFIX": run.get("run_id_prefix", "phase10_phase4_congestion"),
    "CFG_LOG_ROOT_BASE": paths.get("log_root_base", "/mnt/nfs_share/cts_experiments"),
    "CFG_SWITCH_ENABLE": int(switch.get("enable", 1)),
    "CFG_SWITCH_LOGGER_ROOT": switch.get("switch_logger_root", "/home/ubuntu/hyoeun/switch_setup_task/switch_congestion_logger"),
    "CFG_SWITCH_LOG_INTERVAL_SEC": switch.get("switch_log_interval_sec", 1),
    "CFG_SWITCH_LOG_SHARED_ROOT": switch.get("switch_log_shared_root", "/mnt/nfs_share/cts_experiments/switch_log"),
}
for k, v in pairs.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

RUN_ID=${RUN_ID:-${CFG_RUN_ID_PREFIX}_$(date +%y%m%d_%H%M%S)}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-${CFG_LOG_ROOT_BASE}}
LOG_ROOT=${LOG_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-${CFG_SWITCH_ENABLE}}
SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT:-${CFG_SWITCH_LOGGER_ROOT}}
SWITCH_LOG_INTERVAL_SEC=${SWITCH_LOG_INTERVAL_SEC:-${CFG_SWITCH_LOG_INTERVAL_SEC}}
SWITCH_LOG_SHARED_ROOT=${SWITCH_LOG_SHARED_ROOT:-${CFG_SWITCH_LOG_SHARED_ROOT}}

DPU_NODE_HOST=${DPU_NODE_HOST:-172.16.0.100}
DPU_NODE_USER=${DPU_NODE_USER:-ubuntu}
DPU_NODE_PORT=${DPU_NODE_PORT:-22}
NETWORK_NODE_PORT=${NETWORK_NODE_PORT:-}

SWITCH_LOG_RUN_ID=
SWITCH_LOG_DIR=
SWITCH_LOG_LOCAL_DIR=
SWITCH_LOG_PID_FILE=
SWITCH_LOG_MARKERS_JSONL=
SWITCH_LOG_STARTED=0
SWITCH_ENV_FILE="${LOG_ROOT}/switch_logger.env"

mkdir -p "${LOG_ROOT}"

require_sshpass() {
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase10-matrix] sshpass is required for switch logging." >&2
    exit 1
  fi
}

remote_dpu_bash() {
  local cmd=$1
  require_sshpass
  if [[ -z "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase10-matrix] DPU_NODE_PWD must be set when switch logging is enabled." >&2
    exit 1
  fi
  sshpass -p "${DPU_NODE_PWD}" \
    ssh -p "${DPU_NODE_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${DPU_NODE_USER}@${DPU_NODE_HOST}" \
    "bash -lc $(printf '%q' "${cmd}")"
}

write_switch_metadata() {
  cat > "${SWITCH_ENV_FILE}" <<EOF
SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE}
SWITCH_LOG_RUN_ID=${SWITCH_LOG_RUN_ID}
SWITCH_LOG_DIR=${SWITCH_LOG_DIR}
SWITCH_LOG_LOCAL_DIR=${SWITCH_LOG_LOCAL_DIR}
SWITCH_LOG_PID_FILE=${SWITCH_LOG_PID_FILE}
SWITCH_LOG_MARKERS_JSONL=${SWITCH_LOG_MARKERS_JSONL}
EOF
}

start_switch_logger() {
  if [[ "${SWITCH_LOG_ENABLE}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${NETWORK_NODE_PASSWORD:-}" || -z "${SWITCH_PASSWORD:-}" ]]; then
    echo "[phase10-matrix] NETWORK_NODE_PASSWORD and SWITCH_PASSWORD must be set when switch logging is enabled." >&2
    exit 1
  fi
  local cmd output
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && "
  if [[ -n "${NETWORK_NODE_PORT}" ]]; then
    cmd+="export NETWORK_NODE_PORT=$(printf '%q' "${NETWORK_NODE_PORT}") && "
  fi
  cmd+="./start_switch_congestion_loggers.sh --interval-sec $(printf '%q' "${SWITCH_LOG_INTERVAL_SEC}") --network-node-password $(printf '%q' "${NETWORK_NODE_PASSWORD}") --switch-password $(printf '%q' "${SWITCH_PASSWORD}")"
  output=$(remote_dpu_bash "${cmd}")
  while IFS='=' read -r key value; do
    case "${key}" in
      RUN_ID) SWITCH_LOG_RUN_ID=${value} ;;
      LOG_DIR) SWITCH_LOG_DIR=${value} ;;
      PID_FILE) SWITCH_LOG_PID_FILE=${value} ;;
      MARKERS_JSONL) SWITCH_LOG_MARKERS_JSONL=${value} ;;
    esac
  done <<< "${output}"
  if [[ -z "${SWITCH_LOG_RUN_ID}" || -z "${SWITCH_LOG_PID_FILE}" ]]; then
    echo "[phase10-matrix] failed to parse switch logger metadata" >&2
    echo "${output}" >&2
    exit 1
  fi
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
  SWITCH_LOG_STARTED=1
  write_switch_metadata
  echo "[phase10-matrix] switch logger started run_id=${SWITCH_LOG_RUN_ID} local_dir=${SWITCH_LOG_LOCAL_DIR}"
  remote_dpu_bash "cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --marker matrix_start --source phase10_matrix --message $(printf '%q' "run_id=${RUN_ID}")" || true
}

stop_switch_logger() {
  if [[ "${SWITCH_LOG_STARTED}" != "1" ]]; then
    return 0
  fi
  remote_dpu_bash "cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --marker matrix_end --source phase10_matrix --message $(printf '%q' "run_id=${RUN_ID}")" || true
  remote_dpu_bash "cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./stop_switch_congestion_loggers.sh --pid-file $(printf '%q' "${SWITCH_LOG_PID_FILE}")" || true
  echo "[phase10-matrix] switch logger stopped run_id=${SWITCH_LOG_RUN_ID}"
}

cleanup() {
  stop_switch_logger
}
trap cleanup EXIT INT TERM

export RUN_ID
export LOG_ROOT
start_switch_logger
set +e
bash "${REPO_ROOT}/research/code/phase4/run_b4_matrix.sh" --config "${CONFIG_PATH}"
rc=$?
set -e
exit "${rc}"
