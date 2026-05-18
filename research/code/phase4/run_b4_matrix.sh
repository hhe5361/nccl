#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_PATH=${SCRIPT_DIR}/phase4_b4_post_receive_ddp.json
NET_BURST=${NET_BURST:-0}
while (( $# > 0 )); do
  case "${1}" in
    --config)
      CONFIG_PATH=${2:?config path required}
      shift 2
      ;;
    --net-burst)
      NET_BURST=${2:?net-burst value required}
      shift 2
      ;;
    *)
      CONFIG_PATH=${1}
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[phase4-matrix] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT:-/home/ubuntu/hyoeun/switch_setup_task}
SWITCH_LOGGER_V2_SUBDIR=${SWITCH_LOGGER_V2_SUBDIR:-switch_congestion_logger_v2}
SWITCH_LOG_INTERVAL_SEC=${SWITCH_LOG_INTERVAL_SEC:-1}
SWITCH_LOG_SHARED_ROOT=${SWITCH_LOG_SHARED_ROOT:-/mnt/nfs_share/cts_experiments/switch_log}
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

eval "$(
python3 - "${CONFIG_PATH}" <<'PY'
import json, shlex, sys

cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
phase_name = cfg.get("phase", "phase4")
run = cfg.get("run", {})
model = cfg.get("model", {})
container = cfg.get("container", {})
ssh = cfg.get("ssh", {})
paths = cfg.get("paths", {})
nccl = cfg.get("nccl", {})
modes = [m.get("mode_id") for m in cfg.get("mode_matrix", []) if m.get("mode_id")]
mode_matrix_json = json.dumps(cfg.get("mode_matrix", []), separators=(",", ":"))
workers = run.get("all_workers", [])

pairs = {
    "CFG_RUN_ID_PREFIX": run.get("run_id_prefix", "phase4_b4_ddp"),
    "CFG_MASTER_SERVER": run.get("master_server", "worker01"),
    "CFG_MASTER_ADDR": run.get("master_addr", "172.16.0.101"),
    "CFG_MASTER_PORT": run.get("master_port", 40100),
    "CFG_LOG_ROOT_BASE": paths.get("log_root_base", "/mnt/nfs_share/cts_experiments"),
    "CFG_ALL_WORKERS": ",".join(workers),
    "CFG_RUN_MODES": ",".join(modes),
    "CFG_MODE_MATRIX_JSON": mode_matrix_json,
    "CFG_STEPS": run.get("steps", 40),
    "CFG_WARMUP_STEPS": run.get("warmup_steps", 5),
    "CFG_REPEATS": run.get("repeats", 1),
    "CFG_MODE_TIMEOUT_SEC": run.get("mode_timeout_sec", 600),
    "CFG_MODE_DELAY_SEC": run.get("mode_delay_sec", 180),
    "CFG_TORCH_ENV": run.get("torch_env", "/workspace/venvs/torch-cu121-custom/bin/activate"),
    "CFG_TARGET_SCRIPT": run.get("target_script", "research/code/phase4/ddp_b4.py"),
    "CFG_WORKER_SCRIPT": run.get("worker_script", "research/code/phase4/run_b4_mode_worker.sh"),
    "CFG_COMPARE_SCRIPT": run.get("compare_script", "research/code/phase4/compare_b4_vs_stock.py"),
    "CFG_SUMMARY_SCRIPT": run.get("summary_script", "research/code/phase4/phase4_ncc_event_summary.py"),
    "CFG_SUMMARY_OUTPUT_NAME": run.get("summary_output_name", "phase4_ncc_event_summary.json"),
    "CFG_HOST_REPO_ROOT": paths.get("host_repo_root", ""),
    "CFG_REMOTE_REPO_ROOT": paths.get("remote_repo_root", ""),
    "CFG_REMOTE_REPO_ROOT_MAP": paths.get("remote_repo_root_map", ""),
    "CFG_CONTAINER_REPO_ROOT": paths.get("container_repo_root", ""),
    "CFG_NETWORK_TOPOLOGY_FILE": paths.get("network_topology_file", "research/env/network_topology_internal_ips.txt"),
    "CFG_CONTAINER_NAME": container.get("name", "nccl-cu121-dev"),
    "CFG_CONTAINER_RESET_AT_START": int(container.get("reset_at_start", 1)),
    "CFG_RESTART_CONTAINERS_ON_FAILURE": int(container.get("restart_on_failure", 1)),
    "CFG_WORKER_SSH_USER": ssh.get("worker_ssh_user", ""),
    "CFG_WORKER_SSH_USER_MAP": ssh.get("worker_ssh_user_map", ""),
    "CFG_WORKER_SSH_PORT": ssh.get("worker_ssh_port", 22),
    "CFG_WORKER_SSH_PORT_MAP": ssh.get("worker_ssh_port_map", ""),
    "CFG_SSH_CONNECT_TIMEOUT_SEC": ssh.get("ssh_connect_timeout_sec", 10),
    "CFG_DTYPE": model.get("dtype", "float32"),
    "CFG_HIDDEN_DIM": model.get("hidden_dim", 128),
    "CFG_NUM_LAYERS": model.get("num_layers", 2),
    "CFG_BATCH_SIZE": model.get("batch_size", 8),
    "CFG_BUCKET_CAP_MB": model.get("bucket_cap_mb", 1),
    "CFG_LR": model.get("lr", 0.01),
    "CFG_MODEL_SEED": model.get("model_seed", 20260504),
    "CFG_NCCL_ALGO": nccl.get("algo", "auto"),
    "CFG_NCCL_PROTO": nccl.get("proto", "auto"),
    "CFG_NCCL_PHASE0_LOG": nccl.get("phase0_log", 1),
    "CFG_NCCL_PHASE4_LOG": nccl.get("phase4_log", 0),
    "CFG_NCCL_PHASE6_LOG": nccl.get("phase6_log", 0),
    "CFG_NCCL_PHASE7_LOG": nccl.get("phase7_log", 0),
    "CFG_NCCL_PHASE7_BURST_FLOOR_POSTS": nccl.get("phase7_burst_floor_posts", 4),
    "CFG_NCCL_PHASE10_LOG": nccl.get("phase10_log", 0),
    "CFG_NCCL_APPENDIX2_GROUP_LOG": nccl.get("appendix2_group_log", 0),
    "CFG_NCCL_DEBUG": nccl.get("debug", "INFO"),
    "CFG_NCCL_DEBUG_SUBSYS": nccl.get("debug_subsys", "NET"),
    "CFG_NCCL_NET_GDR_LEVEL": nccl.get("net_gdr_level", 0),
    "CFG_PHASE_NAME": phase_name,
}

for key, value in pairs.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

RUN_ID=${RUN_ID:-${CFG_RUN_ID_PREFIX}_$(date +%y%m%d_%H%M%S)}
MASTER_SERVER=${MASTER_SERVER:-${CFG_MASTER_SERVER}}
MASTER_ADDR=${MASTER_ADDR:-${CFG_MASTER_ADDR}}
MASTER_PORT=${MASTER_PORT:-${CFG_MASTER_PORT}}
LOG_ROOT_BASE=${LOG_ROOT_BASE:-${CFG_LOG_ROOT_BASE}}
LOG_ROOT=${LOG_ROOT:-${LOG_ROOT_BASE}/${RUN_ID}}
ALL_WORKERS=${ALL_WORKERS:-${CFG_ALL_WORKERS}}
RUN_MODES=${RUN_MODES:-${CFG_RUN_MODES}}
MODE_MATRIX_JSON=${MODE_MATRIX_JSON:-${CFG_MODE_MATRIX_JSON}}
STEPS=${STEPS:-${CFG_STEPS}}
WARMUP_STEPS=${WARMUP_STEPS:-${CFG_WARMUP_STEPS}}
REPEATS=${REPEATS:-${CFG_REPEATS}}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-${CFG_MODE_TIMEOUT_SEC}}
MODE_DELAY_SEC=${MODE_DELAY_SEC:-${CFG_MODE_DELAY_SEC}}
TORCH_ENV=${TORCH_ENV:-${CFG_TORCH_ENV}}
TARGET_SCRIPT=${TARGET_SCRIPT:-${CFG_TARGET_SCRIPT}}
WORKER_SCRIPT=${WORKER_SCRIPT:-${CFG_WORKER_SCRIPT}}
COMPARE_SCRIPT=${COMPARE_SCRIPT:-${CFG_COMPARE_SCRIPT}}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-${CFG_SUMMARY_SCRIPT}}
SUMMARY_OUTPUT_NAME=${SUMMARY_OUTPUT_NAME:-${CFG_SUMMARY_OUTPUT_NAME}}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-${CFG_HOST_REPO_ROOT:-${REPO_ROOT}}}
REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-${CFG_REMOTE_REPO_ROOT:-${HOST_REPO_ROOT}}}
REMOTE_REPO_ROOT_MAP=${REMOTE_REPO_ROOT_MAP:-${CFG_REMOTE_REPO_ROOT_MAP}}
CONTAINER_REPO_ROOT=${CONTAINER_REPO_ROOT:-${CFG_CONTAINER_REPO_ROOT:-/workspace/$(basename "${REPO_ROOT}")}}
NETWORK_TOPOLOGY_FILE=${NETWORK_TOPOLOGY_FILE:-${CFG_NETWORK_TOPOLOGY_FILE}}
CONTAINER_NAME=${CONTAINER_NAME:-${CFG_CONTAINER_NAME}}
CONTAINER_RESET_AT_START=${CONTAINER_RESET_AT_START:-${CFG_CONTAINER_RESET_AT_START}}
RESTART_CONTAINERS_ON_FAILURE=${RESTART_CONTAINERS_ON_FAILURE:-${CFG_RESTART_CONTAINERS_ON_FAILURE}}
WORKER_SSH_USER=${WORKER_SSH_USER:-${CFG_WORKER_SSH_USER}}
WORKER_SSH_USER_MAP=${WORKER_SSH_USER_MAP:-${CFG_WORKER_SSH_USER_MAP}}
WORKER_SSH_PORT=${WORKER_SSH_PORT:-${CFG_WORKER_SSH_PORT}}
WORKER_SSH_PORT_MAP=${WORKER_SSH_PORT_MAP:-${CFG_WORKER_SSH_PORT_MAP}}
SSH_CONNECT_TIMEOUT_SEC=${SSH_CONNECT_TIMEOUT_SEC:-${CFG_SSH_CONNECT_TIMEOUT_SEC}}
DTYPE=${DTYPE:-${CFG_DTYPE}}
HIDDEN_DIM=${HIDDEN_DIM:-${CFG_HIDDEN_DIM}}
NUM_LAYERS=${NUM_LAYERS:-${CFG_NUM_LAYERS}}
BATCH_SIZE=${BATCH_SIZE:-${CFG_BATCH_SIZE}}
BUCKET_CAP_MB=${BUCKET_CAP_MB:-${CFG_BUCKET_CAP_MB}}
LR=${LR:-${CFG_LR}}
MODEL_SEED=${MODEL_SEED:-${CFG_MODEL_SEED}}
NCCL_ALGO=${NCCL_ALGO:-${CFG_NCCL_ALGO}}
NCCL_PROTO=${NCCL_PROTO:-${CFG_NCCL_PROTO}}
NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-${CFG_NCCL_PHASE0_LOG}}
NCCL_PHASE4_LOG=${NCCL_PHASE4_LOG:-${CFG_NCCL_PHASE4_LOG}}
NCCL_PHASE6_LOG=${NCCL_PHASE6_LOG:-${CFG_NCCL_PHASE6_LOG}}
NCCL_PHASE6_W_BASE=${NCCL_PHASE6_W_BASE:-}
NCCL_PHASE6_W_MIN=${NCCL_PHASE6_W_MIN:-}
NCCL_PHASE6_W_MAX=${NCCL_PHASE6_W_MAX:-}
NCCL_PHASE6_W_STEP=${NCCL_PHASE6_W_STEP:-}
NCCL_PHASE6_KP=${NCCL_PHASE6_KP:-}
NCCL_PHASE6_KI=${NCCL_PHASE6_KI:-}
NCCL_PHASE6_EPOCH_MS=${NCCL_PHASE6_EPOCH_MS:-20}
NCCL_PHASE6_MIN_SAMPLES=${NCCL_PHASE6_MIN_SAMPLES:-16}
NCCL_PHASE6_WARMUP_EPOCHS=${NCCL_PHASE6_WARMUP_EPOCHS:-10}
NCCL_PHASE6_COOLDOWN_EPOCHS=${NCCL_PHASE6_COOLDOWN_EPOCHS:-3}
NCCL_PHASE6_STABLE_EPOCHS=${NCCL_PHASE6_STABLE_EPOCHS:-5}
NCCL_PHASE6_THRESHOLD_HIGH_PCT=${NCCL_PHASE6_THRESHOLD_HIGH_PCT:-25}
NCCL_PHASE6_THRESHOLD_LOW_PCT=${NCCL_PHASE6_THRESHOLD_LOW_PCT:-10}
NCCL_PHASE6_THROUGHPUT_LOW_PCT=${NCCL_PHASE6_THROUGHPUT_LOW_PCT:-85}
NCCL_PHASE6_WSTALL_HIGH_PCT=${NCCL_PHASE6_WSTALL_HIGH_PCT:-20}
NCCL_PHASE6_INTEGRAL_LIMIT_PCT=${NCCL_PHASE6_INTEGRAL_LIMIT_PCT:-500}
NCCL_PHASE6_ALPHA_FAST_PCT=${NCCL_PHASE6_ALPHA_FAST_PCT:-30}
NCCL_PHASE6_ALPHA_SLOW_PCT=${NCCL_PHASE6_ALPHA_SLOW_PCT:-5}
NCCL_PHASE7_LOG=${NCCL_PHASE7_LOG:-${CFG_NCCL_PHASE7_LOG}}
NCCL_PHASE7_BURST_FLOOR_POSTS=${NCCL_PHASE7_BURST_FLOOR_POSTS:-${CFG_NCCL_PHASE7_BURST_FLOOR_POSTS}}
NCCL_PHASE10_LOG=${NCCL_PHASE10_LOG:-${CFG_NCCL_PHASE10_LOG}}
NCCL_APPENDIX2_GROUP_LOG=${NCCL_APPENDIX2_GROUP_LOG:-${CFG_NCCL_APPENDIX2_GROUP_LOG}}
NCCL_DEBUG=${NCCL_DEBUG:-${CFG_NCCL_DEBUG}}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-${CFG_NCCL_DEBUG_SUBSYS}}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-${CFG_NCCL_NET_GDR_LEVEL}}
PHASE_NAME=${PHASE_NAME:-${CFG_PHASE_NAME}}

if [[ "${NETWORK_TOPOLOGY_FILE}" != /* ]]; then
  NETWORK_TOPOLOGY_FILE="${REPO_ROOT}/${NETWORK_TOPOLOGY_FILE}"
fi

mkdir -p "${LOG_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[phase4-matrix] docker is required on the master host." >&2
  exit 1
fi

if [[ "$(hostname -s)" != "${MASTER_SERVER}" ]]; then
  echo "[phase4-matrix] run this script on ${MASTER_SERVER}. current host=$(hostname -s)" >&2
  exit 1
fi

IFS=',' read -r -a ALL_WORKER_ARRAY <<< "${ALL_WORKERS}"
IFS=',' read -r -a MODE_VALUES <<< "${RUN_MODES}"

if (( ${#ALL_WORKER_ARRAY[@]} == 0 )); then
  echo "[phase4-matrix] ALL_WORKERS is empty." >&2
  exit 1
fi

if [[ "${ALL_WORKER_ARRAY[0]}" != "${MASTER_SERVER}" ]]; then
  echo "[phase4-matrix] ALL_WORKERS must start with MASTER_SERVER=${MASTER_SERVER}." >&2
  exit 1
fi

count_in_list() {
  local list=$1
  local count=0
  local item
  IFS=',' read -r -a _items <<< "${list}"
  for item in "${_items[@]}"; do
    [[ -n "${item}" ]] && count=$((count + 1))
  done
  echo "${count}"
}

NNODES=$(count_in_list "${ALL_WORKERS}")

require_sshpass() {
  if [[ -n "${WORKER_SSH_PASSWORD:-}" ]] && ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase4-matrix] sshpass is required when WORKER_SSH_PASSWORD is set." >&2
    exit 1
  fi
}

require_switch_sshpass() {
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase4-matrix] sshpass is required for switch logging." >&2
    exit 1
  fi
}

remote_dpu_bash() {
  local cmd=$1
  require_switch_sshpass
  if [[ -z "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase4-matrix] DPU_NODE_PWD must be set when switch logging is enabled." >&2
    exit 1
  fi
  sshpass -p "${DPU_NODE_PWD}" \
    ssh -p "${DPU_NODE_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${DPU_NODE_USER}@${DPU_NODE_HOST}" \
    "bash -lc $(printf '%q' "${cmd}")"
}

switch_logger_dir_on_dpu() {
  local root=${SWITCH_LOGGER_ROOT%/}
  local subdir=${SWITCH_LOGGER_V2_SUBDIR#/}
  printf '%s/%s' "${root}" "${subdir}"
}

start_switch_logger() {
  if [[ -z "${NETWORK_NODE_PASSWORD:-}" || -z "${SWITCH_PASSWORD:-}" ]]; then
    echo "[phase4-matrix] NETWORK_NODE_PASSWORD and SWITCH_PASSWORD must be set when switch logging is enabled." >&2
    exit 1
  fi
  local cmd output switch_logger_dir
  switch_logger_dir=$(switch_logger_dir_on_dpu)
  cmd="cd $(printf '%q' "${switch_logger_dir}") && "
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
    echo "[phase4-matrix] failed to parse switch logger metadata" >&2
    echo "${output}" >&2
    exit 1
  fi
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
  SWITCH_LOG_STARTED=1
  echo "[phase4-matrix] switch logger started run_id=${SWITCH_LOG_RUN_ID} local_dir=${SWITCH_LOG_LOCAL_DIR}"
  remote_dpu_bash "cd $(printf '%q' "${switch_logger_dir}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --output-jsonl $(printf '%q' "${SWITCH_LOG_MARKERS_JSONL}") --marker matrix_start --source phase4_matrix --message $(printf '%q' "run_id=${RUN_ID} net_burst=${NET_BURST}")" || true
  cat > "${LOG_ROOT}/switch_logger.env" <<EOF
SWITCH_LOG_RUN_ID=${SWITCH_LOG_RUN_ID}
SWITCH_LOG_DIR=${SWITCH_LOG_DIR}
SWITCH_LOG_LOCAL_DIR=${SWITCH_LOG_LOCAL_DIR}
SWITCH_LOG_PID_FILE=${SWITCH_LOG_PID_FILE}
SWITCH_LOG_MARKERS_JSONL=${SWITCH_LOG_MARKERS_JSONL}
SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT}
SWITCH_LOGGER_V2_SUBDIR=${SWITCH_LOGGER_V2_SUBDIR}
NET_BURST=${NET_BURST}
EOF
}

stop_switch_logger() {
  if [[ "${SWITCH_LOG_STARTED}" != "1" ]]; then
    return 0
  fi
  local switch_logger_dir
  switch_logger_dir=$(switch_logger_dir_on_dpu)
  remote_dpu_bash "cd $(printf '%q' "${switch_logger_dir}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --output-jsonl $(printf '%q' "${SWITCH_LOG_MARKERS_JSONL}") --marker matrix_end --source phase4_matrix --message $(printf '%q' "run_id=${RUN_ID} net_burst=${NET_BURST}")" || true
  remote_dpu_bash "cd $(printf '%q' "${switch_logger_dir}") && ./stop_switch_congestion_loggers.sh --pid-file $(printf '%q' "${SWITCH_LOG_PID_FILE}")" || true
  echo "[phase4-matrix] switch logger stopped run_id=${SWITCH_LOG_RUN_ID}"
}

emit_switch_marker() {
  local marker=$1
  local message=$2
  local switch_logger_dir
  if [[ "${SWITCH_LOG_STARTED}" != "1" ]]; then
    return 0
  fi
  switch_logger_dir=$(switch_logger_dir_on_dpu)
  remote_dpu_bash "cd $(printf '%q' "${switch_logger_dir}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --output-jsonl $(printf '%q' "${SWITCH_LOG_MARKERS_JSONL}") --marker $(printf '%q' "${marker}") --source phase4_matrix --message $(printf '%q' "${message}")" || true
}

resolve_worker_ssh_user() {
  local worker=$1
  local mapping entry map_worker map_user
  if [[ -n "${WORKER_SSH_USER_MAP}" ]]; then
    IFS=',' read -r -a mapping <<< "${WORKER_SSH_USER_MAP}"
    for entry in "${mapping[@]}"; do
      map_worker=${entry%%=*}
      map_user=${entry#*=}
      if [[ "${map_worker}" == "${worker}" && -n "${map_user}" ]]; then
        echo "${map_user}"
        return 0
      fi
    done
  fi
  if [[ -n "${WORKER_SSH_USER}" ]]; then
    echo "${WORKER_SSH_USER}"
    return 0
  fi
  echo "${worker}"
}

resolve_worker_ssh_host() {
  local worker=$1
  local ssh_host
  if [[ ! -f "${NETWORK_TOPOLOGY_FILE}" ]]; then
    echo "[phase4-matrix] NETWORK_TOPOLOGY_FILE not found: ${NETWORK_TOPOLOGY_FILE}" >&2
    return 1
  fi
  ssh_host=$(awk -v target="${worker}" '
    /^\[Workers\]/ { in_workers=1; next }
    /^\[/ && $0 !~ /^\[Workers\]/ { in_workers=0 }
    in_workers && $1 == "-" {
      gsub(":", "", $2)
      if ($2 == target) {
        print $3
        exit
      }
    }
  ' "${NETWORK_TOPOLOGY_FILE}")
  if [[ -z "${ssh_host}" ]]; then
    echo "[phase4-matrix] failed to resolve SSH host for worker=${worker} from ${NETWORK_TOPOLOGY_FILE}" >&2
    return 1
  fi
  echo "${ssh_host}"
}

resolve_all_worker_hosts_csv() {
  local hosts=()
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    hosts+=("$(resolve_worker_ssh_host "${worker}")")
  done
  local IFS=,
  echo "${hosts[*]}"
}

resolve_remote_repo_root() {
  local worker=$1
  local ssh_user root mapping entry map_worker map_root path_tail
  ssh_user=$(resolve_worker_ssh_user "${worker}")
  if [[ -n "${REMOTE_REPO_ROOT_MAP}" ]]; then
    IFS=',' read -r -a mapping <<< "${REMOTE_REPO_ROOT_MAP}"
    for entry in "${mapping[@]}"; do
      map_worker=${entry%%=*}
      map_root=${entry#*=}
      if [[ "${map_worker}" == "${worker}" && -n "${map_root}" ]]; then
        echo "${map_root}"
        return 0
      fi
    done
  fi
  root="${REMOTE_REPO_ROOT}"
  root="${root//\{worker\}/${worker}}"
  root="${root//\{user\}/${ssh_user}}"
  if [[ "${root}" == "${REMOTE_REPO_ROOT}" && "${root}" == /home/*/* ]]; then
    path_tail=${root#/home/*/}
    root="/home/${ssh_user}/${path_tail}"
  fi
  echo "${root}"
}

resolve_worker_ssh_port() {
  local worker=$1
  local mapping entry map_worker map_port
  if [[ -n "${WORKER_SSH_PORT_MAP}" ]]; then
    IFS=',' read -r -a mapping <<< "${WORKER_SSH_PORT_MAP}"
    for entry in "${mapping[@]}"; do
      map_worker=${entry%%=*}
      map_port=${entry#*=}
      if [[ "${map_worker}" == "${worker}" && -n "${map_port}" ]]; then
        echo "${map_port}"
        return 0
      fi
    done
  fi
  echo "${WORKER_SSH_PORT}"
}

mode_field() {
  local mode_id=$1
  local field=$2
  local default_value=${3:-}
  python3 - "${MODE_MATRIX_JSON}" "${mode_id}" "${field}" "${default_value}" <<'PY'
import json
import sys

matrix = json.loads(sys.argv[1])
mode_id = sys.argv[2]
field = sys.argv[3]
default_value = sys.argv[4]

for row in matrix:
    if row.get("mode_id") == mode_id:
        value = row.get(field, default_value)
        if value is None:
            value = default_value
        print(value)
        sys.exit(0)

print(default_value)
PY
}

remote_worker_bash() {
  local worker=$1
  local cmd=$2
  local ssh_user
  local ssh_host
  local ssh_port
  if [[ "${worker}" == "${MASTER_SERVER}" ]]; then
    bash -lc "${cmd}"
    return
  fi

  require_sshpass
  ssh_user=$(resolve_worker_ssh_user "${worker}")
  ssh_host=$(resolve_worker_ssh_host "${worker}")
  ssh_port=$(resolve_worker_ssh_port "${worker}")
  if [[ -n "${WORKER_SSH_PASSWORD:-}" ]]; then
    sshpass -p "${WORKER_SSH_PASSWORD}" \
      ssh -p "${ssh_port}" -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${ssh_user}@${ssh_host}" \
      "bash -lc $(printf '%q' "${cmd}")"
  else
    ssh -p "${ssh_port}" -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SEC}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${ssh_user}@${ssh_host}" \
      "bash -lc $(printf '%q' "${cmd}")"
  fi
}

wait_for_master_port_free() {
  local timeout_sec=${1:-30}
  local waited=0
  while (( waited < timeout_sec )); do
    if ! ss -ltn 2>/dev/null | awk -v port=":${MASTER_PORT}" 'index($4, port) { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[phase4-matrix] master port is free port=${MASTER_PORT}"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[phase4-matrix] ERROR: master port remained busy port=${MASTER_PORT}" >&2
  return 1
}

wait_for_master_port_listen() {
  local rank0_pid=$1
  local waited=0
  while (( waited < 60 )); do
    if ss -ltn 2>/dev/null | awk -v port=":${MASTER_PORT}" 'index($4, port) { found=1 } END { exit(found ? 0 : 1) }'; then
      echo "[phase4-matrix] rank0 port is listening port=${MASTER_PORT}"
      return 0
    fi
    if ! kill -0 "${rank0_pid}" >/dev/null 2>&1; then
      echo "[phase4-matrix] ERROR: rank0 launcher exited before opening port=${MASTER_PORT}" >&2
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "[phase4-matrix] ERROR: timed out waiting for rank0 port listen port=${MASTER_PORT}" >&2
  return 1
}

write_pending_status() {
  local status_file=$1
  local worker=$2
  local rank=$3
  local mode=$4
  mkdir -p "$(dirname "${status_file}")"
  cat > "${status_file}" <<EOF
status=9
state=pending
worker=${worker}
node_rank=${rank}
mode=${mode}
experiment=phase4_ddp_tiny
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
rc=
pid=
updated_at=$(date +%s)
message=pending
EOF
}

read_status_field() {
  local file=$1
  local key=$2
  awk -F= -v k="${key}" '$1 == k { print substr($0, length(k) + 2); exit }' "${file}" 2>/dev/null || true
}

prepare_worker_container() {
  local worker=$1
  local remote_repo_root
  local cmd
  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  if [[ "${CONTAINER_RESET_AT_START}" == "1" ]]; then
    cmd="docker rm -f $(printf '%q' "${CONTAINER_NAME}") >/dev/null 2>&1 || true; cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh true >/dev/null"
  else
    cmd="cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh true >/dev/null"
  fi
  echo "[phase4-matrix] prepare container worker=${worker} reset=${CONTAINER_RESET_AT_START}"
  remote_worker_bash "${worker}" "${cmd}"
}

cleanup_worker_processes() {
  local worker=$1
  local remote_repo_root
  local cmd
  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  cmd="cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh bash -lc $(printf '%q' "pkill -f 'torchrun|torch\\.distributed\\.run|ddp_b4\\.py' >/dev/null 2>&1 || true; sleep 1; ps -ef | grep -E 'torchrun|torch\\.distributed\\.run|ddp_b4\\.py' | grep -v grep || true")"
  remote_worker_bash "${worker}" "${cmd}" >/dev/null 2>&1 || true
}

cleanup_all_workers() {
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    cleanup_worker_processes "${worker}"
  done
}

build_worker_host_command() {
  local worker=$1
  local rank=$2
  local mode=$3
  local repeat_label=$4
  local phase4_enable_value=$5
  local phase4_post_receive_w_value=$6
  local net_burst_value=$7
  local phase6_enable_value=$8
  local phase6_post_rate_value=$9
  local phase6_post_burst_value=${10}
  local phase7_enable_value=${11}
  local phase7_ratio_pct_value=${12}
  local phase7_observe_ms_value=${13}
  local phase7_burst_window_ms_value=${14}
  local run_root=${15}
  local status_file=${16}
  local remote_repo_root

  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  local inner
  inner=$(cat <<EOF
cd $(printf '%q' "${CONTAINER_REPO_ROOT}") && \
RUN_ID=$(printf '%q' "${RUN_ID}") \
EXPERIMENT_LABEL=$(printf '%q' "phase4_ddp_tiny") \
MODE=$(printf '%q' "${mode}") \
REPEAT_LABEL=$(printf '%q' "${repeat_label}") \
PHASE4_ENABLE_VALUE=$(printf '%q' "${phase4_enable_value}") \
PHASE4_POST_RECEIVE_W_VALUE=$(printf '%q' "${phase4_post_receive_w_value}") \
NET_BURST_VALUE=$(printf '%q' "${net_burst_value}") \
PHASE6_ENABLE_VALUE=$(printf '%q' "${phase6_enable_value}") \
PHASE6_POST_RATE_VALUE=$(printf '%q' "${phase6_post_rate_value}") \
PHASE6_POST_BURST_VALUE=$(printf '%q' "${phase6_post_burst_value}") \
PHASE7_ENABLE_VALUE=$(printf '%q' "${phase7_enable_value}") \
PHASE7_RATIO_PCT_VALUE=$(printf '%q' "${phase7_ratio_pct_value}") \
PHASE7_OBSERVE_MS_VALUE=$(printf '%q' "${phase7_observe_ms_value}") \
PHASE7_BURST_WINDOW_MS_VALUE=$(printf '%q' "${phase7_burst_window_ms_value}") \
MASTER_ADDR=$(printf '%q' "${MASTER_ADDR}") \
MASTER_PORT=$(printf '%q' "${MASTER_PORT}") \
NNODES=$(printf '%q' "${NNODES}") \
NPROC_PER_NODE=1 \
NODE_RANK=$(printf '%q' "${rank}") \
WORKER_NAME=$(printf '%q' "${worker}") \
TORCH_ENV=$(printf '%q' "${TORCH_ENV}") \
TARGET_SCRIPT=$(printf '%q' "${TARGET_SCRIPT}") \
LOG_ROOT=$(printf '%q' "${LOG_ROOT}") \
RUN_ROOT=$(printf '%q' "${run_root}") \
STATUS_FILE=$(printf '%q' "${status_file}") \
STATUS_DIR=$(printf '%q' "$(dirname "${status_file}")") \
STEPS=$(printf '%q' "${STEPS}") \
WARMUP_STEPS=$(printf '%q' "${WARMUP_STEPS}") \
DTYPE=$(printf '%q' "${DTYPE}") \
MODE_TIMEOUT_SEC=$(printf '%q' "${MODE_TIMEOUT_SEC}") \
NCCL_ALGO=$(printf '%q' "${NCCL_ALGO}") \
NCCL_PROTO=$(printf '%q' "${NCCL_PROTO}") \
NCCL_PHASE0_LOG=$(printf '%q' "${NCCL_PHASE0_LOG}") \
NCCL_PHASE4_LOG=$(printf '%q' "${NCCL_PHASE4_LOG}") \
NCCL_PHASE6_LOG=$(printf '%q' "${NCCL_PHASE6_LOG}") \
NCCL_PHASE6_W_BASE=$(printf '%q' "${NCCL_PHASE6_W_BASE}") \
NCCL_PHASE6_W_MIN=$(printf '%q' "${NCCL_PHASE6_W_MIN}") \
NCCL_PHASE6_W_MAX=$(printf '%q' "${NCCL_PHASE6_W_MAX}") \
NCCL_PHASE6_W_STEP=$(printf '%q' "${NCCL_PHASE6_W_STEP}") \
NCCL_PHASE6_KP=$(printf '%q' "${NCCL_PHASE6_KP}") \
NCCL_PHASE6_KI=$(printf '%q' "${NCCL_PHASE6_KI}") \
NCCL_PHASE6_EPOCH_MS=$(printf '%q' "${NCCL_PHASE6_EPOCH_MS}") \
NCCL_PHASE6_MIN_SAMPLES=$(printf '%q' "${NCCL_PHASE6_MIN_SAMPLES}") \
NCCL_PHASE6_WARMUP_EPOCHS=$(printf '%q' "${NCCL_PHASE6_WARMUP_EPOCHS}") \
NCCL_PHASE6_COOLDOWN_EPOCHS=$(printf '%q' "${NCCL_PHASE6_COOLDOWN_EPOCHS}") \
NCCL_PHASE6_STABLE_EPOCHS=$(printf '%q' "${NCCL_PHASE6_STABLE_EPOCHS}") \
NCCL_PHASE6_THRESHOLD_HIGH_PCT=$(printf '%q' "${NCCL_PHASE6_THRESHOLD_HIGH_PCT}") \
NCCL_PHASE6_THRESHOLD_LOW_PCT=$(printf '%q' "${NCCL_PHASE6_THRESHOLD_LOW_PCT}") \
NCCL_PHASE6_THROUGHPUT_LOW_PCT=$(printf '%q' "${NCCL_PHASE6_THROUGHPUT_LOW_PCT}") \
NCCL_PHASE6_WSTALL_HIGH_PCT=$(printf '%q' "${NCCL_PHASE6_WSTALL_HIGH_PCT}") \
NCCL_PHASE6_INTEGRAL_LIMIT_PCT=$(printf '%q' "${NCCL_PHASE6_INTEGRAL_LIMIT_PCT}") \
NCCL_PHASE6_ALPHA_FAST_PCT=$(printf '%q' "${NCCL_PHASE6_ALPHA_FAST_PCT}") \
NCCL_PHASE6_ALPHA_SLOW_PCT=$(printf '%q' "${NCCL_PHASE6_ALPHA_SLOW_PCT}") \
NCCL_PHASE7_LOG=$(printf '%q' "${NCCL_PHASE7_LOG}") \
NCCL_PHASE7_BURST_FLOOR_POSTS=$(printf '%q' "${NCCL_PHASE7_BURST_FLOOR_POSTS}") \
NCCL_APPENDIX2_GROUP_LOG=$(printf '%q' "${NCCL_APPENDIX2_GROUP_LOG}") \
NCCL_DEBUG=$(printf '%q' "${NCCL_DEBUG}") \
NCCL_DEBUG_SUBSYS=$(printf '%q' "${NCCL_DEBUG_SUBSYS}") \
NCCL_NET_GDR_LEVEL=$(printf '%q' "${NCCL_NET_GDR_LEVEL}") \
HIDDEN_DIM=$(printf '%q' "${HIDDEN_DIM}") \
NUM_LAYERS=$(printf '%q' "${NUM_LAYERS}") \
BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") \
BUCKET_CAP_MB=$(printf '%q' "${BUCKET_CAP_MB}") \
LR=$(printf '%q' "${LR}") \
MODEL_SEED=$(printf '%q' "${MODEL_SEED}") \
bash $(printf '%q' "${WORKER_SCRIPT}")
EOF
)

  cat <<EOF
cd $(printf '%q' "${remote_repo_root}") && bash ./research/code/deploy/run_dev_container.sh bash -lc $(printf '%q' "${inner}")
EOF
}

declare -A LAUNCH_PIDS=()
declare -A LAUNCH_LOGS=()

launch_worker_mode() {
  local worker=$1
  local rank=$2
  local mode=$3
  local repeat_label=$4
  local phase4_enable_value=$5
  local phase4_post_receive_w_value=$6
  local net_burst_value=$7
  local phase6_enable_value=$8
  local phase6_post_rate_value=$9
  local phase6_post_burst_value=${10}
  local phase7_enable_value=${11}
  local phase7_ratio_pct_value=${12}
  local phase7_observe_ms_value=${13}
  local phase7_burst_window_ms_value=${14}
  local run_root=${15}
  local status_file=${16}
  local mode_upper_value
  mode_upper_value=$(echo "${mode}" | tr '[:lower:]' '[:upper:]')
  local worker_log_root="${run_root}/${worker}"
  local launcher_log="${worker_log_root}/launcher.log"
  mkdir -p "${worker_log_root}"
  write_pending_status "${status_file}" "${worker}" "${rank}" "${mode_upper_value}"
  local host_cmd
  host_cmd=$(build_worker_host_command "${worker}" "${rank}" "${mode}" "${repeat_label}" "${phase4_enable_value}" "${phase4_post_receive_w_value}" "${net_burst_value}" "${phase6_enable_value}" "${phase6_post_rate_value}" "${phase6_post_burst_value}" "${phase7_enable_value}" "${phase7_ratio_pct_value}" "${phase7_observe_ms_value}" "${phase7_burst_window_ms_value}" "${run_root}" "${status_file}")

  echo "[phase4-matrix] launch worker=${worker} rank=${rank} mode=${mode_upper_value}"
  if [[ "${worker}" == "${MASTER_SERVER}" ]]; then
    bash -lc "${host_cmd}" >"${launcher_log}" 2>&1 &
  else
    remote_worker_bash "${worker}" "${host_cmd}" >"${launcher_log}" 2>&1 &
  fi
  LAUNCH_PIDS["${worker}"]=$!
  LAUNCH_LOGS["${worker}"]="${launcher_log}"
}

status_snapshot() {
  local status_dir=$1
  local snapshot=""
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    local status_file="${status_dir}/${worker}.status"
    local status="missing"
    if [[ -f "${status_file}" ]]; then
      status=$(read_status_field "${status_file}" "status")
    fi
    snapshot+="${worker}:${status} "
  done
  echo "${snapshot% }"
}

wait_for_mode_completion() {
  local status_dir=$1
  local mode_upper_value=$2
  local start_ts=$3
  while true; do
    local success_count=0
    local running_count=0
    local failed_count=0
    local pending_count=0
    local worker
    for worker in "${ALL_WORKER_ARRAY[@]}"; do
      local status_file="${status_dir}/${worker}.status"
      local status="missing"
      if [[ -f "${status_file}" ]]; then
        status=$(read_status_field "${status_file}" "status")
      fi
      case "${status}" in
        0) success_count=$((success_count + 1)) ;;
        1) running_count=$((running_count + 1)) ;;
        -1) failed_count=$((failed_count + 1)) ;;
        *) pending_count=$((pending_count + 1)) ;;
      esac
    done

    local elapsed=$(( $(date +%s) - start_ts ))
    echo "[phase4-matrix] mode=${mode_upper_value} elapsed=${elapsed}s success=${success_count}/${NNODES} running=${running_count} pending=${pending_count} failed=${failed_count} statuses=$(status_snapshot "${status_dir}")"

    if (( failed_count > 0 )); then
      return 1
    fi
    if (( success_count == NNODES )); then
      return 0
    fi
    if (( elapsed >= MODE_TIMEOUT_SEC )); then
      echo "[phase4-matrix] ERROR: mode timeout reached mode=${mode_upper_value} timeout=${MODE_TIMEOUT_SEC}s" >&2
      return 2
    fi
    sleep 5
  done
}

wait_for_launchers() {
  local rc=0
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    local pid="${LAUNCH_PIDS[${worker}]:-}"
    if [[ -n "${pid}" ]]; then
      wait "${pid}" || rc=1
    fi
  done
  return "${rc}"
}

print_failed_worker_logs() {
  local status_dir=$1
  local worker
  for worker in "${ALL_WORKER_ARRAY[@]}"; do
    local status_file="${status_dir}/${worker}.status"
    if [[ -f "${status_file}" ]] && [[ "$(read_status_field "${status_file}" "status")" == "-1" ]]; then
      local launcher_log="${LAUNCH_LOGS[${worker}]:-}"
      echo "[phase4-matrix] failed worker=${worker} launcher_log=${launcher_log}"
      if [[ -n "${launcher_log}" && -f "${launcher_log}" ]]; then
        tail -n 80 "${launcher_log}" || true
      fi
    fi
  done
}

compare_outputs() {
  local experiment_root=$1
  local output_json=$2
  if [[ ! -f "${REPO_ROOT}/${COMPARE_SCRIPT}" ]]; then
    echo "[phase4-matrix] compare script missing path=${COMPARE_SCRIPT}"
    return 0
  fi
  if [[ ! -d "${experiment_root}/STOCK" ]]; then
    echo "[phase4-matrix] skip STOCK comparison: STOCK directory not present experiment_root=${experiment_root}"
    return 0
  fi
  echo "[phase4-matrix] validating final outputs against STOCK experiment_root=${experiment_root}"
  python3 "${REPO_ROOT}/${COMPARE_SCRIPT}" \
    --experiment-root "${experiment_root}" \
    --output-json "${output_json}" || true
}

summarize_nccl_events() {
  local experiment_root=$1
  local output_json=$2
  local summary_script="${REPO_ROOT}/${SUMMARY_SCRIPT}"
  if [[ ! -f "${summary_script}" ]]; then
    echo "[phase4-matrix] ncc event summary script missing path=${summary_script}"
    return 0
  fi
  echo "[phase4-matrix] summarizing NCCL recv/group events experiment_root=${experiment_root}"
  python3 "${summary_script}" \
    --experiment-root "${experiment_root}" \
    --output-json "${output_json}" || true
}

cleanup() {
  stop_switch_logger
}
trap cleanup EXIT INT TERM

MANIFEST_JSON="${LOG_ROOT}/${PHASE_NAME}_manifest.json"
cat > "${MANIFEST_JSON}" <<EOF
{
  "phase": "${PHASE_NAME}",
  "run_id": "${RUN_ID}",
  "master_server": "${MASTER_SERVER}",
  "master_addr": "${MASTER_ADDR}",
  "master_port": ${MASTER_PORT},
  "run_modes": "${RUN_MODES}",
  "repeats": ${REPEATS},
  "steps": ${STEPS},
  "warmup_steps": ${WARMUP_STEPS},
  "dtype": "${DTYPE}",
  "hidden_dim": ${HIDDEN_DIM},
  "num_layers": ${NUM_LAYERS},
  "batch_size": ${BATCH_SIZE},
  "bucket_cap_mb": ${BUCKET_CAP_MB},
  "lr": ${LR},
  "mode_timeout_sec": ${MODE_TIMEOUT_SEC},
  "worker_pool": "${ALL_WORKERS}",
  "execution_model": "master_orchestrated_single_port_ddp_tiny"
}
EOF

echo "[phase4-matrix] RUN_ID=${RUN_ID}"
echo "[phase4-matrix] MASTER_SERVER=${MASTER_SERVER} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[phase4-matrix] WORKERS=${ALL_WORKERS}"
echo "[phase4-matrix] MODES=${RUN_MODES}"
echo "[phase4-matrix] REPEATS=${REPEATS}"
echo "[phase4-matrix] MODEL hidden_dim=${HIDDEN_DIM} num_layers=${NUM_LAYERS} batch_size=${BATCH_SIZE} bucket_cap_mb=${BUCKET_CAP_MB} lr=${LR}"
echo "[phase4-matrix] LOG_ROOT=${LOG_ROOT}"
echo "[phase4-matrix] NET_BURST=${NET_BURST}"
echo "[phase4-matrix] MODE_DELAY_SEC=${MODE_DELAY_SEC}"

start_switch_logger

wait_for_master_port_free 5

for worker in "${ALL_WORKER_ARRAY[@]}"; do
  prepare_worker_container "${worker}"
done
cleanup_all_workers
wait_for_master_port_free 10

overall_rc=0
for repeat_idx in $(seq 1 "${REPEATS}"); do
  repeat_label=$(printf 'repeat_%02d' "${repeat_idx}")
  repeat_root="${LOG_ROOT}/${repeat_label}"
  repeat_status_root="${LOG_ROOT}/.ddp_status/${repeat_label}"
  mkdir -p "${repeat_root}" "${repeat_status_root}"

  echo "[phase4-matrix] ============================================================"
  echo "[phase4-matrix] start repeat=${repeat_label}"

  mode_index=0
  for mode in "${MODE_VALUES[@]}"; do
    if (( mode_index > 0 && MODE_DELAY_SEC > 0 )); then
      echo "[phase4-matrix] sleep before next mode repeat=${repeat_label} delay_sec=${MODE_DELAY_SEC}"
      sleep "${MODE_DELAY_SEC}"
    fi
    mode_upper_value=$(echo "${mode}" | tr '[:lower:]' '[:upper:]')
    phase4_enable_value=$(mode_field "${mode}" "phase4_enable" "0")
    phase4_post_receive_w_value=$(mode_field "${mode}" "post_receive_w" "0")
    net_burst_value=${NET_BURST}
    phase6_enable_value=$(mode_field "${mode}" "phase6_enable" "0")
    phase6_post_rate_value=$(mode_field "${mode}" "post_rate" "0")
    phase6_post_burst_value=$(mode_field "${mode}" "post_burst" "0")
    phase7_enable_value=$(mode_field "${mode}" "phase7_enable" "0")
    phase7_ratio_pct_value=$(mode_field "${mode}" "post_rate_ratio_pct" "0")
    phase7_observe_ms_value=$(mode_field "${mode}" "observe_ms" "10")
    phase7_burst_window_ms_value=$(mode_field "${mode}" "burst_window_ms" "2")
    run_root="${repeat_root}/${mode_upper_value}"
    status_dir="${repeat_status_root}/${mode_upper_value}"
    mkdir -p "${run_root}" "${status_dir}"

    echo "[phase4-matrix] ------------------------------------------------------------"
    echo "[phase4-matrix] start repeat=${repeat_label} mode=${mode_upper_value} port=${MASTER_PORT} phase4_enable=${phase4_enable_value} phase4_post_receive_w=${phase4_post_receive_w_value} net_burst=${net_burst_value} phase6_enable=${phase6_enable_value} phase6_post_rate=${phase6_post_rate_value} phase6_post_burst=${phase6_post_burst_value} phase7_enable=${phase7_enable_value} phase7_ratio_pct=${phase7_ratio_pct_value} phase7_observe_ms=${phase7_observe_ms_value} phase7_burst_window_ms=${phase7_burst_window_ms_value}"
    emit_switch_marker "mode_start" "run_id=${RUN_ID} repeat=${repeat_label} mode=${mode_upper_value} net_burst=${net_burst_value}"

    cleanup_all_workers
    wait_for_master_port_free 10

    unset LAUNCH_PIDS
    unset LAUNCH_LOGS
    declare -A LAUNCH_PIDS=()
    declare -A LAUNCH_LOGS=()

    launch_worker_mode "${ALL_WORKER_ARRAY[0]}" 0 "${mode}" "${repeat_label}" "${phase4_enable_value}" "${phase4_post_receive_w_value}" "${net_burst_value}" "${phase6_enable_value}" "${phase6_post_rate_value}" "${phase6_post_burst_value}" "${phase7_enable_value}" "${phase7_ratio_pct_value}" "${phase7_observe_ms_value}" "${phase7_burst_window_ms_value}" "${run_root}" "${status_dir}/${ALL_WORKER_ARRAY[0]}.status"

    rank0_pid="${LAUNCH_PIDS[${ALL_WORKER_ARRAY[0]}]}"
    if ! wait_for_master_port_listen "${rank0_pid}"; then
      overall_rc=1
      print_failed_worker_logs "${status_dir}"
      break 2
    fi

    for idx in "${!ALL_WORKER_ARRAY[@]}"; do
      if (( idx == 0 )); then
        continue
      fi
      worker="${ALL_WORKER_ARRAY[$idx]}"
      launch_worker_mode "${worker}" "${idx}" "${mode}" "${repeat_label}" "${phase4_enable_value}" "${phase4_post_receive_w_value}" "${net_burst_value}" "${phase6_enable_value}" "${phase6_post_rate_value}" "${phase6_post_burst_value}" "${phase7_enable_value}" "${phase7_ratio_pct_value}" "${phase7_observe_ms_value}" "${phase7_burst_window_ms_value}" "${run_root}" "${status_dir}/${worker}.status"
    done

    mode_start_ts=$(date +%s)
    if ! wait_for_mode_completion "${status_dir}" "${mode_upper_value}" "${mode_start_ts}"; then
      overall_rc=1
      print_failed_worker_logs "${status_dir}"
      wait_for_launchers || true
      cleanup_all_workers
      wait_for_master_port_free 15 || true
      break 2
    fi

    wait_for_launchers || overall_rc=1
    cleanup_all_workers
    if ! wait_for_master_port_free 15; then
      overall_rc=1
      break 2
    fi

    echo "[phase4-matrix] complete repeat=${repeat_label} mode=${mode_upper_value}"
    emit_switch_marker "mode_end" "run_id=${RUN_ID} repeat=${repeat_label} mode=${mode_upper_value} net_burst=${net_burst_value}"
    mode_index=$((mode_index + 1))
  done

  compare_outputs "${repeat_root}" "${repeat_root}/final_output_validation.json"
  summarize_nccl_events "${repeat_root}" "${repeat_root}/${SUMMARY_OUTPUT_NAME}"
done

if (( overall_rc != 0 )); then
  echo "[phase4-matrix] experiment failed" >&2
  exit "${overall_rc}"
fi

echo "[phase4-matrix] experiment complete"
