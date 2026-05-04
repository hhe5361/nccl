#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

CONFIG_PATH=${SCRIPT_DIR}/phase4_b4_post_receive_ddp.json
if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH=${2:?config path required}
elif [[ -n "${1:-}" ]]; then
  CONFIG_PATH=${1}
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[phase4-matrix] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

eval "$(
python3 - "${CONFIG_PATH}" <<'PY'
import json, shlex, sys

cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
run = cfg.get("run", {})
model = cfg.get("model", {})
container = cfg.get("container", {})
ssh = cfg.get("ssh", {})
paths = cfg.get("paths", {})
nccl = cfg.get("nccl", {})
modes = [m.get("mode_id") for m in cfg.get("mode_matrix", []) if m.get("mode_id")]
workers = run.get("all_workers", [])

pairs = {
    "CFG_RUN_ID_PREFIX": run.get("run_id_prefix", "phase4_b4_ddp"),
    "CFG_MASTER_SERVER": run.get("master_server", "worker01"),
    "CFG_MASTER_ADDR": run.get("master_addr", "172.16.0.101"),
    "CFG_MASTER_PORT": run.get("master_port", 40100),
    "CFG_LOG_ROOT_BASE": paths.get("log_root_base", "/mnt/nfs_share/cts_experiments"),
    "CFG_ALL_WORKERS": ",".join(workers),
    "CFG_RUN_MODES": ",".join(modes),
    "CFG_STEPS": run.get("steps", 40),
    "CFG_WARMUP_STEPS": run.get("warmup_steps", 5),
    "CFG_MODE_TIMEOUT_SEC": run.get("mode_timeout_sec", 600),
    "CFG_TORCH_ENV": run.get("torch_env", "/workspace/venvs/torch-cu121-custom/bin/activate"),
    "CFG_TARGET_SCRIPT": run.get("target_script", "research/code/phase4/ddp_b4.py"),
    "CFG_WORKER_SCRIPT": run.get("worker_script", "research/code/phase4/run_b4_mode_worker.sh"),
    "CFG_COMPARE_SCRIPT": run.get("compare_script", "research/code/phase4/compare_b4_vs_stock.py"),
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
    "CFG_NCCL_ALGO": nccl.get("algo", "auto"),
    "CFG_NCCL_PROTO": nccl.get("proto", "auto"),
    "CFG_NCCL_PHASE0_LOG": nccl.get("phase0_log", 1),
    "CFG_NCCL_PHASE4_LOG": nccl.get("phase4_log", 0),
    "CFG_NCCL_DEBUG": nccl.get("debug", "INFO"),
    "CFG_NCCL_DEBUG_SUBSYS": nccl.get("debug_subsys", "NET"),
    "CFG_NCCL_NET_GDR_LEVEL": nccl.get("net_gdr_level", 0),
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
STEPS=${STEPS:-${CFG_STEPS}}
WARMUP_STEPS=${WARMUP_STEPS:-${CFG_WARMUP_STEPS}}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-${CFG_MODE_TIMEOUT_SEC}}
TORCH_ENV=${TORCH_ENV:-${CFG_TORCH_ENV}}
TARGET_SCRIPT=${TARGET_SCRIPT:-${CFG_TARGET_SCRIPT}}
WORKER_SCRIPT=${WORKER_SCRIPT:-${CFG_WORKER_SCRIPT}}
COMPARE_SCRIPT=${COMPARE_SCRIPT:-${CFG_COMPARE_SCRIPT}}
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
NCCL_ALGO=${NCCL_ALGO:-${CFG_NCCL_ALGO}}
NCCL_PROTO=${NCCL_PROTO:-${CFG_NCCL_PROTO}}
NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-${CFG_NCCL_PHASE0_LOG}}
NCCL_PHASE4_LOG=${NCCL_PHASE4_LOG:-${CFG_NCCL_PHASE4_LOG}}
NCCL_DEBUG=${NCCL_DEBUG:-${CFG_NCCL_DEBUG}}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-${CFG_NCCL_DEBUG_SUBSYS}}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-${CFG_NCCL_NET_GDR_LEVEL}}

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
  local run_root=$4
  local status_file=$5
  local remote_repo_root

  remote_repo_root=$(resolve_remote_repo_root "${worker}")
  local inner
  inner=$(cat <<EOF
cd $(printf '%q' "${CONTAINER_REPO_ROOT}") && \
RUN_ID=$(printf '%q' "${RUN_ID}") \
EXPERIMENT_LABEL=$(printf '%q' "phase4_ddp_tiny") \
MODE=$(printf '%q' "${mode}") \
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
NCCL_DEBUG=$(printf '%q' "${NCCL_DEBUG}") \
NCCL_DEBUG_SUBSYS=$(printf '%q' "${NCCL_DEBUG_SUBSYS}") \
NCCL_NET_GDR_LEVEL=$(printf '%q' "${NCCL_NET_GDR_LEVEL}") \
HIDDEN_DIM=$(printf '%q' "${HIDDEN_DIM}") \
NUM_LAYERS=$(printf '%q' "${NUM_LAYERS}") \
BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") \
BUCKET_CAP_MB=$(printf '%q' "${BUCKET_CAP_MB}") \
LR=$(printf '%q' "${LR}") \
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
  local run_root=$4
  local status_file=$5
  local mode_upper_value
  mode_upper_value=$(echo "${mode}" | tr '[:lower:]' '[:upper:]')
  local worker_log_root="${run_root}/${worker}"
  local launcher_log="${worker_log_root}/launcher.log"
  mkdir -p "${worker_log_root}"
  write_pending_status "${status_file}" "${worker}" "${rank}" "${mode_upper_value}"
  local host_cmd
  host_cmd=$(build_worker_host_command "${worker}" "${rank}" "${mode}" "${run_root}" "${status_file}")

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

VALIDATION_JSON="${LOG_ROOT}/final_output_validation.json"

compare_outputs() {
  if [[ ! -f "${REPO_ROOT}/${COMPARE_SCRIPT}" ]]; then
    echo "[phase4-matrix] compare script missing path=${COMPARE_SCRIPT}"
    return 0
  fi
  echo "[phase4-matrix] validating final outputs against STOCK"
  python3 "${REPO_ROOT}/${COMPARE_SCRIPT}" \
    --experiment-root "${LOG_ROOT}" \
    --output-json "${VALIDATION_JSON}" || true
}

MANIFEST_JSON="${LOG_ROOT}/phase4_manifest.json"
cat > "${MANIFEST_JSON}" <<EOF
{
  "phase": "phase4",
  "run_id": "${RUN_ID}",
  "master_server": "${MASTER_SERVER}",
  "master_addr": "${MASTER_ADDR}",
  "master_port": ${MASTER_PORT},
  "run_modes": "${RUN_MODES}",
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
echo "[phase4-matrix] MODEL hidden_dim=${HIDDEN_DIM} num_layers=${NUM_LAYERS} batch_size=${BATCH_SIZE} bucket_cap_mb=${BUCKET_CAP_MB} lr=${LR}"
echo "[phase4-matrix] LOG_ROOT=${LOG_ROOT}"

wait_for_master_port_free 5

for worker in "${ALL_WORKER_ARRAY[@]}"; do
  prepare_worker_container "${worker}"
done
cleanup_all_workers
wait_for_master_port_free 10

overall_rc=0
for mode in "${MODE_VALUES[@]}"; do
  mode_upper_value=$(echo "${mode}" | tr '[:lower:]' '[:upper:]')
  run_root="${LOG_ROOT}/${mode_upper_value}"
  status_dir="${LOG_ROOT}/.ddp_status/${mode_upper_value}"
  mkdir -p "${run_root}" "${status_dir}"

  echo "[phase4-matrix] ------------------------------------------------------------"
  echo "[phase4-matrix] start mode=${mode_upper_value} port=${MASTER_PORT}"

  cleanup_all_workers
  wait_for_master_port_free 10

  unset LAUNCH_PIDS
  unset LAUNCH_LOGS
  declare -A LAUNCH_PIDS=()
  declare -A LAUNCH_LOGS=()

  launch_worker_mode "${ALL_WORKER_ARRAY[0]}" 0 "${mode}" "${run_root}" "${status_dir}/${ALL_WORKER_ARRAY[0]}.status"

  rank0_pid="${LAUNCH_PIDS[${ALL_WORKER_ARRAY[0]}]}"
  if ! wait_for_master_port_listen "${rank0_pid}"; then
    overall_rc=1
    print_failed_worker_logs "${status_dir}"
    break
  fi

  for idx in "${!ALL_WORKER_ARRAY[@]}"; do
    if (( idx == 0 )); then
      continue
    fi
    worker="${ALL_WORKER_ARRAY[$idx]}"
    launch_worker_mode "${worker}" "${idx}" "${mode}" "${run_root}" "${status_dir}/${worker}.status"
  done

  mode_start_ts=$(date +%s)
  if ! wait_for_mode_completion "${status_dir}" "${mode_upper_value}" "${mode_start_ts}"; then
    overall_rc=1
    print_failed_worker_logs "${status_dir}"
    wait_for_launchers || true
    cleanup_all_workers
    wait_for_master_port_free 15 || true
    break
  fi

  wait_for_launchers || overall_rc=1
  cleanup_all_workers
  if ! wait_for_master_port_free 15; then
    overall_rc=1
    break
  fi

  echo "[phase4-matrix] complete mode=${mode_upper_value}"
done

compare_outputs

if (( overall_rc != 0 )); then
  echo "[phase4-matrix] experiment failed" >&2
  exit "${overall_rc}"
fi

echo "[phase4-matrix] experiment complete"
