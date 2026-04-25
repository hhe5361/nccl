#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

infer_node_rank() {
  local worker_name=$1
  if [[ "${worker_name}" =~ ^worker([0-9]+)$ ]]; then
    local idx=${BASH_REMATCH[1]}
    if (( 1 <= 10#${idx} && 10#${idx} <= 8 )); then
      echo $((10#${idx} - 1))
      return 0
    fi
  fi
  echo "cannot infer NODE_RANK from WORKER_NAME='${worker_name}'" >&2
  return 1
}

RUN_ID=${RUN_ID:-phase2_b2_collective_$(date +%y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-31500}
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
NODE_RANK=${NODE_RANK:-$(infer_node_rank "${WORKER_NAME}")}
MASTER_SERVER=${MASTER_SERVER:-worker01}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase2/collective_b2.py}
LOG_ROOT=${LOG_ROOT:-/mnt/nfs_share/cts_experiments/${RUN_ID}}
RUN_MODES=${RUN_MODES:-stock,b2_w2,b2_w4,b2_w5,b2_w6,b2_w7,b2_w8}
COLLECTIVE=${COLLECTIVE:-allreduce}
EXPERIMENT_LABEL=${EXPERIMENT_LABEL:-}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
PAYLOAD_MB=${PAYLOAD_MB:-128}
DTYPE=${DTYPE:-float32}
SLEEP_MS=${SLEEP_MS:-0}
ALGO_SETTING=${NCCL_ALGO:-auto}
PROTO_SETTING=${NCCL_PROTO:-auto}

SWITCH_LOG_ENABLE=${SWITCH_LOG_ENABLE:-0}
SWITCH_METADATA_FILE=${SWITCH_METADATA_FILE:-}
DPU_NODE_HOST=${DPU_NODE_HOST:-172.16.0.100}
DPU_NODE_USER=${DPU_NODE_USER:-ubuntu}
SWITCH_LOGGER_ROOT=${SWITCH_LOGGER_ROOT:-/home/ubuntu/hyoeun/switch_setup_task/switch_congestion_logger}
SWITCH_LOG_SHARED_ROOT=${SWITCH_LOG_SHARED_ROOT:-/mnt/nfs/cts_experiments/switch_log}

mkdir -p "${LOG_ROOT}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_PRELOAD="/workspace/nccl/build/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-1}
export NCCL_PHASE1_STATIC_W=0
export NCCL_PHASE2_B2_ENABLE=0
export NCCL_PHASE2_LOG=${NCCL_PHASE2_LOG:-0}
export NCCL_PHASE3_B3_ENABLE=0
export NCCL_PHASE3_LOG=${NCCL_PHASE3_LOG:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ "${ALGO_SETTING}" == "auto" ]]; then
  unset NCCL_ALGO
else
  export NCCL_ALGO="${ALGO_SETTING}"
fi

if [[ "${PROTO_SETTING}" == "auto" ]]; then
  unset NCCL_PROTO
else
  export NCCL_PROTO="${PROTO_SETTING}"
fi

if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun)
elif python - <<'PYTORCHCHECK' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('torch.distributed.run') else 1)
PYTORCHCHECK
then
  LAUNCHER=(python -m torch.distributed.run)
else
  echo "[phase2] PyTorch launcher not found in current environment." >&2
  exit 1
fi

is_master_server() {
  [[ "${WORKER_NAME}" == "${MASTER_SERVER}" || "$(hostname -f 2>/dev/null || true)" == "${MASTER_SERVER}" || "$(hostname -s 2>/dev/null || true)" == "${MASTER_SERVER}" ]]
}

require_sshpass() {
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "[phase2] sshpass is required for switch marker integration." >&2
    exit 1
  fi
}

remote_dpu_bash() {
  local cmd=$1
  require_sshpass
  if [[ -z "${DPU_NODE_PWD:-}" ]]; then
    echo "[phase2] DPU_NODE_PWD must be set when SWITCH_LOG_ENABLE=1 on MASTER_SERVER." >&2
    exit 1
  fi
  sshpass -p "${DPU_NODE_PWD}" \
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${DPU_NODE_USER}@${DPU_NODE_HOST}" \
    "bash -lc $(printf '%q' "${cmd}")"
}

SWITCH_LOG_RUN_ID=
SWITCH_LOG_DIR=
SWITCH_LOG_LOCAL_DIR=
SWITCH_LOG_MARKERS_JSONL=
if [[ -n "${SWITCH_METADATA_FILE}" && -f "${SWITCH_METADATA_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SWITCH_METADATA_FILE}"
fi
if [[ -z "${SWITCH_LOG_LOCAL_DIR:-}" && -n "${SWITCH_LOG_RUN_ID:-}" ]]; then
  SWITCH_LOG_LOCAL_DIR="${SWITCH_LOG_SHARED_ROOT}/${SWITCH_LOG_RUN_ID}"
fi

emit_switch_marker() {
  local marker=$1
  local message=${2:-}
  local source_tag=${3:-phase2_collective}
  [[ "${SWITCH_LOG_ENABLE}" == "1" ]] || return 0
  is_master_server || return 0
  [[ -n "${SWITCH_LOG_RUN_ID:-}" ]] || return 0
  local cmd
  cmd="cd $(printf '%q' "${SWITCH_LOGGER_ROOT}") && ./log_run_marker.sh --run-id $(printf '%q' "${SWITCH_LOG_RUN_ID}") --marker $(printf '%q' "${marker}") --source $(printf '%q' "${source_tag}")"
  if [[ -n "${message}" ]]; then
    cmd+=" --message $(printf '%q' "${message}")"
  fi
  remote_dpu_bash "${cmd}" >/dev/null
}

case "${COLLECTIVE}" in
  allreduce|allgather|reducescatter|alltoall) ;;
  *)
    echo "[phase2] unsupported COLLECTIVE=${COLLECTIVE}" >&2
    exit 1
    ;;
esac

IFS=',' read -r -a MODE_VALUES <<< "${RUN_MODES}"

SWITCH_ENV_JSON=""
if [[ "${SWITCH_LOG_ENABLE}" == "1" && -n "${SWITCH_LOG_RUN_ID:-}" ]]; then
  SWITCH_ENV_JSON=$(cat <<EOF
,
  "switch_log_enable": 1,
  "switch_log_run_id": "${SWITCH_LOG_RUN_ID}",
  "switch_log_dir": "${SWITCH_LOG_DIR:-}",
  "switch_log_local_dir": "${SWITCH_LOG_LOCAL_DIR:-}",
  "switch_log_markers_jsonl": "${SWITCH_LOG_MARKERS_JSONL:-}",
  "dpu_node_host": "${DPU_NODE_HOST}",
  "master_server": "${MASTER_SERVER}"
EOF
)
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  cat > "${LOG_ROOT}/env_setup.json" <<EOF
{
  "phase": "B2",
  "run_id": "${RUN_ID}",
  "experiment_label": "${EXPERIMENT_LABEL}",
  "collective": "${COLLECTIVE}",
  "run_modes": "${RUN_MODES}",
  "master_addr": "${MASTER_ADDR}",
  "master_port_base": ${MASTER_PORT_BASE},
  "nnodes": ${NNODES},
  "nproc_per_node": ${NPROC_PER_NODE},
  "steps": ${STEPS},
  "warmup_steps": ${WARMUP_STEPS},
  "payload_mb": ${PAYLOAD_MB},
  "dtype": "${DTYPE}",
  "nccl_algo": "${ALGO_SETTING}",
  "nccl_proto": "${PROTO_SETTING}",
  "policy_name": "static_receiver_window_sweep",
  "policy_formula": "W_eff = min(W_base, W_cfg) with PHASE1_STATIC_W override",
  "phase2_b2_enable": 0,
  "phase3_b3_enable": 0,
  "b3_disabled_by_runner": true,
  "worker_pool": "worker01,worker03,worker05,worker07"${SWITCH_ENV_JSON}
}
EOF
fi

echo "[phase2] RUN_ID=${RUN_ID}"
echo "[phase2] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES}"
echo "[phase2] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT_BASE=${MASTER_PORT_BASE}"
echo "[phase2] MASTER_SERVER=${MASTER_SERVER}"
echo "[phase2] LOG_ROOT=${LOG_ROOT}"
echo "[phase2] TARGET_SCRIPT=${TARGET_SCRIPT}"
echo "[phase2] COLLECTIVE=${COLLECTIVE} RUN_MODES=${RUN_MODES}"
echo "[phase2] PAYLOAD_MB=${PAYLOAD_MB} DTYPE=${DTYPE} NCCL_ALGO=${ALGO_SETTING} NCCL_PROTO=${PROTO_SETTING}"
if [[ "${SWITCH_LOG_ENABLE}" == "1" && -n "${SWITCH_LOG_RUN_ID:-}" ]]; then
  echo "[phase2] SWITCH_LOG_RUN_ID=${SWITCH_LOG_RUN_ID} SWITCH_LOG_DIR=${SWITCH_LOG_DIR:-} SWITCH_LOG_LOCAL_DIR=${SWITCH_LOG_LOCAL_DIR:-}"
fi

for idx in "${!MODE_VALUES[@]}"; do
  MODE="${MODE_VALUES[$idx]}"
  MODE_UPPER=$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')
  RUN_ROOT="${LOG_ROOT}/${MODE_UPPER}"
  WORKER_LOG_ROOT="${RUN_ROOT}/${WORKER_NAME}"
  MASTER_PORT=$((MASTER_PORT_BASE + idx))

  mkdir -p "${RUN_ROOT}" "${WORKER_LOG_ROOT}"

  case "${MODE}" in
    stock)
      export NCCL_PHASE1_STATIC_W=0
      ;;
    b2_w*)
      export NCCL_PHASE1_STATIC_W="${MODE#b2_w}"
      ;;
    *)
      echo "[phase2] unsupported RUN_MODES entry=${MODE}" >&2
      exit 1
      ;;
  esac

  export PHASE2_MODE="${MODE}"
  export PHASE2_OUTPUT_DIR="${RUN_ROOT}"
  export NCCL_DEBUG_FILE="${WORKER_LOG_ROOT}/nccl.%h.%p.log"

  echo "[phase2] starting MODE=${MODE} MASTER_PORT=${MASTER_PORT} STATIC_W=${NCCL_PHASE1_STATIC_W}"
  echo "[phase2] NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
  emit_switch_marker "mode_start" "run_id=${RUN_ID} mode=${MODE_UPPER} collective=${COLLECTIVE} algo=${ALGO_SETTING} payload_mb=${PAYLOAD_MB} static_w=${NCCL_PHASE1_STATIC_W}" "phase2_collective"

  "${LAUNCHER[@]}" \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/${TARGET_SCRIPT}" \
    --collective "${COLLECTIVE}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --payload-mb "${PAYLOAD_MB}" \
    --dtype "${DTYPE}" \
    --sleep-ms "${SLEEP_MS}" \
    --output-dir "${RUN_ROOT}" \
    --tag "${MODE_UPPER}" \
    "$@"

  emit_switch_marker "mode_end" "run_id=${RUN_ID} mode=${MODE_UPPER} collective=${COLLECTIVE} algo=${ALGO_SETTING} payload_mb=${PAYLOAD_MB} static_w=${NCCL_PHASE1_STATIC_W}" "phase2_collective"
  sleep 2
done
