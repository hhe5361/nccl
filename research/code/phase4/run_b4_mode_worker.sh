#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

RUN_ID=${RUN_ID:?RUN_ID is required}
EXPERIMENT_LABEL=${EXPERIMENT_LABEL:-${RUN_ID}}
MODE=${MODE:?MODE is required}
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR is required}
MASTER_PORT=${MASTER_PORT:?MASTER_PORT is required}
NNODES=${NNODES:?NNODES is required}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:?NODE_RANK is required}
WORKER_NAME=${WORKER_NAME:-$(hostname -s)}
TORCH_ENV=${TORCH_ENV:-/workspace/venvs/torch-cu121-custom/bin/activate}
TARGET_SCRIPT=${TARGET_SCRIPT:-research/code/phase4/ddp_b4.py}
LOG_ROOT=${LOG_ROOT:?LOG_ROOT is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
STATUS_FILE=${STATUS_FILE:?STATUS_FILE is required}
STATUS_DIR=${STATUS_DIR:-$(dirname "${STATUS_FILE}")}
STEPS=${STEPS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-5}
DTYPE=${DTYPE:-float32}
MODE_TIMEOUT_SEC=${MODE_TIMEOUT_SEC:-600}
ALGO_SETTING=${NCCL_ALGO:-auto}
PROTO_SETTING=${NCCL_PROTO:-auto}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_LAYERS=${NUM_LAYERS:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
BUCKET_CAP_MB=${BUCKET_CAP_MB:-1}
LR=${LR:-0.01}
MODEL_SEED=${MODEL_SEED:-20260504}
REPEAT_LABEL=${REPEAT_LABEL:-repeat_01}
PHASE4_ENABLE_VALUE=${PHASE4_ENABLE_VALUE:-}
PHASE4_POST_RECEIVE_W_VALUE=${PHASE4_POST_RECEIVE_W_VALUE:-}
PHASE6_ENABLE_VALUE=${PHASE6_ENABLE_VALUE:-0}
PHASE6_POST_RATE_VALUE=${PHASE6_POST_RATE_VALUE:-0}
PHASE6_POST_BURST_VALUE=${PHASE6_POST_BURST_VALUE:-0}
NET_BURST_VALUE=${NET_BURST_VALUE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
PHASE4_NETWORK_TOPOLOGY_FILE=${PHASE4_NETWORK_TOPOLOGY_FILE:-${REPO_ROOT}/research/env/network_topology_internal_ips.txt}
PHASE4_LOAD_DURATION_SEC=${PHASE4_LOAD_DURATION_SEC:-240}
PHASE4_LOAD_LEADIN_SEC=${PHASE4_LOAD_LEADIN_SEC:-5}
PHASE4_IB_DEVICE=${PHASE4_IB_DEVICE:-mlx5_0}
PHASE4_IB_GID_INDEX=${PHASE4_IB_GID_INDEX:-3}
PHASE4_PAIR_PORT_BASE=${PHASE4_PAIR_PORT_BASE:-18600}
PHASE4_FULL_PAIR_GBPS=${PHASE4_FULL_PAIR_GBPS:-26.0}

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/${WORKER_NAME}" "${STATUS_DIR}"

# shellcheck disable=SC1090
source "${TORCH_ENV}"

export LD_PRELOAD="/workspace/nccl/build/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export NCCL_PHASE0_LOG=${NCCL_PHASE0_LOG:-1}
export NCCL_PHASE2_B2_ENABLE=0
export NCCL_PHASE2_LOG=${NCCL_PHASE2_LOG:-0}
export NCCL_PHASE3_B3_ENABLE=0
export NCCL_PHASE3_LOG=${NCCL_PHASE3_LOG:-0}
export NCCL_PHASE4_LOG=${NCCL_PHASE4_LOG:-0}
export NCCL_PHASE5_LOG=${NCCL_PHASE5_LOG:-1}
export NCCL_PHASE6_LOG=${NCCL_PHASE6_LOG:-0}
export NCCL_APPENDIX2_GROUP_LOG=${NCCL_APPENDIX2_GROUP_LOG:-0}
export NCCL_APPENDIX2_DISABLE_WSTALL_LOG=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_PHASE1_STATIC_W=0

case "${MODE}" in
  stock)
    export NCCL_PHASE4_ENABLE=${PHASE4_ENABLE_VALUE:-0}
    export NCCL_PHASE4_POST_RECEIVE_W=${PHASE4_POST_RECEIVE_W_VALUE:-0}
    ;;
  stock_2)
    export NCCL_PHASE4_ENABLE=${PHASE4_ENABLE_VALUE:-0}
    export NCCL_PHASE4_POST_RECEIVE_W=${PHASE4_POST_RECEIVE_W_VALUE:-0}
    ;;
  b4_w*)
    export NCCL_PHASE4_ENABLE=${PHASE4_ENABLE_VALUE:-1}
    export NCCL_PHASE4_POST_RECEIVE_W=${PHASE4_POST_RECEIVE_W_VALUE:-${MODE#b4_w}}
    ;;
  p6*)
    export NCCL_PHASE4_ENABLE=${PHASE4_ENABLE_VALUE:-0}
    export NCCL_PHASE4_POST_RECEIVE_W=${PHASE4_POST_RECEIVE_W_VALUE:-0}
    ;;
  *)
    echo "[phase4-worker] unsupported MODE=${MODE}" >&2
    exit 1
    ;;
esac

export NCCL_PHASE6_ENABLE=${PHASE6_ENABLE_VALUE:-0}
export NCCL_PHASE6_POST_RATE=${PHASE6_POST_RATE_VALUE:-0}
export NCCL_PHASE6_POST_BURST=${PHASE6_POST_BURST_VALUE:-0}

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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[phase4-worker] Python launcher not found: ${PYTHON_BIN}" >&2
  exit 1
fi

MODE_UPPER=$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')
export PHASE4_MODE="${MODE}"
export PHASE4_OUTPUT_DIR="${RUN_ROOT}"
export PHASE4_REPEAT_LABEL="${REPEAT_LABEL}"
export NCCL_DEBUG_FILE="${RUN_ROOT}/${WORKER_NAME}/nccl.%h.%p.log"
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE="${NNODES}"
export RANK="${NODE_RANK}"
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export GROUP_RANK="${NODE_RANK}"

STATUS_DDP=-1
RUN_RC=1
RUN_STATE=failed
STATUS_MESSAGE=init
STARTED_AT=$(date +%s)
LOAD_PIDS=()

write_status() {
  cat > "${STATUS_FILE}" <<EOF
status=${STATUS_DDP}
state=${RUN_STATE}
worker=${WORKER_NAME}
node_rank=${NODE_RANK}
mode=${MODE_UPPER}
experiment=${EXPERIMENT_LABEL}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
rc=${RUN_RC}
pid=$$
updated_at=$(date +%s)
started_at=${STARTED_AT}
message=${STATUS_MESSAGE}
EOF
}

mark_running() {
  export STATUS_DDP=1
  RUN_STATE=running
  RUN_RC=
  STATUS_MESSAGE=running
  write_status
}

mark_success() {
  export STATUS_DDP=0
  RUN_STATE=success
  RUN_RC=0
  STATUS_MESSAGE=success
  write_status
}

mark_failure() {
  local rc=${1:-1}
  local message=${2:-failed}
  export STATUS_DDP=-1
  RUN_STATE=failed
  RUN_RC=${rc}
  STATUS_MESSAGE=${message}
  write_status
}

cleanup() {
  local pid
  for pid in "${LOAD_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  if [[ "${RUN_STATE}" != "success" ]]; then
    mark_failure "${RUN_RC:-1}" "${STATUS_MESSAGE:-failed}"
  fi
}
trap cleanup EXIT INT TERM

resolve_worker_ip() {
  local worker=$1
  awk -v target="${worker}" '
    /^\[Workers\]/ { in_workers=1; next }
    /^\[/ && $0 !~ /^\[Workers\]/ { in_workers=0 }
    in_workers && $1 == "-" {
      gsub(":", "", $2)
      if ($2 == target) {
        print $3
        exit
      }
    }
  ' "${PHASE4_NETWORK_TOPOLOGY_FILE}"
}

pair_worker_names() {
  local pair_index=$1
  case "${pair_index}" in
    0) echo "worker01 worker02" ;;
    1) echo "worker03 worker04" ;;
    2) echo "worker05 worker06" ;;
    3) echo "worker07 worker08" ;;
    *) return 1 ;;
  esac
}

launch_phase4_load() {
  local load_pct=$1
  if (( load_pct <= 0 )); then
    return 0
  fi
  if ! command -v ib_write_bw >/dev/null 2>&1; then
    echo "[phase4-worker] ib_write_bw not found; skip background load" >&2
    return 0
  fi
  if [[ ! -f "${PHASE4_NETWORK_TOPOLOGY_FILE}" ]]; then
    echo "[phase4-worker] topology file not found: ${PHASE4_NETWORK_TOPOLOGY_FILE}" >&2
    return 0
  fi

  local worker_num pair_index pair_role active_pairs
  worker_num=${WORKER_NAME#worker}
  worker_num=$((10#${worker_num}))
  pair_index=$(((worker_num - 1) / 2))
  if (( worker_num % 2 == 1 )); then
    pair_role=client
  else
    pair_role=server
  fi

  active_pairs=$(((load_pct + 24) / 25))
  (( active_pairs < 1 )) && active_pairs=1
  (( active_pairs > 4 )) && active_pairs=4
  if (( pair_index >= active_pairs )); then
    return 0
  fi

  read -r src_worker dst_worker <<< "$(pair_worker_names "${pair_index}")"
  local src_ip dst_ip port pair_rate_gbps load_dir
  src_ip=$(resolve_worker_ip "${src_worker}")
  dst_ip=$(resolve_worker_ip "${dst_worker}")
  port=$((PHASE4_PAIR_PORT_BASE + pair_index))
  pair_rate_gbps=$(python3 - <<PY
full_pair = float(${PHASE4_FULL_PAIR_GBPS})
load_pct = float(${load_pct})
active_pairs = float(${active_pairs})
target_total = full_pair * 4.0 * load_pct / 100.0
pair_rate = target_total / active_pairs if active_pairs > 0 else 0.0
pair_rate = min(pair_rate, full_pair)
print(f"{pair_rate:.3f}")
PY
)
  load_dir="${RUN_ROOT}/${WORKER_NAME}/phase4_load"
  mkdir -p "${load_dir}"

  if [[ "${pair_role}" == "server" ]]; then
    timeout --signal=TERM --kill-after=5 "${PHASE4_LOAD_DURATION_SEC}" \
      ib_write_bw -R -d "${PHASE4_IB_DEVICE}" -x "${PHASE4_IB_GID_INDEX}" -F \
      -q 1 -p "${port}" -D "${PHASE4_LOAD_DURATION_SEC}" --report_gbits \
      > "${load_dir}/server.log" 2>&1 &
    LOAD_PIDS+=($!)
  else
    (
      sleep 3
      timeout --signal=TERM --kill-after=5 "${PHASE4_LOAD_DURATION_SEC}" \
        ib_write_bw "${dst_ip}" -R -d "${PHASE4_IB_DEVICE}" -x "${PHASE4_IB_GID_INDEX}" -F \
        -q 1 -p "${port}" -D "${PHASE4_LOAD_DURATION_SEC}" --report_gbits \
        --rate_limit "${pair_rate_gbps}" --rate_units=gbps \
        > "${load_dir}/client.log" 2>&1
    ) &
    LOAD_PIDS+=($!)
  fi
}

echo "[phase4-worker] RUN_ID=${RUN_ID} EXPERIMENT=${EXPERIMENT_LABEL}"
echo "[phase4-worker] REPEAT_LABEL=${REPEAT_LABEL}"
echo "[phase4-worker] WORKER_NAME=${WORKER_NAME} NODE_RANK=${NODE_RANK}/${NNODES} MODE=${MODE_UPPER}"
echo "[phase4-worker] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[phase4-worker] PHASE4_ENABLE=${NCCL_PHASE4_ENABLE} PHASE4_POST_RECEIVE_W=${NCCL_PHASE4_POST_RECEIVE_W}"
echo "[phase4-worker] PHASE5_LOG=${NCCL_PHASE5_LOG}"
echo "[phase4-worker] PHASE6_ENABLE=${NCCL_PHASE6_ENABLE} PHASE6_LOG=${NCCL_PHASE6_LOG} PHASE6_POST_RATE=${NCCL_PHASE6_POST_RATE} PHASE6_POST_BURST=${NCCL_PHASE6_POST_BURST}"
echo "[phase4-worker] APPENDIX2_GROUP_LOG=${NCCL_APPENDIX2_GROUP_LOG} APPENDIX2_DISABLE_WSTALL_LOG=${NCCL_APPENDIX2_DISABLE_WSTALL_LOG}"
echo "[phase4-worker] HIDDEN_DIM=${HIDDEN_DIM} NUM_LAYERS=${NUM_LAYERS} BATCH_SIZE=${BATCH_SIZE} BUCKET_CAP_MB=${BUCKET_CAP_MB} LR=${LR}"
echo "[phase4-worker] MODEL_SEED=${MODEL_SEED}"
echo "[phase4-worker] NET_BURST=${NET_BURST_VALUE}"
echo "[phase4-worker] PYTHON_BIN=${PYTHON_BIN} RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} LOCAL_RANK=${LOCAL_RANK}"
echo "[phase4-worker] NCCL_ALGO=${ALGO_SETTING} NCCL_PROTO=${PROTO_SETTING} STATUS_FILE=${STATUS_FILE}"
echo "[phase4-worker] NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"

mark_running
launch_phase4_load "${NET_BURST_VALUE}"
if (( NET_BURST_VALUE > 0 )); then
  sleep "${PHASE4_LOAD_LEADIN_SEC}"
fi

set +e
timeout --signal=TERM --kill-after=30 "${MODE_TIMEOUT_SEC}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/${TARGET_SCRIPT}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --dtype "${DTYPE}" \
    --output-dir "${RUN_ROOT}" \
    --tag "${MODE_UPPER}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --batch-size "${BATCH_SIZE}" \
    --bucket-cap-mb "${BUCKET_CAP_MB}" \
    --lr "${LR}" \
    --model-seed "${MODEL_SEED}" \
    --net-burst "${NET_BURST_VALUE}"
RUN_RC=$?
set -e

if (( RUN_RC == 0 )); then
  mark_success
  echo "[phase4-worker] complete worker=${WORKER_NAME} mode=${MODE_UPPER}"
  exit 0
fi

if (( RUN_RC == 124 )); then
  STATUS_MESSAGE=timeout
else
  STATUS_MESSAGE=launcher_failed
fi
echo "[phase4-worker] failure worker=${WORKER_NAME} mode=${MODE_UPPER} rc=${RUN_RC} message=${STATUS_MESSAGE}" >&2
exit "${RUN_RC}"
