#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

RUN_ID=${RUN_ID:-phase0_$(date +%y%m%d_%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-172.16.0.101}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu132-dev}
REMOTE_REPO=${REMOTE_REPO:-/workspace/nccl}
WORKERS_DEFAULT=(worker01 worker02 worker03 worker04 worker05 worker06 worker07 worker08)

if [[ $# -gt 0 ]]; then
  PHASE0_ARGS=("$@")
else
  PHASE0_ARGS=(--steps 30 --tensor-mb 16 --dtype float32)
fi

if [[ -n "${WORKERS:-}" ]]; then
  # shellcheck disable=SC2206
  WORKERS_ARR=(${WORKERS})
else
  WORKERS_ARR=("${WORKERS_DEFAULT[@]}")
fi

if [[ ${#WORKERS_ARR[@]} -ne ${NNODES} ]]; then
  echo "WORKERS count (${#WORKERS_ARR[@]}) must match NNODES (${NNODES})" >&2
  exit 1
fi

shell_quote() {
  printf '%q' "$1"
}

join_quoted() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(shell_quote "$arg") "
  done
  printf '%s' "${out}"
}

FORWARDED_ARGS=$(join_quoted "${PHASE0_ARGS[@]}")

echo "[phase0-launch] RUN_ID=${RUN_ID}"
echo "[phase0-launch] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[phase0-launch] WORKERS=${WORKERS_ARR[*]}"
echo "[phase0-launch] CONTAINER_NAME=${CONTAINER_NAME}"
echo "[phase0-launch] REMOTE_REPO=${REMOTE_REPO}"
echo "[phase0-launch] PHASE0_ARGS=${PHASE0_ARGS[*]}"

pids=()
workers_started=()

for worker in "${WORKERS_ARR[@]}"; do
  inner_cmd="cd $(shell_quote "${REMOTE_REPO}") && RUN_ID=$(shell_quote "${RUN_ID}") MASTER_ADDR=$(shell_quote "${MASTER_ADDR}") MASTER_PORT=$(shell_quote "${MASTER_PORT}") NNODES=$(shell_quote "${NNODES}") NPROC_PER_NODE=$(shell_quote "${NPROC_PER_NODE}") CUDA_VISIBLE_DEVICES=$(shell_quote "${CUDA_VISIBLE_DEVICES}") WORKER_NAME=$(shell_quote "${worker}") ./research/code/phase0/run_phase0_torch.sh ${FORWARDED_ARGS}"
  remote_cmd="docker exec $(shell_quote "${CONTAINER_NAME}") bash -lc $(shell_quote "${inner_cmd}")"
  echo "[phase0-launch] starting ${worker}"
  ssh "${worker}" "${remote_cmd}" &
  pids+=("$!")
  workers_started+=("${worker}")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[phase0-launch] worker failed: ${workers_started[$i]}" >&2
    fail=1
  fi
done

if [[ ${fail} -ne 0 ]]; then
  exit 1
fi

echo "[phase0-launch] completed RUN_ID=${RUN_ID}"
