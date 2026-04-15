#!/usr/bin/env bash
# Persistent 개발용 컨테이너 실행 스크립트.
# - 컨테이너가 없으면 detached로 생성
# - 컨테이너가 멈춰 있으면 start
# - 이후 docker exec 로 shell/command 진입

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}
IMAGE_TAG=${IMAGE_TAG:-nccl-cu121-dev:latest}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu121-dev}
HOST_USER=${HOST_USER:-$(id -un)}
HOST_UID=${HOST_UID:-$(id -u)}
HOST_GID=${HOST_GID:-$(id -g)}
WORKDIR_IN_CONTAINER=${WORKDIR_IN_CONTAINER:-/workspace/$(basename "${REPO_ROOT}")}

RUN_ARGS=(
  -d
  --name "${CONTAINER_NAME}"
  --gpus all
  --network host
  --ipc host
  --ulimit memlock=-1
  --cap-add IPC_LOCK
  -e HOME="/home/${HOST_USER}"
  -e USER="${HOST_USER}"
  -e HOST_UID="${HOST_UID}"
  -e HOST_GID="${HOST_GID}"
  -e CUDA_HOME=/usr/local/cuda
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1}"
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  -e NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  -v "${WORKSPACE_ROOT}:/workspace"
  -w "${WORKDIR_IN_CONTAINER}"
)

if [[ -d /dev/infiniband ]]; then
  RUN_ARGS+=(--device /dev/infiniband)
fi

if [[ -d /mnt/nfs_share ]]; then
  RUN_ARGS+=(-v "/mnt/nfs_share:/mnt/nfs_share")
fi

if [[ -d "${HOME}/.cache/pip" ]]; then
  RUN_ARGS+=(-v "${HOME}/.cache/pip:/home/${HOST_USER}/.cache/pip")
fi

EXEC_ARGS=(
  -w "${WORKDIR_IN_CONTAINER}"
  -e HOME="/home/${HOST_USER}"
  -e USER="${HOST_USER}"
  -e HOST_UID="${HOST_UID}"
  -e HOST_GID="${HOST_GID}"
  -e CUDA_HOME=/usr/local/cuda
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1}"
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  -e NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
)

if [[ -t 0 && -t 1 ]]; then
  EXEC_ARGS=(-it "${EXEC_ARGS[@]}")
fi

if [[ $# -gt 0 ]]; then
  CMD=("$@")
else
  CMD=(bash)
fi

container_exists() {
  docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

if ! container_exists; then
  echo "[docker] creating persistent container ${CONTAINER_NAME} from ${IMAGE_TAG}"
  docker run "${RUN_ARGS[@]}" "${IMAGE_TAG}" tail -f /dev/null >/dev/null
elif ! container_running; then
  echo "[docker] starting existing container ${CONTAINER_NAME}"
  docker start "${CONTAINER_NAME}" >/dev/null
else
  echo "[docker] reusing running container ${CONTAINER_NAME}"
fi

exec docker exec "${EXEC_ARGS[@]}" "${CONTAINER_NAME}" "${CMD[@]}"
