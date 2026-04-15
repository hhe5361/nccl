#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}
IMAGE_TAG=${IMAGE_TAG:-nccl-cu118-dev:latest}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu118-dev}
HOST_USER=${HOST_USER:-$(id -un)}
HOST_UID=${HOST_UID:-$(id -u)}
HOST_GID=${HOST_GID:-$(id -g)}
WORKDIR_IN_CONTAINER=${WORKDIR_IN_CONTAINER:-/workspace/$(basename "${REPO_ROOT}")}

DOCKER_ARGS=(
  --rm
  -it
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
  DOCKER_ARGS+=(--device /dev/infiniband)
fi

if [[ -d "${HOME}/.cache/pip" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.cache/pip:/home/${HOST_USER}/.cache/pip")
fi

if [[ $# -gt 0 ]]; then
  CMD=("$@")
else
  CMD=(bash)
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" "${CMD[@]}"
