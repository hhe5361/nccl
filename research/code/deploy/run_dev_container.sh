#!/usr/bin/env bash
# docker 이미지 실행 스크립트. 

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd) #NCCL repo root directory 
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")} #repo 외부의 디렉토리를 workspace로 사용, 기본값은 repo의 부모 디렉토리
IMAGE_TAG=${IMAGE_TAG:-nccl-cu121-dev:latest}
CONTAINER_NAME=${CONTAINER_NAME:-nccl-cu121-dev}
HOST_USER=${HOST_USER:-$(id -un)}
HOST_UID=${HOST_UID:-$(id -u)}
HOST_GID=${HOST_GID:-$(id -g)}
WORKDIR_IN_CONTAINER=${WORKDIR_IN_CONTAINER:-/workspace/$(basename "${REPO_ROOT}")}

#gpu_all 이 GPU 컨테이너에 노출하는거, network host는 컨테이너가 호스트 네트워크 그대ㅗㄹ 쓰도록
#--ipc host는 shared memory를 호스트와 공유하도록함. 
#ulimit memlock=-1과 --cap-add IPC_LOCK는 NCCL이 필요한 메모리를 잠글 수 있도록 해줌.
#RDMA 장치가 있으면 컨테이너에 같이 넘기는 방식

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

#RDMA 장치가 있을 경우, 컨테이너에 같이 넘기도록
if [[ -d /dev/infiniband ]]; then
  DOCKER_ARGS+=(--device /dev/infiniband)
fi

if [[ -d "${HOME}/.cache/pip" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.cache/pip:/home/${HOST_USER}/.cache/pip")
fi

#인자 안 주면 bash 실행, 인자 주면 그거 실행
if [[ $# -gt 0 ]]; then
  CMD=("$@")
else
  CMD=(bash)
fi


docker run "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" "${CMD[@]}"
