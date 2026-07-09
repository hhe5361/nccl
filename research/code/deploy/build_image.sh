#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE_TAG=${IMAGE_TAG:-nccl-cu132-dev:latest}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04}
HOST_UID=${HOST_UID:-$(id -u)}
HOST_GID=${HOST_GID:-$(id -g)}
HOST_USER=${HOST_USER:-$(id -un)}

docker build \
  --build-arg CUDA_IMAGE="${CUDA_IMAGE}" \
  --build-arg HOST_UID="${HOST_UID}" \
  --build-arg HOST_GID="${HOST_GID}" \
  --build-arg HOST_USER="${HOST_USER}" \
  -t "${IMAGE_TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"
