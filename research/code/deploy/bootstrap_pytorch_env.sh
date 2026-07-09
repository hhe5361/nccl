#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NCCL_HOME=${NCCL_HOME:-${REPO_ROOT}/build}
NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-${WORKSPACE_ROOT}/nccl-tests}
PYTORCH_DIR=${PYTORCH_DIR:-${WORKSPACE_ROOT}/pytorch}
PYTORCH_REF=${PYTORCH_REF:-main}
VENV_DIR=${VENV_DIR:-${WORKSPACE_ROOT}/venvs/torch-cu132-custom}
DEFAULT_MAX_JOBS=$(nproc)
if [[ "${DEFAULT_MAX_JOBS}" -gt 4 ]]; then
  DEFAULT_MAX_JOBS=4
fi
MAX_JOBS=${MAX_JOBS:-${DEFAULT_MAX_JOBS}}
NVCC_GENCODE_DEFAULT="-gencode=arch=compute_120,code=sm_120"
NVCC_GENCODE=${NVCC_GENCODE:-${NVCC_GENCODE_DEFAULT}}
MPI_HOME=${MPI_HOME:-}
if [[ -z "${MPI_HOME}" ]]; then
  if [[ -d /usr/lib/x86_64-linux-gnu/openmpi ]]; then
    MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
  elif command -v mpicc >/dev/null 2>&1; then
    MPI_COMPILE_FLAGS=$(mpicc --showme:compile 2>/dev/null || true)
    for flag in ${MPI_COMPILE_FLAGS}; do
      case "${flag}" in
        -I*/include)
          MPI_INCLUDE_DIR=${flag#-I}
          MPI_HOME=$(dirname "${MPI_INCLUDE_DIR}")
          break
          ;;
      esac
    done
  fi
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export MAX_JOBS

usage() {
  cat <<'USAGE'
Usage: bootstrap_pytorch_env.sh [nccl|tests|pytorch|all]

Targets:
  nccl     Build the custom NCCL in the current repo.
  tests    Clone and build nccl-tests against the custom NCCL.
  pytorch  Clone and build PyTorch against the custom NCCL.
  all      Run nccl, tests, pytorch in order.

Environment:
  MAX_JOBS   Parallel build jobs. Defaults to min(nproc, 4).
  MPI_HOME   MPI installation prefix for nccl-tests. Auto-detected when possible.
USAGE
}

build_nccl() {
  echo "[bootstrap] building NCCL at ${REPO_ROOT} with MAX_JOBS=${MAX_JOBS}"
  make -C "${REPO_ROOT}" -j"${MAX_JOBS}" src.build \
    CUDA_HOME="${CUDA_HOME}" \
    NVCC_GENCODE="${NVCC_GENCODE}"
}

build_tests() {
  echo "[bootstrap] building nccl-tests at ${NCCL_TESTS_DIR} with MAX_JOBS=${MAX_JOBS}"
  if [[ ! -d "${NCCL_TESTS_DIR}/.git" ]]; then
    git clone https://github.com/NVIDIA/nccl-tests.git "${NCCL_TESTS_DIR}"
  fi

  local mpi_args=(MPI=1)
  if [[ -n "${MPI_HOME}" ]]; then
    echo "[bootstrap] using MPI_HOME=${MPI_HOME}"
    mpi_args+=(MPI_HOME="${MPI_HOME}")
  else
    echo "[bootstrap] MPI_HOME not detected; nccl-tests MPI build may fail"
  fi

  make -C "${NCCL_TESTS_DIR}" -j"${MAX_JOBS}" \
    "${mpi_args[@]}" \
    CUDA_HOME="${CUDA_HOME}" \
    NCCL_HOME="${NCCL_HOME}"
}

build_pytorch() {
  echo "[bootstrap] building PyTorch at ${PYTORCH_DIR} with MAX_JOBS=${MAX_JOBS}"
  if [[ ! -d "${PYTORCH_DIR}/.git" ]]; then
    git clone --recursive https://github.com/pytorch/pytorch "${PYTORCH_DIR}"
  fi
  git -C "${PYTORCH_DIR}" fetch --tags --force
  git -C "${PYTORCH_DIR}" checkout "${PYTORCH_REF}"
  git -C "${PYTORCH_DIR}" submodule sync
  git -C "${PYTORCH_DIR}" submodule update --init --recursive

  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip setuptools wheel ninja "cmake<4"
  pip install -r "${PYTORCH_DIR}/requirements.txt"
  # Prefer PyTorch's bundled pybind11 CMake package over any pip-installed one.
  pip uninstall -y pybind11 >/dev/null 2>&1 || true

  if ! python - <<'PYEOF'
import pathlib, sysconfig
inc = pathlib.Path(sysconfig.get_paths()["include"])
header = inc / "Python.h"
raise SystemExit(0 if header.exists() else 1)
PYEOF
  then
    echo "[bootstrap] Python development headers are missing for $(python -V 2>&1). Install python3-dev in the container or rebuild the image." >&2
    return 1
  fi

  local pyexe
  pyexe=$(command -v python)

  export USE_CUDA=1
  export USE_DISTRIBUTED=1
  export USE_NCCL=1
  export USE_SYSTEM_NCCL=1
  export NCCL_ROOT="${NCCL_HOME}"
  export NCCL_INCLUDE_DIR="${NCCL_HOME}/include"
  export NCCL_LIB_DIR="${NCCL_HOME}/lib"
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DBUILD_PYTHON=ON -DPython_EXECUTABLE=${pyexe} -DPython3_EXECUTABLE=${pyexe} -DPython_FIND_STRATEGY=LOCATION -DPython3_FIND_STRATEGY=LOCATION"

  (cd "${PYTORCH_DIR}" && python setup.py develop)
}

TARGET=${1:-all}
case "${TARGET}" in
  nccl)
    build_nccl
    ;;
  tests)
    build_tests
    ;;
  pytorch)
    build_pytorch
    ;;
  all)
    build_nccl
    build_tests
    build_pytorch
    ;;
  *)
    usage
    exit 1
    ;;
esac
