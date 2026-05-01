# Docker Deployment Scripts

This directory packages a CUDA 12.1 development container for the custom NCCL
workflow discussed in `research/docs`.

## Files

- `Dockerfile`
  - CUDA 12.1 devel image with RDMA, OpenMPI, Python, and build dependencies.
- `build_image.sh`
  - Builds the dev image with the current host UID/GID mapped into the image.
- `run_dev_container.sh`
  - Starts an interactive container with GPU access, host networking, and
    `/dev/infiniband` forwarded when present.
  - Reuses a persistent container by default.
  - Supports `--reset --start-only` for clean matrix startup.
- `bootstrap_pytorch_env.sh`
  - Runs inside the container and builds:
    - the custom NCCL in this repo
    - `nccl-tests`
    - a source-built PyTorch linked against the custom NCCL
- `common.sh`
  - Shared worker/IP/status helper functions for deploy orchestration.
- `run_dev_worker_once.sh`
  - Single-shot worker launcher.
  - Writes NFS-backed `STATUS_DDP` state files with `1 -> 0` or `-1` on failure.
- `run_dev_matrix_master.sh`
  - Master-side serial orchestrator.
  - Dispatches one experiment at a time to all workers, waits for status-file
    barriers, optionally validates results, and only then proceeds.

## Recommended Flow

```bash
cd research/code/deploy
chmod +x build_image.sh run_dev_container.sh bootstrap_pytorch_env.sh

./build_image.sh
./run_dev_container.sh
```

Inside the container:

```bash
./research/code/deploy/bootstrap_pytorch_env.sh nccl
./research/code/deploy/bootstrap_pytorch_env.sh tests
./research/code/deploy/bootstrap_pytorch_env.sh pytorch
```

Or all at once:

```bash
./research/code/deploy/bootstrap_pytorch_env.sh all
```

## Runtime Notes

- The scripts assume:
  - NVIDIA driver is installed on the host
  - Docker is installed on the host
  - NVIDIA Container Toolkit is installed and Docker is configured with it
- The matrix orchestration flow is intentionally centralized:
  - worker01 is expected to act as the master orchestrator
  - workers are launched by SSH as single-shot jobs
  - master waits for NFS status files to reach `0` before moving forward
- `run_dev_container.sh` defaults to `NCCL_NET_GDR_LEVEL=0` because the target
  setup discussed so far is a RoCE v2 environment where GPUDirect RDMA may be
  unavailable.
- The default PyTorch build target assumes GTX 1080 Ti, so
  `TORCH_CUDA_ARCH_LIST=6.1` and
  `NVCC_GENCODE=-gencode=arch=compute_61,code=sm_61`.
