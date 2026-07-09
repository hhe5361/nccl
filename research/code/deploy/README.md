# Docker Deployment Scripts

This directory packages a CUDA 13.2 development container for the custom NCCL
workflow discussed in `research/docs`.

## Files

- `Dockerfile`
  - CUDA 13.2 cuDNN devel image with RDMA, OpenMPI, Python, and build dependencies.
- `build_image.sh`
  - Builds the dev image with the current host UID/GID mapped into the image.
- `run_dev_container.sh`
  - Starts an interactive container with GPU access, host networking, and
    `/dev/infiniband` forwarded when present.
- `bootstrap_pytorch_env.sh`
  - Runs inside the container and builds:
    - the custom NCCL in this repo
    - `nccl-tests`
    - a source-built PyTorch linked against the custom NCCL

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
- `run_dev_container.sh` defaults to `NCCL_NET_GDR_LEVEL=0` because the target
  setup discussed so far is a RoCE v2 environment where GPUDirect RDMA may be
  unavailable.
- The default PyTorch build target assumes RTX 5070 Ti, so
  `TORCH_CUDA_ARCH_LIST=12.0` and
  `NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120`.
