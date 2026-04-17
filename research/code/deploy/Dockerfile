FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=nccl

SHELL ["/bin/bash", "-lc"]

RUN set -eux; \
    install_deps() { \
      apt-get update -o Acquire::Retries=5; \
      apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gdb \
        git \
        ibverbs-providers \
        iproute2 \
        iputils-ping \
        less \
        libibverbs-dev \
        libnuma-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pciutils \
        pkg-config \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        rdma-core \
        rsync \
        vim \
        wget; \
    }; \
    for attempt in 1 2 3 4 5; do \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
      if install_deps; then \
        break; \
      fi; \
      if [[ "${attempt}" == 5 ]]; then \
        exit 1; \
      fi; \
      echo "apt mirror transient failure on attempt ${attempt}; retrying in 10s"; \
      sleep 10; \
    done; \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${HOST_GID}" "${HOST_USER}" \
    && useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home --shell /bin/bash "${HOST_USER}"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=6.1
ENV WORKSPACE_ROOT=/workspace

USER ${HOST_USER}
WORKDIR /workspace/nccl
