# Phase 0 PyTorch and Log Collection

These scripts assume the container PyTorch environment was built against the
custom NCCL in this repository.

## Files

- `smoke_allreduce.py`
  - Small `torch.distributed` smoke test.
- `ddp_minimal_train.py`
  - Minimal `DistributedDataParallel` training loop on synthetic tensors.
- `ring_allreduce_loop.py`
  - Repeated `dist.all_reduce()` loop for cleaner Phase 0 transport logging.
- `run_phase0_torch.sh`
  - Wrapper that sets custom NCCL, Phase 0 log env vars, `NCCL_ALGO=Ring`, and runs `torchrun`.
- `phase0_log_report.py`
  - Parses `PHASE0 event=...` logs and renders an HTML report.

## Recommended Phase 0 path

Start with one GPU per node across 8 workers so the ring is mostly inter-node
and crosses the rack uplinks in a simple, predictable way.

- `MASTER_ADDR=172.16.0.101`
- `NNODES=8`
- `NPROC_PER_NODE=1`
- `NCCL_ALGO=Ring`
- `NCCL_PROTO=Simple`
- `CUDA_VISIBLE_DEVICES=0`

## Run one worker

Run inside the container on each worker. Replace `NODE_RANK` with `0..7`.

```bash
cd /workspace/nccl
export RUN_ID=phase0_250416_120000  # 모든 worker에서 동일한 값 사용
export MASTER_ADDR=172.16.0.101
export MASTER_PORT=29500
export NNODES=8
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0
export NODE_RANK=0

./research/code/phase0/run_phase0_torch.sh --steps 30 --tensor-mb 16 --dtype float32
```

The script writes NCCL debug logs to:

```bash
/mnt/nfs_share/cts_experiments/${RUN_ID}/workerXX/
```

## Generate the report

After all workers finish, run on any node that can see the shared NFS path:

```bash
cd /workspace/nccl
python3 research/code/phase0/phase0_log_report.py \
  --input /mnt/nfs_share/cts_experiments/${RUN_ID} \
  --output /mnt/nfs_share/cts_experiments/${RUN_ID}/phase0-report.html \
  --topology-file research/env/network_topology_internal_ips.txt
```

This writes:

- `phase0-report.html`
- `events.csv`
- `summary.json`

## Lightweight DDP option

If you want a real model path after the all-reduce loop is stable:

```bash
source /workspace/venvs/torch-cu121-custom/bin/activate
NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_PHASE0_LOG=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET \
NCCL_DEBUG_FILE=/mnt/nfs_share/cts_experiments/${RUN_ID}/workerXX/nccl-phase0.%h.%p.log \
torchrun \
  --nnodes=8 \
  --nproc_per_node=1 \
  --node_rank=${NODE_RANK} \
  --master_addr=172.16.0.101 \
  --master_port=29500 \
  research/code/phase0/ddp_minimal_train.py --steps 20 --batch-size 8 --input-dim 128 --hidden-dim 64 --output-dim 16
```
