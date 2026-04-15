# Phase 0 PyTorch Examples

These examples assume that PyTorch inside the container was built against the
custom NCCL in this repository.

## Files

- `smoke_allreduce.py`
  - Small `torch.distributed` smoke test.
  - Confirms that `torchrun` + NCCL process group + all-reduce work.
- `ddp_minimal_train.py`
  - Minimal `DistributedDataParallel` training loop on synthetic tensors.
  - Useful for verifying that a real DDP backward/update path works with the
    custom NCCL build.

## Single-node examples

Run inside the container after activating the custom PyTorch environment:

```bash
source /workspace/venvs/torch-cu118-custom/bin/activate

# 4 GPUs on one node
torchrun --standalone --nproc_per_node=4 \
  research/code/phase0/smoke_allreduce.py

torchrun --standalone --nproc_per_node=4 \
  research/code/phase0/ddp_minimal_train.py --steps 20 --batch-size 32
```

## Multi-node shape

```bash
torchrun \
  --nnodes=8 \
  --nproc_per_node=4 \
  --node_rank=<node_rank> \
  --master_addr=<master_ip> \
  --master_port=29500 \
  research/code/phase0/smoke_allreduce.py
```

Replace `smoke_allreduce.py` with `ddp_minimal_train.py` once the smoke test is
stable.
