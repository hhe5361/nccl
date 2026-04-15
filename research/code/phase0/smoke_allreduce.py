#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tensor = torch.arange(8, device=device, dtype=torch.float32) + (rank + 1)
    before = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()

    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"before={before.tolist()} after={tensor.tolist()}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
