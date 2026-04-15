#!/usr/bin/env python3
import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated NCCL all-reduce loop for Phase 0 logging")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--tensor-mb", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between iterations")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype = resolve_dtype(args.dtype)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    numel = (args.tensor_mb * 1024 * 1024) // bytes_per_elem

    if rank == 0:
        print(
            f"phase0 ring_allreduce steps={args.steps} tensor_mb={args.tensor_mb} "
            f"dtype={args.dtype} world_size={world_size} numel={numel}"
        )

    for step in range(args.steps):
        value = float(rank + 1 + step)
        tensor = torch.full((numel,), value, device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        end = time.perf_counter()

        expected = sum(float(r + 1 + step) for r in range(world_size))
        sample = float(tensor[0].float().item())
        if rank == 0:
            print(
                f"step={step:04d} expected={expected:.1f} sample={sample:.1f} "
                f"allreduce_ms={(end - start) * 1000:.3f}"
            )

        if abs(sample - expected) > 1e-2:
            raise RuntimeError(
                f"rank {rank}: validation failed at step {step}: got {sample}, expected {expected}"
            )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
