#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collective", choices=["allreduce", "alltoall"], required=True)
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--payload-mb", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def bytes_to_elems(payload_mb: int, dtype: torch.dtype, world_size: int, collective: str) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = payload_mb * 1024 * 1024
    elems = max(1, total_bytes // element_size)
    if collective == "alltoall":
        remainder = elems % world_size
        if remainder:
            elems += world_size - remainder
    return elems


def make_allreduce_tensor(numel: int, rank: int, device: torch.device) -> torch.Tensor:
    return torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)


def make_alltoall_input(numel: int, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    split = numel // world_size
    tensor = torch.empty((numel,), dtype=torch.float32, device=device)
    for peer in range(world_size):
        start = peer * split
        end = start + split
        tensor[start:end].fill_(float(rank * 1000 + peer))
    return tensor


def verify_allreduce(tensor: torch.Tensor, world_size: int) -> dict:
    expected = float(world_size * (world_size + 1) // 2)
    max_abs = float((tensor - expected).abs().max().item())
    return {"ok": max_abs == 0.0, "expected": expected, "max_abs_err": max_abs}


def verify_alltoall(tensor: torch.Tensor, rank: int, world_size: int) -> dict:
    split = tensor.numel() // world_size
    max_abs = 0.0
    for src in range(world_size):
        start = src * split
        end = start + split
        expected = float(src * 1000 + rank)
        err = float((tensor[start:end] - expected).abs().max().item())
        max_abs = max(max_abs, err)
    return {"ok": max_abs == 0.0, "max_abs_err": max_abs}


def build_probe(tensor: torch.Tensor) -> dict:
    sample_len = min(8, tensor.numel())
    flat = tensor.reshape(-1)
    sample = flat[:sample_len].detach().cpu().tolist()
    return {
        "sum": float(tensor.sum().item()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "sample": [float(v) for v in sample],
    }


def run_step(collective: str, rank: int, world_size: int, numel: int, device: torch.device):
    if collective == "allreduce":
        tensor = make_allreduce_tensor(numel, rank, device)
        dist.barrier()
        torch.cuda.synchronize(device)
        start_ns = time.time_ns()
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        end_ns = time.time_ns()
        return tensor, start_ns, end_ns

    if collective == "alltoall":
        input_tensor = make_alltoall_input(numel, rank, world_size, device)
        output_tensor = torch.empty_like(input_tensor)
        dist.barrier()
        torch.cuda.synchronize(device)
        start_ns = time.time_ns()
        dist.all_to_all_single(output_tensor, input_tensor)
        torch.cuda.synchronize(device)
        end_ns = time.time_ns()
        return output_tensor, start_ns, end_ns

    raise ValueError(f"Unsupported collective: {collective}")


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"rank{rank:02d}_step_trace.jsonl"
    summary_path = output_dir / f"rank{rank:02d}_summary.json"

    numel = bytes_to_elems(args.payload_mb, torch.float32, world_size, args.collective)
    step_records = []
    verification = []

    with trace_path.open("w", encoding="utf-8") as trace_fp:
        total_steps = args.warmup_steps + args.steps
        for step_idx in range(total_steps):
            phase = "warmup" if step_idx < args.warmup_steps else "steady"
            tensor, start_ns, end_ns = run_step(args.collective, rank, world_size, numel, device)
            duration_ms = (end_ns - start_ns) / 1_000_000.0

            if args.collective == "allreduce":
                verify = verify_allreduce(tensor, world_size)
            else:
                verify = verify_alltoall(tensor, rank, world_size)
            verification.append(verify["ok"])
            probe = build_probe(tensor)

            record = {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "collective": args.collective,
                "algorithm": args.algorithm,
                "phase": phase,
                "step_index": step_idx if phase == "warmup" else step_idx - args.warmup_steps,
                "duration_ms": duration_ms,
                "ts_start_unix_ns": start_ns,
                "ts_end_unix_ns": end_ns,
                "payload_mb": args.payload_mb,
                "numel": numel,
                "verification_ok": verify["ok"],
                "verification_detail": verify,
                "probe": probe,
            }
            trace_fp.write(json.dumps(record) + "\n")
            step_records.append(record)

    steady = [r["duration_ms"] for r in step_records if r["phase"] == "steady"]
    summary = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "collective": args.collective,
        "algorithm": args.algorithm,
        "payload_mb": args.payload_mb,
        "numel": numel,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "verification_all_ok": all(verification),
        "steady_avg_ms": sum(steady) / len(steady) if steady else None,
        "steady_min_ms": min(steady) if steady else None,
        "steady_max_ms": max(steady) if steady else None,
        "trace_file": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
