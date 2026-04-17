#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class SingleVectorModel(nn.Module):
    """Communication-heavy, compute-light model for DDP ring sensitivity tests."""

    def __init__(self, numel: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(numel, dtype=dtype) * 1e-3)

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (self.weight * scale).sum() / self.weight.numel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 B1 DDP ring sensitivity workload")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--param-mb", type=int, default=64, help="Model parameter size in MiB")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bucket-cap-mb", type=int, default=128)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--output-dir", default=os.environ.get("PHASE1_OUTPUT_DIR"))
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = resolve_dtype(args.dtype)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = (args.param_mb * 1024 * 1024) // elem_size
    run_tag = args.tag or f"W{os.environ.get('NCCL_PHASE1_STATIC_W', 'unset')}"

    torch.manual_seed(20260417 + rank)

    model = SingleVectorModel(numel=numel, dtype=dtype).to(device)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=args.bucket_cap_mb,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.lr)

    grad_bytes = sum(param.numel() * param.element_size() for param in ddp_model.module.parameters())
    # Approximate ring traffic volume for one all-reduce of this gradient tensor.
    ring_bytes_est = grad_bytes * (2.0 * (world_size - 1) / world_size)
    ring_mb_est = ring_bytes_est / (1024.0 * 1024.0)

    output_dir = Path(args.output_dir) if args.output_dir else None
    step_metrics_path = None
    if rank == 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        step_metrics_path = output_dir / f"{run_tag}_step_metrics.jsonl"

    if rank == 0:
        print(
            f"phase1_b1 ddp_ring steps={args.steps} warmup_steps={args.warmup_steps} "
            f"param_mb={args.param_mb} dtype={args.dtype} bucket_cap_mb={args.bucket_cap_mb} "
            f"world_size={world_size} run_tag={run_tag} ring_mb_est={ring_mb_est:.3f}"
        )

    records = []
    dist.barrier()
    handle = step_metrics_path.open("w", encoding="utf-8") if step_metrics_path is not None else None
    try:
        for step in range(args.steps):
            scale = torch.tensor(1.0 + rank + step * 0.001, device=device, dtype=dtype)
            target = torch.tensor(0.5 + step * 0.01, device=device, dtype=torch.float32)

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            out = ddp_model(scale)

            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            loss = (out.float() - target).pow(2)
            loss.backward()

            torch.cuda.synchronize(device)
            t2 = time.perf_counter()

            optimizer.step()

            torch.cuda.synchronize(device)
            t3 = time.perf_counter()

            local_times = torch.tensor(
                [
                    (t3 - t0) * 1000.0,
                    (t1 - t0) * 1000.0,
                    (t2 - t1) * 1000.0,
                    (t3 - t2) * 1000.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            max_times = local_times.clone()
            mean_times = local_times.clone()
            dist.all_reduce(max_times, op=dist.ReduceOp.MAX)
            dist.all_reduce(mean_times, op=dist.ReduceOp.SUM)
            mean_times /= world_size

            loss_tensor = torch.tensor([loss.detach().item()], device=device, dtype=torch.float64)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_mean = float((loss_tensor / world_size).item())

            backward_s = max(float(max_times[2].item()), 1e-9) / 1000.0
            ring_gbps_est = (ring_bytes_est * 8.0) / backward_s / 1e9

            record = {
                "step": step,
                "tag": run_tag,
                "world_size": world_size,
                "param_mb": args.param_mb,
                "ring_mb_est": ring_mb_est,
                "step_ms_max": float(max_times[0].item()),
                "step_ms_mean": float(mean_times[0].item()),
                "forward_ms_max": float(max_times[1].item()),
                "backward_ms_max": float(max_times[2].item()),
                "optimizer_ms_max": float(max_times[3].item()),
                "loss_mean": loss_mean,
                "ring_gbps_est": ring_gbps_est,
                "warmup": step < args.warmup_steps,
                "phase1_w": int(os.environ.get("NCCL_PHASE1_STATIC_W", "0")),
                "algo": os.environ.get("NCCL_ALGO", ""),
                "proto": os.environ.get("NCCL_PROTO", ""),
            }

            if rank == 0:
                print(json.dumps(record, sort_keys=True))
                records.append(record)
                if handle is not None:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        if handle is not None:
            handle.close()

    if rank == 0 and output_dir is not None:
        effective = [r for r in records if not r["warmup"]]
        summary = {
            "run_tag": run_tag,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "effective_steps": len(effective),
            "param_mb": args.param_mb,
            "dtype": args.dtype,
            "bucket_cap_mb": args.bucket_cap_mb,
            "phase1_w": int(os.environ.get("NCCL_PHASE1_STATIC_W", "0")),
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p50": percentile([r["step_ms_max"] for r in effective], 0.50),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "backward_ms_avg": mean(r["backward_ms_max"] for r in effective),
            "backward_ms_p50": percentile([r["backward_ms_max"] for r in effective], 0.50),
            "backward_ms_p95": percentile([r["backward_ms_max"] for r in effective], 0.95),
            "ring_gbps_avg": mean(r["ring_gbps_est"] for r in effective),
            "ring_gbps_p50": percentile([r["ring_gbps_est"] for r in effective], 0.50),
            "ring_gbps_p95": percentile([r["ring_gbps_est"] for r in effective], 0.95),
            "loss_mean_last": effective[-1]["loss_mean"] if effective else 0.0,
        }
        summary_path = output_dir / f"{run_tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
