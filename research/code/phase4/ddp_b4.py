#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 tiny DDP training workload")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--output-dir", default=os.environ.get("PHASE4_OUTPUT_DIR"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bucket-cap-mb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
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


class TinyStack(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = resolve_dtype(args.dtype)

    run_tag = args.tag or os.environ.get("PHASE4_MODE", "B4").upper()
    output_dir = Path(args.output_dir) if args.output_dir else None
    worker_name = os.environ.get("WORKER_NAME", f"worker_rank{rank}")
    phase4_mode = os.environ.get("PHASE4_MODE", "stock").lower()
    phase4_enable = int(os.environ.get("NCCL_PHASE4_ENABLE", "0"))
    phase4_post_receive_w = int(os.environ.get("NCCL_PHASE4_POST_RECEIVE_W", "0"))

    if rank == 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    step_metrics_path = output_dir / f"{run_tag}_step_metrics.jsonl" if rank == 0 and output_dir is not None else None
    worker_dir = output_dir / worker_name if output_dir is not None else None
    step_timing_path = worker_dir / f"{run_tag}_worker_step_timing.jsonl" if worker_dir is not None else None
    validation_path = worker_dir / f"{run_tag}_rank_validation.json" if worker_dir is not None else None

    model = TinyStack(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device=device, dtype=dtype)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        bucket_cap_mb=args.bucket_cap_mb,
        static_graph=True,
    )
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    global_batch_size = args.batch_size * world_size
    total_param_bytes = sum(p.numel() * p.element_size() for p in ddp_model.module.parameters())
    total_param_mb = total_param_bytes / (1024.0 * 1024.0)

    if rank == 0:
        print(
            f"phase4_b4 ddp steps={args.steps} warmup_steps={args.warmup_steps} "
            f"hidden_dim={args.hidden_dim} num_layers={args.num_layers} batch_size={args.batch_size} "
            f"bucket_cap_mb={args.bucket_cap_mb} dtype={args.dtype} world_size={world_size} "
            f"phase4_mode={phase4_mode} phase4_enable={phase4_enable} phase4_post_receive_w={phase4_post_receive_w} "
            f"param_mb={total_param_mb:.6f}"
        )

    records = []
    dist.barrier()
    handle = step_metrics_path.open("w", encoding="utf-8") if step_metrics_path is not None else None
    timing_handle = None
    if worker_dir is not None:
        worker_dir.mkdir(parents=True, exist_ok=True)
        timing_handle = step_timing_path.open("w", encoding="utf-8")

    try:
        for step in range(args.steps):
            step_seed = 20260504 + step
            gen = torch.Generator(device=device)
            gen.manual_seed(step_seed)
            inputs = torch.randn((args.batch_size, args.hidden_dim), device=device, dtype=dtype, generator=gen)
            targets = torch.randn((args.batch_size, args.hidden_dim), device=device, dtype=dtype, generator=gen)

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_start_ns = time.monotonic_ns()
            ts_start_unix_ns = time.time_ns()
            t0 = time.perf_counter()

            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            ts_end_unix_ns = time.time_ns()
            step_end_ns = time.monotonic_ns()

            local_ms = (t1 - t0) * 1000.0
            local_times = torch.tensor([local_ms], device=device, dtype=torch.float64)
            max_times = local_times.clone()
            mean_times = local_times.clone()
            dist.all_reduce(max_times, op=dist.ReduceOp.MAX)
            dist.all_reduce(mean_times, op=dist.ReduceOp.SUM)
            mean_times /= world_size

            duration_s = max(float(max_times[0].item()) / 1000.0, 1e-9)
            steps_per_sec = 1.0 / duration_s
            samples_per_sec = global_batch_size / duration_s

            record = {
                "step": step,
                "tag": run_tag,
                "phase4_mode": phase4_mode,
                "phase4_enable": phase4_enable,
                "phase4_post_receive_w": phase4_post_receive_w,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "global_batch_size": global_batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "bucket_cap_mb": args.bucket_cap_mb,
                "lr": args.lr,
                "dtype": args.dtype,
                "param_mb": total_param_mb,
                "ts_start_unix_ns": int(ts_start_unix_ns),
                "ts_end_unix_ns": int(ts_end_unix_ns),
                "ts_mid_unix_ns": int((ts_start_unix_ns + ts_end_unix_ns) // 2),
                "step_ms_max": float(max_times[0].item()),
                "step_ms_mean": float(mean_times[0].item()),
                "steps_per_sec": steps_per_sec,
                "samples_per_sec": samples_per_sec,
                "loss": float(loss.detach().float().item()),
                "warmup": step < args.warmup_steps,
                "algo": os.environ.get("NCCL_ALGO", ""),
                "proto": os.environ.get("NCCL_PROTO", ""),
            }

            if rank == 0:
                print(json.dumps(record, sort_keys=True))
                records.append(record)
                if handle is not None:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()

            if timing_handle is not None:
                timing_row = {
                    "step": step,
                    "rank": rank,
                    "worker": worker_name,
                    "phase4_mode": phase4_mode,
                    "phase4_enable": phase4_enable,
                    "phase4_post_receive_w": phase4_post_receive_w,
                    "warmup": step < args.warmup_steps,
                    "start_ns": step_start_ns,
                    "end_ns": step_end_ns,
                    "local_step_ms": local_ms,
                }
                timing_handle.write(json.dumps(timing_row, sort_keys=True) + "\n")
                timing_handle.flush()
    finally:
        if handle is not None:
            handle.close()
        if timing_handle is not None:
            timing_handle.close()

    if validation_path is not None:
        cpu_tensor = flatten_params(ddp_model.module).contiguous().cpu()
        tensor_bytes = cpu_tensor.numpy().tobytes()
        validation_row = {
            "worker": worker_name,
            "rank": rank,
            "tag": run_tag,
            "phase4_mode": phase4_mode,
            "phase4_enable": phase4_enable,
            "phase4_post_receive_w": phase4_post_receive_w,
            "world_size": world_size,
            "dtype": args.dtype,
            "numel": int(cpu_tensor.numel()),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "bucket_cap_mb": args.bucket_cap_mb,
            "local_validation_passed": bool(torch.isfinite(cpu_tensor).all().item()),
            "final_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
            "final_head": cpu_tensor.flatten()[:8].tolist(),
        }
        validation_path.write_text(json.dumps(validation_row, indent=2, sort_keys=True), encoding="utf-8")

    if rank == 0 and output_dir is not None:
        effective = [r for r in records if not r["warmup"]]
        summary = {
            "run_tag": run_tag,
            "phase4_mode": phase4_mode,
            "phase4_enable": phase4_enable,
            "phase4_post_receive_w": phase4_post_receive_w,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "effective_steps": len(effective),
            "batch_size": args.batch_size,
            "global_batch_size": global_batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "bucket_cap_mb": args.bucket_cap_mb,
            "lr": args.lr,
            "dtype": args.dtype,
            "param_mb": total_param_mb,
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p50": percentile([r["step_ms_max"] for r in effective], 0.50),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "steps_per_sec_avg": mean(r["steps_per_sec"] for r in effective),
            "steps_per_sec_p50": percentile([r["steps_per_sec"] for r in effective], 0.50),
            "steps_per_sec_p95": percentile([r["steps_per_sec"] for r in effective], 0.95),
            "samples_per_sec_avg": mean(r["samples_per_sec"] for r in effective),
            "samples_per_sec_p50": percentile([r["samples_per_sec"] for r in effective], 0.50),
            "samples_per_sec_p95": percentile([r["samples_per_sec"] for r in effective], 0.95),
            "loss_avg": mean(r["loss"] for r in effective),
            "loss_p50": percentile([r["loss"] for r in effective], 0.50),
            "loss_p95": percentile([r["loss"] for r in effective], 0.95),
        }
        summary_path = output_dir / f"{run_tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
