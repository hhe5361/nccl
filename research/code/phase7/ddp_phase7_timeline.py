#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 DDP backward/communication timeline workload")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--output-dir", default=os.environ.get("PHASE4_OUTPUT_DIR"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bucket-cap-mb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--model-seed", type=int, default=20260504)
    parser.add_argument("--net-burst", type=float, default=0.0)
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
    return sum(values) / len(values) if values else 0.0


def distributed_stat(local_value: float, device: torch.device) -> tuple[float, float]:
    values = torch.tensor([local_value], device=device, dtype=torch.float64)
    max_values = values.clone()
    mean_values = values.clone()
    dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
    dist.all_reduce(mean_values, op=dist.ReduceOp.SUM)
    mean_values /= dist.get_world_size()
    return float(max_values[0].item()), float(mean_values[0].item())


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


def write_jsonl(handle, row: dict) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = resolve_dtype(args.dtype)

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)

    run_tag = args.tag or os.environ.get("PHASE4_MODE", "STOCK").upper()
    output_dir = Path(args.output_dir) if args.output_dir else None
    worker_name = os.environ.get("WORKER_NAME", f"worker_rank{rank}")
    mode = os.environ.get("PHASE4_MODE", "stock").lower()
    repeat_label = os.environ.get("PHASE4_REPEAT_LABEL", "repeat_01")
    net_burst = float(args.net_burst)

    worker_dir = output_dir / worker_name if output_dir is not None else None
    if output_dir is not None and rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if worker_dir is not None:
        worker_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = worker_dir / f"{run_tag}_phase7_backward_timeline.jsonl" if worker_dir else None
    validation_path = worker_dir / f"{run_tag}_rank_validation.json" if worker_dir else None
    final_stage_path = worker_dir / f"{run_tag}_final_stage.json" if worker_dir else None
    step_metrics_path = output_dir / f"{run_tag}_phase7_step_metrics.jsonl" if rank == 0 and output_dir else None

    def stage_log(stage: str, **extra: object) -> None:
        row = {
            "ts_unix_ns": time.time_ns(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "worker": worker_name,
            "tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": mode,
            "net_burst": net_burst,
            "stage": stage,
        }
        row.update(extra)
        print(f"[phase7-ddp] {json.dumps(row, sort_keys=True)}", file=sys.stderr, flush=True)
        if final_stage_path is not None:
            final_stage_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")

    stage_log("startup")

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
            f"phase7 ddp timeline steps={args.steps} warmup_steps={args.warmup_steps} "
            f"hidden_dim={args.hidden_dim} num_layers={args.num_layers} batch_size={args.batch_size} "
            f"bucket_cap_mb={args.bucket_cap_mb} dtype={args.dtype} world_size={world_size} "
            f"mode={mode} net_burst={net_burst} repeat_label={repeat_label} param_mb={total_param_mb:.6f}"
        )

    stage_log("before_startup_barrier")
    dist.barrier()
    stage_log("after_startup_barrier")

    records: List[dict] = []
    timeline_handle = timeline_path.open("w", encoding="utf-8") if timeline_path is not None else None
    metrics_handle = step_metrics_path.open("w", encoding="utf-8") if step_metrics_path is not None else None

    try:
        for step in range(args.steps):
            step_seed = args.model_seed + step
            gen = torch.Generator(device=device)
            gen.manual_seed(step_seed)
            inputs = torch.randn((args.batch_size, args.hidden_dim), device=device, dtype=dtype, generator=gen)
            targets = torch.randn((args.batch_size, args.hidden_dim), device=device, dtype=dtype, generator=gen)

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)

            step_start_mono_ns = time.monotonic_ns()
            step_start_unix_ns = time.time_ns()
            step_t0 = time.perf_counter()

            fwd_start_unix_ns = time.time_ns()
            fwd_t0 = time.perf_counter()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            torch.cuda.synchronize(device)
            fwd_t1 = time.perf_counter()
            fwd_end_unix_ns = time.time_ns()

            backward_start_unix_ns = time.time_ns()
            bwd_t0 = time.perf_counter()
            loss.backward()
            backward_return_unix_ns = time.time_ns()
            bwd_return_t = time.perf_counter()
            torch.cuda.synchronize(device)
            backward_comm_complete_unix_ns = time.time_ns()
            bwd_t1 = time.perf_counter()

            opt_start_unix_ns = time.time_ns()
            opt_t0 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            opt_t1 = time.perf_counter()
            opt_end_unix_ns = time.time_ns()

            step_t1 = time.perf_counter()
            step_end_unix_ns = time.time_ns()
            step_end_mono_ns = time.monotonic_ns()

            local_step_ms = (step_t1 - step_t0) * 1000.0
            local_forward_ms = (fwd_t1 - fwd_t0) * 1000.0
            local_backward_call_ms = (bwd_return_t - bwd_t0) * 1000.0
            local_backward_sync_ms = (bwd_t1 - bwd_return_t) * 1000.0
            local_backward_total_ms = (bwd_t1 - bwd_t0) * 1000.0
            local_optimizer_ms = (opt_t1 - opt_t0) * 1000.0

            max_step_ms, mean_step_ms = distributed_stat(local_step_ms, device)
            max_forward_ms, mean_forward_ms = distributed_stat(local_forward_ms, device)
            max_backward_ms, mean_backward_ms = distributed_stat(local_backward_total_ms, device)
            max_backward_call_ms, mean_backward_call_ms = distributed_stat(local_backward_call_ms, device)
            max_backward_sync_ms, mean_backward_sync_ms = distributed_stat(local_backward_sync_ms, device)
            max_optimizer_ms, mean_optimizer_ms = distributed_stat(local_optimizer_ms, device)

            duration_s = max(max_step_ms / 1000.0, 1e-9)
            record = {
                "kind": "phase7_step_metrics",
                "step": step,
                "warmup": step < args.warmup_steps,
                "tag": run_tag,
                "repeat_label": repeat_label,
                "phase4_mode": mode,
                "net_burst": net_burst,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "global_batch_size": global_batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "bucket_cap_mb": args.bucket_cap_mb,
                "dtype": args.dtype,
                "param_mb": total_param_mb,
                "ts_start_unix_ns": int(step_start_unix_ns),
                "ts_end_unix_ns": int(step_end_unix_ns),
                "ts_mid_unix_ns": int((step_start_unix_ns + step_end_unix_ns) // 2),
                "step_ms_max": max_step_ms,
                "step_ms_mean": mean_step_ms,
                "forward_ms_max": max_forward_ms,
                "forward_ms_mean": mean_forward_ms,
                "backward_ms_max": max_backward_ms,
                "backward_ms_mean": mean_backward_ms,
                "backward_call_ms_max": max_backward_call_ms,
                "backward_call_ms_mean": mean_backward_call_ms,
                "backward_sync_ms_max": max_backward_sync_ms,
                "backward_sync_ms_mean": mean_backward_sync_ms,
                "optimizer_ms_max": max_optimizer_ms,
                "optimizer_ms_mean": mean_optimizer_ms,
                "steps_per_sec": 1.0 / duration_s,
                "samples_per_sec": global_batch_size / duration_s,
                "loss": float(loss.detach().float().item()),
                "algo": os.environ.get("NCCL_ALGO", ""),
                "proto": os.environ.get("NCCL_PROTO", ""),
            }
            timeline_row = {
                "kind": "phase7_backward_timeline",
                "step": step,
                "rank": rank,
                "worker": worker_name,
                "repeat_label": repeat_label,
                "tag": run_tag,
                "phase4_mode": mode,
                "warmup": step < args.warmup_steps,
                "net_burst": net_burst,
                "step_start_mono_ns": int(step_start_mono_ns),
                "step_end_mono_ns": int(step_end_mono_ns),
                "step_start_unix_ns": int(step_start_unix_ns),
                "forward_start_unix_ns": int(fwd_start_unix_ns),
                "forward_end_unix_ns": int(fwd_end_unix_ns),
                "backward_start_unix_ns": int(backward_start_unix_ns),
                "backward_return_unix_ns": int(backward_return_unix_ns),
                "backward_comm_complete_unix_ns": int(backward_comm_complete_unix_ns),
                "optimizer_start_unix_ns": int(opt_start_unix_ns),
                "optimizer_end_unix_ns": int(opt_end_unix_ns),
                "step_end_unix_ns": int(step_end_unix_ns),
                "local_step_ms": local_step_ms,
                "local_forward_ms": local_forward_ms,
                "local_backward_call_ms": local_backward_call_ms,
                "local_backward_sync_ms": local_backward_sync_ms,
                "local_backward_total_ms": local_backward_total_ms,
                "local_optimizer_ms": local_optimizer_ms,
                "comm_complete_proxy": "backward_comm_complete_unix_ns_after_cuda_synchronize",
            }

            if timeline_handle is not None:
                write_jsonl(timeline_handle, timeline_row)
            if rank == 0:
                print(json.dumps(record, sort_keys=True))
                records.append(record)
                if metrics_handle is not None:
                    write_jsonl(metrics_handle, record)
    finally:
        if timeline_handle is not None:
            timeline_handle.close()
        if metrics_handle is not None:
            metrics_handle.close()

    stage_log("step_loop_complete", records=len(records))

    if validation_path is not None:
        cpu_tensor = flatten_params(ddp_model.module).contiguous().cpu()
        validation_row = {
            "worker": worker_name,
            "rank": rank,
            "tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": mode,
            "world_size": world_size,
            "dtype": args.dtype,
            "model_seed": args.model_seed,
            "numel": int(cpu_tensor.numel()),
            "local_validation_passed": bool(torch.isfinite(cpu_tensor).all().item()),
            "final_sha256": hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest(),
        }
        validation_path.write_text(json.dumps(validation_row, indent=2, sort_keys=True), encoding="utf-8")

    if rank == 0 and output_dir is not None:
        effective = [r for r in records if not r["warmup"]]
        summary = {
            "run_tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": mode,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "effective_steps": len(effective),
            "global_batch_size": global_batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "bucket_cap_mb": args.bucket_cap_mb,
            "dtype": args.dtype,
            "param_mb": total_param_mb,
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "backward_ms_avg": mean(r["backward_ms_max"] for r in effective),
            "backward_ms_p95": percentile([r["backward_ms_max"] for r in effective], 0.95),
            "backward_sync_ms_avg": mean(r["backward_sync_ms_max"] for r in effective),
            "backward_sync_ms_p95": percentile([r["backward_sync_ms_max"] for r in effective], 0.95),
            "samples_per_sec_avg": mean(r["samples_per_sec"] for r in effective),
            "samples_per_sec_p50": percentile([r["samples_per_sec"] for r in effective], 0.50),
            "samples_per_sec_p95": percentile([r["samples_per_sec"] for r in effective], 0.95),
        }
        (output_dir / f"{run_tag}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    stage_log("before_final_barrier")
    dist.barrier()
    stage_log("after_final_barrier")
    dist.destroy_process_group()
    stage_log("after_destroy_process_group")


if __name__ == "__main__":
    main()
