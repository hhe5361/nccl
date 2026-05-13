#!/usr/bin/env python3
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
    if not values:
        return 0.0
    return sum(values) / len(values)


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


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = resolve_dtype(args.dtype)

    # Keep model initialization identical across modes/runs so STOCK vs B4
    # validation reflects communication behavior rather than random init drift.
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)

    run_tag = args.tag or os.environ.get("PHASE4_MODE", "B4").upper()
    output_dir = Path(args.output_dir) if args.output_dir else None
    worker_name = os.environ.get("WORKER_NAME", f"worker_rank{rank}")
    phase4_mode = os.environ.get("PHASE4_MODE", "stock").lower()
    repeat_label = os.environ.get("PHASE4_REPEAT_LABEL", "repeat_01")
    phase4_enable = int(os.environ.get("NCCL_PHASE4_ENABLE", "0"))
    phase4_post_receive_w = float(os.environ.get("NCCL_PHASE4_POST_RECEIVE_W", "0"))
    net_burst = float(args.net_burst)

    if rank == 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    step_metrics_path = output_dir / f"{run_tag}_step_metrics.jsonl" if rank == 0 and output_dir is not None else None
    worker_dir = output_dir / worker_name if output_dir is not None else None
    step_timing_path = worker_dir / f"{run_tag}_worker_step_timing.jsonl" if worker_dir is not None else None
    validation_path = worker_dir / f"{run_tag}_rank_validation.json" if worker_dir is not None else None
    final_stage_path = worker_dir / f"{run_tag}_final_stage.json" if worker_dir is not None else None
    progress_path = worker_dir / f"{run_tag}_progress.json" if worker_dir is not None else None

    if worker_dir is not None:
        worker_dir.mkdir(parents=True, exist_ok=True)

    def stage_log(stage: str, **extra: object) -> None:
        row = {
            "ts_unix_ns": time.time_ns(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "worker": worker_name,
            "tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": phase4_mode,
            "phase4_enable": phase4_enable,
            "phase4_post_receive_w": phase4_post_receive_w,
            "net_burst": net_burst,
            "stage": stage,
        }
        row.update(extra)
        print(f"[phase4-ddp] {json.dumps(row, sort_keys=True)}", file=sys.stderr, flush=True)
        if final_stage_path is not None:
            atomic_write_json(final_stage_path, row)

    def progress_log(phase: str, step: int | None = None, **extra: object) -> None:
        row = {
            "ts_unix_ns": time.time_ns(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "worker": worker_name,
            "tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": phase4_mode,
            "phase4_enable": phase4_enable,
            "phase4_post_receive_w": phase4_post_receive_w,
            "net_burst": net_burst,
            "phase": phase,
        }
        if step is not None:
            row["step"] = step
            row["warmup"] = step < args.warmup_steps
        row.update(extra)
        print(f"[phase4-progress] {json.dumps(row, sort_keys=True)}", file=sys.stderr, flush=True)
        if progress_path is not None:
            atomic_write_json(progress_path, row)

    stage_log("startup")
    progress_log("startup")

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
            f"net_burst={net_burst} repeat_label={repeat_label} param_mb={total_param_mb:.6f}"
        )

    records = []
    stage_log("before_startup_barrier")
    progress_log("before_startup_barrier")
    dist.barrier()
    stage_log("after_startup_barrier")
    progress_log("after_startup_barrier")
    handle = step_metrics_path.open("w", encoding="utf-8") if step_metrics_path is not None else None
    timing_handle = None
    if worker_dir is not None:
        timing_handle = step_timing_path.open("w", encoding="utf-8")

    try:
        for step in range(args.steps):
            progress_log("step_begin", step=step)
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

            progress_log("before_forward", step=step)
            fwd_t0 = time.perf_counter()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            torch.cuda.synchronize(device)
            fwd_t1 = time.perf_counter()
            progress_log("after_forward", step=step, local_forward_ms=(fwd_t1 - fwd_t0) * 1000.0)

            progress_log("before_backward", step=step)
            bwd_t0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            bwd_t1 = time.perf_counter()
            progress_log("after_backward", step=step, local_backward_ms=(bwd_t1 - bwd_t0) * 1000.0)

            progress_log("before_optimizer", step=step)
            opt_t0 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            opt_t1 = time.perf_counter()
            progress_log("after_optimizer", step=step, local_optimizer_ms=(opt_t1 - opt_t0) * 1000.0)

            t1 = time.perf_counter()
            ts_end_unix_ns = time.time_ns()
            step_end_ns = time.monotonic_ns()

            local_ms = (t1 - t0) * 1000.0
            local_forward_ms = (fwd_t1 - fwd_t0) * 1000.0
            local_backward_ms = (bwd_t1 - bwd_t0) * 1000.0
            local_optimizer_ms = (opt_t1 - opt_t0) * 1000.0

            max_step_ms, mean_step_ms = distributed_stat(local_ms, device)
            max_forward_ms, mean_forward_ms = distributed_stat(local_forward_ms, device)
            max_backward_ms, mean_backward_ms = distributed_stat(local_backward_ms, device)
            max_optimizer_ms, mean_optimizer_ms = distributed_stat(local_optimizer_ms, device)

            duration_s = max(max_step_ms / 1000.0, 1e-9)
            steps_per_sec = 1.0 / duration_s
            samples_per_sec = global_batch_size / duration_s

            record = {
                "step": step,
                "tag": run_tag,
                "repeat_label": repeat_label,
                "phase4_mode": phase4_mode,
                "phase4_enable": phase4_enable,
                "phase4_post_receive_w": phase4_post_receive_w,
                "net_burst": net_burst,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "global_batch_size": global_batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "bucket_cap_mb": args.bucket_cap_mb,
                "lr": args.lr,
                "model_seed": args.model_seed,
                "dtype": args.dtype,
                "param_mb": total_param_mb,
                "ts_start_unix_ns": int(ts_start_unix_ns),
                "ts_end_unix_ns": int(ts_end_unix_ns),
                "ts_mid_unix_ns": int((ts_start_unix_ns + ts_end_unix_ns) // 2),
                "step_ms_max": max_step_ms,
                "step_ms_mean": mean_step_ms,
                "forward_ms_max": max_forward_ms,
                "forward_ms_mean": mean_forward_ms,
                "backward_ms_max": max_backward_ms,
                "backward_ms_mean": mean_backward_ms,
                "optimizer_ms_max": max_optimizer_ms,
                "optimizer_ms_mean": mean_optimizer_ms,
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
                    "repeat_label": repeat_label,
                    "phase4_mode": phase4_mode,
                    "phase4_enable": phase4_enable,
                    "phase4_post_receive_w": phase4_post_receive_w,
                    "net_burst": net_burst,
                    "model_seed": args.model_seed,
                    "warmup": step < args.warmup_steps,
                    "start_ns": step_start_ns,
                    "end_ns": step_end_ns,
                    "local_step_ms": local_ms,
                    "local_forward_ms": local_forward_ms,
                    "local_backward_ms": local_backward_ms,
                    "local_optimizer_ms": local_optimizer_ms,
                }
                timing_handle.write(json.dumps(timing_row, sort_keys=True) + "\n")
                timing_handle.flush()
            progress_log("step_complete", step=step, local_step_ms=local_ms)
    finally:
        if handle is not None:
            handle.close()
        if timing_handle is not None:
            timing_handle.close()

    stage_log("step_loop_complete", records=len(records))
    progress_log("step_loop_complete", step=args.steps - 1 if args.steps > 0 else None, records=len(records))

    if validation_path is not None:
        stage_log("before_validation_write")
        progress_log("before_validation_write")
        cpu_tensor = flatten_params(ddp_model.module).contiguous().cpu()
        tensor_bytes = cpu_tensor.numpy().tobytes()
        validation_row = {
            "worker": worker_name,
            "rank": rank,
            "tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": phase4_mode,
            "phase4_enable": phase4_enable,
            "phase4_post_receive_w": phase4_post_receive_w,
            "net_burst": net_burst,
            "world_size": world_size,
            "dtype": args.dtype,
            "model_seed": args.model_seed,
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
        stage_log(
            "after_validation_write",
            final_sha256=validation_row["final_sha256"],
            local_validation_passed=validation_row["local_validation_passed"],
        )
        progress_log(
            "after_validation_write",
            final_sha256=validation_row["final_sha256"],
            local_validation_passed=validation_row["local_validation_passed"],
        )

    if rank == 0 and output_dir is not None:
        stage_log("before_summary_write")
        progress_log("before_summary_write")
        effective = [r for r in records if not r["warmup"]]
        summary = {
            "run_tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": phase4_mode,
            "phase4_enable": phase4_enable,
            "phase4_post_receive_w": phase4_post_receive_w,
            "net_burst": net_burst,
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
            "model_seed": args.model_seed,
            "dtype": args.dtype,
            "param_mb": total_param_mb,
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p50": percentile([r["step_ms_max"] for r in effective], 0.50),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "forward_ms_avg": mean(r["forward_ms_max"] for r in effective),
            "forward_ms_p50": percentile([r["forward_ms_max"] for r in effective], 0.50),
            "forward_ms_p95": percentile([r["forward_ms_max"] for r in effective], 0.95),
            "backward_ms_avg": mean(r["backward_ms_max"] for r in effective),
            "backward_ms_p50": percentile([r["backward_ms_max"] for r in effective], 0.50),
            "backward_ms_p95": percentile([r["backward_ms_max"] for r in effective], 0.95),
            "optimizer_ms_avg": mean(r["optimizer_ms_max"] for r in effective),
            "optimizer_ms_p50": percentile([r["optimizer_ms_max"] for r in effective], 0.50),
            "optimizer_ms_p95": percentile([r["optimizer_ms_max"] for r in effective], 0.95),
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
        atomic_write_json(summary_path, summary)
        stage_log("after_summary_write", summary_path=str(summary_path), effective_steps=len(effective))
        progress_log("after_summary_write", summary_path=str(summary_path), effective_steps=len(effective))

    stage_log("before_final_barrier")
    progress_log("before_final_barrier")
    try:
        dist.barrier()
    except Exception as exc:
        stage_log("final_barrier_failed", error=repr(exc))
        progress_log("final_barrier_failed", error=repr(exc))
        raise
    stage_log("after_final_barrier")
    progress_log("after_final_barrier")

    stage_log("before_destroy_process_group")
    progress_log("before_destroy_process_group")
    try:
        dist.destroy_process_group()
    except Exception as exc:
        stage_log("destroy_process_group_failed", error=repr(exc))
        progress_log("destroy_process_group_failed", error=repr(exc))
        raise
    stage_log("after_destroy_process_group")
    progress_log("after_destroy_process_group")
    stage_log("before_python_exit")
    progress_log("before_python_exit")


if __name__ == "__main__":
    main()
