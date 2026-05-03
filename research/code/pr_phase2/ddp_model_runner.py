#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyStack(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def debug_log(rank: int, stage: str, **fields) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    suffix = f" {payload}" if payload else ""
    print(f"[phase2-ddp][{ts}][rank={rank}] {stage}{suffix}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PR Phase2 tiny DDP training-like runner")
    parser.add_argument("--collective", choices=["allreduce"], required=True)
    parser.add_argument("--algorithm", default="ring")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--payload-mb", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bucket-cap-mb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def model_probe(model: nn.Module) -> dict:
    flat = torch.cat([param.detach().reshape(-1) for param in model.parameters()])
    sample_len = min(8, flat.numel())
    sample = flat[:sample_len].cpu().tolist()
    return {
        "sum": float(flat.sum().item()),
        "mean": float(flat.mean().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "sample": [float(v) for v in sample],
    }


def verify_model_sync(model: nn.Module, device: torch.device) -> dict:
    flat = torch.cat([param.detach().reshape(-1) for param in model.parameters()])
    local_sum = flat.sum()
    local_mean = flat.mean()
    local_min = flat.min()
    local_max = flat.max()

    checks = {
        "sum": local_sum.clone(),
        "mean": local_mean.clone(),
        "min": local_min.clone(),
        "max": local_max.clone(),
    }
    mins = {name: tensor.clone() for name, tensor in checks.items()}
    maxs = {name: tensor.clone() for name, tensor in checks.items()}
    for tensor in mins.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    for tensor in maxs.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)

    max_err = 0.0
    for name in checks:
        err = float((maxs[name] - mins[name]).abs().item())
        max_err = max(max_err, err)

    all_finite = bool(torch.isfinite(flat).all().item())
    return {
        "ok": all_finite and max_err < 1e-5,
        "all_finite": all_finite,
        "max_cross_rank_err": max_err,
    }


def estimate_payload_mb(model: nn.Module) -> float:
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024.0 * 1024.0)


def make_batch(batch_size: int, hidden_dim: int, step_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = 0.01 * float(step_idx)
    x = torch.full((batch_size, hidden_dim), base, device=device, dtype=torch.float32)
    target = torch.full((batch_size, hidden_dim), base + 0.5, device=device, dtype=torch.float32)
    return x, target


def main() -> None:
    args = parse_args()
    if args.collective != "allreduce":
        raise ValueError(f"Unsupported collective for phase2 DDP model runner: {args.collective}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    configured_w = os.environ.get("NCCL_PHASE1_INFLIGHT_W")

    debug_log(rank, "startup", local_rank=local_rank, world_size=world_size, configured_w=configured_w, output_dir=args.output_dir)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    debug_log(rank, "process_group_initialized")

    model = TinyStack(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device=device, dtype=torch.float32)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"rank{rank:02d}_step_trace.jsonl"
    summary_path = output_dir / f"rank{rank:02d}_summary.json"

    effective_payload_mb = estimate_payload_mb(ddp_model.module)
    total_params = sum(param.numel() for param in ddp_model.module.parameters())
    global_batch_size = args.batch_size * world_size
    step_records = []
    verification = []

    dist.barrier()
    debug_log(rank, "startup_barrier_complete")

    with trace_path.open("w", encoding="utf-8") as trace_fp:
        total_steps = args.warmup_steps + args.steps
        for raw_step_idx in range(total_steps):
            phase = "warmup" if raw_step_idx < args.warmup_steps else "steady"
            x, target = make_batch(args.batch_size, args.hidden_dim, raw_step_idx, device)

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start_ns = time.time_ns()

            out = ddp_model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize(device)
            end_ns = time.time_ns()
            duration_ms = (end_ns - start_ns) / 1_000_000.0
            duration_s = max(duration_ms / 1000.0, 1e-9)
            steps_per_sec = 1.0 / duration_s
            samples_per_sec = float(global_batch_size) / duration_s

            verify = verify_model_sync(ddp_model.module, device)
            verification.append(bool(verify["ok"]))
            probe = model_probe(ddp_model.module)

            record = {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "collective": args.collective,
                "algorithm": args.algorithm,
                "configured_w": configured_w,
                "phase": phase,
                "step_index": raw_step_idx if phase == "warmup" else raw_step_idx - args.warmup_steps,
                "duration_ms": duration_ms,
                "ts_start_unix_ns": start_ns,
                "ts_end_unix_ns": end_ns,
                "payload_mb": effective_payload_mb,
                "requested_payload_mb": float(args.payload_mb),
                "total_params": total_params,
                "global_batch_size": global_batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "bucket_cap_mb": args.bucket_cap_mb,
                "loss": float(loss.detach().item()),
                "steps_per_sec": steps_per_sec,
                "samples_per_sec": samples_per_sec,
                "verification_ok": verify["ok"],
                "verification_detail": verify,
                "probe": probe,
            }
            trace_fp.write(json.dumps(record) + "\n")
            step_records.append(record)

    debug_log(rank, "step_loop_complete", records=len(step_records), verification_all_ok=all(verification))

    steady = [row["duration_ms"] for row in step_records if row["phase"] == "steady"]
    summary = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "collective": args.collective,
        "algorithm": args.algorithm,
        "configured_w": configured_w,
        "payload_mb": effective_payload_mb,
        "requested_payload_mb": float(args.payload_mb),
        "total_params": total_params,
        "global_batch_size": global_batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "bucket_cap_mb": args.bucket_cap_mb,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "verification_all_ok": all(verification),
        "steady_avg_ms": sum(steady) / len(steady) if steady else None,
        "steady_min_ms": min(steady) if steady else None,
        "steady_max_ms": max(steady) if steady else None,
        "steady_steps_per_sec_avg": (sum(1.0 / max(v / 1000.0, 1e-9) for v in steady) / len(steady)) if steady else None,
        "steady_samples_per_sec_avg": (sum(float(global_batch_size) / max(v / 1000.0, 1e-9) for v in steady) / len(steady)) if steady else None,
        "trace_file": str(trace_path),
    }
    debug_log(rank, "before_summary_write", summary_path=summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    debug_log(rank, "after_summary_write", steady_count=len(steady))

    try:
        if dist.is_initialized():
            debug_log(rank, "before_destroy_process_group")
            dist.destroy_process_group()
            debug_log(rank, "after_destroy_process_group")
    except Exception:
        debug_log(rank, "destroy_process_group_failed")
    sys.stdout.flush()
    sys.stderr.flush()
    debug_log(rank, "before_force_exit")
    os._exit(0)


if __name__ == "__main__":
    main()
