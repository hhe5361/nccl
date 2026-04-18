#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 B1 AllToAll workload")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--payload-mb", type=int, default=64, help="Per-rank total send payload in MiB")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
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

    payload_bytes = args.payload_mb * 1024 * 1024
    elems_total = payload_bytes // elem_size
    elems_per_peer = max(1, elems_total // world_size)
    elems_total = elems_per_peer * world_size
    payload_bytes = elems_total * elem_size
    payload_mb = payload_bytes / (1024.0 * 1024.0)

    run_tag = args.tag or f"W{os.environ.get('NCCL_PHASE1_STATIC_W', 'unset')}"
    output_dir = Path(args.output_dir) if args.output_dir else None
    step_metrics_path = None
    if rank == 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        step_metrics_path = output_dir / f"{run_tag}_step_metrics.jsonl"

    if rank == 0:
        print(
            f"phase1_b1 alltoall steps={args.steps} warmup_steps={args.warmup_steps} "
            f"payload_mb={payload_mb:.3f} dtype={args.dtype} world_size={world_size} "
            f"elems_per_peer={elems_per_peer} run_tag={run_tag}"
        )

    records = []
    handle = step_metrics_path.open("w", encoding="utf-8") if step_metrics_path is not None else None
    dist.barrier()
    try:
        for step in range(args.steps):
            input_tensor = torch.full(
                (elems_total,),
                fill_value=float(rank + 1 + step * 0.001),
                device=device,
                dtype=dtype,
            )
            output_tensor = torch.empty_like(input_tensor)

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dist.all_to_all_single(output_tensor, input_tensor)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            local_ms = (t1 - t0) * 1000.0
            local_times = torch.tensor([local_ms], device=device, dtype=torch.float64)
            max_times = local_times.clone()
            mean_times = local_times.clone()
            dist.all_reduce(max_times, op=dist.ReduceOp.MAX)
            dist.all_reduce(mean_times, op=dist.ReduceOp.SUM)
            mean_times /= world_size

            expected0 = 1.0 + step * 0.001
            sample0 = float(output_tensor[0].float().item())
            if abs(sample0 - expected0) > 1e-2:
                raise RuntimeError(
                    f"rank {rank}: validation failed at step {step}: got {sample0}, expected {expected0}"
                )

            step_s = max(float(max_times[0].item()), 1e-9) / 1000.0
            # Aggregate network volume approximation: each rank sends payload_bytes and receives payload_bytes.
            agg_bytes_est = payload_bytes * world_size
            agg_gbps_est = (agg_bytes_est * 8.0) / step_s / 1e9

            record = {
                "step": step,
                "tag": run_tag,
                "world_size": world_size,
                "payload_mb": payload_mb,
                "elems_per_peer": elems_per_peer,
                "step_ms_max": float(max_times[0].item()),
                "step_ms_mean": float(mean_times[0].item()),
                "alltoall_gbps_est": agg_gbps_est,
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
            "payload_mb": payload_mb,
            "dtype": args.dtype,
            "phase1_w": int(os.environ.get("NCCL_PHASE1_STATIC_W", "0")),
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p50": percentile([r["step_ms_max"] for r in effective], 0.50),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "alltoall_gbps_avg": mean(r["alltoall_gbps_est"] for r in effective),
            "alltoall_gbps_p50": percentile([r["alltoall_gbps_est"] for r in effective], 0.50),
            "alltoall_gbps_p95": percentile([r["alltoall_gbps_est"] for r in effective], 0.95),
        }
        summary_path = output_dir / f"{run_tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
