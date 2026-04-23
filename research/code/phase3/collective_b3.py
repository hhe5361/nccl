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
    parser = argparse.ArgumentParser(description="Phase 3 B3 collective workload")
    parser.add_argument(
        "--collective",
        choices=["allreduce", "allgather", "reducescatter", "alltoall"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--payload-mb", type=int, default=32, help="Per-rank logical payload in MiB")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--output-dir", default=os.environ.get("PHASE3_OUTPUT_DIR"))
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


def expected_sum(world_size: int, step: int) -> float:
    return (world_size * (world_size + 1) / 2.0) + world_size * step * 0.001


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
    run_tag = args.tag or os.environ.get("PHASE3_MODE", "B3").upper()
    output_dir = Path(args.output_dir) if args.output_dir else None
    worker_name = os.environ.get("WORKER_NAME", f"worker_rank{rank}")
    if rank == 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    step_metrics_path = output_dir / f"{run_tag}_step_metrics.jsonl" if rank == 0 and output_dir is not None else None
    worker_dir = output_dir / worker_name if output_dir is not None else None
    step_timing_path = worker_dir / f"{run_tag}_worker_step_timing.jsonl" if worker_dir is not None else None

    traffic_bytes_est_rank = 0.0
    input_tensor = None
    output_tensor = None

    if args.collective == "allreduce":
        elems = max(1, payload_bytes // elem_size)
        payload_bytes = elems * elem_size
        traffic_bytes_est_rank = payload_bytes * (2.0 * (world_size - 1) / world_size)
        input_tensor = torch.empty((elems,), device=device, dtype=dtype)
        output_tensor = input_tensor
    elif args.collective == "allgather":
        elems_total = max(1, payload_bytes // elem_size)
        elems_per_rank = max(1, elems_total // world_size)
        elems_total = elems_per_rank * world_size
        payload_bytes = elems_total * elem_size
        traffic_bytes_est_rank = payload_bytes * ((world_size - 1) / world_size)
        input_tensor = torch.empty((elems_per_rank,), device=device, dtype=dtype)
        output_tensor = torch.empty((elems_total,), device=device, dtype=dtype)
    elif args.collective == "reducescatter":
        elems_total = max(1, payload_bytes // elem_size)
        elems_per_rank = max(1, elems_total // world_size)
        elems_total = elems_per_rank * world_size
        payload_bytes = elems_total * elem_size
        traffic_bytes_est_rank = payload_bytes * ((world_size - 1) / world_size)
        input_tensor = torch.empty((elems_total,), device=device, dtype=dtype)
        output_tensor = torch.empty((elems_per_rank,), device=device, dtype=dtype)
    else:
        elems_total = max(1, payload_bytes // elem_size)
        elems_per_peer = max(1, elems_total // world_size)
        elems_total = elems_per_peer * world_size
        payload_bytes = elems_total * elem_size
        traffic_bytes_est_rank = float(payload_bytes)
        input_tensor = torch.empty((elems_total,), device=device, dtype=dtype)
        output_tensor = torch.empty_like(input_tensor)

    payload_mb = payload_bytes / (1024.0 * 1024.0)
    traffic_mb_est_rank = traffic_bytes_est_rank / (1024.0 * 1024.0)

    if rank == 0:
        print(
            f"phase3_b3 collective={args.collective} steps={args.steps} warmup_steps={args.warmup_steps} "
            f"payload_mb={payload_mb:.3f} dtype={args.dtype} world_size={world_size} "
            f"phase3_mode={os.environ.get('PHASE3_MODE', 'b3')} traffic_mb_est_rank={traffic_mb_est_rank:.3f}"
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
            base_value = float(rank + 1 + step * 0.001)
            input_tensor.fill_(base_value)

            torch.cuda.synchronize(device)
            step_start_ns = time.monotonic_ns()
            ts_start_unix_ns = time.time_ns()
            t0 = time.perf_counter()

            if args.collective == "allreduce":
                dist.all_reduce(input_tensor)
            elif args.collective == "allgather":
                dist.all_gather_into_tensor(output_tensor, input_tensor)
            elif args.collective == "reducescatter":
                dist.reduce_scatter_tensor(output_tensor, input_tensor)
            else:
                dist.all_to_all_single(output_tensor, input_tensor)

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

            if args.collective == "allreduce":
                sample = float(input_tensor[0].float().item())
                expected = expected_sum(world_size, step)
            elif args.collective == "allgather":
                sample = float(output_tensor[0].float().item())
                expected = 1.0 + step * 0.001
            elif args.collective == "reducescatter":
                sample = float(output_tensor[0].float().item())
                expected = expected_sum(world_size, step)
            else:
                sample = float(output_tensor[0].float().item())
                expected = 1.0 + step * 0.001

            if abs(sample - expected) > 1e-2:
                raise RuntimeError(
                    f"rank {rank}: validation failed for {args.collective} at step {step}: got {sample}, expected {expected}"
                )

            step_s = max(float(max_times[0].item()), 1e-9) / 1000.0
            collective_gbps_est = (traffic_bytes_est_rank * 8.0) / step_s / 1e9

            record = {
                "step": step,
                "tag": run_tag,
                "phase3_mode": os.environ.get("PHASE3_MODE", "b3"),
                "phase2_b2_enable": int(os.environ.get("NCCL_PHASE2_B2_ENABLE", "0")),
                "phase3_b3_enable": int(os.environ.get("NCCL_PHASE3_B3_ENABLE", "0")),
                "collective": args.collective,
                "world_size": world_size,
                "payload_mb": payload_mb,
                "traffic_mb_est_rank": traffic_mb_est_rank,
                "ts_start_unix_ns": int(ts_start_unix_ns),
                "ts_end_unix_ns": int(ts_end_unix_ns),
                "ts_mid_unix_ns": int((ts_start_unix_ns + ts_end_unix_ns) // 2),
                "step_ms_max": float(max_times[0].item()),
                "step_ms_mean": float(mean_times[0].item()),
                "collective_gbps_est": collective_gbps_est,
                "warmup": step < args.warmup_steps,
                "algo": os.environ.get("NCCL_ALGO", ""),
                "proto": os.environ.get("NCCL_PROTO", ""),
                "rack_map_file": os.environ.get("NCCL_RACK_MAP_FILE", ""),
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
                    "phase3_mode": os.environ.get("PHASE3_MODE", "b3"),
                    "collective": args.collective,
                    "warmup": step < args.warmup_steps,
                    "start_ns": step_start_ns,
                    "end_ns": step_end_ns,
                    "local_step_ms": local_ms,
                }
                timing_handle.write(json.dumps(timing_row, sort_keys=True) + "\n")
                timing_handle.flush()

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        if handle is not None:
            handle.close()
        if timing_handle is not None:
            timing_handle.close()

    if rank == 0 and output_dir is not None:
        effective = [r for r in records if not r["warmup"]]
        summary = {
            "run_tag": run_tag,
            "phase3_mode": os.environ.get("PHASE3_MODE", "b3"),
            "phase2_b2_enable": int(os.environ.get("NCCL_PHASE2_B2_ENABLE", "0")),
            "phase3_b3_enable": int(os.environ.get("NCCL_PHASE3_B3_ENABLE", "0")),
            "collective": args.collective,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "effective_steps": len(effective),
            "payload_mb": payload_mb,
            "traffic_mb_est_rank": traffic_mb_est_rank,
            "dtype": args.dtype,
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "rack_map_file": os.environ.get("NCCL_RACK_MAP_FILE", ""),
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p50": percentile([r["step_ms_max"] for r in effective], 0.50),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "collective_gbps_avg": mean(r["collective_gbps_est"] for r in effective),
            "collective_gbps_p50": percentile([r["collective_gbps_est"] for r in effective], 0.50),
            "collective_gbps_p95": percentile([r["collective_gbps_est"] for r in effective], 0.95),
        }
        summary_path = output_dir / f"{run_tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
