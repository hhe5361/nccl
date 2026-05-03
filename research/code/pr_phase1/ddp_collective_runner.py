#!/usr/bin/env python3
import argparse
import json
import os
import sys
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
    parser.add_argument("--buckets-per-step", type=int, default=4)
    parser.add_argument("--bucket-ready-gap-ms", type=float, default=0.5)
    parser.add_argument("--rank-jitter-ms", type=float, default=0.25)
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


def split_numel_into_buckets(numel: int, buckets_per_step: int, world_size: int, collective: str) -> list[int]:
    buckets = max(1, buckets_per_step)
    if collective == "alltoall":
        per_peer = max(1, numel // world_size)
        base = per_peer // buckets
        rem = per_peer % buckets
        sizes = [(base + (1 if idx < rem else 0)) * world_size for idx in range(buckets)]
    else:
        base = numel // buckets
        rem = numel % buckets
        sizes = [base + (1 if idx < rem else 0) for idx in range(buckets)]
    return [size for size in sizes if size > 0]


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


def verify_outputs(collective: str, outputs: list[torch.Tensor], rank: int, world_size: int) -> dict:
    if not outputs:
        return {"ok": True, "max_abs_err": 0.0}
    if collective == "allreduce":
        details = [verify_allreduce(tensor, world_size) for tensor in outputs]
        max_abs = max(float(item["max_abs_err"]) for item in details)
        return {
            "ok": all(bool(item["ok"]) for item in details),
            "expected": details[0]["expected"],
            "max_abs_err": max_abs,
            "bucket_count": len(outputs),
        }
    details = [verify_alltoall(tensor, rank, world_size) for tensor in outputs]
    max_abs = max(float(item["max_abs_err"]) for item in details)
    return {
        "ok": all(bool(item["ok"]) for item in details),
        "max_abs_err": max_abs,
        "bucket_count": len(outputs),
    }


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


def build_probe_from_outputs(outputs: list[torch.Tensor]) -> dict:
    if not outputs:
        return {"sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "sample": []}
    total_sum = 0.0
    total_numel = 0
    min_val = None
    max_val = None
    sample: list[float] = []
    for tensor in outputs:
        total_sum += float(tensor.sum().item())
        total_numel += int(tensor.numel())
        cur_min = float(tensor.min().item())
        cur_max = float(tensor.max().item())
        min_val = cur_min if min_val is None else min(min_val, cur_min)
        max_val = cur_max if max_val is None else max(max_val, cur_max)
        if len(sample) < 8:
            need = 8 - len(sample)
            sample.extend(float(v) for v in tensor.reshape(-1)[:need].detach().cpu().tolist())
    return {
        "sum": total_sum,
        "mean": total_sum / max(1, total_numel),
        "min": float(min_val if min_val is not None else 0.0),
        "max": float(max_val if max_val is not None else 0.0),
        "sample": sample,
    }


def maybe_wait_for_bucket(step_idx: int, bucket_idx: int, rank: int, world_size: int, base_gap_ms: float, rank_jitter_ms: float) -> None:
    delay_ms = max(0.0, base_gap_ms)
    if world_size > 1 and rank_jitter_ms > 0:
        delay_ms += rank_jitter_ms * (rank / float(world_size - 1))
    delay_ms += 0.05 * float((step_idx + bucket_idx) % 2)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


def run_step(
    collective: str,
    rank: int,
    world_size: int,
    bucket_numels: list[int],
    device: torch.device,
    step_idx: int,
    bucket_ready_gap_ms: float,
    rank_jitter_ms: float,
):
    outputs: list[torch.Tensor] = []
    torch.cuda.synchronize(device)
    start_ns = time.time_ns()

    for bucket_idx, numel in enumerate(bucket_numels):
        maybe_wait_for_bucket(step_idx, bucket_idx, rank, world_size, bucket_ready_gap_ms, rank_jitter_ms)
        if collective == "allreduce":
            tensor = make_allreduce_tensor(numel, rank, device)
            dist.all_reduce(tensor)
            outputs.append(tensor)
            continue
        if collective == "alltoall":
            input_tensor = make_alltoall_input(numel, rank, world_size, device)
            output_tensor = torch.empty_like(input_tensor)
            dist.all_to_all_single(output_tensor, input_tensor)
            outputs.append(output_tensor)
            continue
        raise ValueError(f"Unsupported collective: {collective}")

    torch.cuda.synchronize(device)
    end_ns = time.time_ns()
    return outputs, start_ns, end_ns


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    configured_w = os.environ.get("NCCL_PHASE1_INFLIGHT_W")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"rank{rank:02d}_step_trace.jsonl"
    summary_path = output_dir / f"rank{rank:02d}_summary.json"

    numel = bytes_to_elems(args.payload_mb, torch.float32, world_size, args.collective)
    bucket_numels = split_numel_into_buckets(numel, args.buckets_per_step, world_size, args.collective)
    step_records = []
    verification = []

    # One startup barrier is enough to align the run. We intentionally avoid
    # per-step barriers so each rank can flow into the next bucket sequence
    # without creating fully synchronized burst launches every step.
    dist.barrier()

    with trace_path.open("w", encoding="utf-8") as trace_fp:
        total_steps = args.warmup_steps + args.steps
        for step_idx in range(total_steps):
            phase = "warmup" if step_idx < args.warmup_steps else "steady"
            outputs, start_ns, end_ns = run_step(
                args.collective,
                rank,
                world_size,
                bucket_numels,
                device,
                step_idx,
                args.bucket_ready_gap_ms,
                args.rank_jitter_ms,
            )
            duration_ms = (end_ns - start_ns) / 1_000_000.0

            verify = verify_outputs(args.collective, outputs, rank, world_size)
            verification.append(verify["ok"])
            probe = build_probe_from_outputs(outputs)

            record = {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "collective": args.collective,
                "algorithm": args.algorithm,
                "configured_w": configured_w,
                "phase": phase,
                "step_index": step_idx if phase == "warmup" else step_idx - args.warmup_steps,
                "duration_ms": duration_ms,
                "ts_start_unix_ns": start_ns,
                "ts_end_unix_ns": end_ns,
                "payload_mb": args.payload_mb,
                "numel": numel,
                "bucket_numels": bucket_numels,
                "buckets_per_step": len(bucket_numels),
                "bucket_ready_gap_ms": args.bucket_ready_gap_ms,
                "rank_jitter_ms": args.rank_jitter_ms,
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
        "configured_w": configured_w,
        "payload_mb": args.payload_mb,
        "numel": numel,
        "bucket_numels": bucket_numels,
        "buckets_per_step": len(bucket_numels),
        "bucket_ready_gap_ms": args.bucket_ready_gap_ms,
        "rank_jitter_ms": args.rank_jitter_ms,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "verification_all_ok": all(verification),
        "steady_avg_ms": sum(steady) / len(steady) if steady else None,
        "steady_min_ms": min(steady) if steady else None,
        "steady_max_ms": max(steady) if steady else None,
        "trace_file": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Keep teardown minimal. Some runs completed all steps and wrote summaries,
    # then stalled during process-group teardown or launcher shutdown.
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
