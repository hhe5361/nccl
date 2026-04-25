#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Appendix 1 semantic equivalence check for NCCL collectives")
    parser.add_argument("--collective", choices=["allreduce", "alltoall"], required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--payload-mb", type=int, default=32, help="Per-rank logical payload in MiB")
    parser.add_argument("--dtype", choices=["int32"], default="int32")
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--output-dir", default=os.environ.get("APPENDIX1_OUTPUT_DIR"))
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


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


def sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(cpu.numpy())).hexdigest()


def first_values(tensor: torch.Tensor, count: int = 8) -> List[int]:
    cpu = tensor.detach().cpu().contiguous().view(-1)
    return [int(x) for x in cpu[:count].tolist()]


def build_allreduce_input(step: int, rank: int, elems: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(elems, device=device, dtype=torch.int32)
    return base + rank * 10007 + step * 17


def build_allreduce_expected(step: int, world_size: int, elems: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(elems, device=device, dtype=torch.int64) * world_size
    rank_sum = world_size * (world_size - 1) // 2
    expected = base + rank_sum * 10007 + world_size * step * 17
    return expected.to(torch.int32)


def build_alltoall_input(step: int, rank: int, world_size: int, elems_per_peer: int, device: torch.device) -> torch.Tensor:
    chunks = []
    local = torch.arange(elems_per_peer, device=device, dtype=torch.int32)
    for peer in range(world_size):
        chunk = local + rank * 1000003 + peer * 1009 + step * 31
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)


def build_alltoall_expected(step: int, rank: int, world_size: int, elems_per_peer: int, device: torch.device) -> torch.Tensor:
    chunks = []
    local = torch.arange(elems_per_peer, device=device, dtype=torch.int32)
    for sender in range(world_size):
        chunk = local + sender * 1000003 + rank * 1009 + step * 31
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)


def gather_rank_rows(local_row: Dict) -> List[Dict]:
    gathered: List[Dict] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_row)
    return gathered


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    worker_name = os.environ.get("WORKER_NAME", f"worker{rank:02d}")
    mode_name = (args.tag or os.environ.get("APPENDIX1_MODE") or os.environ.get("PHASE3_MODE") or "RUN").upper()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    elem_size = torch.tensor([], dtype=torch.int32).element_size()
    payload_bytes = args.payload_mb * 1024 * 1024
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.collective == "allreduce":
        elems = max(1, payload_bytes // elem_size)
        payload_bytes = elems * elem_size
        input_tensor = torch.empty((elems,), device=device, dtype=torch.int32)
        output_tensor = input_tensor
    else:
        elems_total = max(1, payload_bytes // elem_size)
        elems_per_peer = max(1, elems_total // world_size)
        elems_total = elems_per_peer * world_size
        payload_bytes = elems_total * elem_size
        input_tensor = torch.empty((elems_total,), device=device, dtype=torch.int32)
        output_tensor = torch.empty_like(input_tensor)

    payload_mb = payload_bytes / (1024.0 * 1024.0)
    worker_dir = output_dir / worker_name if output_dir is not None else None
    rank_path = worker_dir / f"{mode_name}_rank_validation.jsonl" if worker_dir is not None else None
    summary_path = output_dir / f"{mode_name}_summary.json" if output_dir is not None and rank == 0 else None
    step_metrics_path = output_dir / f"{mode_name}_step_metrics.jsonl" if output_dir is not None and rank == 0 else None

    if rank == 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if worker_dir is not None:
        worker_dir.mkdir(parents=True, exist_ok=True)

    rank_rows: List[dict] = []
    rank0_step_rows: List[dict] = []

    dist.barrier()
    for step in range(args.steps):
        if args.collective == "allreduce":
            input_tensor.copy_(build_allreduce_input(step, rank, input_tensor.numel(), device))
            expected = build_allreduce_expected(step, world_size, input_tensor.numel(), device)
        else:
            input_tensor.copy_(build_alltoall_input(step, rank, world_size, input_tensor.numel() // world_size, device))
            expected = build_alltoall_expected(step, rank, world_size, input_tensor.numel() // world_size, device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        ts_start_unix_ns = time.time_ns()

        if args.collective == "allreduce":
            dist.all_reduce(output_tensor)
        else:
            dist.all_to_all_single(output_tensor, input_tensor)

        torch.cuda.synchronize(device)
        ts_end_unix_ns = time.time_ns()
        t1 = time.perf_counter()

        mismatch = output_tensor.ne(expected)
        mismatch_count = int(mismatch.sum().item())
        max_abs_diff = int((output_tensor.to(torch.int64) - expected.to(torch.int64)).abs().max().item()) if output_tensor.numel() else 0
        actual_sha = sha256_tensor(output_tensor)
        expected_sha = sha256_tensor(expected)
        valid = mismatch_count == 0 and actual_sha == expected_sha

        local_ms = (t1 - t0) * 1000.0
        local_time = torch.tensor([local_ms], device=device, dtype=torch.float64)
        max_time = local_time.clone()
        mean_time = local_time.clone()
        dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
        dist.all_reduce(mean_time, op=dist.ReduceOp.SUM)
        mean_time /= world_size

        local_row = {
            "step": step,
            "worker": worker_name,
            "rank": rank,
            "mode": mode_name,
            "collective": args.collective,
            "payload_mb": payload_mb,
            "warmup": step < args.warmup_steps,
            "step_ms_local": local_ms,
            "step_ms_max": float(max_time.item()),
            "step_ms_mean": float(mean_time.item()),
            "ts_start_unix_ns": int(ts_start_unix_ns),
            "ts_end_unix_ns": int(ts_end_unix_ns),
            "ts_mid_unix_ns": int((ts_start_unix_ns + ts_end_unix_ns) // 2),
            "valid": valid,
            "mismatch_count": mismatch_count,
            "max_abs_diff": max_abs_diff,
            "actual_sha256": actual_sha,
            "expected_sha256": expected_sha,
            "actual_head": first_values(output_tensor),
            "expected_head": first_values(expected),
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "phase3_b3_enable": int(os.environ.get("NCCL_PHASE3_B3_ENABLE", "0")),
        }
        rank_rows.append(local_row)

        gathered_rows = gather_rank_rows(local_row)
        if rank == 0:
            rank0_step_rows.append(
                {
                    "step": step,
                    "mode": mode_name,
                    "collective": args.collective,
                    "payload_mb": payload_mb,
                    "warmup": step < args.warmup_steps,
                    "step_ms_max": float(max_time.item()),
                    "step_ms_mean": float(mean_time.item()),
                    "all_valid": all(bool(row["valid"]) for row in gathered_rows),
                    "worker_sha256": {row["worker"]: row["actual_sha256"] for row in gathered_rows},
                }
            )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    if rank_path is not None:
        write_jsonl(rank_path, rank_rows)

    if rank == 0 and output_dir is not None:
        effective = [row for row in rank0_step_rows if not row["warmup"]]
        summary = {
            "mode": mode_name,
            "collective": args.collective,
            "payload_mb": payload_mb,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "effective_steps": len(effective),
            "step_ms_avg": mean(row["step_ms_max"] for row in effective),
            "step_ms_p95": percentile([row["step_ms_max"] for row in effective], 0.95),
            "all_steps_valid": all(bool(row["all_valid"]) for row in rank0_step_rows),
            "invalid_steps": [int(row["step"]) for row in rank0_step_rows if not row["all_valid"]],
            "algo": os.environ.get("NCCL_ALGO", ""),
            "proto": os.environ.get("NCCL_PROTO", ""),
            "phase3_b3_enable": int(os.environ.get("NCCL_PHASE3_B3_ENABLE", "0")),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_jsonl(step_metrics_path, rank0_step_rows)
        print(json.dumps(summary, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
