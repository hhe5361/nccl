#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase8 GPT-small random-token DDP workload")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--output-dir", default=os.environ.get("PHASE4_OUTPUT_DIR"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bucket-cap-mb", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model-seed", type=int, default=20260504)
    parser.add_argument("--net-burst", type=float, default=0.0)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


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


def distributed_stat(local_value: float, device: torch.device) -> tuple[float, float]:
    values = torch.tensor([local_value], device=device, dtype=torch.float64)
    max_values = values.clone()
    mean_values = values.clone()
    dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
    dist.all_reduce(mean_values, op=dist.ReduceOp.SUM)
    mean_values /= dist.get_world_size()
    return float(max_values[0].item()), float(mean_values[0].item())


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(f"n_embd={cfg.n_embd} must be divisible by n_head={cfg.n_head}")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], torch.finfo(att.dtype).min)
        att = F.softmax(att.float(), dim=-1).to(dtype=q.dtype)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTSmall(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.size()
        if seq_len > self.cfg.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
        x = self.wte(input_ids) + self.wpe(pos)[None, :, :]
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), labels[:, 1:].contiguous().view(-1))
        return logits, loss


def param_stats_digest(model: nn.Module) -> tuple[str, dict[str, float | int]]:
    device = next(model.parameters()).device
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    numel = 0
    with torch.no_grad():
        for param in model.parameters():
            tensor = param.detach().float()
            totals[0] += tensor.sum(dtype=torch.float64)
            totals[1] += tensor.abs().sum(dtype=torch.float64)
            totals[2] += (tensor * tensor).sum(dtype=torch.float64)
            numel += int(tensor.numel())
    total, abs_total, sq_total = [float(v) for v in totals.cpu().tolist()]
    payload = {
        "numel": numel,
        "sum": total,
        "abs_sum": abs_total,
        "sq_sum": sq_total,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, payload


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

    seq_len = int(os.environ.get("PHASE8_SEQ_LEN", "128"))
    vocab_size = int(os.environ.get("PHASE8_VOCAB_SIZE", "50257"))
    num_heads = int(os.environ.get("PHASE8_NUM_HEADS", "12"))
    dropout = float(os.environ.get("PHASE8_DROPOUT", "0.0"))

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=seq_len,
        n_layer=args.num_layers,
        n_head=num_heads,
        n_embd=args.hidden_dim,
        dropout=dropout,
    )

    run_tag = args.tag or os.environ.get("PHASE4_MODE", "STOCK").upper()
    output_dir = Path(args.output_dir) if args.output_dir else None
    worker_name = os.environ.get("WORKER_NAME", f"worker_rank{rank}")
    phase4_mode = os.environ.get("PHASE4_MODE", "stock").lower()
    repeat_label = os.environ.get("PHASE4_REPEAT_LABEL", "repeat_01")
    phase4_enable = int(os.environ.get("NCCL_PHASE4_ENABLE", "0"))
    phase4_post_receive_w = float(os.environ.get("NCCL_PHASE4_POST_RECEIVE_W", "0"))
    phase6_enable = int(os.environ.get("NCCL_PHASE6_ENABLE", "0"))
    net_burst = float(args.net_burst)

    if output_dir is not None and rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = output_dir / worker_name if output_dir is not None else None
    if worker_dir is not None:
        worker_dir.mkdir(parents=True, exist_ok=True)

    step_metrics_path = output_dir / f"{run_tag}_step_metrics.jsonl" if rank == 0 and output_dir is not None else None
    step_timing_path = worker_dir / f"{run_tag}_worker_step_timing.jsonl" if worker_dir is not None else None
    validation_path = worker_dir / f"{run_tag}_rank_validation.json" if worker_dir is not None else None
    final_stage_path = worker_dir / f"{run_tag}_final_stage.json" if worker_dir is not None else None

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
            "phase6_enable": phase6_enable,
            "net_burst": net_burst,
            "stage": stage,
        }
        row.update(extra)
        print(f"[phase8-gpt-ddp] {json.dumps(row, sort_keys=True)}", file=sys.stderr, flush=True)
        if final_stage_path is not None:
            final_stage_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")

    stage_log("startup", seq_len=seq_len, vocab_size=vocab_size, num_heads=num_heads)

    model = GPTSmall(cfg).to(device=device, dtype=dtype)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        bucket_cap_mb=args.bucket_cap_mb,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=0.01)

    global_batch_size = args.batch_size * world_size
    total_param_bytes = sum(p.numel() * p.element_size() for p in ddp_model.module.parameters())
    total_param_mb = total_param_bytes / (1024.0 * 1024.0)

    if rank == 0:
        print(
            f"phase8 gpt-small ddp steps={args.steps} warmup_steps={args.warmup_steps} "
            f"hidden_dim={args.hidden_dim} num_layers={args.num_layers} heads={num_heads} seq_len={seq_len} "
            f"batch_size={args.batch_size} bucket_cap_mb={args.bucket_cap_mb} dtype={args.dtype} "
            f"world_size={world_size} phase6_enable={phase6_enable} param_mb={total_param_mb:.6f}"
        )

    stage_log("before_startup_barrier")
    dist.barrier()
    stage_log("after_startup_barrier")

    records: list[dict] = []
    handle = step_metrics_path.open("w", encoding="utf-8") if step_metrics_path is not None else None
    timing_handle = step_timing_path.open("w", encoding="utf-8") if step_timing_path is not None else None

    try:
        for step in range(args.steps):
            gen = torch.Generator(device=device)
            gen.manual_seed(args.model_seed + step)
            input_ids = torch.randint(0, vocab_size, (args.batch_size, seq_len), device=device, dtype=torch.long, generator=gen)
            labels = input_ids.clone()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_start_ns = time.monotonic_ns()
            ts_start_unix_ns = time.time_ns()
            t0 = time.perf_counter()

            fwd_t0 = time.perf_counter()
            _, loss = ddp_model(input_ids, labels)
            if loss is None:
                raise RuntimeError("GPT loss was not computed")
            torch.cuda.synchronize(device)
            fwd_t1 = time.perf_counter()

            bwd_t0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            bwd_t1 = time.perf_counter()

            opt_t0 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            opt_t1 = time.perf_counter()

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
            record = {
                "step": step,
                "tag": run_tag,
                "repeat_label": repeat_label,
                "phase4_mode": phase4_mode,
                "phase4_enable": phase4_enable,
                "phase4_post_receive_w": phase4_post_receive_w,
                "phase6_enable": phase6_enable,
                "net_burst": net_burst,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "global_batch_size": global_batch_size,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": num_heads,
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
                "steps_per_sec": 1.0 / duration_s,
                "samples_per_sec": global_batch_size / duration_s,
                "tokens_per_sec": (global_batch_size * seq_len) / duration_s,
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
                    "phase6_enable": phase6_enable,
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
    finally:
        if handle is not None:
            handle.close()
        if timing_handle is not None:
            timing_handle.close()

    stage_log("step_loop_complete", records=len(records))

    if validation_path is not None:
        digest, stats = param_stats_digest(ddp_model.module)
        validation_row = {
            "worker": worker_name,
            "rank": rank,
            "tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": phase4_mode,
            "phase6_enable": phase6_enable,
            "world_size": world_size,
            "dtype": args.dtype,
            "model_seed": args.model_seed,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "batch_size": args.batch_size,
            "bucket_cap_mb": args.bucket_cap_mb,
            "local_validation_passed": True,
            "final_sha256": digest,
            "param_stats": stats,
        }
        validation_path.write_text(json.dumps(validation_row, indent=2, sort_keys=True), encoding="utf-8")

    if rank == 0 and output_dir is not None:
        effective = [r for r in records if not r["warmup"]]
        summary = {
            "run_tag": run_tag,
            "repeat_label": repeat_label,
            "phase4_mode": phase4_mode,
            "phase6_enable": phase6_enable,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "effective_steps": len(effective),
            "batch_size": args.batch_size,
            "global_batch_size": global_batch_size,
            "seq_len": seq_len,
            "tokens_per_global_step": global_batch_size * seq_len,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": num_heads,
            "bucket_cap_mb": args.bucket_cap_mb,
            "lr": args.lr,
            "model_seed": args.model_seed,
            "dtype": args.dtype,
            "param_mb": total_param_mb,
            "step_ms_avg": mean(r["step_ms_max"] for r in effective),
            "step_ms_p50": percentile([r["step_ms_max"] for r in effective], 0.50),
            "step_ms_p95": percentile([r["step_ms_max"] for r in effective], 0.95),
            "forward_ms_avg": mean(r["forward_ms_max"] for r in effective),
            "forward_ms_p95": percentile([r["forward_ms_max"] for r in effective], 0.95),
            "backward_ms_avg": mean(r["backward_ms_max"] for r in effective),
            "backward_ms_p95": percentile([r["backward_ms_max"] for r in effective], 0.95),
            "optimizer_ms_avg": mean(r["optimizer_ms_max"] for r in effective),
            "optimizer_ms_p95": percentile([r["optimizer_ms_max"] for r in effective], 0.95),
            "samples_per_sec_avg": mean(r["samples_per_sec"] for r in effective),
            "tokens_per_sec_avg": mean(r["tokens_per_sec"] for r in effective),
            "loss_avg": mean(r["loss"] for r in effective),
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
