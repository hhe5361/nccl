#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def parse_kv_line(line: str, marker: str) -> dict[str, str] | None:
    if marker not in line:
        return None
    row = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[key] = value
    return row if "event" in row else None


def analyze_mode(mode_dir: Path) -> dict:
    post_times_ns: list[int] = []
    allow_times_ns: list[int] = []
    stall_times_ns: list[int] = []
    allow_costs: list[float] = []
    tokens_before_allow: list[float] = []
    tokens_after_allow: list[float] = []
    tokens_before_stall: list[float] = []
    tokens_after_stall: list[float] = []
    baseline_rates: list[float] = []
    target_rates: list[float] = []
    target_bursts: list[float] = []
    burst_fill_ratio_allow: list[float] = []
    burst_fill_ratio_stall: list[float] = []

    for log_path in sorted(mode_dir.glob("worker*/nccl.*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row0 = parse_kv_line(line, "PHASE0 event=")
            if row0 is not None and row0.get("event") == "PROXY_RECV_POST":
                post_times_ns.append(int(row0.get("tNs", "0")))
                continue
            row7 = parse_kv_line(line, "PHASE7 event=")
            if row7 is None:
                continue
            event = row7.get("event", "")
            if event == "RATE_ALLOW":
                allow_times_ns.append(int(row7.get("tNs", "0")))
                allow_costs.append(float(row7.get("postCost", "0")))
                tokens_before = float(row7.get("tokensBefore", "0"))
                tokens_after_allow.append(float(row7.get("tokensAfter", "0")))
                tokens_before_allow.append(tokens_before)
                target_burst = float(row7.get("targetBurst", "0"))
                if target_burst > 0:
                    burst_fill_ratio_allow.append(tokens_before / target_burst)
            elif event == "RATE_STALL":
                stall_times_ns.append(int(row7.get("tNs", "0")))
                tokens_before = float(row7.get("tokensBefore", "0"))
                tokens_after = float(row7.get("tokensAfter", "0"))
                tokens_before_stall.append(tokens_before)
                tokens_after_stall.append(tokens_after)
                target_burst = float(row7.get("targetBurst", "0"))
                if target_burst > 0:
                    burst_fill_ratio_stall.append(tokens_before / target_burst)
            elif event == "RATE_BASELINE":
                baseline_rates.append(float(row7.get("baselineRatePerMs", "0")))
                target_rates.append(float(row7.get("targetRatePerMs", "0")))
                target_bursts.append(float(row7.get("targetBurst", "0")))

    summary_path = mode_dir / f"{mode_dir.name}_summary.json"
    perf_summary = {}
    if summary_path.exists():
        perf_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "mode": mode_dir.name,
        "post_count": len(post_times_ns),
        "allow_event_count": len(allow_times_ns),
        "stall_event_count": len(stall_times_ns),
        "allow_cost_total": sum(allow_costs),
        "allow_cost_stats": summarize(allow_costs),
        "tokens_before_allow": summarize(tokens_before_allow),
        "tokens_after_allow": summarize(tokens_after_allow),
        "tokens_before_stall": summarize(tokens_before_stall),
        "tokens_after_stall": summarize(tokens_after_stall),
        "baseline_rate_per_ms": summarize(baseline_rates),
        "target_rate_per_ms": summarize(target_rates),
        "target_burst_posts": summarize(target_bursts),
        "burst_fill_ratio_allow": summarize(burst_fill_ratio_allow),
        "burst_fill_ratio_stall": summarize(burst_fill_ratio_stall),
        "perf_summary": {
            "step_ms_avg": perf_summary.get("step_ms_avg", 0),
            "steps_per_sec_avg": perf_summary.get("steps_per_sec_avg", 0),
            "samples_per_sec_avg": perf_summary.get("samples_per_sec_avg", 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase7 runtime post-rate diagnostics")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_root": str(experiment_root),
        "modes": [],
    }
    for mode_dir in sorted(p for p in experiment_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not any(mode_dir.glob("worker*/nccl.*.log")):
            continue
        summary["modes"].append(analyze_mode(mode_dir))

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase7-summary] experiment_root={experiment_root}")
    for mode in summary["modes"]:
        print(
            f"[phase7-summary] mode={mode['mode']} "
            f"posts={mode['post_count']} allowEvents={mode['allow_event_count']} "
            f"stallEvents={mode['stall_event_count']} allowCostTotal={mode['allow_cost_total']}"
        )
    print(f"[phase7-summary] wrote {output_json}")


if __name__ == "__main__":
    main()
