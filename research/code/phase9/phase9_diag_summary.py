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
    inst_rates: list[float] = []
    delta_ns: list[float] = []
    post_costs: list[float] = []
    workers: list[dict] = []

    for worker_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("worker")):
        worker_rates: list[float] = []
        worker_delta_ns: list[float] = []
        worker_post_costs: list[float] = []
        for log_path in sorted(worker_dir.glob("nccl.*.log")):
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                row = parse_kv_line(line, "PHASE9 event=")
                if row is None or row.get("event") != "RECVCOMM_POST_RATE":
                    continue
                rate = float(row.get("instPostRatePerMs", "0"))
                delta = float(row.get("deltaNs", "0"))
                cost = float(row.get("postCost", "0"))
                worker_rates.append(rate)
                worker_delta_ns.append(delta)
                worker_post_costs.append(cost)
                inst_rates.append(rate)
                delta_ns.append(delta)
                post_costs.append(cost)
        workers.append(
            {
                "worker": worker_dir.name,
                "inst_post_rate_per_ms": summarize(worker_rates),
                "delta_ns": summarize(worker_delta_ns),
                "post_cost": summarize(worker_post_costs),
            }
        )

    summary_path = mode_dir / f"{mode_dir.name}_summary.json"
    perf_summary = {}
    if summary_path.exists():
        perf_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "mode": mode_dir.name,
        "inst_post_rate_per_ms": summarize(inst_rates),
        "delta_ns": summarize(delta_ns),
        "post_cost": summarize(post_costs),
        "workers": workers,
        "perf_summary": {
            "step_ms_avg": perf_summary.get("step_ms_avg", 0),
            "steps_per_sec_avg": perf_summary.get("steps_per_sec_avg", 0),
            "samples_per_sec_avg": perf_summary.get("samples_per_sec_avg", 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase9 recvComm post-rate diagnostics")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {"experiment_root": str(experiment_root), "modes": []}
    for mode_dir in sorted(p for p in experiment_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not any(mode_dir.glob("worker*/nccl.*.log")):
            continue
        summary["modes"].append(analyze_mode(mode_dir))

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase9-summary] experiment_root={experiment_root}")
    for mode in summary["modes"]:
        print(
            f"[phase9-summary] mode={mode['mode']} "
            f"inst_rate_p50={mode['inst_post_rate_per_ms'].get('p50', 0)} "
            f"delta_ns_p50={mode['delta_ns'].get('p50', 0)}"
        )
    print(f"[phase9-summary] wrote {output_json}")


if __name__ == "__main__":
    main()
