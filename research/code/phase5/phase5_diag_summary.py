#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[int], q: float) -> float:
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


def summarize_ints(values: list[int]) -> dict[str, float | int]:
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


def parse_phase5_line(line: str) -> dict[str, str] | None:
    if "PHASE5 event=" not in line:
      return None
    row = {}
    for token in line.strip().split():
      if "=" not in token:
        continue
      key, value = token.split("=", 1)
      row[key] = value
    return row if "event" in row else None


def analyze_mode(mode_dir: Path) -> dict:
    progress_deltas_ns: list[int] = []
    post_to_net_done_ns: list[int] = []
    progress_calls_since_post: list[int] = []
    wstall_count = 0
    post_count = 0
    net_done_count = 0
    workers: list[dict] = []

    for worker_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
      worker_progress_deltas_ns: list[int] = []
      worker_post_to_net_done_ns: list[int] = []
      worker_progress_calls_since_post: list[int] = []
      worker_wstall_count = 0
      worker_post_count = 0
      worker_net_done_count = 0

      for log_path in sorted(worker_dir.glob("nccl.*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
          row = parse_phase5_line(line)
          if row is None:
            continue
          event = row.get("event", "")
          if event == "RECV_PROXY_PROGRESS":
            delta = int(row.get("deltaNs", "0"))
            worker_progress_deltas_ns.append(delta)
            progress_deltas_ns.append(delta)
          elif event == "PROXY_RECV_POST":
            worker_post_count += 1
            post_count += 1
          elif event == "PROXY_RECV_NET_DONE":
            delay = int(row.get("postToNetDoneNs", "0"))
            calls = int(row.get("progressCallsSincePost", "0"))
            worker_net_done_count += 1
            net_done_count += 1
            worker_post_to_net_done_ns.append(delay)
            post_to_net_done_ns.append(delay)
            worker_progress_calls_since_post.append(calls)
            progress_calls_since_post.append(calls)
          elif event == "PROXY_RECV_WSTALL":
            worker_wstall_count += 1
            wstall_count += 1

      workers.append(
        {
          "worker": worker_dir.name,
          "progress_delta_ns": summarize_ints(worker_progress_deltas_ns),
          "post_to_net_done_ns": summarize_ints(worker_post_to_net_done_ns),
          "progress_calls_since_post": summarize_ints(worker_progress_calls_since_post),
          "post_count": worker_post_count,
          "net_done_count": worker_net_done_count,
          "wstall_count": worker_wstall_count,
        }
      )

    return {
      "mode": mode_dir.name,
      "progress_delta_ns": summarize_ints(progress_deltas_ns),
      "post_to_net_done_ns": summarize_ints(post_to_net_done_ns),
      "progress_calls_since_post": summarize_ints(progress_calls_since_post),
      "post_count": post_count,
      "net_done_count": net_done_count,
      "wstall_count": wstall_count,
      "workers": workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase5 recv-proxy diagnostics")
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
      if not any(mode_dir.glob("*/nccl.*.log")):
        continue
      summary["modes"].append(analyze_mode(mode_dir))

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase5-summary] experiment_root={experiment_root}")
    for mode in summary["modes"]:
      print(
        f"[phase5-summary] mode={mode['mode']} "
        f"posts={mode['post_count']} net_done={mode['net_done_count']} "
        f"wstall={mode['wstall_count']} "
        f"post_to_net_done_p50_ns={mode['post_to_net_done_ns'].get('p50', 0)} "
        f"progress_delta_p50_ns={mode['progress_delta_ns'].get('p50', 0)}"
      )
    print(f"[phase5-summary] wrote {output_json}")


if __name__ == "__main__":
    main()
