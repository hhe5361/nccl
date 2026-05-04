#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV_RE.finditer(line)}


def iter_mode_dirs(experiment_root: Path):
    for candidate in sorted(p for p in experiment_root.iterdir() if p.is_dir()):
        if candidate.name.startswith("."):
            continue
        yield candidate


def summarize_mode(mode_dir: Path) -> dict:
    recv_event_counts: Counter[str] = Counter()
    recv_event_by_channel: dict[str, Counter[str]] = defaultdict(Counter)
    recv_event_by_peer: dict[str, Counter[str]] = defaultdict(Counter)
    recv_group_sizes: Counter[str] = Counter()
    recv_group_by_channel: dict[str, Counter[str]] = defaultdict(Counter)
    recv_group_by_coll: Counter[str] = Counter()
    event_examples: dict[str, dict] = {}

    nccl_logs = sorted(mode_dir.glob("worker*/nccl.*.log"))
    for log_path in nccl_logs:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PHASE0 event=PROXY_RECV_" in line:
                    row = parse_kv(line)
                    event = row.get("event", "")
                    channel = row.get("channel", "unknown")
                    peer = row.get("peer", "unknown")
                    recv_event_counts[event] += 1
                    recv_event_by_channel[channel][event] += 1
                    recv_event_by_peer[peer][event] += 1
                    event_examples.setdefault(event, row)
                elif "APPENDIX2 event=RECV_GROUP_CFG" in line:
                    row = parse_kv(line)
                    group_size = row.get("groupSize", "unknown")
                    channel = row.get("channel", "unknown")
                    coll = row.get("coll", "unknown")
                    recv_group_sizes[group_size] += 1
                    recv_group_by_channel[channel][group_size] += 1
                    recv_group_by_coll[coll] += 1

    step_summary = None
    summary_candidates = sorted(mode_dir.glob("*_summary.json"))
    if summary_candidates:
      with summary_candidates[0].open("r", encoding="utf-8") as fh:
          step_summary = json.load(fh)

    return {
        "mode": mode_dir.name,
        "nccl_log_files": len(nccl_logs),
        "recv_event_counts": dict(sorted(recv_event_counts.items())),
        "recv_event_by_channel": {
            key: dict(sorted(counter.items(), key=lambda kv: kv[0]))
            for key, counter in sorted(recv_event_by_channel.items(), key=lambda kv: kv[0])
        },
        "recv_event_by_peer": {
            key: dict(sorted(counter.items(), key=lambda kv: kv[0]))
            for key, counter in sorted(recv_event_by_peer.items(), key=lambda kv: kv[0])
        },
        "recv_group_size_counts": dict(sorted(recv_group_sizes.items(), key=lambda kv: kv[0])),
        "recv_group_size_by_channel": {
            key: dict(sorted(counter.items(), key=lambda kv: kv[0]))
            for key, counter in sorted(recv_group_by_channel.items(), key=lambda kv: kv[0])
        },
        "recv_group_coll_counts": dict(sorted(recv_group_by_coll.items(), key=lambda kv: kv[0])),
        "event_examples": event_examples,
        "step_summary": step_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase4 NCCL recv/group events")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_root": str(experiment_root),
        "modes": [],
    }

    for mode_dir in iter_mode_dirs(experiment_root):
        report["modes"].append(summarize_mode(mode_dir))

    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[phase4-ncc-summary] experiment_root={experiment_root}")
    for mode in report["modes"]:
        print(
            f"[phase4-ncc-summary] mode={mode['mode']} "
            f"logs={mode['nccl_log_files']} "
            f"recv_events={sum(mode['recv_event_counts'].values())} "
            f"group_logs={sum(mode['recv_group_size_counts'].values())}"
        )
    print(f"[phase4-ncc-summary] wrote {output_json}")


if __name__ == "__main__":
    main()
