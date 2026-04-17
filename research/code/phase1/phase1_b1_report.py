#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


PHASE_RE = re.compile(r"PHASE[01]\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
W_DIR_RE = re.compile(r"^W(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 B1 DDP + NCCL logs")
    parser.add_argument("--input", required=True, help="Run root, e.g. /mnt/nfs_share/cts_experiments/phase1_b1...")
    parser.add_argument("--output-dir", default=None, help="Output directory for summary files")
    return parser.parse_args()


def parse_value(raw: str):
    if raw.startswith("0x"):
        try:
            return int(raw, 16)
        except ValueError:
            return raw
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_nccl_logs(root: Path) -> Dict[str, object]:
    event_counts: Counter = Counter()
    occ_pd: List[float] = []
    occ_tr: List[float] = []
    w_eff_values: List[int] = []
    for path in sorted(root.rglob("*.log")):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "event=" not in line or ("PHASE0" not in line and "PHASE1" not in line):
                    continue
                m = PHASE_RE.search(line)
                if not m:
                    continue
                kvs = {k: parse_value(v) for k, v in KV_RE.findall(m.group(1))}
                event = kvs.get("event")
                if not event:
                    continue
                event_counts[str(event)] += 1
                if "occPd" in kvs:
                    occ_pd.append(float(kvs["occPd"]))
                elif "posted" in kvs and "done" in kvs:
                    occ_pd.append(float(kvs["posted"]) - float(kvs["done"]))
                if "occTr" in kvs:
                    occ_tr.append(float(kvs["occTr"]))
                elif "transmitted" in kvs and "done" in kvs:
                    occ_tr.append(float(kvs["transmitted"]) - float(kvs["done"]))
                if "wEff" in kvs:
                    w_eff_values.append(int(kvs["wEff"]))
    return {
        "event_counts": dict(event_counts),
        "recv_wstall_count": int(event_counts.get("PROXY_RECV_WSTALL", 0)),
        "send_wstall_count": int(event_counts.get("PROXY_SEND_WSTALL", 0)),
        "cts_issue_count": int(event_counts.get("IB_CTS_ISSUE", 0)),
        "send_post_count": int(event_counts.get("IB_SEND_POST", 0)),
        "recv_post_count": int(event_counts.get("PROXY_RECV_POST", 0)),
        "max_occ_pd": max(occ_pd) if occ_pd else 0.0,
        "p99_occ_pd": percentile(occ_pd, 0.99),
        "max_occ_tr": max(occ_tr) if occ_tr else 0.0,
        "p99_occ_tr": percentile(occ_tr, 0.99),
        "w_eff_values": sorted(set(w_eff_values)),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    details: Dict[str, dict] = {}

    for w_dir in sorted(input_root.iterdir()):
        if not w_dir.is_dir():
            continue
        match = W_DIR_RE.match(w_dir.name)
        if not match:
            continue
        w = int(match.group(1))

        summary_path = w_dir / f"W{w}_summary.json"
        if not summary_path.exists():
            summary_candidates = sorted(w_dir.glob("*_summary.json"))
            if not summary_candidates:
                continue
            summary_path = summary_candidates[0]

        summary = load_json(summary_path)
        nccl = parse_nccl_logs(w_dir)
        row = {
            "w": w,
            "steps": summary.get("steps", 0),
            "effective_steps": summary.get("effective_steps", 0),
            "param_mb": summary.get("param_mb", 0),
            "dtype": summary.get("dtype", ""),
            "bucket_cap_mb": summary.get("bucket_cap_mb", 0),
            "step_ms_avg": summary.get("step_ms_avg", 0.0),
            "step_ms_p50": summary.get("step_ms_p50", 0.0),
            "step_ms_p95": summary.get("step_ms_p95", 0.0),
            "backward_ms_avg": summary.get("backward_ms_avg", 0.0),
            "backward_ms_p50": summary.get("backward_ms_p50", 0.0),
            "backward_ms_p95": summary.get("backward_ms_p95", 0.0),
            "ring_gbps_avg": summary.get("ring_gbps_avg", 0.0),
            "ring_gbps_p50": summary.get("ring_gbps_p50", 0.0),
            "ring_gbps_p95": summary.get("ring_gbps_p95", 0.0),
            "recv_wstall_count": nccl["recv_wstall_count"],
            "send_wstall_count": nccl["send_wstall_count"],
            "cts_issue_count": nccl["cts_issue_count"],
            "send_post_count": nccl["send_post_count"],
            "recv_post_count": nccl["recv_post_count"],
            "max_occ_pd": nccl["max_occ_pd"],
            "p99_occ_pd": nccl["p99_occ_pd"],
            "max_occ_tr": nccl["max_occ_tr"],
            "p99_occ_tr": nccl["p99_occ_tr"],
            "w_eff_values": ",".join(str(v) for v in nccl["w_eff_values"]),
        }
        rows.append(row)
        details[f"W{w}"] = {
            "summary": summary,
            "nccl": nccl,
        }

    rows.sort(key=lambda row: row["w"])

    csv_path = output_dir / "phase1_b1_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    json_path = output_dir / "phase1_b1_summary.json"
    json_path.write_text(json.dumps({"rows": rows, "details": details}, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[phase1-report] wrote {csv_path}")
    print(f"[phase1-report] wrote {json_path}")
    for row in rows:
        print(
            f"W={row['w']} step_ms_p95={row['step_ms_p95']:.3f} "
            f"backward_ms_p95={row['backward_ms_p95']:.3f} "
            f"ring_gbps_avg={row['ring_gbps_avg']:.3f} "
            f"recv_wstall={row['recv_wstall_count']} send_wstall={row['send_wstall_count']} "
            f"wEff={row['w_eff_values']}"
        )


if __name__ == "__main__":
    main()
