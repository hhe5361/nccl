#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SWITCH_SHARED_ROOTS = (
    Path("/mnt/nfs_share/cts_experiments/switch_log"),
    Path("/mnt/nfs/cts_experiments/switch_log"),
)
PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
NUMERIC_FIELDS = {
    "tNs",
    "rank",
    "peer",
    "channel",
    "elapsedNs",
    "samples",
    "bytes",
    "posts",
    "wstalls",
    "delayMeanNs",
    "delayMaxNs",
    "delayBaselineNs",
    "delayFastNs",
    "delaySlowNs",
    "gbps",
    "gbpsBaseline",
    "gbpsFast",
    "gbpsSlow",
    "eDelay",
    "eTrend",
    "eThroughput",
    "e",
    "u",
    "wBefore",
    "wAfter",
    "cooldown",
    "stableLow",
    "wstallRatio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Phase6 W adjustments with switch/network congestion metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="Phase6 experiment run root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--bucket-ms", type=float, default=1000.0, help="Network bucket width in ms")
    parser.add_argument("--event-window-sec", type=float, default=2.0, help="Before/after window around W adjustment events")
    parser.add_argument(
        "--event-plot-window-sec",
        type=float,
        default=5.0,
        help="Event-centered plot window on each side of a W adjustment event",
    )
    parser.add_argument("--max-event-plots", type=int, default=80, help="Maximum number of event-centered plots to write")
    parser.add_argument(
        "--workers",
        default="worker01",
        help="Comma-separated worker directory names to scan for NCCL logs. Default: worker01",
    )
    parser.add_argument("--all-workers", action="store_true", help="Scan all workers. Slower on large NCCL logs.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            index = 0
            while index < len(line):
                while index < len(line) and line[index].isspace():
                    index += 1
                if index >= len(line):
                    break
                try:
                    row, next_index = decoder.raw_decode(line, index)
                except json.JSONDecodeError:
                    break
                if isinstance(row, dict):
                    rows.append(row)
                index = next_index
    return rows


def to_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def percentile(values: Iterable[float], pct: float) -> float:
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def parse_pairs(line: str) -> dict:
    row = {key: value for key, value in PAIR_RE.findall(line)}
    for key in NUMERIC_FIELDS:
        if key in row:
            row[key] = to_float(row[key])
    return row


def run_rg(input_dir: Path, workers: Optional[set[str]]) -> list[tuple[Path, str]]:
    if workers:
        log_paths = []
        for worker in sorted(workers):
            log_paths.extend(sorted(input_dir.glob(f"repeat_*/P6*/{worker}/nccl.*.log")))
    else:
        log_paths = sorted(input_dir.glob("repeat_*/P6*/*/nccl.*.log"))
    if not log_paths:
        if workers:
            log_paths = []
            for worker in sorted(workers):
                log_paths.extend(sorted(input_dir.glob(f"repeat_*/*/{worker}/nccl.*.log")))
        else:
            log_paths = sorted(input_dir.glob("repeat_*/*/*/nccl.*.log"))
    if not log_paths:
        return []
    cmd = [
        "rg",
        "--fixed-strings",
        "--no-heading",
        "--with-filename",
        "--line-number",
        "--threads",
        "4",
        "PHASE6 event=CTRL_EPOCH",
        *[str(path) for path in log_paths],
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return run_python_scan(log_paths)
    if proc.returncode not in (0, 1):
        return run_python_scan(log_paths)
    matches: list[tuple[Path, str]] = []
    for line in proc.stdout.splitlines():
        path_str, _line_no, payload = line.split(":", 2)
        matches.append((Path(path_str), payload))
    return matches


def run_python_scan(log_paths: Iterable[Path]) -> list[tuple[Path, str]]:
    matches: list[tuple[Path, str]] = []
    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PHASE6 event=CTRL_EPOCH" in line:
                    matches.append((log_path, line))
    return matches


def load_phase6_epochs(run_root: Path, workers: Optional[set[str]]) -> list[dict]:
    rows: list[dict] = []
    for log_path, line in run_rg(run_root, workers):
        parts = log_path.relative_to(run_root).parts
        if len(parts) < 4:
            continue
        repeat, mode, worker = parts[:3]
        row = parse_pairs(line)
        row["repeat"] = repeat
        row["mode"] = mode
        row["worker"] = worker
        rows.append(row)
    rows.sort(key=lambda r: (str(r["repeat"]), str(r["mode"]), to_float(r.get("tNs")), str(r["worker"])))
    return rows


def resolve_switch_log_dir(run_root: Path) -> Optional[Path]:
    env_path = run_root / "switch_logger.env"
    if not env_path.exists():
        return None
    candidates: list[Path] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("SWITCH_LOG_LOCAL_DIR=") or line.startswith("SWITCH_LOG_DIR="):
            candidates.append(Path(line.split("=", 1)[1].strip()))
        elif line.startswith("SWITCH_LOG_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            candidates.extend(root / run_id for root in SWITCH_SHARED_ROOTS)
    seen: set[str] = set()
    for candidate in candidates:
        variants = [candidate]
        raw = candidate.as_posix()
        if raw.startswith("/mnt/nfs/"):
            variants.append(Path("/mnt/nfs_share/" + raw[len("/mnt/nfs/") :]))
        elif raw.startswith("/mnt/nfs_share/"):
            variants.append(Path("/mnt/nfs/" + raw[len("/mnt/nfs_share/") :]))
        for path in variants:
            key = path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return path
    return None


def extract_ts(row: dict) -> Optional[int]:
    for key in ("ts_mid_unix_ns", "ts_unix_ns", "timestamp_ns"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def load_cumulative_series(path: Path, keys: tuple[str, ...]) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for row in read_jsonl(path):
        ts_ns = extract_ts(row)
        if ts_ns is None:
            continue
        total = 0.0
        found = False
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += max(0.0, float(value))
                found = True
        if found:
            series.append((ts_ns, total))
    series.sort(key=lambda item: item[0])
    return series


def interpolate(series: list[tuple[int, float]], ts_ns: int) -> Optional[float]:
    if not series:
        return None
    if ts_ns <= series[0][0]:
        return series[0][1]
    if ts_ns >= series[-1][0]:
        return series[-1][1]
    lo = 0
    hi = len(series) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] < ts_ns:
            lo = mid + 1
        else:
            hi = mid - 1
    left_ts, left_value = series[max(0, lo - 1)]
    right_ts, right_value = series[min(len(series) - 1, lo)]
    if right_ts == left_ts:
        return right_value
    ratio = (ts_ns - left_ts) / float(right_ts - left_ts)
    return left_value + (right_value - left_value) * ratio


def window_delta(series: list[tuple[int, float]], start_ns: int, end_ns: int) -> float:
    start = interpolate(series, start_ns)
    end = interpolate(series, end_ns)
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def combine_series(series_list: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    timestamps = sorted({ts for series in series_list for ts, _value in series})
    out: list[tuple[int, float]] = []
    for ts_ns in timestamps:
        total = 0.0
        found = False
        for series in series_list:
            value = interpolate(series, ts_ns)
            if value is None:
                continue
            total += value
            found = True
        if found:
            out.append((ts_ns, total))
    return out


def bucket_delta(series: list[tuple[int, float]], start_ns: int, end_ns: int, bucket_ms: float) -> list[tuple[float, float]]:
    if not series or end_ns <= start_ns:
        return []
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    points: list[tuple[float, float]] = []
    current = start_ns
    while current < end_ns:
        nxt = min(end_ns, current + bucket_ns)
        rel_sec = (nxt - start_ns) / 1_000_000_000.0
        points.append((rel_sec, window_delta(series, current, nxt)))
        current = nxt
    return points


def bucket_rate(series: list[tuple[int, float]], start_ns: int, end_ns: int, bucket_ms: float, scale: float = 1.0) -> list[tuple[float, float]]:
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    points = []
    for rel_sec, delta in bucket_delta(series, start_ns, end_ns, bucket_ms):
        rate = (delta * scale) / (bucket_ns / 1_000_000_000.0)
        points.append((rel_sec, rate))
    return points


def load_switch_bundle(run_root: Path) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(run_root)
    if log_dir is None:
        return None
    markers = load_mode_markers(log_dir / "markers.jsonl")
    rack_a = load_cumulative_series(log_dir / "rackA_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    rack_b = load_cumulative_series(log_dir / "rackB_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    spine_pfc = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_pause_packets_total", "tx_pause_packets_total"))
    spine_ecn = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_ecn_marked_packets_total",))
    total_pfc = combine_series([rack_a, rack_b, spine_pfc])
    worker_bytes = load_cumulative_series(
        log_dir / "worker_roce_counters_aggregate.jsonl",
        ("tx_prio_bytes_total", "rx_prio_bytes_total", "tx_prio_bytes", "rx_prio_bytes"),
    )
    worker_cnp = load_cumulative_series(
        log_dir / "worker_roce_counters_aggregate.jsonl",
        ("np_cnp_sent_total", "rp_cnp_handled_total", "np_cnp_sent", "rp_cnp_handled"),
    )
    return {
        "log_dir": log_dir,
        "markers": markers,
        "rackA_pfc": rack_a,
        "rackB_pfc": rack_b,
        "spine_pfc": spine_pfc,
        "spine_ecn": spine_ecn,
        "total_pfc": total_pfc,
        "worker_bytes": worker_bytes,
        "worker_cnp": worker_cnp,
    }


def load_mode_markers(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    markers: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return markers
    for row in read_jsonl(path):
        marker = row.get("marker")
        message = str(row.get("message", ""))
        repeat_match = re.search(r"\brepeat=(\S+)", message)
        mode_match = re.search(r"\bmode=(\S+)", message)
        ts_ns = row.get("ts_unix_ns")
        if marker not in ("mode_start", "mode_end"):
            continue
        if not repeat_match or not mode_match or not isinstance(ts_ns, (int, float)):
            continue
        key = (repeat_match.group(1), mode_match.group(1).upper())
        markers[key]["start_ns" if marker == "mode_start" else "end_ns"] = int(ts_ns)
    return markers


def aligned_switch_window(
    repeat: str,
    mode: str,
    rows: list[dict],
    switch: Optional[dict],
) -> tuple[int, int, int, bool]:
    ctrl_start_ns = int(min(to_float(r.get("tNs")) for r in rows))
    ctrl_end_ns = int(max(to_float(r.get("tNs")) for r in rows))
    if switch:
        marker = switch.get("markers", {}).get((repeat, mode))
        if marker and "start_ns" in marker:
            switch_start_ns = int(marker["start_ns"])
            return ctrl_start_ns, switch_start_ns, switch_start_ns + (ctrl_end_ns - ctrl_start_ns), True
    return ctrl_start_ns, ctrl_start_ns, ctrl_end_ns, False


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["repeat"]), str(row["mode"]))].append(row)
    for group in groups.values():
        group.sort(key=lambda r: to_float(r.get("tNs")))
    return groups


def downsample(rows: list[dict], limit: int = 6000) -> list[dict]:
    if len(rows) <= limit:
        return rows
    step = max(1, len(rows) // limit)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def bucket_ctrl_rows(rows: list[dict], start_ns: int, end_ns: int, bucket_ms: float) -> list[dict]:
    if not rows or end_ns <= start_ns:
        return []
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        t_ns = int(to_float(row.get("tNs")))
        if t_ns < start_ns or t_ns > end_ns:
            continue
        bucket_idx = max(0, int((t_ns - start_ns) // bucket_ns))
        buckets[bucket_idx].append(row)

    out: list[dict] = []
    last_idx = max(0, int(math.ceil((end_ns - start_ns) / bucket_ns)))
    for idx in range(last_idx):
        group = buckets.get(idx, [])
        bucket_start_ns = start_ns + idx * bucket_ns
        bucket_end_ns = min(end_ns, bucket_start_ns + bucket_ns)
        rel_mid_s = ((bucket_start_ns + bucket_end_ns) / 2.0 - start_ns) / 1_000_000_000.0
        delays = [to_float(r.get("delayMeanNs")) / 1e6 for r in group]
        delay_max = [to_float(r.get("delayMaxNs")) / 1e6 for r in group]
        w_vals = [to_float(r.get("wAfter")) for r in group]
        e_vals = [to_float(r.get("e")) for r in group]
        wstall_vals = [to_float(r.get("wstallRatio")) for r in group]
        out.append(
            {
                "rel_mid_s": rel_mid_s,
                "bucket_start_ns": bucket_start_ns,
                "bucket_end_ns": bucket_end_ns,
                "samples": len(group),
                "delay_p50_ms": percentile(delays, 50),
                "delay_p95_ms": percentile(delays, 95),
                "delay_p99_ms": percentile(delays, 99),
                "delay_mean_max_ms": max(finite(delays), default=math.nan),
                "delay_max_peak_ms": max(finite(delay_max), default=math.nan),
                "w_min": min(finite(w_vals), default=math.nan),
                "w_max": max(finite(w_vals), default=math.nan),
                "e_max": max(finite(e_vals), default=math.nan),
                "wstall_max": max(finite(wstall_vals), default=math.nan),
                "decrease_count": sum(1 for r in group if r.get("action") == "decrease"),
                "increase_count": sum(1 for r in group if r.get("action") == "increase"),
            }
        )
    return out


def write_bucket_csv(
    groups: dict[tuple[str, str], list[dict]],
    switch: Optional[dict],
    output_dir: Path,
    bucket_ms: float,
) -> Path:
    path = output_dir / "phase6_bucket_metrics.csv"
    fields = [
        "repeat",
        "mode",
        "rel_mid_s",
        "samples",
        "delay_p50_ms",
        "delay_p95_ms",
        "delay_p99_ms",
        "delay_mean_max_ms",
        "delay_max_peak_ms",
        "w_min",
        "w_max",
        "e_max",
        "wstall_max",
        "decrease_count",
        "increase_count",
        "pfc_delta",
        "ecn_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (repeat, mode), rows in sorted(groups.items()):
            ctrl_start_ns, switch_start_ns, switch_end_ns, _marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
            ctrl_end_ns = int(max(to_float(r.get("tNs")) for r in rows))
            buckets = bucket_ctrl_rows(rows, ctrl_start_ns, ctrl_end_ns, bucket_ms)
            switch_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms) if switch else []
            switch_ecn = bucket_delta(switch["spine_ecn"], switch_start_ns, switch_end_ns, bucket_ms) if switch else []
            for idx, bucket in enumerate(buckets):
                out = {key: bucket.get(key, "") for key in fields}
                out["repeat"] = repeat
                out["mode"] = mode
                out["pfc_delta"] = switch_pfc[idx][1] if idx < len(switch_pfc) else 0.0
                out["ecn_delta"] = switch_ecn[idx][1] if idx < len(switch_ecn) else 0.0
                writer.writerow(out)
    return path


def plot_group(repeat: str, mode: str, rows: list[dict], switch: Optional[dict], output_dir: Path, bucket_ms: float) -> str:
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    plot_rows = downsample(rows)
    xs = [(to_float(r.get("tNs")) - start_ns) / 1_000_000_000.0 for r in plot_rows]
    decrease_rows = [r for r in rows if r.get("action") == "decrease"]
    increase_rows = [r for r in rows if r.get("action") == "increase"]

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    axes[0].plot(xs, [to_float(r.get("wAfter")) for r in plot_rows], linewidth=1.5, label="W", color="#1f77b4")
    axes[0].set_ylabel("W")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, [to_float(r.get("delayMeanNs")) / 1e6 for r in plot_rows], linewidth=1.0, label="delayMean ms", color="#d62728")
    axes[1].plot(xs, [to_float(r.get("delaySlowNs")) / 1e6 for r in plot_rows], linewidth=1.0, label="delaySlow ms", color="#9467bd", alpha=0.8)
    axes[1].set_ylabel("POST-DONE ms")
    axes[1].legend(loc="upper right")

    axes[2].plot(xs, [to_float(r.get("e")) for r in plot_rows], linewidth=1.0, label="spike e", color="#ff7f0e")
    axes[2].plot(xs, [to_float(r.get("wstallRatio")) for r in plot_rows], linewidth=1.0, label="wstallRatio", color="#2ca02c", alpha=0.8)
    axes[2].set_ylabel("Controller signal")
    axes[2].legend(loc="upper right")

    if switch:
        total_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
        spine_ecn = bucket_delta(switch["spine_ecn"], switch_start_ns, switch_end_ns, bucket_ms)
        worker_gbps = bucket_rate(switch["worker_bytes"], switch_start_ns, switch_end_ns, bucket_ms, scale=8e-9)
        if total_pfc:
            axes[3].plot([x for x, _ in total_pfc], [y for _, y in total_pfc], label="total PFC delta/bucket", color="#111827")
        if spine_ecn:
            axes[3].plot([x for x, _ in spine_ecn], [y for _, y in spine_ecn], label="spine ECN delta/bucket", color="#8c564b")
        if worker_gbps:
            ax2 = axes[3].twinx()
            ax2.plot([x for x, _ in worker_gbps], [y for _, y in worker_gbps], label="worker NIC Gbps", color="#17becf", alpha=0.7)
            ax2.set_ylabel("NIC Gbps")
            ax2.legend(loc="upper right")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper left")

    for ax in axes:
        for row in decrease_rows:
            x = (to_float(row.get("tNs")) - start_ns) / 1_000_000_000.0
            ax.axvline(x, color="#b91c1c", alpha=0.22, linewidth=1.2)
        for row in increase_rows:
            x = (to_float(row.get("tNs")) - start_ns) / 1_000_000_000.0
            ax.axvline(x, color="#047857", alpha=0.22, linewidth=1.2)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time since P6 mode start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    fig.suptitle(f"{repeat} {mode}: Phase6 W Adjustment vs Network Congestion ({alignment})")
    fig.tight_layout()
    name = f"{repeat}_{mode}_w_network_overlay.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def plot_bucket_group(
    repeat: str,
    mode: str,
    rows: list[dict],
    switch: Optional[dict],
    output_dir: Path,
    bucket_ms: float,
) -> str:
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    end_ns = int(max(to_float(r.get("tNs")) for r in rows))
    buckets = bucket_ctrl_rows(rows, start_ns, end_ns, bucket_ms)
    if not buckets:
        return ""
    xs = [b["rel_mid_s"] for b in buckets]
    decrease_rows = [r for r in rows if r.get("action") == "decrease"]
    increase_rows = [r for r in rows if r.get("action") == "increase"]

    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=False)
    axes[0].plot(xs, [b["delay_p50_ms"] for b in buckets], label="POST-DONE p50", color="#9ca3af", linewidth=1.0)
    axes[0].plot(xs, [b["delay_p95_ms"] for b in buckets], label="POST-DONE p95", color="#f97316", linewidth=1.0)
    axes[0].plot(xs, [b["delay_p99_ms"] for b in buckets], label="POST-DONE p99", color="#dc2626", linewidth=1.2)
    axes[0].plot(xs, [b["delay_mean_max_ms"] for b in buckets], label="POST-DONE mean max", color="#7f1d1d", linewidth=0.9, alpha=0.75)
    axes[0].set_ylabel("POST-DONE ms")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, [b["w_min"] for b in buckets], label="W min", color="#1d4ed8")
    axes[1].plot(xs, [b["w_max"] for b in buckets], label="W max", color="#60a5fa", alpha=0.8)
    axes[1].bar(xs, [b["decrease_count"] for b in buckets], width=bucket_ms / 1000.0 * 0.8, label="W decrease count", color="#ef4444", alpha=0.35)
    axes[1].set_ylabel("W / events")
    axes[1].legend(loc="upper right")

    axes[2].plot(xs, [b["e_max"] for b in buckets], label="spike e max", color="#f59e0b")
    axes[2].plot(xs, [b["wstall_max"] for b in buckets], label="wstallRatio max", color="#16a34a")
    axes[2].set_ylabel("Controller signal")
    axes[2].legend(loc="upper right")

    if switch:
        total_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
        spine_ecn = bucket_delta(switch["spine_ecn"], switch_start_ns, switch_end_ns, bucket_ms)
        if total_pfc:
            axes[3].plot([x for x, _ in total_pfc], [y for _, y in total_pfc], label="total PFC delta/bucket", color="#111827")
        if spine_ecn:
            axes[3].plot([x for x, _ in spine_ecn], [y for _, y in spine_ecn], label="spine ECN delta/bucket", color="#8c564b")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper right")

    if switch:
        total_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
        pfc_vals = [y for _, y in total_pfc[: len(buckets)]]
        delay_vals = [b["delay_p99_ms"] for b in buckets[: len(pfc_vals)]]
        axes[4].scatter(pfc_vals, delay_vals, s=18, alpha=0.65, color="#4b5563")
        axes[4].set_xlabel("PFC delta / bucket")
        axes[4].set_ylabel("POST-DONE p99 ms")
    else:
        axes[4].set_ylabel("PFC unavailable")

    for ax in axes[:4]:
        for row in decrease_rows:
            x = (to_float(row.get("tNs")) - start_ns) / 1_000_000_000.0
            ax.axvline(x, color="#b91c1c", alpha=0.22, linewidth=1.2)
        for row in increase_rows:
            x = (to_float(row.get("tNs")) - start_ns) / 1_000_000_000.0
            ax.axvline(x, color="#047857", alpha=0.22, linewidth=1.2)
        ax.grid(True, alpha=0.25)
    axes[4].grid(True, alpha=0.25)

    axes[3].set_xlabel("Time since P6 CTRL_EPOCH start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    fig.suptitle(f"{repeat} {mode}: Bucketed POST-DONE/W vs PFC ({bucket_ms:g} ms, {alignment})")
    fig.tight_layout()
    name = f"{repeat}_{mode}_bucketed_w_pfc_delay.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def plot_event_centered(
    repeat: str,
    mode: str,
    rows: list[dict],
    event_row: dict,
    event_idx: int,
    switch: Optional[dict],
    output_dir: Path,
    bucket_ms: float,
    window_sec: float,
) -> str:
    event_ns = int(to_float(event_row.get("tNs")))
    window_ns = int(window_sec * 1_000_000_000.0)
    channel = event_row.get("channel")
    worker = event_row.get("worker")
    local_rows = [
        r
        for r in rows
        if r.get("worker") == worker
        and r.get("channel") == channel
        and event_ns - window_ns <= int(to_float(r.get("tNs"))) <= event_ns + window_ns
    ]
    if not local_rows:
        return ""
    xs = [(to_float(r.get("tNs")) - event_ns) / 1_000_000_000.0 for r in local_rows]
    ctrl_start_ns, switch_start_ns, _switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    aligned_event_ns = switch_start_ns + (event_ns - ctrl_start_ns) if marker_aligned else event_ns

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(xs, [to_float(r.get("delayMeanNs")) / 1e6 for r in local_rows], label="delayMean ms", color="#dc2626")
    axes[0].plot(xs, [to_float(r.get("delaySlowNs")) / 1e6 for r in local_rows], label="delaySlow ms", color="#7c3aed", alpha=0.8)
    axes[0].set_ylabel("POST-DONE ms")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, [to_float(r.get("wAfter")) for r in local_rows], label="W", color="#2563eb")
    axes[1].set_ylabel("W")
    axes[1].legend(loc="upper right")

    axes[2].plot(xs, [to_float(r.get("e")) for r in local_rows], label="spike e", color="#f59e0b")
    axes[2].plot(xs, [to_float(r.get("wstallRatio")) for r in local_rows], label="wstallRatio", color="#16a34a")
    axes[2].set_ylabel("Controller")
    axes[2].legend(loc="upper right")

    if switch:
        pfc = bucket_delta(switch["total_pfc"], aligned_event_ns - window_ns, aligned_event_ns + window_ns, bucket_ms)
        ecn = bucket_delta(switch["spine_ecn"], aligned_event_ns - window_ns, aligned_event_ns + window_ns, bucket_ms)
        if pfc:
            axes[3].plot([x - window_sec for x, _ in pfc], [y for _, y in pfc], label="total PFC delta/bucket", color="#111827")
        if ecn:
            axes[3].plot([x - window_sec for x, _ in ecn], [y for _, y in ecn], label="spine ECN delta/bucket", color="#8c564b")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper right")

    for ax in axes:
        ax.axvline(0.0, color="#b91c1c", linewidth=1.5, alpha=0.75, label="_nolegend_")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time relative to W decrease event (s)")
    fig.suptitle(
        f"{repeat} {mode} event {event_idx:03d}: W {event_row.get('wBefore')} -> {event_row.get('wAfter')} "
        f"worker={worker} channel={channel}"
    )
    fig.tight_layout()
    safe_channel = str(channel).replace(".", "_")
    name = f"{repeat}_{mode}_event_{event_idx:03d}_{worker}_ch{safe_channel}_centered.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def write_event_csv(groups: dict[tuple[str, str], list[dict]], switch: Optional[dict], output_dir: Path, window_sec: float) -> Path:
    path = output_dir / "phase6_w_adjustment_network_windows.csv"
    fields = [
        "repeat",
        "mode",
        "worker",
        "channel",
        "action",
        "tNs",
        "aligned_switch_unix_ns",
        "wBefore",
        "wAfter",
        "e",
        "delayMeanMs",
        "delaySlowMs",
        "pfc_before",
        "pfc_after",
        "ecn_before",
        "ecn_after",
        "worker_gb_before",
        "worker_gb_after",
        "cnp_before",
        "cnp_after",
    ]
    window_ns = int(window_sec * 1_000_000_000)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (repeat, mode), rows in sorted(groups.items()):
            ctrl_start_ns, switch_start_ns, _switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
            for row in rows:
                if row.get("action") not in ("decrease", "increase"):
                    continue
                t_ns = int(to_float(row.get("tNs")))
                aligned_t_ns = switch_start_ns + (t_ns - ctrl_start_ns) if marker_aligned else t_ns
                out = {
                    "repeat": repeat,
                    "mode": mode,
                    "worker": row.get("worker"),
                    "channel": row.get("channel"),
                    "action": row.get("action"),
                    "tNs": t_ns,
                    "aligned_switch_unix_ns": aligned_t_ns if marker_aligned else "",
                    "wBefore": row.get("wBefore"),
                    "wAfter": row.get("wAfter"),
                    "e": row.get("e"),
                    "delayMeanMs": to_float(row.get("delayMeanNs")) / 1e6,
                    "delaySlowMs": to_float(row.get("delaySlowNs")) / 1e6,
                    "pfc_before": 0.0,
                    "pfc_after": 0.0,
                    "ecn_before": 0.0,
                    "ecn_after": 0.0,
                    "worker_gb_before": 0.0,
                    "worker_gb_after": 0.0,
                    "cnp_before": 0.0,
                    "cnp_after": 0.0,
                }
                if switch:
                    out["pfc_before"] = window_delta(switch["total_pfc"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["pfc_after"] = window_delta(switch["total_pfc"], aligned_t_ns, aligned_t_ns + window_ns)
                    out["ecn_before"] = window_delta(switch["spine_ecn"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["ecn_after"] = window_delta(switch["spine_ecn"], aligned_t_ns, aligned_t_ns + window_ns)
                    out["worker_gb_before"] = window_delta(switch["worker_bytes"], aligned_t_ns - window_ns, aligned_t_ns) * 8e-9
                    out["worker_gb_after"] = window_delta(switch["worker_bytes"], aligned_t_ns, aligned_t_ns + window_ns) * 8e-9
                    out["cnp_before"] = window_delta(switch["worker_cnp"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["cnp_after"] = window_delta(switch["worker_cnp"], aligned_t_ns, aligned_t_ns + window_ns)
                writer.writerow(out)
    return path


def write_html(
    output_dir: Path,
    raw_images: list[str],
    bucket_images: list[str],
    event_images: list[str],
    event_csv: Path,
    bucket_csv: Path,
    switch: Optional[dict],
) -> Path:
    path = output_dir / "phase6_network_overlay_report.html"
    switch_msg = f"switch_log_dir={html.escape(str(switch['log_dir']))}" if switch else "switch_log_dir=not found"
    def image_section(title: str, names: list[str]) -> str:
        if not names:
            return ""
        images = "\n".join(f"<h3>{html.escape(name)}</h3><img src='{html.escape(name)}'>" for name in names)
        return f"<h2>{html.escape(title)}</h2>\n{images}"

    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase6 Network Overlay</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
img {{ width: 100%; max-width: 1900px; border: 1px solid #ddd; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style></head><body>
<h1>Phase6 W Adjustment vs Network Congestion</h1>
<p><code>{switch_msg}</code></p>
<p>Red vertical lines are W decrease events. Green vertical lines are W increase events. Switch metrics are aligned with <code>mode_start</code> markers from <code>markers.jsonl</code>.</p>
<p>Bucket metrics are in <code>{html.escape(bucket_csv.name)}</code>. Event before/after network deltas are in <code>{html.escape(event_csv.name)}</code>.</p>
{image_section("Bucketed Timeline: POST-DONE p99/max, W, PFC", bucket_images)}
{image_section("Event-Centered W Decrease Windows", event_images)}
{image_section("Raw Timeline", raw_images)}
</body></html>
""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    run_root = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = None if args.all_workers else {item.strip() for item in args.workers.split(",") if item.strip()}
    rows = load_phase6_epochs(run_root, workers)
    groups = group_rows(rows)
    switch = load_switch_bundle(run_root)
    event_csv = write_event_csv(groups, switch, output_dir, args.event_window_sec)
    bucket_csv = write_bucket_csv(groups, switch, output_dir, args.bucket_ms)
    raw_images = [
        plot_group(repeat, mode, group, switch, output_dir, args.bucket_ms)
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    bucket_images = [
        plot_bucket_group(repeat, mode, group, switch, output_dir, args.bucket_ms)
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    bucket_images = [name for name in bucket_images if name]

    event_images: list[str] = []
    event_count = 0
    for (repeat, mode), group in sorted(groups.items()):
        for event_idx, event_row in enumerate((r for r in group if r.get("action") == "decrease"), start=1):
            if event_count >= args.max_event_plots:
                break
            name = plot_event_centered(
                repeat,
                mode,
                group,
                event_row,
                event_idx,
                switch,
                output_dir,
                args.bucket_ms,
                args.event_plot_window_sec,
            )
            if name:
                event_images.append(name)
                event_count += 1
        if event_count >= args.max_event_plots:
            break

    html_path = write_html(output_dir, raw_images, bucket_images, event_images, event_csv, bucket_csv, switch)
    worker_msg = "all" if workers is None else ",".join(sorted(workers))
    print(f"[phase6-network] rows={len(rows)} groups={len(groups)} workers={worker_msg} switch={bool(switch)} html={html_path}")


if __name__ == "__main__":
    main()
