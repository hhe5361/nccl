#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
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
    "tsUnixNs",
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
    parser.add_argument("--bin-ms", type=float, default=None, help="Alias for --bucket-ms")
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
    parser.add_argument("--switch-log-dir", type=Path, default=None, help="Explicit local switch log directory")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw timeline plots. Recommended for all-worker reports.")
    parser.add_argument(
        "--delay-plot-pct",
        type=float,
        default=99.0,
        help="Percentile used as the trimmed POST-DONE tail line in binned plots",
    )
    parser.add_argument(
        "--delay-trim-top",
        type=int,
        default=3,
        help="Drop this many largest POST-DONE samples per analysis bin before plotting tail/max lines",
    )
    parser.add_argument(
        "--delay-ymax-pct",
        type=float,
        default=99.5,
        help="Percentile used to choose POST-DONE y-axis upper limit in binned plots",
    )
    parser.add_argument("--delay-ymax-ms", type=float, default=None, help="Explicit POST-DONE y-axis upper limit in ms")
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
    if len(vals) == 1:
        return vals[0]
    pos = min(len(vals) - 1, max(0.0, (pct / 100.0) * (len(vals) - 1)))
    lo = int(math.floor(pos))
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def drop_top(values: Iterable[float], count: int) -> list[float]:
    vals = sorted(finite(values))
    if count <= 0 or len(vals) <= count:
        return vals
    return vals[:-count]


def parse_pairs(line: str) -> dict:
    row = {key: value for key, value in PAIR_RE.findall(line)}
    for key in NUMERIC_FIELDS:
        if key in row:
            row[key] = to_float(row[key])
    return row


def run_rg(input_dir: Path, workers: Optional[set[str]], needle: str = "PHASE6 event=CTRL_EPOCH") -> list[tuple[Path, str]]:
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
        needle,
        *[str(path) for path in log_paths],
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return run_python_scan(log_paths, needle)
    if proc.returncode not in (0, 1):
        return run_python_scan(log_paths, needle)
    matches: list[tuple[Path, str]] = []
    for line in proc.stdout.splitlines():
        path_str, _line_no, payload = line.split(":", 2)
        matches.append((Path(path_str), payload))
    return matches


def run_python_scan(log_paths: Iterable[Path], needle: str) -> list[tuple[Path, str]]:
    matches: list[tuple[Path, str]] = []
    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if needle in line:
                    matches.append((log_path, line))
    return matches


def load_phase6_anchors(run_root: Path, workers: Optional[set[str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    anchors: dict[tuple[str, str, str], dict[str, float]] = {}
    for log_path, line in run_rg(run_root, workers, "PHASE6 event=CTRL_ANCHOR"):
        parts = log_path.relative_to(run_root).parts
        if len(parts) < 4:
            continue
        repeat, mode, worker = parts[:3]
        row = parse_pairs(line)
        t_ns = to_float(row.get("tNs"))
        unix_ns = to_float(row.get("tsUnixNs"))
        if not math.isfinite(t_ns) or not math.isfinite(unix_ns):
            continue
        key = (repeat, mode, worker)
        previous = anchors.get(key)
        if previous is None or t_ns < previous["anchor_t_ns"]:
            anchors[key] = {"anchor_t_ns": t_ns, "anchor_unix_ns": unix_ns}
    return anchors


def load_phase6_epochs(run_root: Path, workers: Optional[set[str]]) -> list[dict]:
    anchors = load_phase6_anchors(run_root, workers)
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
    add_worker_relative_times(rows)
    apply_anchor_unix_times(rows, anchors)
    rows.sort(
        key=lambda r: (
            str(r["repeat"]),
            str(r["mode"]),
            to_float(r.get("unixNs"), to_float(r.get("tNs"))),
            str(r["worker"]),
        )
    )
    return rows


def apply_anchor_unix_times(rows: list[dict], anchors: dict[tuple[str, str, str], dict[str, float]]) -> None:
    for row in rows:
        key = (str(row["repeat"]), str(row["mode"]), str(row["worker"]))
        t_ns = to_float(row.get("tNs"))
        anchor = anchors.get(key)
        if anchor is None or not math.isfinite(t_ns):
            row["anchorAligned"] = 0
            continue
        row["unixNs"] = anchor["anchor_unix_ns"] + (t_ns - anchor["anchor_t_ns"])
        row["anchorAligned"] = 1


def add_worker_relative_times(rows: list[dict]) -> None:
    starts: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (str(row["repeat"]), str(row["mode"]), str(row["worker"]))
        t_ns = to_float(row.get("tNs"))
        if not math.isfinite(t_ns):
            continue
        starts[key] = min(starts.get(key, t_ns), t_ns)
    for row in rows:
        key = (str(row["repeat"]), str(row["mode"]), str(row["worker"]))
        t_ns = to_float(row.get("tNs"))
        start_ns = starts.get(key, t_ns)
        row["relNs"] = max(0.0, t_ns - start_ns)


def apply_switch_relative_times(rows: list[dict], switch: Optional[dict]) -> int:
    if not switch:
        return 0
    aligned = 0
    markers = switch.get("markers", {})
    for row in rows:
        marker = markers.get((str(row["repeat"]), str(row["mode"])))
        unix_ns = to_float(row.get("unixNs"))
        if not marker or "start_ns" not in marker or not math.isfinite(unix_ns):
            row["switchAligned"] = 0
            continue
        row["relNs"] = unix_ns - int(marker["start_ns"])
        row["switchAligned"] = 1
        aligned += 1
    return aligned


def resolve_switch_log_dir(run_root: Path, explicit_switch_log_dir: Optional[Path] = None) -> Optional[Path]:
    if explicit_switch_log_dir is not None:
        explicit = explicit_switch_log_dir.expanduser().resolve()
        if explicit.exists():
            return explicit
    env_path = run_root / "switch_logger.env"
    if not env_path.exists():
        return None
    candidates: list[Path] = []
    run_ids: list[str] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("SWITCH_LOG_LOCAL_DIR=") or line.startswith("SWITCH_LOG_DIR="):
            candidates.append(Path(line.split("=", 1)[1].strip()))
        elif line.startswith("SWITCH_LOG_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            run_ids.append(run_id)
            candidates.extend(root / run_id for root in SWITCH_SHARED_ROOTS)
    for run_id in run_ids:
        for parent in [run_root, *run_root.parents]:
            candidates.extend(
                [
                    parent / "switch_log" / run_id,
                    parent / "switch_log" / "switch_log" / run_id,
                    parent.parent / "switch_log" / run_id if parent.parent != parent else parent / "switch_log" / run_id,
                    parent.parent / "switch_log" / "switch_log" / run_id
                    if parent.parent != parent
                    else parent / "switch_log" / "switch_log" / run_id,
                ]
            )
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


def load_switch_bundle(run_root: Path, explicit_switch_log_dir: Optional[Path] = None) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(run_root, explicit_switch_log_dir)
    if log_dir is None:
        return None
    markers = load_mode_markers(log_dir / "markers.jsonl")
    rack_a = load_cumulative_series(log_dir / "rackA_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    rack_b = load_cumulative_series(log_dir / "rackB_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    spine_pfc = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_pause_packets_total", "tx_pause_packets_total"))
    rack_a_deadlock = load_cumulative_series(log_dir / "rackA_pfc_deadlock_aggregate.jsonl", ("deadlock_count_total",))
    rack_b_deadlock = load_cumulative_series(log_dir / "rackB_pfc_deadlock_aggregate.jsonl", ("deadlock_count_total",))
    total_pfc = combine_series([rack_a, rack_b, spine_pfc])
    total_deadlock = combine_series([rack_a_deadlock, rack_b_deadlock])
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
        "total_pfc": total_pfc,
        "rackA_deadlock": rack_a_deadlock,
        "rackB_deadlock": rack_b_deadlock,
        "total_deadlock": total_deadlock,
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
    rel_values = finite(to_float(r.get("relNs")) for r in rows)
    ctrl_start_ns = int(min(0.0, min(rel_values, default=0.0)))
    ctrl_end_ns = int(max(rel_values, default=0.0))
    if switch:
        marker = switch.get("markers", {}).get((repeat, mode))
        if marker and "start_ns" in marker:
            marker_start_ns = int(marker["start_ns"])
            if any(int(to_float(row.get("switchAligned"), 0)) == 1 for row in rows):
                return ctrl_start_ns, marker_start_ns + ctrl_start_ns, marker_start_ns + ctrl_end_ns, True
            return 0, marker_start_ns, marker_start_ns + ctrl_end_ns, True
    return ctrl_start_ns, ctrl_start_ns, ctrl_end_ns, False


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["repeat"]), str(row["mode"]))].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (to_float(r.get("relNs")), str(r.get("worker")), to_float(r.get("tNs"))))
    return groups


def downsample(rows: list[dict], limit: int = 6000) -> list[dict]:
    if len(rows) <= limit:
        return rows
    step = max(1, len(rows) // limit)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def bucket_ctrl_rows(
    rows: list[dict],
    start_ns: int,
    end_ns: int,
    bucket_ms: float,
    plot_pct: float = 99.0,
    trim_top: int = 0,
) -> list[dict]:
    if not rows or end_ns <= start_ns:
        return []
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        t_ns = int(to_float(row.get("relNs")))
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
        plot_delays = drop_top(delays, trim_top)
        delay_max = [to_float(r.get("delayMaxNs")) / 1e6 for r in group]
        by_worker: dict[str, list[float]] = defaultdict(list)
        for row in group:
            by_worker[str(row.get("worker", ""))].append(to_float(row.get("delayMeanNs")) / 1e6)
        worker_p99 = [percentile(values, 99) for values in by_worker.values()]
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
                "delay_plot_tail_ms": percentile(plot_delays, plot_pct),
                "delay_trimmed_max_ms": max(finite(plot_delays), default=math.nan),
                "delay_worker_p99_max_ms": max(finite(worker_p99), default=math.nan),
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
    plot_pct: float,
    trim_top: int,
) -> Path:
    path = output_dir / "phase6_bin_metrics.csv"
    fields = [
        "repeat",
        "mode",
        "rel_mid_s",
        "samples",
        "delay_p50_ms",
        "delay_p95_ms",
        "delay_p99_ms",
        "delay_plot_tail_ms",
        "delay_trimmed_max_ms",
        "delay_worker_p99_max_ms",
        "delay_mean_max_ms",
        "delay_max_peak_ms",
        "w_min",
        "w_max",
        "e_max",
        "wstall_max",
        "decrease_count",
        "increase_count",
        "pfc_delta",
        "deadlock_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (repeat, mode), rows in sorted(groups.items()):
            ctrl_start_ns, switch_start_ns, switch_end_ns, _marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
            ctrl_end_ns = int(max(to_float(r.get("relNs")) for r in rows))
            buckets = bucket_ctrl_rows(rows, ctrl_start_ns, ctrl_end_ns, bucket_ms, plot_pct, trim_top)
            switch_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms) if switch else []
            switch_deadlock = bucket_delta(switch["total_deadlock"], switch_start_ns, switch_end_ns, bucket_ms) if switch else []
            for idx, bucket in enumerate(buckets):
                out = {key: bucket.get(key, "") for key in fields}
                out["repeat"] = repeat
                out["mode"] = mode
                out["pfc_delta"] = switch_pfc[idx][1] if idx < len(switch_pfc) else 0.0
                out["deadlock_delta"] = switch_deadlock[idx][1] if idx < len(switch_deadlock) else 0.0
                writer.writerow(out)
    return path


def plot_group(repeat: str, mode: str, rows: list[dict], switch: Optional[dict], output_dir: Path, bucket_ms: float) -> str:
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    plot_rows = downsample(rows)
    xs = [to_float(r.get("relNs")) / 1_000_000_000.0 for r in plot_rows]

    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
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
        worker_gbps = bucket_rate(switch["worker_bytes"], switch_start_ns, switch_end_ns, bucket_ms, scale=8e-9)
        if total_pfc:
            axes[3].plot([x for x, _ in total_pfc], [y for _, y in total_pfc], label="total PFC delta/bin", color="#111827")
        if worker_gbps:
            ax2 = axes[3].twinx()
            ax2.plot([x for x, _ in worker_gbps], [y for _, y in worker_gbps], label="worker NIC Gbps", color="#17becf", alpha=0.7)
            ax2.set_ylabel("NIC Gbps")
            ax2.legend(loc="upper right")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper left")

    if switch:
        total_deadlock = bucket_delta(switch["total_deadlock"], switch_start_ns, switch_end_ns, bucket_ms)
        if total_deadlock:
            axes[4].plot(
                [x for x, _ in total_deadlock],
                [y for _, y in total_deadlock],
                label="total PFC deadlock delta/bin",
                color="#7f1d1d",
            )
    axes[4].set_ylabel("Deadlock delta")
    axes[4].legend(loc="upper right")

    for ax in axes:
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
    plot_pct: float,
    trim_top: int,
    ymax_pct: float,
    explicit_ymax_ms: Optional[float],
) -> str:
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    end_ns = int(max(to_float(r.get("relNs")) for r in rows))
    buckets = bucket_ctrl_rows(rows, start_ns, end_ns, bucket_ms, plot_pct, trim_top)
    if not buckets:
        return ""
    xs = [b["rel_mid_s"] for b in buckets]

    fig, axes = plt.subplots(5, 1, figsize=(18, 15), sharex=False)
    axes[0].plot(xs, [b["delay_p50_ms"] for b in buckets], label="POST-DONE p50", color="#9ca3af", linewidth=1.0)
    axes[0].plot(xs, [b["delay_p95_ms"] for b in buckets], label="POST-DONE p95", color="#f97316", linewidth=1.0)
    axes[0].plot(xs, [b["delay_p99_ms"] for b in buckets], label="POST-DONE p99", color="#dc2626", linewidth=1.2)
    axes[0].plot(xs, [b["delay_worker_p99_max_ms"] for b in buckets], label="max worker p99", color="#991b1b", linewidth=1.0, linestyle="--")
    axes[0].plot(xs, [b["delay_plot_tail_ms"] for b in buckets], label=f"trim-top{trim_top} p{plot_pct:g}", color="#7f1d1d", linewidth=0.9, alpha=0.75)
    axes[0].plot(xs, [b["delay_trimmed_max_ms"] for b in buckets], label=f"trim-top{trim_top} max", color="#374151", linewidth=0.8, alpha=0.65)
    y_candidates = []
    for bucket in buckets:
        y_candidates.extend(
            [
                bucket["delay_p50_ms"],
                bucket["delay_p95_ms"],
                bucket["delay_p99_ms"],
                bucket["delay_worker_p99_max_ms"],
                bucket["delay_plot_tail_ms"],
                bucket["delay_trimmed_max_ms"],
            ]
        )
    if explicit_ymax_ms is not None and explicit_ymax_ms > 0:
        y_max = explicit_ymax_ms
    else:
        y_max = percentile(y_candidates, ymax_pct)
        if not math.isfinite(y_max) or y_max <= 0:
            y_max = max(finite(y_candidates), default=1.0)
        y_max *= 1.15
    axes[0].set_ylim(bottom=0, top=max(y_max, 1.0))
    axes[0].set_ylabel("POST-DONE ms")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, [b["w_min"] for b in buckets], label="W min", color="#1d4ed8")
    axes[1].plot(xs, [b["w_max"] for b in buckets], label="W max", color="#60a5fa", alpha=0.8)
    axes[1].set_ylabel("W")
    axes[1].legend(loc="upper right")

    axes[2].plot(xs, [b["e_max"] for b in buckets], label="spike e max", color="#f59e0b")
    axes[2].plot(xs, [b["wstall_max"] for b in buckets], label="wstallRatio max", color="#16a34a")
    axes[2].set_ylabel("Controller signal")
    axes[2].legend(loc="upper right")

    if switch:
        total_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
        if total_pfc:
            axes[3].plot([x for x, _ in total_pfc], [y for _, y in total_pfc], label="total PFC delta/bin", color="#111827")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper right")

    if switch:
        total_deadlock = bucket_delta(switch["total_deadlock"], switch_start_ns, switch_end_ns, bucket_ms)
        if total_deadlock:
            axes[4].plot(
                [x for x, _ in total_deadlock],
                [y for _, y in total_deadlock],
                label="total PFC deadlock delta/bin",
                color="#7f1d1d",
            )
    axes[4].set_ylabel("Deadlock delta")
    axes[4].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[4].set_xlabel("Time since aligned Phase6 start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    fig.suptitle(
        f"{repeat} {mode}: Binned POST-DONE/W vs PFC/Deadlock "
        f"({bucket_ms:g} ms, trim-top{trim_top} p{plot_pct:g}, {alignment})"
    )
    fig.tight_layout()
    name = f"{repeat}_{mode}_binned_w_pfc_delay.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def plot_pfc_spike_overlay(
    repeat: str,
    mode: str,
    rows: list[dict],
    switch: Optional[dict],
    output_dir: Path,
    bucket_ms: float,
    plot_pct: float,
    trim_top: int,
) -> str:
    if not switch:
        return ""
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    end_ns = int(max(to_float(r.get("relNs")) for r in rows))
    buckets = bucket_ctrl_rows(rows, start_ns, end_ns, bucket_ms, plot_pct, trim_top)
    pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
    if not buckets or not pfc:
        return ""

    xs = [b["rel_mid_s"] for b in buckets]
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot([x for x, _ in pfc], [y for _, y in pfc], label="total PFC delta/bin", color="#111827", linewidth=1.4)
    ax.set_ylabel("PFC delta / bin", color="#111827")
    ax.tick_params(axis="y", labelcolor="#111827")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(xs, [b["e_max"] for b in buckets], label="spike e max", color="#f59e0b", linewidth=1.1)
    ax2.plot(xs, [b["wstall_max"] for b in buckets], label="wstallRatio max", color="#16a34a", linewidth=0.9, alpha=0.8)
    ax2.set_ylabel("controller signal", color="#92400e")
    ax2.tick_params(axis="y", labelcolor="#92400e")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    ax.set_xlabel("Time since aligned Phase6 start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    ax.set_title(f"{repeat} {mode}: PFC Delta vs Controller Spike Signal ({bucket_ms:g} ms, {alignment})")
    fig.tight_layout()
    name = f"{repeat}_{mode}_pfc_spike_overlay.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def plot_pfc_w_overlay(
    repeat: str,
    mode: str,
    rows: list[dict],
    switch: Optional[dict],
    output_dir: Path,
    bucket_ms: float,
    plot_pct: float,
    trim_top: int,
) -> str:
    if not switch:
        return ""
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    end_ns = int(max(to_float(r.get("relNs")) for r in rows))
    buckets = bucket_ctrl_rows(rows, start_ns, end_ns, bucket_ms, plot_pct, trim_top)
    pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
    if not buckets or not pfc:
        return ""

    xs = [b["rel_mid_s"] for b in buckets]
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot([x for x, _ in pfc], [y for _, y in pfc], label="total PFC delta/bin", color="#111827", linewidth=1.4)
    ax.set_ylabel("PFC delta / bin", color="#111827")
    ax.tick_params(axis="y", labelcolor="#111827")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(xs, [b["w_min"] for b in buckets], label="W min", color="#1d4ed8", linewidth=1.1)
    ax2.plot(xs, [b["w_max"] for b in buckets], label="W max", color="#60a5fa", linewidth=0.9, alpha=0.7)
    ax2.invert_yaxis()
    ax2.set_ylabel("W (inverted; lower W is stronger control)", color="#1d4ed8")
    ax2.tick_params(axis="y", labelcolor="#1d4ed8")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    ax.set_xlabel("Time since aligned Phase6 start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    ax.set_title(f"{repeat} {mode}: PFC Delta vs W Control ({bucket_ms:g} ms, {alignment})")
    fig.tight_layout()
    name = f"{repeat}_{mode}_pfc_w_overlay.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def plot_w_event_counts(
    repeat: str,
    mode: str,
    rows: list[dict],
    output_dir: Path,
    bucket_ms: float,
    plot_pct: float,
    trim_top: int,
) -> str:
    start_ns = int(min(0.0, min(finite(to_float(r.get("relNs")) for r in rows), default=0.0)))
    end_ns = int(max(to_float(r.get("relNs")) for r in rows))
    buckets = bucket_ctrl_rows(rows, start_ns, end_ns, bucket_ms, plot_pct, trim_top)
    if not buckets:
        return ""
    xs = [b["rel_mid_s"] for b in buckets]
    width = bucket_ms / 1000.0 * 0.8

    fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
    axes[0].plot(xs, [b["w_min"] for b in buckets], label="W min", color="#1d4ed8")
    axes[0].plot(xs, [b["w_max"] for b in buckets], label="W max", color="#60a5fa", alpha=0.8)
    axes[0].set_ylabel("W")
    axes[0].legend(loc="upper right")

    axes[1].bar(xs, [b["decrease_count"] for b in buckets], width=width, label="W decrease count", color="#b91c1c", alpha=0.75)
    axes[1].bar(xs, [b["increase_count"] for b in buckets], width=width * 0.45, label="W increase count", color="#047857", alpha=0.65)
    axes[1].set_ylabel("events / bin")
    axes[1].legend(loc="upper right")

    axes[2].plot(xs, [b["e_max"] for b in buckets], label="spike e max", color="#f59e0b")
    axes[2].plot(xs, [b["wstall_max"] for b in buckets], label="wstallRatio max", color="#16a34a")
    axes[2].set_ylabel("controller")
    axes[2].set_xlabel("Time since aligned Phase6 start (s)")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{repeat} {mode}: W Adjustment Events ({bucket_ms:g} ms)")
    fig.tight_layout()
    name = f"{repeat}_{mode}_w_event_counts.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def plot_switch_pfc_components(
    repeat: str,
    mode: str,
    rows: list[dict],
    switch: Optional[dict],
    output_dir: Path,
    bucket_ms: float,
) -> str:
    if not switch:
        return ""
    ctrl_start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    if switch_end_ns <= switch_start_ns:
        return ""

    pfc_series = [
        ("total PFC", "total_pfc", "#111827", "-"),
        ("rackA PFC", "rackA_pfc", "#2563eb", "-"),
        ("rackB PFC", "rackB_pfc", "#16a34a", "-"),
        ("spine PFC", "spine_pfc", "#f97316", "-"),
    ]
    deadlock_series = [
        ("rackA deadlock", "rackA_deadlock", "#7f1d1d", "-"),
        ("rackB deadlock", "rackB_deadlock", "#be123c", "--"),
        ("total deadlock", "total_deadlock", "#450a0a", ":"),
    ]
    if not any(switch.get(key) for _label, key, _color, _style in pfc_series + deadlock_series):
        return ""

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    for label, key, color, linestyle in pfc_series:
        points = bucket_delta(switch.get(key, []), switch_start_ns, switch_end_ns, bucket_ms)
        if points:
            axes[0].plot([x for x, _ in points], [y for _, y in points], label=f"{label} delta/bin", color=color, linestyle=linestyle)
    axes[0].set_ylabel("PFC delta")
    axes[0].legend(loc="upper right")

    for label, key, color, linestyle in deadlock_series:
        points = bucket_delta(switch.get(key, []), switch_start_ns, switch_end_ns, bucket_ms)
        if points:
            axes[1].plot([x for x, _ in points], [y for _, y in points], label=f"{label} delta/bin", color=color, linestyle=linestyle)
    axes[1].set_ylabel("Deadlock delta")
    axes[1].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time since aligned Phase6 start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    fig.suptitle(f"{repeat} {mode}: Switch PFC Component Breakdown ({bucket_ms:g} ms, {alignment})")
    fig.tight_layout()
    name = f"{repeat}_{mode}_switch_pfc_components.png"
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
    event_ns = int(to_float(event_row.get("relNs")))
    window_ns = int(window_sec * 1_000_000_000.0)
    channel = event_row.get("channel")
    worker = event_row.get("worker")
    local_rows = [
        r
        for r in rows
        if r.get("worker") == worker
        and r.get("channel") == channel
        and event_ns - window_ns <= int(to_float(r.get("relNs"))) <= event_ns + window_ns
    ]
    if not local_rows:
        return ""
    xs = [(to_float(r.get("relNs")) - event_ns) / 1_000_000_000.0 for r in local_rows]
    ctrl_start_ns, switch_start_ns, _switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    aligned_event_ns = switch_start_ns + (event_ns - ctrl_start_ns) if marker_aligned else event_ns

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
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
        if pfc:
            axes[3].plot([x - window_sec for x, _ in pfc], [y for _, y in pfc], label="total PFC delta/bin", color="#111827")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper right")

    if switch:
        deadlock = bucket_delta(switch["total_deadlock"], aligned_event_ns - window_ns, aligned_event_ns + window_ns, bucket_ms)
        if deadlock:
            axes[4].plot(
                [x - window_sec for x, _ in deadlock],
                [y for _, y in deadlock],
                label="total PFC deadlock delta/bin",
                color="#7f1d1d",
            )
    axes[4].set_ylabel("Deadlock delta")
    axes[4].legend(loc="upper right")

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
        "relTimeS",
        "aligned_switch_unix_ns",
        "wBefore",
        "wAfter",
        "e",
        "delayMeanMs",
        "delaySlowMs",
        "pfc_before",
        "pfc_after",
        "deadlock_before",
        "deadlock_after",
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
                rel_ns = int(to_float(row.get("relNs")))
                aligned_t_ns = switch_start_ns + (rel_ns - ctrl_start_ns) if marker_aligned else rel_ns
                out = {
                    "repeat": repeat,
                    "mode": mode,
                    "worker": row.get("worker"),
                    "channel": row.get("channel"),
                    "action": row.get("action"),
                    "tNs": t_ns,
                    "relTimeS": rel_ns / 1_000_000_000.0,
                    "aligned_switch_unix_ns": aligned_t_ns if marker_aligned else "",
                    "wBefore": row.get("wBefore"),
                    "wAfter": row.get("wAfter"),
                    "e": row.get("e"),
                    "delayMeanMs": to_float(row.get("delayMeanNs")) / 1e6,
                    "delaySlowMs": to_float(row.get("delaySlowNs")) / 1e6,
                    "pfc_before": 0.0,
                    "pfc_after": 0.0,
                    "deadlock_before": 0.0,
                    "deadlock_after": 0.0,
                    "worker_gb_before": 0.0,
                    "worker_gb_after": 0.0,
                    "cnp_before": 0.0,
                    "cnp_after": 0.0,
                }
                if switch:
                    out["pfc_before"] = window_delta(switch["total_pfc"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["pfc_after"] = window_delta(switch["total_pfc"], aligned_t_ns, aligned_t_ns + window_ns)
                    out["deadlock_before"] = window_delta(switch["total_deadlock"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["deadlock_after"] = window_delta(switch["total_deadlock"], aligned_t_ns, aligned_t_ns + window_ns)
                    out["worker_gb_before"] = window_delta(switch["worker_bytes"], aligned_t_ns - window_ns, aligned_t_ns) * 8e-9
                    out["worker_gb_after"] = window_delta(switch["worker_bytes"], aligned_t_ns, aligned_t_ns + window_ns) * 8e-9
                    out["cnp_before"] = window_delta(switch["worker_cnp"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["cnp_after"] = window_delta(switch["worker_cnp"], aligned_t_ns, aligned_t_ns + window_ns)
                writer.writerow(out)
    return path


def build_random_event_rows(
    groups: dict[tuple[str, str], list[dict]],
    switch: Optional[dict],
    window_sec: float,
    seed: int = 20260526,
) -> list[dict]:
    if not switch:
        return []
    rng = random.Random(seed)
    window_ns = int(window_sec * 1_000_000_000)
    out: list[dict] = []
    for (repeat, mode), rows in sorted(groups.items()):
        decreases = [row for row in rows if row.get("action") == "decrease"]
        if not decreases:
            continue
        ctrl_start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
        if switch_end_ns <= switch_start_ns + 2 * window_ns:
            continue
        start = switch_start_ns + window_ns
        end = switch_end_ns - window_ns
        for idx in range(len(decreases)):
            aligned_t_ns = rng.randint(start, end)
            pfc_before = window_delta(switch["total_pfc"], aligned_t_ns - window_ns, aligned_t_ns)
            pfc_after = window_delta(switch["total_pfc"], aligned_t_ns, aligned_t_ns + window_ns)
            out.append(
                {
                    "repeat": repeat,
                    "mode": mode,
                    "event_type": "random",
                    "event_idx": idx + 1,
                    "aligned_switch_unix_ns": aligned_t_ns if marker_aligned else "",
                    "relTimeS": (aligned_t_ns - switch_start_ns) / 1_000_000_000.0,
                    "pfc_before": pfc_before,
                    "pfc_after": pfc_after,
                    "pfc_after_minus_before": pfc_after - pfc_before,
                    "pfc_reduced_after": 1 if pfc_after < pfc_before else 0,
                }
            )
    return out


def write_random_event_csv(path: Path, rows: list[dict]) -> Path:
    fields = [
        "repeat",
        "mode",
        "event_type",
        "event_idx",
        "aligned_switch_unix_ns",
        "relTimeS",
        "pfc_before",
        "pfc_after",
        "pfc_after_minus_before",
        "pfc_reduced_after",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_event_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def build_w_decrease_compare_rows(event_csv: Path) -> list[dict]:
    out: list[dict] = []
    for idx, row in enumerate(read_event_csv(event_csv), start=1):
        if row.get("action") != "decrease":
            continue
        before = to_float(row.get("pfc_before"))
        after = to_float(row.get("pfc_after"))
        if not math.isfinite(before) or not math.isfinite(after):
            continue
        out.append(
            {
                "repeat": row.get("repeat", ""),
                "mode": row.get("mode", ""),
                "event_type": "w_decrease",
                "event_idx": idx,
                "aligned_switch_unix_ns": row.get("aligned_switch_unix_ns", ""),
                "relTimeS": row.get("relTimeS", ""),
                "pfc_before": before,
                "pfc_after": after,
                "pfc_after_minus_before": after - before,
                "pfc_reduced_after": 1 if after < before else 0,
            }
        )
    return out


def summarize_event_effect(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("event_type", ""))].append(row)
    out: dict[str, dict[str, float]] = {}
    for event_type, group in grouped.items():
        deltas = [to_float(row.get("pfc_after_minus_before")) for row in group]
        reduced = [to_float(row.get("pfc_reduced_after"), 0.0) for row in group]
        before = [to_float(row.get("pfc_before")) for row in group]
        after = [to_float(row.get("pfc_after")) for row in group]
        vals = finite(deltas)
        red_vals = finite(reduced)
        out[event_type] = {
            "events": float(len(group)),
            "median_before": percentile(before, 50),
            "median_after": percentile(after, 50),
            "median_after_minus_before": percentile(vals, 50),
            "p95_after_minus_before": percentile(vals, 95),
            "reduction_ratio": sum(1.0 for value in red_vals if value > 0.0) / len(red_vals) if red_vals else math.nan,
        }
    return out


def write_event_effect_summary(path: Path, rows: list[dict]) -> Path:
    summary = summarize_event_effect(rows)
    fields = [
        "event_type",
        "events",
        "median_before",
        "median_after",
        "median_after_minus_before",
        "p95_after_minus_before",
        "reduction_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for event_type, row in sorted(summary.items()):
            writer.writerow({"event_type": event_type, **row})
    return path


def plot_w_vs_random_pfc_effect(rows: list[dict], output_dir: Path) -> list[str]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get("pfc_after_minus_before"))
        if math.isfinite(value):
            grouped[str(row.get("event_type", ""))].append(value)
    order = [name for name in ("w_decrease", "random") if grouped.get(name)]
    if not order:
        return []

    names: list[str] = []
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([grouped[name] for name in order], labels=order, showfliers=False)
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_ylabel("PFC after - before")
    ax.set_title("PFC Change After Event: W Decrease vs Random")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    name = "pfc_effect_w_decrease_vs_random_boxplot.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    names.append(name)

    fig, ax = plt.subplots(figsize=(10, 5))
    for event_type, color in (("w_decrease", "#1d4ed8"), ("random", "#6b7280")):
        vals = grouped.get(event_type, [])
        if vals:
            ax.hist(vals, bins=60, alpha=0.55, label=event_type, color=color)
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_xlabel("PFC after - before")
    ax.set_ylabel("event count")
    ax.set_title("Distribution of PFC Change After Event")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    name = "pfc_effect_w_decrease_vs_random_histogram.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    names.append(name)
    return names


def write_html(
    output_dir: Path,
    raw_images: list[str],
    bucket_images: list[str],
    pfc_spike_images: list[str],
    pfc_w_images: list[str],
    pfc_effect_images: list[str],
    w_event_images: list[str],
    component_images: list[str],
    event_images: list[str],
    event_csv: Path,
    random_event_csv: Path,
    event_effect_summary_csv: Path,
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
<p>W decrease/increase events are separated into dedicated event-count plots so the main latency/PFC timeline remains readable. If <code>CTRL_ANCHOR</code> exists, NCCL monotonic time is converted to unix time and aligned to switch <code>mode_start</code>. Without anchor logs, the reporter falls back to first <code>CTRL_EPOCH</code> relative alignment.</p>
<p>Analysis-bin metrics are in <code>{html.escape(bucket_csv.name)}</code>. Event before/after network deltas are in <code>{html.escape(event_csv.name)}</code>. Random-event baseline is in <code>{html.escape(random_event_csv.name)}</code>; W-vs-random effect summary is in <code>{html.escape(event_effect_summary_csv.name)}</code>.</p>
<p>PFC deadlock is plotted from <code>rackA_pfc_deadlock_aggregate.jsonl</code> and <code>rackB_pfc_deadlock_aggregate.jsonl</code> using <code>deadlock_count_total</code> deltas.</p>
<p>The main timeline keeps <code>total_pfc = rackA + rackB + spine</code>. The component breakdown below plots rackA, rackB, and spine separately so switch-local congestion can be distinguished from total network pressure.</p>
{image_section("PFC Delta vs Controller Spike Signal", pfc_spike_images)}
{image_section("PFC Delta vs W Control", pfc_w_images)}
{image_section("PFC Change After W Decrease vs Random Events", pfc_effect_images)}
{image_section("Binned Timeline: POST-DONE p99/max, W, PFC, Deadlock", bucket_images)}
{image_section("W Adjustment Event Counts", w_event_images)}
{image_section("Switch PFC Component Breakdown", component_images)}
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
    if args.bin_ms is not None:
        args.bucket_ms = args.bin_ms

    workers = None if args.all_workers else {item.strip() for item in args.workers.split(",") if item.strip()}
    rows = load_phase6_epochs(run_root, workers)
    switch = load_switch_bundle(run_root, args.switch_log_dir)
    switch_aligned_rows = apply_switch_relative_times(rows, switch)
    groups = group_rows(rows)
    event_csv = write_event_csv(groups, switch, output_dir, args.event_window_sec)
    random_event_rows = build_random_event_rows(groups, switch, args.event_window_sec)
    random_event_csv = write_random_event_csv(output_dir / "phase6_random_event_network_windows.csv", random_event_rows)
    event_effect_rows = build_w_decrease_compare_rows(event_csv) + random_event_rows
    event_effect_summary_csv = write_event_effect_summary(output_dir / "phase6_w_vs_random_pfc_effect_summary.csv", event_effect_rows)
    bucket_csv = write_bucket_csv(groups, switch, output_dir, args.bucket_ms, args.delay_plot_pct, args.delay_trim_top)
    raw_images = []
    if not args.skip_raw:
        raw_images = [
            plot_group(repeat, mode, group, switch, output_dir, args.bucket_ms)
            for (repeat, mode), group in sorted(groups.items())
            if group
        ]
    bucket_images = [
        plot_bucket_group(
            repeat,
            mode,
            group,
            switch,
            output_dir,
            args.bucket_ms,
            args.delay_plot_pct,
            args.delay_trim_top,
            args.delay_ymax_pct,
            args.delay_ymax_ms,
        )
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    bucket_images = [name for name in bucket_images if name]
    pfc_spike_images = [
        plot_pfc_spike_overlay(
            repeat,
            mode,
            group,
            switch,
            output_dir,
            args.bucket_ms,
            args.delay_plot_pct,
            args.delay_trim_top,
        )
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    pfc_spike_images = [name for name in pfc_spike_images if name]
    pfc_w_images = [
        plot_pfc_w_overlay(
            repeat,
            mode,
            group,
            switch,
            output_dir,
            args.bucket_ms,
            args.delay_plot_pct,
            args.delay_trim_top,
        )
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    pfc_w_images = [name for name in pfc_w_images if name]
    pfc_effect_images = plot_w_vs_random_pfc_effect(event_effect_rows, output_dir)
    w_event_images = [
        plot_w_event_counts(
            repeat,
            mode,
            group,
            output_dir,
            args.bucket_ms,
            args.delay_plot_pct,
            args.delay_trim_top,
        )
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    w_event_images = [name for name in w_event_images if name]
    component_images = [
        plot_switch_pfc_components(repeat, mode, group, switch, output_dir, args.bucket_ms)
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    component_images = [name for name in component_images if name]

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

    html_path = write_html(
        output_dir,
        raw_images,
        bucket_images,
        pfc_spike_images,
        pfc_w_images,
        pfc_effect_images,
        w_event_images,
        component_images,
        event_images,
        event_csv,
        random_event_csv,
        event_effect_summary_csv,
        bucket_csv,
        switch,
    )
    worker_msg = "all" if workers is None else ",".join(sorted(workers))
    anchor_rows = sum(1 for row in rows if int(to_float(row.get("anchorAligned"), 0)) == 1)
    print(
        f"[phase6-network] rows={len(rows)} groups={len(groups)} workers={worker_msg} "
        f"switch={bool(switch)} anchor_rows={anchor_rows} switch_aligned_rows={switch_aligned_rows} html={html_path}"
    )


if __name__ == "__main__":
    main()
