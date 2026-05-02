#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


PHASE1_EVENT_RE = re.compile(r"PHASE1 event=(?P<event>\S+) (?P<body>.*)")
LOG_TS_PATTERNS = (
    re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"),
    re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"),
)
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
PROGRESS_BINS = 50
SAMPLED_PROGRESS_BINS = (10, 20, 30)


def reporter_log(message: str) -> None:
    print(f"[phase1-reporter] {message}", flush=True)


@dataclass
class RepeatData:
    mode: str
    experiment: str
    repeat_name: str
    repeat_index: int
    worker: str
    rank: int
    payload_mb: float
    worker_dir: Path
    repeat_dir: Path
    summary_path: Path
    trace_path: Path
    launcher_log_path: Path
    validation_path: Optional[Path]


@dataclass
class Phase1Event:
    event: str
    phase: str
    ts_unix_ns: Optional[int]
    ts_mono_ns: Optional[int]
    rank: int
    peer: int
    channel: int
    slot: int
    base: int
    nsteps: int
    posted: int
    received: int
    transmitted: int
    done: int
    logical_posted: int
    logical_received: int
    logical_transmitted: int
    logical_done: int
    occ_pd: int
    occ_pr: int
    occ_tr: int
    max_depth: int
    w_cfg: float
    allow_boundary: int
    stall_reason: str
    slice_steps: int
    chunk_steps: int
    nsubs: int
    log_ts: Optional[str]


@dataclass
class RepeatSwitchMetrics:
    total_delta: Optional[float]
    severity: Optional[float]
    per_switch_delta: Dict[str, float]


@dataclass
class RepeatCtsMetrics:
    total_posts: int
    total_wstalls: int
    wstall_by_reason: Dict[str, int]
    mean_occ_pr_at_post: Optional[float]
    mean_occ_pr_at_wstall: Optional[float]
    posts_by_worker: Dict[str, int]
    posts_by_channel: Dict[int, int]
    progress_post_counts: List[int]
    progress_occ_pr: List[Optional[float]]
    progress_channel_counts: Dict[int, Dict[int, int]]
    exact_step_post_counts: Dict[int, int]
    exact_step_wstall_counts: Dict[int, int]
    exact_step_channel_counts: Dict[int, Dict[int, int]]
    exact_step_occ_pr_post: Dict[int, Optional[float]]
    exact_step_latency_ms: Dict[int, Optional[float]]
    exact_step_throughput_gbps: Dict[int, Optional[float]]
    exact_ts_available: bool
    configured_w: Optional[float]
    observed_max_occ_pr: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate phase1 plots and HTML report.")
    parser.add_argument("--input", required=True, help="Phase1 RUN_ID directory")
    parser.add_argument("--output-dir", required=True, help="Output report directory")
    return parser.parse_args()


def mode_sort_key(mode: str) -> Tuple[int, float, str]:
    if mode == "STOCK":
        return (0, -1.0, mode)
    if mode.startswith("W"):
        try:
            return (1, float(mode[1:].replace("_", ".")), mode)
        except ValueError:
            return (2, math.inf, mode)
    return (2, math.inf, mode)


def experiment_sort_key(experiment: str) -> Tuple[int, str]:
    order = {
        "allreduce_ring": 0,
        "allreduce_tree": 1,
        "treeallreduce": 1,
        "alltoall": 2,
        "alltoall_auto": 2,
    }
    return (order.get(experiment, 99), experiment)


def safe_median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def percentile(sorted_values: List[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def iqr_bounds(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = sorted(float(v) for v in values if v is not None)
    return percentile(vals, 0.25), percentile(vals, 0.75)


def throughput_gbps(payload_mb: float, duration_ms: float) -> float:
    if duration_ms <= 0:
        return 0.0
    payload_bytes = payload_mb * 1024.0 * 1024.0
    return payload_bytes / (duration_ms / 1000.0) / 1e9


def parse_repeat_index(repeat_name: str) -> int:
    m = re.search(r"(\d+)$", repeat_name)
    return int(m.group(1)) if m else 0


def parse_mode_w(mode: str) -> Optional[float]:
    if mode.startswith("W"):
        try:
            return float(mode[1:].replace("_", "."))
        except ValueError:
            return None
    return None


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(run_root: Path) -> Dict[str, Dict[str, Dict[str, List[RepeatData]]]]:
    discovered: Dict[str, Dict[str, Dict[str, List[RepeatData]]]] = {}
    for mode_dir in sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: mode_sort_key(p.name)):
        mode = mode_dir.name
        for experiment_dir in sorted((p for p in mode_dir.iterdir() if p.is_dir()), key=lambda p: experiment_sort_key(p.name)):
            experiment = experiment_dir.name
            for repeat_dir in sorted((p for p in experiment_dir.iterdir() if p.is_dir())):
                validation_path = repeat_dir / "stock_probe_validation.json"
                repeat_index = parse_repeat_index(repeat_dir.name)
                for worker_dir in sorted(p for p in repeat_dir.iterdir() if p.is_dir() and p.name.startswith("worker")):
                    summary_files = sorted(worker_dir.glob("rank*_summary.json"))
                    trace_files = sorted(worker_dir.glob("rank*_step_trace.jsonl"))
                    launcher_log_path = worker_dir / "worker_launcher.log"
                    if not summary_files or not trace_files:
                        continue
                    summary_path = summary_files[0]
                    trace_path = trace_files[0]
                    rank_str = summary_path.stem.split("_", 1)[0].replace("rank", "")
                    rank = int(rank_str)
                    summary = read_json(summary_path)
                    payload_mb = float(summary.get("payload_mb", 0))
                    repeat_data = RepeatData(
                        mode=mode,
                        experiment=experiment,
                        repeat_name=repeat_dir.name,
                        repeat_index=repeat_index,
                        worker=worker_dir.name,
                        rank=rank,
                        payload_mb=payload_mb,
                        worker_dir=worker_dir,
                        repeat_dir=repeat_dir,
                        summary_path=summary_path,
                        trace_path=trace_path,
                        launcher_log_path=launcher_log_path,
                        validation_path=validation_path if validation_path.exists() else None,
                    )
                    discovered.setdefault(experiment, {}).setdefault(mode, {}).setdefault(repeat_dir.name, []).append(repeat_data)
    return discovered


@lru_cache(maxsize=None)
def load_trace(path_str: str) -> List[dict]:
    path = Path(path_str)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_log_ts(line: str) -> Optional[str]:
    for pattern in LOG_TS_PATTERNS:
        m = pattern.search(line)
        if m:
            return m.group("ts")
    return None


@lru_cache(maxsize=None)
def load_phase1_events(path_str: str) -> List[Phase1Event]:
    path = Path(path_str)
    if not path.exists():
        return []
    events: List[Phase1Event] = []
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            m = PHASE1_EVENT_RE.search(line)
            if not m:
                continue
            body = m.group("body")
            kv = {match.group(1): match.group(2) for match in KV_RE.finditer(body)}
            try:
                events.append(
                    Phase1Event(
                        event=m.group("event"),
                        phase=kv.get("phase", "-"),
                        ts_unix_ns=int(kv["ts_unix_ns"]) if "ts_unix_ns" in kv else None,
                        ts_mono_ns=int(kv["ts_mono_ns"]) if "ts_mono_ns" in kv else None,
                        rank=int(kv.get("rank", "0")),
                        peer=int(kv.get("peer", "0")),
                        channel=int(kv.get("channel", "0")),
                        slot=int(kv.get("slot", "0")),
                        base=int(kv.get("base", "0")),
                        nsteps=int(kv.get("nsteps", "0")),
                        posted=int(kv.get("posted", "0")),
                        received=int(kv.get("received", "0")),
                        transmitted=int(kv.get("transmitted", "0")),
                        done=int(kv.get("done", "0")),
                        logical_posted=int(kv.get("logicalPosted", "0")),
                        logical_received=int(kv.get("logicalReceived", "0")),
                        logical_transmitted=int(kv.get("logicalTransmitted", "0")),
                        logical_done=int(kv.get("logicalDone", "0")),
                        occ_pd=int(kv.get("occPd", "0")),
                        occ_pr=int(kv.get("occPr", "0")),
                        occ_tr=int(kv.get("occTr", "0")),
                        max_depth=int(kv.get("maxDepth", "0")),
                        w_cfg=float(kv.get("wCfg", "0")),
                        allow_boundary=int(kv.get("allowBoundary", "0")),
                        stall_reason=kv.get("stallReason", "-"),
                        slice_steps=int(kv.get("sliceSteps", "0")),
                        chunk_steps=int(kv.get("chunkSteps", "0")),
                        nsubs=int(kv.get("nsubs", "0")),
                        log_ts=extract_log_ts(line),
                    )
                )
            except ValueError:
                continue
    return events


def per_repeat_step_series(repeat_items: List[RepeatData]) -> Dict[int, Dict[str, float]]:
    by_step: Dict[int, Dict[str, List[float]]] = {}
    payload_mb = repeat_items[0].payload_mb if repeat_items else 0.0
    for item in repeat_items:
        for row in load_trace(str(item.trace_path)):
            if row.get("phase") != "steady":
                continue
            step_idx = int(row["step_index"])
            latency_ms = float(row["duration_ms"])
            thr = throughput_gbps(payload_mb, latency_ms)
            slot = by_step.setdefault(step_idx, {"latency_ms": [], "throughput_gbps": []})
            slot["latency_ms"].append(latency_ms)
            slot["throughput_gbps"].append(thr)
    result: Dict[int, Dict[str, float]] = {}
    for step_idx, slot in sorted(by_step.items()):
        result[step_idx] = {
            "latency_ms": safe_median(slot["latency_ms"]) or 0.0,
            "throughput_gbps": safe_median(slot["throughput_gbps"]) or 0.0,
        }
    return result


def per_repeat_overall(repeat_items: List[RepeatData]) -> Dict[str, float]:
    latencies: List[float] = []
    throughputs: List[float] = []
    payload_mb = repeat_items[0].payload_mb if repeat_items else 0.0
    for item in repeat_items:
        for row in load_trace(str(item.trace_path)):
            if row.get("phase") != "steady":
                continue
            latency_ms = float(row["duration_ms"])
            latencies.append(latency_ms)
            throughputs.append(throughput_gbps(payload_mb, latency_ms))
    sorted_lat = sorted(latencies)
    return {
        "latency_median_ms": safe_median(latencies) or 0.0,
        "latency_mean_ms": safe_mean(latencies) or 0.0,
        "latency_p95_ms": percentile(sorted_lat, 0.95) or 0.0,
        "throughput_median_gbps": safe_median(throughputs) or 0.0,
        "throughput_mean_gbps": safe_mean(throughputs) or 0.0,
    }


def summarize_validation(validation_path: Optional[Path]) -> Tuple[str, str]:
    if validation_path is None or not validation_path.exists():
        return ("-", "no validation file")
    payload = read_json(validation_path)
    ok = bool(payload.get("ok", False))
    workers = payload.get("workers", [])
    failed = [w.get("worker", "?") for w in workers if not w.get("ok", False)]
    return ("OK" if ok else "FAIL", ",".join(failed) if failed else "-")


def make_mode_palette(modes: List[str]) -> Dict[str, str]:
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    palette: Dict[str, str] = {}
    offset = 0
    for mode in modes:
        if mode == "STOCK":
            palette[mode] = "#222222"
        else:
            palette[mode] = base[offset % len(base)]
            offset += 1
    return palette


def parse_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip("'").strip('"')
    return data


def parse_marker_message(message: str) -> Dict[str, str]:
    return {match.group(1): match.group(2) for match in KV_RE.finditer(message)}


@lru_cache(maxsize=None)
def load_switch_windows(switch_log_dir_str: str) -> Dict[Tuple[str, str, int], Tuple[Optional[int], Optional[int]]]:
    switch_log_dir = Path(switch_log_dir_str)
    markers_path = switch_log_dir / "markers.jsonl"
    windows: Dict[Tuple[str, str, int], Dict[str, Optional[int]]] = defaultdict(lambda: {"start": None, "end": None})
    if not markers_path.exists():
        return {}
    with markers_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            marker = row.get("marker")
            if marker not in {"mode_start", "mode_end"}:
                continue
            fields = parse_marker_message(str(row.get("message", "")))
            experiment = fields.get("experiment")
            mode = fields.get("mode")
            repeat = fields.get("repeat")
            if not experiment or not mode or not repeat:
                continue
            try:
                repeat_index = int(repeat)
            except ValueError:
                continue
            key = (experiment, mode, repeat_index)
            if marker == "mode_start":
                windows[key]["start"] = int(row.get("ts_unix_ns", 0))
            else:
                windows[key]["end"] = int(row.get("ts_unix_ns", 0))
    return {key: (value["start"], value["end"]) for key, value in windows.items()}


@lru_cache(maxsize=None)
def load_switch_aggregates(switch_log_dir_str: str) -> Dict[str, List[dict]]:
    switch_log_dir = Path(switch_log_dir_str)
    samples_by_switch: Dict[str, List[dict]] = defaultdict(list)
    for path in sorted(switch_log_dir.glob("*_pfc_aggregate.jsonl")):
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                switch_name = str(row.get("switch", path.stem))
                samples_by_switch[switch_name].append(row)
    for switch_name in samples_by_switch:
        samples_by_switch[switch_name].sort(key=lambda row: int(row.get("ts_mid_unix_ns", 0)))
    return dict(samples_by_switch)


def switch_metrics_for_interval(samples_by_switch: Dict[str, List[dict]], start_ns: int, end_ns: int) -> RepeatSwitchMetrics:
    per_switch_delta: Dict[str, float] = {}
    severities: List[float] = []
    for switch_name, samples in samples_by_switch.items():
        if not samples:
            continue
        before = None
        inside = []
        prev = None
        max_burst = 0.0
        for row in samples:
            ts = int(row.get("ts_mid_unix_ns", 0))
            value = row.get("rx_pause_total")
            if value is None:
                prev = row
                continue
            value = float(value)
            if ts < start_ns:
                before = row
            if start_ns <= ts <= end_ns:
                inside.append(row)
                if prev is not None:
                    prev_val = prev.get("rx_pause_total")
                    if prev_val is not None:
                        max_burst = max(max_burst, value - float(prev_val))
            prev = row
        if not inside:
            continue
        start_row = before or inside[0]
        end_row = inside[-1]
        start_val = start_row.get("rx_pause_total")
        end_val = end_row.get("rx_pause_total")
        if start_val is None or end_val is None:
            continue
        delta = max(0.0, float(end_val) - float(start_val))
        per_switch_delta[switch_name] = delta
        severities.append(max(0.0, max_burst))
    if not per_switch_delta:
        return RepeatSwitchMetrics(total_delta=None, severity=None, per_switch_delta={})
    return RepeatSwitchMetrics(
        total_delta=sum(per_switch_delta.values()),
        severity=sum(severities) if severities else None,
        per_switch_delta=per_switch_delta,
    )


@lru_cache(maxsize=None)
def load_repeat_switch_metrics(repeat_dir_str: str, experiment: str, mode: str, repeat_index: int) -> RepeatSwitchMetrics:
    repeat_dir = Path(repeat_dir_str)
    meta = parse_env_file(repeat_dir / "switch_logger_meta.env")
    switch_log_dir = meta.get("SWITCH_LOG_DIR")
    if not switch_log_dir:
        return RepeatSwitchMetrics(total_delta=None, severity=None, per_switch_delta={})
    windows = load_switch_windows(switch_log_dir)
    window = windows.get((experiment, mode, repeat_index))
    if not window or window[0] is None or window[1] is None:
        return RepeatSwitchMetrics(total_delta=None, severity=None, per_switch_delta={})
    samples_by_switch = load_switch_aggregates(switch_log_dir)
    return switch_metrics_for_interval(samples_by_switch, int(window[0]), int(window[1]))


def build_progress_bins(events: List[Phase1Event], bins: int = PROGRESS_BINS) -> Tuple[List[int], List[Optional[float]], Dict[int, Dict[int, int]]]:
    posts = [event for event in events if event.event == "PROXY_RECV_POST" and event.phase == "steady"]
    if not posts:
        return [0] * bins, [None] * bins, {idx: {} for idx in range(bins)}
    counts = [0] * bins
    occ_pr_by_bin: List[List[float]] = [[] for _ in range(bins)]
    channel_counts: Dict[int, Dict[int, int]] = {idx: defaultdict(int) for idx in range(bins)}
    total = len(posts)
    for idx, event in enumerate(posts):
        bin_idx = min(bins - 1, int(idx * bins / total))
        counts[bin_idx] += 1
        occ_pr_by_bin[bin_idx].append(float(event.occ_pr))
        channel_counts[bin_idx][event.channel] += 1
    occ_pr = [safe_median(values) for values in occ_pr_by_bin]
    return counts, occ_pr, {idx: dict(counter) for idx, counter in channel_counts.items()}


def build_exact_step_metrics(trace_rows: List[dict], events: List[Phase1Event]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, Dict[int, int]], Dict[int, Optional[float]], Dict[int, Optional[float]], Dict[int, Optional[float]], bool]:
    steady_rows = [row for row in trace_rows if row.get("phase") == "steady"]
    if not steady_rows:
        return {}, {}, {}, {}, {}, {}, False
    if any(event.ts_unix_ns is None for event in events if event.phase == "steady"):
        return {}, {}, {}, {}, {}, {}, False

    step_post_counts: Dict[int, int] = Counter()
    step_wstall_counts: Dict[int, int] = Counter()
    step_channel_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    step_occ_pr_values: Dict[int, List[float]] = defaultdict(list)
    step_latency_ms: Dict[int, float] = {}
    step_throughput_gbps: Dict[int, float] = {}

    sorted_rows = sorted(steady_rows, key=lambda row: int(row.get("step_index", 0)))
    payload_mb = float(sorted_rows[0].get("payload_mb", 0))
    intervals = []
    for row in sorted_rows:
        step_idx = int(row["step_index"])
        start_ns = int(row["ts_start_unix_ns"])
        end_ns = int(row["ts_end_unix_ns"])
        intervals.append((step_idx, start_ns, end_ns))
        latency_ms = float(row["duration_ms"])
        step_latency_ms[step_idx] = latency_ms
        step_throughput_gbps[step_idx] = throughput_gbps(payload_mb, latency_ms)

    for event in events:
        if event.phase != "steady" or event.ts_unix_ns is None:
            continue
        event_ts = event.ts_unix_ns
        matched_step = None
        for step_idx, start_ns, end_ns in intervals:
            if start_ns <= event_ts <= end_ns:
                matched_step = step_idx
                break
        if matched_step is None:
            continue
        if event.event == "PROXY_RECV_POST":
            step_post_counts[matched_step] += 1
            step_channel_counts[matched_step][event.channel] += 1
            step_occ_pr_values[matched_step].append(float(event.occ_pr))
        elif event.event == "PROXY_RECV_WSTALL":
            step_wstall_counts[matched_step] += 1

    step_occ_pr = {step: safe_median(vals) for step, vals in step_occ_pr_values.items()}
    return (
        dict(step_post_counts),
        dict(step_wstall_counts),
        {step: dict(counter) for step, counter in step_channel_counts.items()},
        step_occ_pr,
        step_latency_ms,
        step_throughput_gbps,
        True,
    )


@lru_cache(maxsize=None)
def load_repeat_cts_metrics(repeat_dir_str: str, experiment: str, mode: str, repeat_name: str) -> RepeatCtsMetrics:
    repeat_dir = Path(repeat_dir_str)
    posts_by_worker: Dict[str, int] = {}
    posts_by_channel: Counter[int] = Counter()
    progress_post_counts = [0] * PROGRESS_BINS
    progress_occ_pr_lists: List[List[float]] = [[] for _ in range(PROGRESS_BINS)]
    progress_channel_counts: Dict[int, Counter[int]] = {idx: Counter() for idx in range(PROGRESS_BINS)}
    exact_step_post_counts: Counter[int] = Counter()
    exact_step_wstall_counts: Counter[int] = Counter()
    exact_step_channel_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    exact_step_occ_pr_lists: Dict[int, List[float]] = defaultdict(list)
    exact_step_latency_values: Dict[int, List[float]] = defaultdict(list)
    exact_step_throughput_values: Dict[int, List[float]] = defaultdict(list)
    exact_ts_available = True
    total_posts = 0
    total_wstalls = 0
    wstall_by_reason: Counter[str] = Counter()
    occ_pr_at_post: List[float] = []
    occ_pr_at_wstall: List[float] = []
    configured_ws: List[float] = []
    observed_max_occ_pr: Optional[float] = None

    for worker_dir in sorted(p for p in repeat_dir.iterdir() if p.is_dir() and p.name.startswith("worker")):
        log_path = worker_dir / "worker_launcher.log"
        trace_files = sorted(worker_dir.glob("rank*_step_trace.jsonl"))
        events = load_phase1_events(str(log_path))
        post_events = [event for event in events if event.event == "PROXY_RECV_POST" and event.phase == "steady"]
        wstall_events = [event for event in events if event.event == "PROXY_RECV_WSTALL" and event.phase == "steady"]
        posts_by_worker[worker_dir.name] = len(post_events)
        total_posts += len(post_events)
        total_wstalls += len(wstall_events)
        for event in post_events:
            posts_by_channel[event.channel] += 1
            occ_pr_at_post.append(float(event.occ_pr))
            observed_max_occ_pr = max(observed_max_occ_pr or float(event.occ_pr), float(event.occ_pr))
            if event.w_cfg > 0:
                configured_ws.append(event.w_cfg)
        for event in wstall_events:
            occ_pr_at_wstall.append(float(event.occ_pr))
            wstall_by_reason[event.stall_reason] += 1
            if event.w_cfg > 0:
                configured_ws.append(event.w_cfg)

        counts, occ_pr_bins, channel_bins = build_progress_bins(events)
        for idx in range(PROGRESS_BINS):
            progress_post_counts[idx] += counts[idx]
            if occ_pr_bins[idx] is not None:
                progress_occ_pr_lists[idx].append(float(occ_pr_bins[idx]))
            for channel, count in channel_bins[idx].items():
                progress_channel_counts[idx][channel] += count

        if trace_files:
            step_posts, step_wstalls, step_channels, step_occ_pr, step_latency_ms, step_throughput_gbps, has_exact_ts = build_exact_step_metrics(
                load_trace(str(trace_files[0])),
                events,
            )
            exact_ts_available = exact_ts_available and has_exact_ts
            for step_idx, value in step_posts.items():
                exact_step_post_counts[step_idx] += value
            for step_idx, value in step_wstalls.items():
                exact_step_wstall_counts[step_idx] += value
            for step_idx, channels in step_channels.items():
                for channel, count in channels.items():
                    exact_step_channel_counts[step_idx][channel] += count
            for step_idx, value in step_occ_pr.items():
                if value is not None:
                    exact_step_occ_pr_lists[step_idx].append(float(value))
            for step_idx, value in step_latency_ms.items():
                exact_step_latency_values[step_idx].append(float(value))
            for step_idx, value in step_throughput_gbps.items():
                exact_step_throughput_values[step_idx].append(float(value))
        else:
            exact_ts_available = False

    progress_occ_pr = [safe_median(values) for values in progress_occ_pr_lists]
    return RepeatCtsMetrics(
        total_posts=total_posts,
        total_wstalls=total_wstalls,
        wstall_by_reason=dict(wstall_by_reason),
        mean_occ_pr_at_post=safe_mean(occ_pr_at_post),
        mean_occ_pr_at_wstall=safe_mean(occ_pr_at_wstall),
        posts_by_worker=posts_by_worker,
        posts_by_channel=dict(posts_by_channel),
        progress_post_counts=progress_post_counts,
        progress_occ_pr=progress_occ_pr,
        progress_channel_counts={idx: dict(counter) for idx, counter in progress_channel_counts.items()},
        exact_step_post_counts=dict(exact_step_post_counts),
        exact_step_wstall_counts=dict(exact_step_wstall_counts),
        exact_step_channel_counts={step: dict(counter) for step, counter in exact_step_channel_counts.items()},
        exact_step_occ_pr_post={step: safe_median(vals) for step, vals in exact_step_occ_pr_lists.items()},
        exact_step_latency_ms={step: safe_median(vals) for step, vals in exact_step_latency_values.items()},
        exact_step_throughput_gbps={step: safe_median(vals) for step, vals in exact_step_throughput_values.items()},
        exact_ts_available=exact_ts_available,
        configured_w=safe_median(configured_ws),
        observed_max_occ_pr=observed_max_occ_pr,
    )


def mode_metrics_for_experiment(mode_repeats: Dict[str, Dict[str, List[RepeatData]]], metric: str) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for mode, repeats in mode_repeats.items():
        values[mode] = [per_repeat_overall(items)[metric] for _, items in sorted(repeats.items())]
    return values


def mode_switch_metrics_for_experiment(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], metric: str) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = defaultdict(list)
    for mode, repeats in mode_repeats.items():
        for repeat_name, items in sorted(repeats.items()):
            sample = items[0]
            switch_metrics = load_repeat_switch_metrics(str(sample.repeat_dir), experiment, mode, sample.repeat_index)
            value = switch_metrics.total_delta if metric == "total_delta" else switch_metrics.severity
            if value is not None:
                values[mode].append(value)
    return dict(values)


def plot_step_overlay(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path, metric: str) -> str:
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    palette = make_mode_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 6))
    ylabel = "Latency (ms)" if metric == "latency_ms" else "Throughput (GB/s)"
    title = f"{experiment} Step {'Latency' if metric == 'latency_ms' else 'Throughput'} Overlay"

    for mode in modes:
        repeat_series = [per_repeat_step_series(items) for _, items in sorted(mode_repeats[mode].items())]
        all_steps = sorted({step for series in repeat_series for step in series.keys()})
        xs: List[int] = []
        medians: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []
        for step_idx in all_steps:
            vals = [series[step_idx][metric] for series in repeat_series if step_idx in series]
            if not vals:
                continue
            q1, q3 = iqr_bounds(vals)
            xs.append(step_idx)
            medians.append(safe_median(vals) or 0.0)
            lowers.append(q1 if q1 is not None else medians[-1])
            uppers.append(q3 if q3 is not None else medians[-1])
        ax.plot(xs, medians, label=mode, color=palette[mode], linewidth=2)
        ax.fill_between(xs, lowers, uppers, color=palette[mode], alpha=0.16)

    ax.set_title(title)
    ax.set_xlabel("Step Index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    filename = f"{experiment}_step_{'latency' if metric == 'latency_ms' else 'throughput'}_overlay.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_summary_box(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path, metric: str) -> str:
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    palette = make_mode_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 6))

    series: List[List[float]] = []
    labels: List[str] = []
    colors: List[str] = []
    for mode in modes:
        vals = [per_repeat_overall(items)[metric] for _, items in sorted(mode_repeats[mode].items())]
        series.append(vals)
        labels.append(mode)
        colors.append(palette[mode])

    box = ax.boxplot(series, patch_artist=True, tick_labels=labels, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(2)

    ylabel = "Latency (ms)" if metric == "latency_median_ms" else "Throughput (GB/s)"
    title = f"{experiment} {'Latency' if metric == 'latency_median_ms' else 'Throughput'} by Mode"
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    filename = f"{experiment}_{'latency' if metric == 'latency_median_ms' else 'throughput'}_boxplot.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_delta_vs_stock(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> str:
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    non_stock_modes = [mode for mode in modes if mode != "STOCK"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    stock_repeats = {
        repeat_name: per_repeat_overall(items)
        for repeat_name, items in mode_repeats.get("STOCK", {}).items()
    }

    latency_vals: List[float] = []
    throughput_vals: List[float] = []
    labels: List[str] = []
    for mode in non_stock_modes:
        lat_deltas: List[float] = []
        thr_deltas: List[float] = []
        for repeat_name, items in sorted(mode_repeats[mode].items()):
            if repeat_name not in stock_repeats:
                continue
            cur = per_repeat_overall(items)
            stock = stock_repeats[repeat_name]
            if stock["latency_median_ms"] > 0:
                lat_deltas.append((cur["latency_median_ms"] / stock["latency_median_ms"] - 1.0) * 100.0)
            if stock["throughput_median_gbps"] > 0:
                thr_deltas.append((cur["throughput_median_gbps"] / stock["throughput_median_gbps"] - 1.0) * 100.0)
        labels.append(mode)
        latency_vals.append(safe_median(lat_deltas) or 0.0)
        throughput_vals.append(safe_median(thr_deltas) or 0.0)

    x = list(range(len(labels)))
    axes[0].axhline(0.0, color="#333333", linewidth=1)
    axes[0].bar(x, latency_vals, color="#d62728", alpha=0.7)
    axes[0].set_ylabel("Latency Delta vs STOCK (%)")
    axes[0].set_title(f"{experiment} Delta vs STOCK")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].bar(x, throughput_vals, color="#2ca02c", alpha=0.7)
    axes[1].set_ylabel("Throughput Delta vs STOCK (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    filename = f"{experiment}_delta_vs_stock.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_vs_w_lines(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path, metric: str) -> str:
    modes = sorted([mode for mode in mode_repeats if mode.startswith("W")], key=mode_sort_key)
    xs = [float(mode[1:].replace("_", ".")) for mode in modes]
    ys = []
    q1s = []
    q3s = []
    for mode in modes:
        vals = [per_repeat_overall(items)[metric] for _, items in sorted(mode_repeats[mode].items())]
        q1, q3 = iqr_bounds(vals)
        ys.append(safe_median(vals) or 0.0)
        q1s.append(q1 if q1 is not None else ys[-1])
        q3s.append(q3 if q3 is not None else ys[-1])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, ys, marker="o", linewidth=2, color="#1f77b4")
    ax.fill_between(xs, q1s, q3s, color="#1f77b4", alpha=0.16)
    ax.set_xlabel("W")
    if metric == "latency_median_ms":
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{experiment} Median Latency vs W")
        filename = f"{experiment}_latency_vs_w.png"
    else:
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title(f"{experiment} Median Throughput vs W")
        filename = f"{experiment}_throughput_vs_w.png"
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_step_throughput_selected_modes(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> Tuple[str, str]:
    modes_all = sorted(mode_repeats.keys(), key=mode_sort_key)
    selected = [mode for mode in ("STOCK", "W2_0", "W4_0") if mode in mode_repeats]
    if len(selected) < 2:
        selected = modes_all[: min(3, len(modes_all))]

    def _make_plot(modes: List[str], filename: str, title: str) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.26
        palette = make_mode_palette(modes)
        base_steps = None
        for idx, mode in enumerate(modes):
            repeat_series = [per_repeat_step_series(items) for _, items in sorted(mode_repeats[mode].items())]
            steps = sorted({step for series in repeat_series for step in series.keys()})
            if base_steps is None:
                base_steps = steps
            vals = []
            for step_idx in steps:
                per_repeat_vals = [series[step_idx]["throughput_gbps"] for series in repeat_series if step_idx in series]
                vals.append(safe_median(per_repeat_vals) or 0.0)
            xs = [step + (idx - (len(modes) - 1) / 2.0) * width for step in steps]
            ax.bar(xs, vals, width=width, label=mode, color=palette[mode], alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Throughput (GB/s)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        return filename

    all_modes_plot = _make_plot(
        [mode for mode in modes_all if mode.startswith("W")] or modes_all,
        f"{experiment}_step_throughput_by_w_bar.png",
        f"{experiment} Step Throughput by W",
    )
    selected_plot = _make_plot(
        selected,
        f"{experiment}_step_throughput_by_w_bar_stock_w2_w4.png",
        f"{experiment} Step Throughput Comparison (STOCK, W2, W4)",
    )
    return all_modes_plot, selected_plot


def plot_cts_post_volume_by_worker_vs_w(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> str:
    modes = sorted([mode for mode in mode_repeats if mode.startswith("W")], key=mode_sort_key)
    worker_names = sorted({item.worker for repeats in mode_repeats.values() for items in repeats.values() for item in items})
    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.8 / max(1, len(worker_names))
    palette = plt.cm.tab20.colors

    xs = [parse_mode_w(mode) or 0.0 for mode in modes]
    index_positions = list(range(len(xs)))
    for worker_idx, worker_name in enumerate(worker_names):
        vals = []
        for mode in modes:
            per_repeat_counts = []
            for repeat_name, items in sorted(mode_repeats[mode].items()):
                sample = items[0]
                metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
                per_repeat_counts.append(metrics.posts_by_worker.get(worker_name, 0))
            vals.append(safe_median(per_repeat_counts) or 0.0)
        offsets = [x + (worker_idx - (len(worker_names) - 1) / 2.0) * width for x in index_positions]
        ax.bar(offsets, vals, width=width, label=worker_name, color=palette[worker_idx % len(palette)], alpha=0.85)
    ax.set_xticks(index_positions)
    ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45, ha="right")
    ax.set_xlabel("Configured W")
    ax.set_ylabel("Median CTS/POST Count")
    ax.set_title(f"{experiment} CTS/POST Volume by Worker vs W")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    filename = f"{experiment}_cts_post_volume_by_worker_vs_w.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_post_receive_gate_effect_vs_w(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> str:
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    palette = make_mode_palette(modes)
    xs = list(range(len(modes)))
    posts = []
    stalls = []
    stall_ratios = []
    occ_post = []
    occ_wstall = []
    for mode in modes:
        per_repeat_posts = []
        per_repeat_stalls = []
        per_repeat_occ_post = []
        per_repeat_occ_wstall = []
        for repeat_name, items in sorted(mode_repeats[mode].items()):
            sample = items[0]
            metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
            per_repeat_posts.append(metrics.total_posts)
            per_repeat_stalls.append(metrics.total_wstalls)
            if metrics.mean_occ_pr_at_post is not None:
                per_repeat_occ_post.append(metrics.mean_occ_pr_at_post)
            if metrics.mean_occ_pr_at_wstall is not None:
                per_repeat_occ_wstall.append(metrics.mean_occ_pr_at_wstall)
        post_med = safe_median(per_repeat_posts) or 0.0
        stall_med = safe_median(per_repeat_stalls) or 0.0
        posts.append(post_med)
        stalls.append(stall_med)
        stall_ratios.append(stall_med / max(1.0, post_med + stall_med))
        occ_post.append(safe_median(per_repeat_occ_post) or 0.0)
        occ_wstall.append(safe_median(per_repeat_occ_wstall) or 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(xs, posts, color="#1f77b4", alpha=0.75, label="POST")
    axes[0].bar(xs, stalls, bottom=posts, color="#d62728", alpha=0.75, label="WSTALL")
    axes[0].set_ylabel("Median Event Count")
    axes[0].set_title(f"{experiment} Post-Receive Gate Effect vs W")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(xs, stall_ratios, marker="o", color="#d62728", label="WSTALL ratio")
    axes[1].plot(xs, occ_post, marker="o", color="#1f77b4", label="occPr at POST")
    axes[1].plot(xs, occ_wstall, marker="o", color="#9467bd", label="occPr at WSTALL")
    axes[1].set_ylabel("Ratio / Occupancy")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(modes, rotation=45, ha="right")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    filename = f"{experiment}_post_receive_gate_effect_vs_w.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_cts_progress_by_w(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> str:
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    palette = make_mode_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = list(range(PROGRESS_BINS))
    for mode in modes:
        per_repeat_bins: List[List[int]] = []
        for repeat_name, items in sorted(mode_repeats[mode].items()):
            sample = items[0]
            metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
            per_repeat_bins.append(metrics.progress_post_counts)
        medians = []
        lowers = []
        uppers = []
        for bin_idx in xs:
            vals = [float(bins[bin_idx]) for bins in per_repeat_bins]
            q1, q3 = iqr_bounds(vals)
            medians.append(safe_median(vals) or 0.0)
            lowers.append(q1 if q1 is not None else medians[-1])
            uppers.append(q3 if q3 is not None else medians[-1])
        ax.plot(xs, medians, linewidth=2, color=palette[mode], label=mode)
        ax.fill_between(xs, lowers, uppers, color=palette[mode], alpha=0.16)
    ax.set_title(f"{experiment} CTS/POST Progress-Bin Trace by W")
    ax.set_xlabel("Normalized Progress Bin (50 bins across steady steps)")
    ax.set_ylabel("Median POST Count")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    filename = f"{experiment}_cts_progress_by_w.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_exact_step_cts_vs_w(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> List[str]:
    filenames: List[str] = []
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    numeric_modes = [mode for mode in modes if mode == "STOCK" or mode.startswith("W")]
    palette = make_mode_palette(numeric_modes)
    for step_idx in SAMPLED_PROGRESS_BINS:
        labels = []
        posts = []
        stalls = []
        latencies = []
        throughputs = []
        exact_available = False
        for mode in numeric_modes:
            per_repeat_posts = []
            per_repeat_stalls = []
            per_repeat_lat = []
            per_repeat_thr = []
            for repeat_name, items in sorted(mode_repeats[mode].items()):
                sample = items[0]
                metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
                exact_available = exact_available or metrics.exact_ts_available
                per_repeat_posts.append(float(metrics.exact_step_post_counts.get(step_idx, 0)))
                per_repeat_stalls.append(float(metrics.exact_step_wstall_counts.get(step_idx, 0)))
                if step_idx in metrics.exact_step_latency_ms and metrics.exact_step_latency_ms[step_idx] is not None:
                    per_repeat_lat.append(float(metrics.exact_step_latency_ms[step_idx]))
                if step_idx in metrics.exact_step_throughput_gbps and metrics.exact_step_throughput_gbps[step_idx] is not None:
                    per_repeat_thr.append(float(metrics.exact_step_throughput_gbps[step_idx]))
            labels.append(mode)
            posts.append(safe_median(per_repeat_posts) or 0.0)
            stalls.append(safe_median(per_repeat_stalls) or 0.0)
            latencies.append(safe_median(per_repeat_lat) or 0.0)
            throughputs.append(safe_median(per_repeat_thr) or 0.0)
        if not exact_available:
            continue
        xs = list(range(len(labels)))
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].bar(xs, posts, color=[palette[label] for label in labels], alpha=0.8, label="CTS/POST")
        axes[0].bar(xs, stalls, bottom=posts, color="#d62728", alpha=0.7, label="WSTALL")
        axes[0].set_ylabel("Event Count")
        axes[0].set_title(f"{experiment} Exact Step {step_idx}: CTS/WSTALL by W")
        axes[0].legend()
        axes[0].grid(True, axis="y", alpha=0.25)

        axes[1].plot(xs, latencies, marker="o", color="#1f77b4")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(xs, throughputs, marker="o", color="#2ca02c")
        axes[2].set_ylabel("Throughput (GB/s)")
        axes[2].set_xticks(xs)
        axes[2].set_xticklabels(labels, rotation=45, ha="right")
        axes[2].grid(True, alpha=0.25)

        fig.tight_layout()
        filename = f"{experiment}_exact_step_{step_idx}_cts_vs_w.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def plot_exact_step_channel_volume(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> List[str]:
    filenames: List[str] = []
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    for step_idx in SAMPLED_PROGRESS_BINS:
        channels = sorted(
            {
                channel
                for mode in modes
                for repeat_name, items in sorted(mode_repeats[mode].items())
                for channel in load_repeat_cts_metrics(str(items[0].repeat_dir), experiment, mode, repeat_name).exact_step_channel_counts.get(step_idx, {})
            }
        )
        if not channels:
            continue
        palette = make_mode_palette(modes)
        fig, ax = plt.subplots(figsize=(13, 6))
        width = 0.8 / max(1, len(modes))
        exact_available = False
        for mode_idx, mode in enumerate(modes):
            vals = []
            for channel in channels:
                per_repeat_vals = []
                for repeat_name, items in sorted(mode_repeats[mode].items()):
                    sample = items[0]
                    metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
                    exact_available = exact_available or metrics.exact_ts_available
                    per_repeat_vals.append(float(metrics.exact_step_channel_counts.get(step_idx, {}).get(channel, 0)))
                vals.append(safe_median(per_repeat_vals) or 0.0)
            xs = [idx + (mode_idx - (len(modes) - 1) / 2.0) * width for idx in range(len(channels))]
            ax.bar(xs, vals, width=width, color=palette[mode], label=mode, alpha=0.85)
        if not exact_available:
            plt.close(fig)
            continue
        ax.set_xticks(list(range(len(channels))))
        ax.set_xticklabels([f"ch{channel}" for channel in channels])
        ax.set_title(f"{experiment} Exact Step {step_idx}: CTS Channel Volume by W")
        ax.set_xlabel("Channel")
        ax.set_ylabel("CTS/POST Count")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        filename = f"{experiment}_exact_step_{step_idx}_cts_channel_volume.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def plot_cts_channel_volume_for_bins(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]], output_dir: Path) -> List[str]:
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    filenames: List[str] = []
    for sampled_bin in SAMPLED_PROGRESS_BINS:
        fig, ax = plt.subplots(figsize=(13, 6))
        channels = sorted(
            {
                channel
                for mode in modes
                for repeat_name, items in sorted(mode_repeats[mode].items())
                for channel in load_repeat_cts_metrics(str(items[0].repeat_dir), experiment, mode, repeat_name).progress_channel_counts.get(sampled_bin, {})
            }
        )
        if not channels:
            plt.close(fig)
            continue
        width = 0.8 / max(1, len(modes))
        palette = make_mode_palette(modes)
        for mode_idx, mode in enumerate(modes):
            vals = []
            for channel in channels:
                per_repeat_vals = []
                for repeat_name, items in sorted(mode_repeats[mode].items()):
                    sample = items[0]
                    metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
                    per_repeat_vals.append(float(metrics.progress_channel_counts.get(sampled_bin, {}).get(channel, 0)))
                vals.append(safe_median(per_repeat_vals) or 0.0)
            xs = [idx + (mode_idx - (len(modes) - 1) / 2.0) * width for idx in range(len(channels))]
            ax.bar(xs, vals, width=width, color=palette[mode], label=mode, alpha=0.85)
        ax.set_xticks(list(range(len(channels))))
        ax.set_xticklabels([f"ch{channel}" for channel in channels])
        ax.set_title(f"{experiment} CTS Channel Volume at Normalized Progress Bin {sampled_bin}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Median POST Count")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        filename = f"{experiment}_cts_channel_volume_step_{sampled_bin}.png"
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
        filenames.append(filename)
    return filenames


def build_experiment_table(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]]) -> str:
    rows = []
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    for mode in modes:
        per_repeat = [per_repeat_overall(items) for _, items in sorted(mode_repeats[mode].items())]
        lat_medians = [x["latency_median_ms"] for x in per_repeat]
        thr_medians = [x["throughput_median_gbps"] for x in per_repeat]
        lat_p95s = [x["latency_p95_ms"] for x in per_repeat]
        sample = next(iter(next(iter(mode_repeats[mode].values()))))
        validation_status, validation_detail = summarize_validation(sample.validation_path)
        switch_vals = mode_switch_metrics_for_experiment(experiment, {mode: mode_repeats[mode]}, "total_delta").get(mode, [])
        switch_severity = mode_switch_metrics_for_experiment(experiment, {mode: mode_repeats[mode]}, "severity").get(mode, [])
        rows.append(
            "<tr>"
            f"<td>{escape(mode)}</td>"
            f"<td>{len(per_repeat)}</td>"
            f"<td>{fmt_float(safe_median(lat_medians))}</td>"
            f"<td>{fmt_float(safe_mean(lat_medians))}</td>"
            f"<td>{fmt_float(safe_median(lat_p95s))}</td>"
            f"<td>{fmt_float(safe_median(thr_medians))}</td>"
            f"<td>{fmt_float(safe_median(switch_vals), 1)}</td>"
            f"<td>{fmt_float(safe_median(switch_severity), 1)}</td>"
            f"<td>{validation_status}</td>"
            f"<td>{escape(validation_detail)}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>Mode</th><th>Repeats</th><th>Median Latency (ms)</th><th>Mean Latency (ms)</th>"
        "<th>Median p95 Latency (ms)</th><th>Median Throughput (GB/s)</th><th>Median PFC Count</th>"
        "<th>Median PFC Severity</th><th>Validation</th><th>Failed Workers</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def plot_root_stock_vs_w(discovered: Dict[str, Dict[str, Dict[str, List[RepeatData]]]], output_dir: Path, metric: str, filename: str, title: str, ylabel: str) -> str:
    experiments = sorted(discovered.keys(), key=experiment_sort_key)
    fig, axes = plt.subplots(len(experiments), 1, figsize=(12, 4 * len(experiments)), sharex=True)
    if len(experiments) == 1:
        axes = [axes]
    for ax, experiment in zip(axes, experiments):
        mode_repeats = discovered[experiment]
        stock_vals = [per_repeat_overall(items)[metric] for _, items in sorted(mode_repeats.get("STOCK", {}).items())]
        stock_baseline = safe_median(stock_vals)
        w_modes = sorted([mode for mode in mode_repeats if mode.startswith("W")], key=mode_sort_key)
        xs = [parse_mode_w(mode) or 0.0 for mode in w_modes]
        ys = []
        q1s = []
        q3s = []
        for mode in w_modes:
            vals = [per_repeat_overall(items)[metric] for _, items in sorted(mode_repeats[mode].items())]
            q1, q3 = iqr_bounds(vals)
            ys.append(safe_median(vals) or 0.0)
            q1s.append(q1 if q1 is not None else ys[-1])
            q3s.append(q3 if q3 is not None else ys[-1])
        if stock_baseline is not None:
            ax.axhline(stock_baseline, color="#222222", linestyle="--", linewidth=1.5, label="STOCK median")
        ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2, label="W sweep")
        ax.fill_between(xs, q1s, q3s, color="#1f77b4", alpha=0.16)
        ax.set_title(experiment)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("W")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_root_stock_vs_w_pfc(discovered: Dict[str, Dict[str, Dict[str, List[RepeatData]]]], output_dir: Path) -> str:
    experiments = sorted(discovered.keys(), key=experiment_sort_key)
    fig, axes = plt.subplots(len(experiments), 1, figsize=(12, 4 * len(experiments)), sharex=True)
    if len(experiments) == 1:
        axes = [axes]
    for ax, experiment in zip(axes, experiments):
        mode_repeats = discovered[experiment]
        stock_vals = mode_switch_metrics_for_experiment(experiment, {"STOCK": mode_repeats.get("STOCK", {})}, "total_delta").get("STOCK", [])
        stock_baseline = safe_median(stock_vals)
        w_modes = sorted([mode for mode in mode_repeats if mode.startswith("W")], key=mode_sort_key)
        xs = [parse_mode_w(mode) or 0.0 for mode in w_modes]
        ys = []
        q1s = []
        q3s = []
        for mode in w_modes:
            vals = mode_switch_metrics_for_experiment(experiment, {mode: mode_repeats[mode]}, "total_delta").get(mode, [])
            q1, q3 = iqr_bounds(vals)
            ys.append(safe_median(vals) or 0.0)
            q1s.append(q1 if q1 is not None else ys[-1])
            q3s.append(q3 if q3 is not None else ys[-1])
        if stock_baseline is not None:
            ax.axhline(stock_baseline, color="#222222", linestyle="--", linewidth=1.5, label="STOCK median")
        ax.plot(xs, ys, marker="o", color="#d62728", linewidth=2, label="W sweep")
        ax.fill_between(xs, q1s, q3s, color="#d62728", alpha=0.16)
        ax.set_title(experiment)
        ax.set_ylabel("PFC Count Delta")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("W")
    fig.suptitle("STOCK vs W Switch PFC Count", y=0.995)
    fig.tight_layout()
    filename = "stock_vs_w_switch_pfc_counts.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_root_stock_vs_w_window_trace(discovered: Dict[str, Dict[str, Dict[str, List[RepeatData]]]], output_dir: Path) -> str:
    experiments = sorted(discovered.keys(), key=experiment_sort_key)
    fig, axes = plt.subplots(len(experiments), 1, figsize=(12, 4 * len(experiments)), sharex=True)
    if len(experiments) == 1:
        axes = [axes]
    for ax, experiment in zip(axes, experiments):
        mode_repeats = discovered[experiment]
        modes = sorted(mode_repeats.keys(), key=mode_sort_key)
        palette = make_mode_palette(modes)
        xs = list(range(PROGRESS_BINS))
        for mode in modes:
            per_repeat_occ: List[List[Optional[float]]] = []
            for repeat_name, items in sorted(mode_repeats[mode].items()):
                sample = items[0]
                metrics = load_repeat_cts_metrics(str(sample.repeat_dir), experiment, mode, repeat_name)
                per_repeat_occ.append(metrics.progress_occ_pr)
            medians = []
            for idx in xs:
                vals = [row[idx] for row in per_repeat_occ if row[idx] is not None]
                medians.append(safe_median(vals) or 0.0)
            ax.plot(xs, medians, label=mode, color=palette[mode], linewidth=2)
        ax.set_title(experiment)
        ax.set_ylabel("Observed post-receive occupancy")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Normalized Progress Bin (50 bins across steady steps)")
    fig.suptitle("STOCK vs W Observed Window Trace", y=0.995)
    fig.tight_layout()
    filename = "stock_vs_w_step_window_trace.png"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def plot_root_matrix_heatmap(
    discovered: Dict[str, Dict[str, Dict[str, List[RepeatData]]]],
    output_dir: Path,
    metric_name: str,
    filename: str,
    title: str,
    use_switch_metric: bool = False,
) -> str:
    experiments = sorted(discovered.keys(), key=experiment_sort_key)
    w_modes = sorted({mode for modes in discovered.values() for mode in modes.keys() if mode.startswith("W")}, key=mode_sort_key)
    if not w_modes:
        w_modes = []
    matrix: List[List[float]] = []
    for experiment in experiments:
        row: List[float] = []
        mode_repeats = discovered[experiment]
        for mode in w_modes:
            if mode not in mode_repeats:
                row.append(float("nan"))
                continue
            if use_switch_metric:
                vals = mode_switch_metrics_for_experiment(experiment, {mode: mode_repeats[mode]}, metric_name).get(mode, [])
            else:
                vals = [per_repeat_overall(items)[metric_name] for _, items in sorted(mode_repeats[mode].items())]
            row.append(safe_median(vals) if vals else float("nan"))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(w_modes) * 0.7 + 3), max(4, len(experiments) * 0.7 + 2)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(list(range(len(w_modes))))
    ax.set_xticklabels([mode.replace("_", ".") for mode in w_modes], rotation=45, ha="right")
    ax.set_yticks(list(range(len(experiments))))
    ax.set_yticklabels(experiments)
    ax.set_xlabel("W")
    ax.set_title(title)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if not math.isnan(value):
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def generate_html(run_root: Path, output_dir: Path, root_plots: List[Tuple[str, str]], experiment_sections: List[str], root_meta_html: str) -> None:
    root_plot_html = "".join(
        f'<div><h3>{escape(title)}</h3><img src="{escape(filename)}" alt="{escape(title)}"></div>'
        for title, filename in root_plots
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PR Phase1 Report - {escape(run_root.name)}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --line: #d6dbe7;
      --text: #162033;
      --muted: #62708a;
      --accent: #1248c4;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Pretendard", "Noto Sans KR", sans-serif;
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 24px 60px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      margin-top: 20px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p, li {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      font-size: 14px;
    }}
    th {{ background: #edf2fb; }}
    code {{
      background: #eef3ff;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .note {{
      background: #f7faff;
      border-left: 4px solid #3d6ae6;
      padding: 12px 14px;
      border-radius: 10px;
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>PR Phase1 Report</h1>
      <p>Run root: <code>{escape(str(run_root))}</code></p>
      <p>Primary aggregate: median. Mean is kept as a secondary reference because step outliers can distort network experiments.</p>
      {root_meta_html}
      <div class="note">
        <strong>CTS diagnostics note.</strong>
        Exact step CTS plots use <code>PHASE1 ts_unix_ns</code> from NCCL proxy logs and
        map <code>PROXY_RECV_POST</code>/<code>PROXY_RECV_WSTALL</code> events into each DDP
        step window using the step trace <code>ts_start_unix_ns</code> and
        <code>ts_end_unix_ns</code>. If a run root does not contain the new timestamp fields,
        the reporter falls back to normalized progress-bin plots only.
      </div>
    </div>
    <section class="card">
      <h2>Root Summary</h2>
      <div class="grid">
        {root_plot_html}
      </div>
    </section>
    {''.join(experiment_sections)}
  </div>
</body>
</html>
"""
    (output_dir / "phase1_report.html").write_text(html, encoding="utf-8")


def build_root_meta_html(run_root: Path) -> str:
    meta = parse_env_file(run_root / "switch_logger_meta.env")
    if not meta:
        return ""
    items = []
    for key in ("RUN_ID", "SWITCH_RUN_ID", "SWITCH_LOG_DIR", "PID_FILE", "MARKERS_JSONL"):
        value = meta.get(key)
        if value:
            items.append(f"<li><strong>{escape(key)}</strong>: <code>{escape(value)}</code></li>")
    if not items:
        return ""
    return "<ul>" + "".join(items) + "</ul>"


def build_report(run_root: Path, output_dir: Path) -> None:
    discovered = discover_runs(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_sections: List[str] = []

    reporter_log(f"load run root={run_root}")
    reporter_log(f"discovered experiments={len(discovered)} output_dir={output_dir}")

    reporter_log("build root summary plots")
    root_plots = [
        (
            "STOCK vs W Latency",
            plot_root_stock_vs_w(discovered, output_dir, "latency_median_ms", "stock_vs_w_latency.png", "STOCK vs W Latency", "Latency (ms)"),
        ),
        (
            "STOCK vs W Throughput",
            plot_root_stock_vs_w(discovered, output_dir, "throughput_median_gbps", "stock_vs_w_throughput.png", "STOCK vs W Throughput", "Throughput (GB/s)"),
        ),
        ("STOCK vs W Switch PFC Count", plot_root_stock_vs_w_pfc(discovered, output_dir)),
        ("STOCK vs W Observed Window Trace", plot_root_stock_vs_w_window_trace(discovered, output_dir)),
        ("W Matrix Latency", plot_root_matrix_heatmap(discovered, output_dir, "latency_median_ms", "w_matrix_latency.png", "W Matrix Latency")),
        ("W Matrix Throughput", plot_root_matrix_heatmap(discovered, output_dir, "throughput_median_gbps", "w_matrix_throughput.png", "W Matrix Throughput")),
        ("W Matrix PFC Count", plot_root_matrix_heatmap(discovered, output_dir, "total_delta", "w_matrix_pfc_count.png", "W Matrix PFC Count", use_switch_metric=True)),
        ("W Matrix PFC Severity", plot_root_matrix_heatmap(discovered, output_dir, "severity", "w_matrix_pfc_severity.png", "W Matrix PFC Severity", use_switch_metric=True)),
    ]

    for experiment in sorted(discovered.keys(), key=experiment_sort_key):
        mode_repeats = discovered[experiment]
        reporter_log(f"build experiment start experiment={experiment} modes={len(mode_repeats)}")
        latency_box = plot_summary_box(experiment, mode_repeats, output_dir, "latency_median_ms")
        reporter_log(f"plot done experiment={experiment} file={latency_box}")
        throughput_box = plot_summary_box(experiment, mode_repeats, output_dir, "throughput_median_gbps")
        reporter_log(f"plot done experiment={experiment} file={throughput_box}")
        step_latency = plot_step_overlay(experiment, mode_repeats, output_dir, "latency_ms")
        reporter_log(f"plot done experiment={experiment} file={step_latency}")
        step_throughput = plot_step_overlay(experiment, mode_repeats, output_dir, "throughput_gbps")
        reporter_log(f"plot done experiment={experiment} file={step_throughput}")
        delta_vs_stock = plot_delta_vs_stock(experiment, mode_repeats, output_dir)
        reporter_log(f"plot done experiment={experiment} file={delta_vs_stock}")
        latency_vs_w = plot_vs_w_lines(experiment, mode_repeats, output_dir, "latency_median_ms")
        reporter_log(f"plot done experiment={experiment} file={latency_vs_w}")
        throughput_vs_w = plot_vs_w_lines(experiment, mode_repeats, output_dir, "throughput_median_gbps")
        reporter_log(f"plot done experiment={experiment} file={throughput_vs_w}")
        cts_worker_volume = plot_cts_post_volume_by_worker_vs_w(experiment, mode_repeats, output_dir)
        reporter_log(f"plot done experiment={experiment} file={cts_worker_volume}")
        gate_effect = plot_post_receive_gate_effect_vs_w(experiment, mode_repeats, output_dir)
        reporter_log(f"plot done experiment={experiment} file={gate_effect}")
        cts_progress = plot_cts_progress_by_w(experiment, mode_repeats, output_dir)
        reporter_log(f"plot done experiment={experiment} file={cts_progress}")
        exact_step_cts = plot_exact_step_cts_vs_w(experiment, mode_repeats, output_dir)
        for filename in exact_step_cts:
            reporter_log(f"plot done experiment={experiment} file={filename}")
        exact_step_channels = plot_exact_step_channel_volume(experiment, mode_repeats, output_dir)
        for filename in exact_step_channels:
            reporter_log(f"plot done experiment={experiment} file={filename}")
        throughput_bar_all, throughput_bar_selected = plot_step_throughput_selected_modes(experiment, mode_repeats, output_dir)
        reporter_log(f"plot done experiment={experiment} file={throughput_bar_all}")
        reporter_log(f"plot done experiment={experiment} file={throughput_bar_selected}")
        channel_files = plot_cts_channel_volume_for_bins(experiment, mode_repeats, output_dir)
        for filename in channel_files:
            reporter_log(f"plot done experiment={experiment} file={filename}")
        summary_table = build_experiment_table(experiment, mode_repeats)

        exact_step_html = "".join(
            f'<div><h3>{escape(filename.replace(".png", "").replace("_", " "))}</h3><img src="{escape(filename)}" alt="{escape(experiment)} {escape(filename)}"></div>'
            for filename in exact_step_cts
        )
        exact_channel_html = "".join(
            f'<div><h3>{escape(filename.replace(".png", "").replace("_", " "))}</h3><img src="{escape(filename)}" alt="{escape(experiment)} {escape(filename)}"></div>'
            for filename in exact_step_channels
        )
        channel_html = "".join(
            f'<div><h3>CTS Channel Volume Bin {escape(filename.rsplit("_", 1)[-1].replace(".png", ""))}</h3><img src="{escape(filename)}" alt="{escape(experiment)} {escape(filename)}"></div>'
            for filename in channel_files
        )

        experiment_sections.append(
            f"""
            <section class="card">
              <h2>{escape(experiment)}</h2>
              <p>Mode comparison uses repeat-level medians. CTS diagnostics are derived from steady-phase <code>PROXY_RECV_POST</code> and <code>PROXY_RECV_WSTALL</code> events parsed from <code>worker_launcher.log</code>.</p>
              {summary_table}
              <div class="grid" style="margin-top:18px;">
                <div><h3>Latency by Mode</h3><img src="{escape(latency_box)}" alt="{escape(experiment)} latency boxplot"></div>
                <div><h3>Throughput by Mode</h3><img src="{escape(throughput_box)}" alt="{escape(experiment)} throughput boxplot"></div>
                <div><h3>Step Latency Overlay</h3><img src="{escape(step_latency)}" alt="{escape(experiment)} step latency overlay"></div>
                <div><h3>Step Throughput Overlay</h3><img src="{escape(step_throughput)}" alt="{escape(experiment)} step throughput overlay"></div>
                <div><h3>Latency vs W</h3><img src="{escape(latency_vs_w)}" alt="{escape(experiment)} latency vs W"></div>
                <div><h3>Throughput vs W</h3><img src="{escape(throughput_vs_w)}" alt="{escape(experiment)} throughput vs W"></div>
                <div><h3>CTS/POST Volume by Worker vs W</h3><img src="{escape(cts_worker_volume)}" alt="{escape(experiment)} CTS volume by worker"></div>
                <div><h3>Post-Receive Gate Effect vs W</h3><img src="{escape(gate_effect)}" alt="{escape(experiment)} gate effect"></div>
                <div><h3>CTS Progress by W</h3><img src="{escape(cts_progress)}" alt="{escape(experiment)} CTS progress by W"></div>
                {exact_step_html}
                {exact_channel_html}
                <div><h3>Step Throughput by W</h3><img src="{escape(throughput_bar_all)}" alt="{escape(experiment)} step throughput by W"></div>
                <div><h3>Step Throughput: STOCK vs W2 vs W4</h3><img src="{escape(throughput_bar_selected)}" alt="{escape(experiment)} throughput stock w2 w4"></div>
                {channel_html}
              </div>
              <div style="margin-top:18px;">
                <h3>Delta vs STOCK</h3>
                <img src="{escape(delta_vs_stock)}" alt="{escape(experiment)} delta vs STOCK">
              </div>
            </section>
            """
        )
        reporter_log(f"build experiment end experiment={experiment}")

    reporter_log("write html report")
    generate_html(run_root, output_dir, root_plots, experiment_sections, build_root_meta_html(run_root))
    reporter_log(f"done html={output_dir / 'phase1_report.html'}")


def main() -> None:
    args = parse_args()
    build_report(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
