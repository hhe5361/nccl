#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
SWITCH_SHARED_ROOT_DEFAULT = Path("/mnt/nfs/cts_experiments/switch_log")
SWITCH_SHARED_ROOT_FALLBACK = Path("/mnt/nfs_share/cts_experiments/switch_log")
SWITCH_LABELS = ("rackA", "rackB", "spine")
SWITCH_BUCKET_NS = 1_000_000_000


@dataclass
class StepRecord:
    repeat: str
    mode: str
    step: int
    warmup: bool
    ts_start_unix_ns: int
    ts_end_unix_ns: int
    ts_mid_unix_ns: int
    step_ms_max: float
    steps_per_sec: float
    samples_per_sec: float


@dataclass
class Phase0Event:
    event: str
    t_ns: int
    t_ms: float
    size: int
    posted: int
    received: int
    transmitted: int
    done: int
    channel: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase4 reporter")
    parser.add_argument("--input", required=True, help="phase4 run root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outlier-method", choices=("none", "mad", "iqr"), default="mad")
    parser.add_argument("--outlier-z", type=float, default=3.5)
    parser.add_argument("--outlier-iqr-k", type=float, default=1.5)
    return parser.parse_args()


def parse_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV_RE.finditer(line)}


def discover_repeat_dirs(run_root: Path) -> list[Path]:
    repeat_dirs = sorted(p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    if repeat_dirs:
        return repeat_dirs
    return [run_root]


def iter_mode_dirs(repeat_dir: Path) -> Iterable[Path]:
    for candidate in sorted(p for p in repeat_dir.iterdir() if p.is_dir()):
        if candidate.name.startswith("."):
            continue
        yield candidate


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def median(values: Sequence[float]) -> float:
    return percentile(values, 0.50)


def centered_limits(values: Sequence[float], *, min_half_span: float, frac: float) -> tuple[float, float] | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    med = median(vals)
    half_span = max(min_half_span, abs(med) * frac)
    return max(0.0, med - half_span), med + half_span


def build_palette(keys: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {key: cmap(i % 10) for i, key in enumerate(keys)}


def elapsed_seconds(base_ns: int, ts_ns: int) -> float:
    return (ts_ns - base_ns) / 1e9


def load_step_metrics(repeat_name: str, mode_dir: Path) -> list[StepRecord]:
    metrics_files = sorted(mode_dir.glob("*_step_metrics.jsonl"))
    if not metrics_files:
        return []
    rows: list[StepRecord] = []
    with metrics_files[0].open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                StepRecord(
                    repeat=repeat_name,
                    mode=mode_dir.name,
                    step=int(row["step"]),
                    warmup=bool(row["warmup"]),
                    ts_start_unix_ns=int(row["ts_start_unix_ns"]),
                    ts_end_unix_ns=int(row["ts_end_unix_ns"]),
                    ts_mid_unix_ns=int(row["ts_mid_unix_ns"]),
                    step_ms_max=float(row["step_ms_max"]),
                    steps_per_sec=float(row["steps_per_sec"]),
                    samples_per_sec=float(row["samples_per_sec"]),
                )
            )
    return rows


def _discover_nccl_logs(mode_dir: Path) -> list[Path]:
    worker_log_dir = mode_dir / "worker01"
    nccl_logs = sorted(worker_log_dir.glob("nccl.*.log"))
    if nccl_logs:
        return nccl_logs
    worker_dirs = sorted(p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("worker"))
    if worker_dirs:
        return sorted(worker_dirs[0].glob("nccl.*.log"))
    return []


def load_phase0_events(mode_dir: Path) -> list[Phase0Event]:
    raw_rows: list[dict[str, str]] = []
    for log_path in _discover_nccl_logs(mode_dir):
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PHASE0 event=PROXY_RECV_" not in line:
                    continue
                row = parse_kv(line)
                event = row.get("event", "")
                if row.get("tNs") is None or not event.startswith("PROXY_RECV_"):
                    continue
                raw_rows.append(row)
    if not raw_rows:
        return []
    base_ns = min(int(row["tNs"]) for row in raw_rows)
    events: list[Phase0Event] = []
    for row in sorted(raw_rows, key=lambda item: int(item["tNs"])):
        t_ns = int(row["tNs"])
        events.append(
            Phase0Event(
                event=row["event"],
                t_ns=t_ns,
                t_ms=(t_ns - base_ns) / 1e6,
                size=int(row.get("size", "0")),
                posted=int(row.get("posted", "0")),
                received=int(row.get("received", "0")),
                transmitted=int(row.get("transmitted", "0")),
                done=int(row.get("done", "0")),
                channel=int(row.get("channel", "0")),
            )
        )
    return events


def measured_records(records: Sequence[StepRecord]) -> list[StepRecord]:
    return [record for record in records if not record.warmup]


def compute_outlier_indices(
    records: Sequence[StepRecord],
    *,
    method: str,
    mad_z: float,
    iqr_k: float,
) -> set[int]:
    measured = [(idx, record.step_ms_max) for idx, record in enumerate(records) if not record.warmup]
    if method == "none" or len(measured) < 4:
        return set()
    values = [value for _, value in measured]
    flagged: set[int] = set()
    if method == "mad":
        med = median(values)
        abs_dev = [abs(value - med) for value in values]
        mad = median(abs_dev)
        if mad > 0:
            scale = 1.4826 * mad
            threshold = med + mad_z * scale
            for idx, value in measured:
                if value > threshold:
                    flagged.add(idx)
    if method == "iqr" or (method == "mad" and not flagged):
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        iqr = q3 - q1
        if iqr > 0:
            threshold = q3 + iqr_k * iqr
            for idx, value in measured:
                if value > threshold:
                    flagged.add(idx)
    return flagged


def filtered_measured_records(
    records: Sequence[StepRecord],
    *,
    method: str,
    mad_z: float,
    iqr_k: float,
) -> tuple[list[StepRecord], set[int]]:
    flagged = compute_outlier_indices(records, method=method, mad_z=mad_z, iqr_k=iqr_k)
    filtered = [record for idx, record in enumerate(records) if not record.warmup and idx not in flagged]
    return filtered, flagged


def mode_start_ns(records: Sequence[StepRecord]) -> int:
    return min(record.ts_start_unix_ns for record in records)


def measured_window_bounds(records: Sequence[StepRecord]) -> Optional[tuple[int, int]]:
    measured = measured_records(records)
    if not measured:
        return None
    return (
        min(record.ts_start_unix_ns for record in measured),
        max(record.ts_end_unix_ns for record in measured),
    )


def event_in_measured_window(record: StepRecord, event_ts_ns: int) -> bool:
    return record.ts_start_unix_ns <= event_ts_ns <= record.ts_end_unix_ns


def filter_events_to_measured_steps(events: Sequence[Phase0Event], records: Sequence[StepRecord]) -> list[Phase0Event]:
    measured = measured_records(records)
    if not measured:
        return []
    kept: list[Phase0Event] = []
    step_index = 0
    for event in events:
        while step_index < len(measured) and measured[step_index].ts_end_unix_ns < event.t_ns:
            step_index += 1
        if step_index >= len(measured):
            break
        if event_in_measured_window(measured[step_index], event.t_ns):
            kept.append(event)
    return kept


def plot_metric_overlay(
    records_by_mode: dict[str, list[StepRecord]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    metric_key: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        records = records_by_mode[mode]
        if not records:
            continue
        base_ns = mode_start_ns(records)
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            end_s = elapsed_seconds(base_ns, record.ts_end_unix_ns)
            mid_s = elapsed_seconds(base_ns, record.ts_mid_unix_ns)
            metric_value = getattr(record, metric_key)
            ax.hlines(metric_value, start_s, end_s, color=palette[mode], linewidth=2.0, alpha=0.9)
            ax.scatter([start_s], [metric_value], color=palette[mode], s=14, marker="o", alpha=0.9)
            xs.append(mid_s)
            ys.append(metric_value)
        ax.plot(xs, ys, color=palette[mode], linewidth=1.6, alpha=0.9, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since Mode Start (s)")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_step_timeline(records_by_mode: dict[str, list[StepRecord]], output_path: Path, *, title: str) -> None:
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, max(3.5, 1.1 * len(modes) + 1.5)))
    yticks: list[float] = []
    ylabels: list[str] = []
    for y, mode in enumerate(modes):
        records = records_by_mode[mode]
        if not records:
            continue
        base_ns = mode_start_ns(records)
        for record in records:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            end_s = elapsed_seconds(base_ns, record.ts_end_unix_ns)
            width = max(end_s - start_s, 1e-9)
            ax.barh(y, width, left=start_s, height=0.6, color=palette[mode], alpha=0.85)
        yticks.append(y)
        ylabels.append(mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since Mode Start (s)")
    ax.set_ylabel("Mode")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_repeat_metric_overlay(
    records_by_repeat: dict[str, list[StepRecord]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    metric_key: str,
) -> None:
    repeats = list(records_by_repeat.keys())
    palette = build_palette(repeats)
    fig, ax = plt.subplots(figsize=(12, 5))
    for repeat in repeats:
        records = records_by_repeat[repeat]
        if not records:
            continue
        base_ns = mode_start_ns(records)
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            mid_s = elapsed_seconds(base_ns, record.ts_mid_unix_ns)
            metric_value = getattr(record, metric_key)
            ax.scatter([start_s], [metric_value], color=palette[repeat], s=14, marker="o", alpha=0.9)
            xs.append(mid_s)
            ys.append(metric_value)
        ax.plot(xs, ys, color=palette[repeat], linewidth=1.8, alpha=0.9, label=repeat)
    ax.set_title(title)
    ax.set_xlabel("Time Since Mode Start (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_boxplot_by_mode(
    values_by_mode: dict[str, list[float]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    modes = [mode for mode, values in values_by_mode.items() if values]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot([values_by_mode[mode] for mode in modes], tick_labels=modes, patch_artist=True, showfliers=True)
    for patch, mode in zip(bp["boxes"], modes):
        patch.set_facecolor(palette[mode])
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_post_timestamps_ms(mode_dir: Path) -> list[float]:
    return [event.t_ms for event in load_phase0_events(mode_dir) if event.event == "PROXY_RECV_POST"]


def load_netdone_events(mode_dir: Path) -> list[Phase0Event]:
    return [event for event in load_phase0_events(mode_dir) if event.event == "PROXY_RECV_NET_DONE" and event.size > 0]


def plot_post_cumulative_overlay(post_ms_by_mode: dict[str, list[float]], output_path: Path, *, title: str) -> None:
    modes = [mode for mode, values in post_ms_by_mode.items() if values]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        xs = post_ms_by_mode[mode]
        ys = list(range(1, len(xs) + 1))
        ax.plot(xs, ys, color=palette[mode], linewidth=1.8, alpha=0.95, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since First POST in Mode (ms)")
    ax.set_ylabel("Cumulative PROXY_RECV_POST Count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_post_rate_overlay(
    post_ms_by_mode: dict[str, list[float]],
    output_path: Path,
    *,
    title: str,
    bin_width_ms: float = 0.1,
) -> None:
    modes = [mode for mode, values in post_ms_by_mode.items() if values]
    if not modes:
        return
    palette = build_palette(modes)
    max_t = max(max(values) for values in post_ms_by_mode.values() if values)
    nbins = max(1, int(max_t / bin_width_ms) + 1)
    xs = [(i + 0.5) * bin_width_ms for i in range(nbins)]
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        bins = [0] * nbins
        for t_ms in post_ms_by_mode[mode]:
            idx = min(int(t_ms / bin_width_ms), nbins - 1)
            bins[idx] += 1
        rates = [count / bin_width_ms for count in bins]
        ax.plot(xs, rates, color=palette[mode], linewidth=1.6, alpha=0.95, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since First POST in Mode (ms)")
    ax.set_ylabel("POST Rate (events / ms)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_outstanding_overlay(
    events_by_mode: dict[str, list[Phase0Event]],
    output_path: Path,
    *,
    title: str,
) -> None:
    modes = [mode for mode, events in events_by_mode.items() if events]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        post_events = [event for event in events_by_mode[mode] if event.event == "PROXY_RECV_POST"]
        if not post_events:
            continue
        xs = [event.t_ms for event in post_events]
        ys = [event.posted - event.received for event in post_events]
        ax.plot(xs, ys, color=palette[mode], linewidth=1.8, alpha=0.95, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since First PHASE0 Event in Mode (ms)")
    ax.set_ylabel("Outstanding (posted - received)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_netdone_cumulative_overlay(
    events_by_mode: dict[str, list[Phase0Event]],
    output_path: Path,
    *,
    title: str,
) -> None:
    modes = [mode for mode, events in events_by_mode.items() if events]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        cumulative = 0
        xs: list[float] = []
        ys_gib: list[float] = []
        base_ns = min(event.t_ns for event in events_by_mode[mode])
        for event in events_by_mode[mode]:
            cumulative += event.size
            xs.append((event.t_ns - base_ns) / 1e6)
            ys_gib.append(cumulative / float(1024 ** 3))
        ax.plot(xs, ys_gib, color=palette[mode], linewidth=1.8, alpha=0.95, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since First NET_DONE in Mode (ms)")
    ax.set_ylabel("Cumulative NET_DONE Bytes (GiB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_network_throughput_overlay(
    events_by_mode: dict[str, list[Phase0Event]],
    output_path: Path,
    *,
    title: str,
    bin_width_ms: float = 1.0,
) -> None:
    modes = [mode for mode, events in events_by_mode.items() if events]
    if not modes:
        return
    palette = build_palette(modes)
    max_t = 0.0
    aligned_by_mode: dict[str, list[tuple[float, int]]] = {}
    for mode in modes:
        base_ns = min(event.t_ns for event in events_by_mode[mode])
        aligned = [((event.t_ns - base_ns) / 1e6, event.size) for event in events_by_mode[mode]]
        aligned_by_mode[mode] = aligned
        max_t = max(max_t, max(t_ms for t_ms, _ in aligned))
    nbins = max(1, int(max_t / bin_width_ms) + 1)
    xs = [(i + 0.5) * bin_width_ms for i in range(nbins)]
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        bins = [0] * nbins
        for t_ms, size_bytes in aligned_by_mode[mode]:
            idx = min(int(t_ms / bin_width_ms), nbins - 1)
            bins[idx] += size_bytes
        gbps = [(total_bytes * 8.0) / (bin_width_ms * 1_000_000.0) for total_bytes in bins]
        ax.plot(xs, gbps, color=palette[mode], linewidth=1.6, alpha=0.95, label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since First NET_DONE in Mode (ms)")
    ax.set_ylabel("Network Throughput (Gbps)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    rows: list[dict] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        row, next_index = decoder.raw_decode(text, index)
        if isinstance(row, dict):
            rows.append(row)
        index = next_index
    return rows


def extract_record_ts_ns(row: dict) -> Optional[int]:
    for key in ("ts_unix_ns", "timestamp_ns", "time_ns", "ts_ns", "tNs"):
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def resolve_switch_log_dir(run_root: Path) -> Optional[Path]:
    env_path = run_root / "switch_logger.env"
    if not env_path.exists():
        return None
    candidates: list[Path] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SWITCH_LOG_LOCAL_DIR=") or line.startswith("SWITCH_LOG_DIR="):
            candidates.append(Path(line.split("=", 1)[1].strip()))
        elif line.startswith("SWITCH_LOG_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            candidates.append(SWITCH_SHARED_ROOT_DEFAULT / run_id)
            candidates.append(SWITCH_SHARED_ROOT_FALLBACK / run_id)
    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        raw = candidate.as_posix()
        if raw.startswith("/mnt/nfs/"):
            expanded.append(Path("/mnt/nfs_share/" + raw[len("/mnt/nfs/"):]))
        elif raw.startswith("/mnt/nfs_share/"):
            expanded.append(Path("/mnt/nfs/" + raw[len("/mnt/nfs_share/"):]))
    seen: set[str] = set()
    for candidate in expanded:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def load_aggregate_switch_file(path: Path, switch_label: str) -> dict[str, list[tuple[int, float]]]:
    fields: dict[str, list[tuple[int, float]]] = {}
    carry_forward: dict[str, dict[str, float]] = {}
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        if ts_ns is None:
            continue
        field_names = {
            "rx": "rx_pause_packets_total",
            "tx": "tx_pause_packets_total",
        } if switch_label == "spine" else {
            "rx": "rx_pause_total",
            "tx": "tx_pause_total",
        }
        ports = row.get("ports")
        if not isinstance(ports, dict):
            continue
        for alias, total_key in field_names.items():
            port_key = total_key[:-len("_total")] if total_key.endswith("_total") else total_key
            series = carry_forward.setdefault(alias, {})
            saw_value = False
            for port_name, port_row in ports.items():
                if not isinstance(port_row, dict):
                    continue
                value = port_row.get(port_key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    series[str(port_name)] = max(0.0, float(value))
                    saw_value = True
            if saw_value or series:
                fields.setdefault(alias, []).append((ts_ns, sum(series.values())))
    for rows in fields.values():
        rows.sort(key=lambda item: item[0])
    return fields


def raw_switch_counter_value(row: dict, switch_label: str) -> Optional[float]:
    keys = ("rx_pause_packets", "tx_pause_packets") if switch_label == "spine" else ("rx_pause", "tx_pause")
    total = 0.0
    found = False
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += max(0.0, float(value))
            found = True
    return total if found else None


def load_raw_switch_file(path: Path, switch_label: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        value = raw_switch_counter_value(row, switch_label)
        if ts_ns is None or value is None:
            continue
        sample_id = row.get("sample_id")
        key = f"sample:{sample_id}" if sample_id not in (None, "") else f"bucket:{(ts_ns // SWITCH_BUCKET_NS) * SWITCH_BUCKET_NS}"
        grouped.setdefault(key, []).append((ts_ns, value))
    samples: list[tuple[int, float]] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
        ts_ns = int(sum(ts for ts, _ in rows) / len(rows))
        samples.append((ts_ns, sum(value for _, value in rows)))
    samples.sort(key=lambda item: item[0])
    return {"rx": samples} if samples else {}


def positive_increment_total(samples: list[tuple[int, float]]) -> float:
    total = 0.0
    for (_, prev_total), (_, cur_total) in zip(samples, samples[1:]):
        total += max(0.0, cur_total - prev_total)
    return total


def choose_primary_switch_series(fields: dict[str, list[tuple[int, float]]]) -> list[tuple[int, float]]:
    candidates = [(name, rows) for name, rows in fields.items() if rows]
    if not candidates:
        return []
    packet_candidates = [item for item in candidates if item[0] in {"rx", "tx"}]
    if packet_candidates:
        candidates = packet_candidates
    candidates.sort(key=lambda item: positive_increment_total(item[1]), reverse=True)
    return candidates[0][1]


def compute_switch_delta_series(samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    deltas: list[tuple[int, float]] = []
    for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
        elapsed = (cur_ts - prev_ts) / 1_000_000_000.0
        if elapsed <= 0:
            continue
        deltas.append((cur_ts, max(0.0, cur_total - prev_total) / elapsed))
    return deltas


def compute_counter_delta_series(samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    deltas: list[tuple[int, float]] = []
    for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
        deltas.append((cur_ts, max(0.0, cur_total - prev_total)))
    return deltas


def compute_counter_rate_series(samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    rates: list[tuple[int, float]] = []
    for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
        elapsed = (cur_ts - prev_ts) / 1_000_000_000.0
        if elapsed <= 0:
            continue
        rates.append((cur_ts, max(0.0, cur_total - prev_total) / elapsed))
    return rates


def interpolate_switch_snapshot_value(snapshots: list[tuple[int, float]], ts_ns: int) -> Optional[float]:
    if not snapshots:
        return None
    if ts_ns <= snapshots[0][0]:
        return float(snapshots[0][1])
    if ts_ns >= snapshots[-1][0]:
        return float(snapshots[-1][1])
    for (left_ts, left_value), (right_ts, right_value) in zip(snapshots, snapshots[1:]):
        if left_ts <= ts_ns <= right_ts:
            if right_ts == left_ts:
                return float(right_value)
            ratio = (ts_ns - left_ts) / float(right_ts - left_ts)
            return float(left_value + (right_value - left_value) * ratio)
    return None


def build_phase_aligned_cumulative_series(snapshots: list[tuple[int, float]], start_ns: int, end_ns: int) -> list[tuple[float, float]]:
    if end_ns <= start_ns:
        return []
    start_value = interpolate_switch_snapshot_value(snapshots, start_ns)
    if start_value is None:
        return []
    candidate_ts = [start_ns]
    candidate_ts.extend(ts for ts, _ in snapshots if start_ns < ts < end_ns)
    candidate_ts.append(end_ns)
    samples: list[tuple[int, float]] = []
    seen: set[int] = set()
    for ts_ns in candidate_ts:
        if ts_ns in seen:
            continue
        seen.add(ts_ns)
        value = interpolate_switch_snapshot_value(snapshots, ts_ns)
        if value is None:
            continue
        samples.append((ts_ns, value))
    if not samples:
        return []
    points: list[tuple[float, float]] = []
    cumulative = 0.0
    prev_value = samples[0][1]
    points.append(((samples[0][0] - start_ns) / 1_000_000_000.0, 0.0))
    for ts_ns, value in samples[1:]:
        cumulative += max(0.0, value - prev_value)
        points.append(((ts_ns - start_ns) / 1_000_000_000.0, cumulative))
        prev_value = value
    return points


def compute_worker_total_gbps_series(
    tx_samples: list[tuple[int, float]],
    rx_samples: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    if not tx_samples or not rx_samples:
        return []
    tx_map = {ts: value for ts, value in tx_samples}
    rx_map = {ts: value for ts, value in rx_samples}
    timestamps = sorted(set(tx_map) & set(rx_map))
    if len(timestamps) < 2:
        return []
    totals = [(ts, tx_map[ts] + rx_map[ts]) for ts in timestamps]
    rates = compute_counter_rate_series(totals)
    return [(ts, (bytes_per_sec * 8.0) / 1e9) for ts, bytes_per_sec in rates]


def load_switch_bundle(run_root: Path) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(run_root)
    if log_dir is None or not log_dir.exists():
        return None
    aggregate_paths = {
        "rackA": log_dir / "rackA_pfc_aggregate.jsonl",
        "rackB": log_dir / "rackB_pfc_aggregate.jsonl",
        "spine": log_dir / "spine_roce_aggregate.jsonl",
    }
    raw_paths = {
        "rackA": log_dir / "rackA_pfc_statistics.jsonl",
        "rackB": log_dir / "rackB_pfc_statistics.jsonl",
        "spine": log_dir / "spine_roce_counters.jsonl",
    }
    snapshots: dict[str, list[tuple[int, float]]] = {}
    for label in SWITCH_LABELS:
        if aggregate_paths[label].exists():
            fields = load_aggregate_switch_file(aggregate_paths[label], label)
        elif raw_paths[label].exists():
            fields = load_raw_switch_file(raw_paths[label], label)
        else:
            fields = {}
        primary = choose_primary_switch_series(fields)
        if primary:
            snapshots[label] = primary
    if not snapshots:
        return None
    return {"log_dir": log_dir, "snapshots": snapshots}


def load_total_series_from_jsonl(path: Path, total_field: str) -> list[tuple[int, float]]:
    if not path.exists():
        return []
    samples: list[tuple[int, float]] = []
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        value = row.get(total_field)
        if ts_ns is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        samples.append((ts_ns, float(value)))
    samples.sort(key=lambda item: item[0])
    return samples


def load_grouped_sum_series(path: Path, field_names: Sequence[str]) -> list[tuple[int, float]]:
    if not path.exists():
        return []
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        if ts_ns is None:
            continue
        total = 0.0
        found = False
        for field_name in field_names:
            value = row.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += max(0.0, float(value))
                found = True
        if not found:
            continue
        sample_id = row.get("sample_id")
        key = f"sample:{sample_id}" if sample_id not in (None, "") else f"bucket:{(ts_ns // SWITCH_BUCKET_NS) * SWITCH_BUCKET_NS}"
        grouped.setdefault(key, []).append((ts_ns, total))
    series: list[tuple[int, float]] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
        ts_ns = int(sum(ts for ts, _ in rows) / len(rows))
        series.append((ts_ns, sum(value for _, value in rows)))
    series.sort(key=lambda item: item[0])
    return series


def load_congestion_bundle(run_root: Path) -> Optional[dict]:
    switch_bundle = load_switch_bundle(run_root)
    log_dir = switch_bundle["log_dir"] if switch_bundle else resolve_switch_log_dir(run_root)
    if log_dir is None or not log_dir.exists():
        return switch_bundle

    worker_agg = log_dir / "worker_roce_counters_aggregate.jsonl"
    spine_agg = log_dir / "spine_roce_aggregate.jsonl"
    spine_raw = log_dir / "spine_roce_counters.jsonl"

    worker_bytes_total = load_total_series_from_jsonl(worker_agg, "tx_prio_bytes_total")
    worker_rx_bytes_total = load_total_series_from_jsonl(worker_agg, "rx_prio_bytes_total")
    worker_cnp_total = load_total_series_from_jsonl(worker_agg, "np_cnp_sent_total")
    worker_tx_pause_total = load_total_series_from_jsonl(worker_agg, "tx_prio_pause_total")
    worker_tx_pause_duration_total = load_total_series_from_jsonl(worker_agg, "tx_prio_pause_duration_total")

    spine_ecn_total = load_total_series_from_jsonl(spine_agg, "rx_ecn_marked_packets_total")
    if not spine_ecn_total:
        spine_ecn_total = load_grouped_sum_series(spine_raw, ("rx_ecn_marked_packets",))

    bundle = switch_bundle.copy() if switch_bundle else {"log_dir": log_dir, "snapshots": {}}
    bundle["worker_total_gbps_series"] = compute_worker_total_gbps_series(worker_bytes_total, worker_rx_bytes_total)
    bundle["worker_cnp_rate_series"] = compute_counter_rate_series(worker_cnp_total)
    bundle["worker_tx_pause_rate_series"] = compute_counter_rate_series(worker_tx_pause_total)
    bundle["worker_tx_pause_duration_rate_series"] = compute_counter_rate_series(worker_tx_pause_duration_total)
    bundle["spine_ecn_rate_series"] = compute_counter_rate_series(spine_ecn_total)
    if bundle.get("snapshots") or bundle["worker_total_gbps_series"] or bundle["worker_cnp_rate_series"] or bundle["worker_tx_pause_rate_series"] or bundle["spine_ecn_rate_series"]:
        return bundle
    return None


def compute_switch_metrics_for_modes(
    switch_bundle: Optional[dict],
    records_by_mode: dict[str, list[StepRecord]],
) -> dict[str, dict[str, float]]:
    if not switch_bundle:
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for mode, records in records_by_mode.items():
        bounds = measured_window_bounds(records)
        row: dict[str, float] = {"switch_pfc_total": 0.0, "switch_pfc_peak_rate": 0.0}
        if bounds is None:
            metrics[mode] = row
            continue
        start_ns, end_ns = bounds
        for label in SWITCH_LABELS:
            snapshots = switch_bundle["snapshots"].get(label, [])
            points = build_phase_aligned_cumulative_series(snapshots, start_ns, end_ns)
            delta = points[-1][1] if points else 0.0
            rates = [rate for ts_ns, rate in compute_switch_delta_series(snapshots) if start_ns <= ts_ns <= end_ns]
            peak_rate = max(rates) if rates else 0.0
            row[f"{label}_pfc_delta"] = delta
            row[f"{label}_pfc_peak_rate"] = peak_rate
            row["switch_pfc_total"] += delta
            row["switch_pfc_peak_rate"] = max(row["switch_pfc_peak_rate"], peak_rate)
        metrics[mode] = row
    return metrics


def plot_switch_pfc_overlay(
    switch_bundle: Optional[dict],
    records_by_mode: dict[str, list[StepRecord]],
    output_path: Path,
    *,
    title: str,
) -> None:
    if not switch_bundle:
        return
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=False)
    any_points = False
    for ax, label in zip(axes, SWITCH_LABELS):
        snapshots = switch_bundle["snapshots"].get(label, [])
        if not snapshots:
            continue
        for mode in modes:
            bounds = measured_window_bounds(records_by_mode[mode])
            if bounds is None:
                continue
            start_ns, end_ns = bounds
            points = build_phase_aligned_cumulative_series(snapshots, start_ns, end_ns)
            if not points:
                continue
            any_points = True
            delta = points[-1][1]
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                linewidth=1.7,
                markersize=3.0,
                color=palette[mode],
                label=f"{mode} delta={delta:.0f}",
            )
        ylabel = "pause increase"
        if label == "spine":
            ylabel = "pause packets increase"
        ax.set_title(f"{label} PFC count increase")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    if not any_points:
        plt.close(fig)
        return
    axes[-1].set_xlabel("Seconds From Measured-Phase Start")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_switch_pfc_bar_summary(
    switch_metrics: dict[str, dict[str, float]],
    output_path: Path,
    *,
    title: str,
) -> None:
    if not switch_metrics:
        return
    modes = [mode for mode, row in switch_metrics.items() if row]
    if not modes:
        return
    palette = build_palette(modes)
    totals = [switch_metrics[mode]["switch_pfc_total"] for mode in modes]
    peaks = [switch_metrics[mode]["switch_pfc_peak_rate"] for mode in modes]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    axes[0].bar(modes, totals, color=[palette[mode] for mode in modes], alpha=0.85)
    axes[0].set_title("Switch PFC Total Delta By Mode")
    axes[0].set_ylabel("PFC Delta")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(modes, peaks, color=[palette[mode] for mode in modes], alpha=0.85)
    axes[1].set_title("Switch PFC Peak Rate By Mode")
    axes[1].set_ylabel("PFC / sec")
    axes[1].set_xlabel("Mode")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def align_step_cumulative_counter(records: Sequence[StepRecord], samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    bounds = measured_window_bounds(records)
    if bounds is None or not samples:
        return []
    start_ns, _ = bounds
    start_value = interpolate_switch_snapshot_value(samples, start_ns)
    if start_value is None:
        return []
    points: list[tuple[int, float]] = []
    for record in records:
        end_value = interpolate_switch_snapshot_value(samples, record.ts_end_unix_ns)
        if end_value is None:
            continue
        points.append((record.step, max(0.0, end_value - start_value)))
    return points


def align_step_rate(records: Sequence[StepRecord], rate_series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not rate_series:
        return []
    points: list[tuple[int, float]] = []
    idx = 0
    for record in records:
        best_value = 0.0
        while idx < len(rate_series) and rate_series[idx][0] < record.ts_start_unix_ns:
            idx += 1
        scan = idx
        found = False
        while scan < len(rate_series) and rate_series[scan][0] <= record.ts_end_unix_ns:
            best_value = max(best_value, rate_series[scan][1])
            found = True
            scan += 1
        if found:
            points.append((record.step, best_value))
    return points


def plot_stepwise_overlay(
    points_by_mode: dict[str, list[tuple[int, float]]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    modes = [mode for mode, points in points_by_mode.items() if points]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        xs = [step for step, _ in points_by_mode[mode]]
        ys = [value for _, value in points_by_mode[mode]]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.0, color=palette[mode], label=mode)
    ax.set_title(title)
    ax.set_xlabel("Measured Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rate_overlay_from_series(
    records_by_mode: dict[str, list[StepRecord]],
    series_by_mode: dict[str, list[tuple[int, float]]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    modes = [mode for mode, series in series_by_mode.items() if series]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        records = records_by_mode.get(mode, [])
        if not records:
            continue
        bounds = measured_window_bounds(records)
        if bounds is None:
            continue
        start_ns, end_ns = bounds
        xs: list[float] = []
        ys: list[float] = []
        for ts_ns, value in series_by_mode[mode]:
            if start_ns <= ts_ns <= end_ns:
                xs.append((ts_ns - start_ns) / 1e9)
                ys.append(value)
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=1.8, color=palette[mode], label=mode)
    ax.set_title(title)
    ax.set_xlabel("Time Since Measured-Phase Start (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mode_step_percentiles(
    records_by_repeat: dict[str, list[StepRecord]],
    output_path: Path,
    *,
    title: str,
    metric_key: str,
    ylabel: str,
    quantiles: Sequence[tuple[float, str]],
) -> None:
    step_to_values: dict[int, list[float]] = {}
    for records in records_by_repeat.values():
        for record in records:
            step_to_values.setdefault(record.step, []).append(float(getattr(record, metric_key)))
    if not step_to_values:
        return
    steps = sorted(step_to_values.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for idx, (q, label) in enumerate(quantiles):
        ys = [percentile(step_to_values[step], q) for step in steps]
        ax.plot(steps, ys, marker="o", linewidth=1.8, markersize=3.0, color=cmap(idx), label=label)
    ax.set_title(title)
    ax.set_xlabel("Measured Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarise_records(records: Sequence[StepRecord]) -> dict[str, float]:
    latencies = [record.step_ms_max for record in records]
    throughputs = [record.samples_per_sec for record in records]
    return {
        "count": float(len(records)),
        "latency_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "throughput_avg": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "throughput_p50": percentile(throughputs, 0.50),
        "throughput_p95": percentile(throughputs, 0.95),
    }


def write_html(
    run_root: Path,
    output_dir: Path,
    repeat_sections: list[tuple[str, list[tuple[str, str]]]],
    mode_sections: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Phase4 Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;border:1px solid #ddd;margin-bottom:16px;} code{background:#f4f4f4;padding:2px 4px;}</style>",
        "</head><body>",
        "<h1>Phase4 Report</h1>",
        f"<p><strong>Run root:</strong> <code>{run_root}</code></p>",
        "<p>Latency and throughput plots use measured steps only. POST plots use all NCCL phase0 POST events. Network throughput uses measured-phase <code>PROXY_RECV_NET_DONE size</code> aggregated into ms bins.</p>",
    ]
    for section_title, images in repeat_sections:
        html.append(f"<h2>{section_title}</h2>")
        for title, filename in images:
            html.append(f"<h3>{title}</h3>")
            html.append(f"<img src='{filename}' alt='{title}'>")
    for section_title, images in mode_sections:
        html.append(f"<h2>{section_title}</h2>")
        for title, filename in images:
            html.append(f"<h3>{title}</h3>")
            html.append(f"<img src='{filename}' alt='{title}'>")
    html.extend(["</body></html>"])
    (output_dir / "phase4_report.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase4-reporter] load run root={run_root}")

    raw_records_by_repeat_mode: dict[str, dict[str, list[StepRecord]]] = {}
    filtered_records_by_repeat_mode: dict[str, dict[str, list[StepRecord]]] = {}
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    repeat_dirs = discover_repeat_dirs(run_root)
    for repeat_dir in repeat_dirs:
        repeat_name = repeat_dir.name if repeat_dir != run_root else "repeat_01"
        raw_records_by_repeat_mode[repeat_name] = {}
        filtered_records_by_repeat_mode[repeat_name] = {}
        summary[repeat_name] = {}
        for mode_dir in iter_mode_dirs(repeat_dir):
            records = load_step_metrics(repeat_name, mode_dir)
            if not records:
                continue
            filtered_records, flagged = filtered_measured_records(
                records,
                method=args.outlier_method,
                mad_z=args.outlier_z,
                iqr_k=args.outlier_iqr_k,
            )
            raw_measured = measured_records(records)
            raw_records_by_repeat_mode[repeat_name][mode_dir.name] = raw_measured
            filtered_records_by_repeat_mode[repeat_name][mode_dir.name] = filtered_records
            summary[repeat_name][mode_dir.name] = {
                "raw": summarise_records(raw_measured),
                "filtered": summarise_records(filtered_records),
                "dropped_count": float(len(flagged)),
            }
            print(
                f"[phase4-reporter] loaded repeat={repeat_name} mode={mode_dir.name} "
                f"raw={len(raw_measured)} filtered={len(filtered_records)} dropped={len(flagged)}"
            )

    filtered_records_by_repeat_mode = {
        repeat_name: records_by_mode
        for repeat_name, records_by_mode in filtered_records_by_repeat_mode.items()
        if records_by_mode
    }
    raw_records_by_repeat_mode = {
        repeat_name: records_by_mode
        for repeat_name, records_by_mode in raw_records_by_repeat_mode.items()
        if records_by_mode
    }
    if not filtered_records_by_repeat_mode:
        raise FileNotFoundError(f"no *_step_metrics.jsonl found under {run_root}")

    congestion_bundle = load_congestion_bundle(run_root)

    repeat_sections: list[tuple[str, list[tuple[str, str]]]] = []
    for repeat_name, records_by_mode in filtered_records_by_repeat_mode.items():
        raw_records_by_mode = raw_records_by_repeat_mode[repeat_name]
        images: list[tuple[str, str]] = []
        latency_raw_png = f"{repeat_name}_latency_over_time_raw.png"
        latency_png = f"{repeat_name}_latency_over_time.png"
        latency_raw_median_band_png = f"{repeat_name}_latency_over_time_raw_zoom_median_band.png"
        latency_median_band_png = f"{repeat_name}_latency_over_time_zoom_median_band.png"
        throughput_raw_png = f"{repeat_name}_throughput_over_time_raw.png"
        throughput_png = f"{repeat_name}_throughput_over_time.png"
        throughput_raw_median_band_png = f"{repeat_name}_throughput_over_time_raw_zoom_median_band.png"
        throughput_median_band_png = f"{repeat_name}_throughput_over_time_zoom_median_band.png"
        timeline_png = f"{repeat_name}_step_timeline.png"
        post_overlay_png = f"{repeat_name}_cumulative_post_overlay.png"
        post_rate_png = f"{repeat_name}_post_rate_overlay.png"
        outstanding_png = f"{repeat_name}_outstanding_post_minus_received_overlay.png"
        netdone_cum_png = f"{repeat_name}_cumulative_netdone_bytes_overlay.png"
        network_tput_png = f"{repeat_name}_network_throughput_overlay.png"
        switch_overlay_png = f"{repeat_name}_switch_pfc_overlay.png"
        switch_summary_png = f"{repeat_name}_switch_pfc_summary.png"
        step_pfc_png = f"{repeat_name}_stepwise_pfc_cumulative_overlay.png"
        spine_ecn_png = f"{repeat_name}_spine_ecn_rate_overlay.png"
        worker_cnp_png = f"{repeat_name}_worker_cnp_rate_overlay.png"
        worker_tx_pause_png = f"{repeat_name}_worker_tx_pause_rate_overlay.png"
        worker_nic_tput_png = f"{repeat_name}_worker_nic_throughput_overlay.png"

        raw_latency_values = [record.step_ms_max for mode_records in raw_records_by_mode.values() for record in mode_records]
        raw_throughput_values = [record.samples_per_sec for mode_records in raw_records_by_mode.values() for record in mode_records]
        latency_values = [record.step_ms_max for mode_records in records_by_mode.values() for record in mode_records]
        throughput_values = [record.samples_per_sec for mode_records in records_by_mode.values() for record in mode_records]
        raw_latency_median_limits = centered_limits(raw_latency_values, min_half_span=0.5, frac=0.25)
        raw_throughput_median_limits = centered_limits(raw_throughput_values, min_half_span=500.0, frac=0.08)
        latency_median_limits = centered_limits(latency_values, min_half_span=0.5, frac=0.25)
        throughput_median_limits = centered_limits(throughput_values, min_half_span=500.0, frac=0.08)

        plot_metric_overlay(
            raw_records_by_mode,
            output_dir / latency_raw_png,
            title=f"{repeat_name} Latency Overlay Over Relative Mode Time (raw)",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
        )
        if raw_latency_median_limits is not None:
            plot_metric_overlay(
                raw_records_by_mode,
                output_dir / latency_raw_median_band_png,
                title=f"{repeat_name} Latency Overlay Over Relative Mode Time (raw, median-centered)",
                ylabel="Latency (ms)",
                metric_key="step_ms_max",
                y_limits=raw_latency_median_limits,
            )
        plot_metric_overlay(
            records_by_mode,
            output_dir / latency_png,
            title=f"{repeat_name} Latency Overlay Over Relative Mode Time (filtered)",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
        )
        if latency_median_limits is not None:
            plot_metric_overlay(
                records_by_mode,
                output_dir / latency_median_band_png,
                title=f"{repeat_name} Latency Overlay Over Relative Mode Time (filtered, median-centered)",
                ylabel="Latency (ms)",
                metric_key="step_ms_max",
                y_limits=latency_median_limits,
            )
        plot_metric_overlay(
            raw_records_by_mode,
            output_dir / throughput_raw_png,
            title=f"{repeat_name} Training Throughput Overlay Over Relative Mode Time (raw)",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
        )
        if raw_throughput_median_limits is not None:
            plot_metric_overlay(
                raw_records_by_mode,
                output_dir / throughput_raw_median_band_png,
                title=f"{repeat_name} Training Throughput Overlay Over Relative Mode Time (raw, median-centered)",
                ylabel="Samples / sec",
                metric_key="samples_per_sec",
                y_limits=raw_throughput_median_limits,
            )
        plot_metric_overlay(
            records_by_mode,
            output_dir / throughput_png,
            title=f"{repeat_name} Training Throughput Overlay Over Relative Mode Time (filtered)",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
        )
        if throughput_median_limits is not None:
            plot_metric_overlay(
                records_by_mode,
                output_dir / throughput_median_band_png,
                title=f"{repeat_name} Training Throughput Overlay Over Relative Mode Time (filtered, median-centered)",
                ylabel="Samples / sec",
                metric_key="samples_per_sec",
                y_limits=throughput_median_limits,
            )
        plot_step_timeline(
            records_by_mode,
            output_dir / timeline_png,
            title=f"{repeat_name} Measured Step Timeline Overlay",
        )

        repeat_dir = run_root / repeat_name if (run_root / repeat_name).exists() else run_root
        phase0_events_by_mode = {mode: load_phase0_events(repeat_dir / mode) for mode in records_by_mode.keys()}
        post_ms_by_mode = {
            mode: [event.t_ms for event in phase0_events_by_mode[mode] if event.event == "PROXY_RECV_POST"]
            for mode in records_by_mode.keys()
        }
        measured_netdone_by_mode = {
            mode: filter_events_to_measured_steps(
                [event for event in phase0_events_by_mode[mode] if event.event == "PROXY_RECV_NET_DONE" and event.size > 0],
                records_by_mode[mode],
            )
            for mode in records_by_mode.keys()
        }
        plot_post_cumulative_overlay(
            post_ms_by_mode,
            output_dir / post_overlay_png,
            title=f"{repeat_name} Cumulative PROXY_RECV_POST Over Relative Mode Time",
        )
        plot_post_rate_overlay(
            post_ms_by_mode,
            output_dir / post_rate_png,
            title=f"{repeat_name} PROXY_RECV_POST Rate Over Relative Mode Time",
        )
        plot_outstanding_overlay(
            phase0_events_by_mode,
            output_dir / outstanding_png,
            title=f"{repeat_name} Outstanding (posted-received) Over Relative Mode Time",
        )
        plot_netdone_cumulative_overlay(
            measured_netdone_by_mode,
            output_dir / netdone_cum_png,
            title=f"{repeat_name} Cumulative PROXY_RECV_NET_DONE Bytes Over Relative Mode Time",
        )
        plot_network_throughput_overlay(
            measured_netdone_by_mode,
            output_dir / network_tput_png,
            title=f"{repeat_name} Network Throughput Over Relative Mode Time",
        )

        switch_metrics = compute_switch_metrics_for_modes(congestion_bundle, records_by_mode)
        summary[repeat_name]["switch"] = switch_metrics  # type: ignore[index]
        if congestion_bundle:
            plot_switch_pfc_overlay(
                congestion_bundle,
                records_by_mode,
                output_dir / switch_overlay_png,
                title=f"{repeat_name} Switch PFC Delta Overlay",
            )
            plot_switch_pfc_bar_summary(
                switch_metrics,
                output_dir / switch_summary_png,
                title=f"{repeat_name} Switch Congestion Validation",
            )
            plot_stepwise_overlay(
                {
                    mode: align_step_cumulative_counter(records_by_mode[mode], congestion_bundle["snapshots"].get("rackA", []))
                    for mode in records_by_mode.keys()
                },
                output_dir / f"{repeat_name}_stepwise_rackA_pfc_cumulative_overlay.png",
                title=f"{repeat_name} Stepwise rackA PFC Cumulative Increase",
                ylabel="PFC Delta",
            )
            plot_stepwise_overlay(
                {
                    mode: align_step_cumulative_counter(records_by_mode[mode], congestion_bundle["snapshots"].get("spine", []))
                    for mode in records_by_mode.keys()
                },
                output_dir / f"{repeat_name}_stepwise_spine_pfc_cumulative_overlay.png",
                title=f"{repeat_name} Stepwise spine PFC Cumulative Increase",
                ylabel="PFC Delta",
            )
            total_pfc_points = {}
            for mode in records_by_mode.keys():
                rack_points = align_step_cumulative_counter(records_by_mode[mode], congestion_bundle["snapshots"].get("rackA", []))
                rackb_points = align_step_cumulative_counter(records_by_mode[mode], congestion_bundle["snapshots"].get("rackB", []))
                spine_points = align_step_cumulative_counter(records_by_mode[mode], congestion_bundle["snapshots"].get("spine", []))
                if rack_points or rackb_points or spine_points:
                    merged = {}
                    for points in (rack_points, rackb_points, spine_points):
                        for step, value in points:
                            merged[step] = merged.get(step, 0.0) + value
                    total_pfc_points[mode] = sorted(merged.items())
            plot_stepwise_overlay(
                total_pfc_points,
                output_dir / step_pfc_png,
                title=f"{repeat_name} Stepwise Total Switch PFC Cumulative Increase",
                ylabel="PFC Delta",
            )
            plot_rate_overlay_from_series(
                records_by_mode,
                {mode: congestion_bundle.get("spine_ecn_rate_series", []) for mode in records_by_mode.keys()},
                output_dir / spine_ecn_png,
                title=f"{repeat_name} Spine ECN Marking Rate",
                ylabel="ECN packets / sec",
            )
            plot_rate_overlay_from_series(
                records_by_mode,
                {mode: congestion_bundle.get("worker_cnp_rate_series", []) for mode in records_by_mode.keys()},
                output_dir / worker_cnp_png,
                title=f"{repeat_name} Worker NIC CNP Send Rate",
                ylabel="CNP / sec",
            )
            plot_rate_overlay_from_series(
                records_by_mode,
                {mode: congestion_bundle.get("worker_tx_pause_rate_series", []) for mode in records_by_mode.keys()},
                output_dir / worker_tx_pause_png,
                title=f"{repeat_name} Worker NIC TX Pause Rate",
                ylabel="TX pause / sec",
            )
            plot_rate_overlay_from_series(
                records_by_mode,
                {mode: congestion_bundle.get("worker_total_gbps_series", []) for mode in records_by_mode.keys()},
                output_dir / worker_nic_tput_png,
                title=f"{repeat_name} Worker NIC Total Throughput",
                ylabel="Gbps",
            )

        images.extend([
            ("Latency Overlay Raw", latency_raw_png),
            ("Latency Overlay Raw Median-Centered", latency_raw_median_band_png),
            ("Latency Overlay Filtered", latency_png),
            ("Latency Overlay Filtered Median-Centered", latency_median_band_png),
            ("Training Throughput Overlay Raw", throughput_raw_png),
            ("Training Throughput Overlay Raw Median-Centered", throughput_raw_median_band_png),
            ("Training Throughput Overlay Filtered", throughput_png),
            ("Training Throughput Overlay Filtered Median-Centered", throughput_median_band_png),
            ("Measured Step Timeline Overlay", timeline_png),
            ("Cumulative POST Overlay", post_overlay_png),
            ("POST Rate Overlay", post_rate_png),
            ("Outstanding (posted-received) Overlay", outstanding_png),
            ("Cumulative NET_DONE Bytes Overlay", netdone_cum_png),
            ("Network Throughput Gbps Overlay", network_tput_png),
        ])
        if congestion_bundle:
            images.extend([
                ("Switch PFC Overlay", switch_overlay_png),
                ("Switch PFC Summary", switch_summary_png),
                ("Stepwise Total Switch PFC Cumulative Overlay", step_pfc_png),
                ("Spine ECN Marking Rate Overlay", spine_ecn_png),
                ("Worker NIC CNP Send Rate Overlay", worker_cnp_png),
                ("Worker NIC TX Pause Rate Overlay", worker_tx_pause_png),
                ("Worker NIC Throughput Overlay", worker_nic_tput_png),
            ])
        repeat_sections.append((f"Repeat {repeat_name}", images))

    mode_to_repeat_records_raw: dict[str, dict[str, list[StepRecord]]] = {}
    mode_to_repeat_records: dict[str, dict[str, list[StepRecord]]] = {}
    for repeat_name, records_by_mode in raw_records_by_repeat_mode.items():
        for mode, records in records_by_mode.items():
            mode_to_repeat_records_raw.setdefault(mode, {})[repeat_name] = records
    for repeat_name, records_by_mode in filtered_records_by_repeat_mode.items():
        for mode, records in records_by_mode.items():
            mode_to_repeat_records.setdefault(mode, {})[repeat_name] = records

    mode_sections: list[tuple[str, list[tuple[str, str]]]] = []
    all_latency_values_by_mode_raw: dict[str, list[float]] = {}
    all_throughput_values_by_mode_raw: dict[str, list[float]] = {}
    all_latency_values_by_mode: dict[str, list[float]] = {}
    all_throughput_values_by_mode: dict[str, list[float]] = {}
    for mode in sorted(mode_to_repeat_records.keys()):
        raw_records_by_repeat = mode_to_repeat_records_raw.get(mode, {})
        records_by_repeat = mode_to_repeat_records[mode]
        latency_raw_png = f"{mode}_repeat_latency_overlay_raw.png"
        latency_png = f"{mode}_repeat_latency_overlay.png"
        throughput_raw_png = f"{mode}_repeat_throughput_overlay_raw.png"
        throughput_png = f"{mode}_repeat_throughput_overlay.png"
        latency_pct_png = f"{mode}_step_latency_percentiles.png"
        throughput_pct_png = f"{mode}_step_throughput_percentiles.png"
        plot_repeat_metric_overlay(
            raw_records_by_repeat,
            output_dir / latency_raw_png,
            title=f"{mode} Latency Across Repeats (raw)",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
        )
        plot_repeat_metric_overlay(
            records_by_repeat,
            output_dir / latency_png,
            title=f"{mode} Latency Across Repeats (filtered)",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
        )
        plot_repeat_metric_overlay(
            raw_records_by_repeat,
            output_dir / throughput_raw_png,
            title=f"{mode} Training Throughput Across Repeats (raw)",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
        )
        plot_repeat_metric_overlay(
            records_by_repeat,
            output_dir / throughput_png,
            title=f"{mode} Training Throughput Across Repeats (filtered)",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
        )
        plot_mode_step_percentiles(
            records_by_repeat,
            output_dir / latency_pct_png,
            title=f"{mode} Stepwise Latency Percentiles Across Repeats",
            metric_key="step_ms_max",
            ylabel="Latency (ms)",
            quantiles=((0.50, "p50"), (0.95, "p95"), (0.99, "p99")),
        )
        plot_mode_step_percentiles(
            records_by_repeat,
            output_dir / throughput_pct_png,
            title=f"{mode} Stepwise Throughput Percentiles Across Repeats",
            metric_key="samples_per_sec",
            ylabel="Samples / sec",
            quantiles=((0.01, "p01"), (0.05, "p05"), (0.50, "p50")),
        )
        all_latency_values_by_mode_raw[mode] = [record.step_ms_max for records in raw_records_by_repeat.values() for record in records]
        all_throughput_values_by_mode_raw[mode] = [record.samples_per_sec for records in raw_records_by_repeat.values() for record in records]
        all_latency_values_by_mode[mode] = [record.step_ms_max for records in records_by_repeat.values() for record in records]
        all_throughput_values_by_mode[mode] = [record.samples_per_sec for records in records_by_repeat.values() for record in records]
        mode_sections.append((f"Mode {mode}", [
            ("Latency Across Repeats Raw", latency_raw_png),
            ("Latency Across Repeats Filtered", latency_png),
            ("Training Throughput Across Repeats Raw", throughput_raw_png),
            ("Training Throughput Across Repeats Filtered", throughput_png),
            ("Stepwise Latency Percentiles Across Repeats", latency_pct_png),
            ("Stepwise Throughput Percentiles Across Repeats", throughput_pct_png),
        ]))

    box_latency_raw_png = "all_repeats_latency_boxplot_raw.png"
    box_latency_png = "all_repeats_latency_boxplot.png"
    box_throughput_raw_png = "all_repeats_throughput_boxplot_raw.png"
    box_throughput_png = "all_repeats_throughput_boxplot.png"
    plot_boxplot_by_mode(
        all_latency_values_by_mode_raw,
        output_dir / box_latency_raw_png,
        title="Latency Distribution Across Repeats (raw)",
        ylabel="Latency (ms)",
    )
    plot_boxplot_by_mode(
        all_latency_values_by_mode,
        output_dir / box_latency_png,
        title="Latency Distribution Across Repeats (filtered)",
        ylabel="Latency (ms)",
    )
    plot_boxplot_by_mode(
        all_throughput_values_by_mode_raw,
        output_dir / box_throughput_raw_png,
        title="Training Throughput Distribution Across Repeats (raw)",
        ylabel="Samples / sec",
    )
    plot_boxplot_by_mode(
        all_throughput_values_by_mode,
        output_dir / box_throughput_png,
        title="Training Throughput Distribution Across Repeats (filtered)",
        ylabel="Samples / sec",
    )
    mode_sections.append(("All Repeats Summary", [
        ("Latency Distribution Across Repeats Raw", box_latency_raw_png),
        ("Latency Distribution Across Repeats Filtered", box_latency_png),
        ("Training Throughput Distribution Across Repeats Raw", box_throughput_raw_png),
        ("Training Throughput Distribution Across Repeats Filtered", box_throughput_png),
    ]))

    write_html(run_root, output_dir, repeat_sections, mode_sections)
    (output_dir / "phase4_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[phase4-reporter] done html={output_dir / 'phase4_report.html'}")


if __name__ == "__main__":
    main()
