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
    samples_per_sec: float


@dataclass
class DelayEvent:
    t_ns: int
    delay_ms: float
    size_bytes: int


@dataclass
class MarkerEvent:
    t_ns: int
    marker: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase5 E(t) congestion reporter")
    parser.add_argument("--input", required=True, help="phase5 run root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-ms", type=int, default=100, help="bucket size in ms for E(t) aggregation")
    return parser.parse_args()


def parse_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV_RE.finditer(line)}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_jsonl_decoder(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
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
                    record, next_index = decoder.raw_decode(line, index)
                except json.JSONDecodeError:
                    break
                if isinstance(record, dict):
                    rows.append(record)
                index = next_index
    return rows


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


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def extract_record_ts_ns(row: dict) -> Optional[int]:
    for key in (
        "ts_unix_ns",
        "ts_mid_unix_ns",
        "ts_start_unix_ns",
        "ts_end_unix_ns",
        "timestamp_ns",
        "time_ns",
        "ts_ns",
        "tNs",
    ):
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


def discover_repeat_dirs(run_root: Path) -> list[Path]:
    repeat_dirs = sorted(p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    return repeat_dirs or [run_root]


def iter_mode_dirs(repeat_dir: Path) -> Iterable[Path]:
    for candidate in sorted(p for p in repeat_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        yield candidate


def load_step_metrics(repeat_name: str, mode_dir: Path) -> list[StepRecord]:
    metrics_files = sorted(mode_dir.glob("*_step_metrics.jsonl"))
    if not metrics_files:
        return []
    rows: list[StepRecord] = []
    for row in load_jsonl(metrics_files[0]):
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


def load_phase5_delay_events(mode_dir: Path) -> list[DelayEvent]:
    events: list[DelayEvent] = []
    for log_path in _discover_nccl_logs(mode_dir):
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PHASE5 event=PROXY_RECV_NET_DONE" not in line:
                    continue
                row = parse_kv(line)
                if row.get("event") != "PROXY_RECV_NET_DONE" or row.get("tNs") is None:
                    continue
                try:
                    events.append(
                        DelayEvent(
                            t_ns=int(row["tNs"]),
                            delay_ms=int(row.get("postToNetDoneNs", "0")) / 1e6,
                            size_bytes=int(row.get("size", "0")),
                        )
                    )
                except ValueError:
                    continue
    events.sort(key=lambda item: item.t_ns)
    return events


def measured_records(records: Sequence[StepRecord]) -> list[StepRecord]:
    return [record for record in records if not record.warmup]


def measured_window_bounds(records: Sequence[StepRecord]) -> Optional[tuple[int, int]]:
    measured = measured_records(records)
    if not measured:
        return None
    return min(r.ts_start_unix_ns for r in measured), max(r.ts_end_unix_ns for r in measured)


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


def load_markers(log_dir: Path) -> list[MarkerEvent]:
    markers: list[MarkerEvent] = []
    for row in read_jsonl_decoder(log_dir / "markers.jsonl"):
        ts_ns = extract_record_ts_ns(row)
        marker = row.get("marker")
        if ts_ns is None or not isinstance(marker, str):
            continue
        markers.append(MarkerEvent(t_ns=ts_ns, marker=marker, message=str(row.get("message", ""))))
    markers.sort(key=lambda item: item.t_ns)
    return markers


def load_raw_switch_counter_records(log_dir: Path) -> list[dict]:
    records: list[dict] = []
    for name in ("spine_pfc_ecn.jsonl", "rackA_pfc_statistics.jsonl", "rackB_pfc_statistics.jsonl"):
        records.extend(read_jsonl_decoder(log_dir / name))
    return records


def build_switch_events(records: Iterable[dict]) -> tuple[dict[str, list[tuple[int, int]]], Optional[int]]:
    by_key: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    min_ts_ns: Optional[int] = None
    for record in records:
        kind = record.get("kind")
        switch = record.get("switch")
        port = record.get("port")
        ts_ns = record.get("ts_mid_unix_ns")
        if kind is None or switch is None or port is None or ts_ns is None:
            continue
        metrics: list[tuple[str, Optional[int]]] = []
        if kind == "fsos_pfc_statistics":
            rx = record.get("rx_pause")
            tx = record.get("tx_pause")
            metrics.append(("pause", None if rx is None or tx is None else int(rx) + int(tx)))
        elif kind == "onyx_pfc_ecn":
            rx_pause = record.get("rx_pause_packets")
            tx_pause = record.get("tx_pause_packets")
            ecn = record.get("rx_ecn_marked_packets")
            metrics.append(("pause", None if rx_pause is None or tx_pause is None else int(rx_pause) + int(tx_pause)))
            metrics.append(("ecn", None if ecn is None else int(ecn)))
        else:
            continue
        ts_i = int(ts_ns)
        if min_ts_ns is None or ts_i < min_ts_ns:
            min_ts_ns = ts_i
        for metric, value in metrics:
            if value is None:
                continue
            key = (str(switch), str(port), metric)
            by_key.setdefault(key, []).append((ts_i, value))

    events: dict[str, list[tuple[int, int]]] = {}
    for (switch, _port, metric), samples in by_key.items():
        samples.sort(key=lambda item: item[0])
        prev: Optional[int] = None
        series: list[tuple[int, int]] = []
        for ts_i, cur in samples:
            delta = 0 if prev is None else max(0, cur - prev)
            series.append((ts_i, delta))
            prev = cur
        key = f"{switch}:{metric}"
        events.setdefault(key, []).extend(series)
    for key in events:
        events[key].sort(key=lambda item: item[0])
    return events, min_ts_ns


def bucketize_switch_events(
    events_by_series: dict[str, list[tuple[int, int]]],
    bucket_sec: float,
    *,
    start_ns: int,
    end_ns: int,
) -> dict[str, list[tuple[float, int]]]:
    bucket_ns = int(bucket_sec * 1_000_000_000.0)
    if bucket_ns <= 0:
        return {}
    bucketed: dict[str, dict[int, int]] = {}
    for series_name, events in events_by_series.items():
        for ts_ns, delta in events:
            if ts_ns < start_ns or ts_ns > end_ns:
                continue
            bucket_idx = int((ts_ns - start_ns) // bucket_ns)
            bucketed.setdefault(series_name, {})
            bucketed[series_name][bucket_idx] = bucketed[series_name].get(bucket_idx, 0) + int(delta)
    out: dict[str, list[tuple[float, int]]] = {}
    for name, buckets in bucketed.items():
        out[name] = [((idx * bucket_sec), value) for idx, value in sorted(buckets.items())]
    return out


def parse_marker_repeat_mode(message: str) -> tuple[Optional[str], Optional[str]]:
    repeat = None
    mode = None
    for token in message.split():
        if token.startswith("repeat="):
            repeat = token.split("=", 1)[1]
        elif token.startswith("mode="):
            mode = token.split("=", 1)[1]
    return repeat, mode


def compute_repeat_time_bounds(records_by_mode: dict[str, list[StepRecord]]) -> Optional[tuple[int, int]]:
    starts: list[int] = []
    ends: list[int] = []
    for records in records_by_mode.values():
        bounds = measured_window_bounds(records)
        if bounds is None:
            continue
        starts.append(bounds[0])
        ends.append(bounds[1])
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def plot_switch_overview(
    bucketed: dict[str, list[tuple[float, int]]],
    markers: list[tuple[float, str]],
    output_path: Path,
    *,
    title: str,
) -> None:
    layout = [
        ("spine:pause", "spine PFC delta/bucket", "#1f77b4"),
        ("spine:ecn", "spine ECN delta/bucket", "#ff7f0e"),
        ("rackA:pause", "rackA PFC delta/bucket", "#d62728"),
        ("rackB:pause", "rackB PFC delta/bucket", "#2ca02c"),
    ]
    active = [item for item in layout if bucketed.get(item[0])]
    if not active:
        return
    fig, axes = plt.subplots(len(active), 1, figsize=(14, 9), sharex=True)
    if len(active) == 1:
        axes = [axes]
    all_x: list[float] = []
    for ax, (series_name, ylabel, color) in zip(axes, active):
        xs = [x for x, _ in bucketed[series_name]]
        ys = [y for _, y in bucketed[series_name]]
        all_x.extend(xs)
        ax.plot(xs, ys, linewidth=1.6, color=color)
        ax.fill_between(xs, ys, step="pre", alpha=0.18, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        y_top = max(ys) if ys else 1.0
        for marker_x, marker_name in markers:
            ax.axvline(marker_x, color="#555555", linestyle=":", alpha=0.45, linewidth=1.0)
            ax.text(marker_x, y_top if y_top > 0 else 1.0, marker_name, rotation=90, va="top", ha="right", fontsize=8, color="#555555")
    if all_x:
        axes[-1].set_xlim(0, max(all_x))
    axes[-1].set_xlabel("Elapsed time since repeat start (seconds)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mode_overlay(
    series_by_mode: dict[str, list[tuple[float, int]]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    if not any(series_by_mode.values()):
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for mode_name, series in sorted(series_by_mode.items()):
        if not series:
            continue
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        ax.plot(xs, ys, linewidth=1.7, marker="o", markersize=2.8, label=mode_name)
    ax.set_title(title)
    ax.set_xlabel("Relative mode time (seconds)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_total_series_from_jsonl(path: Path, total_field: str) -> list[tuple[int, float]]:
    samples: list[tuple[int, float]] = []
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        value = row.get(total_field)
        if ts_ns is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        samples.append((ts_ns, float(value)))
    samples.sort(key=lambda item: item[0])
    return samples


def compute_counter_rate_series(samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    rates: list[tuple[int, float]] = []
    for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
        elapsed = (cur_ts - prev_ts) / 1_000_000_000.0
        if elapsed <= 0:
            continue
        rates.append((cur_ts, max(0.0, cur_total - prev_total) / elapsed))
    return rates


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
            "ecn": "rx_ecn_marked_packets_total",
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


def load_congestion_bundle(run_root: Path) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(run_root)
    if log_dir is None or not log_dir.exists():
        return None
    snapshots: dict[str, list[tuple[int, float]]] = {}
    for label, path in {
        "rackA": log_dir / "rackA_pfc_aggregate.jsonl",
        "rackB": log_dir / "rackB_pfc_aggregate.jsonl",
        "spine": log_dir / "spine_pfc_ecn_aggregate.jsonl",
    }.items():
        if not path.exists():
            continue
        primary = choose_primary_switch_series(load_aggregate_switch_file(path, label))
        if primary:
            snapshots[label] = primary
    spine_agg = log_dir / "spine_pfc_ecn_aggregate.jsonl"
    bundle = {
        "log_dir": log_dir,
        "snapshots": snapshots,
        "spine_ecn_rate_series": compute_counter_rate_series(
            load_total_series_from_jsonl(spine_agg, "rx_ecn_marked_packets_total")
        ),
    }
    if snapshots or bundle["spine_ecn_rate_series"]:
        return bundle
    return None


def interpolate_series_value(samples: list[tuple[int, float]], ts_ns: int) -> Optional[float]:
    if not samples:
        return None
    if ts_ns <= samples[0][0]:
        return float(samples[0][1])
    if ts_ns >= samples[-1][0]:
        return float(samples[-1][1])
    for (left_ts, left_value), (right_ts, right_value) in zip(samples, samples[1:]):
        if left_ts <= ts_ns <= right_ts:
            if right_ts == left_ts:
                return float(right_value)
            ratio = (ts_ns - left_ts) / float(right_ts - left_ts)
            return float(left_value + (right_value - left_value) * ratio)
    return None


def align_step_delta_counter(records: Sequence[StepRecord], samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for record in measured_records(records):
        start_value = interpolate_series_value(samples, record.ts_start_unix_ns)
        end_value = interpolate_series_value(samples, record.ts_end_unix_ns)
        if start_value is None or end_value is None:
            continue
        points.append((record.step, max(0.0, end_value - start_value)))
    return points


def align_step_cumulative_counter(records: Sequence[StepRecord], samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    bounds = measured_window_bounds(records)
    if bounds is None or not samples:
        return []
    start_ns, _ = bounds
    start_value = interpolate_series_value(samples, start_ns)
    if start_value is None:
        return []
    points: list[tuple[int, float]] = []
    for record in measured_records(records):
        end_value = interpolate_series_value(samples, record.ts_end_unix_ns)
        if end_value is None:
            continue
        points.append((record.step, max(0.0, end_value - start_value)))
    return points


def align_step_rate(records: Sequence[StepRecord], rate_series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not rate_series:
        return []
    points: list[tuple[int, float]] = []
    idx = 0
    for record in measured_records(records):
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


def combine_step_series(series_list: Sequence[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for series in series_list:
        for step, value in series:
            by_step[step] = by_step.get(step, 0.0) + value
    return sorted(by_step.items())


def compute_stepwise_delay_stats(records: Sequence[StepRecord], events: Sequence[DelayEvent]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    measured = measured_records(records)
    event_idx = 0
    for record in measured:
        delays: list[float] = []
        while event_idx < len(events) and events[event_idx].t_ns < record.ts_start_unix_ns:
            event_idx += 1
        scan = event_idx
        while scan < len(events) and events[scan].t_ns <= record.ts_end_unix_ns:
            delays.append(events[scan].delay_ms)
            scan += 1
        rows.append(
            {
                "step": float(record.step),
                "count": float(len(delays)),
                "mean_ms": mean(delays),
                "p95_ms": percentile(delays, 0.95),
                "p99_ms": percentile(delays, 0.99),
            }
        )
    return rows


def compute_bucket_delay_stats(records: Sequence[StepRecord], events: Sequence[DelayEvent], bucket_ms: int) -> list[dict[str, float]]:
    bounds = measured_window_bounds(records)
    if bounds is None:
        return []
    start_ns, end_ns = bounds
    bucket_ns = int(bucket_ms * 1_000_000)
    if bucket_ns <= 0:
        return []
    buckets: dict[int, list[float]] = {}
    for event in events:
        if not (start_ns <= event.t_ns <= end_ns):
            continue
        bucket_idx = int((event.t_ns - start_ns) // bucket_ns)
        buckets.setdefault(bucket_idx, []).append(event.delay_ms)
    rows: list[dict[str, float]] = []
    for bucket_idx in sorted(buckets):
        delays = buckets[bucket_idx]
        rows.append(
            {
                "bucket_ms": (bucket_idx + 0.5) * bucket_ms,
                "count": float(len(delays)),
                "mean_ms": mean(delays),
                "p95_ms": percentile(delays, 0.95),
                "p99_ms": percentile(delays, 0.99),
            }
        )
    return rows


def build_bucket_rate_series(
    start_ns: int,
    end_ns: int,
    bucket_ms: int,
    rate_series: list[tuple[int, float]],
) -> list[tuple[float, float]]:
    if not rate_series or end_ns <= start_ns:
        return []
    bucket_ns = int(bucket_ms * 1_000_000)
    points: list[tuple[float, float]] = []
    idx = 0
    bucket_start = start_ns
    while bucket_start < end_ns:
        bucket_end = min(end_ns, bucket_start + bucket_ns)
        best = 0.0
        while idx < len(rate_series) and rate_series[idx][0] < bucket_start:
            idx += 1
        scan = idx
        found = False
        while scan < len(rate_series) and rate_series[scan][0] <= bucket_end:
            best = max(best, rate_series[scan][1])
            found = True
            scan += 1
        if found:
            center_ms = ((bucket_start - start_ns) + (bucket_end - bucket_start) / 2) / 1_000_000.0
            points.append((center_ms, best))
        bucket_start += bucket_ns
    return points


def plot_dual_axis(
    xs_left: Sequence[float],
    ys_left: Sequence[float],
    xs_right: Sequence[float],
    ys_right: Sequence[float],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    left_label: str,
    right_label: str,
) -> None:
    if not xs_left and not xs_right:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax2 = ax1.twinx()
    if xs_left:
        ax1.plot(xs_left, ys_left, color="tab:blue", marker="o", markersize=3.0, linewidth=1.7, label=left_label)
    if xs_right:
        ax2.plot(xs_right, ys_right, color="tab:red", marker="s", markersize=3.0, linewidth=1.4, alpha=0.85, label=right_label)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(left_label, color="tab:blue")
    ax2.set_ylabel(right_label, color="tab:red")
    ax1.grid(True, alpha=0.25)
    handles = []
    labels = []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax1.legend(handles, labels, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_html(output_dir: Path, images: list[tuple[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang='ko'><head><meta charset='utf-8'><title>Phase5 Report</title></head><body>",
        "<h1>Phase5 E(t) vs Congestion Report</h1>",
        "<p>E(t)=ts_net_done_ns-ts_data_post_ns 기반의 receiver-local delay proxy와 switch PFC/ECN congestion 신호를 비교한다.</p>",
        "<ul>",
    ]
    for title, filename in images:
        lines.append(f"<li><a href='{filename}'>{title}</a></li>")
    lines.extend(["</ul>", "</body></html>"])
    (output_dir / "phase5_report.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repeats = discover_repeat_dirs(run_root)
    congestion_bundle = load_congestion_bundle(run_root)
    switch_log_dir = resolve_switch_log_dir(run_root)
    raw_switch_records = load_raw_switch_counter_records(switch_log_dir) if switch_log_dir else []
    switch_events, _ = build_switch_events(raw_switch_records) if raw_switch_records else ({}, None)
    switch_markers = load_markers(switch_log_dir) if switch_log_dir else []
    summary: dict[str, dict] = {"bucket_ms": args.bucket_ms, "repeats": {}}
    html_images: list[tuple[str, str]] = []

    for repeat_dir in repeats:
        repeat_name = repeat_dir.name
        summary["repeats"][repeat_name] = {}
        records_by_mode: dict[str, list[StepRecord]] = {}
        for mode_dir in iter_mode_dirs(repeat_dir):
            records = load_step_metrics(repeat_name, mode_dir)
            if not records:
                continue
            records_by_mode[mode_dir.name] = records
            events = load_phase5_delay_events(mode_dir)
            if not events:
                continue

            mode_name = mode_dir.name
            step_stats = compute_stepwise_delay_stats(records, events)
            bucket_stats = compute_bucket_delay_stats(records, events, args.bucket_ms)
            bounds = measured_window_bounds(records)
            if bounds is None:
                continue
            start_ns, end_ns = bounds

            pfc_delta_step: list[tuple[int, float]] = []
            pfc_cum_step: list[tuple[int, float]] = []
            pfc_bucket: list[tuple[float, float]] = []
            spine_ecn_step: list[tuple[int, float]] = []
            if congestion_bundle:
                pfc_delta_step = combine_step_series(
                    [align_step_delta_counter(records, congestion_bundle["snapshots"].get(label, [])) for label in SWITCH_LABELS]
                )
                pfc_cum_step = combine_step_series(
                    [align_step_cumulative_counter(records, congestion_bundle["snapshots"].get(label, [])) for label in SWITCH_LABELS]
                )
                tmp: dict[float, float] = {}
                for label in SWITCH_LABELS:
                    label_rates = compute_counter_rate_series(congestion_bundle["snapshots"].get(label, []))
                    for x_ms, value in build_bucket_rate_series(start_ns, end_ns, args.bucket_ms, label_rates):
                        tmp[x_ms] = tmp.get(x_ms, 0.0) + value
                pfc_bucket = sorted(tmp.items())
                spine_ecn_step = align_step_rate(records, congestion_bundle.get("spine_ecn_rate_series", []))

            step_x = [row["step"] for row in step_stats]
            step_et_p99 = [row["p99_ms"] for row in step_stats]
            bucket_x = [row["bucket_ms"] for row in bucket_stats]
            bucket_et_p99 = [row["p99_ms"] for row in bucket_stats]

            prefix = f"{repeat_name}_{mode_name}"
            name1 = f"{prefix}_stepwise_et_p99_vs_pfc_delta.png"
            plot_dual_axis(
                step_x,
                step_et_p99,
                [x for x, _ in pfc_delta_step],
                [y for _, y in pfc_delta_step],
                output_dir / name1,
                title=f"{repeat_name} {mode_name} Stepwise E(t) p99 vs Switch PFC Delta",
                xlabel="Measured Step",
                left_label="E(t) p99 (ms)",
                right_label="Switch PFC Delta",
            )
            html_images.append((f"{repeat_name} {mode_name} Stepwise E(t) p99 vs PFC Delta", name1))

            name2 = f"{prefix}_bucket_et_p99_vs_pfc_rate.png"
            plot_dual_axis(
                bucket_x,
                bucket_et_p99,
                [x for x, _ in pfc_bucket],
                [y for _, y in pfc_bucket],
                output_dir / name2,
                title=f"{repeat_name} {mode_name} Bucketed E(t) p99 vs Switch PFC Rate",
                xlabel=f"Measured Phase Time ({args.bucket_ms}ms buckets)",
                left_label="E(t) p99 (ms)",
                right_label="Switch PFC / sec",
            )
            html_images.append((f"{repeat_name} {mode_name} Bucketed E(t) p99 vs PFC Rate", name2))

            name3 = f"{prefix}_stepwise_et_p99_vs_spine_ecn.png"
            plot_dual_axis(
                step_x,
                step_et_p99,
                [x for x, _ in spine_ecn_step],
                [y for _, y in spine_ecn_step],
                output_dir / name3,
                title=f"{repeat_name} {mode_name} Stepwise E(t) p99 vs Spine ECN Rate",
                xlabel="Measured Step",
                left_label="E(t) p99 (ms)",
                right_label="Spine ECN / sec",
            )
            html_images.append((f"{repeat_name} {mode_name} Stepwise E(t) p99 vs Spine ECN", name3))

            pfc_vals = [y for _, y in pfc_delta_step]
            ecn_vals = [y for _, y in spine_ecn_step]
            summary["repeats"][repeat_name][mode_name] = {
                "event_count": len(events),
                "et_mean_ms": mean([event.delay_ms for event in events]),
                "et_p95_ms": percentile([event.delay_ms for event in events], 0.95),
                "et_p99_ms": percentile([event.delay_ms for event in events], 0.99),
                "step_corr_pfc_delta": pearson_corr(step_et_p99[: min(len(step_et_p99), len(pfc_vals))], pfc_vals[: min(len(step_et_p99), len(pfc_vals))]) if pfc_vals else 0.0,
                "step_corr_spine_ecn_rate": pearson_corr(step_et_p99[: min(len(step_et_p99), len(ecn_vals))], ecn_vals[: min(len(step_et_p99), len(ecn_vals))]) if ecn_vals else 0.0,
                "step_pfc_total_delta": pfc_cum_step[-1][1] if pfc_cum_step else 0.0
            }

        repeat_bounds = compute_repeat_time_bounds(records_by_mode)
        if repeat_bounds and switch_events:
            rep_start_ns, rep_end_ns = repeat_bounds
            repeat_bucketed = bucketize_switch_events(switch_events, 1.0, start_ns=rep_start_ns, end_ns=rep_end_ns)
            repeat_marker_points: list[tuple[float, str]] = []
            for marker in switch_markers:
                if marker.t_ns < rep_start_ns or marker.t_ns > rep_end_ns:
                    continue
                marker_repeat, marker_mode = parse_marker_repeat_mode(marker.message)
                if marker_repeat != repeat_name:
                    continue
                label = marker.marker if not marker_mode else f"{marker.marker}:{marker_mode}"
                repeat_marker_points.append(((marker.t_ns - rep_start_ns) / 1_000_000_000.0, label))
            overview_name = f"{repeat_name}_switch_pfc_ecn_overview.png"
            plot_switch_overview(
                repeat_bucketed,
                repeat_marker_points,
                output_dir / overview_name,
                title=f"{repeat_name} Switch PFC / ECN Overview",
            )
            html_images.append((f"{repeat_name} Switch PFC / ECN Overview", overview_name))

            mode_pfc_series: dict[str, list[tuple[float, int]]] = {}
            mode_ecn_series: dict[str, list[tuple[float, int]]] = {}
            for mode_name, records in records_by_mode.items():
                bounds = measured_window_bounds(records)
                if bounds is None:
                    continue
                mode_start_ns, mode_end_ns = bounds
                mode_bucketed = bucketize_switch_events(switch_events, 1.0, start_ns=mode_start_ns, end_ns=mode_end_ns)
                pfc_map: dict[float, int] = {}
                for key in ("rackA:pause", "rackB:pause", "spine:pause"):
                    for x, value in mode_bucketed.get(key, []):
                        pfc_map[x] = pfc_map.get(x, 0) + value
                mode_pfc_series[mode_name] = sorted(pfc_map.items())
                mode_ecn_series[mode_name] = mode_bucketed.get("spine:ecn", [])

            pfc_overlay_name = f"{repeat_name}_mode_pfc_overlay.png"
            plot_mode_overlay(
                mode_pfc_series,
                output_dir / pfc_overlay_name,
                title=f"{repeat_name} Mode Overlay: Switch PFC Delta",
                ylabel="Switch PFC delta / 1s bucket",
            )
            html_images.append((f"{repeat_name} Mode Overlay: Switch PFC Delta", pfc_overlay_name))

            ecn_overlay_name = f"{repeat_name}_mode_spine_ecn_overlay.png"
            plot_mode_overlay(
                mode_ecn_series,
                output_dir / ecn_overlay_name,
                title=f"{repeat_name} Mode Overlay: Spine ECN Delta",
                ylabel="Spine ECN delta / 1s bucket",
            )
            html_images.append((f"{repeat_name} Mode Overlay: Spine ECN Delta", ecn_overlay_name))

    (output_dir / "phase5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_dir, html_images)


if __name__ == "__main__":
    main()
