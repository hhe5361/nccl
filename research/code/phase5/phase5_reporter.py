#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SWITCH_SHARED_ROOT_DEFAULT = Path("/mnt/nfs_share/cts_experiments/switch_log")
SWITCH_SHARED_ROOT_FALLBACK = Path("/mnt/nfs/cts_experiments/switch_log")


@dataclass
class StepRecord:
    step: int
    warmup: bool
    ts_start_unix_ns: int
    ts_end_unix_ns: int
    ts_mid_unix_ns: int
    step_ms_max: float
    samples_per_sec: float


@dataclass
class NetDoneEvent:
    t_ns: int
    post_to_net_done_us: float
    progress_delta_us: float
    progress_calls_since_post: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render phase5 congestion validation report")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-ms", type=float, default=100.0, help="Bucket size for time overlays")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for _, idx in indexed[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(rank_values(xs), rank_values(ys))


def build_palette(keys: list[str]) -> dict[str, str]:
    colors = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    return {key: matplotlib.colors.to_hex(colors[idx % len(colors)]) for idx, key in enumerate(keys)}


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
                row, next_index = decoder.raw_decode(line, index)
                if isinstance(row, dict):
                    rows.append(row)
                index = next_index
    return rows


def extract_ts_ns(row: dict) -> Optional[int]:
    for key in ("ts_mid_unix_ns", "ts_unix_ns", "ts_start_unix_ns", "ts_end_unix_ns", "timestamp_ns", "tNs"):
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


def parse_phase5_line(line: str) -> dict[str, str] | None:
    if "PHASE5 event=" not in line:
        return None
    row: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[key] = value
    return row if "event" in row else None


def load_step_records(mode_dir: Path) -> list[StepRecord]:
    candidates = sorted(mode_dir.glob("*_step_metrics.jsonl"))
    if not candidates:
        return []
    records: list[StepRecord] = []
    for row in read_jsonl(candidates[0]):
        try:
            records.append(
                StepRecord(
                    step=int(row.get("step", 0)),
                    warmup=bool(row.get("warmup", False)),
                    ts_start_unix_ns=int(row.get("ts_start_unix_ns", 0)),
                    ts_end_unix_ns=int(row.get("ts_end_unix_ns", 0)),
                    ts_mid_unix_ns=int(row.get("ts_mid_unix_ns", 0)),
                    step_ms_max=float(row.get("step_ms_max", 0.0)),
                    samples_per_sec=float(row.get("samples_per_sec", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return records


def load_mode_events(mode_dir: Path) -> list[NetDoneEvent]:
    events: list[NetDoneEvent] = []
    for log_path in sorted(mode_dir.glob("worker*/nccl.*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_phase5_line(line)
            if row is None:
                continue
            event = row.get("event", "")
            if event == "PROXY_RECV_NET_DONE":
                try:
                    t_ns = int(row.get("tNs", "0"))
                    delay_ns = int(row.get("postToNetDoneNs", "0"))
                    calls = int(row.get("progressCallsSincePost", "0"))
                except ValueError:
                    continue
                if t_ns > 0 and delay_ns >= 0:
                    events.append(
                        NetDoneEvent(
                            t_ns=t_ns,
                            post_to_net_done_us=delay_ns / 1000.0,
                            progress_delta_us=0.0,
                            progress_calls_since_post=float(calls),
                        )
                    )
            elif event == "RECV_PROXY_PROGRESS":
                continue
    events.sort(key=lambda item: item.t_ns)
    return events


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    if not ordered:
        return [], []
    ys = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    return ordered, ys


def plot_ecdf(mode_values: dict[str, list[float]], title: str, xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = build_palette(list(mode_values.keys()))
    for mode, values in mode_values.items():
        xs, ys = ecdf(values)
        if xs:
            ax.plot(xs, ys, label=mode, linewidth=2, color=palette[mode])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    seen: set[str] = set()
    for candidate in candidates:
        raw = candidate.as_posix()
        expanded = [candidate]
        if raw.startswith("/mnt/nfs/"):
            expanded.append(Path("/mnt/nfs_share/" + raw[len("/mnt/nfs/"):]))
        elif raw.startswith("/mnt/nfs_share/"):
            expanded.append(Path("/mnt/nfs/" + raw[len("/mnt/nfs_share/"):]))
        for path in expanded:
            key = path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return path
    return None


def load_markers(log_dir: Path) -> list[dict]:
    return read_jsonl(log_dir / "markers.jsonl")


def load_cumulative_series(path: Path, keys: tuple[str, ...]) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for row in read_jsonl(path):
        ts_ns = extract_ts_ns(row)
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


def interpolate_value(samples: list[tuple[int, float]], ts_ns: int) -> Optional[float]:
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


def window_delta(samples: list[tuple[int, float]], start_ns: int, end_ns: int) -> float:
    start_value = interpolate_value(samples, start_ns)
    end_value = interpolate_value(samples, end_ns)
    if start_value is None or end_value is None:
        return 0.0
    return max(0.0, end_value - start_value)


def combine_cumulative_series(series_list: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    timestamps = sorted({ts for series in series_list for ts, _ in series})
    combined: list[tuple[int, float]] = []
    for ts_ns in timestamps:
        total = 0.0
        found = False
        for series in series_list:
            value = interpolate_value(series, ts_ns)
            if value is None:
                continue
            total += value
            found = True
        if found:
            combined.append((ts_ns, total))
    return combined


def bucketize_counter_delta(
    samples: list[tuple[int, float]],
    start_ns: int,
    end_ns: int,
    bucket_ms: float,
) -> list[tuple[float, float]]:
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    points: list[tuple[float, float]] = []
    current = start_ns
    while current < end_ns:
        nxt = min(current + bucket_ns, end_ns)
        delta = window_delta(samples, current, nxt)
        rel_ms = (nxt - start_ns) / 1_000_000.0
        points.append((rel_ms, delta))
        current = nxt
    return points


def bucketize_event_percentile(
    events: list[NetDoneEvent],
    start_ns: int,
    end_ns: int,
    bucket_ms: float,
    q: float,
) -> list[tuple[float, float]]:
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    buckets: list[list[float]] = []
    nbuckets = max(1, int(math.ceil((end_ns - start_ns) / bucket_ns)))
    for _ in range(nbuckets):
        buckets.append([])
    for event in events:
        if event.t_ns < start_ns or event.t_ns > end_ns:
            continue
        idx = min(int((event.t_ns - start_ns) // bucket_ns), nbuckets - 1)
        buckets[idx].append(event.post_to_net_done_us / 1000.0)
    points: list[tuple[float, float]] = []
    for idx, values in enumerate(buckets):
        rel_ms = ((idx + 1) * bucket_ns) / 1_000_000.0
        points.append((rel_ms, percentile(values, q)))
    return points


def compute_step_et_stats(steps: list[StepRecord], events: list[NetDoneEvent]) -> list[dict]:
    rows: list[dict] = []
    event_index = 0
    ordered_events = sorted(events, key=lambda item: item.t_ns)
    for step in steps:
        values: list[float] = []
        while event_index < len(ordered_events) and ordered_events[event_index].t_ns < step.ts_start_unix_ns:
            event_index += 1
        probe = event_index
        while probe < len(ordered_events) and ordered_events[probe].t_ns <= step.ts_end_unix_ns:
            values.append(ordered_events[probe].post_to_net_done_us / 1000.0)
            probe += 1
        rows.append(
            {
                "step": step.step,
                "et_mean_ms": mean(values),
                "et_p95_ms": percentile(values, 0.95),
                "et_p99_ms": percentile(values, 0.99),
                "event_count": len(values),
            }
        )
    return rows


def build_switch_bundle(run_root: Path) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(run_root)
    if log_dir is None:
        return None
    rack_a = load_cumulative_series(log_dir / "rackA_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    rack_b = load_cumulative_series(log_dir / "rackB_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    spine_pfc = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_pause_packets_total", "tx_pause_packets_total"))
    spine_ecn = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_ecn_marked_packets_total",))
    return {
        "log_dir": log_dir,
        "markers": load_markers(log_dir),
        "rackA_pfc": rack_a,
        "rackB_pfc": rack_b,
        "spine_pfc": spine_pfc,
        "spine_ecn": spine_ecn,
    }


def compute_step_switch_stats(steps: list[StepRecord], switch_bundle: Optional[dict]) -> list[dict]:
    rows: list[dict] = []
    if switch_bundle is None:
        return rows
    for step in steps:
        pfc_delta = (
            window_delta(switch_bundle["rackA_pfc"], step.ts_start_unix_ns, step.ts_end_unix_ns)
            + window_delta(switch_bundle["rackB_pfc"], step.ts_start_unix_ns, step.ts_end_unix_ns)
            + window_delta(switch_bundle["spine_pfc"], step.ts_start_unix_ns, step.ts_end_unix_ns)
        )
        ecn_delta = window_delta(switch_bundle["spine_ecn"], step.ts_start_unix_ns, step.ts_end_unix_ns)
        rows.append(
            {
                "step": step.step,
                "switch_pfc_delta": pfc_delta,
                "spine_ecn_delta": ecn_delta,
            }
        )
    return rows


def plot_dual_axis_stepwise(
    xs: list[int],
    left_values: list[float],
    right_values: list[float],
    *,
    left_label: str,
    right_label: str,
    title: str,
    output_path: Path,
    left_color: str = "#1f77b4",
    right_color: str = "#d62728",
) -> None:
    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()
    ax_left.plot(xs, left_values, color=left_color, linewidth=2.0, marker="o", markersize=3, label=left_label)
    ax_right.plot(xs, right_values, color=right_color, linewidth=1.8, marker="s", markersize=3, label=right_label)
    ax_left.set_title(title)
    ax_left.set_xlabel("Measured Step")
    ax_left.set_ylabel(left_label, color=left_color)
    ax_right.set_ylabel(right_label, color=right_color)
    ax_left.grid(True, alpha=0.25)
    lines = ax_left.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mode_overlay(
    series_by_mode: dict[str, list[tuple[int, float]]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    modes = [mode for mode, values in series_by_mode.items() if values]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        xs = [x for x, _ in series_by_mode[mode]]
        ys = [y for _, y in series_by_mode[mode]]
        ax.plot(xs, ys, linewidth=2.0, marker="o", markersize=3, color=palette[mode], label=mode)
    ax.set_title(title)
    ax.set_xlabel("Measured Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter(
    points_by_mode: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    modes = [mode for mode, values in points_by_mode.items() if values]
    if not modes:
        return
    palette = build_palette(modes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode in modes:
        xs = [x for x, _ in points_by_mode[mode]]
        ys = [y for _, y in points_by_mode[mode]]
        ax.scatter(xs, ys, s=24, alpha=0.85, color=palette[mode], label=mode)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_switch_overview(switch_bundle: Optional[dict], output_path: Path) -> None:
    if switch_bundle is None:
        return
    series_map = {
        "spine PFC delta/s": switch_bundle["spine_pfc"],
        "spine ECN delta/s": switch_bundle["spine_ecn"],
        "rackA PFC delta/s": switch_bundle["rackA_pfc"],
        "rackB PFC delta/s": switch_bundle["rackB_pfc"],
    }
    if not any(series_map.values()):
        return
    all_ts = [ts for samples in series_map.values() for ts, _ in samples]
    if not all_ts:
        return
    base_ts = min(all_ts)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    for ax, (label, samples), color in zip(axes, series_map.items(), colors):
        if len(samples) < 2:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
            elapsed = (cur_ts - prev_ts) / 1_000_000_000.0
            if elapsed <= 0.0:
                continue
            xs.append((cur_ts - base_ts) / 1_000_000_000.0)
            ys.append(max(0.0, cur_total - prev_total) / elapsed)
        ax.plot(xs, ys, linewidth=1.8, color=color)
        ax.fill_between(xs, ys, color=color, alpha=0.16)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ymax = max(ys) if ys else 1.0
        for marker in switch_bundle["markers"]:
            ts_ns = marker.get("ts_unix_ns")
            name = marker.get("marker")
            if not isinstance(ts_ns, int) or not name:
                continue
            marker_x = (ts_ns - base_ts) / 1_000_000_000.0
            ax.axvline(marker_x, color="#666666", linestyle=":", linewidth=1.0, alpha=0.5)
            ax.text(marker_x, ymax, str(name), rotation=90, va="top", ha="right", fontsize=8, color="#666666")
    axes[-1].set_xlabel("Elapsed Time (s)")
    fig.suptitle("Switch PFC / ECN Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repeat_dirs = sorted(p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    if not repeat_dirs:
        repeat_dirs = [input_root]

    switch_bundle = build_switch_bundle(input_root)
    plot_switch_overview(switch_bundle, output_dir / "switch_pfc_ecn_overview.png")

    html_sections: list[str] = []
    global_summary: dict[str, dict[str, float]] = {}

    for repeat_dir in repeat_dirs:
        mode_dirs = sorted(p for p in repeat_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        distributions: dict[str, dict[str, list[float]]] = {}
        step_overlay_et: dict[str, list[tuple[int, float]]] = {}
        step_overlay_pfc_cum: dict[str, list[tuple[int, float]]] = {}
        step_overlay_ecn_cum: dict[str, list[tuple[int, float]]] = {}
        pfc_scatter: dict[str, list[tuple[float, float]]] = {}
        ecn_scatter: dict[str, list[tuple[float, float]]] = {}
        repeat_summary_rows: list[dict[str, float | str]] = []

        post_to_net_done_plot = output_dir / f"{repeat_dir.name}_post_to_net_done_ecdf.png"

        for mode_dir in mode_dirs:
            mode = mode_dir.name
            steps_all = load_step_records(mode_dir)
            steps = [record for record in steps_all if not record.warmup]
            events = load_mode_events(mode_dir)
            distributions[mode] = {
                "post_to_net_done_us": [event.post_to_net_done_us for event in events],
                "progress_calls_since_post": [event.progress_calls_since_post for event in events],
            }
            step_et = compute_step_et_stats(steps, events)
            step_switch = compute_step_switch_stats(steps, switch_bundle)
            switch_by_step = {int(row["step"]): row for row in step_switch}

            xs = [int(row["step"]) for row in step_et]
            et_p99 = [float(row["et_p99_ms"]) for row in step_et]
            pfc_delta = [float(switch_by_step.get(step, {}).get("switch_pfc_delta", 0.0)) for step in xs]
            ecn_delta = [float(switch_by_step.get(step, {}).get("spine_ecn_delta", 0.0)) for step in xs]

            pfc_corr = pearson(et_p99, pfc_delta)
            ecn_corr = pearson(et_p99, ecn_delta)
            pfc_spearman = spearman(et_p99, pfc_delta)
            ecn_spearman = spearman(et_p99, ecn_delta)

            repeat_summary_rows.append(
                {
                    "mode": mode,
                    "et_mean_ms": mean([float(row["et_mean_ms"]) for row in step_et]),
                    "et_p95_ms": percentile([float(row["et_p95_ms"]) for row in step_et], 0.95),
                    "et_p99_ms": percentile(et_p99, 0.95),
                    "switch_pfc_total_delta": sum(pfc_delta),
                    "spine_ecn_total_delta": sum(ecn_delta),
                    "pearson_et_p99_vs_pfc": pfc_corr,
                    "spearman_et_p99_vs_pfc": pfc_spearman,
                    "pearson_et_p99_vs_ecn": ecn_corr,
                    "spearman_et_p99_vs_ecn": ecn_spearman,
                }
            )

            step_overlay_et[mode] = list(zip(xs, et_p99))
            cumulative_pfc = 0.0
            cumulative_ecn = 0.0
            pfc_points: list[tuple[int, float]] = []
            ecn_points: list[tuple[int, float]] = []
            for step, pfc, ecn in zip(xs, pfc_delta, ecn_delta):
                cumulative_pfc += pfc
                cumulative_ecn += ecn
                pfc_points.append((step, cumulative_pfc))
                ecn_points.append((step, cumulative_ecn))
            step_overlay_pfc_cum[mode] = pfc_points
            step_overlay_ecn_cum[mode] = ecn_points
            pfc_scatter[mode] = list(zip(pfc_delta, et_p99))
            ecn_scatter[mode] = list(zip(ecn_delta, et_p99))

            plot_dual_axis_stepwise(
                xs,
                et_p99,
                pfc_delta,
                left_label="E(t) p99 (ms)",
                right_label="Switch PFC Delta",
                title=f"{repeat_dir.name} {mode} Stepwise E(t) p99 vs Switch PFC Delta",
                output_path=output_dir / f"{repeat_dir.name}_{mode}_et_p99_vs_pfc.png",
            )
            plot_dual_axis_stepwise(
                xs,
                et_p99,
                ecn_delta,
                left_label="E(t) p99 (ms)",
                right_label="Spine ECN Delta",
                title=f"{repeat_dir.name} {mode} Stepwise E(t) p99 vs Spine ECN Delta",
                output_path=output_dir / f"{repeat_dir.name}_{mode}_et_p99_vs_ecn.png",
                right_color="#2ca02c",
            )

            if steps:
                mode_start_ns = min(step.ts_start_unix_ns for step in steps)
                mode_end_ns = max(step.ts_end_unix_ns for step in steps)
                bucket_et = bucketize_event_percentile(events, mode_start_ns, mode_end_ns, args.bucket_ms, 0.99)
                total_pfc_samples = combine_cumulative_series(
                    [switch_bundle["rackA_pfc"], switch_bundle["rackB_pfc"], switch_bundle["spine_pfc"]]
                ) if switch_bundle else []
                spine_ecn_samples = switch_bundle["spine_ecn"] if switch_bundle else []
                bucket_pfc = bucketize_counter_delta(total_pfc_samples, mode_start_ns, mode_end_ns, args.bucket_ms)
                bucket_ecn = bucketize_counter_delta(spine_ecn_samples, mode_start_ns, mode_end_ns, args.bucket_ms)
                if bucket_et and bucket_pfc:
                    plot_dual_axis_stepwise(
                        [int(x) for x, _ in bucket_et],
                        [y for _, y in bucket_et],
                        [y for _, y in bucket_pfc],
                        left_label="Bucketed E(t) p99 (ms)",
                        right_label="Bucketed Switch PFC Delta",
                        title=f"{repeat_dir.name} {mode} Bucketed E(t) p99 vs Switch PFC Delta",
                        output_path=output_dir / f"{repeat_dir.name}_{mode}_bucket_et_p99_vs_pfc.png",
                    )
                if bucket_et and bucket_ecn:
                    plot_dual_axis_stepwise(
                        [int(x) for x, _ in bucket_et],
                        [y for _, y in bucket_et],
                        [y for _, y in bucket_ecn],
                        left_label="Bucketed E(t) p99 (ms)",
                        right_label="Bucketed Spine ECN Delta",
                        title=f"{repeat_dir.name} {mode} Bucketed E(t) p99 vs Spine ECN Delta",
                        output_path=output_dir / f"{repeat_dir.name}_{mode}_bucket_et_p99_vs_ecn.png",
                        right_color="#2ca02c",
                    )

        plot_ecdf(
            {mode: values["post_to_net_done_us"] for mode, values in distributions.items()},
            title=f"{repeat_dir.name} POST->NET_DONE Delay",
            xlabel="Delay (us)",
            output_path=post_to_net_done_plot,
        )
        plot_mode_overlay(
            step_overlay_et,
            title=f"{repeat_dir.name} Stepwise E(t) p99 Across Modes",
            ylabel="E(t) p99 (ms)",
            output_path=output_dir / f"{repeat_dir.name}_mode_et_p99_overlay.png",
        )
        plot_mode_overlay(
            step_overlay_pfc_cum,
            title=f"{repeat_dir.name} Stepwise Total Switch PFC Cumulative Increase",
            ylabel="Cumulative PFC Delta",
            output_path=output_dir / f"{repeat_dir.name}_mode_pfc_overlay.png",
        )
        plot_mode_overlay(
            step_overlay_ecn_cum,
            title=f"{repeat_dir.name} Stepwise Spine ECN Cumulative Increase",
            ylabel="Cumulative ECN Delta",
            output_path=output_dir / f"{repeat_dir.name}_mode_ecn_overlay.png",
        )
        plot_scatter(
            pfc_scatter,
            title=f"{repeat_dir.name} E(t) p99 vs Switch PFC Delta",
            xlabel="Switch PFC Delta",
            ylabel="E(t) p99 (ms)",
            output_path=output_dir / f"{repeat_dir.name}_et_p99_vs_pfc_scatter.png",
        )
        plot_scatter(
            ecn_scatter,
            title=f"{repeat_dir.name} E(t) p99 vs Spine ECN Delta",
            xlabel="Spine ECN Delta",
            ylabel="E(t) p99 (ms)",
            output_path=output_dir / f"{repeat_dir.name}_et_p99_vs_ecn_scatter.png",
        )

        (output_dir / f"{repeat_dir.name}_phase5_report_summary.json").write_text(
            json.dumps(repeat_summary_rows, indent=2),
            encoding="utf-8",
        )

        for row in repeat_summary_rows:
            global_summary[f"{repeat_dir.name}:{row['mode']}"] = {
                key: float(value)
                for key, value in row.items()
                if key != "mode"
            }

        html_sections.append(
            f"""
            <section>
              <h2>{repeat_dir.name}</h2>
              <img src="{post_to_net_done_plot.name}" style="max-width: 100%;"><br>
              <img src="{repeat_dir.name}_mode_et_p99_overlay.png" style="max-width: 100%;"><br>
              <img src="{repeat_dir.name}_mode_pfc_overlay.png" style="max-width: 100%;"><br>
              <img src="{repeat_dir.name}_mode_ecn_overlay.png" style="max-width: 100%;"><br>
              <img src="{repeat_dir.name}_et_p99_vs_pfc_scatter.png" style="max-width: 48%;">
              <img src="{repeat_dir.name}_et_p99_vs_ecn_scatter.png" style="max-width: 48%;">
            </section>
            """
        )

    (output_dir / "phase5_summary.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase5 Report</title></head>
<body>
  <h1>Phase5 Congestion Validation Report</h1>
  <p>Primary internal metric: E(t)=POST-&gt;NET_DONE delay. External congestion truth: switch PFC delta and spine ECN delta from switch_congestion_logger_v2.</p>
  <img src="switch_pfc_ecn_overview.png" style="max-width: 100%;"><br>
  {''.join(html_sections)}
</body></html>"""
    (output_dir / "phase5_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
