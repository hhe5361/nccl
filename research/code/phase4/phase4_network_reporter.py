#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt

import phase4_reporter as p4


@dataclass(frozen=True)
class ModeWindow:
    repeat_name: str
    mode_name: str
    start_ns: int
    end_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-repeat phase4 network plots that compare internal NCCL "
            "network metrics against switch-side congestion metrics."
        )
    )
    parser.add_argument("--input", required=True, help="Phase4 run root")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--switch-bucket-sec",
        type=float,
        default=1.0,
        help="Bucket width for switch-side deltas in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--net-bucket-ms",
        type=float,
        default=50.0,
        help="Bucket width for internal NET_DONE throughput in milliseconds. Default: 50.0",
    )
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
                row, next_index = decoder.raw_decode(line, index)
                if isinstance(row, dict):
                    rows.append(row)
                index = next_index
    return rows


def load_counter_records(log_dir: Path) -> list[dict]:
    records: list[dict] = []
    for name in (
        "spine_pfc_ecn.jsonl",
        "rackA_pfc_statistics.jsonl",
        "rackB_pfc_statistics.jsonl",
    ):
        records.extend(read_jsonl(log_dir / name))
    return records


def build_switch_delta_events(records: Iterable[dict]) -> dict[str, list[tuple[int, int]]]:
    by_key: DefaultDict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        kind = record.get("kind")
        switch = record.get("switch")
        port = record.get("port")
        ts_ns = record.get("ts_mid_unix_ns")
        if kind is None or switch is None or port is None or ts_ns is None:
            continue
        metric_values: list[tuple[str, Optional[int]]] = []
        if kind == "fsos_pfc_statistics":
            rx = record.get("rx_pause")
            tx = record.get("tx_pause")
            metric_values.append(("pause", None if rx is None or tx is None else int(rx) + int(tx)))
        elif kind == "onyx_pfc_ecn":
            rx_pause = record.get("rx_pause_packets")
            tx_pause = record.get("tx_pause_packets")
            ecn = record.get("rx_ecn_marked_packets")
            metric_values.append(("pause", None if rx_pause is None or tx_pause is None else int(rx_pause) + int(tx_pause)))
            metric_values.append(("ecn", None if ecn is None else int(ecn)))
        else:
            continue
        for metric, value in metric_values:
            if value is None:
                continue
            by_key[(str(switch), str(port), metric)].append((int(ts_ns), value))

    events: DefaultDict[str, list[tuple[int, int]]] = defaultdict(list)
    for (switch, _port, metric), samples in by_key.items():
        samples.sort(key=lambda item: item[0])
        previous_value: Optional[int] = None
        for ts_ns, current_value in samples:
            delta = 0 if previous_value is None else max(0, current_value - previous_value)
            events[f"{switch}:{metric}"].append((ts_ns, delta))
            previous_value = current_value
    for rows in events.values():
        rows.sort(key=lambda item: item[0])
    return dict(events)


def bucketize_relative_events(
    events: Sequence[tuple[int, float]],
    start_ns: int,
    end_ns: int,
    bucket_sec: float,
) -> list[tuple[float, float]]:
    if end_ns <= start_ns or bucket_sec <= 0:
        return []
    buckets: DefaultDict[float, float] = defaultdict(float)
    for ts_ns, value in events:
        if ts_ns < start_ns or ts_ns > end_ns:
            continue
        rel_sec = (ts_ns - start_ns) / 1_000_000_000.0
        bucket = math.floor(rel_sec / bucket_sec) * bucket_sec
        buckets[bucket] += float(value)
    return sorted(buckets.items())


def build_repeat_windows(run_root: Path) -> tuple[dict[str, dict[str, list[p4.StepRecord]]], dict[str, list[ModeWindow]]]:
    records_by_repeat_mode: dict[str, dict[str, list[p4.StepRecord]]] = {}
    windows_by_repeat: dict[str, list[ModeWindow]] = {}
    for repeat_dir in p4.discover_repeat_dirs(run_root):
        repeat_name = repeat_dir.name if repeat_dir != run_root else "repeat_01"
        records_by_repeat_mode[repeat_name] = {}
        windows_by_repeat[repeat_name] = []
        for mode_dir in p4.iter_mode_dirs(repeat_dir):
            records = p4.load_step_metrics(repeat_name, mode_dir)
            if not records:
                continue
            records_by_repeat_mode[repeat_name][mode_dir.name] = records
            start_ns = min(record.ts_start_unix_ns for record in records)
            end_ns = max(record.ts_end_unix_ns for record in records)
            windows_by_repeat[repeat_name].append(
                ModeWindow(
                    repeat_name=repeat_name,
                    mode_name=mode_dir.name,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            )
        windows_by_repeat[repeat_name].sort(key=lambda item: item.start_ns)
    return records_by_repeat_mode, windows_by_repeat


def build_internal_net_series(
    run_root: Path,
    repeat_name: str,
    mode_name: str,
    records: list[p4.StepRecord],
    bucket_ms: float,
) -> list[tuple[float, float]]:
    repeat_dir = run_root / repeat_name if (run_root / repeat_name).exists() else run_root
    mode_dir = repeat_dir / mode_name
    phase0_events = p4.load_phase0_events(mode_dir)
    netdone_events = [event for event in phase0_events if event.event == "PROXY_RECV_NET_DONE" and event.size > 0]
    if not netdone_events:
        return []
    base_ns = min(event.t_ns for event in netdone_events)
    max_t_ms = max((event.t_ns - base_ns) / 1e6 for event in netdone_events)
    nbins = max(1, int(max_t_ms / bucket_ms) + 1)
    bins = [0] * nbins
    for event in netdone_events:
        rel_ms = (event.t_ns - base_ns) / 1e6
        idx = min(int(rel_ms / bucket_ms), nbins - 1)
        bins[idx] += event.size
    points: list[tuple[float, float]] = []
    for idx, total_bytes in enumerate(bins):
        rel_sec = ((idx + 0.5) * bucket_ms) / 1000.0
        gbps = (total_bytes * 8.0) / (bucket_ms * 1_000_000.0)
        points.append((rel_sec, gbps))
    return points


def plot_switch_overview(
    repeat_name: str,
    windows: list[ModeWindow],
    switch_events: dict[str, list[tuple[int, int]]],
    bucket_sec: float,
    output_path: Path,
) -> None:
    if not windows:
        return
    repeat_start_ns = min(window.start_ns for window in windows)
    repeat_end_ns = max(window.end_ns for window in windows)
    layout = [
        ("spine:pause", "spine PFC delta/bucket", "#1f77b4"),
        ("spine:ecn", "spine ECN delta/bucket", "#ff7f0e"),
        ("rackA:pause", "rackA PFC delta/bucket", "#d62728"),
        ("rackB:pause", "rackB PFC delta/bucket", "#2ca02c"),
    ]
    active_layout = [item for item in layout if item[0] in switch_events]
    if not active_layout:
        return

    fig, axes = plt.subplots(len(active_layout), 1, figsize=(15, 10), sharex=True)
    if len(active_layout) == 1:
        axes = [axes]

    for ax, (series_name, ylabel, color) in zip(axes, active_layout):
        points = bucketize_relative_events(switch_events[series_name], repeat_start_ns, repeat_end_ns, bucket_sec)
        if not points:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, linewidth=1.6, color=color)
        ax.fill_between(xs, ys, step="pre", alpha=0.18, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ymax = max(ys) if ys else 0.0
        label_y = ymax if ymax > 0 else 1.0
        for window in windows:
            start_sec = (window.start_ns - repeat_start_ns) / 1_000_000_000.0
            end_sec = (window.end_ns - repeat_start_ns) / 1_000_000_000.0
            ax.axvline(start_sec, color="#555555", linestyle=":", alpha=0.45, linewidth=1.0)
            ax.axvline(end_sec, color="#999999", linestyle=":", alpha=0.25, linewidth=1.0)
            ax.text(start_sec, label_y, window.mode_name, rotation=90, va="top", ha="right", fontsize=8, color="#555555")

    x_max = (repeat_end_ns - repeat_start_ns) / 1_000_000_000.0
    axes[-1].set_xlim(0, max(x_max, bucket_sec))
    axes[-1].set_xlabel("Elapsed time in repeat (seconds)")
    fig.suptitle(f"{repeat_name} Switch PFC / ECN Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_internal_network_overlay(
    repeat_name: str,
    internal_by_mode: dict[str, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    modes = [mode for mode, points in internal_by_mode.items() if points]
    if not modes:
        return
    palette = p4.build_palette(modes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
        xs = [x for x, _ in internal_by_mode[mode]]
        ys = [y for _, y in internal_by_mode[mode]]
        ax.plot(xs, ys, linewidth=1.8, color=palette[mode], label=mode)
    ax.set_title(f"{repeat_name} Internal Network Metric Overlay")
    ax.set_xlabel("Time Since Measured-Phase Start (s)")
    ax.set_ylabel("NET_DONE Throughput (Gbps)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mode_overlay_figure(
    repeat_name: str,
    windows: list[ModeWindow],
    left_series_by_mode: dict[str, list[tuple[float, float]]],
    right_series_by_mode: dict[str, list[tuple[float, float]]],
    output_path: Path,
    *,
    right_label: str,
    title_suffix: str,
) -> None:
    modes = [window.mode_name for window in windows if left_series_by_mode.get(window.mode_name) or right_series_by_mode.get(window.mode_name)]
    if not modes:
        return
    palette = p4.build_palette(modes)
    fig, axes = plt.subplots(len(modes), 1, figsize=(13, 3.2 * len(modes)), sharex=False)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        left_points = left_series_by_mode.get(mode, [])
        right_points = right_series_by_mode.get(mode, [])
        if left_points:
            ax.plot(
                [x for x, _ in left_points],
                [y for _, y in left_points],
                color=palette[mode],
                linewidth=1.8,
                label=f"{mode} internal",
            )
        ax.set_ylabel("Gbps", color=palette[mode])
        ax.tick_params(axis="y", labelcolor=palette[mode])
        ax.grid(True, alpha=0.25)
        ax.set_title(mode)
        twin = ax.twinx()
        if right_points:
            twin.plot(
                [x for x, _ in right_points],
                [y for _, y in right_points],
                color="#444444",
                linewidth=1.6,
                linestyle="--",
                label=f"{mode} {right_label}",
            )
        twin.set_ylabel(right_label, color="#444444")
        twin.tick_params(axis="y", labelcolor="#444444")
        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        if handles or twin_handles:
            ax.legend(handles + twin_handles, labels + twin_labels, loc="upper right")
        ax.set_xlabel("Time Since Measured-Phase Start (s)")

    fig.suptitle(f"{repeat_name} Internal Metric vs {title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_html(output_dir: Path, repeat_sections: list[tuple[str, list[tuple[str, str]]]]) -> None:
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Phase4 Network Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;border:1px solid #ddd;margin-bottom:16px;} code{background:#f4f4f4;padding:2px 4px;}</style>",
        "</head><body>",
        "<h1>Phase4 Network Report</h1>",
        "<p>Internal network metric uses measured-phase <code>PROXY_RECV_NET_DONE size</code> aggregated into throughput buckets. Switch metrics use switch_congestion_logger_v2 raw PFC / ECN counters bucketed into relative-mode time.</p>",
    ]
    for section_title, images in repeat_sections:
        html.append(f"<h2>{section_title}</h2>")
        for title, filename in images:
            html.append(f"<h3>{title}</h3>")
            html.append(f"<img src='{filename}' alt='{title}'>")
    html.extend(["</body></html>"])
    (output_dir / "phase4_network_report.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    switch_log_dir = p4.resolve_switch_log_dir(run_root)
    if switch_log_dir is None:
        raise FileNotFoundError(f"switch log directory not found for run root: {run_root}")

    records_by_repeat_mode, windows_by_repeat = build_repeat_windows(run_root)
    switch_events = build_switch_delta_events(load_counter_records(switch_log_dir))
    repeat_sections: list[tuple[str, list[tuple[str, str]]]] = []
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for repeat_name, windows in windows_by_repeat.items():
        if not windows:
            continue
        records_by_mode = records_by_repeat_mode.get(repeat_name, {})
        internal_by_mode: dict[str, list[tuple[float, float]]] = {}
        pfc_by_mode: dict[str, list[tuple[float, float]]] = {}
        ecn_by_mode: dict[str, list[tuple[float, float]]] = {}
        summary[repeat_name] = {}
        images: list[tuple[str, str]] = []

        for window in windows:
            mode = window.mode_name
            records = records_by_mode.get(mode, [])
            if not records:
                continue
            internal_points = build_internal_net_series(run_root, repeat_name, mode, records, args.net_bucket_ms)
            pfc_points = bucketize_relative_events(
                switch_events.get("rackA:pause", []) + switch_events.get("rackB:pause", []) + switch_events.get("spine:pause", []),
                window.start_ns,
                window.end_ns,
                args.switch_bucket_sec,
            )
            ecn_points = bucketize_relative_events(
                switch_events.get("spine:ecn", []),
                window.start_ns,
                window.end_ns,
                args.switch_bucket_sec,
            )
            internal_by_mode[mode] = internal_points
            pfc_by_mode[mode] = pfc_points
            ecn_by_mode[mode] = ecn_points
            summary[repeat_name][mode] = {
                "internal_peak_gbps": max((y for _, y in internal_points), default=0.0),
                "pfc_total_delta": sum(y for _, y in pfc_points),
                "pfc_peak_bucket_delta": max((y for _, y in pfc_points), default=0.0),
                "ecn_total_delta": sum(y for _, y in ecn_points),
                "ecn_peak_bucket_delta": max((y for _, y in ecn_points), default=0.0),
            }

        switch_png = f"{repeat_name}_switch_pfc_ecn_overview.png"
        internal_png = f"{repeat_name}_internal_network_metric_overlay.png"
        pfc_overlay_png = f"{repeat_name}_internal_vs_switch_pfc.png"
        ecn_overlay_png = f"{repeat_name}_internal_vs_spine_ecn.png"

        plot_switch_overview(repeat_name, windows, switch_events, args.switch_bucket_sec, output_dir / switch_png)
        plot_internal_network_overlay(repeat_name, internal_by_mode, output_dir / internal_png)
        plot_mode_overlay_figure(
            repeat_name,
            windows,
            internal_by_mode,
            pfc_by_mode,
            output_dir / pfc_overlay_png,
            right_label="PFC delta / bucket",
            title_suffix="Switch PFC",
        )
        plot_mode_overlay_figure(
            repeat_name,
            windows,
            internal_by_mode,
            ecn_by_mode,
            output_dir / ecn_overlay_png,
            right_label="ECN delta / bucket",
            title_suffix="Spine ECN",
        )

        images.extend(
            [
                ("Switch PFC / ECN Overview", switch_png),
                ("Internal Network Metric Overlay", internal_png),
                ("Internal Metric vs Switch PFC", pfc_overlay_png),
                ("Internal Metric vs Spine ECN", ecn_overlay_png),
            ]
        )
        repeat_sections.append((repeat_name, images))

    (output_dir / "phase4_network_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_html(output_dir, repeat_sections)


if __name__ == "__main__":
    main()
