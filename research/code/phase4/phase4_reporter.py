#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


@dataclass
class StepRecord:
    mode: str
    step: int
    warmup: bool
    ts_start_unix_ns: int
    ts_end_unix_ns: int
    ts_mid_unix_ns: int
    step_ms_max: float
    steps_per_sec: float
    samples_per_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase4 reporter")
    parser.add_argument("--input", required=True, help="phase4 run root")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def iter_mode_dirs(run_root: Path) -> Iterable[Path]:
    for candidate in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if candidate.name.startswith("."):
            continue
        yield candidate


def load_step_metrics(mode_dir: Path) -> list[StepRecord]:
    metrics_files = sorted(mode_dir.glob("*_step_metrics.jsonl"))
    if not metrics_files:
        return []

    result: list[StepRecord] = []
    with metrics_files[0].open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result.append(
                StepRecord(
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
    return result


def ns_to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).astimezone()


def build_palette(modes: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {mode: cmap(i % 10) for i, mode in enumerate(modes)}


def plot_latency_over_time(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
      records = records_by_mode[mode]
      xs = [ns_to_datetime(r.ts_mid_unix_ns) for r in records]
      ys = [r.step_ms_max for r in records]
      ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=mode, color=palette[mode])

    ax.set_title("Step Latency Over Time")
    ax.set_xlabel("Wall-clock Time")
    ax.set_ylabel("Latency (ms)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_throughput_over_time(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mode in modes:
      records = records_by_mode[mode]
      xs = [ns_to_datetime(r.ts_mid_unix_ns) for r in records]
      ys = [r.samples_per_sec for r in records]
      ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=mode, color=palette[mode])

    ax.set_title("Sample Throughput Over Time")
    ax.set_xlabel("Wall-clock Time")
    ax.set_ylabel("Samples / sec")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_step_timeline(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = 0
    yticks: list[int] = []
    ylabels: list[str] = []

    for mode in modes:
        for record in records_by_mode[mode]:
            start = mdates.date2num(ns_to_datetime(record.ts_start_unix_ns))
            end = mdates.date2num(ns_to_datetime(record.ts_end_unix_ns))
            width = max(end - start, 1e-12)
            color = palette[mode]
            alpha = 0.45 if record.warmup else 0.85
            ax.barh(y, width, left=start, height=0.7, color=color, alpha=alpha)
            yticks.append(y)
            ylabels.append(f"{mode}: step {record.step}")
            y += 1
        y += 1

    ax.set_title("Step Timeline")
    ax.set_xlabel("Wall-clock Time")
    ax.set_ylabel("Mode / Step")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, axis="x", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_timeline_table(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    rows = []
    for mode, records in records_by_mode.items():
        for record in records:
            rows.append(
                {
                    "mode": mode,
                    "step": record.step,
                    "warmup": record.warmup,
                    "start": ns_to_datetime(record.ts_start_unix_ns).isoformat(),
                    "end": ns_to_datetime(record.ts_end_unix_ns).isoformat(),
                    "latency_ms": record.step_ms_max,
                    "samples_per_sec": record.samples_per_sec,
                    "steps_per_sec": record.steps_per_sec,
                }
            )
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_html(run_root: Path, output_dir: Path, modes: list[str]) -> None:
    sections = [
        ("Latency Over Time", "latency_over_time.png"),
        ("Throughput Over Time", "throughput_over_time.png"),
        ("Step Timeline", "step_timeline.png"),
    ]
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Phase4 Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;border:1px solid #ddd;} code{background:#f4f4f4;padding:2px 4px;} li{margin:4px 0;}</style>",
        "</head><body>",
        f"<h1>Phase4 Report</h1>",
        f"<p><strong>Run root:</strong> <code>{run_root}</code></p>",
        f"<p><strong>Modes:</strong> {', '.join(modes)}</p>",
        "<p><strong>Timeline table:</strong> <code>step_timeline_table.json</code></p>",
    ]
    for title, filename in sections:
        html.append(f"<h2>{title}</h2>")
        html.append(f"<img src='{filename}' alt='{title}'>")
    html.extend(["</body></html>"])
    (output_dir / "phase4_report.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase4-reporter] load run root={run_root}")

    records_by_mode: dict[str, list[StepRecord]] = {}
    for mode_dir in iter_mode_dirs(run_root):
        records = load_step_metrics(mode_dir)
        if not records:
            continue
        records_by_mode[mode_dir.name] = records
        print(f"[phase4-reporter] loaded mode={mode_dir.name} steps={len(records)}")

    if not records_by_mode:
        raise FileNotFoundError(f"no *_step_metrics.jsonl found under {run_root}")

    modes = list(records_by_mode.keys())

    latency_png = output_dir / "latency_over_time.png"
    throughput_png = output_dir / "throughput_over_time.png"
    timeline_png = output_dir / "step_timeline.png"
    timeline_json = output_dir / "step_timeline_table.json"

    print("[phase4-reporter] plot latency over time")
    plot_latency_over_time(records_by_mode, latency_png)
    print("[phase4-reporter] plot throughput over time")
    plot_throughput_over_time(records_by_mode, throughput_png)
    print("[phase4-reporter] plot step timeline")
    plot_step_timeline(records_by_mode, timeline_png)
    print("[phase4-reporter] write step timeline table")
    write_timeline_table(records_by_mode, timeline_json)
    print("[phase4-reporter] write html")
    write_html(run_root, output_dir, modes)
    print(f"[phase4-reporter] done html={output_dir / 'phase4_report.html'}")


if __name__ == "__main__":
    main()
