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


def first_start_ns(records_by_mode: dict[str, list[StepRecord]]) -> int:
    return min(record.ts_start_unix_ns for records in records_by_mode.values() for record in records)


def elapsed_seconds(base_ns: int, ns: int) -> float:
    return (ns - base_ns) / 1e9


def mode_start_ns(records: list[StepRecord]) -> int:
    return min(record.ts_start_unix_ns for record in records)


def plot_metric_over_time(
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
        color = palette[mode]
        base_ns = mode_start_ns(records)
        first = True
        line_xs = []
        line_ys = []
        for record in records:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            end_s = elapsed_seconds(base_ns, record.ts_end_unix_ns)
            mid_s = elapsed_seconds(base_ns, record.ts_mid_unix_ns)
            alpha = 0.35 if record.warmup else 0.9
            metric_value = getattr(record, metric_key)
            ax.hlines(
                metric_value,
                start_s,
                end_s,
                color=color,
                linewidth=2.2,
                alpha=alpha,
                label=None,
            )
            marker = "x" if record.warmup else "o"
            marker_size = 18 if record.warmup else 14
            ax.scatter([start_s], [metric_value], color=color, s=marker_size, marker=marker, alpha=alpha)
            line_xs.append(mid_s)
            line_ys.append(metric_value)
        ax.plot(line_xs, line_ys, color=color, linewidth=1.6, alpha=0.9, label=mode)

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


def plot_latency_over_time(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    plot_metric_over_time(
        records_by_mode,
        output_path,
        title="Latency Overlay Over Relative Mode Time",
        ylabel="Latency (ms)",
        metric_key="step_ms_max",
    )


def plot_latency_over_time_zoom(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    plot_metric_over_time(
        records_by_mode,
        output_path,
        title="Latency Overlay Over Relative Mode Time (0-10 ms)",
        ylabel="Latency (ms)",
        metric_key="step_ms_max",
        y_limits=(0.0, 10.0),
    )


def plot_throughput_over_time(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    plot_metric_over_time(
        records_by_mode,
        output_path,
        title="Throughput Overlay Over Relative Mode Time",
        ylabel="Samples / sec",
        metric_key="samples_per_sec",
    )


def plot_throughput_over_time_zoom(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    plot_metric_over_time(
        records_by_mode,
        output_path,
        title="Throughput Overlay Over Relative Mode Time (25000-40000)",
        ylabel="Samples / sec",
        metric_key="samples_per_sec",
        y_limits=(25000.0, 40000.0),
    )


def plot_step_timeline(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    modes = list(records_by_mode.keys())
    palette = build_palette(modes)

    fig, ax = plt.subplots(figsize=(12, max(3.5, 1.1 * len(modes) + 1.5)))
    yticks: list[float] = []
    ylabels: list[str] = []

    for y, mode in enumerate(modes):
        base_ns = mode_start_ns(records_by_mode[mode])
        for record in records_by_mode[mode]:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            end_s = elapsed_seconds(base_ns, record.ts_end_unix_ns)
            width = max(end_s - start_s, 1e-9)
            alpha = 0.35 if record.warmup else 0.85
            ax.barh(y, width, left=start_s, height=0.6, color=palette[mode], alpha=alpha)
            ax.text(
                start_s,
                y + 0.32,
                f"s{record.step}",
                fontsize=7,
                va="bottom",
                ha="left",
                color=palette[mode],
            )
        yticks.append(y)
        ylabels.append(mode)

    ax.set_title("Mode Step Timeline Overlay Over Relative Mode Time")
    ax.set_xlabel("Time Since Mode Start (s)")
    ax.set_ylabel("Mode")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_timeline_table(records_by_mode: dict[str, list[StepRecord]], output_path: Path) -> None:
    rows = []
    for mode, records in records_by_mode.items():
        base_ns = mode_start_ns(records)
        for record in records:
            rows.append(
                {
                    "mode": mode,
                    "step": record.step,
                    "warmup": record.warmup,
                    "start": ns_to_datetime(record.ts_start_unix_ns).isoformat(),
                    "end": ns_to_datetime(record.ts_end_unix_ns).isoformat(),
                    "elapsed_start_s": elapsed_seconds(base_ns, record.ts_start_unix_ns),
                    "elapsed_end_s": elapsed_seconds(base_ns, record.ts_end_unix_ns),
                    "elapsed_mid_s": elapsed_seconds(base_ns, record.ts_mid_unix_ns),
                    "latency_ms": record.step_ms_max,
                    "samples_per_sec": record.samples_per_sec,
                    "steps_per_sec": record.steps_per_sec,
                }
            )
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_html(run_root: Path, output_dir: Path, modes: list[str]) -> None:
    sections = [
        ("Latency Overlay Over Relative Mode Time", "latency_over_time.png"),
        ("Latency Overlay Over Relative Mode Time (0-10 ms)", "latency_over_time_zoom_0_10ms.png"),
        ("Throughput Overlay Over Relative Mode Time", "throughput_over_time.png"),
        ("Throughput Overlay Over Relative Mode Time (25000-40000)", "throughput_over_time_zoom_25000_40000.png"),
        ("Mode Step Timeline Overlay Over Relative Mode Time", "step_timeline.png"),
    ]
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Phase4 Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;border:1px solid #ddd;} code{background:#f4f4f4;padding:2px 4px;} li{margin:4px 0;}</style>",
        "</head><body>",
        "<h1>Phase4 Report</h1>",
        f"<p><strong>Run root:</strong> <code>{run_root}</code></p>",
        f"<p><strong>Modes:</strong> {', '.join(modes)}</p>",
        "<p><strong>Timeline table:</strong> <code>step_timeline_table.json</code></p>",
        "<p>Each mode is normalized so its first step starts at t=0, then step start-stop segments and mode overlays are drawn on the same relative-time axis.</p>",
        "<p>Warmup steps are marked with <code>x</code>; measured steps are marked with <code>o</code>.</p>",
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
    latency_zoom_png = output_dir / "latency_over_time_zoom_0_10ms.png"
    throughput_png = output_dir / "throughput_over_time.png"
    throughput_zoom_png = output_dir / "throughput_over_time_zoom_25000_40000.png"
    timeline_png = output_dir / "step_timeline.png"
    timeline_json = output_dir / "step_timeline_table.json"

    print("[phase4-reporter] plot latency overlay")
    plot_latency_over_time(records_by_mode, latency_png)
    print("[phase4-reporter] plot latency overlay zoom")
    plot_latency_over_time_zoom(records_by_mode, latency_zoom_png)
    print("[phase4-reporter] plot throughput overlay")
    plot_throughput_over_time(records_by_mode, throughput_png)
    print("[phase4-reporter] plot throughput overlay zoom")
    plot_throughput_over_time_zoom(records_by_mode, throughput_zoom_png)
    print("[phase4-reporter] plot mode timeline overlay")
    plot_step_timeline(records_by_mode, timeline_png)
    print("[phase4-reporter] write step timeline table")
    write_timeline_table(records_by_mode, timeline_json)
    print("[phase4-reporter] write html")
    write_html(run_root, output_dir, modes)
    print(f"[phase4-reporter] done html={output_dir / 'phase4_report.html'}")


if __name__ == "__main__":
    main()
