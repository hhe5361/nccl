#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase4 reporter")
    parser.add_argument("--input", required=True, help="phase4 run root")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV_RE.finditer(line)}


def discover_repeat_dirs(run_root: Path) -> list[Path]:
    repeat_dirs = sorted(
        p for p in run_root.iterdir()
        if p.is_dir() and p.name.startswith("repeat_")
    )
    if repeat_dirs:
        return repeat_dirs
    return [run_root]


def iter_mode_dirs(repeat_dir: Path) -> Iterable[Path]:
    for candidate in sorted(p for p in repeat_dir.iterdir() if p.is_dir()):
        if candidate.name.startswith("."):
            continue
        yield candidate


def load_step_metrics(repeat_name: str, mode_dir: Path) -> list[StepRecord]:
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
    return result


def build_palette(keys: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {key: cmap(i % 10) for i, key in enumerate(keys)}


def mode_start_ns(records: list[StepRecord]) -> int:
    return min(record.ts_start_unix_ns for record in records)


def elapsed_seconds(base_ns: int, ns: int) -> float:
    return (ns - base_ns) / 1e9


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
        color = palette[mode]
        base_ns = mode_start_ns(records)
        line_xs: list[float] = []
        line_ys: list[float] = []
        for record in records:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            end_s = elapsed_seconds(base_ns, record.ts_end_unix_ns)
            mid_s = elapsed_seconds(base_ns, record.ts_mid_unix_ns)
            metric_value = getattr(record, metric_key)
            alpha = 0.35 if record.warmup else 0.9
            marker = "x" if record.warmup else "o"
            marker_size = 18 if record.warmup else 14
            ax.hlines(metric_value, start_s, end_s, color=color, linewidth=2.0, alpha=alpha)
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


def plot_step_timeline(records_by_mode: dict[str, list[StepRecord]], output_path: Path, *, title: str) -> None:
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
        color = palette[repeat]
        base_ns = mode_start_ns(records)
        line_xs: list[float] = []
        line_ys: list[float] = []
        for record in records:
            start_s = elapsed_seconds(base_ns, record.ts_start_unix_ns)
            mid_s = elapsed_seconds(base_ns, record.ts_mid_unix_ns)
            metric_value = getattr(record, metric_key)
            alpha = 0.35 if record.warmup else 0.9
            marker = "x" if record.warmup else "o"
            marker_size = 18 if record.warmup else 14
            ax.scatter([start_s], [metric_value], color=color, s=marker_size, marker=marker, alpha=alpha)
            line_xs.append(mid_s)
            line_ys.append(metric_value)
        ax.plot(line_xs, line_ys, color=color, linewidth=1.8, alpha=0.9, label=repeat)

    ax.set_title(title)
    ax.set_xlabel("Time Since Mode Start (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_post_timestamps_ms(mode_dir: Path) -> list[float]:
    worker_log_dir = mode_dir / "worker01"
    nccl_logs = sorted(worker_log_dir.glob("nccl.*.log"))
    if not nccl_logs:
        worker_dirs = sorted(p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("worker"))
        if worker_dirs:
            nccl_logs = sorted(worker_dirs[0].glob("nccl.*.log"))
    timestamps_ns: list[int] = []
    for log_path in nccl_logs:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PHASE0 event=PROXY_RECV_POST" not in line:
                    continue
                row = parse_kv(line)
                tns = row.get("tNs")
                if tns is None:
                    continue
                timestamps_ns.append(int(tns))
    if not timestamps_ns:
        return []
    base = min(timestamps_ns)
    return sorted((ts - base) / 1e6 for ts in timestamps_ns)


def plot_post_cumulative_overlay(
    post_ms_by_mode: dict[str, list[float]],
    output_path: Path,
    *,
    title: str,
) -> None:
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
        "<p>Warmup steps are marked with <code>x</code>; measured steps are marked with <code>o</code>.</p>",
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

    repeat_dirs = discover_repeat_dirs(run_root)
    records_by_repeat_mode: dict[str, dict[str, list[StepRecord]]] = {}
    for repeat_dir in repeat_dirs:
        repeat_name = repeat_dir.name if repeat_dir != run_root else "repeat_01"
        records_by_mode: dict[str, list[StepRecord]] = {}
        for mode_dir in iter_mode_dirs(repeat_dir):
            records = load_step_metrics(repeat_name, mode_dir)
            if not records:
                continue
            records_by_mode[mode_dir.name] = records
            print(f"[phase4-reporter] loaded repeat={repeat_name} mode={mode_dir.name} steps={len(records)}")
        if records_by_mode:
            records_by_repeat_mode[repeat_name] = records_by_mode

    if not records_by_repeat_mode:
        raise FileNotFoundError(f"no *_step_metrics.jsonl found under {run_root}")

    repeat_sections: list[tuple[str, list[tuple[str, str]]]] = []
    for repeat_name, records_by_mode in records_by_repeat_mode.items():
        images: list[tuple[str, str]] = []
        latency_png = f"{repeat_name}_latency_over_time.png"
        latency_zoom_png = f"{repeat_name}_latency_over_time_zoom_0_10ms.png"
        latency_zoom_3_png = f"{repeat_name}_latency_over_time_zoom_0_3ms.png"
        throughput_png = f"{repeat_name}_throughput_over_time.png"
        throughput_zoom_png = f"{repeat_name}_throughput_over_time_zoom_25000_40000.png"
        throughput_zoom_32_38_png = f"{repeat_name}_throughput_over_time_zoom_32000_38000.png"
        timeline_png = f"{repeat_name}_step_timeline.png"
        post_overlay_png = f"{repeat_name}_cumulative_post_overlay.png"

        print(f"[phase4-reporter] plot repeat latency overlay repeat={repeat_name}")
        plot_metric_overlay(
            records_by_mode,
            output_dir / latency_png,
            title=f"{repeat_name} Latency Overlay Over Relative Mode Time",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
        )
        plot_metric_overlay(
            records_by_mode,
            output_dir / latency_zoom_png,
            title=f"{repeat_name} Latency Overlay Over Relative Mode Time (0-10 ms)",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
            y_limits=(0.0, 10.0),
        )
        plot_metric_overlay(
            records_by_mode,
            output_dir / latency_zoom_3_png,
            title=f"{repeat_name} Latency Overlay Over Relative Mode Time (0-3 ms)",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
            y_limits=(0.0, 3.0),
        )
        print(f"[phase4-reporter] plot repeat throughput overlay repeat={repeat_name}")
        plot_metric_overlay(
            records_by_mode,
            output_dir / throughput_png,
            title=f"{repeat_name} Throughput Overlay Over Relative Mode Time",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
        )
        plot_metric_overlay(
            records_by_mode,
            output_dir / throughput_zoom_png,
            title=f"{repeat_name} Throughput Overlay Over Relative Mode Time (25000-40000)",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
            y_limits=(25000.0, 40000.0),
        )
        plot_metric_overlay(
            records_by_mode,
            output_dir / throughput_zoom_32_38_png,
            title=f"{repeat_name} Throughput Overlay Over Relative Mode Time (32000-38000)",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
            y_limits=(32000.0, 38000.0),
        )
        print(f"[phase4-reporter] plot repeat timeline repeat={repeat_name}")
        plot_step_timeline(
            records_by_mode,
            output_dir / timeline_png,
            title=f"{repeat_name} Mode Step Timeline Overlay Over Relative Mode Time",
        )
        print(f"[phase4-reporter] plot repeat cumulative post overlay repeat={repeat_name}")
        repeat_dir = run_root / repeat_name if (run_root / repeat_name).exists() else run_root
        post_ms_by_mode = {mode: load_post_timestamps_ms(repeat_dir / mode) for mode in records_by_mode.keys()}
        plot_post_cumulative_overlay(
            post_ms_by_mode,
            output_dir / post_overlay_png,
            title=f"{repeat_name} Cumulative PROXY_RECV_POST Over Relative Mode Time",
        )

        images.extend([
            ("Latency Overlay", latency_png),
            ("Latency Overlay 0-10 ms", latency_zoom_png),
            ("Latency Overlay 0-3 ms", latency_zoom_3_png),
            ("Throughput Overlay", throughput_png),
            ("Throughput Overlay 25000-40000", throughput_zoom_png),
            ("Throughput Overlay 32000-38000", throughput_zoom_32_38_png),
            ("Step Timeline Overlay", timeline_png),
            ("Cumulative POST Overlay", post_overlay_png),
        ])
        repeat_sections.append((f"Repeat {repeat_name}", images))

    mode_to_repeat_records: dict[str, dict[str, list[StepRecord]]] = {}
    for repeat_name, records_by_mode in records_by_repeat_mode.items():
        for mode, records in records_by_mode.items():
            mode_to_repeat_records.setdefault(mode, {})[repeat_name] = records

    mode_sections: list[tuple[str, list[tuple[str, str]]]] = []
    for mode in sorted(mode_to_repeat_records.keys()):
        records_by_repeat = mode_to_repeat_records[mode]
        latency_png = f"{mode}_repeat_latency_overlay.png"
        throughput_png = f"{mode}_repeat_throughput_overlay.png"
        print(f"[phase4-reporter] plot mode repeat overlay mode={mode}")
        plot_repeat_metric_overlay(
            records_by_repeat,
            output_dir / latency_png,
            title=f"{mode} Latency Across Repeats",
            ylabel="Latency (ms)",
            metric_key="step_ms_max",
        )
        plot_repeat_metric_overlay(
            records_by_repeat,
            output_dir / throughput_png,
            title=f"{mode} Throughput Across Repeats",
            ylabel="Samples / sec",
            metric_key="samples_per_sec",
        )
        mode_sections.append((f"Mode {mode}", [
            ("Latency Across Repeats", latency_png),
            ("Throughput Across Repeats", throughput_png),
        ]))

    print("[phase4-reporter] write html")
    write_html(run_root, output_dir, repeat_sections, mode_sections)
    print(f"[phase4-reporter] done html={output_dir / 'phase4_report.html'}")


if __name__ == "__main__":
    main()
