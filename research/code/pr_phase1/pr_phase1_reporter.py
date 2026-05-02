#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class RepeatData:
    mode: str
    experiment: str
    repeat_name: str
    worker: str
    rank: int
    payload_mb: float
    summary_path: Path
    trace_path: Path
    validation_path: Optional[Path]


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


def safe_median(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return statistics.median(vals) if vals else None


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
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
    vals = sorted(values)
    return percentile(vals, 0.25), percentile(vals, 0.75)


def throughput_gbps(payload_mb: float, duration_ms: float) -> float:
    if duration_ms <= 0:
        return 0.0
    payload_bytes = payload_mb * 1024.0 * 1024.0
    return payload_bytes / (duration_ms / 1000.0) / 1e9


def discover_runs(run_root: Path) -> Dict[str, Dict[str, Dict[str, List[RepeatData]]]]:
    discovered: Dict[str, Dict[str, Dict[str, List[RepeatData]]]] = {}
    for mode_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        mode = mode_dir.name
        for experiment_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
            experiment = experiment_dir.name
            for repeat_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
                validation_path = repeat_dir / "stock_probe_validation.json"
                for worker_dir in sorted(p for p in repeat_dir.iterdir() if p.is_dir() and p.name.startswith("worker")):
                    summary_files = sorted(worker_dir.glob("rank*_summary.json"))
                    trace_files = sorted(worker_dir.glob("rank*_step_trace.jsonl"))
                    if not summary_files or not trace_files:
                        continue
                    summary_path = summary_files[0]
                    trace_path = trace_files[0]
                    rank_str = summary_path.stem.split("_", 1)[0].replace("rank", "")
                    rank = int(rank_str)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    payload_mb = float(summary.get("payload_mb", 0))
                    repeat_data = RepeatData(
                        mode=mode,
                        experiment=experiment,
                        repeat_name=repeat_dir.name,
                        worker=worker_dir.name,
                        rank=rank,
                        payload_mb=payload_mb,
                        summary_path=summary_path,
                        trace_path=trace_path,
                        validation_path=validation_path if validation_path.exists() else None,
                    )
                    discovered.setdefault(experiment, {}).setdefault(mode, {}).setdefault(repeat_dir.name, []).append(repeat_data)
    return discovered


def load_trace(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_repeat_step_series(repeat_items: List[RepeatData]) -> Dict[int, Dict[str, float]]:
    by_step: Dict[int, Dict[str, List[float]]] = {}
    payload_mb = repeat_items[0].payload_mb if repeat_items else 0.0
    for item in repeat_items:
        for row in load_trace(item.trace_path):
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
        for row in load_trace(item.trace_path):
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
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    ok = bool(payload.get("ok", False))
    workers = payload.get("workers", [])
    failed = [w.get("worker", "?") for w in workers if not w.get("ok", False)]
    return ("OK" if ok else "FAIL", ",".join(failed) if failed else "-")


def make_mode_palette(modes: List[str]) -> Dict[str, str]:
    base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    palette: Dict[str, str] = {}
    for idx, mode in enumerate(modes):
        if mode == "STOCK":
            palette[mode] = "#222222"
        else:
            palette[mode] = base[idx % len(base)]
    return palette


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

    box = ax.boxplot(series, patch_artist=True, labels=labels, showmeans=True)
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


def build_experiment_table(experiment: str, mode_repeats: Dict[str, Dict[str, List[RepeatData]]]) -> str:
    rows = []
    modes = sorted(mode_repeats.keys(), key=mode_sort_key)
    for mode in modes:
        per_repeat = [per_repeat_overall(items) for _, items in sorted(mode_repeats[mode].items())]
        lat_medians = [x["latency_median_ms"] for x in per_repeat]
        thr_medians = [x["throughput_median_gbps"] for x in per_repeat]
        lat_p95s = [x["latency_p95_ms"] for x in per_repeat]
        validation_status, validation_detail = summarize_validation(next(iter(mode_repeats[mode].values()))[0].validation_path if mode_repeats[mode] else None)
        rows.append(
            "<tr>"
            f"<td>{escape(mode)}</td>"
            f"<td>{len(per_repeat)}</td>"
            f"<td>{(safe_median(lat_medians) or 0.0):.3f}</td>"
            f"<td>{(safe_mean(lat_medians) or 0.0):.3f}</td>"
            f"<td>{(safe_median(lat_p95s) or 0.0):.3f}</td>"
            f"<td>{(safe_median(thr_medians) or 0.0):.3f}</td>"
            f"<td>{validation_status}</td>"
            f"<td>{escape(validation_detail)}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>Mode</th><th>Repeats</th><th>Median Latency (ms)</th><th>Mean Latency (ms)</th>"
        "<th>Median p95 Latency (ms)</th><th>Median Throughput (GB/s)</th><th>Validation</th><th>Failed Workers</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def generate_html(run_root: Path, output_dir: Path, experiment_sections: List[str]) -> None:
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
      max-width: 1400px;
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
    p {{ color: var(--muted); }}
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>PR Phase1 Report</h1>
      <p>Run root: <code>{escape(str(run_root))}</code></p>
      <p>Primary aggregate: median. Mean is kept as a secondary reference because step outliers can distort network experiments.</p>
    </div>
    {''.join(experiment_sections)}
  </div>
</body>
</html>
"""
    (output_dir / "phase1_report.html").write_text(html, encoding="utf-8")


def build_report(run_root: Path, output_dir: Path) -> None:
    discovered = discover_runs(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []

    for experiment in sorted(discovered.keys()):
        mode_repeats = discovered[experiment]
        latency_box = plot_summary_box(experiment, mode_repeats, output_dir, "latency_median_ms")
        throughput_box = plot_summary_box(experiment, mode_repeats, output_dir, "throughput_median_gbps")
        step_latency = plot_step_overlay(experiment, mode_repeats, output_dir, "latency_ms")
        step_throughput = plot_step_overlay(experiment, mode_repeats, output_dir, "throughput_gbps")
        delta_vs_stock = plot_delta_vs_stock(experiment, mode_repeats, output_dir)
        latency_vs_w = plot_vs_w_lines(experiment, mode_repeats, output_dir, "latency_median_ms")
        throughput_vs_w = plot_vs_w_lines(experiment, mode_repeats, output_dir, "throughput_median_gbps")
        summary_table = build_experiment_table(experiment, mode_repeats)

        sections.append(
            f"""
            <section class="card">
              <h2>{escape(experiment)}</h2>
              <p>Mode comparison is reported with repeat-level medians. Step overlays use per-step medians with IQR shading across repeats.</p>
              {summary_table}
              <div class="grid" style="margin-top:18px;">
                <div><h3>Latency by Mode</h3><img src="{escape(latency_box)}" alt="{escape(experiment)} latency boxplot"></div>
                <div><h3>Throughput by Mode</h3><img src="{escape(throughput_box)}" alt="{escape(experiment)} throughput boxplot"></div>
                <div><h3>Step Latency Overlay</h3><img src="{escape(step_latency)}" alt="{escape(experiment)} step latency overlay"></div>
                <div><h3>Step Throughput Overlay</h3><img src="{escape(step_throughput)}" alt="{escape(experiment)} step throughput overlay"></div>
                <div><h3>Latency vs W</h3><img src="{escape(latency_vs_w)}" alt="{escape(experiment)} latency vs W"></div>
                <div><h3>Throughput vs W</h3><img src="{escape(throughput_vs_w)}" alt="{escape(experiment)} throughput vs W"></div>
              </div>
              <div style="margin-top:18px;">
                <h3>Delta vs STOCK</h3>
                <img src="{escape(delta_vs_stock)}" alt="{escape(experiment)} delta vs STOCK">
              </div>
            </section>
            """
        )

    generate_html(run_root, output_dir, sections)


def main() -> None:
    args = parse_args()
    build_report(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
