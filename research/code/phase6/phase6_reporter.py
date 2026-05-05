#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def centered_limits(values: list[float], min_half_span: float, frac: float) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    median = percentile(values, 0.50)
    half_span = max(min_half_span, abs(median) * frac)
    lower = max(0.0, median - half_span)
    upper = median + half_span
    if upper <= lower:
        upper = lower + max(1.0, min_half_span)
    return lower, upper


def parse_kv_line(line: str, marker: str) -> dict[str, str] | None:
    if marker not in line:
        return None
    row = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[key] = value
    return row if "event" in row else None


def load_mode_series(mode_dir: Path) -> dict[str, list[tuple[float, float]]]:
    worker_dirs = sorted(p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("worker"))
    chosen_worker = next((p for p in worker_dirs if p.name == "worker01"), worker_dirs[0] if worker_dirs else None)

    post_times_ns: list[int] = []
    allow_series: list[tuple[int, float]] = []
    stall_times_ns: list[int] = []

    if chosen_worker is None:
        return {
            "worker": "",
            "cumulative_posts": [],
            "cumulative_allow": [],
            "cumulative_stalls": [],
        }

    for log_path in sorted(chosen_worker.glob("nccl.*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row0 = parse_kv_line(line, "PHASE0 event=")
            if row0 is not None and row0.get("event") == "PROXY_RECV_POST":
                post_times_ns.append(int(row0.get("tNs", "0")))
                continue
            row6 = parse_kv_line(line, "PHASE6 event=")
            if row6 is None:
                continue
            event = row6.get("event", "")
            if event == "RATE_ALLOW":
                allow_series.append((int(row6.get("tNs", "0")), float(row6.get("postCost", "0"))))
            elif event == "RATE_STALL":
                stall_times_ns.append(int(row6.get("tNs", "0")))

    anchors = [t for t in post_times_ns if t > 0]
    if not anchors:
        anchors = [t for t, _ in allow_series if t > 0]
    if not anchors:
        anchors = [t for t in stall_times_ns if t > 0]
    t0 = min(anchors) if anchors else 0

    posts_sorted = sorted(t for t in post_times_ns if t > 0)
    cumulative_posts = [((t - t0) / 1e6, idx + 1) for idx, t in enumerate(posts_sorted)]

    allow_sorted = sorted((t, c) for t, c in allow_series if t > 0)
    cumulative_allow: list[tuple[float, float]] = []
    running_allow = 0.0
    for t, cost in allow_sorted:
        running_allow += cost
        cumulative_allow.append(((t - t0) / 1e6, running_allow))

    stalls_sorted = sorted(t for t in stall_times_ns if t > 0)
    cumulative_stalls = [((t - t0) / 1e6, idx + 1) for idx, t in enumerate(stalls_sorted)]

    return {
        "worker": chosen_worker.name,
        "cumulative_posts": cumulative_posts,
        "cumulative_allow": cumulative_allow,
        "cumulative_stalls": cumulative_stalls,
    }


def plot_step_overlay(
    records_by_mode: dict[str, list[dict]],
    title: str,
    ylabel: str,
    value_key: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for mode, records in records_by_mode.items():
        if not records:
            continue
        xs_mid = [r["elapsed_mid_ms"] for r in records]
        ys = [r[value_key] for r in records]
        ax.plot(xs_mid, ys, linewidth=2, label=mode)
        for r in records:
            ax.hlines(r[value_key], r["elapsed_start_ms"], r["elapsed_end_ms"], linewidth=1.2, alpha=0.55)
            ax.text(
                r["elapsed_start_ms"],
                r[value_key],
                str(r["step_plot_index"]),
                fontsize=7,
                ha="center",
                va="bottom",
                clip_on=True,
            )
    ax.set_title(title)
    ax.set_xlabel("Relative mode time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cumulative_overlay(mode_series: dict[str, list[tuple[float, float]]], title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode, series in mode_series.items():
        if not series:
            continue
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        ax.step(xs, ys, where="post", label=mode, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Relative mode time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_step_records(mode_dir: Path) -> list[dict]:
    metrics_path = mode_dir / f"{mode_dir.name}_step_metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
    if not rows:
        return []
    t0 = min(int(r["ts_start_unix_ns"]) for r in rows)
    out = []
    for idx, row in enumerate(rows, start=1):
        start_ns = int(row["ts_start_unix_ns"])
        end_ns = int(row["ts_end_unix_ns"])
        out.append(
            {
                "step_index": int(row.get("step", idx - 1)),
                "step_plot_index": idx,
                "warmup": bool(row.get("warmup", False)),
                "elapsed_start_ms": (start_ns - t0) / 1e6,
                "elapsed_end_ms": (end_ns - t0) / 1e6,
                "elapsed_mid_ms": ((start_ns + end_ns) / 2 - t0) / 1e6,
                "latency_ms": float(row.get("step_ms_max", row.get("step_ms_mean", 0.0))),
                "throughput": float(row.get("samples_per_sec", 0.0)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render phase6 diagnostic plots")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repeat_dirs = sorted(p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    if not repeat_dirs:
        repeat_dirs = [input_root]

    html_sections: list[str] = []

    for repeat_dir in repeat_dirs:
        mode_dirs = sorted(p for p in repeat_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        cumulative_data = {mode_dir.name: load_mode_series(mode_dir) for mode_dir in mode_dirs}
        step_data = {mode_dir.name: load_step_records(mode_dir) for mode_dir in mode_dirs}

        posts_plot = output_dir / f"{repeat_dir.name}_cumulative_posts_overlay.png"
        allow_plot = output_dir / f"{repeat_dir.name}_cumulative_rate_allow_overlay.png"
        stalls_plot = output_dir / f"{repeat_dir.name}_cumulative_rate_stall_overlay.png"
        latency_plot = output_dir / f"{repeat_dir.name}_latency_overlay_full.png"
        latency_median_plot = output_dir / f"{repeat_dir.name}_latency_overlay_median_band.png"
        throughput_plot = output_dir / f"{repeat_dir.name}_throughput_overlay_full.png"
        throughput_median_plot = output_dir / f"{repeat_dir.name}_throughput_overlay_median_band.png"

        plot_cumulative_overlay(
            {mode: values["cumulative_posts"] for mode, values in cumulative_data.items()},
            title=f"{repeat_dir.name} cumulative recv POST count (worker01)",
            ylabel="Cumulative POST count",
            output_path=posts_plot,
        )
        plot_cumulative_overlay(
            {mode: values["cumulative_allow"] for mode, values in cumulative_data.items()},
            title=f"{repeat_dir.name} cumulative rate-allow cost (worker01)",
            ylabel="Cumulative allowed post cost",
            output_path=allow_plot,
        )
        plot_cumulative_overlay(
            {mode: values["cumulative_stalls"] for mode, values in cumulative_data.items()},
            title=f"{repeat_dir.name} cumulative rate-stall events (worker01)",
            ylabel="Cumulative stall events",
            output_path=stalls_plot,
        )

        latency_values = [r["latency_ms"] for records in step_data.values() for r in records if not r["warmup"]]
        throughput_values = [r["throughput"] for records in step_data.values() for r in records if not r["warmup"]]
        latency_limits = centered_limits(latency_values, min_half_span=0.5, frac=0.25)
        throughput_limits = centered_limits(throughput_values, min_half_span=500.0, frac=0.08)

        plot_step_overlay(
            step_data,
            title=f"{repeat_dir.name} latency overlay over relative mode time",
            ylabel="Latency (ms)",
            value_key="latency_ms",
            output_path=latency_plot,
        )
        plot_step_overlay(
            step_data,
            title=f"{repeat_dir.name} latency overlay median band",
            ylabel="Latency (ms)",
            value_key="latency_ms",
            output_path=latency_median_plot,
            y_limits=latency_limits,
        )
        plot_step_overlay(
            step_data,
            title=f"{repeat_dir.name} throughput overlay over relative mode time",
            ylabel="Throughput (samples/sec)",
            value_key="throughput",
            output_path=throughput_plot,
        )
        plot_step_overlay(
            step_data,
            title=f"{repeat_dir.name} throughput overlay median band",
            ylabel="Throughput (samples/sec)",
            value_key="throughput",
            output_path=throughput_median_plot,
            y_limits=throughput_limits,
        )

        summary_rows = []
        for mode_dir in mode_dirs:
            summary_path = mode_dir / f"{mode_dir.name}_summary.json"
            perf_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            summary_rows.append(
                {
                    "mode": mode_dir.name,
                    "worker_for_cumulative": cumulative_data[mode_dir.name]["worker"],
                    "step_ms_avg": perf_summary.get("step_ms_avg", 0),
                    "steps_per_sec_avg": perf_summary.get("steps_per_sec_avg", 0),
                    "samples_per_sec_avg": perf_summary.get("samples_per_sec_avg", 0),
                    "post_count": len(cumulative_data[mode_dir.name]["cumulative_posts"]),
                    "allow_count": len(cumulative_data[mode_dir.name]["cumulative_allow"]),
                    "stall_count": len(cumulative_data[mode_dir.name]["cumulative_stalls"]),
                }
            )
        (output_dir / f"{repeat_dir.name}_phase6_report_summary.json").write_text(
            json.dumps(summary_rows, indent=2), encoding="utf-8"
        )

        html_sections.append(
            f"""
            <section>
              <h2>{repeat_dir.name}</h2>
              <p>Raw cumulative NCCL plots use a single representative worker per mode: worker01 if present, otherwise the first worker directory.</p>
              <img src="{posts_plot.name}" style="max-width: 100%;"><br>
              <img src="{allow_plot.name}" style="max-width: 100%;"><br>
              <img src="{stalls_plot.name}" style="max-width: 100%;"><br>
              <img src="{latency_plot.name}" style="max-width: 100%;"><br>
              <img src="{latency_median_plot.name}" style="max-width: 100%;"><br>
              <img src="{throughput_plot.name}" style="max-width: 100%;"><br>
              <img src="{throughput_median_plot.name}" style="max-width: 100%;">
            </section>
            """
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase6 Report</title></head>
<body>
  <h1>Phase6 Post-rate Control Report</h1>
  <p>Relative-time overlays for cumulative recv POST, rate limiter events, latency, and throughput.</p>
  {''.join(html_sections)}
</body></html>"""
    (output_dir / "phase6_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
