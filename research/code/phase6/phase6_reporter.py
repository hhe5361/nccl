#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    post_times_ns: list[int] = []
    allow_series: list[tuple[int, float]] = []
    stall_times_ns: list[int] = []

    for log_path in sorted(mode_dir.glob("worker*/nccl.*.log")):
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
        "cumulative_posts": cumulative_posts,
        "cumulative_allow": cumulative_allow,
        "cumulative_stalls": cumulative_stalls,
    }


def plot_overlay(mode_series: dict[str, list[tuple[float, float]]], title: str, ylabel: str, output_path: Path) -> None:
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
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
        data = {mode_dir.name: load_mode_series(mode_dir) for mode_dir in mode_dirs}

        posts_plot = output_dir / f"{repeat_dir.name}_cumulative_posts_overlay.png"
        allow_plot = output_dir / f"{repeat_dir.name}_cumulative_rate_allow_overlay.png"
        stalls_plot = output_dir / f"{repeat_dir.name}_cumulative_rate_stall_overlay.png"

        plot_overlay(
            {mode: values["cumulative_posts"] for mode, values in data.items()},
            title=f"{repeat_dir.name} cumulative recv POST count",
            ylabel="Cumulative POST count",
            output_path=posts_plot,
        )
        plot_overlay(
            {mode: values["cumulative_allow"] for mode, values in data.items()},
            title=f"{repeat_dir.name} cumulative rate-allow cost",
            ylabel="Cumulative allowed post cost",
            output_path=allow_plot,
        )
        plot_overlay(
            {mode: values["cumulative_stalls"] for mode, values in data.items()},
            title=f"{repeat_dir.name} cumulative rate-stall events",
            ylabel="Cumulative stall events",
            output_path=stalls_plot,
        )

        summary_rows = []
        for mode_dir in mode_dirs:
            summary_path = mode_dir / f"{mode_dir.name}_summary.json"
            perf_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            summary_rows.append(
                {
                    "mode": mode_dir.name,
                    "step_ms_avg": perf_summary.get("step_ms_avg", 0),
                    "steps_per_sec_avg": perf_summary.get("steps_per_sec_avg", 0),
                    "samples_per_sec_avg": perf_summary.get("samples_per_sec_avg", 0),
                    "post_count": len(data[mode_dir.name]["cumulative_posts"]),
                    "allow_count": len(data[mode_dir.name]["cumulative_allow"]),
                    "stall_count": len(data[mode_dir.name]["cumulative_stalls"]),
                }
            )
        (output_dir / f"{repeat_dir.name}_phase6_report_summary.json").write_text(
            json.dumps(summary_rows, indent=2), encoding="utf-8"
        )

        html_sections.append(
            f"""
            <section>
              <h2>{repeat_dir.name}</h2>
              <img src="{posts_plot.name}" style="max-width: 100%;"><br>
              <img src="{allow_plot.name}" style="max-width: 100%;"><br>
              <img src="{stalls_plot.name}" style="max-width: 100%;">
            </section>
            """
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase6 Report</title></head>
<body>
  <h1>Phase6 Post-rate Control Report</h1>
  <p>Relative-time overlay of actual recv POSTs, allowed post cost, and stall events.</p>
  {''.join(html_sections)}
</body></html>"""
    (output_dir / "phase6_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
