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


def parse_phase5_line(line: str) -> dict[str, str] | None:
    if "PHASE5 event=" not in line:
        return None
    row = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[key] = value
    return row if "event" in row else None


def load_mode_distributions(mode_dir: Path) -> dict[str, list[float]]:
    progress_delta_us: list[float] = []
    post_to_net_done_us: list[float] = []
    progress_calls_since_post: list[float] = []
    for log_path in sorted(mode_dir.glob("worker*/nccl.*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_phase5_line(line)
            if row is None:
                continue
            event = row.get("event", "")
            if event == "RECV_PROXY_PROGRESS":
                delta_ns = int(row.get("deltaNs", "0"))
                if delta_ns > 0:
                    progress_delta_us.append(delta_ns / 1000.0)
            elif event == "PROXY_RECV_NET_DONE":
                delay_ns = int(row.get("postToNetDoneNs", "0"))
                calls = int(row.get("progressCallsSincePost", "0"))
                if delay_ns > 0:
                    post_to_net_done_us.append(delay_ns / 1000.0)
                progress_calls_since_post.append(float(calls))
    return {
        "progress_delta_us": progress_delta_us,
        "post_to_net_done_us": post_to_net_done_us,
        "progress_calls_since_post": progress_calls_since_post,
    }


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    if not ordered:
        return [], []
    ys = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    return ordered, ys


def plot_ecdf(mode_values: dict[str, list[float]], title: str, xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode, values in mode_values.items():
        xs, ys = ecdf(values)
        if xs:
            ax.plot(xs, ys, label=mode, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render phase5 diagnostic plots")
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
        distributions = {mode_dir.name: load_mode_distributions(mode_dir) for mode_dir in mode_dirs}

        post_to_net_done_plot = output_dir / f"{repeat_dir.name}_post_to_net_done_ecdf.png"
        progress_delta_plot = output_dir / f"{repeat_dir.name}_recv_proxy_progress_delta_ecdf.png"
        progress_calls_plot = output_dir / f"{repeat_dir.name}_progress_calls_since_post_ecdf.png"

        plot_ecdf(
            {mode: values["post_to_net_done_us"] for mode, values in distributions.items()},
            title=f"{repeat_dir.name} POST->NET_DONE Delay",
            xlabel="Delay (us)",
            output_path=post_to_net_done_plot,
        )
        plot_ecdf(
            {mode: values["progress_delta_us"] for mode, values in distributions.items()},
            title=f"{repeat_dir.name} recvProxyProgress Delta",
            xlabel="Delta between recvProxyProgress calls (us)",
            output_path=progress_delta_plot,
        )
        plot_ecdf(
            {mode: values["progress_calls_since_post"] for mode, values in distributions.items()},
            title=f"{repeat_dir.name} recvProxyProgress calls between POST and NET_DONE",
            xlabel="Calls since POST",
            output_path=progress_calls_plot,
        )

        summary_rows = []
        for mode, values in distributions.items():
            summary_rows.append(
                {
                    "mode": mode,
                    "post_to_net_done_p50_us": percentile(values["post_to_net_done_us"], 0.50),
                    "post_to_net_done_p95_us": percentile(values["post_to_net_done_us"], 0.95),
                    "progress_delta_p50_us": percentile(values["progress_delta_us"], 0.50),
                    "progress_delta_p95_us": percentile(values["progress_delta_us"], 0.95),
                    "progress_calls_since_post_p50": percentile(values["progress_calls_since_post"], 0.50),
                    "progress_calls_since_post_p95": percentile(values["progress_calls_since_post"], 0.95),
                }
            )
        (output_dir / f"{repeat_dir.name}_phase5_report_summary.json").write_text(
            json.dumps(summary_rows, indent=2), encoding="utf-8"
        )

        html_sections.append(
            f"""
            <section>
              <h2>{repeat_dir.name}</h2>
              <img src="{post_to_net_done_plot.name}" style="max-width: 100%;"><br>
              <img src="{progress_delta_plot.name}" style="max-width: 100%;"><br>
              <img src="{progress_calls_plot.name}" style="max-width: 100%;">
            </section>
            """
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase5 Report</title></head>
<body>
  <h1>Phase5 Diagnostic Report</h1>
  <p>POST->NET_DONE delay and recvProxyProgress cadence.</p>
  {''.join(html_sections)}
</body></html>"""
    (output_dir / "phase5_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
