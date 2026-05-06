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


def load_step_records(mode_dir: Path) -> list[dict]:
    metrics_path = mode_dir / f"{mode_dir.name}_step_metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return []
    t0 = min(int(r["ts_start_unix_ns"]) for r in rows)
    out = []
    for idx, row in enumerate(rows, start=1):
        start_ns = int(row["ts_start_unix_ns"])
        out.append(
            {
                "step_plot_index": idx,
                "warmup": bool(row.get("warmup", False)),
                "elapsed_start_ms": (start_ns - t0) / 1e6,
            }
        )
    return out


def build_rate_series(samples: list[tuple[int, float]], bin_ms: float = 1.0) -> list[tuple[float, float]]:
    if not samples:
        return []
    t0 = min(t for t, _ in samples)
    t1 = max(t for t, _ in samples)
    if t1 <= t0:
        return [(0.0, sum(v for _, v in samples) / max(bin_ms, 1e-6))]
    bin_ns = max(int(bin_ms * 1e6), 1)
    nbins = ((t1 - t0) // bin_ns) + 1
    totals = [0.0] * nbins
    for t, v in samples:
      idx = (t - t0) // bin_ns
      totals[int(idx)] += v
    return [((i + 0.5) * bin_ms, totals[i] / bin_ms) for i in range(nbins)]


def load_mode_series(mode_dir: Path) -> dict[str, object]:
    worker_dirs = sorted(p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("worker"))
    chosen_worker = next((p for p in worker_dirs if p.name == "worker01"), worker_dirs[0] if worker_dirs else None)
    if chosen_worker is None:
        return {"worker": "", "post_rate": []}
    samples: list[tuple[int, float]] = []
    for log_path in sorted(chosen_worker.glob("nccl.*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            row = parse_kv_line(line, "PHASE9 event=")
            if row is None or row.get("event") != "RECVCOMM_POST_RATE":
                continue
            t_ns = int(row.get("tNs", "0"))
            post_cost = float(row.get("postCost", "0"))
            if t_ns > 0:
                samples.append((t_ns, post_cost))
    return {"worker": chosen_worker.name, "post_rate": build_rate_series(samples, bin_ms=1.0)}


def plot_repeat_overlay(repeat_name: str, mode_series: dict[str, dict[str, object]], step_records_by_mode: dict[str, list[dict]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    global_ymax = 0.0
    for mode, payload in mode_series.items():
        series = payload["post_rate"]
        if not series:
            continue
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        global_ymax = max(global_ymax, max(ys))
        ax.plot(xs, ys, linewidth=2, label=mode)

    marker_y = global_ymax * 1.03 if global_ymax > 0 else 1.0
    for mode, records in step_records_by_mode.items():
        warmup_xs = [r["elapsed_start_ms"] for r in records if r["warmup"]]
        measure_xs = [r["elapsed_start_ms"] for r in records if not r["warmup"]]
        if warmup_xs:
            ax.scatter(warmup_xs, [marker_y] * len(warmup_xs), marker="x", s=30, alpha=0.8, label=f"{mode} warmup start")
        if measure_xs:
            ax.scatter(measure_xs, [marker_y] * len(measure_xs), marker="o", s=22, alpha=0.7, facecolors="none", label=f"{mode} measured start")

    ax.set_title(f"{repeat_name} recvComm POST rate overlay")
    ax.set_xlabel("Relative mode time (ms)")
    ax.set_ylabel("POST cost / ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render phase9 post-rate plots")
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
        mode_series = {mode_dir.name: load_mode_series(mode_dir) for mode_dir in mode_dirs}
        step_records = {mode_dir.name: load_step_records(mode_dir) for mode_dir in mode_dirs}

        plot_path = output_dir / f"{repeat_dir.name}_post_rate_overlay.png"
        plot_repeat_overlay(repeat_dir.name, mode_series, step_records, plot_path)

        html_sections.append(
            f"""
            <section>
              <h2>{repeat_dir.name}</h2>
              <p>Warmup step starts use <code>x</code>; measured step starts use <code>o</code>.</p>
              <img src="{plot_path.name}" style="max-width: 100%;">
            </section>
            """
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase9 Report</title></head>
<body>
  <h1>Phase9 Online recvComm POST Rate Report</h1>
  <p>STOCK-only observation run. POST rate is tracked per recvComm-group POST issuance on worker01 and plotted over relative mode time.</p>
  {''.join(html_sections)}
</body></html>"""
    (output_dir / "phase9_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
