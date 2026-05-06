#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
PHASE3_REPORTER_PATH = REPO_ROOT / "research" / "code" / "phase3" / "phase3_log_reporter.py"
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


@dataclass
class ModeMetrics:
    mode: str
    load_pct: int
    w: Optional[int]
    label: str
    latency_ms: float
    throughput_sps: float
    post_to_net_done_p50_us: float
    post_to_net_done_p95_us: float
    occpr_p50: float
    occpr_p95: float
    switch_pfc_total: float
    switch_pfc_peak_rate: float
    start_ns: int
    end_ns: int
    occpr_series: list[tuple[float, float]]


def percentile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
      return 0.0
    if len(vals) == 1:
      return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def parse_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV_RE.finditer(line)}


def parse_mode_name(mode_name: str) -> tuple[int, Optional[int], str]:
    upper = mode_name.upper()
    m = re.match(r"^STOCK_L(\d+)$", upper)
    if m:
        load = int(m.group(1))
        return load, None, f"STOCK"
    m = re.match(r"^B4_W(\d+)_L(\d+)$", upper)
    if m:
        w = int(m.group(1))
        load = int(m.group(2))
        return load, w, f"W={w}"
    raise ValueError(f"unsupported mode name: {mode_name}")


def load_step_summary(mode_dir: Path) -> tuple[float, float, int, int]:
    summary_path = sorted(mode_dir.glob("*_summary.json"))[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    latency_ms = float(summary.get("step_ms_avg", 0.0))
    throughput_sps = float(summary.get("samples_per_sec_avg", 0.0))
    step_rows = [json.loads(line) for line in sorted(mode_dir.glob("*_step_metrics.jsonl"))[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    start_ns = min(int(row["ts_start_unix_ns"]) for row in step_rows)
    end_ns = max(int(row["ts_end_unix_ns"]) for row in step_rows)
    return latency_ms, throughput_sps, start_ns, end_ns


def load_phase5_metrics(mode_dir: Path) -> tuple[float, float, float, float, list[tuple[float, float]]]:
    log_paths = sorted(mode_dir.glob("worker01/nccl.*.log"))
    delays_us: list[float] = []
    occpr_vals: list[float] = []
    occpr_series: list[tuple[float, float]] = []
    base_ts = None
    for path in log_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "PHASE5 event=PROXY_RECV_" not in line:
                continue
            row = parse_kv(line)
            event = row.get("event", "")
            ts_ns = int(row.get("tNs", "0"))
            occpr = float(row.get("occPr", "0"))
            if base_ts is None:
                base_ts = ts_ns
            rel_ms = (ts_ns - base_ts) / 1_000_000.0 if base_ts else 0.0
            if event == "PROXY_RECV_NET_DONE":
                delay_ns = int(row.get("postToNetDoneNs", "0"))
                if delay_ns > 0:
                    delays_us.append(delay_ns / 1000.0)
            if event in {"PROXY_RECV_POST", "PROXY_RECV_NET_DONE", "PROXY_RECV_VISIBLE", "PROXY_RECV_CONSUMED"}:
                occpr_vals.append(occpr)
                occpr_series.append((rel_ms, occpr))
    return (
        percentile(delays_us, 0.50),
        percentile(delays_us, 0.95),
        percentile(occpr_vals, 0.50),
        percentile(occpr_vals, 0.95),
        occpr_series,
    )


def load_phase3_helper():
    spec = importlib.util.spec_from_file_location("phase3_log_reporter_base_for_phase10", PHASE3_REPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load phase3 helper from {PHASE3_REPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_switch_metrics(repeat_root: Path, start_ns: int, end_ns: int) -> tuple[float, float]:
    helper = load_phase3_helper()
    bundle = helper.load_switch_bundle(repeat_root, None, None)
    if not bundle:
        return 0.0, 0.0
    total = 0.0
    peak = 0.0
    for label, snapshots in bundle.get("snapshots", {}).items():
        start_value = helper.interpolate_switch_snapshot_value(snapshots, start_ns)
        end_value = helper.interpolate_switch_snapshot_value(snapshots, end_ns)
        if start_value is not None and end_value is not None:
            total += max(0.0, float(end_value) - float(start_value))
        rates = [rate for ts_ns, rate in bundle.get("series", {}).get(label, []) if start_ns <= ts_ns <= end_ns]
        if rates:
            peak = max(peak, max(rates))
    return total, peak


def load_mode_metrics(repeat_root: Path, mode_dir: Path) -> ModeMetrics:
    load_pct, w, label = parse_mode_name(mode_dir.name)
    latency_ms, throughput_sps, start_ns, end_ns = load_step_summary(mode_dir)
    p50_us, p95_us, occ50, occ95, occ_series = load_phase5_metrics(mode_dir)
    switch_total, switch_peak = compute_switch_metrics(repeat_root, start_ns, end_ns)
    return ModeMetrics(
        mode=mode_dir.name,
        load_pct=load_pct,
        w=w,
        label=label,
        latency_ms=latency_ms,
        throughput_sps=throughput_sps,
        post_to_net_done_p50_us=p50_us,
        post_to_net_done_p95_us=p95_us,
        occpr_p50=occ50,
        occpr_p95=occ95,
        switch_pfc_total=switch_total,
        switch_pfc_peak_rate=switch_peak,
        start_ns=start_ns,
        end_ns=end_ns,
        occpr_series=occ_series,
    )


def save_heatmap(path: Path, rows: list[ModeMetrics], metric_name: str, title: str, cbar_label: str) -> Optional[Path]:
    loads = sorted({row.load_pct for row in rows})
    w_values = [None, 2, 4, 6, 8]
    matrix = []
    for load in loads:
        row_vals = []
        for w in w_values:
            match = next((item for item in rows if item.load_pct == load and item.w == w), None)
            row_vals.append(float(getattr(match, metric_name)) if match else math.nan)
        matrix.append(row_vals)
    fig, ax = plt.subplots(figsize=(8.2, 3.8 + 0.35 * len(loads)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(w_values)))
    ax.set_xticklabels(["STOCK", "W2", "W4", "W6", "W8"])
    ax.set_yticks(range(len(loads)))
    ax.set_yticklabels([f"L{load}" for load in loads])
    ax.set_xlabel("Mode")
    ax.set_ylabel("Background load")
    ax.set_title(title)
    for y, load in enumerate(loads):
        for x, _w in enumerate(w_values):
            value = matrix[y][x]
            label = "NA" if math.isnan(value) else f"{value:.1f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_occpr_overlay(path: Path, rows: list[ModeMetrics], load_pct: int) -> Optional[Path]:
    subset = [row for row in rows if row.load_pct == load_pct and row.occpr_series]
    if not subset:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for row in sorted(subset, key=lambda item: (-1 if item.w is None else item.w)):
        xs = [x for x, _ in row.occpr_series]
        ys = [y for _, y in row.occpr_series]
        ax.plot(xs, ys, linewidth=1.1, alpha=0.9, label=row.label)
    ax.set_title(f"Load {load_pct}% occPr lifecycle overlay")
    ax.set_xlabel("Relative NCCL event time (ms)")
    ax.set_ylabel("occPr = posted - received")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_html(output_dir: Path, repeat_cards: list[dict]) -> Path:
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>phase10 report</title>",
        "<style>body{font:15px/1.5 sans-serif;background:#f6f8fb;color:#142033;margin:0} .wrap{width:min(1280px,calc(100%-40px));margin:0 auto;padding:24px 0 48px} .card{background:#fff;border:1px solid #dbe3ee;border-radius:16px;padding:16px 18px;margin:14px 0} img{max-width:100%;border:1px solid #dbe3ee;border-radius:12px} code{background:#eef2f7;padding:2px 5px;border-radius:6px}</style></head><body><div class='wrap'>",
        "<div class='card'><h1>Phase10 Report</h1><p>phase4 posted-received&lt;W under background RDMA load and switch logging.</p></div>",
    ]
    for card in repeat_cards:
        parts.append(f"<div class='card'><h2>{card['repeat']}</h2>")
        for path, caption in card["images"]:
            if path is None:
                continue
            rel = path.relative_to(output_dir).as_posix()
            parts.append(f"<h3>{caption}</h3><img src='{rel}' alt='{caption}'>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    html_path = output_dir / "phase10_report.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render phase10 report")
    parser.add_argument("--input", required=True, help="phase10 run root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repeat_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("repeat_")])
    report = {"run_root": str(run_root), "repeats": []}
    repeat_cards = []

    for repeat_dir in repeat_dirs:
      rows = [load_mode_metrics(repeat_dir, mode_dir) for mode_dir in sorted(p for p in repeat_dir.iterdir() if p.is_dir() and not p.name.startswith("."))]
      repeat_output = output_dir / repeat_dir.name
      repeat_output.mkdir(parents=True, exist_ok=True)
      images = [
          (save_heatmap(repeat_output / "latency_heatmap.png", rows, "latency_ms", f"{repeat_dir.name} latency heatmap", "step_ms_avg"), "Latency heatmap"),
          (save_heatmap(repeat_output / "throughput_heatmap.png", rows, "throughput_sps", f"{repeat_dir.name} throughput heatmap", "samples_per_sec_avg"), "Throughput heatmap"),
          (save_heatmap(repeat_output / "post_to_net_done_p95_heatmap.png", rows, "post_to_net_done_p95_us", f"{repeat_dir.name} POST->NET_DONE p95 heatmap", "usec"), "POST->NET_DONE p95 heatmap"),
          (save_heatmap(repeat_output / "occpr_p95_heatmap.png", rows, "occpr_p95", f"{repeat_dir.name} occPr p95 heatmap", "occPr"), "occPr p95 heatmap"),
          (save_heatmap(repeat_output / "switch_pfc_total_heatmap.png", rows, "switch_pfc_total", f"{repeat_dir.name} switch PFC total heatmap", "delta"), "Switch PFC total heatmap"),
      ]
      for load_pct in sorted({row.load_pct for row in rows}):
          images.append((save_occpr_overlay(repeat_output / f"load_{load_pct}_occpr_overlay.png", rows, load_pct), f"Load {load_pct}% occPr overlay"))
      report["repeats"].append({
          "repeat": repeat_dir.name,
          "modes": [row.__dict__ for row in rows],
      })
      repeat_cards.append({"repeat": repeat_dir.name, "images": images})

    (output_dir / "phase10_report_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    html_path = build_html(output_dir, repeat_cards)
    print(f"[phase10-reporter] done html={html_path}")


if __name__ == "__main__":
    main()
