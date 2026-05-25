#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOWER_IS_BETTER = {
    "delay_mean_p50_ms",
    "delay_mean_p99_ms",
    "delay_max_peak_ms",
    "post_done_p99_median_ms",
    "post_done_p99_p95_ms",
    "post_done_worker_p99_max_ms",
    "pfc_delta_sum",
    "deadlock_delta_sum",
}
HIGHER_IS_BETTER = {
    "gbps_fast_p50",
    "gbps_fast_min",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase6 modes against STOCK baseline.")
    parser.add_argument("--input", required=True, type=Path, help="Report root, run root, or plots root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Explicit phase6_summary.csv")
    parser.add_argument("--bin-csv", type=Path, default=None, help="Explicit phase6_bin_metrics.csv")
    parser.add_argument("--baseline-mode", default="STOCK", help="Baseline mode name. Default: STOCK")
    return parser.parse_args()


def to_float(value, default=math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite(values) -> list[float]:
    return [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def percentile(values, pct: float) -> float:
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_csv(input_root: Path, explicit: Path | None, name: str, preferred_subdir: str) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        input_root / preferred_subdir / name,
        input_root / "plots_all" / preferred_subdir / name,
        input_root / "plots" / preferred_subdir / name,
        input_root / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(input_root.rglob(f"{preferred_subdir}/{name}"))
    if matches:
        return matches[0]
    matches = sorted(input_root.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"{name} not found under {input_root}")


def summarize_bins(rows: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        repeat = row.get("repeat", "")
        mode = row.get("mode", "")
        if repeat and mode:
            grouped[(repeat, mode)].append(row)

    out: dict[tuple[str, str], dict] = {}
    for key, group in grouped.items():
        p99 = [to_float(row.get("delay_p99_ms")) for row in group]
        worker_p99 = [to_float(row.get("delay_worker_p99_max_ms")) for row in group]
        pfc = [to_float(row.get("pfc_delta"), 0.0) for row in group]
        deadlock = [to_float(row.get("deadlock_delta"), 0.0) for row in group]
        decreases = [to_float(row.get("decrease_count"), 0.0) for row in group]
        out[key] = {
            "post_done_p99_median_ms": percentile(p99, 50),
            "post_done_p99_p95_ms": percentile(p99, 95),
            "post_done_worker_p99_max_ms": max(finite(worker_p99), default=math.nan),
            "pfc_delta_sum": sum(finite(pfc)),
            "pfc_nonzero_bins": sum(1 for value in pfc if value > 0.0),
            "deadlock_delta_sum": sum(finite(deadlock)),
            "w_decrease_events": sum(finite(decreases)),
            "analysis_bins": len(group),
        }
    return out


def merge_mode_metrics(summary_rows: list[dict], bin_rows: list[dict]) -> dict[tuple[str, str], dict]:
    merged: dict[tuple[str, str], dict] = {}
    for row in summary_rows:
        repeat = row.get("repeat", "")
        mode = row.get("mode", "")
        if not repeat or not mode:
            continue
        key = (repeat, mode)
        merged[key] = {
            "repeat": repeat,
            "mode": mode,
            "epochs": to_float(row.get("epochs"), 0.0),
            "delay_mean_p50_ms": to_float(row.get("delay_mean_p50_ms")),
            "delay_mean_p99_ms": to_float(row.get("delay_mean_p99_ms")),
            "delay_max_peak_ms": to_float(row.get("delay_max_peak_ms")),
            "gbps_fast_p50": to_float(row.get("gbps_fast_p50")),
            "gbps_fast_min": to_float(row.get("gbps_fast_min")),
            "w_min": to_float(row.get("w_min")),
            "w_max": to_float(row.get("w_max")),
            "w_final": to_float(row.get("w_final")),
            "decrease": to_float(row.get("decrease"), 0.0),
            "increase": to_float(row.get("increase"), 0.0),
        }
    for key, metrics in summarize_bins(bin_rows).items():
        merged.setdefault(key, {"repeat": key[0], "mode": key[1]}).update(metrics)
    return merged


def pct_change(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or baseline == 0:
        return math.nan
    return (value - baseline) / baseline * 100.0


def improvement_pct(metric: str, value: float, baseline: float) -> float:
    change = pct_change(value, baseline)
    if not math.isfinite(change):
        return math.nan
    if metric in LOWER_IS_BETTER:
        return -change
    if metric in HIGHER_IS_BETTER:
        return change
    return math.nan


def build_comparison_rows(metrics: dict[tuple[str, str], dict], baseline_mode: str) -> list[dict]:
    baseline_upper = baseline_mode.upper()
    by_repeat: dict[str, dict[str, dict]] = defaultdict(dict)
    for (repeat, mode), row in metrics.items():
        by_repeat[repeat][mode.upper()] = row

    fields = [
        "delay_mean_p50_ms",
        "delay_mean_p99_ms",
        "delay_max_peak_ms",
        "gbps_fast_p50",
        "gbps_fast_min",
        "post_done_p99_median_ms",
        "post_done_p99_p95_ms",
        "post_done_worker_p99_max_ms",
        "pfc_delta_sum",
        "pfc_nonzero_bins",
        "deadlock_delta_sum",
        "w_decrease_events",
    ]
    out = []
    for repeat, modes in sorted(by_repeat.items()):
        baseline = modes.get(baseline_upper)
        if not baseline:
            continue
        for mode_upper, row in sorted(modes.items()):
            if mode_upper == baseline_upper:
                continue
            item = {
                "repeat": repeat,
                "baseline_mode": baseline.get("mode", baseline_mode),
                "mode": row.get("mode", mode_upper),
            }
            for field in fields:
                value = to_float(row.get(field))
                base = to_float(baseline.get(field))
                item[field] = value
                item[f"stock_{field}"] = base
                item[f"{field}_change_pct"] = pct_change(value, base)
                item[f"{field}_improvement_pct"] = improvement_pct(field, value, base)
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def html_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return "<p>No comparison rows. Check that STOCK and P6 modes exist in the same repeat.</p>"
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(fmt(row.get(field, '')))}</td>" for field in fields) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def plot_improvement(rows: list[dict], output_dir: Path) -> list[str]:
    if not rows:
        return []
    plot_specs = [
        ("latency_improvement_pct.png", "Latency Improvement vs STOCK", ["delay_mean_p50_ms", "delay_mean_p99_ms", "post_done_p99_median_ms", "post_done_p99_p95_ms"]),
        ("throughput_improvement_pct.png", "Throughput Improvement vs STOCK", ["gbps_fast_p50", "gbps_fast_min"]),
        ("pfc_reduction_pct.png", "PFC Reduction vs STOCK", ["pfc_delta_sum", "deadlock_delta_sum"]),
    ]
    names = []
    labels = [f"{row['repeat']} {row['mode']}" for row in rows]
    x = list(range(len(rows)))
    for filename, title, metrics in plot_specs:
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.5), 5))
        width = 0.8 / max(1, len(metrics))
        for idx, metric in enumerate(metrics):
            vals = [to_float(row.get(f"{metric}_improvement_pct")) for row in rows]
            offs = [pos - 0.4 + width / 2 + idx * width for pos in x]
            ax.bar(offs, vals, width=width, label=metric)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("improvement vs STOCK (%)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        names.append(filename)
    return names


def write_html(path: Path, summary_csv: Path, bin_csv: Path, compare_csv: Path, rows: list[dict], images: list[str]) -> None:
    table_fields = [
        "repeat",
        "mode",
        "delay_mean_p99_ms_improvement_pct",
        "post_done_p99_median_ms_improvement_pct",
        "gbps_fast_p50_improvement_pct",
        "pfc_delta_sum_improvement_pct",
        "delay_mean_p99_ms",
        "stock_delay_mean_p99_ms",
        "pfc_delta_sum",
        "stock_pfc_delta_sum",
        "w_decrease_events",
    ]
    image_html = "\n".join(f"<h2>{html.escape(name)}</h2><img src='{html.escape(name)}'>" for name in images)
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase6 STOCK vs P6</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
img {{ width: 100%; max-width: 1600px; border: 1px solid #ddd; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style></head><body>
<h1>Phase6 STOCK Baseline Comparison</h1>
<p>Positive improvement means better than STOCK. For latency/PFC metrics, lower raw values become positive improvement. For throughput metrics, higher raw values become positive improvement.</p>
<p>controller summary: <code>{html.escape(str(summary_csv))}</code></p>
<p>network bins: <code>{html.escape(str(bin_csv))}</code></p>
<p>comparison csv: <code>{html.escape(str(compare_csv))}</code></p>
<h2>Key Comparison Table</h2>
{html_table(rows, table_fields)}
{image_html}
</body></html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = resolve_csv(input_root, args.summary_csv, "phase6_summary.csv", "phase6_plot_latest")
    bin_csv = resolve_csv(input_root, args.bin_csv, "phase6_bin_metrics.csv", "network_overlay_allworker_no_ecn_trimtop")
    metrics = merge_mode_metrics(read_csv(summary_csv), read_csv(bin_csv))
    rows = build_comparison_rows(metrics, args.baseline_mode)

    compare_csv = output_dir / "phase6_stock_vs_p6_comparison.csv"
    write_csv(compare_csv, rows)
    images = plot_improvement(rows, output_dir)
    html_path = output_dir / "phase6_stock_vs_p6_report.html"
    write_html(html_path, summary_csv, bin_csv, compare_csv, rows, images)
    print(f"[phase6-stock-vs-p6] rows={len(rows)} html={html_path}")


if __name__ == "__main__":
    main()
