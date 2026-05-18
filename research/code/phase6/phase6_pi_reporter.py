#!/usr/bin/env python3
import argparse
import csv
import html
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
NUMERIC_FIELDS = {
    "tNs",
    "rank",
    "peer",
    "channel",
    "elapsedNs",
    "samples",
    "bytes",
    "posts",
    "wstalls",
    "delayMeanNs",
    "delayMaxNs",
    "delayBaselineNs",
    "delayFastNs",
    "delaySlowNs",
    "gbps",
    "gbpsBaseline",
    "gbpsFast",
    "gbpsSlow",
    "eDelay",
    "eTrend",
    "eThroughput",
    "e",
    "integral",
    "u",
    "wBefore",
    "wAfter",
    "cooldown",
    "stableLow",
    "wstallRatio",
    "baselineReady",
    "warmupEpochsSeen",
}


def parse_pairs(line: str) -> dict:
    row = {k: v for k, v in PAIR_RE.findall(line)}
    for key in NUMERIC_FIELDS:
        if key in row:
            row[key] = to_float(row[key])
    return row


def to_float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def load_ctrl_epochs(input_dir: Path) -> list[dict]:
    rows = []
    for log_path in sorted(input_dir.glob("repeat_*/*/*/nccl.*.log")):
        parts = log_path.relative_to(input_dir).parts
        if len(parts) < 4:
            continue
        repeat, mode, worker = parts[:3]
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "PHASE6 event=CTRL_EPOCH" not in line:
                    continue
                row = parse_pairs(line)
                row["repeat"] = repeat
                row["mode"] = mode
                row["worker"] = worker
                row["source"] = str(log_path)
                rows.append(row)
    rows.sort(key=lambda r: (r["repeat"], r["mode"], r["worker"], to_float(r.get("tNs"), 0.0)))
    add_relative_times(rows)
    return rows


def add_relative_times(rows: list[dict]) -> None:
    by_mode = defaultdict(list)
    by_repeat = defaultdict(list)
    for row in rows:
        by_mode[(row["repeat"], row["mode"])].append(row)
        by_repeat[row["repeat"]].append(row)

    for group in by_mode.values():
        t0 = min(finite([r.get("tNs") for r in group]) or [0.0])
        for row in group:
            row["modeRelTimeS"] = (to_float(row.get("tNs"), t0) - t0) / 1e9

    for group in by_repeat.values():
        t0 = min(finite([r.get("tNs") for r in group]) or [0.0])
        for row in group:
            row["repeatRelTimeS"] = (to_float(row.get("tNs"), t0) - t0) / 1e9


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    fields = [
        "repeat",
        "mode",
        "worker",
        "modeRelTimeS",
        "repeatRelTimeS",
        "tNs",
        "rank",
        "peer",
        "channel",
        "action",
        "elapsedNs",
        "samples",
        "bytes",
        "posts",
        "wstalls",
        "delayMeanNs",
        "delayMaxNs",
        "delayBaselineNs",
        "delayFastNs",
        "delaySlowNs",
        "gbps",
        "gbpsBaseline",
        "gbpsFast",
        "gbpsSlow",
        "eDelay",
        "eTrend",
        "eThroughput",
        "e",
        "integral",
        "u",
        "wBefore",
        "wAfter",
        "cooldown",
        "stableLow",
        "wstallRatio",
        "baselineReady",
        "warmupEpochsSeen",
        "source",
    ]
    path = output_dir / "phase6_ctrl_epoch.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def pctl(values, pct):
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for (repeat, mode), group in sorted(grouped(rows, ("repeat", "mode")).items()):
        actions = Counter(r.get("action", "unknown") for r in group)
        w_vals = finite([r.get("wAfter") for r in group])
        delay_mean_ms = [to_float(r.get("delayMeanNs")) / 1e6 for r in group]
        delay_max_ms = [to_float(r.get("delayMaxNs")) / 1e6 for r in group]
        gbps = finite([r.get("gbpsFast") for r in group])
        u_vals = finite([r.get("u") for r in group])
        summaries.append(
            {
                "repeat": repeat,
                "mode": mode,
                "epochs": len(group),
                "decrease": actions["decrease"],
                "increase": actions["increase"],
                "hold": actions["hold"],
                "cooldown": actions["cooldown"],
                "warmup": actions["warmup"] + actions["baseline_ready"],
                "w_min": min(w_vals, default=math.nan),
                "w_max": max(w_vals, default=math.nan),
                "w_final": w_vals[-1] if w_vals else math.nan,
                "delay_mean_p50_ms": pctl(delay_mean_ms, 50),
                "delay_mean_p99_ms": pctl(delay_mean_ms, 99),
                "delay_max_peak_ms": max(finite(delay_max_ms), default=math.nan),
                "gbps_fast_p50": pctl(gbps, 50),
                "gbps_fast_min": min(gbps, default=math.nan),
                "u_max": max(u_vals, default=math.nan),
            }
        )
    return summaries


def write_summary_csv(summaries: list[dict], output_dir: Path) -> Path:
    path = output_dir / "phase6_summary.csv"
    fields = list(summaries[0].keys()) if summaries else [
        "repeat",
        "mode",
        "epochs",
        "decrease",
        "increase",
        "hold",
        "cooldown",
        "warmup",
        "w_min",
        "w_max",
        "w_final",
        "delay_mean_p50_ms",
        "delay_mean_p99_ms",
        "delay_max_peak_ms",
        "gbps_fast_p50",
        "gbps_fast_min",
        "u_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    return path


def action_marks(ax, rows: list[dict], y_min=None, y_max=None):
    colors = {"decrease": "#d62728", "increase": "#2ca02c", "baseline_ready": "#9467bd"}
    for action, color in colors.items():
        xs = [r["modeRelTimeS"] for r in rows if r.get("action") == action]
        for x in xs:
            ax.axvline(x, color=color, alpha=0.18, linewidth=1)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)


def plot_repeat_panel(rows: list[dict], repeat: str, output_dir: Path) -> Path | None:
    repeat_rows = [r for r in rows if r["repeat"] == repeat]
    modes = sorted({r["mode"] for r in repeat_rows})
    if not modes:
        return None

    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=False)
    for mode in modes:
        mode_rows = sorted([r for r in repeat_rows if r["mode"] == mode], key=lambda r: r["modeRelTimeS"])
        xs = [r["modeRelTimeS"] for r in mode_rows]
        axes[0].plot(xs, [r.get("wAfter") for r in mode_rows], marker="o", markersize=2, linewidth=1, label=mode)
        axes[1].plot(xs, [to_float(r.get("delayMeanNs")) / 1e6 for r in mode_rows], linewidth=1, label=f"{mode} mean")
        axes[1].plot(xs, [to_float(r.get("delayMaxNs")) / 1e6 for r in mode_rows], linestyle="--", linewidth=1, label=f"{mode} max")
        axes[2].plot(xs, [r.get("eDelay") for r in mode_rows], linewidth=1, label=f"{mode} eDelay")
        axes[2].plot(xs, [r.get("eTrend") for r in mode_rows], linestyle="--", linewidth=1, label=f"{mode} eTrend")
        axes[2].plot(xs, [r.get("eThroughput") for r in mode_rows], linestyle=":", linewidth=1, label=f"{mode} eThroughput")
        axes[3].plot(xs, [r.get("gbpsFast") for r in mode_rows], linewidth=1, label=f"{mode} fast")
        axes[3].plot(xs, [r.get("gbpsBaseline") for r in mode_rows], linestyle="--", linewidth=1, label=f"{mode} baseline")
        axes[4].plot(xs, [r.get("u") for r in mode_rows], linewidth=1, label=f"{mode} u")
        axes[4].plot(xs, [r.get("wstallRatio") for r in mode_rows], linestyle="--", linewidth=1, label=f"{mode} wstallRatio")
        for ax in axes:
            action_marks(ax, mode_rows)

    axes[0].set_title(f"{repeat} Phase6 Controller Timeline")
    axes[0].set_ylabel("W")
    axes[1].set_ylabel("POST-DONE ms")
    axes[2].set_ylabel("Error")
    axes[3].set_ylabel("Internal Gbps")
    axes[4].set_ylabel("u / WSTALL ratio")
    axes[4].set_xlabel("Time Since Mode Start (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = output_dir / f"{repeat}_phase6_controller_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_repeat_overlay(rows: list[dict], output_dir: Path) -> list[Path]:
    paths = []
    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        repeats = sorted({r["repeat"] for r in mode_rows})
        if not repeats:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
        for repeat in repeats:
            group = sorted([r for r in mode_rows if r["repeat"] == repeat], key=lambda r: r["modeRelTimeS"])
            xs = [r["modeRelTimeS"] for r in group]
            axes[0].plot(xs, [r.get("wAfter") for r in group], linewidth=1, label=repeat)
            axes[1].plot(xs, [to_float(r.get("delayMeanNs")) / 1e6 for r in group], linewidth=1, label=repeat)
            axes[2].plot(xs, [r.get("gbpsFast") for r in group], linewidth=1, label=repeat)
        axes[0].set_title(f"{mode} Repeat Overlay")
        axes[0].set_ylabel("W")
        axes[1].set_ylabel("delayMean ms")
        axes[2].set_ylabel("gbpsFast")
        axes[2].set_xlabel("Time Since Mode Start (s)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{mode}_repeat_overlay.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_scatter(rows: list[dict], output_dir: Path) -> Path | None:
    valid = [r for r in rows if math.isfinite(to_float(r.get("wAfter"))) and math.isfinite(to_float(r.get("delayMeanNs")))]
    if not valid:
        return None
    fig, ax = plt.subplots(figsize=(12, 8))
    for mode in sorted({r["mode"] for r in valid}):
        group = [r for r in valid if r["mode"] == mode]
        ax.scatter([r["wAfter"] for r in group], [to_float(r.get("delayMeanNs")) / 1e6 for r in group], s=12, alpha=0.45, label=mode)
    ax.set_title("Phase6 W vs POST-DONE Mean Delay")
    ax.set_xlabel("W")
    ax.set_ylabel("delayMean ms")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "phase6_w_vs_delay_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_action_summary(summaries: list[dict], output_dir: Path) -> Path | None:
    if not summaries:
        return None
    labels = [f"{s['repeat']} {s['mode']}" for s in summaries]
    decrease = [s["decrease"] for s in summaries]
    increase = [s["increase"] for s in summaries]
    cooldown = [s["cooldown"] for s in summaries]
    hold = [s["hold"] for s in summaries]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.75), 7))
    x = range(len(labels))
    ax.bar(x, decrease, label="decrease")
    ax.bar(x, increase, bottom=decrease, label="increase")
    bottom = [a + b for a, b in zip(decrease, increase)]
    ax.bar(x, cooldown, bottom=bottom, label="cooldown")
    bottom = [a + b for a, b in zip(bottom, cooldown)]
    ax.bar(x, hold, bottom=bottom, label="hold")
    ax.set_title("Phase6 Action Count Summary")
    ax.set_ylabel("Epoch count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "phase6_action_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_html_report(rows: list[dict], summaries: list[dict], images: list[Path], output_dir: Path, csv_path: Path, summary_path: Path) -> Path:
    html_path = output_dir / "phase6_spike_report.html"
    if summaries:
        headers = list(summaries[0].keys())
    else:
        headers = ["repeat", "mode", "epochs"]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    row_html = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(format_cell(s.get(h)))}</td>" for h in headers) + "</tr>"
        for s in summaries
    )
    image_html = "\n".join(
        f'<section><h2>{html.escape(img.name)}</h2><img src="{html.escape(img.name)}" /></section>'
        for img in images
    )
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Phase6 Spike Controller Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202020; }}
    h1, h2 {{ margin: 16px 0 8px; }}
    img {{ width: 100%; max-width: 1900px; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    .note {{ color: #555; line-height: 1.45; max-width: 1100px; }}
  </style>
</head>
<body>
  <h1>Phase6 Spike Controller Report</h1>
  <p class="note">
    This report uses only <code>PHASE6 event=CTRL_EPOCH</code> logs. Switch/PFC plots are intentionally separated into the network reporter.
    Red vertical markers indicate W decrease epochs, green markers indicate W increase epochs, and purple markers indicate spike-detector-ready epochs.
  </p>
  <p>Raw CSV: <code>{html.escape(csv_path.name)}</code> / Summary CSV: <code>{html.escape(summary_path.name)}</code></p>
  <h2>Summary</h2>
  <table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>
  {image_html}
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def format_cell(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Create a controller-only Phase6 spike report from NCCL PHASE6 CTRL_EPOCH logs.")
    parser.add_argument("--input", required=True, type=Path, help="Experiment run root, e.g. /mnt/nfs_share/cts_experiments/phase6_pi_ddp_xxx")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for phase6 plots and CSVs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_ctrl_epochs(args.input)
    csv_path = write_csv(rows, args.output_dir)
    summaries = summarize(rows)
    summary_path = write_summary_csv(summaries, args.output_dir)

    images = []
    action_plot = plot_action_summary(summaries, args.output_dir)
    if action_plot:
        images.append(action_plot)
    scatter_plot = plot_scatter(rows, args.output_dir)
    if scatter_plot:
        images.append(scatter_plot)
    for repeat in sorted({r["repeat"] for r in rows}):
        path = plot_repeat_panel(rows, repeat, args.output_dir)
        if path:
            images.append(path)
    images.extend(plot_repeat_overlay(rows, args.output_dir))

    html_path = write_html_report(rows, summaries, images, args.output_dir, csv_path, summary_path)
    print(f"[phase6-reporter] rows={len(rows)} summaries={len(summaries)} output={html_path}")


if __name__ == "__main__":
    main()
