#!/usr/bin/env python3
import argparse
import csv
import html
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_pairs(line: str) -> dict:
    return {k: v for k, v in PAIR_RE.findall(line)}


def to_float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    rows.sort(key=lambda r: (r["repeat"], r["mode"], to_float(r.get("tNs"), 0), r["worker"]))
    return rows


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    fields = [
        "repeat",
        "mode",
        "worker",
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


def group_rows(rows: list[dict]):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["repeat"], row["mode"]), []).append(row)
    return grouped


def rel_time_s(rows: list[dict]) -> list[float]:
    t0 = min(to_float(r.get("tNs"), 0) for r in rows)
    return [(to_float(r.get("tNs"), 0) - t0) / 1e9 for r in rows]


def plot_repeat(rows: list[dict], repeat: str, output_dir: Path) -> list[Path]:
    paths = []
    repeat_rows = [r for r in rows if r["repeat"] == repeat]
    modes = sorted({r["mode"] for r in repeat_rows})
    if not modes:
        return paths

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
    for mode in modes:
        mode_rows = [r for r in repeat_rows if r["mode"] == mode]
        xs = rel_time_s(mode_rows)
        axes[0].plot(xs, [to_float(r.get("wAfter")) for r in mode_rows], marker="o", markersize=2, linewidth=1, label=mode)
        axes[1].plot(xs, [to_float(r.get("delayMeanNs")) / 1e6 for r in mode_rows], linewidth=1, label=f"{mode} mean")
        axes[1].plot(xs, [to_float(r.get("delayMaxNs")) / 1e6 for r in mode_rows], linestyle="--", linewidth=1, label=f"{mode} max")
        axes[2].plot(xs, [to_float(r.get("gbpsFast")) for r in mode_rows], linewidth=1, label=f"{mode} fast")
        axes[2].plot(xs, [to_float(r.get("gbpsBaseline")) for r in mode_rows], linestyle="--", linewidth=1, label=f"{mode} baseline")
        axes[3].plot(xs, [to_float(r.get("u")) for r in mode_rows], linewidth=1, label=f"{mode} u")
        dec_x = [x for x, r in zip(xs, mode_rows) if r.get("action") == "decrease"]
        inc_x = [x for x, r in zip(xs, mode_rows) if r.get("action") == "increase"]
        if dec_x:
            axes[0].scatter(dec_x, [to_float(r.get("wAfter")) for x, r in zip(xs, mode_rows) if r.get("action") == "decrease"], marker="v", s=40)
        if inc_x:
            axes[0].scatter(inc_x, [to_float(r.get("wAfter")) for x, r in zip(xs, mode_rows) if r.get("action") == "increase"], marker="^", s=40)

    axes[0].set_ylabel("W")
    axes[0].set_title(f"{repeat} Phase6 W Timeline")
    axes[1].set_ylabel("POST-DONE ms")
    axes[2].set_ylabel("Internal Gbps")
    axes[3].set_ylabel("PI output u")
    axes[3].set_xlabel("Time Since First Phase6 Epoch in Mode (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / f"{repeat}_phase6_ctrl_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    paths.append(out)
    return paths


def write_html(rows: list[dict], images: list[Path], output_dir: Path, csv_path: Path) -> Path:
    html_path = output_dir / "phase6_pi_report.html"
    summary = []
    for (repeat, mode), mode_rows in group_rows(rows).items():
        decreases = sum(1 for r in mode_rows if r.get("action") == "decrease")
        increases = sum(1 for r in mode_rows if r.get("action") == "increase")
        w_values = [to_float(r.get("wAfter")) for r in mode_rows if not math.isnan(to_float(r.get("wAfter")))]
        delay_values = [to_float(r.get("delayMeanNs")) / 1e6 for r in mode_rows if not math.isnan(to_float(r.get("delayMeanNs")))]
        summary.append((repeat, mode, len(mode_rows), decreases, increases, min(w_values, default=math.nan), max(w_values, default=math.nan), max(delay_values, default=math.nan)))

    rows_html = "\n".join(
        f"<tr><td>{html.escape(repeat)}</td><td>{html.escape(mode)}</td><td>{count}</td><td>{dec}</td><td>{inc}</td><td>{wmin:.3f}</td><td>{wmax:.3f}</td><td>{dmax:.3f}</td></tr>"
        for repeat, mode, count, dec, inc, wmin, wmax, dmax in summary
    )
    images_html = "\n".join(f'<h2>{html.escape(img.name)}</h2><img src="{html.escape(img.name)}" />' for img in images)
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Phase6 PI Runtime Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    img {{ width: 100%; max-width: 1800px; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Phase6 PI Runtime Report</h1>
  <p>CSV: {html.escape(csv_path.name)}</p>
  <table>
    <thead><tr><th>repeat</th><th>mode</th><th>epochs</th><th>W decrease</th><th>W increase</th><th>W min</th><th>W max</th><th>max delay mean ms</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  {images_html}
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_ctrl_epochs(args.input)
    csv_path = write_csv(rows, args.output_dir)
    images = []
    for repeat in sorted({r["repeat"] for r in rows}):
        images.extend(plot_repeat(rows, repeat, args.output_dir))
    html_path = write_html(rows, images, args.output_dir, csv_path)
    print(f"[phase6-reporter] rows={len(rows)} csv={csv_path} html={html_path}")


if __name__ == "__main__":
    main()
