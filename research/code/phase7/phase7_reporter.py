#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable


NS_PER_SEC = 1_000_000_000


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 DDP backward timeline and switch congestion reporter")
    parser.add_argument("--input", required=True, help="Phase7 experiment run directory")
    parser.add_argument("--output-dir", required=True, help="Report output directory")
    parser.add_argument("--switch-log-dir", default="", help="Optional switch_congestion_logger_v2 run directory")
    parser.add_argument("--bucket-ms", type=float, default=1000.0, help="Switch metric bucket size in ms")
    parser.add_argument("--include-warmup", action="store_true", help="Include warmup steps in duration/throughput plots")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            index = 0
            while index < len(line):
                while index < len(line) and line[index].isspace():
                    index += 1
                if index >= len(line):
                    break
                try:
                    record, next_index = decoder.raw_decode(line, index)
                except json.JSONDecodeError as exc:
                    print(f"[phase7-reporter] skip malformed JSONL {path}:{line_no}: {exc}", file=sys.stderr)
                    break
                if isinstance(record, dict):
                    rows.append(record)
                index = next_index
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def resolve_switch_log_dir(input_dir: Path, explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    env = parse_env_file(input_dir / "switch_logger.env")
    candidates: list[Path] = []
    for key in ["SWITCH_LOG_LOCAL_DIR", "SWITCH_LOG_DIR"]:
        value = env.get(key)
        if value:
            candidates.append(Path(value))

    run_id = env.get("SWITCH_LOG_RUN_ID")
    if run_id:
        candidates.extend(
            [
                input_dir / "switch_log" / run_id,
                input_dir.parent / "switch_log" / run_id,
                input_dir.parent / "switch_log" / "switch_log" / run_id,
                Path("/mnt/nfs_share/cts_experiments/switch_log") / run_id,
                Path("/mnt/c/Users/hyoeun/Desktop/research/result/switch_log/switch_log") / run_id,
                Path("/mnt/c/Users/hyoeun/Desktop/exp_data/switch_log/switch_log") / run_id,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_timeline_rows(input_dir: Path, include_warmup: bool) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("repeat_*/*/worker*/*_phase7_backward_timeline.jsonl")):
        repeat = path.parents[2].name
        mode = path.parents[1].name
        worker = path.parent.name
        for row in read_jsonl(path):
            if not include_warmup and row.get("warmup"):
                continue
            row = dict(row)
            row.setdefault("repeat_label", repeat)
            row.setdefault("mode", mode)
            row.setdefault("worker", worker)
            row["source_path"] = str(path)
            rows.append(row)
    return rows


def load_step_metrics(input_dir: Path, include_warmup: bool) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("repeat_*/*/*_phase7_step_metrics.jsonl")):
        repeat = path.parents[1].name
        mode = path.parent.name
        for row in read_jsonl(path):
            if not include_warmup and row.get("warmup"):
                continue
            row = dict(row)
            row.setdefault("repeat_label", repeat)
            row.setdefault("mode", mode)
            row["source_path"] = str(path)
            rows.append(row)
    return rows


def marker_mode_from_message(message: str) -> str:
    match = re.search(r"\bmode=([^ ]+)", message)
    return match.group(1) if match else ""


def load_markers(switch_dir: Path | None) -> list[dict]:
    if switch_dir is None:
        return []
    rows = []
    for row in read_jsonl(switch_dir / "markers.jsonl"):
        ts = row.get("ts_unix_ns")
        if ts is None:
            continue
        row = dict(row)
        row["ts_unix_ns"] = int(ts)
        row["mode_from_message"] = marker_mode_from_message(str(row.get("message", "")))
        rows.append(row)
    return sorted(rows, key=lambda item: item["ts_unix_ns"])


def numeric_value(row: dict, keys: Iterable[str]) -> int | None:
    total = 0
    found = False
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
        found = True
    return total if found else None


def cumulative_series_from_aggregate(
    switch_dir: Path,
    files: Iterable[str],
    keys: Iterable[str],
    label: str,
) -> list[dict]:
    series: list[dict] = []
    for name in files:
        for row in read_jsonl(switch_dir / name):
            ts = row.get("ts_mid_unix_ns") or row.get("ts_end_unix_ns") or row.get("ts_start_unix_ns")
            value = numeric_value(row, keys)
            if ts is None or value is None:
                continue
            series.append(
                {
                    "ts_unix_ns": int(ts),
                    "series": label,
                    "switch": row.get("switch", name.split("_")[0]),
                    "value": int(value),
                }
            )
    return sorted(series, key=lambda item: item["ts_unix_ns"])


def cumulative_series_from_raw(
    switch_dir: Path,
    files: Iterable[str],
    keys: Iterable[str],
    label: str,
) -> list[dict]:
    by_sample: DefaultDict[tuple[str, int], int] = defaultdict(int)
    ts_by_sample: dict[tuple[str, int], int] = {}
    for name in files:
        for row in read_jsonl(switch_dir / name):
            ts = row.get("ts_mid_unix_ns") or row.get("ts_end_unix_ns") or row.get("ts_start_unix_ns")
            sample = row.get("sample_id")
            switch = str(row.get("switch", name.split("_")[0]))
            value = numeric_value(row, keys)
            if ts is None or sample is None or value is None:
                continue
            key = (switch, int(sample))
            by_sample[key] += int(value)
            ts_by_sample[key] = max(int(ts), ts_by_sample.get(key, 0))

    series = [
        {"ts_unix_ns": ts_by_sample[key], "series": label, "switch": key[0], "value": value}
        for key, value in by_sample.items()
    ]
    return sorted(series, key=lambda item: item["ts_unix_ns"])


def load_pfc_cumulative(switch_dir: Path | None) -> list[dict]:
    if switch_dir is None:
        return []
    rows: list[dict] = []
    rows.extend(
        cumulative_series_from_aggregate(
            switch_dir,
            ["rackA_pfc_aggregate.jsonl", "rackB_pfc_aggregate.jsonl"],
            ["rx_pause_total", "tx_pause_total"],
            "rack_pfc_pause",
        )
    )
    rows.extend(
        cumulative_series_from_aggregate(
            switch_dir,
            ["spine_roce_aggregate.jsonl", "spine_pfc_ecn_aggregate.jsonl"],
            ["rx_pause_packets_total", "tx_pause_packets_total"],
            "spine_pfc_pause",
        )
    )
    if rows:
        return sorted(rows, key=lambda item: item["ts_unix_ns"])
    rows.extend(
        cumulative_series_from_raw(
            switch_dir,
            ["rackA_pfc_statistics.jsonl", "rackB_pfc_statistics.jsonl"],
            ["rx_pause", "tx_pause"],
            "rack_pfc_pause",
        )
    )
    rows.extend(
        cumulative_series_from_raw(
            switch_dir,
            ["spine_roce_counters.jsonl", "spine_pfc_ecn.jsonl"],
            ["rx_pause_packets", "tx_pause_packets"],
            "spine_pfc_pause",
        )
    )
    return sorted(rows, key=lambda item: item["ts_unix_ns"])


def load_deadlock_cumulative(switch_dir: Path | None) -> list[dict]:
    if switch_dir is None:
        return []
    files = sorted(switch_dir.glob("*deadlock*.jsonl"))
    rows: list[dict] = []
    for path in files:
        for row in read_jsonl(path):
            ts = row.get("ts_mid_unix_ns") or row.get("ts_end_unix_ns") or row.get("ts_start_unix_ns")
            if ts is None:
                continue
            value = numeric_value(row, ["deadlock_count_total", "deadlock_count", "packet_dispose_total", "packet_dispose"])
            if value is None:
                continue
            rows.append(
                {
                    "ts_unix_ns": int(ts),
                    "series": "pfc_deadlock",
                    "switch": row.get("switch", path.name.split("_")[0]),
                    "value": int(value),
                }
            )
    return sorted(rows, key=lambda item: item["ts_unix_ns"])


def delta_rows(cumulative: list[dict]) -> list[dict]:
    previous: dict[tuple[str, str], int] = {}
    out: list[dict] = []
    for row in sorted(cumulative, key=lambda item: item["ts_unix_ns"]):
        key = (str(row.get("series", "")), str(row.get("switch", "")))
        value = int(row.get("value", 0))
        if key not in previous:
            delta = 0
        else:
            delta = max(0, value - previous[key])
        previous[key] = value
        new_row = dict(row)
        new_row["delta"] = delta
        out.append(new_row)
    return out


def bucket_deltas(rows: list[dict], base_ns: int, bucket_ms: float, label: str) -> list[dict]:
    bucket_ns = max(int(bucket_ms * 1_000_000), 1)
    buckets: DefaultDict[int, int] = defaultdict(int)
    for row in rows:
        ts = int(row["ts_unix_ns"])
        bucket = ((ts - base_ns) // bucket_ns) * bucket_ns
        buckets[int(bucket)] += int(row.get("delta", 0))
    return [
        {
            "rel_sec": bucket / NS_PER_SEC,
            "metric": label,
            "delta": value,
            "bucket_ms": bucket_ms,
        }
        for bucket, value in sorted(buckets.items())
    ]


def choose_base_ns(timeline_rows: list[dict], metric_rows: list[dict], markers: list[dict]) -> int:
    candidates: list[int] = []
    candidates.extend(int(row["backward_start_unix_ns"]) for row in timeline_rows if row.get("backward_start_unix_ns"))
    candidates.extend(int(row["ts_mid_unix_ns"]) for row in metric_rows if row.get("ts_mid_unix_ns"))
    candidates.extend(int(row["ts_unix_ns"]) for row in markers if row.get("marker") == "mode_start")
    if not candidates:
        candidates.extend(int(row["ts_unix_ns"]) for row in markers if row.get("ts_unix_ns"))
    if not candidates:
        return 0
    return min(candidates)


def rel_sec(ts_ns: int, base_ns: int) -> float:
    return (int(ts_ns) - base_ns) / NS_PER_SEC


def write_timeline_csv(output_dir: Path, timeline_rows: list[dict], base_ns: int) -> Path:
    rows: list[dict] = []
    for row in timeline_rows:
        rows.append(
            {
                "repeat": row.get("repeat_label", ""),
                "mode": row.get("mode", row.get("tag", "")),
                "worker": row.get("worker", ""),
                "rank": row.get("rank", ""),
                "step": row.get("step", ""),
                "warmup": row.get("warmup", ""),
                "backward_start_sec": rel_sec(int(row["backward_start_unix_ns"]), base_ns),
                "backward_return_sec": rel_sec(int(row["backward_return_unix_ns"]), base_ns),
                "backward_comm_complete_sec": rel_sec(int(row["backward_comm_complete_unix_ns"]), base_ns),
                "backward_call_ms": row.get("local_backward_call_ms", ""),
                "backward_sync_ms": row.get("local_backward_sync_ms", ""),
                "backward_total_ms": row.get("local_backward_total_ms", ""),
                "step_ms": row.get("local_step_ms", ""),
            }
        )
    path = output_dir / "phase7_backward_timeline.csv"
    write_csv(
        path,
        rows,
        [
            "repeat",
            "mode",
            "worker",
            "rank",
            "step",
            "warmup",
            "backward_start_sec",
            "backward_return_sec",
            "backward_comm_complete_sec",
            "backward_call_ms",
            "backward_sync_ms",
            "backward_total_ms",
            "step_ms",
        ],
    )
    return path


def summarize(timeline_rows: list[dict], step_rows: list[dict], pfc_delta: list[dict], deadlock_delta: list[dict]) -> list[dict]:
    by_group: DefaultDict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "steps": set(),
            "workers": set(),
            "backward_total": [],
            "backward_sync": [],
            "throughput": [],
        }
    )
    for row in timeline_rows:
        key = (str(row.get("repeat_label", "")), str(row.get("mode", row.get("tag", ""))))
        by_group[key]["steps"].add(int(row.get("step", -1)))
        by_group[key]["workers"].add(str(row.get("worker", "")))
        by_group[key]["backward_total"].append(float(row.get("local_backward_total_ms", 0.0)))
        by_group[key]["backward_sync"].append(float(row.get("local_backward_sync_ms", 0.0)))
    for row in step_rows:
        key = (str(row.get("repeat_label", "")), str(row.get("mode", row.get("tag", ""))))
        by_group[key]["throughput"].append(float(row.get("samples_per_sec", 0.0)))

    total_pfc = sum(int(row.get("delta", 0)) for row in pfc_delta)
    total_deadlock = sum(int(row.get("delta", 0)) for row in deadlock_delta)
    summary: list[dict] = []
    for (repeat, mode), values in sorted(by_group.items()):
        bwd = values["backward_total"]
        sync = values["backward_sync"]
        thr = values["throughput"]
        summary.append(
            {
                "repeat": repeat,
                "mode": mode,
                "steps": len(values["steps"]),
                "workers": len(values["workers"]),
                "backward_ms_mean": mean(bwd),
                "backward_ms_p95": percentile(bwd, 0.95),
                "backward_sync_ms_mean": mean(sync),
                "backward_sync_ms_p95": percentile(sync, 0.95),
                "samples_per_sec_mean": mean(thr),
                "samples_per_sec_p50": percentile(thr, 0.50),
                "total_switch_pfc_delta": total_pfc,
                "total_switch_deadlock_delta": total_deadlock,
            }
        )
    return summary


def plot_reports(
    output_dir: Path,
    timeline_rows: list[dict],
    step_rows: list[dict],
    markers: list[dict],
    pfc_bucket: list[dict],
    deadlock_bucket: list[dict],
    base_ns: int,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for phase7 reporter") from exc

    plots: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if timeline_rows:
        fig, ax = plt.subplots(figsize=(16, 9))
        groups = sorted({(str(r.get("repeat_label", "")), str(r.get("mode", r.get("tag", ""))), str(r.get("worker", "")), int(r.get("rank", 0))) for r in timeline_rows})
        y_map = {group: idx for idx, group in enumerate(groups)}
        for row in timeline_rows:
            group = (str(row.get("repeat_label", "")), str(row.get("mode", row.get("tag", ""))), str(row.get("worker", "")), int(row.get("rank", 0)))
            y = y_map[group]
            start = rel_sec(int(row["backward_start_unix_ns"]), base_ns)
            complete = rel_sec(int(row["backward_comm_complete_unix_ns"]), base_ns)
            ret = rel_sec(int(row["backward_return_unix_ns"]), base_ns)
            ax.broken_barh([(start, max(complete - start, 1e-6))], (y - 0.35, 0.7), facecolors="#3b82f6", alpha=0.55)
            ax.plot([ret], [y], marker="|", color="#111827", markersize=7, alpha=0.7)
        for marker in markers:
            if marker.get("marker") in {"mode_start", "mode_end"}:
                x = rel_sec(int(marker["ts_unix_ns"]), base_ns)
                ax.axvline(x, color="#991b1b", linestyle="--", linewidth=1.0, alpha=0.45)
        ax.set_yticks(list(y_map.values()))
        ax.set_yticklabels([f"{rep}/{mode}/{worker}/r{rank}" for rep, mode, worker, rank in groups], fontsize=8)
        ax.set_xlabel("Time since first measured event (s)")
        ax.set_ylabel("Worker/rank")
        ax.set_title("Phase7 backward execution window and communication-complete proxy")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.tight_layout()
        path = output_dir / "phase7_backward_timeline.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append(path.name)

    if timeline_rows:
        by_step: DefaultDict[int, list[float]] = defaultdict(list)
        for row in timeline_rows:
            by_step[int(row["step"])].append(float(row.get("local_backward_total_ms", 0.0)))
        xs = sorted(by_step)
        ys_mean = [mean(by_step[x]) for x in xs]
        ys_p95 = [percentile(by_step[x], 0.95) for x in xs]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(xs, ys_mean, label="backward total mean", linewidth=1.6)
        ax.plot(xs, ys_p95, label="backward total p95", linewidth=1.2)
        ax.set_xlabel("Measured step")
        ax.set_ylabel("ms")
        ax.set_title("Backward duration by step")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / "phase7_backward_duration_by_step.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append(path.name)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    if step_rows:
        xs = [rel_sec(int(row["ts_mid_unix_ns"]), base_ns) for row in step_rows]
        ys = [float(row.get("samples_per_sec", 0.0)) for row in step_rows]
        axes[0].plot(xs, ys, marker="o", markersize=3, linewidth=1.0, color="#2563eb", label="DDP samples/sec")
    axes[0].set_ylabel("samples/sec")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.25)

    if pfc_bucket:
        axes[1].plot([r["rel_sec"] for r in pfc_bucket], [r["delta"] for r in pfc_bucket], color="#dc2626", linewidth=1.5, label="PFC pause delta/bin")
    axes[1].set_ylabel("PFC delta")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.25)

    if deadlock_bucket:
        axes[2].plot([r["rel_sec"] for r in deadlock_bucket], [r["delta"] for r in deadlock_bucket], color="#7c3aed", linewidth=1.5, label="PFC deadlock delta/bin")
    else:
        axes[2].text(0.02, 0.5, "No PFC deadlock jsonl data found", transform=axes[2].transAxes)
    axes[2].set_ylabel("deadlock delta")
    axes[2].set_xlabel("Time since first measured event (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.25)

    for ax in axes:
        for marker in markers:
            if marker.get("marker") in {"mode_start", "mode_end"}:
                x = rel_sec(int(marker["ts_unix_ns"]), base_ns)
                ax.axvline(x, color="#111827", linestyle=":", linewidth=1.0, alpha=0.4)
    fig.suptitle("Phase7 throughput vs switch PFC/deadlock")
    fig.tight_layout()
    path = output_dir / "phase7_throughput_pfc_deadlock_overlay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(path.name)

    return plots


def write_html(output_dir: Path, input_dir: Path, switch_dir: Path | None, summary: list[dict], plots: list[str]) -> Path:
    def table(rows: list[dict]) -> str:
        if not rows:
            return "<p>No summary rows.</p>"
        fields = list(rows[0].keys())
        head = "".join(f"<th>{html.escape(str(field))}</th>" for field in fields)
        body = []
        for row in rows:
            cells = []
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                cells.append(f"<td>{html.escape(str(value))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"

    images = "\n".join(f'<h2>{html.escape(plot)}</h2><img src="{html.escape(plot)}" alt="{html.escape(plot)}">' for plot in plots)
    switch_text = str(switch_dir) if switch_dir is not None else "not found"
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Phase7 DDP Timeline Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #111827; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th {{ background: #f3f4f6; }}
    td:first-child, td:nth-child(2) {{ text-align: left; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Phase7 DDP Timeline Report</h1>
  <p>Input: <code>{html.escape(str(input_dir))}</code></p>
  <p>Switch log: <code>{html.escape(switch_text)}</code></p>
  <p>Communication complete is represented by <code>backward_comm_complete_unix_ns</code>, recorded after <code>loss.backward()</code> returns and <code>torch.cuda.synchronize()</code> completes. This is a DDP/NCCL completion proxy, not a per-bucket NCCL internal completion timestamp.</p>
  <h2>Summary</h2>
  {table(summary)}
  {images}
</body>
</html>
"""
    path = output_dir / "phase7_timeline_report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_rows = load_timeline_rows(input_dir, args.include_warmup)
    step_rows = load_step_metrics(input_dir, args.include_warmup)
    switch_dir = resolve_switch_log_dir(input_dir, args.switch_log_dir)
    markers = load_markers(switch_dir)
    base_ns = choose_base_ns(timeline_rows, step_rows, markers)

    if not timeline_rows:
        raise SystemExit(f"No phase7 timeline rows found under {input_dir}")

    pfc_delta = delta_rows(load_pfc_cumulative(switch_dir))
    deadlock_delta = delta_rows(load_deadlock_cumulative(switch_dir))
    pfc_bucket = bucket_deltas(pfc_delta, base_ns, args.bucket_ms, "pfc_pause") if pfc_delta else []
    deadlock_bucket = bucket_deltas(deadlock_delta, base_ns, args.bucket_ms, "pfc_deadlock") if deadlock_delta else []

    timeline_csv = write_timeline_csv(output_dir, timeline_rows, base_ns)
    write_csv(
        output_dir / "phase7_switch_pfc_delta.csv",
        pfc_delta,
        ["ts_unix_ns", "series", "switch", "value", "delta"],
    )
    write_csv(
        output_dir / "phase7_switch_deadlock_delta.csv",
        deadlock_delta,
        ["ts_unix_ns", "series", "switch", "value", "delta"],
    )
    write_csv(
        output_dir / "phase7_switch_bucket_delta.csv",
        pfc_bucket + deadlock_bucket,
        ["rel_sec", "metric", "delta", "bucket_ms"],
    )

    summary = summarize(timeline_rows, step_rows, pfc_delta, deadlock_delta)
    write_csv(
        output_dir / "phase7_summary.csv",
        summary,
        [
            "repeat",
            "mode",
            "steps",
            "workers",
            "backward_ms_mean",
            "backward_ms_p95",
            "backward_sync_ms_mean",
            "backward_sync_ms_p95",
            "samples_per_sec_mean",
            "samples_per_sec_p50",
            "total_switch_pfc_delta",
            "total_switch_deadlock_delta",
        ],
    )
    plots = plot_reports(output_dir, timeline_rows, step_rows, markers, pfc_bucket, deadlock_bucket, base_ns)
    html_path = write_html(output_dir, input_dir, switch_dir, summary, plots)

    print(f"[phase7-reporter] timeline_rows={len(timeline_rows)} step_rows={len(step_rows)} switch_dir={switch_dir}")
    print(f"[phase7-reporter] wrote {html_path}")
    print(f"[phase7-reporter] wrote {timeline_csv}")


if __name__ == "__main__":
    main()
