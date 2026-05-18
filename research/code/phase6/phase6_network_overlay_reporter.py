#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SWITCH_SHARED_ROOTS = (
    Path("/mnt/nfs_share/cts_experiments/switch_log"),
    Path("/mnt/nfs/cts_experiments/switch_log"),
)
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
    "u",
    "wBefore",
    "wAfter",
    "cooldown",
    "stableLow",
    "wstallRatio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Phase6 W adjustments with switch/network congestion metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="Phase6 experiment run root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--bucket-ms", type=float, default=1000.0, help="Network bucket width in ms")
    parser.add_argument("--event-window-sec", type=float, default=2.0, help="Before/after window around W adjustment events")
    parser.add_argument(
        "--workers",
        default="worker01",
        help="Comma-separated worker directory names to scan for NCCL logs. Default: worker01",
    )
    parser.add_argument("--all-workers", action="store_true", help="Scan all workers. Slower on large NCCL logs.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            index = 0
            while index < len(line):
                while index < len(line) and line[index].isspace():
                    index += 1
                if index >= len(line):
                    break
                try:
                    row, next_index = decoder.raw_decode(line, index)
                except json.JSONDecodeError:
                    break
                if isinstance(row, dict):
                    rows.append(row)
                index = next_index
    return rows


def to_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_pairs(line: str) -> dict:
    row = {key: value for key, value in PAIR_RE.findall(line)}
    for key in NUMERIC_FIELDS:
        if key in row:
            row[key] = to_float(row[key])
    return row


def run_rg(input_dir: Path, workers: Optional[set[str]]) -> list[tuple[Path, str]]:
    if workers:
        log_paths = []
        for worker in sorted(workers):
            log_paths.extend(sorted(input_dir.glob(f"repeat_*/P6*/{worker}/nccl.*.log")))
    else:
        log_paths = sorted(input_dir.glob("repeat_*/P6*/*/nccl.*.log"))
    if not log_paths:
        if workers:
            log_paths = []
            for worker in sorted(workers):
                log_paths.extend(sorted(input_dir.glob(f"repeat_*/*/{worker}/nccl.*.log")))
        else:
            log_paths = sorted(input_dir.glob("repeat_*/*/*/nccl.*.log"))
    if not log_paths:
        return []
    cmd = [
        "rg",
        "--fixed-strings",
        "--no-heading",
        "--with-filename",
        "--line-number",
        "--threads",
        "4",
        "PHASE6 event=CTRL_EPOCH",
        *[str(path) for path in log_paths],
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return run_python_scan(log_paths)
    if proc.returncode not in (0, 1):
        return run_python_scan(log_paths)
    matches: list[tuple[Path, str]] = []
    for line in proc.stdout.splitlines():
        path_str, _line_no, payload = line.split(":", 2)
        matches.append((Path(path_str), payload))
    return matches


def run_python_scan(log_paths: Iterable[Path]) -> list[tuple[Path, str]]:
    matches: list[tuple[Path, str]] = []
    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PHASE6 event=CTRL_EPOCH" in line:
                    matches.append((log_path, line))
    return matches


def load_phase6_epochs(run_root: Path, workers: Optional[set[str]]) -> list[dict]:
    rows: list[dict] = []
    for log_path, line in run_rg(run_root, workers):
        parts = log_path.relative_to(run_root).parts
        if len(parts) < 4:
            continue
        repeat, mode, worker = parts[:3]
        row = parse_pairs(line)
        row["repeat"] = repeat
        row["mode"] = mode
        row["worker"] = worker
        rows.append(row)
    rows.sort(key=lambda r: (str(r["repeat"]), str(r["mode"]), to_float(r.get("tNs")), str(r["worker"])))
    return rows


def resolve_switch_log_dir(run_root: Path) -> Optional[Path]:
    env_path = run_root / "switch_logger.env"
    if not env_path.exists():
        return None
    candidates: list[Path] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("SWITCH_LOG_LOCAL_DIR=") or line.startswith("SWITCH_LOG_DIR="):
            candidates.append(Path(line.split("=", 1)[1].strip()))
        elif line.startswith("SWITCH_LOG_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            candidates.extend(root / run_id for root in SWITCH_SHARED_ROOTS)
    seen: set[str] = set()
    for candidate in candidates:
        variants = [candidate]
        raw = candidate.as_posix()
        if raw.startswith("/mnt/nfs/"):
            variants.append(Path("/mnt/nfs_share/" + raw[len("/mnt/nfs/") :]))
        elif raw.startswith("/mnt/nfs_share/"):
            variants.append(Path("/mnt/nfs/" + raw[len("/mnt/nfs_share/") :]))
        for path in variants:
            key = path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return path
    return None


def extract_ts(row: dict) -> Optional[int]:
    for key in ("ts_mid_unix_ns", "ts_unix_ns", "timestamp_ns"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def load_cumulative_series(path: Path, keys: tuple[str, ...]) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for row in read_jsonl(path):
        ts_ns = extract_ts(row)
        if ts_ns is None:
            continue
        total = 0.0
        found = False
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += max(0.0, float(value))
                found = True
        if found:
            series.append((ts_ns, total))
    series.sort(key=lambda item: item[0])
    return series


def interpolate(series: list[tuple[int, float]], ts_ns: int) -> Optional[float]:
    if not series:
        return None
    if ts_ns <= series[0][0]:
        return series[0][1]
    if ts_ns >= series[-1][0]:
        return series[-1][1]
    lo = 0
    hi = len(series) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] < ts_ns:
            lo = mid + 1
        else:
            hi = mid - 1
    left_ts, left_value = series[max(0, lo - 1)]
    right_ts, right_value = series[min(len(series) - 1, lo)]
    if right_ts == left_ts:
        return right_value
    ratio = (ts_ns - left_ts) / float(right_ts - left_ts)
    return left_value + (right_value - left_value) * ratio


def window_delta(series: list[tuple[int, float]], start_ns: int, end_ns: int) -> float:
    start = interpolate(series, start_ns)
    end = interpolate(series, end_ns)
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def combine_series(series_list: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    timestamps = sorted({ts for series in series_list for ts, _value in series})
    out: list[tuple[int, float]] = []
    for ts_ns in timestamps:
        total = 0.0
        found = False
        for series in series_list:
            value = interpolate(series, ts_ns)
            if value is None:
                continue
            total += value
            found = True
        if found:
            out.append((ts_ns, total))
    return out


def bucket_delta(series: list[tuple[int, float]], start_ns: int, end_ns: int, bucket_ms: float) -> list[tuple[float, float]]:
    if not series or end_ns <= start_ns:
        return []
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    points: list[tuple[float, float]] = []
    current = start_ns
    while current < end_ns:
        nxt = min(end_ns, current + bucket_ns)
        rel_sec = (nxt - start_ns) / 1_000_000_000.0
        points.append((rel_sec, window_delta(series, current, nxt)))
        current = nxt
    return points


def bucket_rate(series: list[tuple[int, float]], start_ns: int, end_ns: int, bucket_ms: float, scale: float = 1.0) -> list[tuple[float, float]]:
    bucket_ns = max(1, int(bucket_ms * 1_000_000.0))
    points = []
    for rel_sec, delta in bucket_delta(series, start_ns, end_ns, bucket_ms):
        rate = (delta * scale) / (bucket_ns / 1_000_000_000.0)
        points.append((rel_sec, rate))
    return points


def load_switch_bundle(run_root: Path) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(run_root)
    if log_dir is None:
        return None
    markers = load_mode_markers(log_dir / "markers.jsonl")
    rack_a = load_cumulative_series(log_dir / "rackA_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    rack_b = load_cumulative_series(log_dir / "rackB_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    spine_pfc = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_pause_packets_total", "tx_pause_packets_total"))
    spine_ecn = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_ecn_marked_packets_total",))
    total_pfc = combine_series([rack_a, rack_b, spine_pfc])
    worker_bytes = load_cumulative_series(
        log_dir / "worker_roce_counters_aggregate.jsonl",
        ("tx_prio_bytes_total", "rx_prio_bytes_total", "tx_prio_bytes", "rx_prio_bytes"),
    )
    worker_cnp = load_cumulative_series(
        log_dir / "worker_roce_counters_aggregate.jsonl",
        ("np_cnp_sent_total", "rp_cnp_handled_total", "np_cnp_sent", "rp_cnp_handled"),
    )
    return {
        "log_dir": log_dir,
        "markers": markers,
        "rackA_pfc": rack_a,
        "rackB_pfc": rack_b,
        "spine_pfc": spine_pfc,
        "spine_ecn": spine_ecn,
        "total_pfc": total_pfc,
        "worker_bytes": worker_bytes,
        "worker_cnp": worker_cnp,
    }


def load_mode_markers(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    markers: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return markers
    for row in read_jsonl(path):
        marker = row.get("marker")
        message = str(row.get("message", ""))
        repeat_match = re.search(r"\brepeat=(\S+)", message)
        mode_match = re.search(r"\bmode=(\S+)", message)
        ts_ns = row.get("ts_unix_ns")
        if marker not in ("mode_start", "mode_end"):
            continue
        if not repeat_match or not mode_match or not isinstance(ts_ns, (int, float)):
            continue
        key = (repeat_match.group(1), mode_match.group(1).upper())
        markers[key]["start_ns" if marker == "mode_start" else "end_ns"] = int(ts_ns)
    return markers


def aligned_switch_window(
    repeat: str,
    mode: str,
    rows: list[dict],
    switch: Optional[dict],
) -> tuple[int, int, int, bool]:
    ctrl_start_ns = int(min(to_float(r.get("tNs")) for r in rows))
    ctrl_end_ns = int(max(to_float(r.get("tNs")) for r in rows))
    if switch:
        marker = switch.get("markers", {}).get((repeat, mode))
        if marker and "start_ns" in marker:
            switch_start_ns = int(marker["start_ns"])
            return ctrl_start_ns, switch_start_ns, switch_start_ns + (ctrl_end_ns - ctrl_start_ns), True
    return ctrl_start_ns, ctrl_start_ns, ctrl_end_ns, False


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["repeat"]), str(row["mode"]))].append(row)
    for group in groups.values():
        group.sort(key=lambda r: to_float(r.get("tNs")))
    return groups


def downsample(rows: list[dict], limit: int = 6000) -> list[dict]:
    if len(rows) <= limit:
        return rows
    step = max(1, len(rows) // limit)
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def plot_group(repeat: str, mode: str, rows: list[dict], switch: Optional[dict], output_dir: Path, bucket_ms: float) -> str:
    start_ns, switch_start_ns, switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
    plot_rows = downsample(rows)
    xs = [(to_float(r.get("tNs")) - start_ns) / 1_000_000_000.0 for r in plot_rows]
    decrease_rows = [r for r in rows if r.get("action") == "decrease"]
    increase_rows = [r for r in rows if r.get("action") == "increase"]

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    axes[0].plot(xs, [to_float(r.get("wAfter")) for r in plot_rows], linewidth=1.5, label="W", color="#1f77b4")
    axes[0].set_ylabel("W")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, [to_float(r.get("delayMeanNs")) / 1e6 for r in plot_rows], linewidth=1.0, label="delayMean ms", color="#d62728")
    axes[1].plot(xs, [to_float(r.get("delaySlowNs")) / 1e6 for r in plot_rows], linewidth=1.0, label="delaySlow ms", color="#9467bd", alpha=0.8)
    axes[1].set_ylabel("POST-DONE ms")
    axes[1].legend(loc="upper right")

    axes[2].plot(xs, [to_float(r.get("e")) for r in plot_rows], linewidth=1.0, label="spike e", color="#ff7f0e")
    axes[2].plot(xs, [to_float(r.get("wstallRatio")) for r in plot_rows], linewidth=1.0, label="wstallRatio", color="#2ca02c", alpha=0.8)
    axes[2].set_ylabel("Controller signal")
    axes[2].legend(loc="upper right")

    if switch:
        total_pfc = bucket_delta(switch["total_pfc"], switch_start_ns, switch_end_ns, bucket_ms)
        spine_ecn = bucket_delta(switch["spine_ecn"], switch_start_ns, switch_end_ns, bucket_ms)
        worker_gbps = bucket_rate(switch["worker_bytes"], switch_start_ns, switch_end_ns, bucket_ms, scale=8e-9)
        if total_pfc:
            axes[3].plot([x for x, _ in total_pfc], [y for _, y in total_pfc], label="total PFC delta/bucket", color="#111827")
        if spine_ecn:
            axes[3].plot([x for x, _ in spine_ecn], [y for _, y in spine_ecn], label="spine ECN delta/bucket", color="#8c564b")
        if worker_gbps:
            ax2 = axes[3].twinx()
            ax2.plot([x for x, _ in worker_gbps], [y for _, y in worker_gbps], label="worker NIC Gbps", color="#17becf", alpha=0.7)
            ax2.set_ylabel("NIC Gbps")
            ax2.legend(loc="upper right")
    axes[3].set_ylabel("Switch delta")
    axes[3].legend(loc="upper left")

    for ax in axes:
        for row in decrease_rows:
            x = (to_float(row.get("tNs")) - start_ns) / 1_000_000_000.0
            ax.axvline(x, color="#b91c1c", alpha=0.22, linewidth=1.2)
        for row in increase_rows:
            x = (to_float(row.get("tNs")) - start_ns) / 1_000_000_000.0
            ax.axvline(x, color="#047857", alpha=0.22, linewidth=1.2)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time since P6 mode start (s)")
    alignment = "switch mode_start aligned" if marker_aligned else "not switch-aligned"
    fig.suptitle(f"{repeat} {mode}: Phase6 W Adjustment vs Network Congestion ({alignment})")
    fig.tight_layout()
    name = f"{repeat}_{mode}_w_network_overlay.png"
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def write_event_csv(groups: dict[tuple[str, str], list[dict]], switch: Optional[dict], output_dir: Path, window_sec: float) -> Path:
    path = output_dir / "phase6_w_adjustment_network_windows.csv"
    fields = [
        "repeat",
        "mode",
        "worker",
        "channel",
        "action",
        "tNs",
        "aligned_switch_unix_ns",
        "wBefore",
        "wAfter",
        "e",
        "delayMeanMs",
        "delaySlowMs",
        "pfc_before",
        "pfc_after",
        "ecn_before",
        "ecn_after",
        "worker_gb_before",
        "worker_gb_after",
        "cnp_before",
        "cnp_after",
    ]
    window_ns = int(window_sec * 1_000_000_000)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (repeat, mode), rows in sorted(groups.items()):
            ctrl_start_ns, switch_start_ns, _switch_end_ns, marker_aligned = aligned_switch_window(repeat, mode, rows, switch)
            for row in rows:
                if row.get("action") not in ("decrease", "increase"):
                    continue
                t_ns = int(to_float(row.get("tNs")))
                aligned_t_ns = switch_start_ns + (t_ns - ctrl_start_ns) if marker_aligned else t_ns
                out = {
                    "repeat": repeat,
                    "mode": mode,
                    "worker": row.get("worker"),
                    "channel": row.get("channel"),
                    "action": row.get("action"),
                    "tNs": t_ns,
                    "aligned_switch_unix_ns": aligned_t_ns if marker_aligned else "",
                    "wBefore": row.get("wBefore"),
                    "wAfter": row.get("wAfter"),
                    "e": row.get("e"),
                    "delayMeanMs": to_float(row.get("delayMeanNs")) / 1e6,
                    "delaySlowMs": to_float(row.get("delaySlowNs")) / 1e6,
                    "pfc_before": 0.0,
                    "pfc_after": 0.0,
                    "ecn_before": 0.0,
                    "ecn_after": 0.0,
                    "worker_gb_before": 0.0,
                    "worker_gb_after": 0.0,
                    "cnp_before": 0.0,
                    "cnp_after": 0.0,
                }
                if switch:
                    out["pfc_before"] = window_delta(switch["total_pfc"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["pfc_after"] = window_delta(switch["total_pfc"], aligned_t_ns, aligned_t_ns + window_ns)
                    out["ecn_before"] = window_delta(switch["spine_ecn"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["ecn_after"] = window_delta(switch["spine_ecn"], aligned_t_ns, aligned_t_ns + window_ns)
                    out["worker_gb_before"] = window_delta(switch["worker_bytes"], aligned_t_ns - window_ns, aligned_t_ns) * 8e-9
                    out["worker_gb_after"] = window_delta(switch["worker_bytes"], aligned_t_ns, aligned_t_ns + window_ns) * 8e-9
                    out["cnp_before"] = window_delta(switch["worker_cnp"], aligned_t_ns - window_ns, aligned_t_ns)
                    out["cnp_after"] = window_delta(switch["worker_cnp"], aligned_t_ns, aligned_t_ns + window_ns)
                writer.writerow(out)
    return path


def write_html(output_dir: Path, image_names: list[str], event_csv: Path, switch: Optional[dict]) -> Path:
    path = output_dir / "phase6_network_overlay_report.html"
    switch_msg = f"switch_log_dir={html.escape(str(switch['log_dir']))}" if switch else "switch_log_dir=not found"
    images = "\n".join(f"<h2>{html.escape(name)}</h2><img src='{html.escape(name)}'>" for name in image_names)
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase6 Network Overlay</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
img {{ width: 100%; max-width: 1900px; border: 1px solid #ddd; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style></head><body>
<h1>Phase6 W Adjustment vs Network Congestion</h1>
<p><code>{switch_msg}</code></p>
<p>Red vertical lines are W decrease events. Green vertical lines are W increase events. Switch metrics are aligned with <code>mode_start</code> markers from <code>markers.jsonl</code>. Event before/after network deltas are in <code>{html.escape(event_csv.name)}</code>.</p>
{images}
</body></html>
""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    run_root = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = None if args.all_workers else {item.strip() for item in args.workers.split(",") if item.strip()}
    rows = load_phase6_epochs(run_root, workers)
    groups = group_rows(rows)
    switch = load_switch_bundle(run_root)
    event_csv = write_event_csv(groups, switch, output_dir, args.event_window_sec)
    image_names = [
        plot_group(repeat, mode, group, switch, output_dir, args.bucket_ms)
        for (repeat, mode), group in sorted(groups.items())
        if group
    ]
    html_path = write_html(output_dir, image_names, event_csv, switch)
    worker_msg = "all" if workers is None else ",".join(sorted(workers))
    print(f"[phase6-network] rows={len(rows)} groups={len(groups)} workers={worker_msg} switch={bool(switch)} html={html_path}")


if __name__ == "__main__":
    main()
