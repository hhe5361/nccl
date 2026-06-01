#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SWITCH_SHARED_ROOTS = (
    Path("/mnt/nfs_share/cts_experiments/switch_log"),
    Path("/mnt/nfs/cts_experiments/switch_log"),
)

LOWER_IS_BETTER = {
    "app_step_ms_avg",
    "app_step_ms_p50",
    "app_step_ms_p95",
    "app_backward_ms_p95",
    "post_done_p99_median_ms",
    "post_done_p99_p95_ms",
    "post_done_worker_p99_max_ms",
    "total_pfc_delta_sum",
    "rackA_pfc_delta_sum",
    "rackB_pfc_delta_sum",
    "spine_pfc_delta_sum",
    "total_deadlock_delta_sum",
}
HIGHER_IS_BETTER = {
    "app_samples_per_sec_avg",
    "app_samples_per_sec_p50",
}

COMPARE_FIELDS = [
    "app_step_ms_avg",
    "app_step_ms_p50",
    "app_step_ms_p95",
    "app_backward_ms_p95",
    "app_samples_per_sec_avg",
    "app_samples_per_sec_p50",
    "post_done_p99_median_ms",
    "post_done_p99_p95_ms",
    "post_done_worker_p99_max_ms",
    "total_pfc_delta_sum",
    "rackA_pfc_delta_sum",
    "rackB_pfc_delta_sum",
    "spine_pfc_delta_sum",
    "total_deadlock_delta_sum",
]

MODE_ACTIVITY_FIELDS = [
    "effective_steps",
    "phase6_w_min",
    "phase6_w_max",
    "phase6_w_final",
    "phase6_decrease",
    "phase6_increase",
    "analysis_bins",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase6 modes against STOCK baseline.")
    parser.add_argument("--input", required=True, type=Path, help="Report root, run root, or plots root")
    parser.add_argument("--run-root", type=Path, default=None, help="Phase6 experiment run root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--switch-log-dir", type=Path, default=None, help="Explicit switch logger directory")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional phase6_summary.csv")
    parser.add_argument("--bin-csv", type=Path, default=None, help="Optional phase6_bin_metrics.csv")
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


def read_csv(path: Optional[Path]) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def resolve_run_root(input_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    root = input_root.expanduser().resolve()
    if any(root.glob("repeat_*")):
        return root
    for parent in [root, *root.parents]:
        if any(parent.glob("repeat_*")):
            return parent
    return root


def resolve_optional_csv(input_root: Path, explicit: Optional[Path], name: str, preferred_subdir: str) -> Optional[Path]:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [
        input_root / preferred_subdir / name,
        input_root / "plots_all" / preferred_subdir / name,
        input_root / "plots" / preferred_subdir / name,
        input_root / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = sorted(input_root.rglob(f"{preferred_subdir}/{name}"))
    if matches:
        return matches[0].resolve()
    matches = sorted(input_root.rglob(name))
    if matches:
        return matches[0].resolve()
    return None


def resolve_switch_log_dir(run_root: Path, explicit_switch_log_dir: Optional[Path] = None) -> Optional[Path]:
    if explicit_switch_log_dir is not None:
        explicit = explicit_switch_log_dir.expanduser().resolve()
        if explicit.exists():
            return explicit
    env_path = run_root / "switch_logger.env"
    if not env_path.exists():
        return None
    candidates: list[Path] = []
    run_ids: list[str] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("SWITCH_LOG_LOCAL_DIR=") or line.startswith("SWITCH_LOG_DIR="):
            candidates.append(Path(line.split("=", 1)[1].strip()))
        elif line.startswith("SWITCH_LOG_RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            run_ids.append(run_id)
            candidates.extend(root / run_id for root in SWITCH_SHARED_ROOTS)
    for run_id in run_ids:
        for parent in [run_root, *run_root.parents]:
            candidates.extend(
                [
                    parent / "switch_log" / run_id,
                    parent / "switch_log" / "switch_log" / run_id,
                ]
            )
            if parent.parent != parent:
                candidates.extend(
                    [
                        parent.parent / "switch_log" / run_id,
                        parent.parent / "switch_log" / "switch_log" / run_id,
                    ]
                )
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


def load_mode_markers(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    markers: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
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
    if end_ns <= start_ns:
        return math.nan
    start = interpolate(series, start_ns)
    end = interpolate(series, end_ns)
    if start is None or end is None:
        return math.nan
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


def load_ddp_app_metrics(run_root: Path) -> dict[tuple[str, str], dict]:
    metrics: dict[tuple[str, str], dict] = {}
    for summary_path in sorted(run_root.glob("repeat_*/*/*_summary.json")):
        rel = summary_path.relative_to(run_root).parts
        if len(rel) < 3:
            continue
        repeat = rel[0]
        mode = rel[1].upper()
        try:
            row = json.loads(summary_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        key = (repeat, mode)
        metrics[key] = {
            "repeat": repeat,
            "mode": mode,
            "effective_steps": to_float(row.get("effective_steps")),
            "app_step_ms_avg": to_float(row.get("step_ms_avg")),
            "app_step_ms_p50": to_float(row.get("step_ms_p50")),
            "app_step_ms_p95": to_float(row.get("step_ms_p95")),
            "app_backward_ms_p95": to_float(row.get("backward_ms_p95")),
            "app_samples_per_sec_avg": to_float(row.get("samples_per_sec_avg")),
            "app_samples_per_sec_p50": to_float(row.get("samples_per_sec_p50")),
            "param_mb": to_float(row.get("param_mb")),
            "summary_path": str(summary_path),
        }

    step_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for step_path in sorted(run_root.glob("repeat_*/*/*_step_metrics.jsonl")):
        rel = step_path.relative_to(run_root).parts
        if len(rel) < 3:
            continue
        repeat = rel[0]
        mode = rel[1].upper()
        for row in read_jsonl(step_path):
            if bool(row.get("warmup")):
                continue
            step_groups[(repeat, mode)].append(row)

    for key, rows in step_groups.items():
        repeat, mode = key
        item = metrics.setdefault(key, {"repeat": repeat, "mode": mode})
        step_ms = [to_float(row.get("step_ms_max")) for row in rows]
        backward_ms = [to_float(row.get("backward_ms_max")) for row in rows]
        sps = [to_float(row.get("samples_per_sec")) for row in rows]
        starts = finite(to_float(row.get("ts_start_unix_ns")) for row in rows)
        ends = finite(to_float(row.get("ts_end_unix_ns")) for row in rows)
        item.setdefault("effective_steps", float(len(rows)))
        if not math.isfinite(to_float(item.get("app_step_ms_avg"))):
            item["app_step_ms_avg"] = percentile(step_ms, 50) if step_ms else math.nan
        if not math.isfinite(to_float(item.get("app_step_ms_p50"))):
            item["app_step_ms_p50"] = percentile(step_ms, 50)
        if not math.isfinite(to_float(item.get("app_step_ms_p95"))):
            item["app_step_ms_p95"] = percentile(step_ms, 95)
        if not math.isfinite(to_float(item.get("app_backward_ms_p95"))):
            item["app_backward_ms_p95"] = percentile(backward_ms, 95)
        if not math.isfinite(to_float(item.get("app_samples_per_sec_avg"))):
            vals = finite(sps)
            item["app_samples_per_sec_avg"] = sum(vals) / len(vals) if vals else math.nan
        if not math.isfinite(to_float(item.get("app_samples_per_sec_p50"))):
            item["app_samples_per_sec_p50"] = percentile(sps, 50)
        if starts:
            item["step_start_ns"] = int(min(starts))
        if ends:
            item["step_end_ns"] = int(max(ends))
    return metrics


def load_switch_metrics(run_root: Path, switch_log_dir: Optional[Path], metrics: dict[tuple[str, str], dict]) -> None:
    log_dir = resolve_switch_log_dir(run_root, switch_log_dir)
    for item in metrics.values():
        item["switch_log_dir"] = str(log_dir) if log_dir else ""
    if log_dir is None:
        return

    markers = load_mode_markers(log_dir / "markers.jsonl")
    rack_a = load_cumulative_series(log_dir / "rackA_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    rack_b = load_cumulative_series(log_dir / "rackB_pfc_aggregate.jsonl", ("rx_pause_total", "tx_pause_total"))
    spine = load_cumulative_series(log_dir / "spine_pfc_ecn_aggregate.jsonl", ("rx_pause_packets_total", "tx_pause_packets_total"))
    rack_a_dead = load_cumulative_series(log_dir / "rackA_pfc_deadlock_aggregate.jsonl", ("deadlock_count_total",))
    rack_b_dead = load_cumulative_series(log_dir / "rackB_pfc_deadlock_aggregate.jsonl", ("deadlock_count_total",))
    total_pfc = combine_series([rack_a, rack_b, spine])
    total_dead = combine_series([rack_a_dead, rack_b_dead])

    for key, item in metrics.items():
        marker = markers.get(key, {})
        start_ns = marker.get("start_ns") or item.get("step_start_ns")
        end_ns = marker.get("end_ns") or item.get("step_end_ns")
        if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns <= start_ns:
            continue
        item["mode_start_ns"] = start_ns
        item["mode_end_ns"] = end_ns
        item["mode_duration_sec"] = (end_ns - start_ns) / 1e9
        item["rackA_pfc_delta_sum"] = window_delta(rack_a, start_ns, end_ns)
        item["rackB_pfc_delta_sum"] = window_delta(rack_b, start_ns, end_ns)
        item["spine_pfc_delta_sum"] = window_delta(spine, start_ns, end_ns)
        item["total_pfc_delta_sum"] = window_delta(total_pfc, start_ns, end_ns)
        item["total_deadlock_delta_sum"] = window_delta(total_dead, start_ns, end_ns)


def summarize_bins(rows: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        repeat = row.get("repeat", "")
        mode = row.get("mode", "")
        if repeat and mode:
            grouped[(repeat, mode.upper())].append(row)

    out: dict[tuple[str, str], dict] = {}
    for key, group in grouped.items():
        p99 = [to_float(row.get("delay_p99_ms")) for row in group]
        worker_p99 = [to_float(row.get("delay_worker_p99_max_ms")) for row in group]
        out[key] = {
            "post_done_p99_median_ms": percentile(p99, 50),
            "post_done_p99_p95_ms": percentile(p99, 95),
            "post_done_worker_p99_max_ms": max(finite(worker_p99), default=math.nan),
            "analysis_bins": len(group),
        }
    return out


def merge_optional_phase6_metrics(metrics: dict[tuple[str, str], dict], summary_csv: Optional[Path], bin_csv: Optional[Path]) -> None:
    for row in read_csv(summary_csv):
        repeat = row.get("repeat", "")
        mode = row.get("mode", "")
        if not repeat or not mode:
            continue
        item = metrics.setdefault((repeat, mode.upper()), {"repeat": repeat, "mode": mode.upper()})
        item.update(
            {
                "phase6_w_min": to_float(row.get("w_min")),
                "phase6_w_max": to_float(row.get("w_max")),
                "phase6_w_final": to_float(row.get("w_final")),
                "phase6_decrease": to_float(row.get("decrease"), 0.0),
                "phase6_increase": to_float(row.get("increase"), 0.0),
            }
        )
    for key, row in summarize_bins(read_csv(bin_csv)).items():
        item = metrics.setdefault(key, {"repeat": key[0], "mode": key[1]})
        item.update(row)


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
            for field in COMPARE_FIELDS:
                value = to_float(row.get(field))
                base = to_float(baseline.get(field))
                item[field] = value
                item[f"stock_{field}"] = base
                item[f"{field}_change_pct"] = pct_change(value, base)
                item[f"{field}_improvement_pct"] = improvement_pct(field, value, base)
            for field in MODE_ACTIVITY_FIELDS:
                item[field] = row.get(field, "")
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["repeat", "baseline_mode", "mode"]
    for field in COMPARE_FIELDS:
        fieldnames.extend([field, f"stock_{field}", f"{field}_change_pct", f"{field}_improvement_pct"])
    fieldnames.extend(MODE_ACTIVITY_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.3f}"
    return str(value)


SUMMARY_SPECS = [
    ("평균 step latency", "app_step_ms_avg", "ms"),
    ("p50 step latency", "app_step_ms_p50", "ms"),
    ("p95 step latency", "app_step_ms_p95", "ms"),
    ("p95 backward latency", "app_backward_ms_p95", "ms"),
    ("평균 samples/sec", "app_samples_per_sec_avg", ""),
    ("p50 samples/sec", "app_samples_per_sec_p50", ""),
    ("total PFC delta", "total_pfc_delta_sum", "count"),
    ("rackA PFC", "rackA_pfc_delta_sum", "count"),
    ("rackB PFC", "rackB_pfc_delta_sum", "count"),
    ("deadlock", "total_deadlock_delta_sum", "count"),
]


def fmt_summary_value(value: float, unit: str) -> str:
    if not math.isfinite(value):
        return ""
    if unit == "ms":
        return f"{value:,.2f} ms"
    if unit == "count":
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def interpretation_class(improvement: float) -> str:
    if not math.isfinite(improvement) or abs(improvement) < 0.1:
        return "neutral"
    return "positive" if improvement > 0 else "negative"


def interpretation_text(field: str, value: float, baseline: float, improvement: float) -> str:
    if not math.isfinite(value) or not math.isfinite(baseline):
        return ""
    if field == "total_deadlock_delta_sum" and value == baseline:
        return "변화 없음"
    if abs(improvement) < 0.1:
        if value == baseline:
            return "변화 없음"
        direction = "개선" if improvement > 0 else "악화"
        return f"거의 동일, {abs(improvement):.2f}% {direction}"
    direction = "개선" if improvement > 0 else "악화"
    return f"{abs(improvement):.2f}% {direction}"


def summary_table(rows: list[dict], diagnostics: str) -> str:
    if not rows:
        return f"<p>No summary rows. {html.escape(diagnostics)}</p>"
    sections = []
    for row in rows:
        title = f"{row.get('repeat', '')} {row.get('mode', '')} vs {row.get('baseline_mode', 'STOCK')}"
        body = []
        for label, field, unit in SUMMARY_SPECS:
            value = to_float(row.get(field))
            baseline = to_float(row.get(f"stock_{field}"))
            improvement = to_float(row.get(f"{field}_improvement_pct"))
            klass = interpretation_class(improvement)
            body.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(fmt_summary_value(baseline, unit))}</td>"
                f"<td>{html.escape(fmt_summary_value(value, unit))}</td>"
                f"<td class='{klass}'>{html.escape(interpretation_text(field, value, baseline, improvement))}</td>"
                "</tr>"
            )
        sections.append(
            f"<h3>{html.escape(title)}</h3>"
            "<table class='summary-table'>"
            "<thead><tr><th>Metric</th><th>STOCK</th><th>P6</th><th>Interpretation</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
        )
    return "\n".join(sections)


def html_table(rows: list[dict], fields: list[str], diagnostics: str) -> str:
    if not rows:
        return f"<p>No comparison rows. {html.escape(diagnostics)}</p>"
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(fmt(row.get(field, '')))}</td>" for field in fields) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def plot_improvement(rows: list[dict], output_dir: Path) -> list[str]:
    if not rows:
        return []
    plot_specs = [
        ("app_latency_improvement_pct.png", "DDP Step Latency Improvement vs STOCK", ["app_step_ms_p50", "app_step_ms_p95", "app_backward_ms_p95"]),
        ("app_throughput_improvement_pct.png", "DDP Throughput Improvement vs STOCK", ["app_samples_per_sec_p50", "app_samples_per_sec_avg"]),
        ("switch_pfc_reduction_pct.png", "Switch PFC Reduction vs STOCK", ["total_pfc_delta_sum", "rackA_pfc_delta_sum", "rackB_pfc_delta_sum", "spine_pfc_delta_sum"]),
        ("switch_deadlock_reduction_pct.png", "Switch PFC Deadlock Reduction vs STOCK", ["total_deadlock_delta_sum"]),
    ]
    names = []
    labels = [f"{row['repeat']} {row['mode']}" for row in rows]
    x = list(range(len(rows)))
    for filename, title, fields in plot_specs:
        usable = [
            field
            for field in fields
            if any(math.isfinite(to_float(row.get(f"{field}_improvement_pct"))) for row in rows)
        ]
        if not usable:
            continue
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.5), 5))
        width = 0.8 / max(1, len(usable))
        for idx, field in enumerate(usable):
            vals = [to_float(row.get(f"{field}_improvement_pct")) for row in rows]
            offs = [pos - 0.4 + width / 2 + idx * width for pos in x]
            ax.bar(offs, vals, width=width, label=field)
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


def write_metric_inventory(path: Path, metrics: dict[tuple[str, str], dict]) -> None:
    fieldnames = ["repeat", "mode"] + sorted({key for row in metrics.values() for key in row if key not in {"repeat", "mode", "summary_path"}})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for _key, row in sorted(metrics.items()):
            writer.writerow(row)


def write_html(
    path: Path,
    run_root: Path,
    summary_csv: Optional[Path],
    bin_csv: Optional[Path],
    compare_csv: Path,
    metric_csv: Path,
    rows: list[dict],
    images: list[str],
    metrics: dict[tuple[str, str], dict],
) -> None:
    table_fields = [
        "repeat",
        "mode",
        "app_step_ms_p95_improvement_pct",
        "app_samples_per_sec_p50_improvement_pct",
        "total_pfc_delta_sum_improvement_pct",
        "rackB_pfc_delta_sum_improvement_pct",
        "app_step_ms_p95",
        "stock_app_step_ms_p95",
        "app_samples_per_sec_p50",
        "stock_app_samples_per_sec_p50",
        "total_pfc_delta_sum",
        "stock_total_pfc_delta_sum",
        "phase6_decrease",
        "phase6_w_min",
        "phase6_w_final",
    ]
    available = ", ".join(f"{repeat}/{mode}" for repeat, mode in sorted(metrics))
    diagnostics = f"Available mode keys: {available or 'none'}"
    image_html = "\n".join(f"<h2>{html.escape(name)}</h2><img src='{html.escape(name)}'>" for name in images)
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase6 STOCK vs P6</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
.summary-table {{ margin-bottom: 18px; min-width: 760px; }}
.summary-table th {{ background: #f3f4f6; }}
.summary-table td:first-child {{ font-weight: 600; }}
.positive {{ color: #047857; font-weight: 700; }}
.negative {{ color: #b91c1c; font-weight: 700; }}
.neutral {{ color: #374151; font-weight: 600; }}
img {{ width: 100%; max-width: 1600px; border: 1px solid #ddd; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style></head><body>
<h1>Phase6 STOCK Baseline Comparison</h1>
<p>Positive improvement means better than STOCK. For latency/PFC metrics, lower raw values become positive improvement. For throughput metrics, higher raw values become positive improvement.</p>
<p>This report uses DDP application summaries for both STOCK and P6 modes. Switch PFC/deadlock deltas are computed over each mode window using switch <code>mode_start</code>/<code>mode_end</code> markers.</p>
<p>run root: <code>{html.escape(str(run_root))}</code></p>
<p>controller summary: <code>{html.escape(str(summary_csv or 'not found'))}</code></p>
<p>network bins: <code>{html.escape(str(bin_csv or 'not found'))}</code></p>
<p>mode metric csv: <code>{html.escape(str(metric_csv))}</code></p>
<p>comparison csv: <code>{html.escape(str(compare_csv))}</code></p>
<h2>Readable Summary</h2>
{summary_table(rows, diagnostics)}
<h2>Key Comparison Table</h2>
{html_table(rows, table_fields, diagnostics)}
{image_html}
</body></html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    run_root = resolve_run_root(input_root, args.run_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = resolve_optional_csv(input_root, args.summary_csv, "phase6_summary.csv", "phase6_plot_latest")
    bin_csv = resolve_optional_csv(input_root, args.bin_csv, "phase6_bin_metrics.csv", "network_overlay_allworker_no_ecn_trimtop")

    metrics = load_ddp_app_metrics(run_root)
    load_switch_metrics(run_root, args.switch_log_dir, metrics)
    merge_optional_phase6_metrics(metrics, summary_csv, bin_csv)
    rows = build_comparison_rows(metrics, args.baseline_mode)

    metric_csv = output_dir / "phase6_stock_vs_p6_mode_metrics.csv"
    compare_csv = output_dir / "phase6_stock_vs_p6_comparison.csv"
    write_metric_inventory(metric_csv, metrics)
    write_csv(compare_csv, rows)
    images = plot_improvement(rows, output_dir)
    html_path = output_dir / "phase6_stock_vs_p6_report.html"
    write_html(html_path, run_root, summary_csv, bin_csv, compare_csv, metric_csv, rows, images, metrics)
    print(
        f"[phase6-stock-vs-p6] modes={len(metrics)} rows={len(rows)} "
        f"summary_csv={summary_csv or 'not_found'} bin_csv={bin_csv or 'not_found'} html={html_path}"
    )


if __name__ == "__main__":
    main()
