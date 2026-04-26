#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for phase3_log_reporter.py. "
        "Install it first, for example: pip install matplotlib"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SWITCH_SHARED_ROOT_DEFAULT = Path("/mnt/nfs/cts_experiments/switch_log")
SWITCH_DOC_ROOT = REPO_ROOT / "research" / "docs" / "switch"
SWITCH_BUCKET_NS = 1_000_000_000

PHASE_RE = re.compile(r"(PHASE\d+)\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
WORKER_RE = re.compile(r"worker(\d+)$")
EXPERIMENT_RE = re.compile(r"^\d+_.+")
MODE_ORDER = {"STOCK": 0, "B2": 1, "B3": 2}
MODE_COLORS = {"STOCK": "#1f77b4", "B2": "#ff7f0e", "B3": "#2ca02c"}
SWITCH_LABELS = ("rackA", "rackB", "spine")
COLLECTIVE_TO_COLLAPI = {
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "reducescatter": "ReduceScatter",
    "alltoall": "AllToAll",
}

REPORT_LOG_PATH: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase3 B3 experiment plots and HTML report")
    parser.add_argument("--input", "--input-dir", dest="input", required=True, help="Single experiment root, matrix root, or zip")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input>/report or <zip>_report")
    parser.add_argument("--switch-log-dir", default=None, help="Override switch log directory for all experiments")
    parser.add_argument("--top-events", type=int, default=12, help="Top NCCL events to include in event-count plots")
    return parser.parse_args()


def log_progress(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[phase3-log-reporter] {timestamp} {message}"
    print(line, flush=True)
    if REPORT_LOG_PATH is not None:
        with REPORT_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
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


def percentile_from_counter(counter: Counter, q: float) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    threshold = max(1, int(math.ceil(total * q)))
    seen = 0
    for value in sorted(counter.keys(), key=float):
        seen += counter[value]
        if seen >= threshold:
            return float(value)
    return float(max(counter.keys(), key=float))


def rel_change(base: float, candidate: float) -> float:
    if abs(float(base)) < 1e-12:
        return 0.0
    return 100.0 * (float(candidate) - float(base)) / float(base)


def parse_value(raw: str):
    if raw.startswith("0x"):
        try:
            return int(raw, 16)
        except ValueError:
            return raw
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    rows: List[dict] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        row, next_index = decoder.raw_decode(text, index)
        if isinstance(row, dict):
            rows.append(row)
        index = next_index
    return rows


def worker_sort_key(name: str) -> Tuple[int, str]:
    match = WORKER_RE.match(str(name))
    if match:
        return int(match.group(1)), str(name)
    return 10**9, str(name)


def mode_sort_key(name: str) -> Tuple[int, str]:
    upper = str(name or "").upper()
    if upper.startswith("B2_W"):
        try:
            return 1, upper
        except ValueError:
            pass
    return MODE_ORDER.get(upper, 10**9), upper


def relpath(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def positive_ylim(max_value: float) -> Optional[Tuple[float, float]]:
    if max_value <= 0:
        return None
    return 0.0, max_value * 1.08


def value_ylim(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        pad = max(abs(lo) * 0.1, 1.0)
        return lo - pad, hi + pad
    pad = max((hi - lo) * 0.08, 0.1)
    return min(0.0, lo - pad), hi + pad


def save_line_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: List[Tuple[str, List[float], List[float]]],
) -> Optional[Path]:
    usable = [(label, xs, ys) for label, xs, ys in series if xs and ys]
    if not usable:
        return None
    plt.figure(figsize=(11, 5.5))
    all_values: List[float] = []
    for label, xs, ys in usable:
        count = min(len(xs), len(ys))
        plt.plot(xs[:count], ys[:count], marker="o", linewidth=1.6, markersize=2.8, label=label)
        all_values.extend(float(v) for v in ys[:count])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ylim = value_ylim(all_values)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_grouped_bar(
    path: Path,
    title: str,
    categories: List[str],
    series: List[Tuple[str, List[float]]],
    ylabel: str,
    rotate_labels: bool = False,
) -> Optional[Path]:
    if not categories or not series:
        return None
    plt.figure(figsize=(max(10, len(categories) * 0.85), 5.8))
    x_positions = list(range(len(categories)))
    width = 0.8 / max(len(series), 1)
    all_values: List[float] = []
    for idx, (label, values) in enumerate(series):
        vals = [float(v) for v in values]
        offset = (idx - (len(series) - 1) / 2.0) * width
        plt.bar([x + offset for x in x_positions], vals, width=width, label=label)
        all_values.extend(vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(
        x_positions,
        categories,
        rotation=35 if rotate_labels else 0,
        ha="right" if rotate_labels else "center",
    )
    ylim = value_ylim(all_values)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def render_table(headers: List[str], rows: List[List[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def render_table_raw(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def render_html(title: str, sections: List[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eef3f7;
      --panel: #ffffff;
      --panel-soft: #f7f9fc;
      --text: #18212b;
      --muted: #5a6675;
      --border: #d9e1ea;
      --accent: #0b5fa5;
      --shadow: 0 14px 30px rgba(17, 31, 48, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--text);
      line-height: 1.6;
      background:
        radial-gradient(circle at top right, rgba(11,95,165,0.08), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
    }}
    .page {{ width: min(1240px, calc(100% - 32px)); margin: 24px auto 48px; }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
      padding: 24px 28px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 12px;
      border-radius: 999px;
      background: #e7f1fb;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    h1 {{ margin: 14px 0 10px; font-size: clamp(30px, 4vw, 42px); line-height: 1.08; }}
    h2 {{ margin: 0 0 12px; font-size: 24px; line-height: 1.2; }}
    p {{ margin: 10px 0 0; color: var(--muted); }}
    .grid-2 {{ display: grid; gap: 14px; margin-top: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .card {{ background: var(--panel-soft); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    figure {{ margin: 18px 0 0; padding: 12px; border-radius: 14px; border: 1px solid var(--border); background: var(--panel-soft); }}
    img {{ width: 100%; height: auto; border-radius: 10px; background: #fff; }}
    figcaption {{ margin-top: 10px; color: var(--muted); font-size: 14px; }}
    code {{
      font-family: Consolas, "SFMono-Regular", Monaco, monospace;
      font-size: 0.93em;
      background: #f2f5fa;
      border: 1px solid #e1e7f0;
      border-radius: 6px;
      padding: 1px 6px;
      color: #1d2b3a;
    }}
    @media (max-width: 920px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">Phase3 B3 Reporter</div>
      <h1>{html.escape(title)}</h1>
      <p>STOCK vs B3 receiver admission window, NCCL pressure decisions, and switch PFC counter deltas.</p>
    </header>
    {''.join(sections)}
  </div>
</body>
</html>"""


def find_mode_dirs(experiment_root: Path) -> List[Path]:
    mode_dirs: List[Path] = []
    for child in sorted(experiment_root.iterdir(), key=lambda p: mode_sort_key(p.name)):
        if child.is_dir() and list(child.glob("*_summary.json")):
            mode_dirs.append(child)
    return mode_dirs


def find_experiment_dirs(matrix_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for child in sorted(matrix_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if EXPERIMENT_RE.match(child.name) and (child / "env_setup.json").exists():
            dirs.append(child)
    return dirs


def is_matrix_root(path: Path) -> bool:
    return (path / "matrix_manifest.json").exists() or bool(find_experiment_dirs(path))


def effective_rows(step_rows: List[dict]) -> List[dict]:
    rows = [row for row in step_rows if not row.get("warmup", False)]
    return rows if rows else step_rows


def compute_summary_from_steps(step_rows: List[dict]) -> dict:
    rows = effective_rows(step_rows)
    return {
        "step_ms_avg": mean(float(row.get("step_ms_max", 0.0)) for row in rows),
        "step_ms_p50": percentile([float(row.get("step_ms_max", 0.0)) for row in rows], 0.50),
        "step_ms_p95": percentile([float(row.get("step_ms_max", 0.0)) for row in rows], 0.95),
        "collective_gbps_avg": mean(float(row.get("collective_gbps_est", 0.0)) for row in rows),
        "collective_gbps_p50": percentile([float(row.get("collective_gbps_est", 0.0)) for row in rows], 0.50),
        "collective_gbps_p95": percentile([float(row.get("collective_gbps_est", 0.0)) for row in rows], 0.95),
    }


def compute_step_tail_metrics(step_rows: List[dict]) -> dict:
    rows = effective_rows(step_rows)
    values = [float(row.get("step_ms_max", 0.0)) for row in rows]
    if not values:
        return {
            "step_ms_p99": 0.0,
            "step_ms_max": 0.0,
            "top_outlier_step": -1,
            "top_outlier_step_ms": 0.0,
            "top_outlier_warmup": False,
        }
    top_row = max(rows, key=lambda row: float(row.get("step_ms_max", 0.0)))
    return {
        "step_ms_p99": percentile(values, 0.99),
        "step_ms_max": max(values),
        "top_outlier_step": int(top_row.get("step", -1)),
        "top_outlier_step_ms": float(top_row.get("step_ms_max", 0.0)),
        "top_outlier_warmup": bool(top_row.get("warmup", False)),
    }


def load_worker_step_timings(mode_dir: Path) -> Dict[str, List[dict]]:
    timing_map: Dict[str, List[dict]] = {}
    for timing_path in sorted(mode_dir.glob("*/**/*_worker_step_timing.jsonl")):
        rows = load_jsonl(timing_path)
        if not rows:
            continue
        worker = timing_path.parent.name
        timing_map.setdefault(worker, []).extend(rows)
    for rows in timing_map.values():
        rows.sort(key=lambda row: (int(row.get("step", 0)), int(row.get("start_ns", 0))))
    return timing_map


def lookup_worker_step(step_rows: List[dict], timestamp_ns: int) -> Optional[int]:
    if not step_rows or timestamp_ns <= 0:
        return None
    for row in step_rows:
        start_ns = int(row.get("start_ns", 0) or 0)
        end_ns = int(row.get("end_ns", 0) or 0)
        if start_ns <= timestamp_ns <= end_ns:
            return int(row.get("step", 0))
    first_start = int(step_rows[0].get("start_ns", 0) or 0)
    last_end = int(step_rows[-1].get("end_ns", 0) or 0)
    if timestamp_ns < first_start:
        return int(step_rows[0].get("step", 0))
    if timestamp_ns > last_end:
        return int(step_rows[-1].get("step", 0))
    return None


def worker_window_defaults() -> dict:
    return {
        "trace_x": [],
        "trace_step": [],
        "trace_tns": [],
        "trace_w": [],
        "trace_w_sem": [],
        "trace_w_fb": [],
        "trace_pressure": [],
        "trace_occ_ratio": [],
        "trace_lag_ratio": [],
        "trace_delay_ratio": [],
        "decision_steps": [],
        "decision_reasons": [],
        "effective_decision_steps": [],
        "effective_decision_reasons": [],
        "effective_decision_old_w": [],
        "effective_decision_new_w": [],
        "sample_index": 0,
        "w_values": [],
        "w_sem_values": [],
        "w_fb_values": [],
    }


def parse_nccl_event(line: str) -> Optional[dict]:
    match = PHASE_RE.search(line)
    if not match:
        return None
    entry = {key: parse_value(value) for key, value in KV_RE.findall(match.group(2))}
    event_name = str(entry.get("event", ""))
    if not event_name:
        return None
    entry["phase"] = match.group(1)
    return entry


def should_record_window_sample(entry: dict) -> bool:
    event = str(entry.get("event", ""))
    if "wEff" not in entry:
        return False
    if event in {"PROXY_WINDOW_CFG", "PROXY_B2_WINDOW_CFG", "PROXY_B3_WINDOW_CFG", "PROXY_B3_DECISION", "PROXY_B3_PRESSURE"}:
        return True
    return event.startswith("PROXY_RECV_")


def target_collapi_from_summary(summary: Optional[dict]) -> Optional[str]:
    if not summary:
        return None
    collective = str(summary.get("collective", "") or "").strip().lower()
    return COLLECTIVE_TO_COLLAPI.get(collective)


def nccl_entry_matches_target(entry: dict, target_collapi: Optional[str]) -> bool:
    if not target_collapi:
        return True
    coll_api = str(entry.get("collApi", "") or "")
    coll = str(entry.get("coll", "") or "")
    return coll_api == target_collapi or coll == target_collapi


def add_window_sample(worker_info: dict, worker_steps: List[dict], entry: dict) -> None:
    try:
        t_ns = int(entry.get("tNs", 0) or 0)
        w_eff = float(entry["wEff"])
    except (TypeError, ValueError, KeyError):
        return
    worker_info["sample_index"] += 1
    step = lookup_worker_step(worker_steps, t_ns)
    worker_info["trace_x"].append(worker_info["sample_index"])
    worker_info["trace_step"].append(step if step is not None else -1)
    worker_info["trace_tns"].append(t_ns)
    worker_info["trace_w"].append(w_eff)
    worker_info["trace_w_sem"].append(float(entry.get("wSem", math.nan)) if "wSem" in entry else math.nan)
    worker_info["trace_w_fb"].append(float(entry.get("wFb", math.nan)) if "wFb" in entry else math.nan)
    worker_info["trace_pressure"].append(float(entry.get("pressureScore", math.nan)) if "pressureScore" in entry else math.nan)
    worker_info["trace_occ_ratio"].append(float(entry.get("occRatioPct", math.nan)) if "occRatioPct" in entry else math.nan)
    worker_info["trace_lag_ratio"].append(float(entry.get("lagRatioPct", math.nan)) if "lagRatioPct" in entry else math.nan)
    worker_info["trace_delay_ratio"].append(float(entry.get("delayRatioPct", math.nan)) if "delayRatioPct" in entry else math.nan)
    worker_info["w_values"].append(w_eff)
    if "wSem" in entry:
        worker_info["w_sem_values"].append(float(entry["wSem"]))
    if "wFb" in entry:
        worker_info["w_fb_values"].append(float(entry["wFb"]))


def summarize_nccl_logs(
    mode_dir: Path,
    worker_step_timings: Optional[Dict[str, List[dict]]] = None,
    target_collapi: Optional[str] = None,
) -> dict:
    worker_step_timings = worker_step_timings or {}
    log_paths = sorted(mode_dir.glob("*/nccl.*.log"))
    log_progress(f"scan NCCL logs mode={mode_dir.name} files={len(log_paths)} target_collapi={target_collapi or 'ALL'}")

    event_counts: Counter = Counter()
    worker_event_counts: Dict[str, Counter] = defaultdict(Counter)
    all_event_counts: Counter = Counter()
    filtered_event_counts: Counter = Counter()
    worker_wstall_counts: Counter = Counter()
    decision_counts: Counter = Counter()
    worker_decision_counts: Dict[str, Counter] = defaultdict(Counter)
    effective_decision_counts: Counter = Counter()
    worker_effective_decision_counts: Dict[str, Counter] = defaultdict(Counter)
    w_eff_counter: Counter = Counter()
    occ_pd_counter: Counter = Counter()
    occ_tr_counter: Counter = Counter()
    pressure_scores: List[float] = []
    occ_ratios: List[float] = []
    lag_ratios: List[float] = []
    delay_ratios: List[float] = []
    recv_lags: List[float] = []
    worker_windows: Dict[str, dict] = defaultdict(worker_window_defaults)

    recv_wstall_count = 0
    send_wstall_count = 0

    for log_path in log_paths:
        worker = log_path.parent.name
        worker_steps = worker_step_timings.get(worker, [])
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                entry = parse_nccl_event(raw_line)
                if entry is None:
                    continue
                event = str(entry["event"])
                all_event_counts[event] += 1
                if not nccl_entry_matches_target(entry, target_collapi):
                    filtered_event_counts[event] += 1
                    continue
                event_counts[event] += 1
                worker_event_counts[worker][event] += 1

                if "WSTALL" in event:
                    worker_wstall_counts[worker] += 1
                    if "RECV" in event:
                        recv_wstall_count += 1
                    if "SEND" in event:
                        send_wstall_count += 1

                if "wEff" in entry:
                    try:
                        w_eff_counter[float(entry["wEff"])] += 1
                    except (TypeError, ValueError):
                        pass
                if "occPd" in entry:
                    try:
                        occ_pd_counter[float(entry["occPd"])] += 1
                    except (TypeError, ValueError):
                        pass
                if "occTr" in entry:
                    try:
                        occ_tr_counter[float(entry["occTr"])] += 1
                    except (TypeError, ValueError):
                        pass
                if "pressureScore" in entry:
                    try:
                        pressure_scores.append(float(entry["pressureScore"]))
                    except (TypeError, ValueError):
                        pass
                for key, target in (
                    ("occRatioPct", occ_ratios),
                    ("lagRatioPct", lag_ratios),
                    ("delayRatioPct", delay_ratios),
                    ("recvLag", recv_lags),
                ):
                    if key in entry:
                        try:
                            target.append(float(entry[key]))
                        except (TypeError, ValueError):
                            pass

                if event == "PROXY_B3_DECISION":
                    reason = str(entry.get("reason", "unknown"))
                    decision_counts[reason] += 1
                    worker_decision_counts[worker][reason] += 1
                    step = lookup_worker_step(worker_steps, int(entry.get("tNs", 0) or 0))
                    worker_windows[worker]["decision_steps"].append(step if step is not None else -1)
                    worker_windows[worker]["decision_reasons"].append(reason)
                    try:
                        old_w = float(entry.get("oldW", 0.0))
                        new_w = float(entry.get("newW", 0.0))
                    except (TypeError, ValueError):
                        old_w = new_w = 0.0
                    if old_w != new_w:
                        if new_w < old_w:
                            effective_reason = "effective_shrink"
                        elif new_w > old_w:
                            effective_reason = "effective_recover"
                        else:
                            effective_reason = f"effective_{reason}"
                        effective_decision_counts[effective_reason] += 1
                        worker_effective_decision_counts[worker][effective_reason] += 1
                        worker_windows[worker]["effective_decision_steps"].append(step if step is not None else -1)
                        worker_windows[worker]["effective_decision_reasons"].append(effective_reason)
                        worker_windows[worker]["effective_decision_old_w"].append(old_w)
                        worker_windows[worker]["effective_decision_new_w"].append(new_w)

                if should_record_window_sample(entry):
                    add_window_sample(worker_windows[worker], worker_steps, entry)

    worker_summary: Dict[str, dict] = {}
    for worker, info in worker_windows.items():
        values = [float(v) for v in info["w_values"]]
        worker_summary[worker] = {
            "samples": len(values),
            "initial_w": values[0] if values else 0.0,
            "final_w": values[-1] if values else 0.0,
            "min_w": min(values) if values else 0.0,
            "max_w": max(values) if values else 0.0,
            "mean_w": mean(values),
            "p10_w": percentile(values, 0.10),
            "p90_w": percentile(values, 0.90),
            "decision_counts": dict(worker_decision_counts.get(worker, Counter())),
            "effective_decision_counts": dict(worker_effective_decision_counts.get(worker, Counter())),
            "wstall_count": int(worker_wstall_counts.get(worker, 0)),
        }

    return {
        "total_events": int(sum(event_counts.values())),
        "event_counts": event_counts,
        "total_events_all_collectives": int(sum(all_event_counts.values())),
        "filtered_events_non_target": int(sum(filtered_event_counts.values())),
        "target_collapi": target_collapi or "ALL",
        "worker_event_counts": worker_event_counts,
        "recv_wstall_count": int(recv_wstall_count),
        "send_wstall_count": int(send_wstall_count),
        "total_wstall_count": int(recv_wstall_count + send_wstall_count),
        "worker_wstall_counts": worker_wstall_counts,
        "decision_counts": decision_counts,
        "worker_decision_counts": worker_decision_counts,
        "effective_decision_counts": effective_decision_counts,
        "worker_effective_decision_counts": worker_effective_decision_counts,
        "effective_shrink_count": int(effective_decision_counts.get("effective_shrink", 0)),
        "effective_recover_count": int(effective_decision_counts.get("effective_recover", 0)),
        "w_eff_counter": w_eff_counter,
        "w_eff_values": sorted(w_eff_counter.keys(), key=float),
        "p99_occ_pd": percentile_from_counter(occ_pd_counter, 0.99),
        "p99_occ_tr": percentile_from_counter(occ_tr_counter, 0.99),
        "pressure_scores": pressure_scores,
        "pressure_score_p50": percentile(pressure_scores, 0.50),
        "pressure_score_p95": percentile(pressure_scores, 0.95),
        "occ_ratio_p95": percentile(occ_ratios, 0.95),
        "lag_ratio_p95": percentile(lag_ratios, 0.95),
        "delay_ratio_p95": percentile(delay_ratios, 0.95),
        "recv_lag_p95": percentile(recv_lags, 0.95),
        "worker_windows": worker_windows,
        "worker_summary": worker_summary,
    }


def load_mode_data(mode_dir: Path) -> dict:
    log_progress(f"load mode start mode_dir={mode_dir.as_posix()}")
    summary_files = sorted(mode_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"no *_summary.json found in {mode_dir}")
    summary_path = summary_files[0]
    summary = load_json(summary_path)
    step_files = sorted(mode_dir.glob("*_step_metrics.jsonl"))
    step_path = step_files[0] if step_files else None
    step_rows = load_jsonl(step_path) if step_path else []
    if "step_ms_avg" not in summary and step_rows:
        summary.update(compute_summary_from_steps(step_rows))
    if step_rows:
        summary.update(compute_step_tail_metrics(step_rows))

    worker_step_timings = load_worker_step_timings(mode_dir)
    target_collapi = target_collapi_from_summary(summary)
    nccl = summarize_nccl_logs(mode_dir, worker_step_timings, target_collapi)
    mode_name = str(summary.get("run_tag") or summary.get("phase3_mode") or summary.get("phase2_mode") or mode_dir.name).upper()
    return {
        "mode": mode_name,
        "summary": summary,
        "summary_path": summary_path,
        "step_path": step_path,
        "step_rows": step_rows,
        "mode_dir": mode_dir,
        "worker_step_timings": worker_step_timings,
        "nccl": nccl,
    }


def canonical_algo(value) -> str:
    text = str(value or "").strip()
    return text if text else "auto"


def workload_label(collective: str, algo: str) -> str:
    if collective == "allreduce":
        return f"{collective}_{str(algo or 'auto').lower()}"
    return f"{collective}_{str(algo or 'auto').lower()}"


def annotate_summary_rows(summary_rows: List[dict], env_setup: Optional[dict]) -> List[dict]:
    env_algo = canonical_algo((env_setup or {}).get("nccl_algo", ""))
    placement = str((env_setup or {}).get("placement", ""))
    threshold_keys = {
        "threshold_warmup": "phase3_warmup_intervals",
        "threshold_hi": "phase3_hi_intervals",
        "threshold_lo": "phase3_lo_intervals",
        "threshold_occ": "phase3_occ_ratio_high_pct",
        "threshold_lag": "phase3_lag_ratio_high_pct",
        "threshold_delay": "phase3_delay_ratio_high_pct",
    }
    for row in summary_rows:
        algo = canonical_algo(row.get("algo", ""))
        if algo == "auto" and env_algo != "auto":
            algo = env_algo
        row["algo"] = algo
        row["placement"] = placement
        row["workload"] = workload_label(str(row.get("collective", "")), algo)
        for out_key, env_key in threshold_keys.items():
            row[out_key] = (env_setup or {}).get(env_key, "")
        row["threshold_label"] = (
            f"warmup={row['threshold_warmup']},hi={row['threshold_hi']},lo={row['threshold_lo']},"
            f"occ={row['threshold_occ']},lag={row['threshold_lag']},delay={row['threshold_delay']}"
        )
    return summary_rows


def summarize_modes(mode_data: List[dict]) -> List[dict]:
    stock = next((item for item in mode_data if item["mode"] == "STOCK"), None)
    rows: List[dict] = []
    for item in mode_data:
        summary = item["summary"]
        nccl = item["nccl"]
        w_values = [float(v) for v in nccl["w_eff_counter"].elements()]
        row = {
            "mode": item["mode"],
            "algo": canonical_algo(summary.get("algo", "")),
            "collective": summary.get("collective", ""),
            "payload_mb": float(summary.get("payload_mb", 0.0)),
            "step_ms_avg": float(summary.get("step_ms_avg", 0.0)),
            "step_ms_p50": float(summary.get("step_ms_p50", 0.0)),
            "step_ms_p95": float(summary.get("step_ms_p95", 0.0)),
            "step_ms_p99": float(summary.get("step_ms_p99", 0.0)),
            "step_ms_max": float(summary.get("step_ms_max", 0.0)),
            "top_outlier_step": int(summary.get("top_outlier_step", -1)),
            "top_outlier_step_ms": float(summary.get("top_outlier_step_ms", 0.0)),
            "top_outlier_warmup": bool(summary.get("top_outlier_warmup", False)),
            "collective_gbps_avg": float(summary.get("collective_gbps_avg", 0.0)),
            "collective_gbps_p50": float(summary.get("collective_gbps_p50", 0.0)),
            "collective_gbps_p95": float(summary.get("collective_gbps_p95", 0.0)),
            "total_events": int(nccl["total_events"]),
            "total_events_all_collectives": int(nccl["total_events_all_collectives"]),
            "filtered_events_non_target": int(nccl["filtered_events_non_target"]),
            "target_collapi": str(nccl["target_collapi"]),
            "recv_wstall_count": int(nccl["recv_wstall_count"]),
            "send_wstall_count": int(nccl["send_wstall_count"]),
            "total_wstall_count": int(nccl["total_wstall_count"]),
            "p99_occ_pd": float(nccl["p99_occ_pd"]),
            "p99_occ_tr": float(nccl["p99_occ_tr"]),
            "w_eff_values": ",".join(str(int(v)) if float(v).is_integer() else str(v) for v in nccl["w_eff_values"]) or "-",
            "w_min": min(w_values) if w_values else 0.0,
            "w_mean": mean(w_values),
            "w_p90": percentile(w_values, 0.90),
            "decision_counts": ", ".join(f"{k}:{v}" for k, v in sorted(nccl["decision_counts"].items())) or "-",
            "effective_decision_counts": ", ".join(f"{k}:{v}" for k, v in sorted(nccl["effective_decision_counts"].items())) or "-",
            "effective_shrink_count": int(nccl["effective_shrink_count"]),
            "effective_recover_count": int(nccl["effective_recover_count"]),
            "pressure_score_p95": float(nccl["pressure_score_p95"]),
            "occ_ratio_p95": float(nccl["occ_ratio_p95"]),
            "lag_ratio_p95": float(nccl["lag_ratio_p95"]),
            "delay_ratio_p95": float(nccl["delay_ratio_p95"]),
            "delta_step_vs_stock_pct": 0.0,
            "delta_bw_vs_stock_pct": 0.0,
            "delta_occ_tr_vs_stock_pct": 0.0,
            "delta_wstall_vs_stock_pct": 0.0,
        }
        if stock is not None and item["mode"] != "STOCK":
            row["delta_step_vs_stock_pct"] = rel_change(float(stock["summary"].get("step_ms_avg", 0.0)), row["step_ms_avg"])
            row["delta_bw_vs_stock_pct"] = rel_change(float(stock["summary"].get("collective_gbps_avg", 0.0)), row["collective_gbps_avg"])
            row["delta_occ_tr_vs_stock_pct"] = rel_change(float(stock["nccl"]["p99_occ_tr"]), row["p99_occ_tr"])
            row["delta_wstall_vs_stock_pct"] = rel_change(float(stock["nccl"]["total_wstall_count"]), row["total_wstall_count"])
        rows.append(row)
    return rows


def extract_ts_ns(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value > 10**17:
            return value
        if value > 10**14:
            return value * 1000
        if value > 10**11:
            return value * 1_000_000
        return value * 1_000_000_000
    if isinstance(value, float):
        return int(value * 1_000_000_000) if value < 10**11 else int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return extract_ts_ns(int(raw))
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return int(dt.datetime.fromisoformat(normalized).timestamp() * 1_000_000_000)
        except ValueError:
            return None
    return None


def extract_record_ts_ns(record: dict) -> Optional[int]:
    for key in ("ts_mid_unix_ns", "ts_start_unix_ns", "ts_end_unix_ns", "ts_unix_ns", "timestamp_ns", "unix_ns"):
        if key in record:
            ts_ns = extract_ts_ns(record.get(key))
            if ts_ns is not None:
                return ts_ns
    return None


def parse_marker_message(message: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for token in str(message or "").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def resolve_switch_log_dir(
    experiment_root: Path,
    env_setup: Optional[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[Path]:
    if switch_log_dir_override:
        return Path(switch_log_dir_override)
    candidates: List[Path] = []
    if env_setup:
        for key in ("switch_log_local_dir", "switch_log_dir"):
            value = env_setup.get(key)
            if value:
                candidates.append(Path(str(value)))
        run_id = env_setup.get("switch_log_run_id")
        if run_id:
            candidates.append(SWITCH_SHARED_ROOT_DEFAULT / str(run_id))
            candidates.append(SWITCH_DOC_ROOT / str(run_id))
    matrix_switch_env = experiment_root.parent / "switch_logger.env"
    if matrix_switch_env.exists():
        for line in matrix_switch_env.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("SWITCH_LOG_LOCAL_DIR="):
                candidates.append(Path(line.split("=", 1)[1].strip()))
            if line.startswith("SWITCH_LOG_RUN_ID="):
                candidates.append(SWITCH_DOC_ROOT / line.split("=", 1)[1].strip())
    seen = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def load_aggregate_switch_file(path: Path, switch_label: str) -> Dict[str, List[Tuple[int, float]]]:
    fields: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        if ts_ns is None:
            continue
        if switch_label == "spine":
            field_names = {
                "rx": "rx_pause_packets_total",
                "tx": "tx_pause_packets_total",
                "rx_duration": "rx_pause_duration_total",
                "tx_duration": "tx_pause_duration_total",
            }
        else:
            field_names = {"rx": "rx_pause_total", "tx": "tx_pause_total"}
        for alias, key in field_names.items():
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields[alias].append((ts_ns, float(value)))
    for rows in fields.values():
        rows.sort(key=lambda item: item[0])
    return fields


def raw_switch_counter_value(row: dict, switch_label: str) -> Optional[float]:
    if switch_label == "spine":
        keys = ("rx_pause_packets", "tx_pause_packets")
    else:
        keys = ("rx_pause", "tx_pause")
    total = 0.0
    found = False
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += max(0.0, float(value))
            found = True
    return total if found else None


def load_raw_switch_file(path: Path, switch_label: str) -> Dict[str, List[Tuple[int, float]]]:
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        value = raw_switch_counter_value(row, switch_label)
        if ts_ns is None or value is None:
            continue
        sample_id = row.get("sample_id")
        if sample_id not in (None, ""):
            group_key = f"sample:{sample_id}"
        else:
            group_key = f"bucket:{(ts_ns // SWITCH_BUCKET_NS) * SWITCH_BUCKET_NS}"
        grouped[group_key].append((ts_ns, value))
    samples: List[Tuple[int, float]] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
        ts_ns = int(sum(ts for ts, _ in rows) / len(rows))
        samples.append((ts_ns, sum(value for _, value in rows)))
    samples.sort(key=lambda item: item[0])
    return {"rx": samples} if samples else {}


def load_switch_counter_snapshots(path: Path) -> List[Tuple[int, float]]:
    name = path.name
    if "spine" in name:
        fields = load_aggregate_switch_file(path, "spine") if "aggregate" in name else load_raw_switch_file(path, "spine")
    elif "rackA" in name:
        fields = load_aggregate_switch_file(path, "rackA") if "aggregate" in name else load_raw_switch_file(path, "rackA")
    else:
        fields = load_aggregate_switch_file(path, "rackB") if "aggregate" in name else load_raw_switch_file(path, "rackB")
    return fields.get("rx", [])


def compute_switch_delta_series(samples: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    deltas: List[Tuple[int, float]] = []
    for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
        elapsed = (cur_ts - prev_ts) / 1_000_000_000.0
        if elapsed <= 0:
            continue
        deltas.append((cur_ts, max(0.0, cur_total - prev_total) / elapsed))
    return deltas


def interpolate_switch_snapshot_value(snapshots: List[Tuple[int, float]], ts_ns: int) -> Optional[float]:
    if not snapshots:
        return None
    if ts_ns <= snapshots[0][0]:
        return float(snapshots[0][1])
    if ts_ns >= snapshots[-1][0]:
        return float(snapshots[-1][1])
    for (left_ts, left_value), (right_ts, right_value) in zip(snapshots, snapshots[1:]):
        if left_ts <= ts_ns <= right_ts:
            if right_ts == left_ts:
                return float(right_value)
            ratio = (ts_ns - left_ts) / float(right_ts - left_ts)
            return float(left_value + (right_value - left_value) * ratio)
    return None


def build_phase_aligned_cumulative_series(
    snapshots: List[Tuple[int, float]],
    start_ns: int,
    end_ns: int,
) -> List[Tuple[float, float]]:
    if end_ns <= start_ns:
        return []
    start_value = interpolate_switch_snapshot_value(snapshots, start_ns)
    if start_value is None:
        return []
    candidate_ts = [start_ns]
    candidate_ts.extend(ts for ts, _ in snapshots if start_ns < ts < end_ns)
    candidate_ts.append(end_ns)
    points: List[Tuple[float, float]] = []
    seen = set()
    for ts_ns in candidate_ts:
        if ts_ns in seen:
            continue
        seen.add(ts_ns)
        value = interpolate_switch_snapshot_value(snapshots, ts_ns)
        if value is None:
            continue
        points.append(((ts_ns - start_ns) / 1_000_000_000.0, max(0.0, value - start_value)))
    return points


def load_switch_bundle(
    experiment_root: Path,
    env_setup: Optional[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(experiment_root, env_setup, switch_log_dir_override)
    if log_dir is None or not log_dir.exists():
        return None

    aggregate_paths = {
        "rackA": log_dir / "rackA_pfc_aggregate.jsonl",
        "rackB": log_dir / "rackB_pfc_aggregate.jsonl",
        "spine": log_dir / "spine_roce_aggregate.jsonl",
    }
    raw_paths = {
        "rackA": log_dir / "rackA_pfc_statistics.jsonl",
        "rackB": log_dir / "rackB_pfc_statistics.jsonl",
        "spine": log_dir / "spine_roce_counters.jsonl",
    }

    switch_fields: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    snapshots: Dict[str, List[Tuple[int, float]]] = {}
    series: Dict[str, List[Tuple[int, float]]] = {}
    for label in SWITCH_LABELS:
        if aggregate_paths[label].exists():
            fields = load_aggregate_switch_file(aggregate_paths[label], label)
        elif raw_paths[label].exists():
            fields = load_raw_switch_file(raw_paths[label], label)
        else:
            fields = {}
        if fields:
            switch_fields[label] = fields
            primary = fields.get("rx") or fields.get("tx") or []
            snapshots[label] = primary
            series[label] = compute_switch_delta_series(primary)

    markers_path = log_dir / "markers.jsonl"
    markers = load_jsonl(markers_path) if markers_path.exists() else []
    if not snapshots and not markers:
        return None
    return {
        "log_dir": log_dir,
        "switch_fields": switch_fields,
        "snapshots": snapshots,
        "series": series,
        "markers": markers,
    }


def step_time_bounds(step_rows: List[dict]) -> Optional[Tuple[int, int]]:
    starts = [extract_ts_ns(row.get("ts_start_unix_ns")) for row in step_rows]
    ends = [extract_ts_ns(row.get("ts_end_unix_ns")) for row in step_rows]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if starts and ends:
        return min(starts), max(ends)
    mids = [extract_ts_ns(row.get("ts_mid_unix_ns")) for row in step_rows]
    mids = [value for value in mids if value is not None]
    if mids:
        return min(mids), max(mids)
    return None


def marker_matches_experiment(fields: Dict[str, str], env_setup: Optional[dict]) -> bool:
    if not env_setup:
        return True
    candidates = {
        str(env_setup.get("run_id", "") or ""),
        str(env_setup.get("experiment_label", "") or ""),
    }
    candidates = {value for value in candidates if value}
    if not candidates:
        return True
    marker_experiment = fields.get("experiment")
    marker_run_id = fields.get("run_id")
    return marker_experiment in candidates or marker_run_id in candidates


def find_mode_marker_bounds(markers: List[dict], env_setup: Optional[dict], mode: str) -> Optional[Tuple[int, int]]:
    starts: List[int] = []
    ends: List[int] = []
    target_mode = str(mode).upper()
    for record in markers:
        marker = str(record.get("marker") or "")
        fields = parse_marker_message(str(record.get("message") or ""))
        if str(fields.get("mode", "")).upper() != target_mode:
            continue
        if not marker_matches_experiment(fields, env_setup):
            continue
        ts_ns = extract_record_ts_ns(record)
        if ts_ns is None:
            continue
        if marker == "mode_start":
            starts.append(ts_ns)
        elif marker == "mode_end":
            ends.append(ts_ns)
    if starts and ends:
        return min(starts), max(ends)
    return None


def build_mode_time_windows(mode_data: List[dict], env_setup: Optional[dict], markers: List[dict]) -> Dict[str, Tuple[int, int]]:
    windows: Dict[str, Tuple[int, int]] = {}
    for item in mode_data:
        bounds = find_mode_marker_bounds(markers, env_setup, item["mode"])
        if bounds is None:
            bounds = step_time_bounds(item["step_rows"])
        if bounds is not None:
            windows[item["mode"]] = bounds
    return windows


def compute_switch_metrics(
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Dict[str, dict]:
    bundle = load_switch_bundle(experiment_root, env_setup, switch_log_dir_override)
    if not bundle:
        return {}
    windows = build_mode_time_windows(mode_data, env_setup, bundle.get("markers", []))
    metrics: Dict[str, dict] = {}
    for item in mode_data:
        mode = item["mode"]
        bounds = windows.get(mode)
        row = {
            "switch_pfc_total": 0.0,
            "switch_pfc_peak_rate": 0.0,
            "rackA_pfc_delta": 0.0,
            "rackB_pfc_delta": 0.0,
            "spine_pfc_delta": 0.0,
            "rackA_pfc_peak_rate": 0.0,
            "rackB_pfc_peak_rate": 0.0,
            "spine_pfc_peak_rate": 0.0,
        }
        if bounds is None:
            metrics[mode] = row
            continue
        start_ns, end_ns = bounds
        for label in SWITCH_LABELS:
            snapshots = bundle["snapshots"].get(label, [])
            start_value = interpolate_switch_snapshot_value(snapshots, start_ns)
            end_value = interpolate_switch_snapshot_value(snapshots, end_ns)
            if start_value is None or end_value is None:
                continue
            delta = max(0.0, end_value - start_value)
            row[f"{label}_pfc_delta"] = delta
            row["switch_pfc_total"] += delta
            rates = [rate for ts_ns, rate in compute_switch_delta_series(snapshots) if start_ns <= ts_ns <= end_ns]
            if rates:
                row[f"{label}_pfc_peak_rate"] = max(rates)
        row["switch_pfc_peak_rate"] = max(row["rackA_pfc_peak_rate"], row["rackB_pfc_peak_rate"], row["spine_pfc_peak_rate"])
        metrics[mode] = row

    stock = metrics.get("STOCK")
    if stock:
        for mode, row in metrics.items():
            if mode == "STOCK":
                row["delta_pfc_total_vs_stock_pct"] = 0.0
                row["delta_pfc_peak_vs_stock_pct"] = 0.0
            else:
                row["delta_pfc_total_vs_stock_pct"] = rel_change(stock["switch_pfc_total"], row["switch_pfc_total"])
                row["delta_pfc_peak_vs_stock_pct"] = rel_change(stock["switch_pfc_peak_rate"], row["switch_pfc_peak_rate"])
    return metrics


def save_pfc_overlay_plot(
    path: Path,
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[Path]:
    bundle = load_switch_bundle(experiment_root, env_setup, switch_log_dir_override)
    if not bundle or not bundle.get("snapshots"):
        return None
    windows = build_mode_time_windows(mode_data, env_setup, bundle.get("markers", []))
    if not windows:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=False)
    any_points = False
    for ax, label in zip(axes, SWITCH_LABELS):
        snapshots = bundle["snapshots"].get(label, [])
        for item in mode_data:
            mode = item["mode"]
            if mode not in windows:
                continue
            start_ns, end_ns = windows[mode]
            points = build_phase_aligned_cumulative_series(snapshots, start_ns, end_ns)
            if not points:
                continue
            any_points = True
            delta = points[-1][1]
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                linewidth=1.7,
                markersize=3.0,
                label=f"{mode} delta={delta:.0f}",
                color=MODE_COLORS.get(mode),
            )
        ylabel = "rx pause increase"
        if label == "spine":
            ylabel = "rx pause packets increase"
        ax.set_title(f"{label} PFC count increase")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    if not any_points:
        plt.close(fig)
        return None
    axes[-1].set_xlabel("seconds from mode_start")
    fig.suptitle(f"{experiment_root.name} STOCK/B3 PFC Delta Overlay", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_switch_overlay_plot(
    path: Path,
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[Path]:
    return save_pfc_overlay_plot(path, experiment_root, env_setup, mode_data, switch_log_dir_override)


def save_switch_phase_pfc_plot(
    path: Path,
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[Path]:
    return save_pfc_overlay_plot(path, experiment_root, env_setup, mode_data, switch_log_dir_override)


def save_event_counts_plot(path: Path, mode_data: List[dict], top_n: int) -> Optional[Path]:
    total = Counter()
    for item in mode_data:
        total.update(item["nccl"]["event_counts"])
    labels = [name for name, _ in total.most_common(top_n)]
    if not labels:
        return None
    series = []
    for item in mode_data:
        series.append((item["mode"], [float(item["nccl"]["event_counts"].get(label, 0)) for label in labels]))
    return save_grouped_bar(path, "Top NCCL Event Counts by Mode", labels, series, "count", rotate_labels=True)


def save_worker_wstall_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    workers = sorted({worker for item in mode_data for worker in item["nccl"]["worker_wstall_counts"].keys()}, key=worker_sort_key)
    if not workers:
        return None
    series = []
    for item in mode_data:
        counter = item["nccl"]["worker_wstall_counts"]
        series.append((item["mode"], [float(counter.get(worker, 0)) for worker in workers]))
    return save_grouped_bar(path, "Window Stall Count by Worker", workers, series, "WSTALL count")


def save_window_distribution_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    windows = sorted({window for item in mode_data for window in item["nccl"]["w_eff_counter"].keys()}, key=float)
    if not windows:
        return None
    labels = [str(int(w)) if float(w).is_integer() else f"{w:.2f}" for w in windows]
    series = []
    for item in mode_data:
        counter = item["nccl"]["w_eff_counter"]
        series.append((item["mode"], [float(counter.get(window, 0)) for window in windows]))
    return save_grouped_bar(path, "Receiver W_eff Distribution", labels, series, "count")


def save_decision_counts_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    workers = sorted({worker for item in mode_data for worker in item["nccl"]["worker_decision_counts"].keys()}, key=worker_sort_key)
    reasons = sorted({reason for item in mode_data for reason in item["nccl"]["decision_counts"].keys()})
    if not workers or not reasons:
        return None
    fig, ax = plt.subplots(figsize=(max(10, len(workers) * 0.8), 5.8))
    bottom = [0.0] * len(workers)
    b3_item = next((item for item in mode_data if item["mode"] == "B3"), None)
    if b3_item is None:
        return None
    for reason in reasons:
        values = [float(b3_item["nccl"]["worker_decision_counts"].get(worker, Counter()).get(reason, 0)) for worker in workers]
        ax.bar(workers, values, bottom=bottom, label=reason)
        bottom = [left + right for left, right in zip(bottom, values)]
    ax.set_title("B3 Decision Reason Counts by Worker")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_effective_decision_counts_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    b3_item = next((item for item in mode_data if item["mode"] == "B3"), None)
    if b3_item is None:
        return None
    workers = sorted(b3_item["nccl"]["worker_windows"].keys(), key=worker_sort_key)
    reasons = sorted(b3_item["nccl"]["effective_decision_counts"].keys())
    if not workers or not reasons:
        return None
    fig, ax = plt.subplots(figsize=(max(10, len(workers) * 0.8), 5.8))
    bottom = [0.0] * len(workers)
    for reason in reasons:
        values = [float(b3_item["nccl"]["worker_effective_decision_counts"].get(worker, Counter()).get(reason, 0)) for worker in workers]
        ax.bar(workers, values, bottom=bottom, label=reason)
        bottom = [left + right for left, right in zip(bottom, values)]
    ax.set_title("B3 Effective W Change Counts by Worker")
    ax.set_ylabel("oldW != newW count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_pressure_summary_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    b3_items = [item for item in mode_data if item["mode"] == "B3" and item["nccl"]["pressure_scores"]]
    if not b3_items:
        return None
    item = b3_items[0]
    modes = ["B3"]
    series = [
        ("pressure_p95", [float(item["nccl"]["pressure_score_p95"])]),
        ("occ_ratio_p95", [float(item["nccl"]["occ_ratio_p95"])]),
        ("lag_ratio_p95", [float(item["nccl"]["lag_ratio_p95"])]),
        ("delay_ratio_p95", [float(item["nccl"]["delay_ratio_p95"])]),
    ]
    return save_grouped_bar(path, "B3 Pressure Basis Summary", modes, series, "score / percent")


def save_occupancy_summary_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    modes = [item["mode"] for item in mode_data]
    if not modes:
        return None
    series = [
        ("p99_occ_pd", [float(item["nccl"]["p99_occ_pd"]) for item in mode_data]),
        ("p99_occ_tr", [float(item["nccl"]["p99_occ_tr"]) for item in mode_data]),
    ]
    return save_grouped_bar(path, "Outstanding Depth Summary by Mode", modes, series, "depth")


def save_delta_vs_stock_plot(path: Path, summary_rows: List[dict]) -> Optional[Path]:
    rows = [row for row in summary_rows if row["mode"] != "STOCK"]
    if not rows:
        return None
    categories = [row["mode"] for row in rows]
    series = [
        ("latency_pct", [float(row["delta_step_vs_stock_pct"]) for row in rows]),
        ("throughput_pct", [float(row["delta_bw_vs_stock_pct"]) for row in rows]),
        ("pfc_total_pct", [float(row.get("delta_pfc_total_vs_stock_pct", 0.0)) for row in rows]),
        ("wstall_pct", [float(row["delta_wstall_vs_stock_pct"]) for row in rows]),
    ]
    return save_grouped_bar(path, "Change vs STOCK", categories, series, "percent")


def trace_by_step(worker_info: dict, value_key: str = "trace_w") -> Tuple[List[int], List[float]]:
    steps = worker_info.get("trace_step", [])
    values = worker_info.get(value_key, [])
    per_step: Dict[int, float] = {}
    for step, value in zip(steps, values):
        if step is None or int(step) < 0:
            continue
        try:
            if math.isnan(float(value)):
                continue
        except (TypeError, ValueError):
            continue
        per_step[int(step)] = float(value)
    if per_step:
        ordered_steps = sorted(per_step.keys())
        return ordered_steps, [per_step[step] for step in ordered_steps]
    xs = worker_info.get("trace_x", [])
    clean_values = []
    clean_x = []
    for x_value, value in zip(xs, values):
        try:
            if math.isnan(float(value)):
                continue
        except (TypeError, ValueError):
            continue
        clean_x.append(int(x_value))
        clean_values.append(float(value))
    return clean_x, clean_values


def trace_by_step_pairs(worker_info: dict, value_key: str = "trace_w") -> Tuple[List[int], List[float]]:
    return trace_by_step(worker_info, value_key)


def save_worker_window_trace_plot(path: Path, mode_data_or_item, title: Optional[str] = None) -> Optional[Path]:
    if isinstance(mode_data_or_item, dict):
        mode_data = [mode_data_or_item]
    else:
        mode_data = list(mode_data_or_item)
    usable = [item for item in mode_data if item["nccl"]["worker_windows"]]
    if not usable:
        return None

    fig, axes = plt.subplots(len(usable), 1, figsize=(12, max(4.5, 3.8 * len(usable))), sharex=False)
    if len(usable) == 1:
        axes = [axes]
    any_points = False
    for ax, item in zip(axes, usable):
        for worker, info in sorted(item["nccl"]["worker_windows"].items(), key=lambda pair: worker_sort_key(pair[0])):
            xs, ys = trace_by_step(info, "trace_w")
            if not xs or not ys:
                continue
            any_points = True
            ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=2.3, label=worker)
        ax.set_title(f"{item['mode']} worker W_eff trace")
        ax.set_ylabel("W_eff")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
    if not any_points:
        plt.close(fig)
        return None
    axes[-1].set_xlabel("collective step if aligned, otherwise control sample")
    fig.suptitle(title or "Worker Receiver Window Trace", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_pressure_vs_w_plot(path: Path, mode_data: List[dict], experiment_name: str) -> Optional[Path]:
    b3 = next((item for item in mode_data if item["mode"] == "B3"), None)
    if b3 is None:
        return None
    worker_windows = b3["nccl"]["worker_windows"]
    workers = [worker for worker, info in sorted(worker_windows.items(), key=lambda pair: worker_sort_key(pair[0])) if info["trace_w"]]
    if not workers:
        return None

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    plotted = False
    for worker in workers:
        info = worker_windows[worker]
        xs, w_values = trace_by_step_pairs(info, "trace_w")
        pressure_xs, pressure = trace_by_step_pairs(info, "trace_pressure")
        occ_xs, occ = trace_by_step_pairs(info, "trace_occ_ratio")
        lag_xs, lag = trace_by_step_pairs(info, "trace_lag_ratio")
        delay_xs, delay = trace_by_step_pairs(info, "trace_delay_ratio")
        if xs and w_values:
            plotted = True
            axes[0].plot(xs, w_values, linewidth=1.2, marker="o", markersize=2.2, label=worker)
        if pressure_xs and pressure:
            axes[1].plot(pressure_xs, pressure, linewidth=1.0, alpha=0.75)
        if occ_xs and occ:
            axes[2].plot(occ_xs, occ, linewidth=1.0, alpha=0.75)
        if lag_xs and lag:
            axes[3].plot(lag_xs, lag, linewidth=1.0, alpha=0.75)
        if delay_xs and delay:
            axes[4].plot(delay_xs, delay, linewidth=1.0, alpha=0.75)
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_title(f"{experiment_name} B3 W and Pressure Basis")
    axes[0].set_ylabel("W_eff")
    axes[1].set_ylabel("pressureScore")
    axes[2].set_ylabel("occRatioPct")
    axes[3].set_ylabel("lagRatioPct")
    axes[4].set_ylabel("delayRatioPct")
    axes[4].set_xlabel("collective step if aligned, otherwise control sample")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_step_outlier_plot(path: Path, mode_data: List[dict], experiment_name: str) -> Optional[Path]:
    series = []
    for item in mode_data:
        rows = item["step_rows"]
        if not rows:
            continue
        xs = [int(row.get("step", idx)) for idx, row in enumerate(rows)]
        ys = [float(row.get("step_ms_max", 0.0)) for row in rows]
        series.append((item["mode"], xs, ys, rows))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for mode, xs, ys, rows in series:
        color = MODE_COLORS.get(mode)
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=mode, color=color)
        effective = [row for row in rows if not row.get("warmup", False)] or rows
        top_rows = sorted(effective, key=lambda row: float(row.get("step_ms_max", 0.0)), reverse=True)[:3]
        for row in top_rows:
            step = int(row.get("step", -1))
            value = float(row.get("step_ms_max", 0.0))
            ax.scatter([step], [value], s=60, color=color, edgecolors="black", linewidths=0.8, zorder=3)
            ax.annotate(f"{mode} s{step}", (step, value), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_title(f"{experiment_name} Step Latency Outliers")
    ax.set_xlabel("collective step")
    ax.set_ylabel("step_ms_max")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_worker_window_table(mode_item: dict) -> str:
    rows = []
    for worker, summary in sorted(mode_item["nccl"]["worker_summary"].items(), key=lambda pair: worker_sort_key(pair[0])):
        rows.append([
            worker,
            int(summary["samples"]),
            f'{summary["initial_w"]:.2f}',
            f'{summary["final_w"]:.2f}',
            f'{summary["min_w"]:.2f}',
            f'{summary["max_w"]:.2f}',
            f'{summary["mean_w"]:.2f}',
            f'{summary["p90_w"]:.2f}',
            summary["wstall_count"],
            ", ".join(f"{k}:{v}" for k, v in sorted(summary["decision_counts"].items())) or "-",
            ", ".join(f"{k}:{v}" for k, v in sorted(summary["effective_decision_counts"].items())) or "-",
        ])
    return render_table(["worker", "samples", "initial_w", "final_w", "min_w", "max_w", "mean_w", "p90_w", "wstall", "decision_counts", "effective_w_changes"], rows)


def build_effective_decision_table(mode_data: List[dict]) -> str:
    b3_item = next((item for item in mode_data if item["mode"] == "B3"), None)
    if b3_item is None:
        return "<p>No B3 mode found.</p>"
    rows = []
    for worker, counts in sorted(b3_item["nccl"]["worker_effective_decision_counts"].items(), key=lambda pair: worker_sort_key(pair[0])):
        rows.append([
            worker,
            counts.get("effective_shrink", 0),
            counts.get("effective_recover", 0),
            sum(counts.values()),
        ])
    if not rows:
        rows = [["-", 0, 0, 0]]
    return render_table(["worker", "effective_shrink", "effective_recover", "total_effective_changes"], rows)


def build_collection_card(experiment_root: Path, mode_data: List[dict], env_setup: Optional[dict], switch_log_dir_override: Optional[str]) -> str:
    total_logs = sum(len(list(item["mode_dir"].glob("*/nccl.*.log"))) for item in mode_data)
    total_step_timing = sum(len(list(item["mode_dir"].glob("*/**/*_worker_step_timing.jsonl"))) for item in mode_data)
    switch_dir = resolve_switch_log_dir(experiment_root, env_setup, switch_log_dir_override)
    rows = [
        ["env_setup.json", "Experiment metadata and B3 controller parameters."],
        ["MODE/*_summary.json", "Mode-level latency and throughput summary."],
        ["MODE/*_step_metrics.jsonl", "Per-step latency/throughput timeline."],
        [f"MODE/workerXX/*_worker_step_timing.jsonl ({total_step_timing})", "Worker-local step timing for W trace alignment."],
        [f"MODE/workerXX/nccl.*.log ({total_logs})", "PHASE0/PHASE1/PHASE3 W, pressure, decision, and WSTALL events filtered to the target collApi."],
        ["switch aggregate jsonl", str(switch_dir) if switch_dir else "not found"],
    ]
    return '<div class="card"><h2>Collected Data</h2>' + render_table(["source", "meaning"], rows) + "</div>"


def build_visualization_card() -> str:
    rows = [
        ["pfc_overlay_<experiment>.png", "rackA/rackB/spine STOCK vs B3 cumulative PFC delta from mode_start to mode_end."],
        ["worker_w_trace_<experiment>.png", "Worker-level STOCK/B3 W_eff trace and stock W confirmation."],
        ["b3_decision_<experiment>.png", "B3 decision reason counts by worker."],
        ["b3_effective_decision_<experiment>.png", "B3 effective W changes where oldW != newW."],
        ["pressure_vs_w_<experiment>.png", "B3 pressureScore and pressure components against W_eff."],
        ["step_outliers_<experiment>.png", "Per-step latency timeline with top outlier steps highlighted."],
        ["summary plots", "Latency, throughput, W distribution, WSTALL, and stock-relative deltas."],
    ]
    return '<div class="card"><h2>Visualization Outputs</h2>' + render_table(["plot", "meaning"], rows) + "</div>"


def load_validation_summary(experiment_root: Path) -> dict:
    compare_path = experiment_root / "final_output_validation.json"
    rank_paths = sorted(experiment_root.glob("*/worker*/*_rank_validation.json"))
    result = {"compare_status": "missing", "rank_validation_files": len(rank_paths)}
    if compare_path.exists():
        try:
            result["compare_status"] = "present"
            result["compare"] = load_json(compare_path)
        except json.JSONDecodeError:
            result["compare_status"] = "invalid_json"
    return result


def merge_switch_metrics(summary_rows: List[dict], switch_metrics: Dict[str, dict]) -> None:
    stock_switch = switch_metrics.get("STOCK", {})
    for row in summary_rows:
        metrics = switch_metrics.get(row["mode"], {})
        for key in (
            "switch_pfc_total",
            "switch_pfc_peak_rate",
            "rackA_pfc_delta",
            "rackB_pfc_delta",
            "spine_pfc_delta",
            "rackA_pfc_peak_rate",
            "rackB_pfc_peak_rate",
            "spine_pfc_peak_rate",
            "delta_pfc_total_vs_stock_pct",
            "delta_pfc_peak_vs_stock_pct",
        ):
            row[key] = float(metrics.get(key, 0.0))
        if stock_switch and row["mode"] != "STOCK" and "delta_pfc_total_vs_stock_pct" not in metrics:
            row["delta_pfc_total_vs_stock_pct"] = rel_change(float(stock_switch.get("switch_pfc_total", 0.0)), row["switch_pfc_total"])
            row["delta_pfc_peak_vs_stock_pct"] = rel_change(float(stock_switch.get("switch_pfc_peak_rate", 0.0)), row["switch_pfc_peak_rate"])


def build_single_experiment_report(
    experiment_root: Path,
    output_dir: Path,
    top_events: int,
    switch_log_dir_override: Optional[str] = None,
) -> Tuple[Path, List[dict]]:
    log_progress(f"build single report start experiment={experiment_root.name}")
    env_setup = load_json(experiment_root / "env_setup.json") if (experiment_root / "env_setup.json").exists() else None
    mode_dirs = find_mode_dirs(experiment_root)
    if not mode_dirs:
        raise FileNotFoundError(f"no mode directories found in {experiment_root}")

    mode_data = [load_mode_data(mode_dir) for mode_dir in mode_dirs]
    mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
    summary_rows = annotate_summary_rows(summarize_modes(mode_data), env_setup)
    switch_metrics = compute_switch_metrics(experiment_root, env_setup, mode_data, switch_log_dir_override)
    merge_switch_metrics(summary_rows, switch_metrics)
    validation = load_validation_summary(experiment_root)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    step_series = []
    bw_series = []
    for item in mode_data:
        rows = item["step_rows"]
        if not rows:
            continue
        steps = [int(row.get("step", idx)) for idx, row in enumerate(rows)]
        step_series.append((f"{item['mode']} step_ms_max", steps, [float(row.get("step_ms_max", 0.0)) for row in rows]))
        bw_series.append((item["mode"], steps, [float(row.get("collective_gbps_est", 0.0)) for row in rows]))

    p_step = save_line_plot(plots_dir / "step_timeline.png", f"{experiment_root.name} Step Latency", "collective step", "ms", step_series)
    p_bw = save_line_plot(plots_dir / "throughput_timeline.png", f"{experiment_root.name} Throughput", "collective step", "Gbps", bw_series)
    p_latency = save_grouped_bar(
        plots_dir / "summary_latency.png",
        "Latency Summary by Mode",
        [row["mode"] for row in summary_rows],
        [
            ("avg", [row["step_ms_avg"] for row in summary_rows]),
            ("p95", [row["step_ms_p95"] for row in summary_rows]),
            ("p99", [row["step_ms_p99"] for row in summary_rows]),
            ("max", [row["step_ms_max"] for row in summary_rows]),
        ],
        "ms",
    )
    p_throughput = save_grouped_bar(
        plots_dir / "summary_throughput.png",
        "Throughput Summary by Mode",
        [row["mode"] for row in summary_rows],
        [("avg", [row["collective_gbps_avg"] for row in summary_rows]), ("p95", [row["collective_gbps_p95"] for row in summary_rows])],
        "Gbps",
    )
    p_pfc = save_pfc_overlay_plot(plots_dir / f"pfc_overlay_{experiment_root.name}.png", experiment_root, env_setup, mode_data, switch_log_dir_override)
    p_worker_w = save_worker_window_trace_plot(plots_dir / f"worker_w_trace_{experiment_root.name}.png", mode_data, f"{experiment_root.name} Worker W_eff Trace")
    p_decision = save_decision_counts_plot(plots_dir / f"b3_decision_{experiment_root.name}.png", mode_data)
    p_effective_decision = save_effective_decision_counts_plot(plots_dir / f"b3_effective_decision_{experiment_root.name}.png", mode_data)
    p_pressure_w = save_pressure_vs_w_plot(plots_dir / f"pressure_vs_w_{experiment_root.name}.png", mode_data, experiment_root.name)
    p_step_outlier = save_step_outlier_plot(plots_dir / f"step_outliers_{experiment_root.name}.png", mode_data, experiment_root.name)
    p_events = save_event_counts_plot(plots_dir / "event_counts.png", mode_data, top_events)
    p_wstall = save_worker_wstall_plot(plots_dir / "wstall_by_worker.png", mode_data)
    p_window_dist = save_window_distribution_plot(plots_dir / "window_distribution.png", mode_data)
    p_occ = save_occupancy_summary_plot(plots_dir / "occupancy_summary.png", mode_data)
    p_delta = save_delta_vs_stock_plot(plots_dir / "delta_vs_stock.png", summary_rows)

    info_rows = [["experiment_root", experiment_root.as_posix()], ["modes", ", ".join(item["mode"] for item in mode_data)]]
    if env_setup:
        for key in (
            "run_id",
            "experiment_label",
            "collective",
            "nccl_algo",
            "payload_mb",
            "nnodes",
            "worker_pool",
            "policy_formula",
            "phase3_warmup_intervals",
            "phase3_hi_intervals",
            "phase3_lo_intervals",
            "phase3_occ_ratio_high_pct",
            "phase3_lag_ratio_high_pct",
            "phase3_delay_ratio_high_pct",
            "switch_log_run_id",
            "switch_log_local_dir",
        ):
            if key in env_setup:
                info_rows.append([key, env_setup[key]])
    info_rows.append(["validation", validation.get("compare_status", "missing")])

    summary_headers = [
        "mode",
        "workload",
        "target_collapi",
        "filtered_non_target",
        "step_ms_avg",
        "step_ms_p95",
        "step_ms_p99",
        "step_ms_max",
        "top_outlier_step",
        "top_outlier_step_ms",
        "gbps_avg",
        "pfc_total",
        "rackA_delta",
        "rackB_delta",
        "spine_delta",
        "w_values",
        "w_min",
        "w_mean",
        "p99_occ_tr",
        "wstall",
        "decision_counts",
        "effective_w_changes",
        "lat_vs_stock_pct",
        "bw_vs_stock_pct",
        "pfc_vs_stock_pct",
        "threshold",
    ]
    summary_table = []
    for row in summary_rows:
        summary_table.append([
            row["mode"],
            row.get("workload", ""),
            row["target_collapi"],
            row["filtered_events_non_target"],
            f'{row["step_ms_avg"]:.3f}',
            f'{row["step_ms_p95"]:.3f}',
            f'{row["step_ms_p99"]:.3f}',
            f'{row["step_ms_max"]:.3f}',
            row["top_outlier_step"],
            f'{row["top_outlier_step_ms"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            f'{row["switch_pfc_total"]:.0f}',
            f'{row["rackA_pfc_delta"]:.0f}',
            f'{row["rackB_pfc_delta"]:.0f}',
            f'{row["spine_pfc_delta"]:.0f}',
            row["w_eff_values"],
            f'{row["w_min"]:.2f}',
            f'{row["w_mean"]:.2f}',
            f'{row["p99_occ_tr"]:.2f}',
            row["total_wstall_count"],
            row["decision_counts"],
            row["effective_decision_counts"],
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
            f'{row["delta_pfc_total_vs_stock_pct"]:.2f}',
            row["threshold_label"],
        ])

    sections = [
        '<section class="section"><h2>Experiment Metadata</h2>' + render_table(["field", "value"], info_rows) + "</section>",
        '<section class="section"><div class="grid-2">' + build_collection_card(experiment_root, mode_data, env_setup, switch_log_dir_override) + build_visualization_card() + "</div></section>",
        '<section class="section"><h2>Mode Summary</h2>' + render_table(summary_headers, summary_table) + "</section>",
    ]

    for item in mode_data:
        sections.append(f'<section class="section"><h2>{html.escape(item["mode"])} Worker W Summary</h2>{build_worker_window_table(item)}</section>')
    sections.append('<section class="section"><h2>B3 Effective W Changes</h2>' + build_effective_decision_table(mode_data) + "</section>")

    figure_paths = [
        (p_pfc, "rackA/rackB/spine PFC cumulative count increase overlaid by mode"),
        (p_worker_w, "worker-level receiver W_eff trace, including STOCK W confirmation"),
        (p_decision, "B3 decision reason counts by worker"),
        (p_effective_decision, "B3 effective W changes where oldW != newW"),
        (p_pressure_w, "B3 pressure basis vs W_eff"),
        (p_step_outlier, "step latency outliers with top steps highlighted"),
        (p_step, "step latency timeline"),
        (p_bw, "throughput timeline"),
        (p_latency, "latency summary"),
        (p_throughput, "throughput summary"),
        (p_delta, "change vs STOCK"),
        (p_window_dist, "W_eff distribution"),
        (p_occ, "posted/transmitted occupancy summary"),
        (p_wstall, "WSTALL count by worker"),
        (p_events, "top NCCL event counts"),
    ]
    figures = []
    for plot_path, caption in figure_paths:
        if plot_path is None or not plot_path.exists():
            continue
        figures.append(
            f'<figure><img src="{html.escape(relpath(plot_path, output_dir))}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figures) + "</section>")

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "phase3_report.html"
    html_path.write_text(render_html(f"Phase3 B3 Report - {experiment_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase3_report.json"
    report_json.write_text(
        json.dumps({"experiment": experiment_root.name, "env_setup": env_setup, "mode_summary": summary_rows, "validation": validation}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_csv = output_dir / "phase3_report.csv"
    if summary_rows:
        with report_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    log_progress(f"build single report complete html={html_path.as_posix()}")
    return html_path, summary_rows


def save_matrix_tradeoff_plot(path: Path, rows: List[dict]) -> Optional[Path]:
    b3_rows = [row for row in rows if row["mode"] == "B3"]
    if not b3_rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for row in b3_rows:
        x_val = -float(row.get("delta_pfc_total_vs_stock_pct", 0.0))
        y_val = -float(row.get("delta_step_vs_stock_pct", 0.0))
        ax.scatter([x_val], [y_val], s=70)
        ax.annotate(str(row["experiment"]), (x_val, y_val), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.axhline(0.0, color="#6c7f99", linewidth=0.9, linestyle="--")
    ax.axvline(0.0, color="#6c7f99", linewidth=0.9, linestyle="--")
    ax.set_xlabel("PFC total reduction vs STOCK (%)")
    ax.set_ylabel("latency improvement vs STOCK (%)")
    ax.set_title("B3 Benefit Tradeoff")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_matrix_bar(path: Path, rows: List[dict], metric: str, title: str, ylabel: str) -> Optional[Path]:
    experiments = sorted({row["experiment"] for row in rows})
    modes = sorted({row["mode"] for row in rows}, key=mode_sort_key)
    series = []
    for mode in modes:
        values = []
        for experiment in experiments:
            row = next((item for item in rows if item["experiment"] == experiment and item["mode"] == mode), None)
            values.append(float(row.get(metric, 0.0)) if row else 0.0)
        series.append((mode, values))
    return save_grouped_bar(path, title, experiments, series, ylabel, rotate_labels=True)


def build_matrix_report(
    matrix_root: Path,
    output_dir: Path,
    top_events: int,
    switch_log_dir_override: Optional[str] = None,
) -> Path:
    experiment_dirs = find_experiment_dirs(matrix_root)
    if not experiment_dirs:
        raise FileNotFoundError(f"no experiment directories found in {matrix_root}")
    log_progress(f"build matrix report start matrix={matrix_root.as_posix()} experiments={len(experiment_dirs)}")

    rows: List[dict] = []
    links: Dict[str, str] = {}
    for experiment_dir in experiment_dirs:
        per_output = output_dir / experiment_dir.name
        per_html, per_rows = build_single_experiment_report(experiment_dir, per_output, top_events, switch_log_dir_override)
        links[experiment_dir.name] = relpath(per_html, output_dir)
        for row in per_rows:
            flat = dict(row)
            flat["experiment"] = experiment_dir.name
            rows.append(flat)

    plots_dir = output_dir / "plots"
    p_latency = save_matrix_bar(plots_dir / "matrix_latency.png", rows, "step_ms_avg", "Matrix Latency", "step_ms_avg")
    p_throughput = save_matrix_bar(plots_dir / "matrix_throughput.png", rows, "collective_gbps_avg", "Matrix Throughput", "Gbps")
    p_pfc = save_matrix_bar(plots_dir / "matrix_pfc_total.png", rows, "switch_pfc_total", "Matrix Switch PFC Delta", "PFC delta")
    p_pfc_change = save_matrix_bar(plots_dir / "matrix_pfc_delta_vs_stock.png", rows, "delta_pfc_total_vs_stock_pct", "PFC Change vs STOCK", "percent")
    p_tradeoff = save_matrix_tradeoff_plot(plots_dir / "matrix_b3_tradeoff.png", rows)

    headers = [
        "experiment",
        "mode",
        "workload",
        "target_collapi",
        "filtered_non_target",
        "step_ms_avg",
        "step_ms_p99",
        "step_ms_max",
        "top_outlier_step",
        "gbps_avg",
        "pfc_total",
        "rackA",
        "rackB",
        "spine",
        "lat_vs_stock_pct",
        "bw_vs_stock_pct",
        "pfc_vs_stock_pct",
        "effective_w_changes",
        "threshold",
        "report",
    ]
    table_rows = []
    for row in rows:
        link = links.get(row["experiment"], "")
        table_rows.append([
            html.escape(str(row["experiment"])),
            html.escape(str(row["mode"])),
            html.escape(str(row.get("workload", ""))),
            html.escape(str(row.get("target_collapi", ""))),
            str(row.get("filtered_events_non_target", 0)),
            f'{row["step_ms_avg"]:.3f}',
            f'{row.get("step_ms_p99", 0.0):.3f}',
            f'{row.get("step_ms_max", 0.0):.3f}',
            str(row.get("top_outlier_step", "")),
            f'{row["collective_gbps_avg"]:.3f}',
            f'{row.get("switch_pfc_total", 0.0):.0f}',
            f'{row.get("rackA_pfc_delta", 0.0):.0f}',
            f'{row.get("rackB_pfc_delta", 0.0):.0f}',
            f'{row.get("spine_pfc_delta", 0.0):.0f}',
            f'{row.get("delta_step_vs_stock_pct", 0.0):.2f}',
            f'{row.get("delta_bw_vs_stock_pct", 0.0):.2f}',
            f'{row.get("delta_pfc_total_vs_stock_pct", 0.0):.2f}',
            html.escape(str(row.get("effective_decision_counts", "-"))),
            html.escape(str(row.get("threshold_label", ""))),
            f'<a href="{html.escape(link)}">open</a>' if link else "",
        ])

    sections = [
        '<section class="section"><h2>Matrix Summary</h2>' + render_table_raw(headers, table_rows) + "</section>",
    ]
    figures = []
    for plot_path, caption in [
        (p_latency, "experiment/mode latency"),
        (p_throughput, "experiment/mode throughput"),
        (p_pfc, "experiment/mode PFC start-stop delta"),
        (p_pfc_change, "PFC total change vs STOCK"),
        (p_tradeoff, "B3 PFC reduction vs latency improvement tradeoff"),
    ]:
        if plot_path is None or not plot_path.exists():
            continue
        figures.append(
            f'<figure><img src="{html.escape(relpath(plot_path, output_dir))}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figures) + "</section>")
    report_list = "".join(f'<li><a href="{html.escape(link)}">{html.escape(name)}</a></li>' for name, link in sorted(links.items()))
    sections.append('<section class="section"><h2>Per Experiment Reports</h2><ul>' + report_list + "</ul></section>")

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "phase3_report.html"
    html_path.write_text(render_html(f"Phase3 B3 Matrix Report - {matrix_root.name}", sections), encoding="utf-8")
    (output_dir / "phase3_report.json").write_text(json.dumps({"matrix_root": matrix_root.name, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (output_dir / "phase3_report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    log_progress(f"build matrix report complete html={html_path.as_posix()}")
    return html_path


def main() -> None:
    global REPORT_LOG_PATH
    args = parse_args()
    input_path = Path(args.input).resolve()
    if input_path.suffix.lower() == ".zip":
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.with_name(f"{input_path.stem}_report")
        output_dir.mkdir(parents=True, exist_ok=True)
        REPORT_LOG_PATH = output_dir / "phase3_reporter.log"
        REPORT_LOG_PATH.write_text("", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="phase3_log_reporter_") as tempdir:
            temp_root = Path(tempdir)
            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(temp_root)
                top_level_dirs = sorted({Path(name).parts[0] for name in zf.namelist() if name.strip("/")})
            if not top_level_dirs:
                raise FileNotFoundError(f"zip archive is empty: {input_path}")
            root = temp_root / top_level_dirs[0]
            html_path = build_matrix_report(root, output_dir, args.top_events, args.switch_log_dir) if is_matrix_root(root) else build_single_experiment_report(root, output_dir, args.top_events, args.switch_log_dir)[0]
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path / "report"
        output_dir.mkdir(parents=True, exist_ok=True)
        REPORT_LOG_PATH = output_dir / "phase3_reporter.log"
        REPORT_LOG_PATH.write_text("", encoding="utf-8")
        html_path = build_matrix_report(input_path, output_dir, args.top_events, args.switch_log_dir) if is_matrix_root(input_path) else build_single_experiment_report(input_path, output_dir, args.top_events, args.switch_log_dir)[0]
    log_progress(f"wrote html={html_path.as_posix()}")


if __name__ == "__main__":
    main()
