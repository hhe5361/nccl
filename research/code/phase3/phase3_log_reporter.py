#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import html
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for phase3_log_reporter.py. Install it first, for example: pip install matplotlib"
    ) from exc


PHASE_RE = __import__("re").compile(r"(PHASE\d+)\s+(.*)")
KV_RE = __import__("re").compile(r"(\w+)=([^\s]+)")
WORKER_RE = __import__("re").compile(r"worker(\d+)$")
EXPERIMENT_RE = __import__("re").compile(r"^\d+_.+")

MODE_ORDER = {"STOCK": 0, "B2": 1, "B3": 2}
REPORT_LOG_PATH: Optional[Path] = None
SWITCH_SHARED_ROOT_DEFAULT = Path("/mnt/nfs/cts_experiments/switch_log")
SWITCH_AGGREGATE_BUCKET_NS = 1_000_000_000


RUN_SUMMARY_FIELD_HELP = {
    "mode": "실험 모드. STOCK, B2, B3.",
    "placement": "실험 placement. intra-rack, inter-rack, mixed-8.",
    "collective": "실험 collective 종류.",
    "algo": "실험에서 강제하거나 선택된 NCCL algorithm.",
    "workload": "보고서에서 쓰는 workload label. allreduce는 ring/tree를 포함해 구분한다.",
    "payload_mb": "rank당 logical payload 크기(MiB).",
    "step_ms_avg": "step max latency 평균(ms).",
    "step_ms_p95": "step max latency p95(ms).",
    "collective_gbps_avg": "collective 예상 traffic volume 기반 처리량 평균(Gbps).",
    "collective_gbps_p95": "collective 예상 처리량 p95(Gbps).",
    "delta_step_vs_stock_pct": "같은 experiment 안에서 stock 대비 step_ms_avg 변화율(%). 음수면 latency 개선.",
    "delta_bw_vs_stock_pct": "같은 experiment 안에서 stock 대비 collective_gbps_avg 변화율(%). 양수면 throughput 개선.",
    "delta_occ_tr_vs_stock_pct": "stock 대비 p99_occ_tr 변화율(%). 음수면 outstanding depth 감소.",
    "delta_wstall_vs_stock_pct": "stock 대비 total_wstall_count 변화율(%). 음수면 stall 감소.",
    "delta_step_vs_b2_pct": "B2 대비 step_ms_avg 변화율(%). 음수면 latency 개선.",
    "delta_bw_vs_b2_pct": "B2 대비 collective_gbps_avg 변화율(%). 양수면 throughput 개선.",
}

NCCL_SUMMARY_FIELD_HELP = {
    "total_events": "수집된 PHASE 이벤트 총 수.",
    "recv_wstall_count": "RECV 관련 WSTALL 총 횟수.",
    "send_wstall_count": "SEND 관련 WSTALL 총 횟수.",
    "p99_occ_pd": "posted-done의 p99.",
    "p99_occ_tr": "transmitted-done의 p99.",
    "w_eff_values": "receiver-side에서 관측한 effective window 집합.",
    "decision_counts": "B3 decision 이벤트의 reason별 빈도.",
    "pressure_score_p95": "pressureScore 계열 로그의 p95.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 3 PNG plots and HTML report")
    parser.add_argument("--input", required=True, help="Single experiment root, matrix root, or zip")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <input>_report or <input>/report")
    parser.add_argument("--top-events", type=int, default=12, help="Number of top NCCL events to visualize")
    parser.add_argument("--switch-log-dir", default=None, help="Override switch log directory for all experiments in this report")
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


def percentile_from_counter(counter: Counter, q: float) -> float:
    if not counter:
        return 0.0
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    threshold = max(1, math.ceil(total * q))
    seen = 0
    for value in sorted(counter.keys(), key=float):
        seen += counter[value]
        if seen >= threshold:
            return float(value)
    return float(max(counter.keys(), key=float))


def rel_change(base: float, candidate: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return 100.0 * (candidate - base) / base


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


def worker_sort_key(name: str) -> Tuple[int, str]:
    match = WORKER_RE.match(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def mode_sort_key(name: str) -> Tuple[int, str]:
    return MODE_ORDER.get(name.upper(), 10**9), name


def relpath(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    rows: List[dict] = []
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        row, next_index = decoder.raw_decode(text, index)
        rows.append(row)
        index = next_index
    return rows


def load_worker_step_timings(mode_dir: Path) -> Dict[str, List[dict]]:
    timing_map: Dict[str, List[dict]] = {}
    for timing_path in sorted(mode_dir.glob("*/**/*_worker_step_timing.jsonl")):
        rows = load_jsonl(timing_path)
        if not rows:
            continue
        worker = timing_path.parent.name
        timing_map.setdefault(worker, []).extend(rows)
    for worker, rows in timing_map.items():
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


def positive_ylim(max_value: float) -> Optional[Tuple[float, float]]:
    if max_value <= 0:
        return None
    return 0.0, max_value * 1.05


def value_ylim(values: List[float]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    min_value = min(values)
    max_value = max(values)
    if min_value >= 0:
        return positive_ylim(max_value)
    span = max_value - min_value
    margin = max(span * 0.05, 0.1)
    lower = min_value - margin
    upper = max_value + margin
    if lower > 0:
        lower = 0.0
    if upper < 0:
        upper = 0.0
    return lower, upper


def save_line_plot(path: Path, title: str, xlabel: str, ylabel: str, series: List[Tuple[str, List[float], List[float]]]) -> Optional[Path]:
    if not series:
        return None
    plt.figure(figsize=(11, 5.5))
    max_value = 0.0
    for label, x_values, y_values in series:
        if not x_values or not y_values:
            continue
        plt.plot(x_values, y_values, marker="o", linewidth=1.7, markersize=3, label=label)
        max_value = max(max_value, max(y_values))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ylim = positive_ylim(max_value)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_grouped_bar(path: Path, title: str, categories: List[str], series: List[Tuple[str, List[float]]], ylabel: str, rotate_labels: bool = False) -> Optional[Path]:
    if not categories or not series:
        return None
    plt.figure(figsize=(max(10, len(categories) * 0.9), 5.8))
    x = list(range(len(categories)))
    width = 0.8 / max(len(series), 1)
    all_values: List[float] = []
    for idx, (label, values) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        plt.bar([pos + offset for pos in x], values, width=width, label=label)
        all_values.extend(float(v) for v in values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, categories, rotation=35 if rotate_labels else 0, ha="right" if rotate_labels else "center")
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


def render_field_help_card(title: str, field_help: Dict[str, str], wanted_fields: List[str]) -> str:
    rows = [[field, field_help[field]] for field in wanted_fields if field in field_help]
    return '<div class="card"><h2>' + html.escape(title) + '</h2>' + render_table(["field", "meaning"], rows) + '</div>'


def render_plot_help_card(title: str, rows: List[Tuple[str, str, str]]) -> str:
    formatted = [[name, source, meaning] for name, source, meaning in rows]
    return '<div class="card"><h2>' + html.escape(title) + '</h2>' + render_table(["plot", "source", "meaning"], formatted) + '</div>'


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
    .page {{
      width: min(1240px, calc(100% - 32px));
      margin: 24px auto 48px;
    }}
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
    h1 {{
      margin: 14px 0 10px;
      font-size: clamp(30px, 4vw, 42px);
      line-height: 1.08;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 24px;
      line-height: 1.2;
    }}
    p {{ margin: 10px 0 0; color: var(--muted); }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .chip {{
      border: 1px solid var(--border);
      background: var(--panel-soft);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .grid-2 {{ display: grid; gap: 14px; margin-top: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .card {{
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f4f7fb; }}
    figure {{
      margin: 18px 0 0;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--panel-soft);
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      background: #fff;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    ul {{ margin: 10px 0 0; padding-left: 20px; }}
    li + li {{ margin-top: 8px; }}
    code {{
      font-family: Consolas, "SFMono-Regular", Monaco, monospace;
      font-size: 0.93em;
      background: #f2f5fa;
      border: 1px solid #e1e7f0;
      border-radius: 6px;
      padding: 1px 6px;
      color: #1d2b3a;
    }}
    @media (max-width: 920px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">Phase3 Reporter</div>
      <h1>{html.escape(title)}</h1>
      <p>Phase3 runner가 남긴 env_setup, summary, step metrics, NCCL PHASE logs를 읽어 PNG plot과 HTML report를 생성한다.</p>
    </header>
    {''.join(sections)}
  </div>
</body>
</html>"""


def find_mode_dirs(experiment_root: Path) -> List[Path]:
    mode_dirs: List[Path] = []
    for child in sorted(experiment_root.iterdir(), key=lambda p: mode_sort_key(p.name)):
        if not child.is_dir():
            continue
        if list(child.glob("*_summary.json")) or list(child.glob("*_step_metrics.jsonl")):
            mode_dirs.append(child)
    return mode_dirs


def find_experiment_dirs(matrix_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for child in sorted(matrix_root.iterdir()):
        if child.is_dir() and EXPERIMENT_RE.match(child.name):
            if child.name == ".matrix_status":
                continue
            if child.joinpath("env_setup.json").exists() or find_mode_dirs(child):
                dirs.append(child)
    return dirs


def is_matrix_root(path: Path) -> bool:
    return path.joinpath("matrix_manifest.json").exists() or bool(find_experiment_dirs(path))


def effective_rows(step_rows: List[dict]) -> List[dict]:
    rows = [row for row in step_rows if not row.get("warmup", False)]
    return rows if rows else step_rows


def compute_summary_from_steps(step_rows: List[dict]) -> dict:
    rows = effective_rows(step_rows)
    return {
        "step_ms_avg": mean(float(row["step_ms_max"]) for row in rows),
        "step_ms_p95": percentile([float(row["step_ms_max"]) for row in rows], 0.95),
        "collective_gbps_avg": mean(float(row["collective_gbps_est"]) for row in rows),
        "collective_gbps_p95": percentile([float(row["collective_gbps_est"]) for row in rows], 0.95),
    }


def canonical_algo(value) -> str:
    text = str(value or "").strip()
    return text if text else "auto"


def workload_label(collective: str, algo: str) -> str:
    if collective == "allreduce":
        return f"{collective}_{algo.lower()}"
    return collective


def annotate_summary_rows(summary_rows: List[dict], env_setup: Optional[dict]) -> List[dict]:
    placement = str((env_setup or {}).get("placement", ""))
    env_algo = canonical_algo((env_setup or {}).get("nccl_algo", ""))
    for row in summary_rows:
        algo = canonical_algo(row.get("algo", ""))
        if algo == "auto" and env_algo != "auto":
            algo = env_algo
        row["placement"] = placement
        row["algo"] = algo
        row["workload"] = workload_label(str(row.get("collective", "")), algo)
    return summary_rows


def _worker_window_defaults() -> dict:
    return {
        "first_ts": None,
        "last_ts": None,
        "initial_w": set(),
        "initial_penalty": set(),
        "initial_w_sem": set(),
        "final_w": set(),
        "final_penalty": set(),
        "final_w_sem": set(),
        "trace_x": [],
        "trace_step": [],
        "trace_w": [],
        "trace_pressure": [],
        "sample_index": 0,
    }


def summarize_nccl_logs(mode_dir: Path, worker_step_timings: Optional[Dict[str, List[dict]]] = None) -> dict:
    log_paths = sorted(mode_dir.glob("*/nccl.*.log"))
    log_progress(f"scan mode logs start mode_dir={mode_dir.as_posix()} files={len(log_paths)}")

    event_counts: Counter = Counter()
    worker_event_counts: Dict[str, Counter] = defaultdict(Counter)
    worker_wstall_counts: Counter = Counter()
    decision_counts: Counter = Counter()
    w_eff_values: List[float] = []
    pressure_scores: List[float] = []
    occ_pd_counter: Counter = Counter()
    occ_tr_counter: Counter = Counter()
    worker_windows: Dict[str, dict] = defaultdict(_worker_window_defaults)

    recv_wstall_count = 0
    send_wstall_count = 0

    for log_path in log_paths:
        worker = log_path.parent.name
        log_progress(f"scan events file={log_path.name} worker={worker} size={log_path.stat().st_size}")
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                match = PHASE_RE.search(line)
                if not match:
                    continue
                payload = match.group(2)
                entry = {key: parse_value(value) for key, value in KV_RE.findall(payload)}
                event_name = str(entry.get("event", ""))
                if not event_name:
                    continue

                event_counts[event_name] += 1
                worker_event_counts[worker][event_name] += 1

                if "WSTALL" in event_name:
                    worker_wstall_counts[worker] += 1
                    if "RECV" in event_name:
                        recv_wstall_count += 1
                    if "SEND" in event_name:
                        send_wstall_count += 1

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

                if event_name.endswith("DECISION") and "reason" in entry:
                    decision_counts[str(entry["reason"])] += 1

                ts = int(entry.get("tNs", 0) or 0)
                window_info = worker_windows[worker]

                is_receiver_window_sample = (
                    event_name in {"PROXY_B2_WINDOW_CFG", "PROXY_B3_WINDOW_CFG"}
                    or event_name.startswith("PROXY_RECV_")
                )

                sample_w = entry.get("wEff")
                sample_penalty = entry.get("penTotal", 0)
                sample_w_sem = entry.get("wSem", sample_w)
                if sample_w is not None and is_receiver_window_sample:
                    try:
                        w_eff = float(sample_w)
                        w_eff_values.append(w_eff)
                        if window_info["first_ts"] is None or ts < window_info["first_ts"]:
                            window_info["first_ts"] = ts
                            window_info["initial_w"] = {w_eff}
                            window_info["initial_penalty"] = {sample_penalty}
                            window_info["initial_w_sem"] = {sample_w_sem}
                        elif ts == window_info["first_ts"]:
                            window_info["initial_w"].add(w_eff)
                            window_info["initial_penalty"].add(sample_penalty)
                            window_info["initial_w_sem"].add(sample_w_sem)

                        if window_info["last_ts"] is None or ts > window_info["last_ts"]:
                            window_info["last_ts"] = ts
                            window_info["final_w"] = {w_eff}
                            window_info["final_penalty"] = {sample_penalty}
                            window_info["final_w_sem"] = {sample_w_sem}
                        elif ts == window_info["last_ts"]:
                            window_info["final_w"].add(w_eff)
                            window_info["final_penalty"].add(sample_penalty)
                            window_info["final_w_sem"].add(sample_w_sem)
                    except (TypeError, ValueError):
                        pass

                trace_w = None
                trace_x = None
                trace_step = None
                trace_pressure = None
                if event_name == "PROXY_B3_DECISION":
                    trace_x = int(entry.get("ctrlStep", 0) or 0)
                    trace_w = entry.get("newW", entry.get("wEff"))
                    trace_pressure = entry.get("pressureScore")
                elif event_name == "PROXY_B3_WINDOW_CFG":
                    trace_x = int(entry.get("ctrlStep", 0) or 0)
                    trace_w = entry.get("wEff")
                    trace_pressure = None
                elif event_name == "PROXY_B2_WINDOW_CFG":
                    window_info["sample_index"] += 1
                    trace_x = window_info["sample_index"]
                    trace_w = entry.get("wEff")
                elif event_name.startswith("PROXY_RECV_") and sample_w is not None:
                    window_info["sample_index"] += 1
                    trace_x = window_info["sample_index"]
                    trace_w = sample_w

                if trace_x and worker_step_timings is not None:
                    trace_step = lookup_worker_step(worker_step_timings.get(worker, []), ts)

                if trace_x and trace_w is not None:
                    try:
                        window_info["trace_x"].append(int(trace_x))
                        window_info["trace_step"].append(trace_step if trace_step is not None else -1)
                        window_info["trace_w"].append(float(trace_w))
                        window_info["trace_pressure"].append(float(trace_pressure) if trace_pressure is not None else float("nan"))
                    except (TypeError, ValueError):
                        pass

    normalized_workers = {}
    for worker, info in sorted(worker_windows.items(), key=lambda item: worker_sort_key(item[0])):
        normalized_workers[worker] = {
            "initial_w_values": sorted(info["initial_w"], key=float),
            "initial_penalty_values": sorted(info["initial_penalty"], key=float),
            "initial_w_sem_values": sorted(info["initial_w_sem"], key=float),
            "final_w_values": sorted(info["final_w"], key=float),
            "final_penalty_values": sorted(info["final_penalty"], key=float),
            "final_w_sem_values": sorted(info["final_w_sem"], key=float),
            "trace_x": info["trace_x"],
            "trace_step": info["trace_step"],
            "trace_w": info["trace_w"],
            "trace_pressure": info["trace_pressure"],
        }

    return {
        "event_counts": event_counts,
        "worker_event_counts": worker_event_counts,
        "worker_wstall_counts": worker_wstall_counts,
        "decision_counts": decision_counts,
        "w_eff_values": sorted({int(v) if float(v).is_integer() else v for v in w_eff_values}, key=float),
        "w_eff_counter": Counter(int(v) if float(v).is_integer() else v for v in w_eff_values),
        "pressure_scores": pressure_scores,
        "worker_windows": normalized_workers,
        "total_events": int(sum(event_counts.values())),
        "recv_wstall_count": int(recv_wstall_count),
        "send_wstall_count": int(send_wstall_count),
        "total_wstall_count": int(recv_wstall_count + send_wstall_count),
        "p99_occ_pd": percentile_from_counter(occ_pd_counter, 0.99),
        "p99_occ_tr": percentile_from_counter(occ_tr_counter, 0.99),
        "pressure_score_p95": percentile(pressure_scores, 0.95),
    }


def extract_ts_ns(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        abs_value = abs(value)
        if abs_value >= 10**17:
            return value
        if abs_value >= 10**14:
            return value * 1000
        if abs_value >= 10**11:
            return value * 1_000_000
        if abs_value >= 10**9:
            return value * 1_000_000_000
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        abs_value = abs(value)
        if abs_value >= 10**17:
            return int(value)
        if abs_value >= 10**9:
            return int(value * 1_000_000_000)
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return extract_ts_ns(int(raw))
        try:
            return extract_ts_ns(float(raw))
        except ValueError:
            pass
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return int(dt.datetime.fromisoformat(normalized).timestamp() * 1_000_000_000)
        except ValueError:
            return None
    return None


def extract_record_ts_ns(record: dict) -> Optional[int]:
    for key in (
        "ts_mid_unix_ns",
        "ts_start_unix_ns",
        "ts_end_unix_ns",
        "ts_unix_ns",
        "timestamp_ns",
        "unix_ns",
        "ts_ns",
        "time_ns",
        "timestamp",
        "ts",
        "time",
        "collected_at",
        "sample_time",
    ):
        if key in record:
            ts_ns = extract_ts_ns(record.get(key))
            if ts_ns is not None:
                return ts_ns
    for value in record.values():
        if isinstance(value, dict):
            ts_ns = extract_record_ts_ns(value)
            if ts_ns is not None:
                return ts_ns
    return None


def extract_switch_counter_value(record: dict) -> Optional[float]:
    kind = str(record.get("kind") or "")
    if kind == "fsos_pfc_statistics":
        total = 0.0
        found = False
        for key in ("rx_pause", "tx_pause"):
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += max(0.0, float(value))
                found = True
        return total if found else None
    if kind == "onyx_roce_counters":
        total = 0.0
        found = False
        for key in ("rx_pause_packets", "tx_pause_packets"):
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += max(0.0, float(value))
                found = True
        return total if found else None
    return None


def switch_snapshot_group_key(record: dict, ts_ns: int) -> str:
    sample_id = record.get("sample_id")
    if sample_id not in (None, ""):
        return f"sample:{sample_id}"
    bucket_ts = (ts_ns // SWITCH_AGGREGATE_BUCKET_NS) * SWITCH_AGGREGATE_BUCKET_NS
    return f"bucket:{bucket_ts}"


def load_switch_counter_snapshots(path: Path) -> List[Tuple[int, float]]:
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in load_jsonl(path):
        ts_ns = extract_record_ts_ns(row)
        value = extract_switch_counter_value(row)
        if ts_ns is None or value is None:
            continue
        grouped[switch_snapshot_group_key(row, ts_ns)].append((ts_ns, value))
    snapshots: List[Tuple[int, float]] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
        ts_ns = int(sum(ts for ts, _ in rows) / len(rows))
        total = float(sum(value for _, value in rows))
        snapshots.append((ts_ns, total))
    snapshots.sort(key=lambda item: item[0])
    return snapshots


def compute_switch_delta_series(samples: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    deltas: List[Tuple[int, float]] = []
    for (prev_ts, prev_total), (cur_ts, cur_total) in zip(samples, samples[1:]):
        dt_sec = (cur_ts - prev_ts) / 1_000_000_000.0
        if dt_sec <= 0.0:
            continue
        delta = max(0.0, cur_total - prev_total)
        deltas.append((cur_ts, delta / dt_sec))
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
    snapshots: List[Tuple[int, float]], start_ns: int, end_ns: int
) -> List[Tuple[float, float]]:
    if end_ns <= start_ns:
        return []
    start_value = interpolate_switch_snapshot_value(snapshots, start_ns)
    end_value = interpolate_switch_snapshot_value(snapshots, end_ns)
    if start_value is None or end_value is None:
        return []

    candidate_ts = [start_ns]
    candidate_ts.extend(ts_ns for ts_ns, _ in snapshots if start_ns < ts_ns < end_ns)
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


def resolve_switch_log_dir(experiment_root: Path, env_setup: Optional[dict], switch_log_dir_override: Optional[str] = None) -> Optional[Path]:
    if switch_log_dir_override:
        return Path(switch_log_dir_override)
    if not env_setup or int(env_setup.get("switch_log_enable", 0) or 0) != 1:
        return None
    candidates: List[Path] = []
    local_dir = env_setup.get("switch_log_local_dir")
    if local_dir:
        candidates.append(Path(str(local_dir)))
    run_id = env_setup.get("switch_log_run_id")
    if run_id:
        candidates.append(SWITCH_SHARED_ROOT_DEFAULT / str(run_id))
    remote_dir = env_setup.get("switch_log_dir")
    if remote_dir:
        candidates.append(Path(str(remote_dir)))
    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    for candidate in unique:
        if candidate.exists():
            return candidate
    return unique[0] if unique else None


def step_time_bounds(step_rows: List[dict]) -> Optional[Tuple[int, int]]:
    starts = [extract_ts_ns(row.get("ts_start_unix_ns")) for row in step_rows]
    starts = [value for value in starts if value is not None]
    ends = [extract_ts_ns(row.get("ts_end_unix_ns")) for row in step_rows]
    ends = [value for value in ends if value is not None]
    mids = [extract_ts_ns(row.get("ts_mid_unix_ns")) for row in step_rows]
    mids = [value for value in mids if value is not None]
    if starts and ends:
        return min(starts), max(ends)
    if mids:
        return min(mids), max(mids)
    return None


def step_midpoint_series(step_rows: List[dict]) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for row in [row for row in step_rows if not row.get("warmup", False)] or step_rows:
        ts_ns = extract_ts_ns(row.get("ts_mid_unix_ns"))
        if ts_ns is None:
            ts_ns = extract_ts_ns(row.get("ts_end_unix_ns"))
        if ts_ns is None:
            ts_ns = extract_ts_ns(row.get("ts_start_unix_ns"))
        if ts_ns is None:
            continue
        points.append((ts_ns, float(row["step_ms_max"])))
    points.sort(key=lambda item: item[0])
    return points


def find_mode_marker_bounds(markers: List[dict], run_id: str, mode: str) -> Optional[Tuple[int, int]]:
    starts: List[int] = []
    ends: List[int] = []
    mode_token = f"mode={mode.upper()}"
    run_token = f"run_id={run_id}" if run_id else ""
    for record in markers:
        marker_name = str(record.get("marker") or record.get("event") or "")
        message = str(record.get("message") or "")
        if mode_token not in message:
            continue
        if run_token and run_token not in message:
            continue
        ts_ns = extract_record_ts_ns(record)
        if ts_ns is None:
            continue
        if marker_name == "mode_start":
            starts.append(ts_ns)
        elif marker_name == "mode_end":
            ends.append(ts_ns)
    if starts and ends:
        return min(starts), max(ends)
    return None


def build_mode_time_windows(mode_data: List[dict], env_setup: Optional[dict], markers: List[dict]) -> Dict[str, Tuple[int, int]]:
    run_id = str(env_setup.get("run_id", "")) if env_setup else ""
    windows: Dict[str, Tuple[int, int]] = {}
    for item in mode_data:
        bounds = step_time_bounds(item["step_rows"])
        if bounds is None:
            bounds = find_mode_marker_bounds(markers, run_id, item["mode"])
        if bounds is not None:
            windows[item["mode"]] = bounds
    return windows


def load_switch_bundle(experiment_root: Path, env_setup: Optional[dict], switch_log_dir_override: Optional[str] = None) -> Optional[dict]:
    log_dir = resolve_switch_log_dir(experiment_root, env_setup, switch_log_dir_override)
    if log_dir is None or not log_dir.exists():
        return None
    series = {}
    snapshots = {}
    file_map = {
        "spine": log_dir / "spine_roce_counters.jsonl",
        "rackA": log_dir / "rackA_pfc_statistics.jsonl",
        "rackB": log_dir / "rackB_pfc_statistics.jsonl",
    }
    for label, path in file_map.items():
        if not path.exists():
            continue
        snapshot_series = load_switch_counter_snapshots(path)
        if snapshot_series:
            snapshots[label] = snapshot_series
        rate_series = compute_switch_delta_series(snapshot_series)
        if rate_series:
            series[label] = rate_series
    markers_path = log_dir / "markers.jsonl"
    markers = load_jsonl(markers_path) if markers_path.exists() else []
    if not series and not snapshots and not markers:
        return None
    return {
        "log_dir": log_dir,
        "series": series,
        "snapshots": snapshots,
        "markers": markers,
        "bucket_ns": SWITCH_AGGREGATE_BUCKET_NS,
    }


def save_switch_overlay_plot(
    path: Path,
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[Path]:
    switch_bundle = load_switch_bundle(experiment_root, env_setup, switch_log_dir_override)
    if not switch_bundle or not switch_bundle.get("series"):
        return None

    mode_windows = build_mode_time_windows(mode_data, env_setup, switch_bundle.get("markers", []))
    if not mode_windows:
        return None

    global_start = min(start for start, _ in mode_windows.values())
    global_end = max(end for _, end in mode_windows.values())
    if global_end <= global_start:
        return None

    switch_series = {}
    for label, rows in switch_bundle["series"].items():
        filtered = [((ts_ns - global_start) / 1_000_000_000.0, rate) for ts_ns, rate in rows if global_start <= ts_ns <= global_end]
        if filtered:
            switch_series[label] = filtered
    if not switch_series:
        return None

    step_series = []
    for item in mode_data:
        points = [((ts_ns - global_start) / 1_000_000_000.0, value) for ts_ns, value in step_midpoint_series(item["step_rows"]) if global_start <= ts_ns <= global_end]
        if points:
            step_series.append((item["mode"], points))

    colors = {"spine": "#d04e00", "rackA": "#1f77b4", "rackB": "#2ca02c"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax_top, ax_bottom = axes

    for label, points in switch_series.items():
        ax_top.plot([x for x, _ in points], [y for _, y in points], linewidth=1.8, label=label, color=colors.get(label))

    for _, (start_ns, end_ns) in mode_windows.items():
        rel_start = (start_ns - global_start) / 1_000_000_000.0
        rel_end = (end_ns - global_start) / 1_000_000_000.0
        ax_top.axvspan(rel_start, rel_end, color="#8aa1c1", alpha=0.06)
        ax_bottom.axvspan(rel_start, rel_end, color="#8aa1c1", alpha=0.06)
        ax_top.axvline(rel_start, color="#6c7f99", linestyle="--", alpha=0.35, linewidth=0.9)
        ax_bottom.axvline(rel_start, color="#6c7f99", linestyle="--", alpha=0.35, linewidth=0.9)

    for mode, points in step_series:
        ax_bottom.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=1.5, markersize=2.8, label=mode)

    ax_top.set_title(f"{experiment_root.name} Switch Pressure Overlay")
    ax_top.set_ylabel("aggregated counter delta / sec")
    ax_top.grid(True, alpha=0.25)
    if switch_series:
        ax_top.legend()

    ax_bottom.set_xlabel("elapsed sec")
    ax_bottom.set_ylabel("step_ms_max")
    ax_bottom.grid(True, alpha=0.25)
    if step_series:
        ax_bottom.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_switch_phase_pfc_plot(
    path: Path,
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str] = None,
) -> Optional[Path]:
    switch_bundle = load_switch_bundle(experiment_root, env_setup, switch_log_dir_override)
    if not switch_bundle or not switch_bundle.get("snapshots"):
        return None

    mode_windows = build_mode_time_windows(mode_data, env_setup, switch_bundle.get("markers", []))
    if not mode_windows:
        return None

    target_labels = [label for label in ("rackA", "rackB") if label in switch_bundle["snapshots"]]
    if not target_labels:
        return None

    mode_colors = {"STOCK": "#1f77b4", "B2": "#ff7f0e", "B3": "#2ca02c"}
    fig, axes = plt.subplots(len(target_labels), 1, figsize=(12, 4.6 + 2.2 * max(0, len(target_labels) - 1)), sharex=False)
    if len(target_labels) == 1:
        axes = [axes]

    any_points = False
    for ax, label in zip(axes, target_labels):
        for item in mode_data:
            bounds = mode_windows.get(item["mode"])
            if bounds is None:
                continue
            start_ns, end_ns = bounds
            points = build_phase_aligned_cumulative_series(switch_bundle["snapshots"][label], start_ns, end_ns)
            if not points:
                continue
            any_points = True
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                linewidth=1.6,
                markersize=2.8,
                label=item["mode"],
                color=mode_colors.get(item["mode"]),
            )
        ax.set_title(f"{label} PFC pause increase by phase")
        ax.set_ylabel("pause count increase")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.axvline(0.0, color="#6c7f99", linestyle="--", alpha=0.35, linewidth=0.9)

    if not any_points:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("elapsed sec from phase start")
    fig.suptitle(f"{experiment_root.name} Phase-Aligned PFC Pause Count Overlay", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


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

    worker_step_timings = load_worker_step_timings(mode_dir)
    nccl_summary = summarize_nccl_logs(mode_dir, worker_step_timings)
    mode_name = str(summary.get("run_tag") or summary.get("phase3_mode") or mode_dir.name).upper()
    log_progress(f"load mode complete mode={mode_name} step_rows={len(step_rows)} total_events={nccl_summary['total_events']}")
    return {
        "mode": mode_name,
        "summary": summary,
        "summary_path": summary_path,
        "step_path": step_path,
        "step_rows": step_rows,
        "nccl": nccl_summary,
        "mode_dir": mode_dir,
        "worker_step_timings": worker_step_timings,
    }


def summarize_modes(mode_data: List[dict]) -> List[dict]:
    stock = next((item for item in mode_data if item["mode"] == "STOCK"), None)
    b2 = next((item for item in mode_data if item["mode"] == "B2"), None)
    rows = []
    for item in mode_data:
        summary = item["summary"]
        nccl = item["nccl"]
        row = {
            "mode": item["mode"],
            "algo": canonical_algo(summary.get("algo", "")),
            "collective": summary.get("collective", ""),
            "payload_mb": float(summary.get("payload_mb", 0.0)),
            "step_ms_avg": float(summary.get("step_ms_avg", 0.0)),
            "step_ms_p95": float(summary.get("step_ms_p95", 0.0)),
            "collective_gbps_avg": float(summary.get("collective_gbps_avg", 0.0)),
            "collective_gbps_p95": float(summary.get("collective_gbps_p95", 0.0)),
            "total_events": int(nccl["total_events"]),
            "recv_wstall_count": int(nccl["recv_wstall_count"]),
            "send_wstall_count": int(nccl["send_wstall_count"]),
            "total_wstall_count": int(nccl["total_wstall_count"]),
            "p99_occ_pd": float(nccl["p99_occ_pd"]),
            "p99_occ_tr": float(nccl["p99_occ_tr"]),
            "w_eff_values": ",".join(str(v) for v in nccl["w_eff_values"]) or "-",
            "decision_counts": ", ".join(f"{k}:{v}" for k, v in sorted(nccl["decision_counts"].items())) or "-",
            "pressure_score_p95": float(nccl["pressure_score_p95"]),
            "delta_step_vs_stock_pct": 0.0,
            "delta_bw_vs_stock_pct": 0.0,
            "delta_occ_tr_vs_stock_pct": 0.0,
            "delta_wstall_vs_stock_pct": 0.0,
            "delta_step_vs_b2_pct": 0.0,
            "delta_bw_vs_b2_pct": 0.0,
        }
        if stock is not None and item["mode"] != "STOCK":
            row["delta_step_vs_stock_pct"] = rel_change(float(stock["summary"].get("step_ms_avg", 0.0)), row["step_ms_avg"])
            row["delta_bw_vs_stock_pct"] = rel_change(float(stock["summary"].get("collective_gbps_avg", 0.0)), row["collective_gbps_avg"])
            row["delta_occ_tr_vs_stock_pct"] = rel_change(float(stock["nccl"]["p99_occ_tr"]), row["p99_occ_tr"])
            row["delta_wstall_vs_stock_pct"] = rel_change(float(stock["nccl"]["total_wstall_count"]), row["total_wstall_count"])
        if b2 is not None and item["mode"] not in {"STOCK", "B2"}:
            row["delta_step_vs_b2_pct"] = rel_change(float(b2["summary"].get("step_ms_avg", 0.0)), row["step_ms_avg"])
            row["delta_bw_vs_b2_pct"] = rel_change(float(b2["summary"].get("collective_gbps_avg", 0.0)), row["collective_gbps_avg"])
        rows.append(row)
    return rows


def build_collection_plan_card_single(
    experiment_root: Path,
    mode_data: List[dict],
    env_setup: Optional[dict] = None,
    switch_log_dir_override: Optional[str] = None,
) -> str:
    total_logs = sum(len(list(item["mode_dir"].glob("*/nccl.*.log"))) for item in mode_data)
    total_step_timing = sum(len(list(item["mode_dir"].glob("*/**/*_worker_step_timing.jsonl"))) for item in mode_data)
    rows = [
        ["env_setup.json", "실험 설정. placement, collective, algorithm, payload, run modes, 정책식."],
        ["MODE/*_summary.json", "mode별 최종 요약. step latency와 collective throughput의 평균/p95."],
        ["MODE/*_step_metrics.jsonl", "collective iteration step 시계열."],
        [f"MODE/workerXX/*_worker_step_timing.jsonl ({total_step_timing} files)", "worker-local monotonic step timing. receiver W trace를 collective step 축에 정렬할 때 사용."],
        [f"MODE/workerXX/nccl.*.log ({total_logs} files)", "PHASE0/2/3 이벤트 로그. WINDOW_CFG, PRESSURE, DECISION, WSTALL을 포함."],
    ]
    if switch_log_dir_override:
        rows.append(["switch_log_override", switch_log_dir_override])
        rows.append(["switch_log/*.jsonl", "override 경로에서 읽는 switch pressure 로그. spine ROCE, rackA/rackB PFC, markers.jsonl 을 포함."])
    elif env_setup and int(env_setup.get("switch_log_enable", 0) or 0) == 1:
        switch_dir = env_setup.get("switch_log_local_dir") or (str(SWITCH_SHARED_ROOT_DEFAULT / str(env_setup.get("switch_log_run_id", ""))) if env_setup.get("switch_log_run_id") else "")
        if switch_dir:
            rows.append(["switch_log_local_dir", switch_dir])
        rows.append(["switch_log/*.jsonl", "switch pressure 로그. spine ROCE, rackA/rackB PFC, markers.jsonl 을 포함."])
    return '<div class="card"><h2>Collected Logs</h2>' + render_table(["source", "meaning"], rows) + '</div>'


def build_visualization_plan_card_single(env_setup: Optional[dict] = None, switch_log_dir_override: Optional[str] = None) -> str:
    rows = [
        ("step_timeline.png", "*_step_metrics.jsonl", "mode별 collective step latency 시계열"),
        ("throughput_timeline.png", "*_step_metrics.jsonl", "mode별 collective throughput estimate 시계열"),
        ("summary_latency.png", "*_summary.json", "mode별 absolute latency 비교"),
        ("summary_throughput.png", "*_summary.json", "mode별 absolute throughput 비교"),
        ("delta_vs_stock.png", "summary + stock baseline", "mode별 stock 대비 개선율"),
        ("selected_window_distribution.png", "receiver-side WINDOW logs", "mode별 receiver W_eff 분포"),
        ("decision_counts.png", "PROXY_B3_DECISION", "B3 shrink / hold / recover 비중"),
        ("pressure_summary.png", "PROXY_B3_PRESSURE", "pressure score p50 / p95"),
        ("worker_window_trace_<MODE>.png", "WINDOW_CFG / DECISION / RECV events + worker step timing", "worker별 receiver W 변화를 collective step 축에 정렬"),
    ]
    if switch_log_dir_override or (env_setup and int(env_setup.get("switch_log_enable", 0) or 0) == 1):
        rows.append(("switch_overlay.png", "switch_log/*.jsonl + *_step_metrics.jsonl", "switch PFC/ROCE delta/sec 와 step latency overlay"))
        rows.append(("switch_pfc_phase_overlay.png", "switch_log/*.jsonl + *_step_metrics.jsonl", "phase start 기준으로 정렬한 rackA/rackB PFC pause count overlay"))
    return render_plot_help_card("Visualization Plan", rows)


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
    return save_grouped_bar(path, "Window Stall Count by Worker", workers, series, "wstall count")


def save_window_distribution_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    windows = sorted({window for item in mode_data for window in item["nccl"]["w_eff_counter"].keys()}, key=float)
    if not windows:
        return None
    labels = [str(window) for window in windows]
    series = []
    for item in mode_data:
        counter = item["nccl"]["w_eff_counter"]
        series.append((item["mode"], [float(counter.get(window, 0)) for window in windows]))
    return save_grouped_bar(path, "Receiver Window Distribution", labels, series, "count")


def save_decision_counts_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    reasons = sorted({reason for item in mode_data for reason in item["nccl"]["decision_counts"].keys()})
    if not reasons:
        return None
    series = []
    for item in mode_data:
        counts = item["nccl"]["decision_counts"]
        series.append((item["mode"], [float(counts.get(reason, 0)) for reason in reasons]))
    return save_grouped_bar(path, "Decision Reason Counts", reasons, series, "count")


def save_pressure_summary_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    modes = [item["mode"] for item in mode_data if item["nccl"]["pressure_scores"]]
    if not modes:
        return None
    p50_values = [percentile(item["nccl"]["pressure_scores"], 0.50) for item in mode_data if item["nccl"]["pressure_scores"]]
    p95_values = [percentile(item["nccl"]["pressure_scores"], 0.95) for item in mode_data if item["nccl"]["pressure_scores"]]
    return save_grouped_bar(path, "Pressure Score Summary", modes, [("p50", p50_values), ("p95", p95_values)], "pressure score")


def save_occupancy_summary_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    modes = [item["mode"] for item in mode_data]
    series = [
        ("p99_occ_pd", [float(item["nccl"]["p99_occ_pd"]) for item in mode_data]),
        ("p99_occ_tr", [float(item["nccl"]["p99_occ_tr"]) for item in mode_data]),
    ]
    return save_grouped_bar(path, "Outstanding Depth Summary by Mode", modes, series, "depth")


def save_delta_vs_stock_plot(path: Path, summary_rows: List[dict]) -> Optional[Path]:
    modes = [row["mode"] for row in summary_rows if row["mode"] != "STOCK"]
    if not modes:
        return None
    series = [
        ("delta_step_vs_stock_pct", [float(row["delta_step_vs_stock_pct"]) for row in summary_rows if row["mode"] != "STOCK"]),
        ("delta_bw_vs_stock_pct", [float(row["delta_bw_vs_stock_pct"]) for row in summary_rows if row["mode"] != "STOCK"]),
        ("delta_occ_tr_vs_stock_pct", [float(row["delta_occ_tr_vs_stock_pct"]) for row in summary_rows if row["mode"] != "STOCK"]),
        ("delta_wstall_vs_stock_pct", [float(row["delta_wstall_vs_stock_pct"]) for row in summary_rows if row["mode"] != "STOCK"]),
    ]
    return save_grouped_bar(path, "Change vs STOCK", modes, series, "percent")


def save_worker_window_trace_plot(path: Path, mode_item: dict) -> Optional[Path]:
    worker_windows = mode_item["nccl"]["worker_windows"]
    workers = [worker for worker, info in sorted(worker_windows.items(), key=lambda item: worker_sort_key(item[0])) if info["trace_x"] and info["trace_w"]]
    if not workers:
        return None
    plt.figure(figsize=(11, 5.5))
    max_value = 0.0
    x_label = "receiver control sample"
    for worker in workers:
        raw_trace_x = worker_windows[worker]["trace_x"]
        trace_step = worker_windows[worker].get("trace_step", [])
        trace_w = worker_windows[worker]["trace_w"]
        if not raw_trace_x or not trace_w:
            continue
        if trace_step and any(step >= 0 for step in trace_step):
            per_step: Dict[int, float] = {}
            for step, w_value in zip(trace_step, trace_w):
                if step < 0:
                    continue
                per_step[int(step)] = float(w_value)
            if not per_step:
                continue
            plot_x = sorted(per_step.keys())
            plot_y = [per_step[idx] for idx in plot_x]
            x_label = "collective step"
        else:
            plot_x = raw_trace_x
            plot_y = trace_w
        plt.plot(plot_x, plot_y, marker="o", linewidth=1.5, markersize=3, label=worker)
        max_value = max(max_value, max(plot_y))
    plt.title(f"{mode_item['mode']} Receiver Window Trace by Worker")
    plt.xlabel(x_label)
    plt.ylabel("W_eff")
    ylim = positive_ylim(max_value)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def build_worker_window_table(mode_item: dict) -> str:
    rows = []
    worker_windows = mode_item["nccl"]["worker_windows"]
    for worker, info in sorted(worker_windows.items(), key=lambda item: worker_sort_key(item[0])):
        rows.append([
            worker,
            ",".join(str(v) for v in info["initial_w_values"]) or "-",
            ",".join(str(v) for v in info["initial_penalty_values"]) or "-",
            ",".join(str(v) for v in info["initial_w_sem_values"]) or "-",
            ",".join(str(v) for v in info["final_w_values"]) or "-",
            ",".join(str(v) for v in info["final_penalty_values"]) or "-",
            ",".join(str(v) for v in info["final_w_sem_values"]) or "-",
            len(info["trace_x"]),
        ])
    return render_table(
        ["worker", "initial_w", "initial_penalty", "initial_w_sem", "final_w", "final_penalty", "final_w_sem", "samples"],
        rows,
    )


def build_single_experiment_report(
    experiment_root: Path,
    output_dir: Path,
    top_events: int,
    switch_log_dir_override: Optional[str] = None,
) -> Path:
    log_progress(f"build single report start experiment={experiment_root.name}")
    env_setup = load_json(experiment_root / "env_setup.json") if (experiment_root / "env_setup.json").exists() else None
    mode_dirs = find_mode_dirs(experiment_root)
    if not mode_dirs:
        raise FileNotFoundError(f"no mode directories found in {experiment_root}")

    mode_data = [load_mode_data(mode_dir) for mode_dir in mode_dirs]
    mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
    summary_rows = summarize_modes(mode_data)
    summary_rows = annotate_summary_rows(summary_rows, env_setup)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    step_series_max = []
    step_series_mean = []
    bw_series = []
    for item in mode_data:
        rows = item["step_rows"]
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        step_series_max.append((f'{item["mode"]} step_ms_max', steps, [float(row["step_ms_max"]) for row in rows]))
        step_series_mean.append((f'{item["mode"]} step_ms_mean', steps, [float(row["step_ms_mean"]) for row in rows]))
        bw_series.append((item["mode"], steps, [float(row["collective_gbps_est"]) for row in rows]))

    p_step = save_line_plot(plots_dir / "step_timeline.png", f"{experiment_root.name} Step Timeline", "collective step", "latency (ms)", step_series_max + step_series_mean)
    p_bw = save_line_plot(plots_dir / "throughput_timeline.png", f"{experiment_root.name} Throughput Estimate", "collective step", "Gbps", bw_series)

    categories = [row["mode"] for row in summary_rows]
    p_step_summary = save_grouped_bar(
        plots_dir / "summary_latency.png",
        "Latency Summary by Mode",
        categories,
        [("avg", [float(row["step_ms_avg"]) for row in summary_rows]), ("p95", [float(row["step_ms_p95"]) for row in summary_rows])],
        "ms",
    )
    p_bw_summary = save_grouped_bar(
        plots_dir / "summary_throughput.png",
        "Throughput Summary by Mode",
        categories,
        [("avg", [float(row["collective_gbps_avg"]) for row in summary_rows]), ("p95", [float(row["collective_gbps_p95"]) for row in summary_rows])],
        "Gbps",
    )
    p_delta = save_delta_vs_stock_plot(plots_dir / "delta_vs_stock.png", summary_rows)
    p_events = save_event_counts_plot(plots_dir / "event_counts.png", mode_data, top_events)
    p_wstall = save_worker_wstall_plot(plots_dir / "wstall_by_worker.png", mode_data)
    p_window = save_window_distribution_plot(plots_dir / "selected_window_distribution.png", mode_data)
    p_occ = save_occupancy_summary_plot(plots_dir / "occupancy_summary.png", mode_data)
    p_decision = save_decision_counts_plot(plots_dir / "decision_counts.png", mode_data)
    p_pressure = save_pressure_summary_plot(plots_dir / "pressure_summary.png", mode_data)
    p_switch = save_switch_overlay_plot(plots_dir / "switch_overlay.png", experiment_root, env_setup, mode_data, switch_log_dir_override)
    p_switch_phase = save_switch_phase_pfc_plot(plots_dir / "switch_pfc_phase_overlay.png", experiment_root, env_setup, mode_data, switch_log_dir_override)

    worker_trace_paths = []
    for item in mode_data:
        trace_path = save_worker_window_trace_plot(plots_dir / f"worker_window_trace_{item['mode'].lower()}.png", item)
        if trace_path is not None:
            worker_trace_paths.append((trace_path, f"{item['mode']} worker window trace"))

    summary_headers = [
        "mode", "placement", "workload", "collective", "algo", "payload_mb", "step_ms_avg", "step_ms_p95",
        "collective_gbps_avg", "collective_gbps_p95", "total_wstall_count",
        "p99_occ_tr", "w_eff_values", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "delta_step_vs_b2_pct", "delta_bw_vs_b2_pct",
    ]
    summary_table = [
        [
            row["mode"],
            row["placement"],
            row["workload"],
            row["collective"],
            row["algo"],
            f'{row["payload_mb"]:.3f}',
            f'{row["step_ms_avg"]:.3f}',
            f'{row["step_ms_p95"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            f'{row["collective_gbps_p95"]:.3f}',
            row["total_wstall_count"],
            f'{row["p99_occ_tr"]:.3f}',
            row["w_eff_values"],
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
            f'{row["delta_step_vs_b2_pct"]:.2f}',
            f'{row["delta_bw_vs_b2_pct"]:.2f}',
        ]
        for row in summary_rows
    ]

    info_rows = [["experiment_root", experiment_root.as_posix()], ["modes", ", ".join(item["mode"] for item in mode_data)]]
    if env_setup:
        for key in ["placement", "collective", "nccl_algo", "nccl_proto", "run_modes", "payload_mb", "dtype", "policy_name", "policy_formula", "master_server", "switch_log_run_id", "switch_log_local_dir"]:
            if key in env_setup:
                info_rows.append([key, env_setup[key]])
    if switch_log_dir_override:
        info_rows.append(["switch_log_override", switch_log_dir_override])

    sections = [
        '<section class="section"><h2>Experiment Metadata</h2>' + render_table(["field", "value"], info_rows) + "</section>",
        '<section class="section"><div class="grid-2">' + build_collection_plan_card_single(experiment_root, mode_data, env_setup, switch_log_dir_override) + build_visualization_plan_card_single(env_setup, switch_log_dir_override) + "</div></section>",
        '<section class="section"><div class="grid-2">'
        + render_field_help_card("Run Summary Field Meanings", RUN_SUMMARY_FIELD_HELP, ["mode", "placement", "workload", "collective", "algo", "payload_mb", "step_ms_avg", "step_ms_p95", "collective_gbps_avg", "collective_gbps_p95", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "delta_occ_tr_vs_stock_pct", "delta_wstall_vs_stock_pct", "delta_step_vs_b2_pct", "delta_bw_vs_b2_pct"])
        + render_field_help_card("NCCL Summary Field Meanings", NCCL_SUMMARY_FIELD_HELP, ["total_events", "recv_wstall_count", "send_wstall_count", "p99_occ_pd", "p99_occ_tr", "w_eff_values", "decision_counts", "pressure_score_p95"])
        + "</div></section>",
        '<section class="section"><h2>Mode Summary</h2>' + render_table(summary_headers, summary_table) + "</section>",
    ]

    for item in mode_data:
        sections.append(f'<section class="section"><h2>{html.escape(item["mode"])} Receiver Window by Worker</h2>{build_worker_window_table(item)}</section>')

    figure_fragments = []
    for path, caption in [
        (p_step, "mode별 collective step latency timeline"),
        (p_bw, "mode별 collective throughput estimate"),
        (p_step_summary, "mode별 absolute latency"),
        (p_bw_summary, "mode별 absolute throughput"),
        (p_delta, "stock 대비 개선율"),
        (p_events, "mode별 top NCCL event counts"),
        (p_wstall, "worker별 WSTALL count"),
        (p_window, "receiver W_eff distribution"),
        (p_occ, "occupancy summary"),
        (p_decision, "decision reason count"),
        (p_pressure, "pressure score summary"),
        (p_switch, "switch PFC / ROCE pressure overlay"),
        (p_switch_phase, "phase-aligned PFC pause count overlay"),
    ] + worker_trace_paths:
        if path is None or not path.exists():
            continue
        figure_fragments.append(f'<figure><img src="{html.escape(relpath(path, output_dir))}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figure_fragments) + "</section>")

    html_path = output_dir / "phase3_report.html"
    html_path.write_text(render_html(f"Phase3 Report - {experiment_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase3_report.json"
    report_json.write_text(
        json.dumps(
            {
                "experiment": experiment_root.name,
                "env_setup": env_setup,
                "modes": summary_rows,
                "switch_overlay_plot": relpath(p_switch, output_dir) if p_switch is not None and p_switch.exists() else None,
                "switch_pfc_phase_plot": relpath(p_switch_phase, output_dir) if p_switch_phase is not None and p_switch_phase.exists() else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_csv = output_dir / "phase3_report.csv"
    if summary_rows:
        with report_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    log_progress(f"build single report complete experiment={experiment_root.name} html={html_path.as_posix()}")
    return html_path


def build_collection_plan_card_matrix(matrix_root: Path, experiment_dirs: List[Path]) -> str:
    total_logs = 0
    total_step_timing = 0
    for experiment_dir in experiment_dirs:
        for mode_dir in find_mode_dirs(experiment_dir):
            total_logs += len(list(mode_dir.glob("*/nccl.*.log")))
            total_step_timing += len(list(mode_dir.glob("*/**/*_worker_step_timing.jsonl")))
    rows = [
        ["matrix_manifest.json", "matrix 전체 조합과 worker pool 정의."],
        ["NN_experiment/env_setup.json", "placement, collective, algorithm, payload, run modes, 정책식."],
        ["NN_experiment/MODE/*_summary.json", "mode별 절대 성능 요약."],
        ["NN_experiment/MODE/*_step_metrics.jsonl", "collective step 시계열."],
        [f"NN_experiment/MODE/workerXX/*_worker_step_timing.jsonl ({total_step_timing} files)", "worker-local monotonic step timing. per-worker W trace를 collective step 축에 정렬할 때 사용."],
        [f"NN_experiment/MODE/workerXX/nccl.*.log ({total_logs} files)", "NCCL PHASE0/2/3 이벤트 로그."],
    ]
    return '<div class="card"><h2>Collected Logs</h2>' + render_table(["source", "meaning"], rows) + '</div>'


def build_visualization_plan_card_matrix() -> str:
    rows = [
        ("matrix_step_latency.png", "per-experiment summary", "experiment별 mode absolute step latency 비교"),
        ("matrix_throughput.png", "per-experiment summary", "experiment별 mode absolute throughput 비교"),
        ("matrix_delta_step_vs_stock.png", "summary + stock baseline", "stock 대비 latency 변화율 비교"),
        ("matrix_delta_bw_vs_stock.png", "summary + stock baseline", "stock 대비 throughput 변화율 비교"),
        ("matrix_delta_occ_vs_stock.png", "summary + stock baseline", "stock 대비 p99_occ_tr 변화율 비교"),
        ("matrix_delta_wstall_vs_stock.png", "summary + stock baseline", "stock 대비 total_wstall_count 변화율 비교"),
    ]
    return render_plot_help_card("Visualization Plan", rows)


def collect_matrix_rows(experiment_name: str, mode_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for row in mode_rows:
        flat = dict(row)
        flat["experiment"] = experiment_name
        rows.append(flat)
    return rows


def build_matrix_report(matrix_root: Path, output_dir: Path, top_events: int, switch_log_dir_override: Optional[str] = None) -> Path:
    experiment_dirs = find_experiment_dirs(matrix_root)
    if not experiment_dirs:
        raise FileNotFoundError(f"no experiment directories found in {matrix_root}")
    log_progress(f"build matrix report start matrix_root={matrix_root.as_posix()} experiments={len(experiment_dirs)}")

    per_experiment_links: Dict[str, str] = {}
    matrix_rows: List[dict] = []

    for experiment_dir in experiment_dirs:
        per_output = output_dir / experiment_dir.name
        per_output.mkdir(parents=True, exist_ok=True)
        per_html = build_single_experiment_report(experiment_dir, per_output, top_events, switch_log_dir_override)
        per_experiment_links[experiment_dir.name] = relpath(per_html, output_dir)
        env_setup = load_json(experiment_dir / "env_setup.json") if (experiment_dir / "env_setup.json").exists() else None
        mode_data = [load_mode_data(mode_dir) for mode_dir in find_mode_dirs(experiment_dir)]
        mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
        matrix_rows.extend(collect_matrix_rows(experiment_dir.name, annotate_summary_rows(summarize_modes(mode_data), env_setup)))

    experiment_names = [exp.name for exp in experiment_dirs]
    mode_names = sorted({row["mode"] for row in matrix_rows}, key=mode_sort_key)

    def values_for(metric: str, mode_name: str) -> List[float]:
        values = []
        for exp_name in experiment_names:
            row = next((item for item in matrix_rows if item["experiment"] == exp_name and item["mode"] == mode_name), None)
            values.append(float(row[metric]) if row else 0.0)
        return values

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    p_abs_step = save_grouped_bar(plots_dir / "matrix_step_latency.png", "Matrix - Step Latency by Experiment", experiment_names, [(mode_name, values_for("step_ms_avg", mode_name)) for mode_name in mode_names], "step_ms_avg", rotate_labels=True)
    p_abs_bw = save_grouped_bar(plots_dir / "matrix_throughput.png", "Matrix - Throughput by Experiment", experiment_names, [(mode_name, values_for("collective_gbps_avg", mode_name)) for mode_name in mode_names], "collective_gbps_avg", rotate_labels=True)

    candidate_modes = [mode for mode in mode_names if mode != "STOCK"]
    p_delta_step = p_delta_bw = p_delta_occ = p_delta_wstall = None
    if candidate_modes:
        p_delta_step = save_grouped_bar(plots_dir / "matrix_delta_step_vs_stock.png", "Matrix - Step Latency Change vs Stock", experiment_names, [(mode_name, values_for("delta_step_vs_stock_pct", mode_name)) for mode_name in candidate_modes], "delta_step_vs_stock_pct", rotate_labels=True)
        p_delta_bw = save_grouped_bar(plots_dir / "matrix_delta_bw_vs_stock.png", "Matrix - Throughput Change vs Stock", experiment_names, [(mode_name, values_for("delta_bw_vs_stock_pct", mode_name)) for mode_name in candidate_modes], "delta_bw_vs_stock_pct", rotate_labels=True)
        p_delta_occ = save_grouped_bar(plots_dir / "matrix_delta_occ_vs_stock.png", "Matrix - p99_occ_tr Change vs Stock", experiment_names, [(mode_name, values_for("delta_occ_tr_vs_stock_pct", mode_name)) for mode_name in candidate_modes], "delta_occ_tr_vs_stock_pct", rotate_labels=True)
        p_delta_wstall = save_grouped_bar(plots_dir / "matrix_delta_wstall_vs_stock.png", "Matrix - Total WSTALL Change vs Stock", experiment_names, [(mode_name, values_for("delta_wstall_vs_stock_pct", mode_name)) for mode_name in candidate_modes], "delta_wstall_vs_stock_pct", rotate_labels=True)

    summary_headers = [
        "experiment", "mode", "placement", "workload", "collective", "algo", "payload_mb", "step_ms_avg", "collective_gbps_avg",
        "total_wstall_count", "p99_occ_tr", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "delta_step_vs_b2_pct", "delta_bw_vs_b2_pct", "report",
    ]
    summary_rows_html = []
    for row in matrix_rows:
        link = per_experiment_links[row["experiment"]]
        summary_rows_html.append([
            html.escape(str(row["experiment"])),
            html.escape(str(row["mode"])),
            html.escape(str(row["placement"])),
            html.escape(str(row["workload"])),
            html.escape(str(row["collective"])),
            html.escape(str(row["algo"])),
            f'{row["payload_mb"]:.3f}',
            f'{row["step_ms_avg"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            str(row["total_wstall_count"]),
            f'{row["p99_occ_tr"]:.3f}',
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
            f'{row["delta_step_vs_b2_pct"]:.2f}',
            f'{row["delta_bw_vs_b2_pct"]:.2f}',
            f'<a href="{html.escape(link)}">open</a>',
        ])

    sections = [
        '<section class="section"><div class="grid-2">' + build_collection_plan_card_matrix(matrix_root, experiment_dirs) + build_visualization_plan_card_matrix() + "</div></section>",
        '<section class="section"><div class="grid-2">'
        + render_field_help_card("Run Summary Field Meanings", RUN_SUMMARY_FIELD_HELP, ["mode", "placement", "workload", "collective", "algo", "payload_mb", "step_ms_avg", "step_ms_p95", "collective_gbps_avg", "collective_gbps_p95", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "delta_occ_tr_vs_stock_pct", "delta_wstall_vs_stock_pct", "delta_step_vs_b2_pct", "delta_bw_vs_b2_pct"])
        + render_field_help_card("NCCL Summary Field Meanings", NCCL_SUMMARY_FIELD_HELP, ["total_events", "recv_wstall_count", "send_wstall_count", "p99_occ_pd", "p99_occ_tr", "w_eff_values", "decision_counts", "pressure_score_p95"])
        + "</div></section>",
        '<section class="section"><h2>Matrix Summary</h2>' + render_table_raw(summary_headers, summary_rows_html) + "</section>",
    ]

    figure_fragments = []
    for path, caption in [
        (p_abs_step, "experiment별 step latency"),
        (p_abs_bw, "experiment별 throughput"),
        (p_delta_step, "stock 대비 step latency 변화율"),
        (p_delta_bw, "stock 대비 throughput 변화율"),
        (p_delta_occ, "stock 대비 p99_occ_tr 변화율"),
        (p_delta_wstall, "stock 대비 total WSTALL 변화율"),
    ]:
        if path is None or not path.exists():
            continue
        figure_fragments.append(f'<figure><img src="{html.escape(relpath(path, output_dir))}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figure_fragments) + "</section>")

    report_list = "".join(f'<li><a href="{html.escape(link)}">{html.escape(name)}</a></li>' for name, link in sorted(per_experiment_links.items()))
    sections.append('<section class="section"><h2>Per-experiment Reports</h2><ul>' + report_list + "</ul></section>")

    html_path = output_dir / "phase3_report.html"
    html_path.write_text(render_html(f"Phase3 Matrix Report - {matrix_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase3_report.json"
    report_json.write_text(json.dumps({"matrix_root": matrix_root.name, "rows": matrix_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    report_csv = output_dir / "phase3_report.csv"
    if matrix_rows:
        with report_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matrix_rows)

    log_progress(f"build matrix report complete html={html_path.as_posix()}")
    return html_path


def main() -> None:
    global REPORT_LOG_PATH
    args = parse_args()
    input_path = Path(args.input).resolve()

    if input_path.suffix.lower() == ".zip":
        default_output = input_path.with_name(f"{input_path.stem}_report")
        output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
        output_dir.mkdir(parents=True, exist_ok=True)
        REPORT_LOG_PATH = output_dir / "phase3_reporter.log"
        REPORT_LOG_PATH.write_text("", encoding="utf-8")
        log_progress(f"start input={input_path.as_posix()} output={output_dir.as_posix()}")
        with tempfile.TemporaryDirectory(prefix="phase3_log_reporter_") as tempdir:
            temp_root = Path(tempdir)
            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(temp_root)
                top_level_dirs = sorted({Path(name).parts[0] for name in zf.namelist() if name.strip("/")})
            if not top_level_dirs:
                raise FileNotFoundError(f"zip archive is empty: {input_path}")
            extracted_root = temp_root / top_level_dirs[0]
            if is_matrix_root(extracted_root):
                html_path = build_matrix_report(extracted_root, output_dir, args.top_events, args.switch_log_dir)
            else:
                html_path = build_single_experiment_report(extracted_root, output_dir, args.top_events, args.switch_log_dir)
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path / "report"
        output_dir.mkdir(parents=True, exist_ok=True)
        REPORT_LOG_PATH = output_dir / "phase3_reporter.log"
        REPORT_LOG_PATH.write_text("", encoding="utf-8")
        log_progress(f"start input={input_path.as_posix()} output={output_dir.as_posix()}")
        if is_matrix_root(input_path):
            html_path = build_matrix_report(input_path, output_dir, args.top_events, args.switch_log_dir)
        else:
            html_path = build_single_experiment_report(input_path, output_dir, args.top_events, args.switch_log_dir)

    log_progress(f"wrote html={html_path.as_posix()}")


if __name__ == "__main__":
    main()
