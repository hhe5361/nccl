#!/usr/bin/env python3
import argparse
import csv
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
        "matplotlib is required for phase2_log_reporter.py. Install it first, for example: pip install matplotlib"
    ) from exc


PHASE_RE = re.compile(r"(PHASE\d+)\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
WORKER_RE = re.compile(r"worker(\d+)$")
EXPERIMENT_RE = re.compile(r"^\d+_.+")

MODE_ORDER = {"STOCK": 0, "B2": 1, "B3": 2}

RUN_SUMMARY_FIELD_HELP = {
    "mode": "실험 모드. STOCK, B2, B3 등.",
    "phase2_mode": "runner가 기록한 내부 phase2 mode 이름.",
    "collective": "실험 collective 종류.",
    "world_size": "참여 rank 수.",
    "payload_mb": "rank당 logical payload 크기(MiB).",
    "step_ms_avg": "step max latency 평균(ms).",
    "step_ms_p95": "step max latency p95(ms).",
    "collective_gbps_avg": "collective 예상 traffic volume 기반 처리량 평균(Gbps).",
    "collective_gbps_p95": "collective 예상 처리량 p95(Gbps).",
    "delta_step_vs_stock_pct": "같은 experiment 안에서 stock 대비 step_ms_avg 변화율(%). 음수면 latency 개선.",
    "delta_bw_vs_stock_pct": "같은 experiment 안에서 stock 대비 collective_gbps_avg 변화율(%). 양수면 throughput 개선.",
}

NCCL_SUMMARY_FIELD_HELP = {
    "total_events": "수집된 PHASE 이벤트 총 수.",
    "total_wstall_count": "이벤트 이름에 WSTALL이 포함된 총 횟수.",
    "recv_wstall_count": "RECV 관련 WSTALL 총 횟수.",
    "send_wstall_count": "SEND 관련 WSTALL 총 횟수.",
    "p99_occ_pd": "posted-done의 p99.",
    "p99_occ_tr": "transmitted-done의 p99.",
    "w_eff_values": "WINDOW_CFG 계열 이벤트에서 관측한 effective window 집합.",
    "decision_counts": "B3 decision 이벤트의 reason별 빈도.",
    "pressure_score_p95": "pressureScore 계열 로그의 p95.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 2 PNG plots and HTML report")
    parser.add_argument(
        "--input",
        required=True,
        help="Single experiment root or matrix root",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <input>/report",
    )
    parser.add_argument(
        "--top-events",
        type=int,
        default=12,
        help="Number of top NCCL events to visualize",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


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
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def positive_ylim(max_value: float) -> Optional[Tuple[float, float]]:
    if max_value <= 0:
        return None
    return 0.0, max_value * 1.05


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


def collect_nccl_logs(mode_dir: Path) -> List[dict]:
    events: List[dict] = []
    for log_path in sorted(mode_dir.glob("*/nccl.*.log")):
        worker = log_path.parent.name
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PHASE_RE.search(line)
            if not match:
                continue
            phase = match.group(1)
            payload = match.group(2)
            fields = {key: parse_value(value) for key, value in KV_RE.findall(payload)}
            event_name = fields.get("event")
            if not event_name:
                continue
            fields["_phase"] = phase
            fields["_worker"] = worker
            events.append(fields)
    return events


def summarize_nccl_events(events: List[dict]) -> dict:
    specific_alias_keys = set()
    for entry in events:
        event_name = str(entry.get("event", ""))
        family = None
        if event_name.endswith("WINDOW_CFG"):
            family = "WINDOW_CFG"
        elif event_name.endswith("RECV_WSTALL"):
            family = "RECV_WSTALL"
        if family and ("_B2_" in event_name or "_B3_" in event_name):
            specific_alias_keys.add(
                (
                    entry.get("_worker"),
                    entry.get("tNs"),
                    entry.get("peer"),
                    entry.get("channel"),
                    entry.get("slot"),
                    family,
                )
            )

    event_counts: Counter = Counter()
    worker_event_counts: Dict[str, Counter] = defaultdict(Counter)
    worker_wstall_counts: Counter = Counter()
    decision_counts: Counter = Counter()
    w_eff_values: List[float] = []
    pressure_scores: List[float] = []
    occ_pd_values: List[float] = []
    occ_tr_values: List[float] = []

    recv_wstall_count = 0
    send_wstall_count = 0

    for entry in events:
        event_name = str(entry.get("event", ""))
        worker = str(entry.get("_worker", "unknown"))
        if not event_name:
            continue

        family = None
        if event_name == "PROXY_WINDOW_CFG":
            family = "WINDOW_CFG"
        elif event_name == "PROXY_RECV_WSTALL":
            family = "RECV_WSTALL"
        if family:
            key = (
                entry.get("_worker"),
                entry.get("tNs"),
                entry.get("peer"),
                entry.get("channel"),
                entry.get("slot"),
                family,
            )
            if key in specific_alias_keys:
                continue

        event_counts[event_name] += 1
        worker_event_counts[worker][event_name] += 1

        if "WSTALL" in event_name:
            worker_wstall_counts[worker] += 1
            if "RECV" in event_name:
                recv_wstall_count += 1
            if "SEND" in event_name:
                send_wstall_count += 1

        if event_name.endswith("DECISION") and "reason" in entry:
            decision_counts[str(entry["reason"])] += 1

        if "wEff" in entry:
            try:
                w_eff_values.append(float(entry["wEff"]))
            except (TypeError, ValueError):
                pass
        elif "newW" in entry:
            try:
                w_eff_values.append(float(entry["newW"]))
            except (TypeError, ValueError):
                pass

        if "pressureScore" in entry:
            try:
                pressure_scores.append(float(entry["pressureScore"]))
            except (TypeError, ValueError):
                pass

        if "occPd" in entry:
            try:
                occ_pd_values.append(float(entry["occPd"]))
            except (TypeError, ValueError):
                pass
        if "occTr" in entry:
            try:
                occ_tr_values.append(float(entry["occTr"]))
            except (TypeError, ValueError):
                pass

    return {
        "event_counts": event_counts,
        "worker_event_counts": worker_event_counts,
        "worker_wstall_counts": worker_wstall_counts,
        "decision_counts": decision_counts,
        "w_eff_values": sorted({int(v) if float(v).is_integer() else v for v in w_eff_values}, key=float),
        "w_eff_counter": Counter(int(v) if float(v).is_integer() else v for v in w_eff_values),
        "pressure_scores": pressure_scores,
        "occ_pd_values": occ_pd_values,
        "occ_tr_values": occ_tr_values,
        "total_events": int(sum(event_counts.values())),
        "total_wstall_count": int(recv_wstall_count + send_wstall_count),
        "recv_wstall_count": int(recv_wstall_count),
        "send_wstall_count": int(send_wstall_count),
        "max_occ_pd": max(occ_pd_values) if occ_pd_values else 0.0,
        "p99_occ_pd": percentile(occ_pd_values, 0.99),
        "max_occ_tr": max(occ_tr_values) if occ_tr_values else 0.0,
        "p99_occ_tr": percentile(occ_tr_values, 0.99),
        "pressure_score_p95": percentile(pressure_scores, 0.95),
    }


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


def load_mode_data(mode_dir: Path) -> dict:
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

    events = collect_nccl_logs(mode_dir)
    nccl_summary = summarize_nccl_events(events)
    mode_name = str(summary.get("run_tag") or summary.get("phase2_mode") or mode_dir.name).upper()

    return {
        "mode": mode_name,
        "summary_path": summary_path,
        "step_path": step_path,
        "summary": summary,
        "step_rows": step_rows,
        "events": events,
        "nccl": nccl_summary,
        "mode_dir": mode_dir,
    }


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
    .grid-2, .grid-3 {{
      display: grid;
      gap: 14px;
      margin-top: 16px;
    }}
    .grid-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
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
    th {{
      background: #f4f7fb;
    }}
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
    code {{
      font-family: Consolas, "SFMono-Regular", Monaco, monospace;
      font-size: 0.93em;
      background: #f2f5fa;
      border: 1px solid #e1e7f0;
      border-radius: 6px;
      padding: 1px 6px;
      color: #1d2b3a;
    }}
    ul {{ margin: 10px 0 0; padding-left: 20px; }}
    li + li {{ margin-top: 8px; }}
    @media (max-width: 920px) {{
      .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">Phase2 Reporter</div>
      <h1>{html.escape(title)}</h1>
      <p>Phase2 runner가 남긴 env_setup, summary, step_metrics, NCCL PHASE logs를 읽어 PNG plot과 HTML report를 생성한다.</p>
    </header>
    {''.join(sections)}
  </div>
</body>
</html>"""


def save_line_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: List[Tuple[str, List[float], List[float]]],
) -> None:
    if not series:
        return
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


def save_grouped_bar(
    path: Path,
    title: str,
    categories: List[str],
    series: List[Tuple[str, List[float]]],
    ylabel: str,
    rotate_labels: bool = False,
) -> None:
    if not categories or not series:
        return
    plt.figure(figsize=(max(10, len(categories) * 0.9), 5.8))
    x = list(range(len(categories)))
    width = 0.8 / max(len(series), 1)
    max_value = 0.0
    for idx, (label, values) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        plt.bar([pos + offset for pos in x], values, width=width, label=label)
        if values:
            max_value = max(max_value, max(values))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, categories, rotation=35 if rotate_labels else 0, ha="right" if rotate_labels else "center")
    ylim = positive_ylim(max_value)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def save_horizontal_bar(path: Path, title: str, labels: List[str], values: List[float], xlabel: str) -> None:
    if not labels:
        return
    plt.figure(figsize=(10, max(4.5, len(labels) * 0.45)))
    y = list(range(len(labels)))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    max_value = max(values) if values else 0.0
    ylim = positive_ylim(max_value)
    if ylim:
        plt.xlim(*ylim)
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def save_event_counts_plot(path: Path, mode_data: List[dict], top_n: int) -> Optional[Path]:
    total = Counter()
    for item in mode_data:
        total.update(item["nccl"]["event_counts"])
    labels = [name for name, _ in total.most_common(top_n)]
    if not labels:
        return None
    series = []
    for item in mode_data:
        values = [float(item["nccl"]["event_counts"].get(label, 0)) for label in labels]
        series.append((item["mode"], values))
    save_grouped_bar(path, "Top NCCL Event Counts by Mode", labels, series, "count", rotate_labels=True)
    return path


def save_worker_wstall_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    workers = sorted(
        {worker for item in mode_data for worker in item["nccl"]["worker_wstall_counts"].keys()},
        key=worker_sort_key,
    )
    if not workers:
        return None
    series = []
    for item in mode_data:
        counter = item["nccl"]["worker_wstall_counts"]
        series.append((item["mode"], [float(counter.get(worker, 0)) for worker in workers]))
    save_grouped_bar(path, "Window Stall Count by Worker", workers, series, "wstall count")
    return path


def save_window_distribution_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    windows = sorted(
        {window for item in mode_data for window in item["nccl"]["w_eff_counter"].keys()},
        key=float,
    )
    if not windows:
        return None
    labels = [str(window) for window in windows]
    series = []
    for item in mode_data:
        counter = item["nccl"]["w_eff_counter"]
        series.append((item["mode"], [float(counter.get(window, 0)) for window in windows]))
    save_grouped_bar(path, "Selected Window Distribution", labels, series, "count")
    return path


def save_decision_counts_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    reasons = sorted(
        {reason for item in mode_data for reason in item["nccl"]["decision_counts"].keys()}
    )
    if not reasons:
        return None
    series = []
    for item in mode_data:
        counts = item["nccl"]["decision_counts"]
        series.append((item["mode"], [float(counts.get(reason, 0)) for reason in reasons]))
    save_grouped_bar(path, "Decision Reason Counts", reasons, series, "count")
    return path


def save_pressure_summary_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    modes = [item["mode"] for item in mode_data if item["nccl"]["pressure_scores"]]
    if not modes:
        return None
    p50_values = [percentile(item["nccl"]["pressure_scores"], 0.50) for item in mode_data if item["nccl"]["pressure_scores"]]
    p95_values = [percentile(item["nccl"]["pressure_scores"], 0.95) for item in mode_data if item["nccl"]["pressure_scores"]]
    save_grouped_bar(
        path,
        "Pressure Score Summary",
        modes,
        [("p50", p50_values), ("p95", p95_values)],
        "pressure score",
    )
    return path


def save_occupancy_summary_plot(path: Path, mode_data: List[dict]) -> Optional[Path]:
    modes = [item["mode"] for item in mode_data]
    if not any(item["nccl"]["occ_pd_values"] or item["nccl"]["occ_tr_values"] for item in mode_data):
        return None
    save_grouped_bar(
        path,
        "Outstanding Depth Summary by Mode",
        modes,
        [
            ("p99_occ_pd", [float(item["nccl"]["p99_occ_pd"]) for item in mode_data]),
            ("p99_occ_tr", [float(item["nccl"]["p99_occ_tr"]) for item in mode_data]),
        ],
        "depth",
    )
    return path


def summarize_modes(mode_data: List[dict]) -> List[dict]:
    stock_summary = None
    for item in mode_data:
        if item["mode"].upper() == "STOCK":
            stock_summary = item["summary"]
            break

    rows = []
    for item in mode_data:
        summary = item["summary"]
        nccl = item["nccl"]
        row = {
            "mode": item["mode"],
            "phase2_mode": summary.get("phase2_mode", "").upper() or item["mode"],
            "collective": summary.get("collective", ""),
            "world_size": int(summary.get("world_size", 0)),
            "payload_mb": float(summary.get("payload_mb", 0.0)),
            "step_ms_avg": float(summary.get("step_ms_avg", 0.0)),
            "step_ms_p95": float(summary.get("step_ms_p95", 0.0)),
            "collective_gbps_avg": float(summary.get("collective_gbps_avg", 0.0)),
            "collective_gbps_p95": float(summary.get("collective_gbps_p95", 0.0)),
            "total_events": nccl["total_events"],
            "total_wstall_count": nccl["total_wstall_count"],
            "recv_wstall_count": nccl["recv_wstall_count"],
            "send_wstall_count": nccl["send_wstall_count"],
            "p99_occ_pd": float(nccl["p99_occ_pd"]),
            "p99_occ_tr": float(nccl["p99_occ_tr"]),
            "w_eff_values": ",".join(str(v) for v in nccl["w_eff_values"]),
            "decision_counts": ", ".join(f"{k}:{v}" for k, v in sorted(nccl["decision_counts"].items())) or "-",
            "pressure_score_p95": float(nccl["pressure_score_p95"]),
            "delta_step_vs_stock_pct": 0.0,
            "delta_bw_vs_stock_pct": 0.0,
            "delta_occ_tr_vs_stock_pct": 0.0,
            "delta_wstall_vs_stock_pct": 0.0,
        }
        if stock_summary is not None and item["mode"].upper() != "STOCK":
            row["delta_step_vs_stock_pct"] = rel_change(
                float(stock_summary.get("step_ms_avg", 0.0)),
                float(summary.get("step_ms_avg", 0.0)),
            )
            row["delta_bw_vs_stock_pct"] = rel_change(
                float(stock_summary.get("collective_gbps_avg", 0.0)),
                float(summary.get("collective_gbps_avg", 0.0)),
            )
            stock_nccl = next(m["nccl"] for m in mode_data if m["mode"].upper() == "STOCK")
            row["delta_occ_tr_vs_stock_pct"] = rel_change(
                float(stock_nccl["p99_occ_tr"]),
                float(nccl["p99_occ_tr"]),
            )
            row["delta_wstall_vs_stock_pct"] = rel_change(
                float(stock_nccl["total_wstall_count"]),
                float(nccl["total_wstall_count"]),
            )
        rows.append(row)
    return rows


def build_collection_plan_card_single(experiment_root: Path, env_setup: Optional[dict], mode_data: List[dict]) -> str:
    total_logs = sum(len(list(item["mode_dir"].glob("*/nccl.*.log"))) for item in mode_data)
    rows = [
        ["env_setup.json", "single experiment 설정값. collective, payload, mode, rack map 등을 기록."],
        ["MODE/*_summary.json", "mode별 최종 요약. step latency와 collective throughput의 평균/p95를 제공."],
        ["MODE/*_step_metrics.jsonl", "step별 시계열. warmup 여부와 step_ms_max/mean, collective_gbps_est를 포함."],
        [f"MODE/workerXX/nccl.*.log ({total_logs} files)", "PHASE0/1/2 이벤트 로그. WSTALL, WINDOW_CFG, pressure/decision trace를 포함할 수 있음."],
    ]
    if env_setup and "run_modes" in env_setup:
        rows.append(["run_modes", str(env_setup["run_modes"])])
    return '<div class="card"><h2>Collected Logs</h2>' + render_table(["source", "meaning"], rows) + '</div>'


def build_visualization_plan_card_single(mode_data: List[dict]) -> str:
    rows = [
        ("step_timeline.png", "*_step_metrics.jsonl", "mode별 step_ms_max / step_ms_mean 시계열 비교"),
        ("throughput_timeline.png", "*_step_metrics.jsonl", "mode별 collective_gbps_est 시계열 비교"),
        ("summary_latency.png", "*_summary.json", "mode별 step_ms_avg / step_ms_p95 비교"),
        ("summary_throughput.png", "*_summary.json", "mode별 collective_gbps_avg / p95 비교"),
        ("event_counts.png", "nccl.*.log", "top NCCL event 분포 비교"),
        ("wstall_by_worker.png", "nccl.*.log", "worker별 WSTALL 편차 확인"),
        ("selected_window_distribution.png", "WINDOW_CFG / DECISION log", "선택된 W_eff 분포 확인"),
        ("occupancy_summary.png", "nccl.*.log", "p99_occ_pd / p99_occ_tr 비교"),
    ]
    if any(item["nccl"]["decision_counts"] for item in mode_data):
        rows.append(("decision_counts.png", "DECISION log", "shrink / hold / recover reason count"))
    if any(item["nccl"]["pressure_scores"] for item in mode_data):
        rows.append(("pressure_summary.png", "PRESSURE log", "mode별 pressure score p50 / p95"))
    return render_plot_help_card("Visualization Plan", rows)


def build_single_experiment_report(experiment_root: Path, output_dir: Path, top_events: int) -> Path:
    env_setup = load_json(experiment_root / "env_setup.json") if (experiment_root / "env_setup.json").exists() else None
    mode_dirs = find_mode_dirs(experiment_root)
    if not mode_dirs:
        raise FileNotFoundError(f"no mode directories found in {experiment_root}")

    mode_data = [load_mode_data(mode_dir) for mode_dir in mode_dirs]
    mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
    summary_rows = summarize_modes(mode_data)

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

    p_step = plots_dir / "step_timeline.png"
    save_line_plot(
        p_step,
        f"{experiment_root.name} Step Timeline",
        "step",
        "latency (ms)",
        step_series_max + step_series_mean,
    )

    p_bw = plots_dir / "throughput_timeline.png"
    save_line_plot(
        p_bw,
        f"{experiment_root.name} Collective Throughput Estimate",
        "step",
        "Gbps",
        bw_series,
    )

    categories = [row["mode"] for row in summary_rows]
    p_step_summary = plots_dir / "summary_latency.png"
    save_grouped_bar(
        p_step_summary,
        "Latency Summary by Mode",
        categories,
        [
            ("avg", [float(row["step_ms_avg"]) for row in summary_rows]),
            ("p95", [float(row["step_ms_p95"]) for row in summary_rows]),
        ],
        "ms",
    )

    p_bw_summary = plots_dir / "summary_throughput.png"
    save_grouped_bar(
        p_bw_summary,
        "Throughput Summary by Mode",
        categories,
        [
            ("avg", [float(row["collective_gbps_avg"]) for row in summary_rows]),
            ("p95", [float(row["collective_gbps_p95"]) for row in summary_rows]),
        ],
        "Gbps",
    )

    p_events = save_event_counts_plot(plots_dir / "event_counts.png", mode_data, top_events)
    p_wstall = save_worker_wstall_plot(plots_dir / "wstall_by_worker.png", mode_data)
    p_window = save_window_distribution_plot(plots_dir / "selected_window_distribution.png", mode_data)
    p_occ = save_occupancy_summary_plot(plots_dir / "occupancy_summary.png", mode_data)
    p_decision = save_decision_counts_plot(plots_dir / "decision_counts.png", mode_data)
    p_pressure = save_pressure_summary_plot(plots_dir / "pressure_summary.png", mode_data)

    summary_headers = [
        "mode", "collective", "payload_mb", "step_ms_avg", "step_ms_p95",
        "collective_gbps_avg", "collective_gbps_p95", "total_wstall_count",
        "p99_occ_tr", "w_eff_values", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct",
    ]
    summary_table = [
        [
            row["mode"],
            row["collective"],
            f'{row["payload_mb"]:.3f}',
            f'{row["step_ms_avg"]:.3f}',
            f'{row["step_ms_p95"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            f'{row["collective_gbps_p95"]:.3f}',
            row["total_wstall_count"],
            f'{row["p99_occ_tr"]:.3f}',
            row["w_eff_values"] or "-",
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
        ]
        for row in summary_rows
    ]

    info_rows = [
        ["experiment_root", experiment_root.as_posix()],
        ["modes", ", ".join(item["mode"] for item in mode_data)],
    ]
    if env_setup:
        for key in ["collective", "run_modes", "payload_mb", "dtype", "master_addr", "master_port_base", "policy_name"]:
            if key in env_setup:
                info_rows.append([key, env_setup[key]])

    sections = [
        '<section class="section"><h2>Experiment Metadata</h2>' + render_table(["field", "value"], info_rows) + "</section>",
        '<section class="section"><div class="grid-2">'
        + build_collection_plan_card_single(experiment_root, env_setup, mode_data)
        + build_visualization_plan_card_single(mode_data)
        + "</div></section>",
        '<section class="section"><div class="grid-2">'
        + render_field_help_card(
            "Run Summary Field Meanings",
            RUN_SUMMARY_FIELD_HELP,
            ["mode", "phase2_mode", "collective", "world_size", "payload_mb", "step_ms_avg", "step_ms_p95", "collective_gbps_avg", "collective_gbps_p95", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct"],
        )
        + render_field_help_card(
            "NCCL Summary Field Meanings",
            NCCL_SUMMARY_FIELD_HELP,
            ["total_events", "total_wstall_count", "recv_wstall_count", "send_wstall_count", "p99_occ_pd", "p99_occ_tr", "w_eff_values", "decision_counts", "pressure_score_p95"],
        )
        + "</div></section>",
        '<section class="section"><h2>Mode Summary</h2>' + render_table(summary_headers, summary_table) + "</section>",
    ]

    figure_fragments = []
    for path, caption in [
        (p_step, "mode별 step latency timeline"),
        (p_bw, "mode별 collective throughput estimate"),
        (p_step_summary, "mode별 latency summary"),
        (p_bw_summary, "mode별 throughput summary"),
        (p_events, "mode별 top NCCL event counts"),
        (p_wstall, "worker별 WSTALL count"),
        (p_window, "selected W_eff distribution"),
        (p_occ, "occupancy summary"),
        (p_decision, "decision reason count"),
        (p_pressure, "pressure score summary"),
    ]:
        if path is None or not path.exists():
            continue
        figure_fragments.append(
            f'<figure><img src="{html.escape(relpath(path, output_dir))}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figure_fragments) + "</section>")

    html_path = output_dir / "phase2_report.html"
    html_path.write_text(render_html(f"Phase2 Report - {experiment_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase2_report.json"
    report_json.write_text(
        json.dumps(
            {
                "experiment": experiment_root.name,
                "env_setup": env_setup,
                "modes": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_csv = output_dir / "phase2_report.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    return html_path


def collect_matrix_rows(experiment_name: str, mode_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for row in mode_rows:
        flat = dict(row)
        flat["experiment"] = experiment_name
        rows.append(flat)
    return rows


def build_collection_plan_card_matrix(matrix_root: Path, manifest: Optional[dict], experiment_dirs: List[Path]) -> str:
    total_logs = 0
    for experiment_dir in experiment_dirs:
        for mode_dir in find_mode_dirs(experiment_dir):
            total_logs += len(list(mode_dir.glob("*/nccl.*.log")))
    rows = [
        ["matrix_manifest.json", "matrix 전체 조합과 worker pool 정의."],
        ["NN_experiment/env_setup.json", "각 experiment의 collective, payload, run_modes, policy 정보."],
        ["NN_experiment/MODE/*_summary.json", "mode별 절대 성능 요약."],
        ["NN_experiment/MODE/*_step_metrics.jsonl", "step 시계열."],
        [f"NN_experiment/MODE/workerXX/nccl.*.log ({total_logs} files)", "NCCL PHASE 이벤트 로그."],
    ]
    if manifest and "experiments" in manifest:
        rows.append(["experiment_count", len(manifest["experiments"])])
    return '<div class="card"><h2>Collected Logs</h2>' + render_table(["source", "meaning"], rows) + '</div>'


def build_visualization_plan_card_matrix() -> str:
    rows = [
        ("matrix_step_latency.png", "per-experiment summary", "experiment별 mode 절대 step latency 비교"),
        ("matrix_throughput.png", "per-experiment summary", "experiment별 mode 절대 throughput 비교"),
        ("matrix_step_delta_vs_stock.png", "summary + stock baseline", "stock 대비 latency 변화율 비교"),
        ("matrix_bw_delta_vs_stock.png", "summary + stock baseline", "stock 대비 throughput 변화율 비교"),
        ("matrix_occ_tr_delta_vs_stock.png", "NCCL logs + stock baseline", "stock 대비 p99_occ_tr 변화율 비교"),
        ("matrix_wstall_delta_vs_stock.png", "NCCL logs + stock baseline", "stock 대비 total_wstall_count 변화율 비교"),
    ]
    return render_plot_help_card("Visualization Plan", rows)


def build_matrix_report(matrix_root: Path, output_dir: Path, top_events: int) -> Path:
    manifest = load_json(matrix_root / "matrix_manifest.json") if (matrix_root / "matrix_manifest.json").exists() else None
    experiment_dirs = find_experiment_dirs(matrix_root)
    if not experiment_dirs:
        raise FileNotFoundError(f"no experiment directories found in {matrix_root}")

    per_experiment_links: Dict[str, str] = {}
    matrix_rows: List[dict] = []

    for experiment_dir in experiment_dirs:
        per_output = output_dir / experiment_dir.name
        per_output.mkdir(parents=True, exist_ok=True)
        per_html = build_single_experiment_report(experiment_dir, per_output, top_events)
        per_experiment_links[experiment_dir.name] = relpath(per_html, output_dir)

        mode_dirs = find_mode_dirs(experiment_dir)
        mode_data = [load_mode_data(mode_dir) for mode_dir in mode_dirs]
        mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
        matrix_rows.extend(collect_matrix_rows(experiment_dir.name, summarize_modes(mode_data)))

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

    p_abs_step = plots_dir / "matrix_step_latency.png"
    save_grouped_bar(
        p_abs_step,
        "Matrix - Step Latency by Experiment",
        experiment_names,
        [(mode_name, values_for("step_ms_avg", mode_name)) for mode_name in mode_names],
        "step_ms_avg",
        rotate_labels=True,
    )

    p_abs_bw = plots_dir / "matrix_throughput.png"
    save_grouped_bar(
        p_abs_bw,
        "Matrix - Throughput by Experiment",
        experiment_names,
        [(mode_name, values_for("collective_gbps_avg", mode_name)) for mode_name in mode_names],
        "collective_gbps_avg",
        rotate_labels=True,
    )

    candidate_modes = [mode for mode in mode_names if mode != "STOCK"]
    p_delta_step = None
    p_delta_bw = None
    p_delta_occ = None
    p_delta_wstall = None
    if candidate_modes:
        p_delta_step = plots_dir / "matrix_step_delta_vs_stock.png"
        save_grouped_bar(
            p_delta_step,
            "Matrix - Step Latency Change vs Stock",
            experiment_names,
            [(mode_name, values_for("delta_step_vs_stock_pct", mode_name)) for mode_name in candidate_modes],
            "delta_step_vs_stock_pct",
            rotate_labels=True,
        )

        p_delta_bw = plots_dir / "matrix_bw_delta_vs_stock.png"
        save_grouped_bar(
            p_delta_bw,
            "Matrix - Throughput Change vs Stock",
            experiment_names,
            [(mode_name, values_for("delta_bw_vs_stock_pct", mode_name)) for mode_name in candidate_modes],
            "delta_bw_vs_stock_pct",
            rotate_labels=True,
        )

        p_delta_occ = plots_dir / "matrix_occ_tr_delta_vs_stock.png"
        save_grouped_bar(
            p_delta_occ,
            "Matrix - p99_occ_tr Change vs Stock",
            experiment_names,
            [(mode_name, values_for("delta_occ_tr_vs_stock_pct", mode_name)) for mode_name in candidate_modes],
            "delta_occ_tr_vs_stock_pct",
            rotate_labels=True,
        )

        p_delta_wstall = plots_dir / "matrix_wstall_delta_vs_stock.png"
        save_grouped_bar(
            p_delta_wstall,
            "Matrix - Total WSTALL Change vs Stock",
            experiment_names,
            [(mode_name, values_for("delta_wstall_vs_stock_pct", mode_name)) for mode_name in candidate_modes],
            "delta_wstall_vs_stock_pct",
            rotate_labels=True,
        )

    summary_headers = [
        "experiment", "mode", "collective", "payload_mb", "step_ms_avg", "collective_gbps_avg",
        "total_wstall_count", "p99_occ_tr", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "report",
    ]
    summary_rows_html = []
    for row in matrix_rows:
        link = per_experiment_links[row["experiment"]]
        summary_rows_html.append([
            html.escape(str(row["experiment"])),
            html.escape(str(row["mode"])),
            html.escape(str(row["collective"])),
            f'{row["payload_mb"]:.3f}',
            f'{row["step_ms_avg"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            str(row["total_wstall_count"]),
            f'{row["p99_occ_tr"]:.3f}',
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
            f'<a href="{html.escape(link)}">open</a>',
        ])

    sections = [
        '<section class="section"><div class="grid-2">'
        + build_collection_plan_card_matrix(matrix_root, manifest, experiment_dirs)
        + build_visualization_plan_card_matrix()
        + "</div></section>",
        '<section class="section"><div class="grid-2">'
        + render_field_help_card(
            "Run Summary Field Meanings",
            RUN_SUMMARY_FIELD_HELP,
            ["mode", "collective", "payload_mb", "step_ms_avg", "step_ms_p95", "collective_gbps_avg", "collective_gbps_p95", "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct"],
        )
        + render_field_help_card(
            "NCCL Summary Field Meanings",
            NCCL_SUMMARY_FIELD_HELP,
            ["total_events", "total_wstall_count", "recv_wstall_count", "send_wstall_count", "p99_occ_pd", "p99_occ_tr", "w_eff_values", "decision_counts", "pressure_score_p95"],
        )
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
        figure_fragments.append(
            f'<figure><img src="{html.escape(relpath(path, output_dir))}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figure_fragments) + "</section>")

    report_list = "".join(
        f'<li><a href="{html.escape(link)}">{html.escape(name)}</a></li>'
        for name, link in sorted(per_experiment_links.items())
    )
    sections.append('<section class="section"><h2>Per-experiment Reports</h2><ul>' + report_list + "</ul></section>")

    html_path = output_dir / "phase2_report.html"
    html_path.write_text(render_html(f"Phase2 Matrix Report - {matrix_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase2_report.json"
    report_json.write_text(
        json.dumps({"matrix_root": matrix_root.name, "rows": matrix_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_csv = output_dir / "phase2_report.csv"
    if matrix_rows:
        with report_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matrix_rows)

    return html_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()

    if input_path.suffix.lower() == ".zip":
        default_output = input_path.with_name(f"{input_path.stem}_report")
        output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phase2_log_reporter_") as tempdir:
            temp_root = Path(tempdir)
            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(temp_root)
                top_level_dirs = sorted({Path(name).parts[0] for name in zf.namelist() if name.strip("/")})
            if not top_level_dirs:
                raise FileNotFoundError(f"zip archive is empty: {input_path}")
            extracted_root = temp_root / top_level_dirs[0]
            if is_matrix_root(extracted_root):
                html_path = build_matrix_report(extracted_root, output_dir, args.top_events)
            else:
                html_path = build_single_experiment_report(extracted_root, output_dir, args.top_events)
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path / "report"
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_matrix_root(input_path):
            html_path = build_matrix_report(input_path, output_dir, args.top_events)
        else:
            html_path = build_single_experiment_report(input_path, output_dir, args.top_events)

    print(f"[phase2-log-reporter] wrote {html_path}")


if __name__ == "__main__":
    main()
