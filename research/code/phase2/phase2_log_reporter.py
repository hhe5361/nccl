#!/usr/bin/env python3
import argparse
import csv
import html
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR.parent / "phase3" / "phase3_log_reporter.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("phase3_log_reporter_base_for_phase2", BASE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 2 PNG plots and HTML report")
    parser.add_argument("--input-dir", required=True, help="Single experiment root or matrix root")
    parser.add_argument("--output-dir", required=True, help="Report output directory")
    parser.add_argument("--switch-log", default=None, help="Override switch log directory")
    parser.add_argument("--top-events", type=int, default=12, help="Number of top NCCL events to visualize")
    return parser.parse_args()


def mode_sort_key(name: str) -> Tuple[int, int, str]:
    mode = str(name or "").upper()
    if mode == "STOCK":
        return (0, 0, mode)
    if mode.startswith("B2_W"):
        try:
            return (1, int(mode.split("_W", 1)[1]), mode)
        except (IndexError, ValueError):
            return (1, 10**9, mode)
    return (10**9, 10**9, mode)


def static_w_from_mode(mode: str, summary: dict) -> int:
    if "phase1_static_w" in summary:
        try:
            return int(summary["phase1_static_w"])
        except (TypeError, ValueError):
            pass
    upper = str(mode or "").upper()
    if upper == "STOCK":
        return 0
    if upper.startswith("B2_W"):
        try:
            return int(upper.split("_W", 1)[1])
        except (IndexError, ValueError):
            return -1
    return -1


def effective_rows(step_rows: List[dict]) -> List[dict]:
    rows = [row for row in step_rows if not row.get("warmup", False)]
    return rows if rows else step_rows


def experiment_label_from_env(experiment_root: Path, env_setup: Optional[dict]) -> str:
    if env_setup and env_setup.get("experiment_label"):
        return str(env_setup["experiment_label"])
    return experiment_root.name


def workload_label(collective: str, algo: str) -> str:
    algo = str(algo or "auto")
    if collective == "allreduce":
        return f"{collective}_{algo.lower()}"
    return f"{collective}_{algo.lower()}"


def extract_experiment_algo(env_setup: Optional[dict], mode_data: List[dict]) -> str:
    env_algo = str((env_setup or {}).get("nccl_algo", "") or "").strip()
    if env_algo and env_algo != "auto":
        return env_algo
    for item in mode_data:
        algo = str(item["summary"].get("algo", "") or "").strip()
        if algo:
            return algo
    return "auto"


def mode_step_window_series(mode_item: dict) -> Tuple[List[int], List[float]]:
    worker_windows = mode_item["nccl"]["worker_windows"]
    per_step: Dict[int, List[float]] = {}
    for worker, info in worker_windows.items():
        trace_steps = info.get("trace_step", [])
        trace_w = info.get("trace_w", [])
        if not trace_steps or not trace_w:
            continue
        last_by_step: Dict[int, float] = {}
        for step, w_value in zip(trace_steps, trace_w):
            if step is None or int(step) < 0:
                continue
            last_by_step[int(step)] = float(w_value)
        for step, w_value in last_by_step.items():
            per_step.setdefault(step, []).append(w_value)
    if not per_step:
        return [], []
    steps = sorted(per_step.keys())
    values = [base.mean(per_step[step]) for step in steps]
    return steps, values


def compute_switch_metrics(
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str],
) -> Dict[str, dict]:
    switch_bundle = base.load_switch_bundle(experiment_root, env_setup, switch_log_dir_override)
    if not switch_bundle:
        return {}
    mode_windows = base.build_mode_time_windows(mode_data, env_setup, switch_bundle.get("markers", []))
    metrics: Dict[str, dict] = {}
    for item in mode_data:
      mode = item["mode"]
      bounds = mode_windows.get(mode)
      if bounds is None:
          metrics[mode] = {"switch_pfc_total": 0.0, "switch_pfc_peak_rate": 0.0}
          continue
      start_ns, end_ns = bounds
      total = 0.0
      peak = 0.0
      for label in ("rackA", "rackB"):
          snapshots = switch_bundle.get("snapshots", {}).get(label, [])
          if snapshots:
              start_value = base.interpolate_switch_snapshot_value(snapshots, start_ns)
              end_value = base.interpolate_switch_snapshot_value(snapshots, end_ns)
              if start_value is not None and end_value is not None:
                  total += max(0.0, float(end_value) - float(start_value))
          for ts_ns, rate in switch_bundle.get("series", {}).get(label, []):
              if start_ns <= ts_ns <= end_ns:
                  peak = max(peak, float(rate))
      metrics[mode] = {
          "switch_pfc_total": total,
          "switch_pfc_peak_rate": peak,
      }
    return metrics


def load_mode_data(mode_dir: Path) -> dict:
    return base.load_mode_data(mode_dir)


def load_rank_validations(mode_dir: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for path in sorted(mode_dir.glob("*/" + mode_dir.name + "_rank_validation.json")):
        row = base.load_json(path)
        rows[str(row.get("worker") or path.parent.name)] = row
    return rows


def compare_final_output_vs_stock(mode_data: List[dict]) -> Dict[str, dict]:
    validations = {item["mode"]: load_rank_validations(item["mode_dir"]) for item in mode_data}
    stock_rows = validations.get("STOCK", {})
    result: Dict[str, dict] = {}
    stock_total = len(stock_rows)
    result["STOCK"] = {
        "all_match": True,
        "matched_workers": stock_total,
        "total_workers": stock_total,
    }
    for mode, mode_rows in validations.items():
        if mode == "STOCK":
            continue
        workers = sorted(set(stock_rows.keys()) | set(mode_rows.keys()))
        matched = 0
        for worker in workers:
            s = stock_rows.get(worker)
            m = mode_rows.get(worker)
            if s and m and s.get("final_sha256") == m.get("final_sha256"):
                matched += 1
        total = len(workers)
        result[mode] = {
            "all_match": total > 0 and matched == total,
            "matched_workers": matched,
            "total_workers": total,
        }
    return result


def summarize_modes(
    experiment_root: Path,
    env_setup: Optional[dict],
    mode_data: List[dict],
    switch_log_dir_override: Optional[str],
) -> List[dict]:
    switch_metrics = compute_switch_metrics(experiment_root, env_setup, mode_data, switch_log_dir_override)
    stock_digest_match = compare_final_output_vs_stock(mode_data)
    stock = next((item for item in mode_data if item["mode"] == "STOCK"), None)
    rows = []
    experiment_label = experiment_label_from_env(experiment_root, env_setup)
    algo = extract_experiment_algo(env_setup, mode_data)

    for item in mode_data:
        summary = item["summary"]
        nccl = item["nccl"]
        mode = item["mode"]
        static_w_cfg = static_w_from_mode(mode, summary)
        switch_row = switch_metrics.get(mode, {"switch_pfc_total": 0.0, "switch_pfc_peak_rate": 0.0})
        row = {
            "experiment": experiment_label,
            "mode": mode,
            "static_w_cfg": static_w_cfg,
            "collective": summary.get("collective", ""),
            "algo": algo,
            "workload": workload_label(str(summary.get("collective", "")), algo),
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
            "switch_pfc_total": float(switch_row["switch_pfc_total"]),
            "switch_pfc_peak_rate": float(switch_row["switch_pfc_peak_rate"]),
            "final_output_match_vs_stock": "-",
            "final_output_match_workers": "-",
            "delta_step_vs_stock_pct": 0.0,
            "delta_bw_vs_stock_pct": 0.0,
            "delta_occ_tr_vs_stock_pct": 0.0,
            "delta_wstall_vs_stock_pct": 0.0,
            "delta_pfc_total_vs_stock_pct": 0.0,
            "delta_pfc_peak_vs_stock_pct": 0.0,
        }
        digest_row = stock_digest_match.get(mode)
        if digest_row is not None:
            row["final_output_match_vs_stock"] = "PASS" if digest_row["all_match"] else "FAIL"
            row["final_output_match_workers"] = f'{digest_row["matched_workers"]}/{digest_row["total_workers"]}'
        if stock is not None and mode != "STOCK":
            stock_summary = stock["summary"]
            stock_nccl = stock["nccl"]
            stock_switch = switch_metrics.get("STOCK", {"switch_pfc_total": 0.0, "switch_pfc_peak_rate": 0.0})
            row["delta_step_vs_stock_pct"] = base.rel_change(float(stock_summary.get("step_ms_avg", 0.0)), row["step_ms_avg"])
            row["delta_bw_vs_stock_pct"] = base.rel_change(float(stock_summary.get("collective_gbps_avg", 0.0)), row["collective_gbps_avg"])
            row["delta_occ_tr_vs_stock_pct"] = base.rel_change(float(stock_nccl["p99_occ_tr"]), row["p99_occ_tr"])
            row["delta_wstall_vs_stock_pct"] = base.rel_change(float(stock_nccl["total_wstall_count"]), row["total_wstall_count"])
            row["delta_pfc_total_vs_stock_pct"] = base.rel_change(float(stock_switch["switch_pfc_total"]), row["switch_pfc_total"])
            row["delta_pfc_peak_vs_stock_pct"] = base.rel_change(float(stock_switch["switch_pfc_peak_rate"]), row["switch_pfc_peak_rate"])
        rows.append(row)
    rows.sort(key=lambda row: mode_sort_key(str(row["mode"])))
    return rows


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
      <div class="eyebrow">Phase2 Reporter</div>
      <h1>{html.escape(title)}</h1>
      <p>4-node static receiver window sweep report. B3 path is disabled during collection; only stock and PHASE1 static W override runs are visualized.</p>
    </header>
    {''.join(sections)}
  </div>
</body>
</html>"""


def build_collection_plan_card_matrix(experiment_dirs: List[Path]) -> str:
    total_logs = 0
    total_step_timing = 0
    for experiment_dir in experiment_dirs:
        for mode_dir in base.find_mode_dirs(experiment_dir):
            total_logs += len(list(mode_dir.glob("*/nccl.*.log")))
            total_step_timing += len(list(mode_dir.glob("*/**/*_worker_step_timing.jsonl")))
    rows = [
        ["matrix_manifest.json", "experiment matrix, mode sweep, worker pool definition"],
        ["NN_experiment/env_setup.json", "experiment metadata, static-W sweep contract, B3 disabled flag"],
        ["NN_experiment/MODE/*_summary.json", "mode absolute performance summary"],
        ["NN_experiment/MODE/*_step_metrics.jsonl", "per-step collective latency / throughput timeline"],
        [f"NN_experiment/MODE/workerXX/*_worker_step_timing.jsonl ({total_step_timing} files)", "worker-local timing used to align W traces on collective step axis"],
        [f"NN_experiment/MODE/workerXX/nccl.*.log ({total_logs} files)", "PHASE0 logs used for W_eff, outstanding depth, WSTALL"],
        ["switch_log/*.jsonl", "override or env_setup-provided switch PFC / ROCE logs"],
    ]
    return '<div class="card"><h2>Collected Logs</h2>' + base.render_table(["source", "meaning"], rows) + '</div>'


def build_visualization_plan_card_matrix() -> str:
    rows = [
        ["stock_vs_b2_latency.png", "experiment별 STOCK/B2_W step latency 비교"],
        ["stock_vs_b2_throughput.png", "experiment별 STOCK/B2_W throughput 비교"],
        ["stock_vs_b2_switch_pfc_counts.png", "experiment별 STOCK/B2_W PFC total count 비교"],
        ["stock_vs_b2_step_window_trace.png", "experiment별 step-axis 평균 W trace"],
        ["b2_matrix_latency.png", "B2만 남겨 W sweep × workload latency heatmap"],
        ["b2_matrix_throughput.png", "B2만 남겨 W sweep × workload throughput heatmap"],
        ["b2_matrix_pfc_count.png", "B2만 남겨 W sweep × workload PFC total heatmap"],
        ["b2_matrix_pfc_severity.png", "B2만 남겨 W sweep × workload PFC peak rate heatmap"],
    ]
    return '<div class="card"><h2>Visualization Plan</h2>' + base.render_table(["plot", "meaning"], rows) + '</div>'


def save_window_trace_by_mode_plot(path: Path, mode_data: List[dict], title: str) -> Optional[Path]:
    series = []
    for item in mode_data:
        x_values, y_values = mode_step_window_series(item)
        if x_values and y_values:
            series.append((item["mode"], x_values, y_values))
    if not series:
        return None
    return base.save_line_plot(path, title, "collective step", "avg receiver W_eff", series)


def save_switch_pfc_counts_plot(path: Path, summary_rows: List[dict], title: str) -> Optional[Path]:
    categories = [row["mode"] for row in summary_rows]
    if not categories:
        return None
    series = [
        ("pfc_total", [float(row["switch_pfc_total"]) for row in summary_rows]),
        ("pfc_peak_rate", [float(row["switch_pfc_peak_rate"]) for row in summary_rows]),
    ]
    return base.save_grouped_bar(path, title, categories, series, "switch PFC metric")


def save_worker_window_trace_plot(path: Path, mode_item: dict) -> Optional[Path]:
    worker_windows = mode_item["nccl"]["worker_windows"]
    workers = [worker for worker, info in sorted(worker_windows.items(), key=lambda item: base.worker_sort_key(item[0])) if info["trace_x"] and info["trace_w"]]
    if not workers:
        return None
    base.plt.figure(figsize=(11, 5.5))
    max_value = 0.0
    for worker in workers:
        trace_steps = worker_windows[worker].get("trace_step", [])
        trace_w = worker_windows[worker]["trace_w"]
        if trace_steps and any(step >= 0 for step in trace_steps):
            per_step: Dict[int, float] = {}
            for step, w_value in zip(trace_steps, trace_w):
                if int(step) < 0:
                    continue
                per_step[int(step)] = float(w_value)
            if not per_step:
                continue
            plot_x = sorted(per_step.keys())
            plot_y = [per_step[idx] for idx in plot_x]
        else:
            plot_x = worker_windows[worker]["trace_x"]
            plot_y = trace_w
        base.plt.plot(plot_x, plot_y, marker="o", linewidth=1.5, markersize=3, label=worker)
        max_value = max(max_value, max(plot_y))
    base.plt.title(f"{mode_item['mode']} Receiver Window Trace by Worker")
    base.plt.xlabel("collective step")
    base.plt.ylabel("W_eff")
    ylim = base.positive_ylim(max_value)
    if ylim:
        base.plt.ylim(*ylim)
    base.plt.grid(True, alpha=0.25)
    base.plt.legend(ncol=2)
    base.plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(path, dpi=180)
    base.plt.close()
    return path


def save_matrix_step_window_trace_plot(path: Path, experiment_payloads: List[Tuple[str, List[dict]]]) -> Optional[Path]:
    if not experiment_payloads:
        return None
    fig, axes = base.plt.subplots(len(experiment_payloads), 1, figsize=(12, 3.6 * len(experiment_payloads)), sharex=False)
    if len(experiment_payloads) == 1:
        axes = [axes]
    any_series = False
    for ax, (experiment_name, mode_data) in zip(axes, experiment_payloads):
        for item in mode_data:
            x_values, y_values = mode_step_window_series(item)
            if not x_values or not y_values:
                continue
            any_series = True
            ax.plot(x_values, y_values, marker="o", linewidth=1.5, markersize=2.8, label=item["mode"])
        ax.set_title(experiment_name)
        ax.set_xlabel("collective step")
        ax.set_ylabel("avg W_eff")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=4)
    if not any_series:
        base.plt.close(fig)
        return None
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    base.plt.close(fig)
    return path


def save_heatmap(
    path: Path,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: Sequence[Sequence[float]],
    colorbar_label: str,
) -> Optional[Path]:
    if not row_labels or not col_labels or not values:
        return None
    fig, ax = base.plt.subplots(figsize=(max(8, len(col_labels) * 1.0), max(4, len(row_labels) * 0.8 + 1.5)))
    im = ax.imshow(values, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="#111", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    base.plt.close(fig)
    return path


def build_worker_window_table(mode_item: dict) -> str:
    rows = []
    worker_windows = mode_item["nccl"]["worker_windows"]
    for worker, info in sorted(worker_windows.items(), key=lambda item: base.worker_sort_key(item[0])):
        rows.append([
            worker,
            ",".join(str(v) for v in info["initial_w_values"]) or "-",
            ",".join(str(v) for v in info["final_w_values"]) or "-",
            len(info["trace_x"]),
        ])
    return base.render_table(["worker", "initial_w", "final_w", "samples"], rows)


def build_single_experiment_report(
    experiment_root: Path,
    output_dir: Path,
    top_events: int,
    switch_log_dir_override: Optional[str],
) -> Path:
    base.log_progress(f"build single report start experiment={experiment_root.name}")
    env_setup = base.load_json(experiment_root / "env_setup.json") if (experiment_root / "env_setup.json").exists() else None
    mode_dirs = base.find_mode_dirs(experiment_root)
    if not mode_dirs:
        raise FileNotFoundError(f"no mode directories found in {experiment_root}")

    mode_data = [load_mode_data(mode_dir) for mode_dir in mode_dirs]
    mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
    summary_rows = summarize_modes(experiment_root, env_setup, mode_data, switch_log_dir_override)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    step_series_max = []
    bw_series = []
    for item in mode_data:
        rows = item["step_rows"]
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        step_series_max.append((f'{item["mode"]} step_ms_max', steps, [float(row["step_ms_max"]) for row in rows]))
        bw_series.append((item["mode"], steps, [float(row["collective_gbps_est"]) for row in rows]))

    p_step = base.save_line_plot(plots_dir / "step_timeline.png", f"{experiment_root.name} Step Timeline", "collective step", "latency (ms)", step_series_max)
    p_bw = base.save_line_plot(plots_dir / "throughput_timeline.png", f"{experiment_root.name} Throughput Estimate", "collective step", "Gbps", bw_series)
    p_step_summary = base.save_grouped_bar(
        plots_dir / "summary_latency.png",
        "Latency Summary by Mode",
        [row["mode"] for row in summary_rows],
        [("avg", [float(row["step_ms_avg"]) for row in summary_rows]), ("p95", [float(row["step_ms_p95"]) for row in summary_rows])],
        "ms",
    )
    p_bw_summary = base.save_grouped_bar(
        plots_dir / "summary_throughput.png",
        "Throughput Summary by Mode",
        [row["mode"] for row in summary_rows],
        [("avg", [float(row["collective_gbps_avg"]) for row in summary_rows]), ("p95", [float(row["collective_gbps_p95"]) for row in summary_rows])],
        "Gbps",
    )
    p_window = save_window_trace_by_mode_plot(plots_dir / "step_window_trace.png", mode_data, f"{experiment_root.name} Step-Axis Window Trace")
    p_pfc = save_switch_pfc_counts_plot(plots_dir / "switch_pfc_counts.png", summary_rows, "Switch PFC Metrics by Mode")
    p_occ = base.save_grouped_bar(
        plots_dir / "occupancy_summary.png",
        "Outstanding Depth Summary by Mode",
        [row["mode"] for row in summary_rows],
        [("p99_occ_pd", [float(row["p99_occ_pd"]) for row in summary_rows]), ("p99_occ_tr", [float(row["p99_occ_tr"]) for row in summary_rows])],
        "depth",
    )
    p_events = base.save_event_counts_plot(plots_dir / "event_counts.png", mode_data, top_events)
    p_wstall = base.save_worker_wstall_plot(plots_dir / "wstall_by_worker.png", mode_data)
    p_dist = base.save_window_distribution_plot(plots_dir / "window_distribution.png", mode_data)
    p_switch = base.save_switch_overlay_plot(plots_dir / "switch_overlay.png", experiment_root, env_setup, mode_data, switch_log_dir_override)
    p_switch_phase = base.save_switch_phase_pfc_plot(plots_dir / "switch_pfc_phase_overlay.png", experiment_root, env_setup, mode_data, switch_log_dir_override)

    worker_trace_paths = []
    for item in mode_data:
        trace_path = save_worker_window_trace_plot(plots_dir / f"worker_window_trace_{item['mode'].lower()}.png", item)
        if trace_path is not None:
            worker_trace_paths.append((trace_path, f"{item['mode']} worker window trace"))

    info_rows = [["experiment_root", experiment_root.as_posix()], ["modes", ", ".join(item["mode"] for item in mode_data)]]
    if env_setup:
        for key in ["experiment_label", "collective", "run_modes", "payload_mb", "dtype", "nccl_algo", "nccl_proto", "policy_name", "policy_formula", "b3_disabled_by_runner", "switch_log_run_id", "switch_log_local_dir"]:
            if key in env_setup:
                info_rows.append([key, env_setup[key]])
    if switch_log_dir_override:
        info_rows.append(["switch_log_override", switch_log_dir_override])

    summary_headers = [
        "mode", "static_w_cfg", "workload", "collective", "algo", "payload_mb", "step_ms_avg", "step_ms_p95",
        "collective_gbps_avg", "switch_pfc_total", "switch_pfc_peak_rate", "p99_occ_tr", "w_eff_values",
        "final_output_match_vs_stock", "final_output_match_workers",
        "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "delta_pfc_total_vs_stock_pct",
    ]
    summary_table = [
        [
            row["mode"],
            row["static_w_cfg"],
            row["workload"],
            row["collective"],
            row["algo"],
            f'{row["payload_mb"]:.3f}',
            f'{row["step_ms_avg"]:.3f}',
            f'{row["step_ms_p95"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            f'{row["switch_pfc_total"]:.3f}',
            f'{row["switch_pfc_peak_rate"]:.3f}',
            f'{row["p99_occ_tr"]:.3f}',
            row["w_eff_values"],
            row["final_output_match_vs_stock"],
            row["final_output_match_workers"],
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
            f'{row["delta_pfc_total_vs_stock_pct"]:.2f}',
        ]
        for row in summary_rows
    ]

    collection_rows = [
        ["env_setup.json", "experiment metadata, static W sweep setup, B3 disabled flag"],
        ["MODE/*_summary.json", "mode absolute latency / throughput summary"],
        ["MODE/*_step_metrics.jsonl", "per-step collective metrics"],
        ["MODE/workerXX/*_worker_step_timing.jsonl", "worker-local timing for step-axis W alignment"],
        ["MODE/workerXX/*_rank_validation.json", "worker별 final output digest. STOCK 대비 semantic match 표시에 사용"],
        ["MODE/workerXX/nccl.*.log", "PHASE0 logs for W_eff / occupancy / WSTALL"],
        ["switch_log/*.jsonl", "switch PFC / ROCE logs"],
    ]
    visualization_rows = [
        ["step_timeline.png", "mode별 collective step latency timeline"],
        ["throughput_timeline.png", "mode별 throughput timeline"],
        ["step_window_trace.png", "mode별 평균 receiver W step trace"],
        ["switch_pfc_counts.png", "mode별 PFC total / peak 비교"],
        ["switch_overlay.png", "switch pressure and step latency overlay"],
    ]

    sections = [
        '<section class="section"><h2>Experiment Metadata</h2>' + base.render_table(["field", "value"], info_rows) + "</section>",
        '<section class="section"><div class="grid-2"><div class="card"><h2>Collected Logs</h2>' + base.render_table(["source", "meaning"], collection_rows) + '</div><div class="card"><h2>Visualization Plan</h2>' + base.render_table(["plot", "meaning"], visualization_rows) + "</div></div></section>",
        '<section class="section"><h2>Mode Summary</h2>' + base.render_table(summary_headers, summary_table) + "</section>",
    ]

    for item in mode_data:
        sections.append(f'<section class="section"><h2>{html.escape(item["mode"])} Receiver Window by Worker</h2>{build_worker_window_table(item)}</section>')

    figure_fragments = []
    for path, caption in [
        (p_step, "mode별 collective step latency timeline"),
        (p_bw, "mode별 collective throughput estimate"),
        (p_step_summary, "mode별 absolute latency"),
        (p_bw_summary, "mode별 absolute throughput"),
        (p_window, "mode별 step-axis window trace"),
        (p_pfc, "mode별 PFC total / peak"),
        (p_occ, "mode별 outstanding depth"),
        (p_events, "mode별 top NCCL events"),
        (p_wstall, "worker별 WSTALL count"),
        (p_dist, "receiver W distribution"),
        (p_switch, "switch pressure overlay"),
        (p_switch_phase, "phase-aligned PFC overlay"),
    ] + worker_trace_paths:
        if path is None or not path.exists():
            continue
        figure_fragments.append(f'<figure><img src="{html.escape(base.relpath(path, output_dir))}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figure_fragments) + "</section>")

    html_path = output_dir / "phase2_report.html"
    html_path.write_text(render_html(f"Phase2 Report - {experiment_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase2_report.json"
    report_json.write_text(
        json.dumps(
            {
                "experiment": experiment_root.name,
                "env_setup": env_setup,
                "mode_summary": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_csv = output_dir / "phase2_report.csv"
    if summary_rows:
        with report_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    base.log_progress(f"build single report complete experiment={experiment_root.name} html={html_path.as_posix()}")
    return html_path


def save_stock_vs_b2_grouped_plot(path: Path, title: str, experiment_names: List[str], matrix_rows: List[dict], metric: str, ylabel: str) -> Optional[Path]:
    mode_names = sorted({row["mode"] for row in matrix_rows}, key=mode_sort_key)
    series = []
    for mode_name in mode_names:
        values = []
        for experiment_name in experiment_names:
            row = next((item for item in matrix_rows if item["experiment"] == experiment_name and item["mode"] == mode_name), None)
            values.append(float(row[metric]) if row else 0.0)
        series.append((mode_name, values))
    return base.save_grouped_bar(path, title, experiment_names, series, ylabel, rotate_labels=True)


def save_b2_only_heatmap(path: Path, title: str, matrix_rows: List[dict], metric: str, colorbar_label: str) -> Optional[Path]:
    b2_rows = [row for row in matrix_rows if str(row["mode"]).startswith("B2_W")]
    if not b2_rows:
        return None
    experiment_names = sorted({row["experiment"] for row in b2_rows})
    w_values = sorted({int(row["static_w_cfg"]) for row in b2_rows})
    grid: List[List[float]] = []
    for experiment_name in experiment_names:
        row_values = []
        for w_value in w_values:
            row = next((item for item in b2_rows if item["experiment"] == experiment_name and int(item["static_w_cfg"]) == w_value), None)
            row_values.append(float(row[metric]) if row else 0.0)
        grid.append(row_values)
    return save_heatmap(path, title, experiment_names, [str(v) for v in w_values], grid, colorbar_label)


def build_matrix_report(matrix_root: Path, output_dir: Path, top_events: int, switch_log_dir_override: Optional[str]) -> Path:
    experiment_dirs = base.find_experiment_dirs(matrix_root)
    if not experiment_dirs:
        raise FileNotFoundError(f"no experiment directories found in {matrix_root}")
    base.log_progress(f"build matrix report start matrix_root={matrix_root.as_posix()} experiments={len(experiment_dirs)}")

    per_experiment_links: Dict[str, str] = {}
    matrix_rows: List[dict] = []
    experiment_payloads: List[Tuple[str, List[dict]]] = []

    for experiment_dir in experiment_dirs:
        per_output = output_dir / experiment_dir.name
        per_output.mkdir(parents=True, exist_ok=True)
        per_html = build_single_experiment_report(experiment_dir, per_output, top_events, switch_log_dir_override)
        env_setup = base.load_json(experiment_dir / "env_setup.json") if (experiment_dir / "env_setup.json").exists() else None
        experiment_label = experiment_label_from_env(experiment_dir, env_setup)
        per_experiment_links[experiment_label] = base.relpath(per_html, output_dir)
        mode_data = [load_mode_data(mode_dir) for mode_dir in base.find_mode_dirs(experiment_dir)]
        mode_data.sort(key=lambda item: mode_sort_key(item["mode"]))
        summary_rows = summarize_modes(experiment_dir, env_setup, mode_data, switch_log_dir_override)
        matrix_rows.extend(summary_rows)
        experiment_payloads.append((experiment_label, mode_data))

    experiment_names = [label for label, _ in experiment_payloads]

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    p_latency = save_stock_vs_b2_grouped_plot(plots_dir / "stock_vs_b2_latency.png", "STOCK vs B2 Latency", experiment_names, matrix_rows, "step_ms_avg", "step_ms_avg")
    p_bw = save_stock_vs_b2_grouped_plot(plots_dir / "stock_vs_b2_throughput.png", "STOCK vs B2 Throughput", experiment_names, matrix_rows, "collective_gbps_avg", "collective_gbps_avg")
    p_pfc = save_stock_vs_b2_grouped_plot(plots_dir / "stock_vs_b2_switch_pfc_counts.png", "STOCK vs B2 PFC Counts", experiment_names, matrix_rows, "switch_pfc_total", "switch_pfc_total")
    p_trace = save_matrix_step_window_trace_plot(plots_dir / "stock_vs_b2_step_window_trace.png", experiment_payloads)

    p_b2_latency = save_b2_only_heatmap(plots_dir / "b2_matrix_latency.png", "B2 Matrix Latency", matrix_rows, "step_ms_avg", "step_ms_avg")
    p_b2_bw = save_b2_only_heatmap(plots_dir / "b2_matrix_throughput.png", "B2 Matrix Throughput", matrix_rows, "collective_gbps_avg", "collective_gbps_avg")
    p_b2_pfc_count = save_b2_only_heatmap(plots_dir / "b2_matrix_pfc_count.png", "B2 Matrix PFC Count", matrix_rows, "switch_pfc_total", "switch_pfc_total")
    p_b2_pfc_peak = save_b2_only_heatmap(plots_dir / "b2_matrix_pfc_severity.png", "B2 Matrix PFC Severity", matrix_rows, "switch_pfc_peak_rate", "switch_pfc_peak_rate")

    summary_headers = [
        "experiment", "mode", "static_w_cfg", "workload", "step_ms_avg", "collective_gbps_avg",
        "switch_pfc_total", "switch_pfc_peak_rate", "p99_occ_tr", "w_eff_values",
        "final_output_match_vs_stock", "final_output_match_workers",
        "delta_step_vs_stock_pct", "delta_bw_vs_stock_pct", "delta_pfc_total_vs_stock_pct", "report",
    ]
    summary_rows_html = []
    for row in matrix_rows:
        link = per_experiment_links[row["experiment"]]
        summary_rows_html.append([
            html.escape(str(row["experiment"])),
            html.escape(str(row["mode"])),
            str(row["static_w_cfg"]),
            html.escape(str(row["workload"])),
            f'{row["step_ms_avg"]:.3f}',
            f'{row["collective_gbps_avg"]:.3f}',
            f'{row["switch_pfc_total"]:.3f}',
            f'{row["switch_pfc_peak_rate"]:.3f}',
            f'{row["p99_occ_tr"]:.3f}',
            html.escape(str(row["w_eff_values"])),
            html.escape(str(row["final_output_match_vs_stock"])),
            html.escape(str(row["final_output_match_workers"])),
            f'{row["delta_step_vs_stock_pct"]:.2f}',
            f'{row["delta_bw_vs_stock_pct"]:.2f}',
            f'{row["delta_pfc_total_vs_stock_pct"]:.2f}',
            f'<a href="{html.escape(link)}">open</a>',
        ])

    collection_card = build_collection_plan_card_matrix(experiment_dirs)
    visualization_card = build_visualization_plan_card_matrix()
    sections = [
        '<section class="section"><div class="grid-2">' + collection_card + visualization_card + "</div></section>",
        '<section class="section"><h2>Matrix Summary</h2>' + base.render_table_raw(summary_headers, summary_rows_html) + "</section>",
    ]

    figure_fragments = []
    for path, caption in [
        (p_latency, "stock vs B2 latency"),
        (p_bw, "stock vs B2 throughput"),
        (p_pfc, "stock vs B2 PFC count"),
        (p_trace, "stock vs B2 step-axis W trace"),
        (p_b2_latency, "B2 latency heatmap"),
        (p_b2_bw, "B2 throughput heatmap"),
        (p_b2_pfc_count, "B2 PFC count heatmap"),
        (p_b2_pfc_peak, "B2 PFC severity heatmap"),
    ]:
        if path is None or not path.exists():
            continue
        figure_fragments.append(f'<figure><img src="{html.escape(base.relpath(path, output_dir))}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    sections.append('<section class="section"><h2>Plots</h2>' + "".join(figure_fragments) + "</section>")

    report_list = "".join(f'<li><a href="{html.escape(link)}">{html.escape(name)}</a></li>' for name, link in sorted(per_experiment_links.items()))
    sections.append('<section class="section"><h2>Per-experiment Reports</h2><ul>' + report_list + "</ul></section>")

    html_path = output_dir / "phase2_report.html"
    html_path.write_text(render_html(f"Phase2 Matrix Report - {matrix_root.name}", sections), encoding="utf-8")

    report_json = output_dir / "phase2_report.json"
    report_json.write_text(json.dumps({"matrix_root": matrix_root.name, "rows": matrix_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    report_csv = output_dir / "phase2_report.csv"
    if matrix_rows:
        with report_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matrix_rows)

    base.log_progress(f"build matrix report complete html={html_path.as_posix()}")
    return html_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base.REPORT_LOG_PATH = output_dir / "phase2_reporter.log"
    base.REPORT_LOG_PATH.write_text("", encoding="utf-8")
    base.log_progress(f"start input={input_path.as_posix()} output={output_dir.as_posix()}")

    if base.is_matrix_root(input_path):
        html_path = build_matrix_report(input_path, output_dir, args.top_events, args.switch_log)
    else:
        html_path = build_single_experiment_report(input_path, output_dir, args.top_events, args.switch_log)

    base.log_progress(f"wrote html={html_path.as_posix()}")


if __name__ == "__main__":
    main()
