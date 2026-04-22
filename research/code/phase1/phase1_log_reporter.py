#!/usr/bin/env python3
import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for phase1_log_reporter.py. Install it first, for example: pip install matplotlib"
    ) from exc


PHASE_RE = re.compile(r"PHASE[01]\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
WORKER_RE = re.compile(r"worker(\d+)$")
W_DIR_RE = re.compile(r"^W(\d+)$")

RUN_SUMMARY_FIELD_HELP = {
    "workload": "리포터가 자동 판별한 workload 종류. Ring AllReduce 또는 AllToAll.",
    "run_tag": "실험 식별자. 각 W run을 구분하는 tag.",
    "phase1_w": "NCCL_PHASE1_STATIC_W로 설정한 고정 window 크기.",
    "world_size": "분산 rank 총 개수.",
    "steps": "전체 step 수.",
    "effective_steps": "warmup을 제외하고 요약에 사용한 step 수.",
    "algo": "실험에서 강제한 NCCL 알고리즘. 비어 있으면 NCCL 자동 선택.",
    "proto": "실험에서 강제한 NCCL 프로토콜. 비어 있으면 NCCL 자동 선택.",
    "param_mb": "Ring AllReduce DDP workload에서 모델 파라미터 총 크기(MiB).",
    "payload_mb": "AllToAll workload에서 rank당 총 send payload 크기(MiB).",
    "dtype": "통신 tensor dtype.",
    "bucket_cap_mb": "DDP gradient bucket cap size(MiB).",
    "step_ms_avg": "rank max 기준 step latency 평균(ms).",
    "step_ms_p50": "rank max 기준 step latency p50(ms).",
    "step_ms_p95": "rank max 기준 step latency p95(ms).",
    "backward_ms_avg": "rank max 기준 backward 구간 평균(ms). Ring DDP에선 gradient all-reduce 지배 구간.",
    "backward_ms_p95": "rank max 기준 backward 구간 p95(ms).",
    "ring_gbps_avg": "Ring AllReduce 예상 traffic volume을 backward time으로 나눈 근사 처리량(Gbps).",
    "ring_gbps_p95": "step별 ring_gbps_est의 p95(Gbps).",
    "alltoall_gbps_avg": "AllToAll 예상 aggregate volume을 step time으로 나눈 근사 처리량(Gbps).",
    "alltoall_gbps_p95": "step별 alltoall_gbps_est의 p95(Gbps).",
}

NCCL_SUMMARY_FIELD_HELP = {
    "recv_wstall_count": "PROXY_RECV_WSTALL 총 개수. receiver가 window 상한 때문에 추가 recv post를 못 한 횟수.",
    "send_wstall_count": "PROXY_SEND_WSTALL 총 개수. sender가 window 상한 때문에 추가 send-side post를 못 한 횟수.",
    "max_occ_pd": "전체 로그에서 관측된 최대 posted-done.",
    "p99_occ_pd": "posted-done의 p99. proxy가 얼마나 앞질러 가는지.",
    "max_occ_tr": "전체 로그에서 관측된 최대 transmitted-done.",
    "p99_occ_tr": "transmitted-done의 p99. 실제 visible/send outstanding 깊이.",
    "w_eff_values": "로그에서 관측된 effective window 값 집합.",
}

PLOT_EXPLANATIONS_SINGLE: Dict[str, List[Tuple[str, str, str, str]]] = {
    "ring": [
        (
            "step_timeline.png",
            "Step / Backward / Forward / Optimizer Timeline",
            "입력 데이터: rank 0가 기록한 *_step_metrics.jsonl",
            "step_ms_max, backward_ms_max, forward_ms_max, optimizer_ms_max를 step 축으로 그린다. "
            "작은 W가 전체 step을 늘리는지, 병목이 backward인지 즉시 볼 수 있다.",
        ),
        (
            "throughput_timeline.png",
            "Per-step Ring Throughput Estimate",
            "입력 데이터: *_step_metrics.jsonl",
            "ring_gbps_est를 step별로 그린다. W에 따라 ring pipeline이 얼마나 채워지는지 확인한다.",
        ),
    ],
    "alltoall": [
        (
            "step_timeline.png",
            "Step Latency Timeline",
            "입력 데이터: rank 0가 기록한 *_step_metrics.jsonl",
            "step_ms_max와 step_ms_mean을 step 축으로 그린다. W 변화가 all-to-all 전체 step latency에 어떤 영향을 주는지 본다.",
        ),
        (
            "throughput_timeline.png",
            "Per-step AllToAll Throughput Estimate",
            "입력 데이터: *_step_metrics.jsonl",
            "alltoall_gbps_est를 step별로 그린다. aggregate many-to-many traffic이 W에 따라 얼마나 달라지는지 확인한다.",
        ),
    ],
}

COMMON_SINGLE_PLOT_EXPLANATIONS = [
    (
        "event_counts.png",
        "Top NCCL Event Counts",
        "입력 데이터: workerXX/nccl.*.log 의 PHASE0/PHASE1 event",
        "이벤트 총량 분포를 본다. fake cumulative에선 총 CTS 수보다 WSTALL과 proxy event 분포 차이가 더 중요하다.",
    ),
    (
        "wstall_by_worker.png",
        "WSTALL by Worker",
        "입력 데이터: PROXY_RECV_WSTALL, PROXY_SEND_WSTALL 이벤트",
        "worker별 stall 편차를 본다. 특정 worker/rack만 과도하게 막히는지 확인할 수 있다.",
    ),
    (
        "occupancy_timeline.png",
        "Outstanding Depth Timeline by Worker",
        "입력 데이터: PHASE0/PHASE1 로그의 occPd, occTr",
        "worker별 상대시간 기준으로 posted-done과 transmitted-done을 그린다. "
        "여러 worker의 절대 시간을 섞지 않고, 각 worker 내부에서 W가 실제 outstanding depth를 어떻게 제한하는지 보여준다.",
    ),
]

PLOT_EXPLANATIONS_SWEEP: Dict[str, List[Tuple[str, str, str, str]]] = {
    "ring": [
        (
            "sweep_step_ms.png",
            "W Sweep - Step Latency",
            "입력 데이터: 각 W의 *_summary.json",
            "step_ms_avg, step_ms_p95, backward_ms_p95를 비교한다. 작은 W가 성능을 얼마나 깎는지 보여준다.",
        ),
        (
            "sweep_throughput.png",
            "W Sweep - Ring Throughput Estimate",
            "입력 데이터: 각 W의 *_summary.json",
            "ring_gbps_avg, ring_gbps_p95를 비교한다. pipeline이 충분히 찬 W 이후 포화 구간이 있는지 보여준다.",
        ),
    ],
    "alltoall": [
        (
            "sweep_step_ms.png",
            "W Sweep - Step Latency",
            "입력 데이터: 각 W의 *_summary.json",
            "step_ms_avg, step_ms_p50, step_ms_p95를 비교한다. W가 all-to-all 전체 지연에 미치는 영향을 보여준다.",
        ),
        (
            "sweep_throughput.png",
            "W Sweep - AllToAll Throughput Estimate",
            "입력 데이터: 각 W의 *_summary.json",
            "alltoall_gbps_avg, alltoall_gbps_p95를 비교한다. W가 aggregate many-to-many throughput에 미치는 영향을 보여준다.",
        ),
    ],
}

COMMON_SWEEP_PLOT_EXPLANATIONS = [
    (
        "sweep_wstall.png",
        "W Sweep - Window Stall Counts",
        "입력 데이터: PHASE0/PHASE1 로그 집계",
        "recv/send WSTALL 총량을 비교한다. W를 줄였을 때 실제로 admission이 더 자주 걸렸는지 확인한다.",
    ),
    (
        "sweep_occupancy.png",
        "W Sweep - Outstanding Depth",
        "입력 데이터: PHASE0/PHASE1 로그의 occPd, occTr 집계",
        "p99/max occupancy를 비교한다. W가 outstanding depth를 어느 정도 줄이는지 확인한다.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 1 PNG plots and HTML report")
    parser.add_argument(
        "--input",
        required=True,
        help="Either a single W directory (e.g. .../W1) or a run root containing W1/W2/W4/W8",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <input>/report",
    )
    parser.add_argument(
        "--occupancy-workers",
        default=None,
        help="Comma-separated worker names to include in occupancy timeline plots, e.g. worker01,worker04",
    )
    parser.add_argument(
        "--occupancy-ranks",
        default=None,
        help="Comma-separated global ranks to include in occupancy timeline plots, e.g. 0,3. "
             "When both worker and rank filters are given, the union of matches is plotted.",
    )
    return parser.parse_args()


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


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def parse_csv_set(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_csv_int_set(raw: Optional[str]) -> Set[int]:
    values: Set[int] = set()
    for item in parse_csv_set(raw):
        values.add(int(item))
    return values


def detect_workload(summary: dict, step_rows: List[dict]) -> str:
    if "alltoall_gbps_avg" in summary:
        return "alltoall"
    if "ring_gbps_avg" in summary or "backward_ms_avg" in summary:
        return "ring"
    if step_rows:
        first = step_rows[0]
        if "alltoall_gbps_est" in first:
            return "alltoall"
        if "ring_gbps_est" in first or "backward_ms_max" in first:
            return "ring"
    raise ValueError("could not detect workload type from summary/step metrics")


def workload_label(workload: str) -> str:
    return {"ring": "Ring AllReduce", "alltoall": "AllToAll"}[workload]


def positive_ylim(max_value: float) -> Tuple[float, float] | None:
    if max_value <= 0:
        return None
    return 0.0, max_value * 1.05


def worker_sort_key(name: str) -> Tuple[int, str]:
    m = WORKER_RE.match(name)
    if m:
        return int(m.group(1)), name
    return (10**9, name)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_nccl_events(root: Path) -> List[dict]:
    events: List[dict] = []
    for path in sorted(root.rglob("*.log")):
        worker = path.parent.name
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if "event=" not in line or ("PHASE0" not in line and "PHASE1" not in line):
                    continue
                m = PHASE_RE.search(line)
                if not m:
                    continue
                kvs = {k: parse_value(v) for k, v in KV_RE.findall(m.group(1))}
                event = kvs.get("event")
                if not event:
                    continue
                record = {
                    "worker": worker,
                    "line_no": lineno,
                    "event": str(event),
                }
                record.update(kvs)
                events.append(record)
    return events


def summarize_nccl_events(events: List[dict]) -> dict:
    event_counts: Counter = Counter()
    worker_event_counts: Dict[str, Counter] = defaultdict(Counter)
    occ_pd: List[float] = []
    occ_tr: List[float] = []
    worker_recv_wstall: Counter = Counter()
    worker_send_wstall: Counter = Counter()
    t_points_pd: List[Tuple[float, float]] = []
    t_points_tr: List[Tuple[float, float]] = []
    worker_t_points_pd: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    worker_t_points_tr: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    worker_ranks: Dict[str, Set[int]] = defaultdict(set)
    w_eff_values: Counter = Counter()

    for event in events:
        evt = event["event"]
        worker = event["worker"]
        if "rank" in event:
            try:
                worker_ranks[worker].add(int(event["rank"]))
            except (TypeError, ValueError):
                pass
        event_counts[evt] += 1
        worker_event_counts[worker][evt] += 1
        if evt == "PROXY_RECV_WSTALL":
            worker_recv_wstall[worker] += 1
        if evt == "PROXY_SEND_WSTALL":
            worker_send_wstall[worker] += 1
        if "wEff" in event:
            w_eff_values[int(event["wEff"])] += 1
        if "occPd" in event:
            value = float(event["occPd"])
            occ_pd.append(value)
            if "tNs" in event:
                point = (float(event["tNs"]), value)
                t_points_pd.append(point)
                worker_t_points_pd[worker].append(point)
        elif "posted" in event and "done" in event:
            value = float(event["posted"]) - float(event["done"])
            occ_pd.append(value)
        if "occTr" in event:
            value = float(event["occTr"])
            occ_tr.append(value)
            if "tNs" in event:
                point = (float(event["tNs"]), value)
                t_points_tr.append(point)
                worker_t_points_tr[worker].append(point)
        elif "transmitted" in event and "done" in event:
            value = float(event["transmitted"]) - float(event["done"])
            occ_tr.append(value)

    return {
        "event_counts": event_counts,
        "worker_event_counts": worker_event_counts,
        "worker_recv_wstall": worker_recv_wstall,
        "worker_send_wstall": worker_send_wstall,
        "occ_pd": occ_pd,
        "occ_tr": occ_tr,
        "t_points_pd": t_points_pd,
        "t_points_tr": t_points_tr,
        "worker_t_points_pd": dict(worker_t_points_pd),
        "worker_t_points_tr": dict(worker_t_points_tr),
        "worker_ranks": {worker: sorted(ranks) for worker, ranks in worker_ranks.items()},
        "w_eff_values": w_eff_values,
        "recv_wstall_count": int(event_counts.get("PROXY_RECV_WSTALL", 0)),
        "send_wstall_count": int(event_counts.get("PROXY_SEND_WSTALL", 0)),
        "max_occ_pd": max(occ_pd) if occ_pd else 0.0,
        "p99_occ_pd": percentile(occ_pd, 0.99),
        "max_occ_tr": max(occ_tr) if occ_tr else 0.0,
        "p99_occ_tr": percentile(occ_tr, 0.99),
    }


def find_summary_and_steps(w_dir: Path) -> Tuple[Path, Path]:
    summary_candidates = sorted(w_dir.glob("*_summary.json"))
    steps_candidates = sorted(w_dir.glob("*_step_metrics.jsonl"))
    if not summary_candidates:
        raise FileNotFoundError(f"summary json not found under {w_dir}")
    if not steps_candidates:
        raise FileNotFoundError(f"step metrics jsonl not found under {w_dir}")
    return summary_candidates[0], steps_candidates[0]


def is_run_root(path: Path) -> bool:
    return any(child.is_dir() and W_DIR_RE.match(child.name) for child in path.iterdir())


def discover_w_dirs(path: Path) -> List[Path]:
    if is_run_root(path):
        return sorted(
            [child for child in path.iterdir() if child.is_dir() and W_DIR_RE.match(child.name)],
            key=lambda p: int(W_DIR_RE.match(p.name).group(1)),
        )
    if W_DIR_RE.match(path.name):
        return [path]
    raise ValueError(f"input path must be a W directory or a run root with W* subdirectories: {path}")


def save_line_plot(
    path: Path,
    title: str,
    xs: List[float],
    series: List[Tuple[str, List[float], str]],
    xlabel: str,
    ylabel: str,
) -> None:
    if not xs or not any(values for _, values, _ in series):
        return
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    max_value = 0.0
    for label, values, color in series:
        if values:
            ax.plot(xs[:len(values)], values, label=label, linewidth=2.0, color=color)
            max_value = max(max_value, max(values))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ylim = positive_ylim(max_value)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_bar_chart(path: Path, title: str, labels: List[str], values: List[float], ylabel: str) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    bars = ax.bar(labels, values, color="#0c5fb0", alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ylim = positive_ylim(max(values))
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}" if isinstance(value, float) else f"{value}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_horizontal_bar(path: Path, title: str, labels: List[str], values: List[float], xlabel: str) -> None:
    if not labels:
        return
    fig_h = max(4.0, 0.45 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))
    bars = ax.barh(labels, values, color="#136245", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if values:
        ax.set_xlim(0.0, max(values) * 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.0, f" {value}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_worker_occupancy_plot(
    path: Path,
    title: str,
    worker_t_points_pd: Dict[str, List[Tuple[float, float]]],
    worker_t_points_tr: Dict[str, List[Tuple[float, float]]],
) -> None:
    workers = sorted(
        set(worker_t_points_pd.keys()) | set(worker_t_points_tr.keys()),
        key=worker_sort_key,
    )
    if not workers:
        return

    ncols = 2 if len(workers) > 1 else 1
    nrows = (len(workers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, max(4.2, 3.2 * nrows)), squeeze=False)
    max_occ = 0.0

    for idx, worker in enumerate(workers):
        ax = axes[idx // ncols][idx % ncols]
        pd_points = worker_t_points_pd.get(worker, [])
        tr_points = worker_t_points_tr.get(worker, [])
        t0_candidates = [t for t, _ in pd_points] + [t for t, _ in tr_points]
        if not t0_candidates:
            ax.set_visible(False)
            continue
        t0 = min(t0_candidates)
        if pd_points:
            xs_pd = [(t - t0) / 1e6 for t, _ in pd_points]
            ys_pd = [v for _, v in pd_points]
            ax.plot(xs_pd, ys_pd, ".", markersize=2, alpha=0.6, label="occPd", color="#0c5fb0")
            max_occ = max(max_occ, max(ys_pd))
        if tr_points:
            xs_tr = [(t - t0) / 1e6 for t, _ in tr_points]
            ys_tr = [v for _, v in tr_points]
            ax.plot(xs_tr, ys_tr, ".", markersize=2, alpha=0.6, label="occTr", color="#d97706")
            max_occ = max(max_occ, max(ys_tr))
        ax.set_title(worker)
        ax.set_xlabel("time since first worker event (ms)")
        ax.set_ylabel("outstanding depth")
        ax.grid(True, linestyle="--", alpha=0.25)
        if pd_points or tr_points:
            ax.legend(fontsize=8)

    for idx in range(len(workers), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    ylim = positive_ylim(max_occ)
    if ylim:
        for row in axes:
            for ax in row:
                if ax.get_visible():
                    ax.set_ylim(*ylim)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def select_occupancy_workers(
    worker_t_points_pd: Dict[str, List[Tuple[float, float]]],
    worker_t_points_tr: Dict[str, List[Tuple[float, float]]],
    worker_ranks: Dict[str, List[int]],
    occupancy_workers: Set[str],
    occupancy_ranks: Set[int],
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float]]], str]:
    if not occupancy_workers and not occupancy_ranks:
        return worker_t_points_pd, worker_t_points_tr, "all workers"

    selected_workers: Set[str] = set(occupancy_workers)
    if occupancy_ranks:
        for worker, ranks in worker_ranks.items():
            if any(rank in occupancy_ranks for rank in ranks):
                selected_workers.add(worker)

    filtered_pd = {worker: points for worker, points in worker_t_points_pd.items() if worker in selected_workers}
    filtered_tr = {worker: points for worker, points in worker_t_points_tr.items() if worker in selected_workers}

    labels: List[str] = []
    if occupancy_workers:
        labels.append("workers=" + ",".join(sorted(occupancy_workers, key=worker_sort_key)))
    if occupancy_ranks:
        labels.append("ranks=" + ",".join(str(rank) for rank in sorted(occupancy_ranks)))
    if not selected_workers:
        labels.append("no matches")
    return filtered_pd, filtered_tr, "; ".join(labels)


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def render_table(headers: List[str], rows: List[List[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    return (
        '<table class="report-table"><thead><tr>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def render_field_help(title: str, mapping: Dict[str, str], used_fields: List[str]) -> str:
    rows = [[field, mapping.get(field, "")] for field in used_fields]
    return '<div class="card"><h2>' + html.escape(title) + '</h2>' + render_table(["field", "meaning"], rows) + '</div>'


def render_plot_help(title: str, items: List[Tuple[str, str, str, str]]) -> str:
    rows = [[filename, plot_title, source, meaning] for filename, plot_title, source, meaning in items]
    return '<div class="card"><h2>' + html.escape(title) + '</h2>' + render_table(["plot", "title", "source", "meaning"], rows) + '</div>'


def render_html(title: str, sections: List[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --ink: #1b1f23;
      --muted: #5f6b76;
      --line: #d8d2c4;
      --accent: #0d5c63;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
      background: linear-gradient(180deg, #f6f4ef 0%, #efe9dc 100%);
      color: var(--ink);
      line-height: 1.6;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{ color: var(--accent); margin: 0 0 8px; }}
    h2 {{ margin-top: 32px; border-bottom: 2px solid var(--line); padding-bottom: 8px; }}
    .lead {{ color: var(--muted); margin: 0 0 18px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
      margin: 16px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.04);
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      background: #fff;
    }}
    .report-table th, .report-table td {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 0.95rem;
    }}
    .report-table th {{
      background: #e6f2f2;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    figure img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }}
    figure figcaption {{
      margin-top: 8px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    code {{
      background: #f0ece2;
      padding: 2px 6px;
      border-radius: 6px;
      font-family: "Cascadia Code", "Consolas", monospace;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <p class="lead">matplotlib 기반 Phase 1 B1 실험 리포트</p>
    {''.join(sections)}
  </div>
</body>
</html>"""


def build_single_w_report(
    w_dir: Path,
    output_dir: Path,
    occupancy_workers: Optional[Set[str]] = None,
    occupancy_ranks: Optional[Set[int]] = None,
) -> Path:
    summary_path, steps_path = find_summary_and_steps(w_dir)
    summary = load_json(summary_path)
    step_rows = load_jsonl(steps_path)
    workload = detect_workload(summary, step_rows)
    events = parse_nccl_events(w_dir)
    nccl = summarize_nccl_events(events)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = [int(row["step"]) for row in step_rows]
    step_ms = [float(row["step_ms_max"]) for row in step_rows]
    step_series: List[Tuple[str, List[float], str]]
    throughput_series: List[Tuple[str, List[float], str]]
    throughput_caption: str
    throughput_ylabel = "Gbps"
    summary_rows: List[List[object]]

    if workload == "ring":
        backward_ms = [float(row["backward_ms_max"]) for row in step_rows]
        ring_gbps = [float(row["ring_gbps_est"]) for row in step_rows]
        forward_ms = [float(row["forward_ms_max"]) for row in step_rows]
        optimizer_ms = [float(row["optimizer_ms_max"]) for row in step_rows]
        step_series = [
            ("step_ms_max", step_ms, "#0c5fb0"),
            ("backward_ms_max", backward_ms, "#d97706"),
            ("forward_ms_max", forward_ms, "#2f855a"),
            ("optimizer_ms_max", optimizer_ms, "#7c3aed"),
        ]
        throughput_series = [("ring_gbps_est", ring_gbps, "#136245")]
        throughput_caption = "per-step ring throughput estimate"
        summary_rows = [
            ["workload", workload_label(workload)],
            ["run_tag", summary.get("run_tag", "")],
            ["phase1_w", summary.get("phase1_w", "")],
            ["world_size", summary.get("world_size", "")],
            ["steps", summary.get("steps", "")],
            ["effective_steps", summary.get("effective_steps", "")],
            ["algo", summary.get("algo", "")],
            ["proto", summary.get("proto", "")],
            ["param_mb", summary.get("param_mb", "")],
            ["dtype", summary.get("dtype", "")],
            ["bucket_cap_mb", summary.get("bucket_cap_mb", "")],
            ["step_ms_avg", f"{summary.get('step_ms_avg', 0.0):.4f}"],
            ["step_ms_p50", f"{summary.get('step_ms_p50', 0.0):.4f}"],
            ["step_ms_p95", f"{summary.get('step_ms_p95', 0.0):.4f}"],
            ["backward_ms_avg", f"{summary.get('backward_ms_avg', 0.0):.4f}"],
            ["backward_ms_p95", f"{summary.get('backward_ms_p95', 0.0):.4f}"],
            ["ring_gbps_avg", f"{summary.get('ring_gbps_avg', 0.0):.4f}"],
            ["ring_gbps_p95", f"{summary.get('ring_gbps_p95', 0.0):.4f}"],
        ]
    else:
        step_ms_mean = [float(row["step_ms_mean"]) for row in step_rows]
        alltoall_gbps = [float(row["alltoall_gbps_est"]) for row in step_rows]
        step_series = [
            ("step_ms_max", step_ms, "#0c5fb0"),
            ("step_ms_mean", step_ms_mean, "#d97706"),
        ]
        throughput_series = [("alltoall_gbps_est", alltoall_gbps, "#136245")]
        throughput_caption = "per-step alltoall throughput estimate"
        summary_rows = [
            ["workload", workload_label(workload)],
            ["run_tag", summary.get("run_tag", "")],
            ["phase1_w", summary.get("phase1_w", "")],
            ["world_size", summary.get("world_size", "")],
            ["steps", summary.get("steps", "")],
            ["effective_steps", summary.get("effective_steps", "")],
            ["algo", summary.get("algo", "")],
            ["proto", summary.get("proto", "")],
            ["payload_mb", summary.get("payload_mb", "")],
            ["dtype", summary.get("dtype", "")],
            ["step_ms_avg", f"{summary.get('step_ms_avg', 0.0):.4f}"],
            ["step_ms_p50", f"{summary.get('step_ms_p50', 0.0):.4f}"],
            ["step_ms_p95", f"{summary.get('step_ms_p95', 0.0):.4f}"],
            ["alltoall_gbps_avg", f"{summary.get('alltoall_gbps_avg', 0.0):.4f}"],
            ["alltoall_gbps_p95", f"{summary.get('alltoall_gbps_p95', 0.0):.4f}"],
        ]

    p_step = plots_dir / "step_timeline.png"
    save_line_plot(
        p_step,
        f"{w_dir.name} Step Timeline",
        steps,
        step_series,
        xlabel="step",
        ylabel="ms",
    )

    p_bw = plots_dir / "throughput_timeline.png"
    save_line_plot(
        p_bw,
        f"{w_dir.name} {workload_label(workload)} Throughput Estimate",
        steps,
        throughput_series,
        xlabel="step",
        ylabel=throughput_ylabel,
    )

    top_events = nccl["event_counts"].most_common(12)
    p_events = plots_dir / "event_counts.png"
    save_bar_chart(
        p_events,
        f"{w_dir.name} NCCL Event Counts",
        [name for name, _ in top_events],
        [count for _, count in top_events],
        ylabel="count",
    )

    workers = sorted(
        set(list(nccl["worker_recv_wstall"].keys()) + list(nccl["worker_send_wstall"].keys())),
        key=worker_sort_key,
    )
    p_wstall = plots_dir / "wstall_by_worker.png"
    if workers:
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        xs = list(range(len(workers)))
        recv_values = [nccl["worker_recv_wstall"].get(worker, 0) for worker in workers]
        send_values = [nccl["worker_send_wstall"].get(worker, 0) for worker in workers]
        ax.bar([x - 0.18 for x in xs], recv_values, width=0.36, label="PROXY_RECV_WSTALL", color="#d62728")
        ax.bar([x + 0.18 for x in xs], send_values, width=0.36, label="PROXY_SEND_WSTALL", color="#1f77b4")
        ax.set_xticks(xs, workers)
        ax.set_title(f"{w_dir.name} WSTALL by Worker")
        ax.set_ylabel("count")
        ylim = positive_ylim(max(recv_values + send_values) if (recv_values or send_values) else 0.0)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(p_wstall, dpi=150, bbox_inches="tight")
        plt.close(fig)

    p_occ = plots_dir / "occupancy_timeline.png"
    occupancy_filter_label = "all workers"
    filtered_pd = nccl["worker_t_points_pd"]
    filtered_tr = nccl["worker_t_points_tr"]
    if nccl["worker_t_points_pd"] or nccl["worker_t_points_tr"]:
        filtered_pd, filtered_tr, occupancy_filter_label = select_occupancy_workers(
            nccl["worker_t_points_pd"],
            nccl["worker_t_points_tr"],
            nccl["worker_ranks"],
            occupancy_workers or set(),
            occupancy_ranks or set(),
        )
    if filtered_pd or filtered_tr:
        save_worker_occupancy_plot(
            p_occ,
            f"{w_dir.name} Outstanding Depth Timeline by Worker ({occupancy_filter_label})",
            filtered_pd,
            filtered_tr,
        )

    nccl_rows = [
        ["recv_wstall_count", nccl["recv_wstall_count"]],
        ["send_wstall_count", nccl["send_wstall_count"]],
        ["max_occ_pd", f"{nccl['max_occ_pd']:.4f}"],
        ["p99_occ_pd", f"{nccl['p99_occ_pd']:.4f}"],
        ["max_occ_tr", f"{nccl['max_occ_tr']:.4f}"],
        ["p99_occ_tr", f"{nccl['p99_occ_tr']:.4f}"],
        ["w_eff_values", ", ".join(str(k) for k in sorted(nccl["w_eff_values"].keys()))],
    ]

    sections = [
        '<div class="card"><h2>Run Summary</h2>' + render_table(["field", "value"], summary_rows) + "</div>",
        render_field_help(
            "Run Summary Field Meanings",
            RUN_SUMMARY_FIELD_HELP,
            [field for field, _ in summary_rows],
        ),
        '<div class="card"><h2>NCCL Summary</h2>' + render_table(["field", "value"], nccl_rows) + "</div>",
        render_field_help(
            "NCCL Summary Field Meanings",
            NCCL_SUMMARY_FIELD_HELP,
            [field for field, _ in nccl_rows],
        ),
        '<div class="card"><h2>Plots</h2><div class="plot-grid">'
        + f'<figure><img src="{html.escape(relpath(p_step, output_dir))}" alt="step timeline"><figcaption>{html.escape(PLOT_EXPLANATIONS_SINGLE[workload][0][1])}</figcaption></figure>'
        + f'<figure><img src="{html.escape(relpath(p_bw, output_dir))}" alt="throughput"><figcaption>{html.escape(throughput_caption)}</figcaption></figure>'
        + f'<figure><img src="{html.escape(relpath(p_events, output_dir))}" alt="event counts"><figcaption>top NCCL event counts</figcaption></figure>'
        + (f'<figure><img src="{html.escape(relpath(p_wstall, output_dir))}" alt="wstall by worker"><figcaption>worker별 recv/send WSTALL count</figcaption></figure>' if p_wstall.exists() else "")
        + (f'<figure><img src="{html.escape(relpath(p_occ, output_dir))}" alt="occupancy timeline"><figcaption>occPd / occTr timeline ({html.escape(occupancy_filter_label)})</figcaption></figure>' if p_occ.exists() else "")
        + "</div></div>",
        render_plot_help("Plot Descriptions", PLOT_EXPLANATIONS_SINGLE[workload] + COMMON_SINGLE_PLOT_EXPLANATIONS),
    ]

    html_path = output_dir / "phase1_report.html"
    html_path.write_text(render_html(f"Phase1 Report - {w_dir.name} ({workload_label(workload)})", sections), encoding="utf-8")

    json_path = output_dir / "phase1_report.json"
    json_path.write_text(
        json.dumps({"workload": workload, "summary": summary, "nccl": {
            "event_counts": dict(nccl["event_counts"]),
            "recv_wstall_count": nccl["recv_wstall_count"],
            "send_wstall_count": nccl["send_wstall_count"],
            "max_occ_pd": nccl["max_occ_pd"],
            "p99_occ_pd": nccl["p99_occ_pd"],
            "max_occ_tr": nccl["max_occ_tr"],
            "p99_occ_tr": nccl["p99_occ_tr"],
            "w_eff_values": dict(nccl["w_eff_values"]),
        }}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return html_path


def build_run_root_report(
    run_root: Path,
    output_dir: Path,
    occupancy_workers: Optional[Set[str]] = None,
    occupancy_ranks: Optional[Set[int]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    per_w_details: Dict[str, dict] = {}
    per_w_links: Dict[str, str] = {}
    workload: str | None = None
    for w_dir in discover_w_dirs(run_root):
        per_w_output_dir = output_dir / w_dir.name
        per_w_html = build_single_w_report(
            w_dir,
            per_w_output_dir,
            occupancy_workers=occupancy_workers,
            occupancy_ranks=occupancy_ranks,
        )
        per_w_links[w_dir.name] = relpath(per_w_html, output_dir)

        summary_path, steps_path = find_summary_and_steps(w_dir)
        summary = load_json(summary_path)
        step_rows = load_jsonl(steps_path)
        this_workload = detect_workload(summary, step_rows)
        if workload is None:
            workload = this_workload
        elif workload != this_workload:
            raise ValueError(f"mixed workloads under one run root are not supported: {workload} vs {this_workload}")
        events = parse_nccl_events(w_dir)
        nccl = summarize_nccl_events(events)
        w_match = W_DIR_RE.match(w_dir.name)
        w = int(w_match.group(1))
        row = {
            "workload": this_workload,
            "w": w,
            "step_ms_avg": float(summary.get("step_ms_avg", 0.0)),
            "step_ms_p50": float(summary.get("step_ms_p50", 0.0)),
            "step_ms_p95": float(summary.get("step_ms_p95", 0.0)),
            "recv_wstall_count": nccl["recv_wstall_count"],
            "send_wstall_count": nccl["send_wstall_count"],
            "max_occ_pd": float(nccl["max_occ_pd"]),
            "p99_occ_pd": float(nccl["p99_occ_pd"]),
            "max_occ_tr": float(nccl["max_occ_tr"]),
            "p99_occ_tr": float(nccl["p99_occ_tr"]),
            "effective_steps": int(summary.get("effective_steps", 0)),
            "dtype": str(summary.get("dtype", "")),
            "algo": str(summary.get("algo", "")),
            "proto": str(summary.get("proto", "")),
        }
        if this_workload == "ring":
            row.update({
                "param_mb": int(summary.get("param_mb", 0)),
                "backward_ms_avg": float(summary.get("backward_ms_avg", 0.0)),
                "backward_ms_p95": float(summary.get("backward_ms_p95", 0.0)),
                "throughput_avg": float(summary.get("ring_gbps_avg", 0.0)),
                "throughput_p95": float(summary.get("ring_gbps_p95", 0.0)),
            })
        else:
            row.update({
                "payload_mb": float(summary.get("payload_mb", 0.0)),
                "throughput_avg": float(summary.get("alltoall_gbps_avg", 0.0)),
                "throughput_p95": float(summary.get("alltoall_gbps_p95", 0.0)),
            })
        rows.append(row)
        per_w_details[w_dir.name] = {
            "workload": this_workload,
            "summary": summary,
            "nccl": {
                "event_counts": dict(nccl["event_counts"]),
                "w_eff_values": dict(nccl["w_eff_values"]),
            },
            "step_count": len(step_rows),
        }

    rows.sort(key=lambda row: row["w"])
    ws = [row["w"] for row in rows]
    if workload is None:
        raise ValueError(f"no W directories found under {run_root}")

    p_step = plots_dir / "sweep_step_ms.png"
    if workload == "ring":
        step_sweep_series = [
            ("step_ms_avg", [row["step_ms_avg"] for row in rows], "#0c5fb0"),
            ("step_ms_p95", [row["step_ms_p95"] for row in rows], "#d97706"),
            ("backward_ms_p95", [row["backward_ms_p95"] for row in rows], "#136245"),
        ]
        throughput_title = "W Sweep - Ring Throughput Estimate"
        throughput_caption = "estimated ring throughput vs W"
        throughput_help_key = "ring_gbps_avg"
    else:
        step_sweep_series = [
            ("step_ms_avg", [row["step_ms_avg"] for row in rows], "#0c5fb0"),
            ("step_ms_p50", [row["step_ms_p50"] for row in rows], "#d97706"),
            ("step_ms_p95", [row["step_ms_p95"] for row in rows], "#136245"),
        ]
        throughput_title = "W Sweep - AllToAll Throughput Estimate"
        throughput_caption = "estimated alltoall throughput vs W"
        throughput_help_key = "alltoall_gbps_avg"
    save_line_plot(
        p_step,
        "W Sweep - Step Latency",
        ws,
        step_sweep_series,
        xlabel="W",
        ylabel="ms",
    )

    p_bw = plots_dir / "sweep_throughput.png"
    save_line_plot(
        p_bw,
        throughput_title,
        ws,
        [
            ("throughput_avg", [row["throughput_avg"] for row in rows], "#136245"),
            ("throughput_p95", [row["throughput_p95"] for row in rows], "#7c3aed"),
        ],
        xlabel="W",
        ylabel="Gbps",
    )

    p_wstall = plots_dir / "sweep_wstall.png"
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    xs = list(range(len(ws)))
    recv_vals = [row["recv_wstall_count"] for row in rows]
    send_vals = [row["send_wstall_count"] for row in rows]
    ax.bar([x - 0.18 for x in xs], recv_vals, width=0.36, label="PROXY_RECV_WSTALL", color="#d62728")
    ax.bar([x + 0.18 for x in xs], send_vals, width=0.36, label="PROXY_SEND_WSTALL", color="#1f77b4")
    ax.set_xticks(xs, [str(w) for w in ws])
    ax.set_xlabel("W")
    ax.set_ylabel("count")
    ax.set_title("W Sweep - Window Stall Counts")
    ylim = positive_ylim(max(recv_vals + send_vals) if (recv_vals or send_vals) else 0.0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(p_wstall, dpi=150, bbox_inches="tight")
    plt.close(fig)

    p_occ = plots_dir / "sweep_occupancy.png"
    save_line_plot(
        p_occ,
        "W Sweep - Outstanding Depth",
        ws,
        [
            ("p99_occ_pd", [row["p99_occ_pd"] for row in rows], "#0c5fb0"),
            ("p99_occ_tr", [row["p99_occ_tr"] for row in rows], "#d97706"),
            ("max_occ_pd", [row["max_occ_pd"] for row in rows], "#136245"),
            ("max_occ_tr", [row["max_occ_tr"] for row in rows], "#7c3aed"),
        ],
        xlabel="W",
        ylabel="depth",
    )

    csv_path = output_dir / "phase1_report.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    json_path = output_dir / "phase1_report.json"
    json_path.write_text(json.dumps({"workload": workload, "rows": rows, "details": per_w_details}, indent=2, sort_keys=True), encoding="utf-8")

    if workload == "ring":
        table_rows = [
            [
                row["w"],
                f"{row['step_ms_avg']:.4f}",
                f"{row['step_ms_p95']:.4f}",
                f"{row['backward_ms_p95']:.4f}",
                f"{row['throughput_avg']:.4f}",
                row["recv_wstall_count"],
                row["send_wstall_count"],
                f"{row['p99_occ_pd']:.4f}",
                f"{row['p99_occ_tr']:.4f}",
                per_w_links.get(f"W{row['w']}", ""),
            ]
            for row in rows
        ]
        table_header_html = (
            '<th>W</th><th>step_ms_avg</th><th>step_ms_p95</th><th>backward_ms_p95</th>'
            '<th>ring_gbps_avg</th><th>recv_wstall</th><th>send_wstall</th>'
            '<th>p99_occ_pd</th><th>p99_occ_tr</th><th>detail</th>'
        )
        sweep_field_help = {
            "W": "NCCL_PHASE1_STATIC_W 값.",
            "step_ms_avg": "해당 W의 rank max step latency 평균(ms).",
            "step_ms_p95": "해당 W의 rank max step latency p95(ms).",
            "backward_ms_p95": "해당 W의 backward 구간 p95(ms).",
            "ring_gbps_avg": "해당 W의 근사 ring throughput 평균(Gbps).",
            "recv_wstall": "PROXY_RECV_WSTALL 총 개수.",
            "send_wstall": "PROXY_SEND_WSTALL 총 개수.",
            "p99_occ_pd": "posted-done의 p99.",
            "p99_occ_tr": "transmitted-done의 p99.",
            "detail": "해당 W의 상세 HTML 리포트 링크.",
        }
        sweep_field_order = ["W", "step_ms_avg", "step_ms_p95", "backward_ms_p95", "ring_gbps_avg", "recv_wstall", "send_wstall", "p99_occ_pd", "p99_occ_tr", "detail"]
    else:
        table_rows = [
            [
                row["w"],
                f"{row['step_ms_avg']:.4f}",
                f"{row['step_ms_p50']:.4f}",
                f"{row['step_ms_p95']:.4f}",
                f"{row['throughput_avg']:.4f}",
                row["recv_wstall_count"],
                row["send_wstall_count"],
                f"{row['p99_occ_pd']:.4f}",
                f"{row['p99_occ_tr']:.4f}",
                per_w_links.get(f"W{row['w']}", ""),
            ]
            for row in rows
        ]
        table_header_html = (
            '<th>W</th><th>step_ms_avg</th><th>step_ms_p50</th><th>step_ms_p95</th>'
            '<th>alltoall_gbps_avg</th><th>recv_wstall</th><th>send_wstall</th>'
            '<th>p99_occ_pd</th><th>p99_occ_tr</th><th>detail</th>'
        )
        sweep_field_help = {
            "W": "NCCL_PHASE1_STATIC_W 값.",
            "step_ms_avg": "해당 W의 rank max step latency 평균(ms).",
            "step_ms_p50": "해당 W의 rank max step latency p50(ms).",
            "step_ms_p95": "해당 W의 rank max step latency p95(ms).",
            "alltoall_gbps_avg": "해당 W의 근사 all-to-all throughput 평균(Gbps).",
            "recv_wstall": "PROXY_RECV_WSTALL 총 개수.",
            "send_wstall": "PROXY_SEND_WSTALL 총 개수.",
            "p99_occ_pd": "posted-done의 p99.",
            "p99_occ_tr": "transmitted-done의 p99.",
            "detail": "해당 W의 상세 HTML 리포트 링크.",
        }
        sweep_field_order = ["W", "step_ms_avg", "step_ms_p50", "step_ms_p95", "alltoall_gbps_avg", "recv_wstall", "send_wstall", "p99_occ_pd", "p99_occ_tr", "detail"]
    linked_table_rows = []
    for row in table_rows:
        link = row[-1]
        if link:
            row[-1] = f'<a href="{html.escape(link)}">open</a>'
        linked_table_rows.append(row)
    table_html = (
        '<table class="report-table"><thead><tr>'
        + table_header_html +
        '</tr></thead><tbody>'
        + "".join(
            "<tr>" + "".join(
                f"<td>{cell}</td>" if idx == len(row) - 1 else f"<td>{html.escape(str(cell))}</td>"
                for idx, cell in enumerate(row)
            ) + "</tr>"
            for row in linked_table_rows
        )
        + "</tbody></table>"
    )
    sections = [
        '<div class="card"><h2>W Sweep Summary</h2>' + table_html + "</div>",
        render_field_help(
            "Sweep Summary Field Meanings",
            sweep_field_help,
            sweep_field_order,
        ),
        '<div class="card"><h2>Plots</h2><div class="plot-grid">'
        + f'<figure><img src="{html.escape(relpath(p_step, output_dir))}" alt="step sweep"><figcaption>step latency vs W</figcaption></figure>'
        + f'<figure><img src="{html.escape(relpath(p_bw, output_dir))}" alt="throughput sweep"><figcaption>{html.escape(throughput_caption)}</figcaption></figure>'
        + f'<figure><img src="{html.escape(relpath(p_wstall, output_dir))}" alt="wstall sweep"><figcaption>recv/send WSTALL count vs W</figcaption></figure>'
        + f'<figure><img src="{html.escape(relpath(p_occ, output_dir))}" alt="occupancy sweep"><figcaption>outstanding depth vs W</figcaption></figure>'
        + "</div></div>",
        render_plot_help("Sweep Plot Descriptions", PLOT_EXPLANATIONS_SWEEP[workload] + COMMON_SWEEP_PLOT_EXPLANATIONS),
        '<div class="card"><h2>Per-W Detail Reports</h2><ul>'
        + "".join(
            f'<li><a href="{html.escape(link)}">{html.escape(name)}</a></li>'
            for name, link in sorted(per_w_links.items(), key=lambda item: int(W_DIR_RE.match(item[0]).group(1)))
        )
        + "</ul></div>",
    ]

    html_path = output_dir / "phase1_report.html"
    html_path.write_text(render_html(f"Phase1 B1 Sweep Report ({workload_label(workload)})", sections), encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path / "report"
    occupancy_workers = parse_csv_set(args.occupancy_workers)
    occupancy_ranks = parse_csv_int_set(args.occupancy_ranks)

    if is_run_root(input_path):
        html_path = build_run_root_report(
            input_path,
            output_dir,
            occupancy_workers=occupancy_workers,
            occupancy_ranks=occupancy_ranks,
        )
    else:
        html_path = build_single_w_report(
            input_path,
            output_dir,
            occupancy_workers=occupancy_workers,
            occupancy_ranks=occupancy_ranks,
        )

    print(f"[phase1-log-reporter] wrote {html_path}")


if __name__ == "__main__":
    main()
