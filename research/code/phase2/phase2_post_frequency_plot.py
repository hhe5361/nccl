#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_BASE_PATH = SCRIPT_DIR.parent / "phase3" / "phase3_log_reporter.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("phase3_log_reporter_base_for_phase2_post_freq", PHASE3_BASE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render sampled-step CTS/POST worker overlays for a single Phase2 experiment"
    )
    parser.add_argument("--input-dir", required=True, help="Single Phase2 experiment root")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/post_frequency_plots")
    parser.add_argument("--bin-ms", type=float, default=1.0, help="Histogram bin width in milliseconds")
    parser.add_argument(
        "--steps",
        default="10,15,20,25",
        help="Comma-separated collective steps to sample",
    )
    parser.add_argument("--include-warmup", action="store_true", help="Include warmup steps")
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


def parse_step_selection(raw: str) -> List[int]:
    selected: List[int] = []
    seen = set()
    for token in str(raw or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value in seen:
            continue
        seen.add(value)
        selected.append(value)
    return selected


def static_w_from_mode(mode: str, summary: dict) -> str:
    if "phase1_static_w" in summary:
        try:
            value = int(summary["phase1_static_w"])
            return "stock" if value == 0 else str(value)
        except (TypeError, ValueError):
            pass
    upper = str(mode or "").upper()
    if upper == "STOCK":
        return "stock"
    if upper.startswith("B2_W"):
        try:
            return str(int(upper.split("_W", 1)[1]))
        except (IndexError, ValueError):
            return "unknown"
    return "unknown"


def sanitize_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def find_mode_dirs(experiment_root: Path) -> List[Path]:
    mode_dirs: List[Path] = []
    for child in sorted(experiment_root.iterdir(), key=lambda p: mode_sort_key(p.name)):
        if child.is_dir() and list(child.glob("*_summary.json")):
            mode_dirs.append(child)
    return mode_dirs


def load_mode_summary(mode_dir: Path) -> dict:
    summary_files = sorted(mode_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"no *_summary.json found in {mode_dir}")
    return base.load_json(summary_files[0])


def first_existing(mapping: dict, keys: Sequence[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def load_worker_step_rows(mode_dir: Path, include_warmup: bool) -> Dict[str, List[dict]]:
    rows_by_worker: Dict[str, List[dict]] = {}
    for timing_path in sorted(mode_dir.glob("*/**/*_worker_step_timing.jsonl")):
        rows: List[dict] = []
        with timing_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not include_warmup and row.get("warmup", False):
                    continue
                rows.append(row)
        if rows:
            rows_by_worker[timing_path.parent.name] = rows
    return rows_by_worker


def build_step_windows(
    worker_step_rows: Dict[str, List[dict]],
    sampled_steps: Sequence[int],
) -> Dict[str, Dict[int, Tuple[int, int]]]:
    per_worker: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for worker, rows in worker_step_rows.items():
        for row in rows:
            step = int(row.get("step", -1))
            if step not in sampled_steps:
                continue
            start_ns = int(row.get("start_ns", 0) or 0)
            end_ns = int(row.get("end_ns", 0) or 0)
            if start_ns <= 0 or end_ns <= start_ns:
                continue
            per_worker.setdefault(worker, {})[step] = (start_ns, end_ns)
    return per_worker


def init_histograms(
    workers: Sequence[str],
    sampled_steps: Sequence[int],
    per_worker_windows: Dict[str, Dict[int, Tuple[int, int]]],
    bin_width_ns: int,
) -> Dict[int, Dict[str, Dict[str, object]]]:
    histograms: Dict[int, Dict[str, Dict[str, object]]] = {}
    for step in sampled_steps:
        histograms[step] = {}
        for worker in workers:
            bounds = per_worker_windows.get(worker, {}).get(step)
            if bounds is None:
                continue
            start_ns, end_ns = bounds
            num_bins = max(1, int(math.ceil((end_ns - start_ns) / float(bin_width_ns))))
            histograms[step][worker] = {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "counts": [0] * num_bins,
            }
    return histograms


def observed_w_label(values: Sequence[float]) -> str:
    normalized = sorted({int(v) if float(v).is_integer() else float(v) for v in values})
    if not normalized:
        return "unknown"
    return "-".join(str(v) for v in normalized)


def save_step_post_frequency_plot(
    path: Path,
    title: str,
    bin_ms: float,
    worker_series: Dict[str, Dict[str, object]],
) -> Optional[Path]:
    usable: List[Tuple[str, List[float], List[int]]] = []
    for worker, info in sorted(worker_series.items(), key=lambda item: base.worker_sort_key(item[0])):
        counts = list(info.get("counts", []))
        if not counts or not any(counts):
            continue
        x_values_ms = [((idx + 0.5) * int(info["bin_width_ns"])) / 1_000_000.0 for idx in range(len(counts))]
        usable.append((worker, x_values_ms, counts))
    if not usable:
        return None
    base.plt.figure(figsize=(12, 6.0))
    max_value = 0.0
    for worker, x_values_ms, counts in usable:
        max_value = max(max_value, max(float(value) for value in counts))
        base.plt.scatter(
            x_values_ms,
            counts,
            s=10,
            alpha=0.8,
            label=worker,
        )
    base.plt.title(title)
    base.plt.xlabel("time since sampled-step start (ms)")
    base.plt.ylabel(f"PROXY_RECV_POST count per {bin_ms:g} ms bin")
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


def save_step_stream_post_frequency_plot(
    path: Path,
    title: str,
    bin_ms: float,
    stream_series: Dict[str, Dict[str, object]],
) -> Optional[Path]:
    usable: List[Tuple[str, List[float], List[int]]] = []
    for key, info in sorted(stream_series.items()):
        counts = list(info.get("counts", []))
        if not counts or not any(counts):
            continue
        x_values_ms = [((idx + 0.5) * int(info["bin_width_ns"])) / 1_000_000.0 for idx in range(len(counts))]
        usable.append((key, x_values_ms, counts))
    if not usable:
        return None
    base.plt.figure(figsize=(12, 6.4))
    max_value = 0.0
    for key, x_values_ms, counts in usable:
        max_value = max(max_value, max(float(value) for value in counts))
        base.plt.scatter(
            x_values_ms,
            counts,
            s=8,
            alpha=0.7,
            label=key,
        )
    base.plt.title(title)
    base.plt.xlabel("time since sampled-step start (ms)")
    base.plt.ylabel(f"PROXY_RECV_POST count per {bin_ms:g} ms bin")
    ylim = base.positive_ylim(max_value)
    if ylim:
        base.plt.ylim(*ylim)
    base.plt.grid(True, alpha=0.25)
    base.plt.legend(ncol=2, fontsize=8)
    base.plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(path, dpi=180)
    base.plt.close()
    return path


def save_step_outstanding_trace_plot(
    path: Path,
    title: str,
    trace_series: Dict[str, List[Tuple[float, float]]],
) -> Optional[Path]:
    usable = [(key, points) for key, points in sorted(trace_series.items()) if points]
    if not usable:
        return None
    base.plt.figure(figsize=(12, 6.4))
    max_value = 0.0
    any_points = False
    for key, points in usable:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if not xs:
            continue
        any_points = True
        max_value = max(max_value, max(float(value) for value in ys))
        base.plt.scatter(xs, ys, s=8, alpha=0.7, label=key)
    if not any_points:
        base.plt.close()
        return None
    base.plt.title(title)
    base.plt.xlabel("time since sampled-step start (ms)")
    base.plt.ylabel("occPd (posted - done)")
    ylim = base.positive_ylim(max_value)
    if ylim:
        base.plt.ylim(*ylim)
    base.plt.grid(True, alpha=0.25)
    base.plt.legend(ncol=2, fontsize=8)
    base.plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(path, dpi=180)
    base.plt.close()
    return path


def save_worker_cts_volume_vs_w_plot(
    path: Path,
    experiment_name: str,
    mode_rows: List[dict],
) -> Optional[Path]:
    if not mode_rows:
        return None
    worker_names = sorted(
        {worker for row in mode_rows for worker in row["post_count_by_worker"].keys()},
        key=base.worker_sort_key,
    )
    if not worker_names:
        return None

    categories = [row["mode"] for row in mode_rows]
    x_positions = list(range(len(categories)))
    base.plt.figure(figsize=(12, 6.2))
    max_value = 0.0
    any_points = False
    for worker in worker_names:
        y_values = [int(row["post_count_by_worker"].get(worker, 0)) for row in mode_rows]
        if not any(y_values):
            continue
        any_points = True
        max_value = max(max_value, max(float(v) for v in y_values))
        base.plt.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=1.6,
            markersize=3.2,
            label=worker,
        )
    if not any_points:
        base.plt.close()
        return None
    base.plt.title(f"{experiment_name} Worker CTS/POST Volume by W")
    base.plt.xlabel("mode / configured W")
    base.plt.ylabel("total PROXY_RECV_POST count across sampled steps")
    base.plt.xticks(x_positions, categories, rotation=25, ha="right")
    ylim = base.positive_ylim(max_value)
    if ylim:
        base.plt.ylim(*ylim)
    base.plt.grid(True, axis="y", alpha=0.25)
    base.plt.legend(ncol=2)
    base.plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(path, dpi=180)
    base.plt.close()
    return path


def load_step_metric_rows(mode_dir: Path) -> List[dict]:
    step_files = sorted(mode_dir.glob("*_step_metrics.jsonl"))
    if not step_files:
        return []
    rows: List[dict] = []
    with step_files[0].open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def detect_slice_steps(mode_dir: Path, summary: dict, step_rows: Sequence[dict]) -> Optional[object]:
    direct = first_existing(
        summary,
        (
            "sliceSteps",
            "slice_steps",
            "slice_step",
            "slice_steps_by_channel",
            "sliceStepsByChannel",
        ),
    )
    if direct is not None:
        return direct
    for row in step_rows:
        direct = first_existing(
            row,
            (
                "sliceSteps",
                "slice_steps",
                "slice_step",
                "slice_steps_by_channel",
                "sliceStepsByChannel",
            ),
        )
        if direct is not None:
            return direct

    rg_path = shutil.which("rg")
    pattern = re.compile(r"sliceSteps(?:=|:|\s+)(\d+)")
    log_paths = sorted(mode_dir.glob("*/nccl.*.log"))
    for log_path in log_paths:
        if rg_path:
            proc = subprocess.Popen(
                [rg_path, "--text", "-m", "1", "sliceSteps", str(log_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            try:
                line = proc.stdout.readline()
            finally:
                proc.stdout.close()
                proc.wait()
            if line:
                matched = pattern.search(line)
                if matched:
                    return int(matched.group(1))
        else:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    if "sliceSteps" not in raw_line:
                        continue
                    matched = pattern.search(raw_line)
                    if matched:
                        return int(matched.group(1))
                    break
    return None


def save_mode_step_latency_bar_plot(
    path: Path,
    experiment_name: str,
    mode_name: str,
    rows: Sequence[dict],
) -> Optional[Path]:
    usable = [row for row in rows if "step_ms_max" in row]
    if not usable:
        return None
    x_values = [int(row.get("step", idx)) for idx, row in enumerate(usable)]
    y_values = [float(row.get("step_ms_max", 0.0)) for row in usable]
    if not x_values:
        return None
    base.plt.figure(figsize=(12, 5.6))
    base.plt.bar(x_values, y_values, width=0.85)
    base.plt.title(f"{experiment_name} {mode_name} Step Duration")
    base.plt.xlabel("collective step")
    base.plt.ylabel("step_ms_max")
    base.plt.ylim(0, 500)
    base.plt.grid(True, axis="y", alpha=0.25)
    base.plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(path, dpi=180)
    base.plt.close()
    return path


def save_step_latency_by_w_plot(
    path: Path,
    experiment_name: str,
    mode_dirs: Sequence[Path],
) -> Optional[Path]:
    series: List[Tuple[str, List[int], List[float]]] = []
    for mode_dir in sorted(mode_dirs, key=lambda item: mode_sort_key(item.name)):
        rows = load_step_metric_rows(mode_dir)
        if not rows:
            continue
        xs = [int(row.get("step", idx)) for idx, row in enumerate(rows)]
        ys = [float(row.get("step_ms_max", 0.0)) for row in rows]
        if not xs:
            continue
        series.append((mode_dir.name, xs, ys))
    if not series:
        return None
    return base.save_line_plot(
        path,
        f"{experiment_name} Step Latency by W",
        "collective step",
        "step_ms_max",
        series,
    )


def save_step_latency_by_w_grouped_bar_plot(
    path: Path,
    experiment_name: str,
    mode_dirs: Sequence[Path],
    metric_key: str,
    metric_label: str,
    title_suffix: str,
    y_max: Optional[float],
) -> Optional[Path]:
    series: List[Tuple[str, Dict[int, float]]] = []
    all_steps = set()
    for mode_dir in sorted(mode_dirs, key=lambda item: mode_sort_key(item.name)):
        rows = load_step_metric_rows(mode_dir)
        if not rows:
            continue
        step_map: Dict[int, float] = {}
        for idx, row in enumerate(rows):
            step = int(row.get("step", idx))
            raw_value = row.get(metric_key, None)
            if raw_value is None:
                continue
            value = float(raw_value)
            step_map[step] = value
            all_steps.add(step)
        if step_map:
            series.append((mode_dir.name, step_map))
    if not series or not all_steps:
        return None

    steps = sorted(all_steps)
    n_modes = len(series)
    group_width = 0.82
    bar_w = group_width / max(1, n_modes)
    centers = list(range(len(steps)))

    base.plt.figure(figsize=(12, 6.2))
    for mode_idx, (mode_name, step_map) in enumerate(series):
        xs = [center - group_width / 2.0 + (mode_idx + 0.5) * bar_w for center in centers]
        ys = [float(step_map.get(step, 0.0)) for step in steps]
        base.plt.bar(xs, ys, width=bar_w * 0.95, label=mode_name)

    base.plt.title(f"{experiment_name} {title_suffix}")
    base.plt.xlabel("collective step")
    base.plt.ylabel(metric_label)
    base.plt.xticks(centers, [str(step) for step in steps])
    if y_max is not None:
        base.plt.ylim(0, y_max)
    base.plt.grid(True, axis="y", alpha=0.25)
    base.plt.legend(ncol=2)
    base.plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.plt.savefig(path, dpi=180)
    base.plt.close()
    return path


def save_step_throughput_selected_modes_bar_plot(
    path: Path,
    experiment_name: str,
    mode_dirs: Sequence[Path],
    selected_modes: Sequence[str],
) -> Optional[Path]:
    selected_upper = {mode.upper() for mode in selected_modes}
    filtered_mode_dirs = [
        mode_dir
        for mode_dir in sorted(mode_dirs, key=lambda item: mode_sort_key(item.name))
        if mode_dir.name.upper() in selected_upper
    ]
    if not filtered_mode_dirs:
        return None
    return save_step_latency_by_w_grouped_bar_plot(
        path,
        experiment_name,
        filtered_mode_dirs,
        "collective_gbps_est",
        "collective_gbps_est",
        "Step Throughput by W (STOCK, W2, W8)",
        None,
    )


def iter_relevant_recv_lines(log_path: Path):
    rg_path = shutil.which("rg")
    if rg_path:
        proc = subprocess.Popen(
            [rg_path, "--text", "-e", "PROXY_RECV_POST|PROXY_RECV_CONSUMED|PROXY_RECV_VISIBLE|PROXY_RECV_NET_DONE|PROXY_RECV_WSTALL", str(log_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.stdout.close()
            proc.wait()
        return
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if "PROXY_RECV_" in raw_line:
                yield raw_line


def stream_key(worker: str, entry: dict) -> str:
    return f"{worker}:ch{entry.get('channel', '?')}:p{entry.get('peer', '?')}"


def process_mode(
    mode_dir: Path,
    summary: dict,
    sampled_steps: Sequence[int],
    include_warmup: bool,
    bin_ms: float,
    output_dir: Path,
    experiment_name: str,
) -> Tuple[int, Dict[str, int], str]:
    configured_w = static_w_from_mode(mode_dir.name, summary)
    worker_step_rows = load_worker_step_rows(mode_dir, include_warmup)
    if not worker_step_rows:
        return 0, {}, "unknown"
    log_paths = sorted(mode_dir.glob("*/nccl.*.log"))
    base.log_progress(
        f"scan mode={mode_dir.name} logs={len(log_paths)} sampled_steps={list(sampled_steps)}"
    )

    target_collapis = base.target_collapis_from_summary(summary)
    per_worker_windows = build_step_windows(worker_step_rows, sampled_steps)
    if not per_worker_windows:
        return 0, {}, "unknown"

    workers = sorted(worker_step_rows.keys(), key=base.worker_sort_key)
    bin_width_ns = max(1, int(bin_ms * 1_000_000.0))
    histograms = init_histograms(workers, sampled_steps, per_worker_windows, bin_width_ns)
    if not histograms:
        return 0, {}, "unknown"

    total_counts_by_worker: Dict[str, int] = {worker: 0 for worker in workers}
    observed_w_values: List[float] = []
    stream_histograms: Dict[int, Dict[str, Dict[str, object]]] = {step: {} for step in sampled_steps}
    stream_traces: Dict[int, Dict[str, List[Tuple[float, float]]]] = {step: defaultdict(list) for step in sampled_steps}

    for log_path in log_paths:
        worker = log_path.parent.name
        worker_windows = per_worker_windows.get(worker, {})
        if not worker_windows:
            continue
        worker_rows = worker_step_rows.get(worker, [])
        base.log_progress(f"scan worker_log mode={mode_dir.name} worker={worker} file={log_path.name}")
        for raw_line in iter_relevant_recv_lines(log_path):
            entry = base.parse_nccl_event(raw_line)
            if entry is None:
                continue
            if not base.nccl_entry_matches_target(entry, target_collapis, worker_rows):
                continue
            event = str(entry.get("event", ""))
            try:
                t_ns = int(entry.get("tNs", 0) or 0)
            except (TypeError, ValueError):
                continue
            matched_step: Optional[int] = None
            for step, (worker_start, worker_end) in worker_windows.items():
                if worker_start <= t_ns <= worker_end:
                    matched_step = step
                    break
            if matched_step is None:
                continue
            stream = stream_key(worker, entry)
            worker_start_for_step = per_worker_windows[worker][matched_step][0]
            rel_ms = (t_ns - worker_start_for_step) / 1_000_000.0
            if "occPd" in entry:
                try:
                    stream_traces[matched_step][stream].append((rel_ms, float(entry["occPd"])))
                except (TypeError, ValueError):
                    pass
            if event != "PROXY_RECV_POST":
                continue
            if "wEff" in entry:
                try:
                    observed_w_values.append(float(entry["wEff"]))
                except (TypeError, ValueError):
                    pass
            worker_hist = histograms.get(matched_step, {}).get(worker)
            if worker_hist is None:
                continue
            worker_start = int(worker_hist["start_ns"])
            worker_end = int(worker_hist["end_ns"])
            if not (worker_start <= t_ns <= worker_end):
                continue
            counts = worker_hist["counts"]
            idx = min(len(counts) - 1, max(0, int((t_ns - worker_start) // bin_width_ns)))
            counts[idx] += 1
            total_counts_by_worker[worker] += 1
            stream_hist = stream_histograms[matched_step].setdefault(
                stream,
                {"bin_width_ns": bin_width_ns, "counts": [0] * len(counts)},
            )
            stream_hist["counts"][idx] += 1

    generated = 0
    observed_w = observed_w_label(observed_w_values)
    for step in sampled_steps:
        if not histograms.get(step, {}):
            continue
        outstanding_path = output_dir / (
            f"recv_outstanding_trace_by_stream_{sanitize_label(mode_dir.name)}"
            f"_step_{step}_cfgW_{sanitize_label(configured_w)}"
            f"_obsW_{sanitize_label(observed_w)}.png"
        )
        outstanding_title = (
            f"{experiment_name} {mode_dir.name} step={step} Stream Outstanding Trace "
            f"(cfgW={configured_w}, observedW={observed_w})"
        )
        saved_outstanding = save_step_outstanding_trace_plot(
            outstanding_path,
            outstanding_title,
            stream_traces.get(step, {}),
        )
        if saved_outstanding is not None:
            generated += 1
            base.log_progress(f"wrote outstanding plot mode={mode_dir.name} step={step} path={saved_outstanding.as_posix()}")
    return generated, total_counts_by_worker, observed_w


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.input_dir).expanduser().resolve()
    if not experiment_root.exists():
        raise SystemExit(f"input-dir not found: {experiment_root}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "post_frequency_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_steps = parse_step_selection(args.steps)
    if not sampled_steps:
        raise SystemExit("no valid --steps specified")

    mode_dirs = find_mode_dirs(experiment_root)
    if not mode_dirs:
        raise SystemExit(f"no mode directories found in {experiment_root}")

    base.log_progress(
        f"phase2 post-frequency start experiment={experiment_root.name} modes={len(mode_dirs)} "
        f"bin_ms={args.bin_ms:g} sampled_steps={sampled_steps}"
    )

    generated_total = 0
    skipped: List[str] = []
    summary_rows: List[dict] = []
    for mode_dir in mode_dirs:
        summary = load_mode_summary(mode_dir)
        mode_step_rows = load_step_metric_rows(mode_dir)
        slice_steps = detect_slice_steps(mode_dir, summary, mode_step_rows)
        generated, total_counts_by_worker, observed_w = process_mode(
            mode_dir,
            summary,
            sampled_steps,
            args.include_warmup,
            args.bin_ms,
            output_dir,
            experiment_root.name,
        )
        if generated == 0:
            skipped.append(mode_dir.name)
            base.log_progress(f"skip mode={mode_dir.name} reason=no_post_samples_for_sampled_steps")
            continue
        generated_total += generated
        summary_rows.append(
            {
                "mode": mode_dir.name,
                "configured_w": static_w_from_mode(mode_dir.name, summary),
                "observed_w": observed_w,
                "post_count_by_worker": total_counts_by_worker,
            }
        )
        if slice_steps is not None:
            summary_rows[-1]["slice_steps"] = slice_steps
        mode_latency_plot = save_mode_step_latency_bar_plot(
            output_dir / f"step_latency_bar_{sanitize_label(mode_dir.name)}.png",
            experiment_root.name,
            mode_dir.name,
            mode_step_rows,
        )
        if mode_latency_plot is not None:
            base.log_progress(f"wrote mode latency bar plot path={mode_latency_plot.as_posix()}")

    summary_rows.sort(key=lambda row: mode_sort_key(row["mode"]))
    summary_plot = save_worker_cts_volume_vs_w_plot(
        output_dir / "cts_post_volume_by_worker_vs_w.png",
        experiment_root.name,
        summary_rows,
    )
    if summary_plot is not None:
        base.log_progress(f"wrote summary plot path={summary_plot.as_posix()}")

    latency_plot = save_step_latency_by_w_plot(
        output_dir / "step_latency_by_w.png",
        experiment_root.name,
        mode_dirs,
    )
    if latency_plot is not None:
        base.log_progress(f"wrote latency plot path={latency_plot.as_posix()}")

    latency_bar_plot_500 = save_step_latency_by_w_grouped_bar_plot(
        output_dir / "step_latency_by_w_bar_y500.png",
        experiment_root.name,
        mode_dirs,
        "step_ms_max",
        "step_ms_max",
        "Step Latency by W (0-500 ms)",
        500.0,
    )
    if latency_bar_plot_500 is not None:
        base.log_progress(f"wrote latency bar plot path={latency_bar_plot_500.as_posix()}")

    latency_bar_plot_300 = save_step_latency_by_w_grouped_bar_plot(
        output_dir / "step_latency_by_w_bar_y300.png",
        experiment_root.name,
        mode_dirs,
        "step_ms_max",
        "step_ms_max",
        "Step Latency by W (0-300 ms)",
        300.0,
    )
    if latency_bar_plot_300 is not None:
        base.log_progress(f"wrote latency bar plot path={latency_bar_plot_300.as_posix()}")

    latency_bar_plot_700 = save_step_latency_by_w_grouped_bar_plot(
        output_dir / "step_latency_by_w_bar_y700.png",
        experiment_root.name,
        mode_dirs,
        "step_ms_max",
        "step_ms_max",
        "Step Latency by W (0-700 ms)",
        700.0,
    )
    if latency_bar_plot_700 is not None:
        base.log_progress(f"wrote latency bar plot path={latency_bar_plot_700.as_posix()}")

    throughput_bar_plot = save_step_latency_by_w_grouped_bar_plot(
        output_dir / "step_throughput_by_w_bar.png",
        experiment_root.name,
        mode_dirs,
        "collective_gbps_est",
        "collective_gbps_est",
        "Step Throughput by W",
        None,
    )
    if throughput_bar_plot is not None:
        base.log_progress(f"wrote throughput bar plot path={throughput_bar_plot.as_posix()}")

    throughput_selected_plot = save_step_throughput_selected_modes_bar_plot(
        output_dir / "step_throughput_by_w_bar_stock_w2_w8.png",
        experiment_root.name,
        mode_dirs,
        ("STOCK", "B2_W2", "B2_W8"),
    )
    if throughput_selected_plot is not None:
        base.log_progress(f"wrote throughput bar plot path={throughput_selected_plot.as_posix()}")

    summary_json_path = output_dir / "post_frequency_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": experiment_root.name,
                "sampled_steps": sampled_steps,
                "bin_ms": args.bin_ms,
                "modes": summary_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    base.log_progress(f"wrote summary json path={summary_json_path.as_posix()}")

    base.log_progress(
        f"phase2 post-frequency complete experiment={experiment_root.name} generated={generated_total} skipped={len(skipped)}"
    )
    if skipped:
        base.log_progress(f"skipped_modes={','.join(skipped)}")


if __name__ == "__main__":
    main()
