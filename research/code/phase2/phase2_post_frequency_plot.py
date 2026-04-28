#!/usr/bin/env python3
import argparse
import importlib.util
import math
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
        description="Render per-mode worker CTS/POST frequency overlays for a single Phase2 experiment"
    )
    parser.add_argument("--input-dir", required=True, help="Single Phase2 experiment root")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <input-dir>/post_frequency_plots",
    )
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=10.0,
        help="Histogram bin width in milliseconds for instantaneous POST counts",
    )
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include warmup steps in the measurement window",
    )
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


def effective_worker_steps(worker_steps: List[dict], include_warmup: bool) -> List[dict]:
    if include_warmup:
        return worker_steps
    rows = [row for row in worker_steps if not row.get("warmup", False)]
    return rows if rows else worker_steps


def worker_bounds(worker_steps: List[dict], include_warmup: bool) -> Optional[Tuple[int, int]]:
    rows = effective_worker_steps(worker_steps, include_warmup)
    if not rows:
        return None
    starts = [int(row.get("start_ns", 0) or 0) for row in rows]
    ends = [int(row.get("end_ns", 0) or 0) for row in rows]
    starts = [value for value in starts if value > 0]
    ends = [value for value in ends if value > 0]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def load_mode_summary(mode_dir: Path) -> dict:
    summary_files = sorted(mode_dir.glob("*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"no *_summary.json found in {mode_dir}")
    return base.load_json(summary_files[0])


def find_mode_dirs(experiment_root: Path) -> List[Path]:
    mode_dirs: List[Path] = []
    for child in sorted(experiment_root.iterdir(), key=lambda p: mode_sort_key(p.name)):
        if child.is_dir() and list(child.glob("*_summary.json")):
            mode_dirs.append(child)
    return mode_dirs


def collect_post_timestamps(
    mode_dir: Path,
    summary: dict,
    include_warmup: bool,
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]], Optional[Tuple[int, int]]]:
    worker_step_timings = base.load_worker_step_timings(mode_dir)
    target_collapis = base.target_collapis_from_summary(summary)

    bounds_by_worker: Dict[str, Tuple[int, int]] = {}
    global_start: Optional[int] = None
    global_end: Optional[int] = None
    for worker, steps in worker_step_timings.items():
        bounds = worker_bounds(steps, include_warmup)
        if bounds is None:
            continue
        bounds_by_worker[worker] = bounds
        start_ns, end_ns = bounds
        global_start = start_ns if global_start is None else min(global_start, start_ns)
        global_end = end_ns if global_end is None else max(global_end, end_ns)

    if global_start is None or global_end is None or global_end <= global_start:
        return {}, {}, None

    post_times: Dict[str, List[int]] = defaultdict(list)
    post_w_values: Dict[str, List[float]] = defaultdict(list)
    for log_path in sorted(mode_dir.glob("*/nccl.*.log")):
        worker = log_path.parent.name
        if worker not in bounds_by_worker:
            continue
        worker_steps = worker_step_timings.get(worker, [])
        worker_start, worker_end = bounds_by_worker[worker]
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                entry = base.parse_nccl_event(raw_line)
                if entry is None:
                    continue
                if str(entry.get("event", "")) != "PROXY_RECV_POST":
                    continue
                if not base.nccl_entry_matches_target(entry, target_collapis, worker_steps):
                    continue
                try:
                    t_ns = int(entry.get("tNs", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if not (worker_start <= t_ns <= worker_end):
                    continue
                post_times[worker].append(t_ns)
                if "wEff" in entry:
                    try:
                        post_w_values[worker].append(float(entry["wEff"]))
                    except (TypeError, ValueError):
                        pass
    return dict(post_times), dict(post_w_values), (global_start, global_end)


def build_histogram_series(
    post_times: Dict[str, List[int]],
    start_ns: int,
    end_ns: int,
    bin_ms: float,
) -> Tuple[List[float], Dict[str, List[int]]]:
    if end_ns <= start_ns:
        return [], {}
    bin_width_ns = max(1, int(bin_ms * 1_000_000.0))
    num_bins = max(1, int(math.ceil((end_ns - start_ns) / float(bin_width_ns))))
    centers = [((idx + 0.5) * bin_width_ns) / 1_000_000_000.0 for idx in range(num_bins)]
    series: Dict[str, List[int]] = {}
    for worker, timestamps in sorted(post_times.items(), key=lambda item: base.worker_sort_key(item[0])):
        counts = [0] * num_bins
        for ts_ns in timestamps:
            if ts_ns < start_ns or ts_ns > end_ns:
                continue
            idx = min(num_bins - 1, max(0, int((ts_ns - start_ns) // bin_width_ns)))
            counts[idx] += 1
        series[worker] = counts
    return centers, series


def observed_w_label(post_w_values: Dict[str, List[float]]) -> str:
    values = sorted({int(v) if float(v).is_integer() else float(v) for rows in post_w_values.values() for v in rows})
    if not values:
        return "unknown"
    return "-".join(str(v) for v in values)


def save_mode_post_frequency_plot(
    path: Path,
    experiment_name: str,
    mode: str,
    configured_w: str,
    observed_w: str,
    bin_ms: float,
    x_values: List[float],
    series: Dict[str, List[int]],
) -> Optional[Path]:
    usable = [(worker, counts) for worker, counts in series.items() if counts and any(counts)]
    if not usable or not x_values:
        return None
    base.plt.figure(figsize=(12, 6.0))
    max_value = 0.0
    for worker, counts in usable:
        max_value = max(max_value, max(float(value) for value in counts))
        base.plt.plot(
            x_values,
            counts,
            linewidth=1.4,
            drawstyle="steps-mid",
            label=worker,
        )
    base.plt.title(
        f"{experiment_name} {mode} CTS/POST Frequency "
        f"(cfgW={configured_w}, observedW={observed_w}, bin={bin_ms:g}ms)"
    )
    base.plt.xlabel("time since measured-step start (s)")
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


def save_worker_cts_volume_vs_w_plot(
    path: Path,
    experiment_name: str,
    mode_rows: List[dict],
) -> Optional[Path]:
    if not mode_rows:
        return None
    worker_names = sorted(
        {
            worker
            for row in mode_rows
            for worker in row["post_count_by_worker"].keys()
        },
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
    base.plt.ylabel("total PROXY_RECV_POST count")
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


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.input_dir).expanduser().resolve()
    if not experiment_root.exists():
        raise SystemExit(f"input-dir not found: {experiment_root}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "post_frequency_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_dirs = find_mode_dirs(experiment_root)
    if not mode_dirs:
        raise SystemExit(f"no mode directories found in {experiment_root}")

    base.log_progress(
        f"phase2 post-frequency start experiment={experiment_root.name} modes={len(mode_dirs)} bin_ms={args.bin_ms:g}"
    )

    generated = 0
    skipped: List[str] = []
    summary_rows: List[dict] = []
    for mode_dir in mode_dirs:
        mode = mode_dir.name
        summary = load_mode_summary(mode_dir)
        configured_w = static_w_from_mode(mode, summary)
        post_times, post_w_values, bounds = collect_post_timestamps(mode_dir, summary, args.include_warmup)
        if bounds is None:
            skipped.append(mode)
            base.log_progress(f"skip mode={mode} reason=no_step_window")
            continue
        x_values, series = build_histogram_series(post_times, bounds[0], bounds[1], args.bin_ms)
        observed_w = observed_w_label(post_w_values)
        filename = (
            f"cts_post_frequency_{sanitize_label(mode)}"
            f"_cfgW_{sanitize_label(configured_w)}"
            f"_obsW_{sanitize_label(observed_w)}.png"
        )
        output_path = output_dir / filename
        saved = save_mode_post_frequency_plot(
            output_path,
            experiment_root.name,
            mode,
            configured_w,
            observed_w,
            args.bin_ms,
            x_values,
            series,
        )
        if saved is None:
            skipped.append(mode)
            base.log_progress(f"skip mode={mode} reason=no_post_samples")
            continue
        summary_rows.append(
            {
                "mode": mode,
                "configured_w": configured_w,
                "observed_w": observed_w,
                "post_count_by_worker": {
                    worker: len(timestamps) for worker, timestamps in sorted(post_times.items(), key=lambda item: base.worker_sort_key(item[0]))
                },
            }
        )
        generated += 1
        base.log_progress(f"wrote plot mode={mode} path={saved.as_posix()}")

    summary_rows.sort(key=lambda row: mode_sort_key(row["mode"]))
    summary_plot = save_worker_cts_volume_vs_w_plot(
        output_dir / "cts_post_volume_by_worker_vs_w.png",
        experiment_root.name,
        summary_rows,
    )
    if summary_plot is not None:
        base.log_progress(f"wrote summary plot path={summary_plot.as_posix()}")

    base.log_progress(
        f"phase2 post-frequency complete experiment={experiment_root.name} generated={generated} skipped={len(skipped)}"
    )
    if skipped:
        base.log_progress(f"skipped_modes={','.join(skipped)}")


if __name__ == "__main__":
    main()
