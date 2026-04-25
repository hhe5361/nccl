#!/usr/bin/env python3
import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MODE_ORDER = {"STOCK": 0, "B2": 1, "B3": 2}
WORKER_RE = re.compile(r"worker(\d+)$")
PHASE_LINE_RE = re.compile(r"PHASE\d+ event=")
ERROR_LINE_RE = re.compile(
    r"Watchdog caught collective operation timeout|Timeout at NCCL work|"
    r"Process group watchdog|SIGABRT|abort called|Some NCCL operations have failed|"
    r"DistBackendError|ChildFailedError"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect focused debug data for NCCL experiment failures")
    parser.add_argument("--experiment-root", required=True, help="Experiment root such as .../07_mixed8_allreduce_ring_32mb")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <experiment-root>/debug_bundle")
    parser.add_argument("--status-dir", default=None, help="Optional .matrix_status/<experiment> directory")
    parser.add_argument("--tail-lines", type=int, default=20, help="Number of matching log lines to keep per worker")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def last_jsonl_row(path: Path) -> Optional[dict]:
    last = None
    for row in iter_jsonl(path):
        last = row
    return last


def relpath(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def worker_sort_key(name: str) -> Tuple[int, str]:
    match = WORKER_RE.match(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def infer_workers(env_setup: dict, experiment_root: Path) -> List[str]:
    workers = set()
    nnodes = int(env_setup.get("nnodes", 0) or 0)
    if nnodes > 0:
        for idx in range(1, nnodes + 1):
            workers.add(f"worker{idx:02d}")
    for mode_dir in experiment_root.iterdir():
        if not mode_dir.is_dir():
            continue
        for worker_dir in mode_dir.iterdir():
            if worker_dir.is_dir() and worker_dir.name.startswith("worker"):
                workers.add(worker_dir.name)
    return sorted(workers, key=worker_sort_key)


def infer_modes(env_setup: dict, experiment_root: Path) -> List[str]:
    modes = set()
    raw = str(env_setup.get("run_modes", "")).strip()
    if raw:
        for item in raw.split(","):
            token = item.strip().upper()
            if token:
                modes.add(token)
    for candidate in ("STOCK", "B2", "B3"):
        if (experiment_root / candidate).exists():
            modes.add(candidate)
    return sorted(modes, key=lambda name: MODE_ORDER.get(name, 999))


def find_single_file(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def collect_matching_lines(path: Path, pattern: re.Pattern, limit: int) -> List[dict]:
    matches: deque = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            if pattern.search(line):
                matches.append({"lineno": lineno, "text": line.rstrip()})
    return list(matches)


def summarize_mode_worker(mode_dir: Path, mode: str, worker: str, experiment_root: Path, tail_lines: int) -> dict:
    worker_dir = mode_dir / worker
    timing_path = worker_dir / f"{mode}_worker_step_timing.jsonl"
    log_path = find_single_file(worker_dir, "nccl.*.log")

    timing_last = last_jsonl_row(timing_path) if timing_path.exists() else None
    phase_tail = collect_matching_lines(log_path, PHASE_LINE_RE, tail_lines) if log_path is not None else []
    error_tail = collect_matching_lines(log_path, ERROR_LINE_RE, tail_lines) if log_path is not None else []

    return {
        "worker": worker,
        "dir_exists": worker_dir.exists(),
        "timing_path": relpath(timing_path if timing_path.exists() else None, experiment_root),
        "timing_last": timing_last,
        "log_path": relpath(log_path, experiment_root),
        "phase_tail": phase_tail,
        "error_tail": error_tail,
    }


def summarize_mode(experiment_root: Path, mode: str, workers: List[str], total_steps: int, tail_lines: int) -> dict:
    mode_dir = experiment_root / mode
    summary_path = mode_dir / f"{mode}_summary.json"
    step_metrics_path = mode_dir / f"{mode}_step_metrics.jsonl"
    worker_rows = [summarize_mode_worker(mode_dir, mode, worker, experiment_root, tail_lines) for worker in workers]

    completed_workers = []
    missing_dirs = []
    missing_timing = []
    missing_logs = []
    for row in worker_rows:
        if not row["dir_exists"]:
            missing_dirs.append(row["worker"])
        if row["timing_last"] is None:
            missing_timing.append(row["worker"])
        else:
            if int(row["timing_last"].get("step", -1)) == total_steps - 1:
                completed_workers.append(row["worker"])
        if row["log_path"] is None:
            missing_logs.append(row["worker"])

    return {
        "mode": mode,
        "mode_dir_exists": mode_dir.exists(),
        "summary_path": relpath(summary_path if summary_path.exists() else None, experiment_root),
        "summary": load_json(summary_path) if summary_path.exists() else None,
        "step_metrics_path": relpath(step_metrics_path if step_metrics_path.exists() else None, experiment_root),
        "step_metrics_lines": count_jsonl(step_metrics_path) if step_metrics_path.exists() else 0,
        "step_metrics_last": last_jsonl_row(step_metrics_path) if step_metrics_path.exists() else None,
        "workers": worker_rows,
        "completed_workers": completed_workers,
        "missing_dirs": missing_dirs,
        "missing_timing": missing_timing,
        "missing_logs": missing_logs,
        "all_workers_completed_final_step": len(completed_workers) == len(workers) and len(workers) > 0,
    }


def load_status_dir(status_dir: Optional[Path], workers: List[str]) -> Optional[dict]:
    if status_dir is None or not status_dir.exists():
        return None
    rows = []
    for worker in workers:
        path = status_dir / f"{worker}.status"
        if not path.exists():
            rows.append({"worker": worker, "exists": False, "participated": None, "rc": None})
            continue
        parsed = {"worker": worker, "exists": True, "participated": None, "rc": None}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key == "participated":
                    parsed["participated"] = int(value)
                elif key == "rc":
                    parsed["rc"] = int(value)
        rows.append(parsed)
    return {"status_dir": str(status_dir), "workers": rows}


def infer_diagnosis(mode_summaries: List[dict]) -> List[str]:
    messages: List[str] = []
    for idx, mode in enumerate(mode_summaries[:-1]):
        next_mode = mode_summaries[idx + 1]
        if mode["all_workers_completed_final_step"]:
            if next_mode["missing_dirs"] or next_mode["missing_timing"]:
                messages.append(
                    f"{mode['mode']} completed all worker step loops, but {next_mode['mode']} is only partially initialized. "
                    f"This points to a teardown/barrier or mode-handoff failure after {mode['mode']} rather than an in-loop payload failure."
                )
                break
    if not messages:
        for mode in mode_summaries:
            if mode["missing_timing"]:
                messages.append(
                    f"{mode['mode']} is missing worker timing for {', '.join(mode['missing_timing'])}. "
                    f"This points to an in-mode failure before all workers finished the step loop."
                )
                break
    if not messages:
        messages.append("No strong handoff failure pattern detected from timing files alone. Inspect per-worker phase tails in the JSON output.")
    return messages


def build_markdown(summary: dict) -> str:
    env = summary["env_setup"]
    lines: List[str] = []
    lines.append(f"# NCCL Failure Debug Bundle: {summary['experiment_root_name']}")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- phase: `{env.get('phase', '')}`")
    lines.append(f"- run_id: `{env.get('run_id', '')}`")
    lines.append(f"- placement: `{env.get('placement', '')}`")
    lines.append(f"- collective: `{env.get('collective', '')}`")
    lines.append(f"- run_modes: `{env.get('run_modes', '')}`")
    lines.append(f"- nnodes: `{env.get('nnodes', '')}`")
    lines.append(f"- steps: `{env.get('steps', '')}`")
    lines.append(f"- payload_mb: `{env.get('payload_mb', '')}`")
    lines.append("")
    lines.append("## Diagnosis")
    for item in summary["diagnosis"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Mode Summary")
    lines.append("| mode | summary | step_metrics lines | completed workers | missing dirs | missing timing | missing logs |")
    lines.append("| --- | --- | ---: | --- | --- | --- | --- |")
    for mode in summary["modes"]:
        lines.append(
            f"| `{mode['mode']}` | "
            f"{'yes' if mode['summary'] is not None else 'no'} | "
            f"{mode['step_metrics_lines']} | "
            f"{len(mode['completed_workers'])}/{len(summary['workers'])} | "
            f"{', '.join(mode['missing_dirs']) or '-'} | "
            f"{', '.join(mode['missing_timing']) or '-'} | "
            f"{', '.join(mode['missing_logs']) or '-'} |"
        )
    lines.append("")
    for mode in summary["modes"]:
        lines.append(f"## {mode['mode']} Worker Tail")
        for worker_row in mode["workers"]:
            last = worker_row["timing_last"]
            last_step = last.get("step") if last else None
            lines.append(
                f"- `{worker_row['worker']}`: dir={'yes' if worker_row['dir_exists'] else 'no'}, "
                f"timing_last_step={last_step if last_step is not None else 'missing'}, "
                f"log={'yes' if worker_row['log_path'] else 'no'}"
            )
        lines.append("")
    if summary.get("status") is not None:
        lines.append("## Matrix Status")
        for row in summary["status"]["workers"]:
            lines.append(
                f"- `{row['worker']}`: exists={row['exists']}, participated={row['participated']}, rc={row['rc']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    if not experiment_root.exists():
        raise SystemExit(f"experiment root not found: {experiment_root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiment_root / "debug_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_setup_path = experiment_root / "env_setup.json"
    if not env_setup_path.exists():
        raise SystemExit(f"env_setup.json not found under {experiment_root}")
    env_setup = load_json(env_setup_path)

    workers = infer_workers(env_setup, experiment_root)
    modes = infer_modes(env_setup, experiment_root)
    total_steps = int(env_setup.get("steps", 0) or 0)

    mode_summaries = [summarize_mode(experiment_root, mode, workers, total_steps, args.tail_lines) for mode in modes]
    status = load_status_dir(Path(args.status_dir).resolve() if args.status_dir else None, workers)

    summary = {
        "experiment_root": str(experiment_root),
        "experiment_root_name": experiment_root.name,
        "env_setup_path": str(env_setup_path),
        "env_setup": env_setup,
        "workers": workers,
        "modes": mode_summaries,
        "status": status,
        "diagnosis": infer_diagnosis(mode_summaries),
    }

    summary_path = output_dir / "debug_summary.json"
    report_path = output_dir / "debug_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(build_markdown(summary), encoding="utf-8")

    print(f"wrote summary: {summary_path}")
    print(f"wrote report:  {report_path}")


if __name__ == "__main__":
    main()
