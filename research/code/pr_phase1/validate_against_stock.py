#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def compare_probe(lhs: dict, rhs: dict):
    checks = {
        "sum": approx_equal(lhs["sum"], rhs["sum"]),
        "mean": approx_equal(lhs["mean"], rhs["mean"]),
        "min": approx_equal(lhs["min"], rhs["min"]),
        "max": approx_equal(lhs["max"], rhs["max"]),
        "sample": len(lhs["sample"]) == len(rhs["sample"]) and all(
            approx_equal(float(x), float(y)) for x, y in zip(lhs["sample"], rhs["sample"])
        ),
    }
    return all(checks.values()), checks


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    repeat_name = f"repeat_{args.repeat_index:02d}"
    mode_dir = run_root / args.mode / args.experiment / repeat_name
    stock_dir = run_root / "STOCK" / args.experiment / repeat_name
    validation_path = mode_dir / "stock_probe_validation.json"

    if not mode_dir.exists():
        raise SystemExit(f"mode dir not found: {mode_dir}")

    if args.mode == "STOCK":
        payload = {
            "mode": args.mode,
            "experiment": args.experiment,
            "repeat": repeat_name,
            "baseline_ready": stock_dir.exists(),
            "comparison_performed": False,
            "ok": True,
        }
        validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if not stock_dir.exists():
        payload = {
            "mode": args.mode,
            "experiment": args.experiment,
            "repeat": repeat_name,
            "comparison_performed": False,
            "ok": False,
            "reason": f"stock baseline not found: {stock_dir}",
        }
        validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit(1)

    worker_results = []
    overall_ok = True
    for worker_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("worker")):
        stock_worker_dir = stock_dir / worker_dir.name
        mode_summary = load_json(worker_dir / "rank00_summary.json")
        stock_summary = load_json(stock_worker_dir / "rank00_summary.json")
        mode_trace = load_jsonl(worker_dir / "rank00_step_trace.jsonl")
        stock_trace = load_jsonl(stock_worker_dir / "rank00_step_trace.jsonl")

        step_checks = []
        if len(mode_trace) != len(stock_trace):
            overall_ok = False
            worker_results.append({
                "worker": worker_dir.name,
                "ok": False,
                "reason": f"trace length mismatch mode={len(mode_trace)} stock={len(stock_trace)}",
            })
            continue

        worker_ok = mode_summary.get("verification_all_ok", False) and stock_summary.get("verification_all_ok", False)
        for idx, (mode_row, stock_row) in enumerate(zip(mode_trace, stock_trace)):
            probe_ok, probe_detail = compare_probe(mode_row["probe"], stock_row["probe"])
            step_ok = (
                mode_row["phase"] == stock_row["phase"]
                and mode_row["step_index"] == stock_row["step_index"]
                and mode_row["verification_ok"]
                and stock_row["verification_ok"]
                and probe_ok
            )
            if not step_ok:
                worker_ok = False
            step_checks.append({
                "row_index": idx,
                "phase": mode_row["phase"],
                "step_index": mode_row["step_index"],
                "ok": step_ok,
                "probe_detail": probe_detail,
            })

        overall_ok = overall_ok and worker_ok
        worker_results.append({
            "worker": worker_dir.name,
            "ok": worker_ok,
            "mode_summary_ok": mode_summary.get("verification_all_ok", False),
            "stock_summary_ok": stock_summary.get("verification_all_ok", False),
            "step_checks": step_checks,
        })

    payload = {
        "mode": args.mode,
        "experiment": args.experiment,
        "repeat": repeat_name,
        "comparison_performed": True,
        "ok": overall_ok,
        "workers": worker_results,
    }
    validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
