#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_validation_files(mode_dir: Path):
    for worker_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
        for candidate in sorted(worker_dir.glob("*_rank_validation.json")):
            yield worker_dir.name, candidate


def build_digest_map(mode_dir: Path):
    result = {}
    for worker_name, path in iter_validation_files(mode_dir):
        row = load_json(path)
        result[worker_name] = {
            "sha256": row.get("final_sha256"),
            "tag": row.get("tag"),
            "rank": row.get("rank"),
            "path": str(path),
        }
    return result


def iter_mode_dirs(experiment_root: Path):
    for candidate in sorted(p for p in experiment_root.iterdir() if p.is_dir()):
        if candidate.name.startswith("."):
            continue
        if candidate.name.upper() == "STOCK":
            continue
        if not any(candidate.glob("*/*_rank_validation.json")):
            continue
        yield candidate


def compare_mode(stock_map, mode_map):
    workers = sorted(set(stock_map) | set(mode_map))
    matches = 0
    details = []
    for worker in workers:
        stock = stock_map.get(worker)
        mode = mode_map.get(worker)
        stock_sha = stock.get("sha256") if stock else None
        mode_sha = mode.get("sha256") if mode else None
        match = bool(stock_sha and mode_sha and stock_sha == mode_sha)
        if match:
            matches += 1
        details.append(
            {
                "worker": worker,
                "match": match,
                "stock_sha256": stock_sha,
                "mode_sha256": mode_sha,
                "stock_path": stock.get("path") if stock else None,
                "mode_path": mode.get("path") if mode else None,
            }
        )
    return {
        "workers_total": len(workers),
        "workers_matched": matches,
        "all_matched": matches == len(workers) and len(workers) > 0,
        "workers": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare B4 final output digests against STOCK")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    stock_dir = experiment_root / "STOCK"
    if not stock_dir.exists():
        raise FileNotFoundError(f"missing STOCK mode directory under {experiment_root}")

    stock_map = build_digest_map(stock_dir)
    summary = {
        "experiment_root": str(experiment_root),
        "stock_workers": sorted(stock_map.keys()),
        "modes": [],
    }

    for mode_dir in iter_mode_dirs(experiment_root):
        mode_name = mode_dir.name.upper()
        mode_map = build_digest_map(mode_dir)
        compared = compare_mode(stock_map, mode_map)
        compared["mode"] = mode_name
        summary["modes"].append(compared)

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[phase4-compare] experiment_root={experiment_root}")
    for mode in summary["modes"]:
        print(
            f"[phase4-compare] mode={mode['mode']} "
            f"matched={mode['workers_matched']}/{mode['workers_total']} "
            f"all_matched={mode['all_matched']}"
        )
    print(f"[phase4-compare] wrote {output_json}")


if __name__ == "__main__":
    main()
