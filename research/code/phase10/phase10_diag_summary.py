#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase10_reporter import load_mode_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase10 metrics")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode_dir in sorted(p for p in experiment_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        rows.append(load_mode_metrics(experiment_root, mode_dir).__dict__)

    output_json.write_text(json.dumps({"experiment_root": str(experiment_root), "modes": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[phase10-summary] wrote {output_json}")


if __name__ == "__main__":
    main()
