#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare STOCK vs B3 semantic equivalence outputs")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_rank_rows(mode_root: Path) -> Dict[Tuple[str, int], dict]:
    rows: Dict[Tuple[str, int], dict] = {}
    for path in sorted(mode_root.glob("worker*/" + mode_root.name + "_rank_validation.jsonl")):
        for row in load_jsonl(path):
            rows[(str(row["worker"]), int(row["step"]))] = row
    return rows


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_dir = experiment_root / "STOCK"
    b3_dir = experiment_root / "B3"
    if not stock_dir.exists() or not b3_dir.exists():
        raise FileNotFoundError(f"expected STOCK and B3 directories under {experiment_root}")

    stock_summary = load_json(next(stock_dir.glob("*_summary.json")))
    b3_summary = load_json(next(b3_dir.glob("*_summary.json")))
    stock_rows = find_rank_rows(stock_dir)
    b3_rows = find_rank_rows(b3_dir)

    keys = sorted(set(stock_rows) | set(b3_rows))
    comparisons = []
    for key in keys:
        s = stock_rows.get(key)
        b = b3_rows.get(key)
        comparisons.append(
            {
                "worker": key[0],
                "step": key[1],
                "present_in_stock": s is not None,
                "present_in_b3": b is not None,
                "stock_valid": bool(s["valid"]) if s else False,
                "b3_valid": bool(b["valid"]) if b else False,
                "sha_match": bool(s and b and s["actual_sha256"] == b["actual_sha256"]),
                "stock_sha256": s["actual_sha256"] if s else "",
                "b3_sha256": b["actual_sha256"] if b else "",
                "stock_head": s.get("actual_head", []) if s else [],
                "b3_head": b.get("actual_head", []) if b else [],
            }
        )

    mismatches = [
        row for row in comparisons
        if not (row["present_in_stock"] and row["present_in_b3"] and row["stock_valid"] and row["b3_valid"] and row["sha_match"])
    ]

    worker_rows: Dict[str, List[dict]] = {}
    for row in comparisons:
        worker_rows.setdefault(row["worker"], []).append(row)

    worker_summary_rows = []
    for worker, rows in sorted(worker_rows.items()):
        total = len(rows)
        match = sum(1 for row in rows if row["sha_match"] and row["stock_valid"] and row["b3_valid"])
        worker_summary_rows.append(
            {
                "worker": worker,
                "steps": total,
                "matched_steps": match,
                "match_rate_pct": (100.0 * match / total) if total else 0.0,
            }
        )

    summary = {
        "experiment": experiment_root.name,
        "collective": stock_summary.get("collective", ""),
        "payload_mb": stock_summary.get("payload_mb", 0.0),
        "stock_all_steps_valid": bool(stock_summary.get("all_steps_valid", False)),
        "b3_all_steps_valid": bool(b3_summary.get("all_steps_valid", False)),
        "stock_step_ms_avg": float(stock_summary.get("step_ms_avg", 0.0)),
        "b3_step_ms_avg": float(b3_summary.get("step_ms_avg", 0.0)),
        "comparison_pairs": len(comparisons),
        "mismatch_count": len(mismatches),
        "all_digest_match": len(mismatches) == 0,
        "worker_summary": worker_summary_rows,
        "mismatches": mismatches[:32],
    }

    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    top_rows = [
        ["experiment", experiment_root.name],
        ["collective", str(summary["collective"])],
        ["payload_mb", f'{float(summary["payload_mb"]):.3f}'],
        ["stock_all_steps_valid", str(summary["stock_all_steps_valid"])],
        ["b3_all_steps_valid", str(summary["b3_all_steps_valid"])],
        ["comparison_pairs", str(summary["comparison_pairs"])],
        ["mismatch_count", str(summary["mismatch_count"])],
        ["all_digest_match", str(summary["all_digest_match"])],
        ["stock_step_ms_avg", f'{float(summary["stock_step_ms_avg"]):.3f}'],
        ["b3_step_ms_avg", f'{float(summary["b3_step_ms_avg"]):.3f}'],
    ]

    worker_table = [
        [row["worker"], str(row["steps"]), str(row["matched_steps"]), f'{row["match_rate_pct"]:.2f}']
        for row in worker_summary_rows
    ]
    mismatch_table = [
        [
            row["worker"],
            str(row["step"]),
            str(row["present_in_stock"]),
            str(row["present_in_b3"]),
            str(row["stock_valid"]),
            str(row["b3_valid"]),
            str(row["sha_match"]),
            row["stock_sha256"][:12],
            row["b3_sha256"][:12],
        ]
        for row in summary["mismatches"]
    ]

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Appendix1 Semantic Equivalence Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-bottom: 10px; }}
    .section {{ margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; font-size: 14px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    .pass {{ color: #0a7a35; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Appendix1 Semantic Equivalence Report</h1>
  <div class="section">
    <p class="{'pass' if summary['all_digest_match'] else 'fail'}">
      Result: {'PASS' if summary['all_digest_match'] else 'FAIL'}
    </p>
    {render_table(["field", "value"], top_rows)}
  </div>
  <div class="section">
    <h2>Worker Match Summary</h2>
    {render_table(["worker", "steps", "matched_steps", "match_rate_pct"], worker_table)}
  </div>
  <div class="section">
    <h2>First Mismatches</h2>
    {render_table(["worker", "step", "present_in_stock", "present_in_b3", "stock_valid", "b3_valid", "sha_match", "stock_sha256", "b3_sha256"], mismatch_table) if mismatch_table else '<p class="pass">No mismatches.</p>'}
  </div>
</body>
</html>
"""
    (output_dir / "comparison_report.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
