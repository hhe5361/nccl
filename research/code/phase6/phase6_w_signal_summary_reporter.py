#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase6 W signal summary tables.")
    parser.add_argument("--input", required=True, type=Path, help="Phase6 run root or plots root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=None,
        help="Optional phase6_w_adjustment_network_windows.csv path",
    )
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional phase6_summary.csv path")
    return parser.parse_args()


def to_float(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_event_csv(input_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        input_root / "network_overlay_allworker_no_ecn_trimtop" / "phase6_w_adjustment_network_windows.csv",
        input_root / "plots" / "network_overlay_allworker_no_ecn_trimtop" / "phase6_w_adjustment_network_windows.csv",
        input_root / "phase6_w_adjustment_network_windows.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(input_root.rglob("network_overlay_allworker_no_ecn_trimtop/phase6_w_adjustment_network_windows.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("phase6_w_adjustment_network_windows.csv not found")


def resolve_summary_csv(input_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        input_root / "phase6_plot_latest" / "phase6_summary.csv",
        input_root / "plots" / "phase6_plot_latest" / "phase6_summary.csv",
        input_root / "phase6_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(input_root.rglob("phase6_plot_latest/phase6_summary.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError("phase6_summary.csv not found")


def summarize_events(rows: list[dict]) -> list[dict]:
    out = []
    repeats = sorted({row.get("repeat", "") for row in rows if row.get("repeat")})
    for repeat in repeats:
        group = [row for row in rows if row.get("repeat") == repeat and row.get("action") == "decrease"]
        if not group:
            continue
        workers = sorted({row.get("worker", "") for row in group if row.get("worker")})
        pfc_around = [to_float(row.get("pfc_before")) + to_float(row.get("pfc_after")) for row in group]
        pfc_before = [to_float(row.get("pfc_before")) for row in group]
        pfc_after = [to_float(row.get("pfc_after")) for row in group]
        pfc_delta = [after - before for before, after in zip(pfc_before, pfc_after)]
        events_with_pfc = sum(1 for value in pfc_around if value > 0.0)
        events_pfc_reduced = sum(1 for before, after in zip(pfc_before, pfc_after) if before > 0.0 and after < before)
        out.append(
            {
                "repeat": repeat,
                "w_decrease_events": len(group),
                "workers_with_w_decrease_count": len(workers),
                "events_with_pfc_nearby": events_with_pfc,
                "pfc_nearby_ratio": events_with_pfc / len(group),
                "events_pfc_reduced_after": events_pfc_reduced,
                "pfc_reduction_ratio": events_pfc_reduced / len(group),
                "median_pfc_before": percentile(pfc_before, 50),
                "median_pfc_after": percentile(pfc_after, 50),
                "median_pfc_after_minus_before": percentile(pfc_delta, 50),
                "median_pfc_around_w_decrease": percentile(pfc_around, 50),
                "p95_pfc_around_w_decrease": percentile(pfc_around, 95),
                "max_e_at_decrease": max(to_float(row.get("e")) for row in group),
            }
        )
    return out


def summarize_controller(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                "repeat": row.get("repeat", ""),
                "mode": row.get("mode", ""),
                "ctrl_intervals": int(to_float(row.get("epochs"))),
                "w_decrease_events": int(to_float(row.get("decrease"))),
                "w_min": to_float(row.get("w_min")),
                "w_max": to_float(row.get("w_max")),
                "delay_mean_p99_ms": to_float(row.get("delay_mean_p99_ms")),
                "u_max": to_float(row.get("u_max")),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def table_html(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"<h2>{html.escape(title)}</h2><p>No rows.</p>"
    fields = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{fmt(row.get(field, ''))}</td>" for field in fields) + "</tr>")
    return f"<h2>{html.escape(title)}</h2><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, event_csv: Path, summary_csv: Path, event_rows: list[dict], controller_rows: list[dict]) -> None:
    path.write_text(
        f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Phase6 W Signal Summary</title>
  <style>
    body {{ font-family: Arial, "Malgun Gothic", sans-serif; margin: 28px; color: #172033; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7deea; padding: 8px 9px; text-align: left; }}
    th {{ background: #f1f4f9; }}
    .note {{ background: #f8fafc; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0 24px; }}
  </style>
</head>
<body>
  <h1>Phase6 W Signal Summary</h1>
  <p>event_csv: <code>{html.escape(str(event_csv))}</code></p>
  <p>controller_csv: <code>{html.escape(str(summary_csv))}</code></p>
  <div class="note">
    W signal 해석: W decrease는 감지된 혼잡성 신호에 대해 W가 제한적으로 낮아진 이벤트이다.
    이 실험에서는 W가 급격히 줄어든 것이 아니라, 감지 구간에서 주로 8.0에서 7.6 수준으로 조정되었다.
    <code>workers_with_w_decrease_count</code>는 전체 worker 수가 아니라 W decrease를 발생시킨 worker 수이다.
    <code>pfc_nearby_ratio</code>는 W decrease 전후 window에서 PFC 증가가 관측된 이벤트 비율이다.
    <code>pfc_reduction_ratio</code>는 W decrease 직후 window의 PFC 증가량이 직전 window보다 작아진 이벤트 비율이다.
  </div>
  {table_html("W Decrease Event vs PFC Window Summary", event_rows)}
  {table_html("Controller Operation Summary", controller_rows)}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    event_csv = resolve_event_csv(input_root, args.event_csv)
    summary_csv = resolve_summary_csv(input_root, args.summary_csv)
    event_rows = summarize_events(read_csv(event_csv))
    controller_rows = summarize_controller(read_csv(summary_csv))

    write_csv(output_dir / "phase6_w_signal_event_summary.csv", event_rows)
    write_csv(output_dir / "phase6_w_signal_controller_summary.csv", controller_rows)
    html_path = output_dir / "phase6_w_signal_summary.html"
    write_html(html_path, event_csv, summary_csv, event_rows, controller_rows)
    print(f"[phase6-w-signal] event_rows={len(event_rows)} controller_rows={len(controller_rows)} html={html_path}")


if __name__ == "__main__":
    main()
