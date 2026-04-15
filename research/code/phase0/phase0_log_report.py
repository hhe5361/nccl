#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PHASE0_RE = re.compile(r"PHASE0\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
WORKER_RE = re.compile(r"worker(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Phase 0 NCCL logs and render an HTML report")
    parser.add_argument("--input", required=True, help="Run root, e.g. /mnt/nfs_share/cts_experiments/phase0_... ")
    parser.add_argument("--output", required=True, help="Output HTML report path")
    parser.add_argument("--topology-file", default=None, help="Optional topology notes file to embed")
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


def parse_phase0_logs(root: Path) -> List[dict]:
    events: List[dict] = []
    for path in sorted(root.rglob("*.log")):
        worker = path.parent.name
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if "PHASE0 event=" not in line:
                    continue
                m = PHASE0_RE.search(line)
                if not m:
                    continue
                kvs = {k: parse_value(v) for k, v in KV_RE.findall(m.group(1))}
                if "event" not in kvs:
                    continue
                prefix = line.split(" PHASE0", 1)[0]
                host = prefix.split(":", 1)[0] if ":" in prefix else worker
                record = {
                    "worker": worker,
                    "host": host,
                    "file": str(path.relative_to(root)),
                    "line_no": lineno,
                    "event": kvs.pop("event"),
                }
                record.update(kvs)
                if "size" in record:
                    record["bytes"] = int(record["size"])
                elif "ctsBytes" in record:
                    record["bytes"] = int(record["ctsBytes"])
                else:
                    record["bytes"] = 0
                events.append(record)
    return events


def write_events_csv(events: List[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "events.csv"
    keys = sorted({key for event in events for key in event.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for event in events:
            writer.writerow(event)
    return csv_path


def worker_sort_key(name: str) -> Tuple[int, str]:
    m = WORKER_RE.match(name)
    if m:
        return int(m.group(1)), name
    return (10**9, name)


def rack_of(worker: str) -> str:
    m = WORKER_RE.match(worker)
    if not m:
        return "unknown"
    idx = int(m.group(1))
    return "rackA" if 1 <= idx <= 4 else "rackB"


def aggregate(events: List[dict]) -> dict:
    event_counts = Counter(event["event"] for event in events)
    worker_event_counts: Dict[str, Counter] = defaultdict(Counter)
    bytes_by_event = Counter()
    slot_counts: Dict[str, Counter] = defaultdict(Counter)
    peer_channel_counts: Counter = Counter()
    req_lifecycle: Dict[Tuple[str, int], set] = defaultdict(set)
    backlog_points: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    coll_counts: Counter = Counter()
    proto_counts: Counter = Counter()

    for seq, event in enumerate(events):
        worker = event["worker"]
        evt = event["event"]
        worker_event_counts[worker][evt] += 1
        bytes_by_event[evt] += int(event.get("bytes", 0) or 0)
        if "slot" in event:
            slot_counts[worker][int(event["slot"])] += 1
        if "peer" in event and "channel" in event:
            peer_channel_counts[(worker, event["peer"], event["channel"], evt)] += 1
        if evt.startswith("IB_") and "reqId" in event:
            req_lifecycle[(worker, int(event["reqId"]))].add(evt)
        if evt.startswith("PROXY_"):
            posted = int(event.get("posted", 0) or 0)
            received = int(event.get("received", 0) or 0)
            transmitted = int(event.get("transmitted", 0) or 0)
            done = int(event.get("done", 0) or 0)
            backlog_points[worker].append((seq, posted - done, received - done, transmitted - done))
        if "collApi" in event:
            coll_counts[str(event["collApi"])] += 1
        if "proto" in event:
            proto_counts[str(event["proto"])] += 1

    lifecycle_summary = Counter()
    for _, states in req_lifecycle.items():
        key = "+".join(sorted(states))
        lifecycle_summary[key] += 1

    return {
        "event_counts": event_counts,
        "worker_event_counts": worker_event_counts,
        "bytes_by_event": bytes_by_event,
        "slot_counts": slot_counts,
        "peer_channel_counts": peer_channel_counts,
        "req_lifecycle": lifecycle_summary,
        "backlog_points": backlog_points,
        "coll_counts": coll_counts,
        "proto_counts": proto_counts,
    }


def format_int(value: int) -> str:
    return f"{value:,}"


def bar_chart(counter: Counter, title: str, width: int = 880, height: int = 260) -> str:
    items = counter.most_common()
    if not items:
        return '<div class="empty">No data</div>'
    max_value = max(v for _, v in items) or 1
    left = 52
    right = 24
    top = 18
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 10
    bar_w = max(18, (plot_w - gap * (len(items) - 1)) / max(1, len(items)))
    svg = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    svg.append(f'<text x="{left}" y="16" class="chart-title">{html.escape(title)}</text>')
    for i, (label, value) in enumerate(items):
        x = left + i * (bar_w + gap)
        h = 0 if max_value == 0 else plot_h * (value / max_value)
        y = top + plot_h - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" class="bar"></rect>')
        svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="value">{value}</text>')
        svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 14}" text-anchor="end" transform="rotate(-35 {x + bar_w / 2:.1f},{height - 14})" class="label">{html.escape(str(label))}</text>')
    svg.append('</svg>')
    return "".join(svg)


def line_chart(points: List[Tuple[int, int, int, int]], title: str, width: int = 880, height: int = 240) -> str:
    if not points:
        return '<div class="empty">No data</div>'
    left, right, top, bottom = 52, 20, 20, 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [p[0] for p in points]
    series = {
        "posted-done": [p[1] for p in points],
        "received-done": [p[2] for p in points],
        "transmitted-done": [p[3] for p in points],
    }
    max_y = max(max(vals) for vals in series.values())
    max_y = max(max_y, 1)
    min_x, max_x = min(xs), max(xs)
    x_span = max(max_x - min_x, 1)

    def map_point(xv: int, yv: int) -> Tuple[float, float]:
        x = left + plot_w * ((xv - min_x) / x_span)
        y = top + plot_h - plot_h * (yv / max_y)
        return x, y

    colors = {
        "posted-done": "#0c5fb0",
        "received-done": "#c46a00",
        "transmitted-done": "#136245",
    }
    svg = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    svg.append(f'<text x="{left}" y="16" class="chart-title">{html.escape(title)}</text>')
    svg.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"></line>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"></line>')
    for name, vals in series.items():
        coords = [map_point(xv, yv) for xv, yv in zip(xs, vals)]
        path = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(coords))
        svg.append(f'<path d="{path}" fill="none" stroke="{colors[name]}" stroke-width="2.5"></path>')
    legend_x = width - 230
    legend_y = 24
    for i, name in enumerate(series):
        y = legend_y + i * 18
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{colors[name]}" stroke-width="3"></line>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 4}" class="legend">{html.escape(name)}</text>')
    svg.append('</svg>')
    return "".join(svg)


def table_event_counts(worker_event_counts: Dict[str, Counter]) -> str:
    workers = sorted(worker_event_counts, key=worker_sort_key)
    events = sorted({evt for counts in worker_event_counts.values() for evt in counts})
    rows = []
    header = "<tr><th>worker</th><th>rack</th>" + "".join(f"<th>{html.escape(evt)}</th>" for evt in events) + "</tr>"
    for worker in workers:
        cells = [f"<td>{html.escape(worker)}</td>", f"<td>{rack_of(worker)}</td>"]
        for evt in events:
            cells.append(f"<td>{worker_event_counts[worker][evt]}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def table_req_lifecycle(counter: Counter) -> str:
    rows = ["<tr><th>lifecycle</th><th>unique req count</th></tr>"]
    for key, value in counter.most_common():
        rows.append(f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>")
    return f"<table><thead>{rows[0]}</thead><tbody>{''.join(rows[1:])}</tbody></table>"


def table_peer_channel(counter: Counter, limit: int = 20) -> str:
    rows = ["<tr><th>worker</th><th>peer</th><th>channel</th><th>event</th><th>count</th></tr>"]
    for (worker, peer, channel, evt), count in counter.most_common(limit):
        rows.append(
            f"<tr><td>{html.escape(str(worker))}</td><td>{peer}</td><td>{channel}</td><td>{html.escape(str(evt))}</td><td>{count}</td></tr>"
        )
    return f"<table><thead>{rows[0]}</thead><tbody>{''.join(rows[1:])}</tbody></table>"


def render_html(events: List[dict], agg: dict, csv_path: Path, topology_notes: str) -> str:
    workers = sorted({event['worker'] for event in events}, key=worker_sort_key)
    backlog_blocks = []
    for worker in workers:
        backlog_blocks.append(
            f'<div class="card"><h3>{html.escape(worker)} backlog</h3>{line_chart(agg["backlog_points"].get(worker, []), f"{worker}: proxy backlog over event order")}</div>'
        )

    topology_html = ""
    if topology_notes:
        topology_html = f'<section class="section"><h2>Topology Notes</h2><div class="code">{html.escape(topology_notes)}</div></section>'

    total_bytes = sum(int(e.get('bytes', 0) or 0) for e in events)
    return f'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 0 NCCL Log Report</title>
  <style>
    :root {{ --bg:#eef2f6; --panel:#fff; --soft:#f7f9fc; --text:#17202b; --muted:#5d6775; --border:#d8dfe8; --accent:#0c5fb0; --shadow:0 14px 34px rgba(18,33,53,.08); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Segoe UI",Helvetica,Arial,sans-serif; background:linear-gradient(180deg,#f7f9fc 0%,var(--bg) 100%); color:var(--text); }}
    .page {{ width:min(1240px, calc(100% - 32px)); margin:24px auto 40px; }}
    .hero,.section {{ background:var(--panel); border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); padding:26px 28px; margin-bottom:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:16px; margin-top:16px; }}
    .card {{ background:var(--soft); border:1px solid var(--border); border-radius:14px; padding:16px; }}
    h1 {{ margin:10px 0 8px; font-size:36px; }}
    h2 {{ margin:0 0 12px; font-size:24px; }}
    h3 {{ margin:0 0 10px; font-size:18px; }}
    p,li {{ line-height:1.6; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }}
    .chip {{ border:1px solid var(--border); background:var(--soft); border-radius:999px; padding:8px 12px; font-size:13px; color:var(--muted); }}
    .chart {{ width:100%; height:auto; display:block; background:#fff; border:1px solid var(--border); border-radius:14px; }}
    .chart-title {{ font-size:14px; fill:#334155; font-weight:700; }}
    .bar {{ fill:#0c5fb0; opacity:.86; }}
    .value {{ fill:#1e293b; font-size:12px; }}
    .label {{ fill:#475569; font-size:11px; }}
    .axis {{ stroke:#cbd5e1; stroke-width:1; }}
    .legend {{ fill:#475569; font-size:12px; }}
    .code {{ background:#101925; color:#e8eef8; padding:16px 18px; border-radius:14px; overflow:auto; white-space:pre-wrap; font-family:Consolas,monospace; font-size:13px; }}
    table {{ width:100%; border-collapse:collapse; border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; font-size:14px; }}
    th {{ background:#f4f7fb; }}
    .empty {{ color:var(--muted); padding:12px 0; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="chip">Phase 0</div>
      <h1>NCCL Phase 0 Log Report</h1>
      <p>Custom NCCL 구조화 로그를 worker별로 수집해 event 분포, IB request lifecycle, slot 사용량, proxy backlog를 요약했다.</p>
      <div class="chips">
        <div class="chip">workers={len(workers)}</div>
        <div class="chip">events={len(events)}</div>
        <div class="chip">bytes={format_int(total_bytes)}</div>
        <div class="chip">events.csv={html.escape(str(csv_path.name))}</div>
      </div>
    </section>

    {topology_html}

    <section class="section">
      <h2>1. What To Read First</h2>
      <ul>
        <li><strong>event count</strong>: 어떤 phase가 가장 많이 발생했는지 본다.</li>
        <li><strong>IB request lifecycle</strong>: <code>IB_CTS_ISSUE</code> → <code>IB_SEND_POST</code> → <code>IB_SEND_COMPLETE</code>가 얼마나 완결되는지 본다.</li>
        <li><strong>proxy backlog</strong>: <code>posted-done</code>, <code>received-done</code>, <code>transmitted-done</code>가 event 순서상 얼마나 벌어지는지 본다.</li>
        <li><strong>peer/channel summary</strong>: 특정 peer/channel에 편중된 traffic이나 반복 이벤트를 본다.</li>
      </ul>
    </section>

    <section class="section">
      <h2>2. Event Volume</h2>
      <div class="grid">
        <div class="card">{bar_chart(agg['event_counts'], 'Event count by type')}</div>
        <div class="card">{bar_chart(agg['bytes_by_event'], 'Bytes by event type')}</div>
      </div>
    </section>

    <section class="section">
      <h2>3. Event Distribution By Worker</h2>
      {table_event_counts(agg['worker_event_counts'])}
    </section>

    <section class="section">
      <h2>4. IB Request Lifecycle</h2>
      <p><code>reqId</code> 기준으로 worker별 IB 이벤트를 묶었다. 이상적으로는 <code>IB_CTS_ISSUE+IB_SEND_POST+IB_SEND_COMPLETE</code> 비율이 높아야 한다.</p>
      {table_req_lifecycle(agg['req_lifecycle'])}
    </section>

    <section class="section">
      <h2>5. Proxy Backlog</h2>
      <div class="grid">{''.join(backlog_blocks)}</div>
    </section>

    <section class="section">
      <h2>6. Top Peer/Channel Activity</h2>
      {table_peer_channel(agg['peer_channel_counts'])}
    </section>

    <section class="section">
      <h2>7. Output Files</h2>
      <ul>
        <li>HTML report: {html.escape(str(csv_path.parent / 'phase0-report.html'))}</li>
        <li>Parsed CSV: {html.escape(str(csv_path))}</li>
      </ul>
    </section>
  </div>
</body>
</html>'''


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_path = Path(args.output)
    events = parse_phase0_logs(input_root)
    if not events:
        raise SystemExit(f"No PHASE0 events found under {input_root}")
    agg = aggregate(events)
    out_dir = output_path.parent
    csv_path = write_events_csv(events, out_dir)
    topology_notes = ""
    if args.topology_file:
        topology_notes = Path(args.topology_file).read_text(encoding="utf-8", errors="replace")
    html_text = render_html(events, agg, csv_path, topology_notes)
    output_path.write_text(html_text, encoding="utf-8")
    summary = {
        "workers": sorted({event["worker"] for event in events}, key=worker_sort_key),
        "event_counts": dict(agg["event_counts"]),
        "bytes_by_event": dict(agg["bytes_by_event"]),
        "req_lifecycle": dict(agg["req_lifecycle"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote report: {output_path}")
    print(f"wrote csv:    {csv_path}")
    print(f"wrote json:   {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
