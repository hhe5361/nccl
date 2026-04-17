#!/usr/bin/env python3
import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for phase0_log_report.py. Install it first, for example: pip install matplotlib"
    ) from exc

PHASE0_RE = re.compile(r"PHASE0\s+(.*)")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
WORKER_RE = re.compile(r"worker(\d+)$")
PLOT_DIRNAME = "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Phase 0 NCCL logs and render an HTML report")
    parser.add_argument("--input", required=True, help="Run root, e.g. /mnt/nfs_share/cts_experiments/phase0_...")
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
        lifecycle_summary["+".join(sorted(states))] += 1

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


def save_bar_chart(counter: Counter, title: str, path: Path, xlabel: str = "", ylabel: str = "count") -> None:
    items = counter.most_common()
    if not items:
        return
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    bars = ax.bar(labels, values, color="#0c5fb0", alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_horizontal_bar(counter: Counter, title: str, path: Path, limit: int = 20, ylabel: str = "") -> None:
    items = counter.most_common(limit)
    if not items:
        return
    labels = [str(k) for k, _ in reversed(items)]
    values = [v for _, v in reversed(items)]
    fig_h = max(4.2, 0.38 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    bars = ax.barh(labels, values, color="#136245", alpha=0.9)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_worker_event_stacked(worker_event_counts: Dict[str, Counter], path: Path) -> None:
    workers = sorted(worker_event_counts, key=worker_sort_key)
    events = sorted({evt for counts in worker_event_counts.values() for evt in counts})
    if not workers or not events:
        return
    color_cycle = [
        "#0c5fb0", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bottom = [0] * len(workers)
    for idx, evt in enumerate(events):
        values = [worker_event_counts[w][evt] for w in workers]
        ax.bar(workers, values, bottom=bottom, label=evt, color=color_cycle[idx % len(color_cycle)])
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_title("Event distribution by worker")
    ax.set_ylabel("event count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_slot_usage_chart(slot_counts: Dict[str, Counter], path: Path) -> None:
    workers = sorted(slot_counts, key=worker_sort_key)
    if not workers:
        return
    slot_ids = sorted({slot for counter in slot_counts.values() for slot in counter})
    if not slot_ids:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bottom = [0] * len(workers)
    cmap = plt.get_cmap("tab20")
    for idx, slot in enumerate(slot_ids):
        values = [slot_counts[w][slot] for w in workers]
        ax.bar(workers, values, bottom=bottom, label=f"slot {slot}", color=cmap(idx % 20))
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_title("Slot usage by worker")
    ax.set_ylabel("event count with slot field")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_backlog_chart(points: List[Tuple[int, int, int, int]], worker: str, path: Path) -> None:
    if not points:
        return
    xs = [p[0] for p in points]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(xs, [p[1] for p in points], label="posted-done", color="#0c5fb0", linewidth=1.8)
    ax.plot(xs, [p[2] for p in points], label="received-done", color="#c46a00", linewidth=1.8)
    ax.plot(xs, [p[3] for p in points], label="transmitted-done", color="#136245", linewidth=1.8)
    ax.set_title(f"{worker}: proxy backlog over event order")
    ax.set_xlabel("event order")
    ax.set_ylabel("step delta")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ensure_plots(agg: dict, out_dir: Path) -> Dict[str, str]:
    plots_dir = out_dir / PLOT_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_refs: Dict[str, str] = {}

    event_counts_path = plots_dir / "event-counts.png"
    save_bar_chart(agg["event_counts"], "Event count by type", event_counts_path, ylabel="events")
    plot_refs["event_counts"] = f"{PLOT_DIRNAME}/{event_counts_path.name}"

    bytes_by_event_path = plots_dir / "bytes-by-event.png"
    save_bar_chart(agg["bytes_by_event"], "Bytes by event type", bytes_by_event_path, ylabel="bytes")
    plot_refs["bytes_by_event"] = f"{PLOT_DIRNAME}/{bytes_by_event_path.name}"

    lifecycle_path = plots_dir / "ib-request-lifecycle.png"
    save_bar_chart(agg["req_lifecycle"], "IB request lifecycle completeness", lifecycle_path, ylabel="unique req count")
    plot_refs["req_lifecycle"] = f"{PLOT_DIRNAME}/{lifecycle_path.name}"

    worker_event_path = plots_dir / "worker-event-distribution.png"
    save_worker_event_stacked(agg["worker_event_counts"], worker_event_path)
    plot_refs["worker_event_distribution"] = f"{PLOT_DIRNAME}/{worker_event_path.name}"

    if agg["slot_counts"]:
        slot_usage_path = plots_dir / "slot-usage-by-worker.png"
        save_slot_usage_chart(agg["slot_counts"], slot_usage_path)
        plot_refs["slot_usage"] = f"{PLOT_DIRNAME}/{slot_usage_path.name}"

    if agg["peer_channel_counts"]:
        peer_counter = Counter({
            f"{worker} peer={peer} ch={channel} {evt}": count
            for (worker, peer, channel, evt), count in agg["peer_channel_counts"].items()
        })
        peer_path = plots_dir / "top-peer-channel-activity.png"
        save_horizontal_bar(peer_counter, "Top peer/channel activity", peer_path, limit=20)
        plot_refs["peer_channel"] = f"{PLOT_DIRNAME}/{peer_path.name}"

    backlog_paths = {}
    for worker, points in sorted(agg["backlog_points"].items(), key=lambda item: worker_sort_key(item[0])):
        path = plots_dir / f"backlog-{worker}.png"
        save_backlog_chart(points, worker, path)
        backlog_paths[worker] = f"{PLOT_DIRNAME}/{path.name}"
    plot_refs["backlog"] = backlog_paths
    return plot_refs


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


def img_card(title: str, src: str, note: str = "") -> str:
    note_html = f"<p>{html.escape(note)}</p>" if note else ""
    return f'<div class="card"><h3>{html.escape(title)}</h3><img class="plot" src="{html.escape(src)}" alt="{html.escape(title)}">{note_html}</div>'


def render_html(events: List[dict], agg: dict, csv_path: Path, summary_path: Path, plot_refs: Dict[str, str], topology_notes: str) -> str:
    workers = sorted({event['worker'] for event in events}, key=worker_sort_key)
    backlog_blocks = []
    for worker in workers:
        src = plot_refs.get("backlog", {}).get(worker)
        if src:
            backlog_blocks.append(img_card(f"{worker} backlog", src, "이 그래프의 x축은 시간축이 아니라 로그 event order다."))

    topology_html = ""
    if topology_notes:
        topology_html = f'<section class="section"><h2>Topology Notes</h2><div class="code">{html.escape(topology_notes)}</div></section>'

    total_bytes = sum(int(e.get('bytes', 0) or 0) for e in events)
    active_workers = ", ".join(html.escape(worker) for worker in workers)
    slot_section = ""
    if plot_refs.get("slot_usage"):
        slot_section = f'''
    <section class="section">
      <h2>6. Slot Usage</h2>
      {img_card("Slot usage by worker", plot_refs['slot_usage'])}
    </section>'''

    peer_plot = ""
    if plot_refs.get("peer_channel"):
        peer_plot = img_card("Top peer/channel activity", plot_refs["peer_channel"])

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
    .code {{ background:#101925; color:#e8eef8; padding:16px 18px; border-radius:14px; overflow:auto; white-space:pre-wrap; font-family:Consolas,monospace; font-size:13px; }}
    table {{ width:100%; border-collapse:collapse; border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; font-size:14px; }}
    th {{ background:#f4f7fb; }}
    .plot {{ width:100%; height:auto; border:1px solid var(--border); border-radius:12px; background:#fff; }}
    .warn {{ background:#fff3e2; border:1px solid #f0d8b7; border-radius:14px; padding:14px 16px; margin-top:14px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="chip">Phase 0</div>
      <h1>NCCL Phase 0 Log Report</h1>
      <p>Custom NCCL 구조화 로그를 worker별로 수집해 event 분포, IB request lifecycle, slot 사용량, proxy backlog를 PNG 차트와 표로 정리했다.</p>
      <div class="chips">
        <div class="chip">workers={len(workers)}</div>
        <div class="chip">active={active_workers}</div>
        <div class="chip">events={len(events)}</div>
        <div class="chip">bytes={format_int(total_bytes)}</div>
      </div>
      <div class="warn">
        <strong>해석 주의</strong><br>
        현재 backlog 그래프의 x축은 wall-clock time이 아니라 로그 event order다. 따라서 지속시간이나 latency로 직접 해석하면 안 되고, 단계별 상대적 queue/backpressure 패턴만 읽어야 한다.
      </div>
    </section>

    {topology_html}

    <section class="section">
      <h2>1. Event Volume</h2>
      <div class="grid">
        {img_card("Event count by type", plot_refs['event_counts'])}
        {img_card("Bytes by event type", plot_refs['bytes_by_event'])}
      </div>
    </section>

    <section class="section">
      <h2>2. Event Distribution By Worker</h2>
      {img_card("Worker event distribution", plot_refs['worker_event_distribution'])}
      {table_event_counts(agg['worker_event_counts'])}
    </section>

    <section class="section">
      <h2>3. IB Request Lifecycle</h2>
      <div class="grid">
        {img_card("IB request lifecycle completeness", plot_refs['req_lifecycle'])}
        <div class="card">
          <h3>Lifecycle table</h3>
          <p><code>reqId</code> 기준으로 worker별 IB 이벤트를 묶었다. 이상적으로는 <code>IB_CTS_ISSUE+IB_SEND_POST+IB_SEND_COMPLETE</code> 비율이 높아야 한다.</p>
          {table_req_lifecycle(agg['req_lifecycle'])}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>4. Proxy Backlog</h2>
      <div class="grid">{''.join(backlog_blocks)}</div>
    </section>

    <section class="section">
      <h2>5. Top Peer/Channel Activity</h2>
      <div class="grid">
        {peer_plot}
        <div class="card">
          <h3>Top peer/channel table</h3>
          {table_peer_channel(agg['peer_channel_counts'])}
        </div>
      </div>
    </section>

    {slot_section}

    <section class="section">
      <h2>7. Output Files</h2>
      <ul>
        <li>HTML report: {html.escape(str(summary_path.parent / 'phase0-report.html'))}</li>
        <li>Parsed CSV: {html.escape(str(csv_path))}</li>
        <li>Summary JSON: {html.escape(str(summary_path))}</li>
        <li>PNG charts: {html.escape(str(summary_path.parent / PLOT_DIRNAME))}</li>
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
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_events_csv(events, out_dir)
    plot_refs = ensure_plots(agg, out_dir)
    topology_notes = ""
    if args.topology_file:
        topology_notes = Path(args.topology_file).read_text(encoding="utf-8", errors="replace")
    summary = {
        "workers": sorted({event["worker"] for event in events}, key=worker_sort_key),
        "event_counts": dict(agg["event_counts"]),
        "bytes_by_event": dict(agg["bytes_by_event"]),
        "req_lifecycle": dict(agg["req_lifecycle"]),
        "plots": plot_refs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html_text = render_html(events, agg, csv_path, summary_path, plot_refs, topology_notes)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"wrote report: {output_path}")
    print(f"wrote csv:    {csv_path}")
    print(f"wrote json:   {summary_path}")
    print(f"wrote plots:  {out_dir / PLOT_DIRNAME}")


if __name__ == "__main__":
    main()
