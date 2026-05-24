#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'USAGE'
Usage:
  run_p6_reporters.sh --input RUN_DIR [--output-dir OUT_DIR] [options]

Generates all Phase6 reports in one pass:
  1. phase6_pi_reporter.py
  2. phase6_network_overlay_reporter.py
  3. phase6_w_signal_summary_reporter.py

Options:
  --input RUN_DIR             Phase6 experiment run root. Required.
  --output-dir OUT_DIR        Report root. Default: RUN_DIR/plots_all
  --switch-log-dir DIR        Explicit switch logger directory.
  --workers LIST             Comma-separated workers for network overlay.
  --worker01-only            Shortcut for --workers worker01.
  --no-all-workers           Use --workers value instead of scanning all workers.
  --bucket-ms MS             Network overlay bucket size. Default: 1000.
  --event-window-sec SEC     Before/after event CSV window. Default: 2.
  --max-event-plots N        Maximum event-centered plots. Default: 80.
  --include-raw              Include raw network timeline plots.
  --python PYTHON            Python command. Default: python3.
  -h, --help                 Show this help.

Default behavior scans all workers and skips raw plots to keep runtime/output size sane.
USAGE
}

INPUT_DIR=""
OUTPUT_ROOT=""
SWITCH_LOG_DIR=""
WORKERS="worker01"
ALL_WORKERS=1
BUCKET_MS="1000"
EVENT_WINDOW_SEC="2"
MAX_EVENT_PLOTS="80"
SKIP_RAW=1
PYTHON_BIN=${PYTHON_BIN:-python3}

log() {
  printf '[phase6-reporters] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

elapsed_msg() {
  local start_ts=$1
  local end_ts
  end_ts=$(date +%s)
  printf '%ss' "$((end_ts - start_ts))"
}

require_file() {
  local path=$1
  local label=$2
  if [[ -s "${path}" ]]; then
    log "ok ${label}: ${path}"
  else
    log "warning ${label} missing_or_empty: ${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_DIR=${2:?--input requires a value}
      shift 2
      ;;
    --output-dir)
      OUTPUT_ROOT=${2:?--output-dir requires a value}
      shift 2
      ;;
    --switch-log-dir)
      SWITCH_LOG_DIR=${2:?--switch-log-dir requires a value}
      shift 2
      ;;
    --workers)
      WORKERS=${2:?--workers requires a value}
      ALL_WORKERS=0
      shift 2
      ;;
    --worker01-only)
      WORKERS="worker01"
      ALL_WORKERS=0
      shift
      ;;
    --no-all-workers)
      ALL_WORKERS=0
      shift
      ;;
    --bucket-ms)
      BUCKET_MS=${2:?--bucket-ms requires a value}
      shift 2
      ;;
    --event-window-sec)
      EVENT_WINDOW_SEC=${2:?--event-window-sec requires a value}
      shift 2
      ;;
    --max-event-plots)
      MAX_EVENT_PLOTS=${2:?--max-event-plots requires a value}
      shift 2
      ;;
    --include-raw)
      SKIP_RAW=0
      shift
      ;;
    --python)
      PYTHON_BIN=${2:?--python requires a value}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[phase6-reporters] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${INPUT_DIR}" ]]; then
  echo "[phase6-reporters] --input is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${INPUT_DIR%/}/plots_all"
fi

PI_OUT="${OUTPUT_ROOT%/}/phase6_plot_latest"
NETWORK_OUT="${OUTPUT_ROOT%/}/network_overlay_allworker_no_ecn_trimtop"
W_SIGNAL_OUT="${OUTPUT_ROOT%/}/plots_w_signal"

mkdir -p "${PI_OUT}" "${NETWORK_OUT}" "${W_SIGNAL_OUT}"

TOTAL_START_TS=$(date +%s)
log "input=${INPUT_DIR}"
log "output_root=${OUTPUT_ROOT}"
if [[ "${ALL_WORKERS}" -eq 1 ]]; then
  log "network_workers=all"
else
  log "network_workers=${WORKERS}"
fi
log "bucket_ms=${BUCKET_MS} event_window_sec=${EVENT_WINDOW_SEC} max_event_plots=${MAX_EVENT_PLOTS} include_raw=$((1 - SKIP_RAW))"

STEP_START_TS=$(date +%s)
log "step 1/3 start controller report -> ${PI_OUT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase6_pi_reporter.py" \
  --input "${INPUT_DIR}" \
  --output-dir "${PI_OUT}"
log "step 1/3 done elapsed=$(elapsed_msg "${STEP_START_TS}")"
require_file "${PI_OUT}/phase6_spike_report.html" "controller_html"
require_file "${PI_OUT}/phase6_summary.csv" "controller_summary_csv"

network_args=(
  --input "${INPUT_DIR}"
  --output-dir "${NETWORK_OUT}"
  --bucket-ms "${BUCKET_MS}"
  --event-window-sec "${EVENT_WINDOW_SEC}"
  --max-event-plots "${MAX_EVENT_PLOTS}"
)

if [[ "${ALL_WORKERS}" -eq 1 ]]; then
  network_args+=(--all-workers)
else
  network_args+=(--workers "${WORKERS}")
fi

if [[ "${SKIP_RAW}" -eq 1 ]]; then
  network_args+=(--skip-raw)
fi

if [[ -n "${SWITCH_LOG_DIR}" ]]; then
  network_args+=(--switch-log-dir "${SWITCH_LOG_DIR}")
fi

STEP_START_TS=$(date +%s)
log "step 2/3 start network overlay -> ${NETWORK_OUT}"
log "step 2/3 command ${PYTHON_BIN} ${SCRIPT_DIR}/phase6_network_overlay_reporter.py ${network_args[*]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase6_network_overlay_reporter.py" "${network_args[@]}"
log "step 2/3 done elapsed=$(elapsed_msg "${STEP_START_TS}")"
require_file "${NETWORK_OUT}/phase6_network_overlay_report.html" "network_html"
require_file "${NETWORK_OUT}/phase6_bin_metrics.csv" "network_bin_csv"
require_file "${NETWORK_OUT}/phase6_w_adjustment_network_windows.csv" "network_event_csv"

STEP_START_TS=$(date +%s)
log "step 3/3 start W-signal summary -> ${W_SIGNAL_OUT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/phase6_w_signal_summary_reporter.py" \
  --input "${OUTPUT_ROOT}" \
  --output-dir "${W_SIGNAL_OUT}" \
  --event-csv "${NETWORK_OUT}/phase6_w_adjustment_network_windows.csv" \
  --summary-csv "${PI_OUT}/phase6_summary.csv"
log "step 3/3 done elapsed=$(elapsed_msg "${STEP_START_TS}")"
require_file "${W_SIGNAL_OUT}/phase6_w_signal_summary.html" "w_signal_html"
require_file "${W_SIGNAL_OUT}/phase6_w_signal_event_summary.csv" "w_signal_event_csv"
require_file "${W_SIGNAL_OUT}/phase6_w_signal_controller_summary.csv" "w_signal_controller_csv"

cat <<EOF
[phase6-reporters] done elapsed=$(elapsed_msg "${TOTAL_START_TS}")
  controller: ${PI_OUT}/phase6_spike_report.html
  network:    ${NETWORK_OUT}/phase6_network_overlay_report.html
  w_signal:   ${W_SIGNAL_OUT}/phase6_w_signal_summary.html
EOF
