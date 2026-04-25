#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUNNER=${RUNNER:-${SCRIPT_DIR}/run_b3_matrix.sh}

OUTPUT_ARG=""
RUN_ID_PREFIX=${RUN_ID_PREFIX:-${RUN_ID:-phase3_b3_threshold_$(date +%y%m%d_%H%M%S)}}
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}

usage() {
  cat <<'EOF'
Usage:
  bash research/code/phase3/run_b3_threshold_matrix.sh --output <path-or-dir> [--run-id-prefix <prefix>]

The wrapper runs run_b3_matrix.sh four times with these B3 threshold presets:
  a_current
  b_conservative
  c_occupancy
  d_lag

If --output ends with .json, that exact file is written.
Otherwise --output is treated as a directory and b3_threshold_matrix_runs.json is written inside it.
EOF
}

while (($# > 0)); do
  case "$1" in
    --output)
      OUTPUT_ARG=${2:-}
      shift 2
      ;;
    --run-id-prefix)
      RUN_ID_PREFIX=${2:-}
      shift 2
      ;;
    --stop-on-failure)
      CONTINUE_ON_FAILURE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[threshold-matrix] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_ARG}" ]]; then
  echo "[threshold-matrix] --output is required." >&2
  usage >&2
  exit 2
fi

if [[ "${OUTPUT_ARG}" == *.json ]]; then
  OUTPUT_JSON=${OUTPUT_ARG}
  OUTPUT_DIR=$(dirname "${OUTPUT_JSON}")
else
  OUTPUT_DIR=${OUTPUT_ARG}
  OUTPUT_JSON="${OUTPUT_DIR}/b3_threshold_matrix_runs.json"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_JSON=$(cd "$(dirname "${OUTPUT_JSON}")" && pwd)/$(basename "${OUTPUT_JSON}")
STATE_DIR="${OUTPUT_JSON}.records"
mkdir -p "${STATE_DIR}"

PRESET_IDS=(a_current b_conservative c_occupancy d_lag)
PRESET_LABELS=("A current baseline" "B conservative runtime activation" "C occupancy-oriented" "D lag-oriented")
PRESET_WARMUP=(4 4 4 4)
PRESET_HI=(2 3 2 2)
PRESET_LO=(8 12 16 16)
PRESET_OCC=(90 80 75 90)
PRESET_LAG=(100 50 100 25)
PRESET_DELAY=(125 125 125 125)

json_record_path() {
  local idx=$1
  local preset_id=$2
  printf '%s/%02d_%s.json' "${STATE_DIR}" "$((idx + 1))" "${preset_id}"
}

read_env_value() {
  local path=$1
  local key=$2
  if [[ -f "${path}" ]]; then
    awk -F= -v k="${key}" '$1 == k { print substr($0, length(k) + 2); exit }' "${path}" 2>/dev/null || true
  fi
}

write_run_record() {
  local record_path=$1
  local idx=$2
  local preset_id=$3
  local preset_label=$4
  local run_id=$5
  local matrix_root=$6
  local rc=$7
  local started_at=$8
  local ended_at=$9
  local switch_env="${matrix_root}/switch_logger.env"
  local switch_run_id switch_dir switch_local_dir switch_markers_jsonl
  switch_run_id=$(read_env_value "${switch_env}" "SWITCH_LOG_RUN_ID")
  switch_dir=$(read_env_value "${switch_env}" "SWITCH_LOG_DIR")
  switch_local_dir=$(read_env_value "${switch_env}" "SWITCH_LOG_LOCAL_DIR")
  switch_markers_jsonl=$(read_env_value "${switch_env}" "SWITCH_LOG_MARKERS_JSONL")

  python3 - "$record_path" <<PY
import json
import sys

record = {
    "index": int(${idx}),
    "preset_id": ${preset_id@Q},
    "preset_label": ${preset_label@Q},
    "run_id": ${run_id@Q},
    "matrix_root": ${matrix_root@Q},
    "return_code": int(${rc}),
    "status": "success" if int(${rc}) == 0 else "failed",
    "started_at_unix": int(${started_at}),
    "ended_at_unix": int(${ended_at}),
    "thresholds": {
        "NCCL_PHASE3_WARMUP_INTERVALS": int(${NCCL_PHASE3_WARMUP_INTERVALS}),
        "NCCL_PHASE3_HI_INTERVALS": int(${NCCL_PHASE3_HI_INTERVALS}),
        "NCCL_PHASE3_LO_INTERVALS": int(${NCCL_PHASE3_LO_INTERVALS}),
        "NCCL_PHASE3_OCC_RATIO_HIGH_PCT": int(${NCCL_PHASE3_OCC_RATIO_HIGH_PCT}),
        "NCCL_PHASE3_LAG_RATIO_HIGH_PCT": int(${NCCL_PHASE3_LAG_RATIO_HIGH_PCT}),
        "NCCL_PHASE3_DELAY_RATIO_HIGH_PCT": int(${NCCL_PHASE3_DELAY_RATIO_HIGH_PCT}),
    },
    "switch": {
        "switch_log_run_id": ${switch_run_id@Q},
        "switch_log_dir": ${switch_dir@Q},
        "switch_log_local_dir": ${switch_local_dir@Q},
        "switch_log_markers_jsonl": ${switch_markers_jsonl@Q},
    },
}
path = sys.argv[1]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
}

write_output_json() {
  local final_status=$1
  local final_rc=$2
  python3 - "${OUTPUT_JSON}" "${STATE_DIR}" "${RUN_ID_PREFIX}" "${final_status}" "${final_rc}" <<'PY'
import json
import socket
import sys
from pathlib import Path

output_json = Path(sys.argv[1])
state_dir = Path(sys.argv[2])
run_id_prefix = sys.argv[3]
status = sys.argv[4]
return_code = int(sys.argv[5])

runs = []
for path in sorted(state_dir.glob("*.json")):
    with path.open(encoding="utf-8") as handle:
        runs.append(json.load(handle))

payload = {
    "kind": "phase3_b3_threshold_matrix",
    "run_id_prefix": run_id_prefix,
    "host": socket.gethostname(),
    "status": status,
    "return_code": return_code,
    "runs_total": 4,
    "runs_recorded": len(runs),
    "runs_success": sum(1 for item in runs if item.get("return_code") == 0),
    "runs_failed": sum(1 for item in runs if item.get("return_code") != 0),
    "runs": runs,
}
output_json.parent.mkdir(parents=True, exist_ok=True)
with output_json.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

FINAL_STATUS=running
FINAL_RC=0
finish() {
  local rc=$?
  if [[ "${FINAL_STATUS}" == "running" ]]; then
    FINAL_STATUS=interrupted
    FINAL_RC=${rc}
  fi
  write_output_json "${FINAL_STATUS}" "${FINAL_RC}" || true
  return "${rc}"
}
trap finish EXIT INT TERM

echo "[threshold-matrix] output_json=${OUTPUT_JSON}"
echo "[threshold-matrix] run_id_prefix=${RUN_ID_PREFIX}"
echo "[threshold-matrix] runner=${RUNNER}"

overall_rc=0
for idx in "${!PRESET_IDS[@]}"; do
  preset_id=${PRESET_IDS[$idx]}
  preset_label=${PRESET_LABELS[$idx]}
  run_id="${RUN_ID_PREFIX}_${preset_id}"
  log_root_base=${LOG_ROOT_BASE:-/mnt/nfs_share/cts_experiments}
  matrix_root="${MATRIX_ROOT_BASE:-${log_root_base}}/${run_id}"

  export RUN_ID="${run_id}"
  export MATRIX_ROOT="${matrix_root}"
  export RUN_MODES=${RUN_MODES:-stock,b3}
  export NCCL_PHASE3_WARMUP_INTERVALS=${PRESET_WARMUP[$idx]}
  export NCCL_PHASE3_HI_INTERVALS=${PRESET_HI[$idx]}
  export NCCL_PHASE3_LO_INTERVALS=${PRESET_LO[$idx]}
  export NCCL_PHASE3_OCC_RATIO_HIGH_PCT=${PRESET_OCC[$idx]}
  export NCCL_PHASE3_LAG_RATIO_HIGH_PCT=${PRESET_LAG[$idx]}
  export NCCL_PHASE3_DELAY_RATIO_HIGH_PCT=${PRESET_DELAY[$idx]}

  echo "[threshold-matrix] ============================================================"
  echo "[threshold-matrix] start preset=${preset_id} run_id=${RUN_ID}"
  echo "[threshold-matrix] thresholds warmup=${NCCL_PHASE3_WARMUP_INTERVALS} hi=${NCCL_PHASE3_HI_INTERVALS} lo=${NCCL_PHASE3_LO_INTERVALS} occ=${NCCL_PHASE3_OCC_RATIO_HIGH_PCT} lag=${NCCL_PHASE3_LAG_RATIO_HIGH_PCT} delay=${NCCL_PHASE3_DELAY_RATIO_HIGH_PCT}"

  started_at=$(date +%s)
  set +e
  bash "${RUNNER}"
  rc=$?
  set -e
  ended_at=$(date +%s)

  write_run_record "$(json_record_path "${idx}" "${preset_id}")" "${idx}" "${preset_id}" "${preset_label}" "${RUN_ID}" "${MATRIX_ROOT}" "${rc}" "${started_at}" "${ended_at}"

  if (( rc != 0 )); then
    overall_rc=1
    echo "[threshold-matrix] failed preset=${preset_id} rc=${rc}" >&2
    write_output_json "running" "${overall_rc}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      break
    fi
  else
    echo "[threshold-matrix] complete preset=${preset_id}"
    write_output_json "running" "${overall_rc}"
  fi
done

FINAL_RC=${overall_rc}
if (( overall_rc == 0 )); then
  FINAL_STATUS=success
else
  FINAL_STATUS=failed
fi
write_output_json "${FINAL_STATUS}" "${FINAL_RC}"
echo "[threshold-matrix] wrote ${OUTPUT_JSON}"
exit "${overall_rc}"
