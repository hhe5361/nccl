#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR=""
RUNNER_TEMPLATE=""
STEPS=50
REPEATS=5
W_VALUES=("1" "1.5" "2" "2.5" "3" "3.5" "4" "4.5" "5" "5.5" "6" "6.5" "7" "7.5" "8")

usage() {
  cat <<'EOF'
Usage:
  run_pr_phase1_prob_w_matrix.sh --output-dir DIR --runner-template TEMPLATE

Required:
  --output-dir DIR
      Root directory for experiment outputs.

  --runner-template TEMPLATE
      Command template to execute one experiment trial.
      Supported placeholders:
        {COLL}     collective name (allreduce, alltoall)
        {ALGO}     algorithm name (ring, tree, auto)
        {LABEL}    experiment label (allreduce_ring, allreduce_tree, alltoall_auto)
        {STEPS}    fixed step count, default 50
        {REPEAT}   repeat index starting from 1
        {W}        configured probabilistic inflight W
        {OUTDIR}   per-run output directory

Example:
  run_pr_phase1_prob_w_matrix.sh \
    --output-dir /tmp/pr_phase1_runs \
    --runner-template 'bash run_one.sh --coll {COLL} --algo {ALGO} --steps {STEPS} --out {OUTDIR}'

Behavior:
  - Sweeps W in 0.5 increments from 1.0 to 8.0
  - Runs 5 repeats per configuration
  - Covers:
      allreduce ring
      allreduce tree
      alltoall auto
  - Exports:
      NCCL_PHASE1_INFLIGHT_W
      NCCL_PHASE1_INFLIGHT_LOG=1
      NCCL_DEBUG=INFO              (if unset)
      NCCL_DEBUG_SUBSYS=NET        (if unset)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --runner-template)
      RUNNER_TEMPLATE="${2:-}"
      shift 2
      ;;
    --steps)
      STEPS="${2:-}"
      shift 2
      ;;
    --repeats)
      REPEATS="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" || -z "${RUNNER_TEMPLATE}" ]]; then
  usage
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export NCCL_PHASE1_INFLIGHT_LOG="${NCCL_PHASE1_INFLIGHT_LOG:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-NET}"

MATRIX_JSON="${OUTPUT_DIR}/pr_phase1_prob_w_matrix.json"
python3 - "${MATRIX_JSON}" "${REPEATS}" "${STEPS}" "${W_VALUES[@]}" <<'PY'
import json
import sys

path = sys.argv[1]
repeats = int(sys.argv[2])
steps = int(sys.argv[3])
w_values = sys.argv[4:]

matrix = {
    "collectives": [
        {"coll": "allreduce", "algo": "ring", "label": "allreduce_ring"},
        {"coll": "allreduce", "algo": "tree", "label": "allreduce_tree"},
        {"coll": "alltoall", "algo": "auto", "label": "alltoall_auto"},
    ],
    "steps": steps,
    "repeats": repeats,
    "w_values": w_values,
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(matrix, f, indent=2)
PY

replace_tokens() {
  local template="$1"
  local coll="$2"
  local algo="$3"
  local label="$4"
  local steps="$5"
  local repeat="$6"
  local w="$7"
  local outdir="$8"

  template="${template//\{COLL\}/${coll}}"
  template="${template//\{ALGO\}/${algo}}"
  template="${template//\{LABEL\}/${label}}"
  template="${template//\{STEPS\}/${steps}}"
  template="${template//\{REPEAT\}/${repeat}}"
  template="${template//\{W\}/${w}}"
  template="${template//\{OUTDIR\}/${outdir}}"
  printf '%s' "${template}"
}

run_case() {
  local coll="$1"
  local algo="$2"
  local label="$3"

  for w in "${W_VALUES[@]}"; do
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
      local run_name="${label}_w${w//./p}_r$(printf '%02d' "${repeat}")"
      local outdir="${OUTPUT_DIR}/${run_name}"
      mkdir -p "${outdir}"

      export NCCL_PHASE1_INFLIGHT_W="${w}"
      local cmd
      cmd="$(replace_tokens "${RUNNER_TEMPLATE}" "${coll}" "${algo}" "${label}" "${STEPS}" "${repeat}" "${w}" "${outdir}")"

      echo "[pr_phase1] label=${label} coll=${coll} algo=${algo} W=${w} repeat=${repeat} outdir=${outdir}"
      bash -lc "${cmd}" > "${outdir}/stdout.log" 2> "${outdir}/stderr.log"
    done
  done
}

run_case "allreduce" "ring" "allreduce_ring"
run_case "allreduce" "tree" "allreduce_tree"
run_case "alltoall" "auto" "alltoall_auto"

echo "[pr_phase1] completed output_dir=${OUTPUT_DIR}"
