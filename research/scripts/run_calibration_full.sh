#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SERVER-SIDE ONLY (GPU + weights). 环节九 · 9.2 full official benchmark
# calibration over the eight tasks used by scripts/run_benchmark.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

RUN_ID="${1:-calib_official8}"
TOLERANCE="${TOLERANCE:-3.0}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"
NUM_GPUS="${NUM_GPUS:-8}"
TASKS="${TASKS:-lambada obqa hellaswag mmlu race siqa squad story_cloze}"

: "${DIT_PATH:?set DIT_PATH (e.g. hf_models/cola_dlm/cola_dit)}"
: "${VAE_PATH:?set VAE_PATH}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"

BASE="${RESULTS_ROOT}/${RUN_ID}"
TASK_DATA_DIR="${BASE}/generate_task_data"
OUTPUT_DIR="${BASE}/tasks_${RUN_ID}"
mkdir -p "${TASK_DATA_DIR}" "${OUTPUT_DIR}"

for TASK in ${TASKS}; do
  REF="eval_output/tasks_default/${TASK}.jsonl"
  if [[ ! -f "${REF}" ]]; then
    echo "[run_calibration_full] ERROR: missing committed reference ${REF}" >&2
    exit 1
  fi
  python -m research.scripts.prep_calibration \
    --task "${TASK}" \
    --reference "${REF}" \
    --output "${TASK_DATA_DIR}/${TASK}.jsonl"
done

TASKS="${TASKS}" \
TASK_DATA_DIR="${TASK_DATA_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DIT_PATH="${DIT_PATH}" \
VAE_PATH="${VAE_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
NUM_GPUS="${NUM_GPUS}" \
MAX_SAMPLES="${MAX_SAMPLES:-1000}" \
  bash scripts/run_benchmark.sh

python scripts/acc_calc.py "${BASE}" "${BASE}/accuracy_summary.csv"

python - "${BASE}/accuracy_summary.csv" "eval_output/accuracy_summary.csv" "${TOLERANCE}" "${BASE}/calibration_verdict.json" <<'PY'
import csv
import json
import sys

measured_csv, reference_csv, tol, verdict_path = sys.argv[1:5]
tol = float(tol)


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    value_cols = [c for c in rows[0] if c != "task"]
    col = value_cols[0]
    return {row["task"]: float(row[col]) for row in rows if row.get("task") and row.get(col)}


measured = read(measured_csv)
reference = read(reference_csv)
tasks = [t for t in reference if t != "tasks_average"]
details = {}
ok = True
for task in tasks + ["tasks_average"]:
    if task not in measured or task not in reference:
        details[task] = {"status": "missing", "measured": measured.get(task), "reference": reference.get(task)}
        ok = False
        continue
    delta = abs(measured[task] - reference[task])
    task_ok = delta <= tol
    details[task] = {
        "status": "PASS" if task_ok else "FAIL",
        "measured": measured[task],
        "reference": reference[task],
        "abs_delta": delta,
        "tolerance": tol,
    }
    ok = ok and task_ok

payload = {"verdict": "PASS" if ok else "FAIL", "tasks": details}
with open(verdict_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
sys.exit(0 if ok else 2)
PY
