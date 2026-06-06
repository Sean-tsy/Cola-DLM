#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SERVER-SIDE ONLY (GPU + weights). 环节九 · 9.2 复现校准.
#
# Reproduces the committed open-source LAMBADA number (eval_output/
# accuracy_summary.csv == 50.80) with the official harness, to prove our
# inference接入 / sampling / scoring pipeline口径 is trustworthy before the H1+
# diagnosis. NOT run locally (no torch, no weights locally).
#
# Why lambada: its prompt template is the question verbatim, so we can rebuild
# the (gitignored, GFW-unreachable) input JSONL offline from the committed
# reference output — a bit-for-bit identical, dataset-download-free target.
#
# Pipeline: prep_calibration -> upstream run_benchmark.sh (lambada) ->
#           acc_calc.py -> compare measured vs reference within tolerance.
#
# Usage (server, run detached via remote/submit.sh):
#   export DIT_PATH=hf_models/cola_dlm/cola_dit \
#          VAE_PATH=hf_models/cola_dlm/cola_vae \
#          TOKENIZER_PATH=hf_models/tokenizer.json
#   NUM_GPUS=8 bash research/scripts/run_calibration.sh calib_lambada
#
# Optional env: TOLERANCE (default 3.0 abs %), MAX_SAMPLES, NUM_GPUS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

RUN_ID="${1:-calib_lambada}"
TASK="lambada"
TOLERANCE="${TOLERANCE:-3.0}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"
NUM_GPUS="${NUM_GPUS:-8}"

: "${DIT_PATH:?set DIT_PATH (e.g. hf_models/cola_dlm/cola_dit)}"
: "${VAE_PATH:?set VAE_PATH}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"

REFERENCE_OUT="eval_output/tasks_default/${TASK}.jsonl"
REFERENCE_CSV="eval_output/accuracy_summary.csv"
if [[ ! -f "${REFERENCE_OUT}" ]]; then
  echo "[run_calibration] ERROR: committed reference output not found: ${REFERENCE_OUT}" >&2
  exit 1
fi

BASE="${RESULTS_ROOT}/${RUN_ID}"
TASK_DATA_DIR="${BASE}/generate_task_data"
OUTPUT_DIR="${BASE}/tasks_${RUN_ID}"
mkdir -p "${TASK_DATA_DIR}" "${OUTPUT_DIR}"

# 1) Reconstruct the calibration input offline from the committed reference.
python -m research.scripts.prep_calibration \
  --task "${TASK}" \
  --reference "${REFERENCE_OUT}" \
  --output "${TASK_DATA_DIR}/${TASK}.jsonl"

# 2) Run the OFFICIAL harness (unchanged) on lambada with default口径
#    (timestep_num=16, temperature=0.0 greedy, noise seed 66, eos/im_end set).
TASKS="${TASK}" \
TASK_DATA_DIR="${TASK_DATA_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DIT_PATH="${DIT_PATH}" \
VAE_PATH="${VAE_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
NUM_GPUS="${NUM_GPUS}" \
MAX_SAMPLES="${MAX_SAMPLES:-1000}" \
  bash scripts/run_benchmark.sh

# 3) Score with the OFFICIAL scorer; acc_calc scans <root>/tasks_<alias>/<task>.jsonl.
python scripts/acc_calc.py "${BASE}" "${BASE}/accuracy_summary.csv"

# 4) Compare measured vs the committed reference within tolerance -> PASS/FAIL.
python - "$TASK" "$RUN_ID" "$BASE/accuracy_summary.csv" "$REFERENCE_CSV" "$TOLERANCE" <<'PY'
import csv, sys

task, run_id, measured_csv, reference_csv, tol = sys.argv[1:6]
tol = float(tol)


def read_acc(path, task):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("task") or "").strip().lower()
            if key == task:
                # take the first non-empty numeric column after 'task'
                for k, v in row.items():
                    if k == "task" or v in (None, ""):
                        continue
                    try:
                        return float(v)
                    except ValueError:
                        continue
    return None


measured = read_acc(measured_csv, task)
reference = read_acc(reference_csv, task)
if measured is None or reference is None:
    print(f"[run_calibration] ERROR: could not read {task} acc (measured={measured}, reference={reference})")
    sys.exit(1)

delta = abs(measured - reference)
verdict = "PASS" if delta <= tol else "FAIL"
print("=============================================")
print(f" Calibration {run_id} | task={task}")
print(f"   measured  : {measured:.2f}%")
print(f"   reference : {reference:.2f}% (eval_output/accuracy_summary.csv)")
print(f"   |delta|   : {delta:.2f}  (tolerance {tol:.2f})")
print(f"   verdict   : {verdict}")
print("=============================================")
sys.exit(0 if verdict == "PASS" else 2)
PY
