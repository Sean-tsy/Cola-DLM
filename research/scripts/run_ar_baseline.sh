#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SERVER-SIDE ONLY (GPU + weights). Run one AR-baseline experiment end to end:
# probe data -> infer_ar_baseline per seed (single GPU, greedy) -> structural
# eval with the same verifiers as the Cola runs.
#
# Usage (server):
#   export BASELINE_MODEL_PATH=hf_models/baselines/Qwen2.5-1.5B
#   bash research/scripts/run_ar_baseline.sh research/configs/experiments/dyck_L64_D8_k1_qwen15b.yaml
set -euo pipefail

CONFIG="${1:?usage: run_ar_baseline.sh <experiment-config.yaml>}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"
: "${BASELINE_MODEL_PATH:?set BASELINE_MODEL_PATH (HF id or local path)}"

python -m research.scripts.gen_data "${CONFIG}" --results-root "${RESULTS_ROOT}"

RUN_ID="$(python -c "from research.experiment import load_experiment; print(load_experiment('${CONFIG}').run_id)")"
SEEDS="$(python -c "from research.experiment import load_experiment; print(' '.join(map(str, load_experiment('${CONFIG}').seeds)))")"
BASE="${RESULTS_ROOT}/${RUN_ID}"

mkdir -p "${BASE}/samples"
for SEED in ${SEEDS}; do
  echo "[run_ar_baseline] ${RUN_ID} seed=${SEED}"
  python -m research.scripts.infer_ar_baseline \
    --config "${CONFIG}" \
    --input-jsonl "${BASE}/data/seed${SEED}.jsonl" \
    --output-jsonl "${BASE}/samples/seed${SEED}.jsonl" \
    --model-path "${BASELINE_MODEL_PATH}" \
    --batch-size "${INFER_BATCH_SIZE:-16}"
done

python -m research.scripts.eval_experiment "${CONFIG}" --results-root "${RESULTS_ROOT}"
echo "[run_ar_baseline] done; archive under ${BASE}"
