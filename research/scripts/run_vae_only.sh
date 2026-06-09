#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SERVER-SIDE ONLY. 9.4 H5 VAE-only GT encode->decode control.
set -euo pipefail

CONFIG="${1:?usage: run_vae_only.sh <experiment-config.yaml>}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"

: "${VAE_PATH:?set VAE_PATH}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"

python -m research.scripts.gen_data "${CONFIG}" --results-root "${RESULTS_ROOT}"

RUN_ID="$(python -c "from research.experiment import load_experiment; print(load_experiment('${CONFIG}').run_id)")"
SEEDS="$(python -c "from research.experiment import load_experiment; print(' '.join(map(str, load_experiment('${CONFIG}').seeds)))")"
BASE="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "${BASE}/samples"

for SEED in ${SEEDS}; do
  python -m research.scripts.vae_roundtrip \
    --input-jsonl "${BASE}/data/seed${SEED}.jsonl" \
    --output-jsonl "${BASE}/samples/seed${SEED}.jsonl" \
    --vae-path "${VAE_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}"
done

python -m research.scripts.eval_experiment "${CONFIG}" --results-root "${RESULTS_ROOT}"
echo "[run_vae_only] done; archive under ${BASE}"
