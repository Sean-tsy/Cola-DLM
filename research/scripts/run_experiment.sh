#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SERVER-SIDE ONLY (GPU + weights). Skeleton job script: runs one experiment
# (one config) end to end on the server. NOT run locally. Fill in the marked
# TODOs against your cluster's launcher / checkpoint layout.
#
# Convention: one experiment == one config + one git commit. This script must
# be invoked at a clean, committed git state so the manifest's git_sha is
# meaningful and the run is reproducible/rollback-able.
#
# Usage (server):
#   export DIT_PATH=... VAE_PATH=... TOKENIZER_PATH=...
#   bash research/scripts/run_experiment.sh research/configs/experiments/dyck_L32_D6_k2_blk4.yaml
set -euo pipefail

CONFIG="${1:?usage: run_experiment.sh <experiment-config.yaml>}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"

: "${DIT_PATH:?set DIT_PATH (checkpoint placeholder ${DIT_PATH})}"
: "${VAE_PATH:?set VAE_PATH}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"

# 1) Materialize probe data + manifest (CPU; reuses the local generator).
python -m research.scripts.gen_data "${CONFIG}" --results-root "${RESULTS_ROOT}"

RUN_ID="$(python -c "from research.experiment import load_experiment; print(load_experiment('${CONFIG}').run_id)")"
SEEDS="$(python -c "from research.experiment import load_experiment; print(' '.join(map(str, load_experiment('${CONFIG}').seeds)))")"
BASE="${RESULTS_ROOT}/${RUN_ID}"

# 2) For each seed: sample with the model and capture diagnostics traces.
for SEED in ${SEEDS}; do
  echo "[run_experiment] ${RUN_ID} seed=${SEED}"
  export COLA_INFER_PER_SAMPLE_NOISE_SEED="${SEED}"   # upstream deterministic noise
  export COLA_DIAG_TRACE=1                            # enable Phase-4 instrumentation
  export COLA_DIAG_TRACE_PATH="${BASE}/traces/seed${SEED}.jsonl"

  # TODO(server): apply model_overrides (block_size / patch_size) from the
  # config to the loaded dit/vae per research/docs/change_map.md (patch_size
  # selects a matching checkpoint; block_size must be set on BOTH dit and vae).
  # TODO(server): invoke the model. Example shape:
  #   python -m cola_dlm.inference \
  #     --dit "${DIT_PATH}" --vae "${VAE_PATH}" --tokenizer "${TOKENIZER_PATH}" \
  #     --input "${BASE}/data/seed${SEED}.jsonl" \
  #     --output "${BASE}/samples/seed${SEED}.jsonl" \
  #     --task "$(python -c "from research.experiment import load_experiment; print(load_experiment('${CONFIG}').task)")"
  echo "[run_experiment] TODO: wire model invocation -> ${BASE}/samples/seed${SEED}.jsonl"
done

# 3) Evaluate structural consistency over the produced samples + traces (CPU).
#    (research/eval is populated in a later phase; metrics land in ${BASE}/metrics.)
echo "[run_experiment] done; samples + traces under ${BASE}"
