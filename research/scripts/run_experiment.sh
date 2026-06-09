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
NUM_GPUS="${NUM_GPUS:-1}"   # multi-GPU degree (passed through by submit_job.sh)

: "${DIT_PATH:?set DIT_PATH (checkpoint placeholder ${DIT_PATH})}"
: "${VAE_PATH:?set VAE_PATH}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"

# 1) Materialize probe data + manifest (CPU; reuses the local generator).
python -m research.scripts.gen_data "${CONFIG}" --results-root "${RESULTS_ROOT}"

RUN_ID="$(python -c "from research.experiment import load_experiment; print(load_experiment('${CONFIG}').run_id)")"
SEEDS="$(python -c "from research.experiment import load_experiment; print(' '.join(map(str, load_experiment('${CONFIG}').seeds)))")"
BASE="${RESULTS_ROOT}/${RUN_ID}"

# 2) For each seed: sample with the model (data-parallel across NUM_GPUS) and
#    capture diagnostics traces. block_size/patch_size overrides + prompt->question
#    are applied inside research.scripts.infer_cola (no upstream core edit).
mkdir -p "${BASE}/samples" "${BASE}/traces"
for SEED in ${SEEDS}; do
  echo "[run_experiment] ${RUN_ID} seed=${SEED} (NUM_GPUS=${NUM_GPUS})"
  IN="${BASE}/data/seed${SEED}.jsonl"
  OUT="${BASE}/samples/seed${SEED}.jsonl"

  pids=()
  for ((r = 0; r < NUM_GPUS; r++)); do
    CUDA_VISIBLE_DEVICES="${r}" \
    COLA_INFER_PER_SAMPLE_NOISE_SEED="${SEED}" \
    COLA_DIAG_TRACE=1 \
    COLA_DIAG_TRACE_PATH="${BASE}/traces/seed${SEED}_rank${r}.jsonl" \
      python -m research.scripts.infer_cola \
        --config "${CONFIG}" \
        --input-jsonl "${IN}" \
        --output-jsonl "${OUT}" \
        --dit-path "${DIT_PATH}" \
        --vae-path "${VAE_PATH}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --rank "${r}" --world-size "${NUM_GPUS}" &
    pids+=("$!")
  done
  wait "${pids[@]}"

  # Merge per-rank shards into a single ordered-by-rank samples file.
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    cat "${OUT%.jsonl}"_rank*.jsonl > "${OUT}"
    rm -f "${OUT%.jsonl}"_rank*.jsonl
  fi
  echo "[run_experiment] seed=${SEED} samples -> ${OUT}"
done

# 3) Evaluate structural consistency over the produced samples + traces (CPU).
python -m research.scripts.eval_experiment "${CONFIG}" --results-root "${RESULTS_ROOT}"
echo "[run_experiment] done; archive under ${BASE}"
