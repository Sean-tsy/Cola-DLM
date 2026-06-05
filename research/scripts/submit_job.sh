#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SERVER-SIDE ONLY. Submit one experiment as a multi-GPU / cluster job. Thin
# launcher around run_experiment.sh: it (optionally) verifies commit
# consistency, then dispatches via your scheduler (srun / torchrun / qsub).
# Prepared locally, run on the server.
#
# Usage (server):
#   NUM_GPUS=8 EXPECTED_SHA=<local sha> \
#     bash research/scripts/submit_job.sh research/configs/experiments/dyck_L32_D6_k2_blk4.yaml
#
# Start small: validate the chain with the smoke config first, then scale.
#   bash research/scripts/submit_job.sh research/configs/experiments/smoke_dyck1.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:?usage: submit_job.sh <experiment-config.yaml>}"

NUM_GPUS="${NUM_GPUS:-1}"
LAUNCHER="${LAUNCHER:-local}"   # local | torchrun | srun

# 1) Confirm the server is at the synced commit (skip with EXPECTED_SHA unset).
if [[ -n "${EXPECTED_SHA:-}" ]]; then
  bash "${SCRIPT_DIR}/check_sync.sh" "${EXPECTED_SHA}"
fi

# 2) Dispatch. run_experiment.sh reads NUM_GPUS for the model invocation.
export NUM_GPUS
case "${LAUNCHER}" in
  local)
    bash "${SCRIPT_DIR}/run_experiment.sh" "${CONFIG}"
    ;;
  torchrun)
    # TODO(server): point run_experiment.sh's model step at a torchrun entry.
    echo "[submit_job] torchrun --nproc_per_node=${NUM_GPUS} ... (wire model step)"
    bash "${SCRIPT_DIR}/run_experiment.sh" "${CONFIG}"
    ;;
  srun)
    # TODO(server): wrap with your Slurm allocation.
    echo "[submit_job] srun --gres=gpu:${NUM_GPUS} bash run_experiment.sh ${CONFIG}"
    srun --gres="gpu:${NUM_GPUS}" bash "${SCRIPT_DIR}/run_experiment.sh" "${CONFIG}"
    ;;
  *)
    echo "unknown LAUNCHER: ${LAUNCHER} (expected local|torchrun|srun)" >&2
    exit 2
    ;;
esac
