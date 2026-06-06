#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# submit.sh -- LOCAL trigger: start one experiment as a DETACHED server job and
# return immediately. The ssh call only *launches* the job (tmux / nohup /
# sbatch); it NEVER stays in the foreground waiting for the GPU work.
#
# Usage (local):
#   REMOTE=cola-gpu bash research/scripts/remote/submit.sh <config> <run_id>
#     config : experiment config path on the server (e.g.
#              research/configs/experiments/smoke_dyck1.yaml)
#     run_id : archive id; log goes to <LOG_DIR>/<run_id>.log, completion
#              marker to <LOG_DIR>/<run_id>.done
#
# Env:
#   LAUNCHER : nohup | tmux | sbatch        (default nohup)
#   JOB_CMD  : command the job runs          (default:
#              bash research/scripts/submit_job.sh <config>)
#              Override with a pure-Python fake job for end-to-end testing.
#
# Auth: key-based ssh via ~/.ssh/config alias REMOTE. No passwords/tokens here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=research/scripts/remote/_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${1:?usage: submit.sh <config> <run_id>}"
RUN_ID="${2:?usage: submit.sh <config> <run_id>}"
LAUNCHER="${LAUNCHER:-nohup}"
JOB_CMD="${JOB_CMD:-bash research/scripts/submit_job.sh ${CONFIG}}"

LOG="${LOG_DIR}/${RUN_ID}.log"
DONE="${LOG_DIR}/${RUN_ID}.done"

# Prepare synchronously so the log exists the moment submit returns; then run
# the job detached, always dropping a .done marker with the exit code (even on
# failure) for status.sh to poll.
PREP="mkdir -p ${LOG_DIR}; rm -f ${DONE}; : > ${LOG}"
INNER="{ ${JOB_CMD}; echo \$? > ${DONE}; } >> ${LOG} 2>&1"

echo "[submit] run_id=${RUN_ID} launcher=${LAUNCHER} -> ${LOG}"
case "${LAUNCHER}" in
  nohup)
    # nohup + & => detached; ssh returns at once (portable, no setsid needed).
    remote_sh "${PREP}; nohup bash -c '${INNER}' >/dev/null 2>&1 < /dev/null & echo launched pid \$!"
    ;;
  tmux)
    remote_sh "${PREP}; tmux new-session -d -s ${RUN_ID} \"bash -c '${INNER}'\" && echo launched tmux ${RUN_ID}"
    ;;
  sbatch)
    # Detached by the scheduler; sbatch itself returns immediately.
    remote_sh "mkdir -p ${LOG_DIR} && sbatch -J ${RUN_ID} -o ${LOG} --wrap \"${JOB_CMD}; echo \\\$? > ${DONE}\""
    ;;
  *)
    echo "unknown LAUNCHER: ${LAUNCHER} (expected nohup|tmux|sbatch)" >&2
    exit 2
    ;;
esac

echo "[submit] launched (detached); poll with: REMOTE=${REMOTE} bash research/scripts/remote/status.sh ${RUN_ID}"
