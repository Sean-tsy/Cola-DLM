#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# status.sh -- LOCAL poll: tail a detached job's log and report its state.
# Returns immediately (one ssh round-trip); the agent polls, never blocks.
#
# Usage (local):
#   REMOTE=cola-gpu bash research/scripts/remote/status.sh <run_id> [tail_lines]
#     tail_lines : log lines to show (default 20)
#
# Exit code: 0 = DONE (job exit 0), 1 = FAILED (job exit != 0),
#            2 = RUNNING, 3 = NOT_FOUND.
#
# Auth: key-based ssh via ~/.ssh/config alias REMOTE. No passwords/tokens here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=research/scripts/remote/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:?usage: status.sh <run_id> [tail_lines]}"
TAIL_LINES="${2:-20}"
LOG="${LOG_DIR}/${RUN_ID}.log"
DONE="${LOG_DIR}/${RUN_ID}.done"

# One round-trip: print state token, exit code (if any), then a log tail.
OUT="$(remote_sh "
  if [ -f ${DONE} ]; then echo \"STATE DONE \$(cat ${DONE})\";
  elif [ -f ${LOG} ]; then echo 'STATE RUNNING';
  else echo 'STATE NOT_FOUND'; fi
  echo '----- tail -----'
  [ -f ${LOG} ] && tail -n ${TAIL_LINES} ${LOG} || true
")"
echo "${OUT}"

STATE_LINE="$(printf '%s\n' "${OUT}" | sed -n '1p')"
case "${STATE_LINE}" in
  "STATE DONE 0") exit 0 ;;
  "STATE DONE "*) exit 1 ;;
  "STATE RUNNING") exit 2 ;;
  *) exit 3 ;;
esac
