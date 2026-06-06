#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# fetch.sh -- LOCAL pull: rsync one run's artifacts back from the server into
# the local results/<run_id>/. Returns when the transfer finishes; no GPU work.
#
# Usage (local):
#   REMOTE=cola-gpu bash research/scripts/remote/fetch.sh <run_id>
#
# Env:
#   LOCAL_RESULTS_ROOT : local destination root (default: RESULTS_ROOT, i.e.
#                        mirror the server path). Override to fetch elsewhere.
#
# Auth: key-based ssh via ~/.ssh/config alias REMOTE. No passwords/tokens here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=research/scripts/remote/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:?usage: fetch.sh <run_id>}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-${RESULTS_ROOT}}"
SRC="${REMOTE}:${REMOTE_REPO}/${RESULTS_ROOT}/${RUN_ID}/"
DST="${LOCAL_RESULTS_ROOT}/${RUN_ID}/"

mkdir -p "${DST}"
echo "[fetch] ${SRC} -> ${DST}"
# shellcheck disable=SC2086
$RSYNC "${SRC}" "${DST}"
echo "[fetch] done: ${DST}"
