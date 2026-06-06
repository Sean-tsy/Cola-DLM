#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# sync.sh -- LOCAL trigger: ship the current commit to the server and VERIFY the
# server is at the same commit before any job. Returns immediately (no GPU work).
#
# Transport note: the GPU server (Tencent China) cannot reach github.com (GFW),
# so the server CANNOT `git fetch`. Code travels by rsync-over-ssh instead:
#   - we still `git push origin` (local->GitHub works) for backup/provenance;
#   - we rsync the working tree INCLUDING .git so the server's HEAD == local SHA
#     and check_sync.sh can confirm consistency without contacting GitHub.
# Large weights/data are excluded (hf_models/ + per-run data/samples/traces).
#
# Usage (local):
#   REMOTE=cityu_tecent bash research/scripts/remote/sync.sh [branch]
#     branch : branch to push to origin (default: current local branch)
#
# Auth: key-based ssh via ~/.ssh/config alias REMOTE. No passwords/tokens here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=research/scripts/remote/_common.sh
source "${SCRIPT_DIR}/_common.sh"

GIT="${GIT:-git}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BRANCH="${1:-$($GIT -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
SHA="$($GIT -C "${REPO_ROOT}" rev-parse HEAD)"

# Push to origin too (local->GitHub is reachable); best-effort, non-fatal so a
# blocked/offline origin never stops the server sync.
echo "[sync] push ${BRANCH} (${SHA:0:12}) to origin (best-effort)"
$GIT -C "${REPO_ROOT}" push origin "${BRANCH}" || echo "[sync] WARN: git push failed; continuing with rsync"

echo "[sync] rsync working tree (incl .git) -> ${REMOTE}:${REMOTE_REPO}"
remote_sh "mkdir -p '${REMOTE_REPO}'"
# shellcheck disable=SC2086
${RSYNC} --delete \
  --exclude='.venv/' --exclude='.venv-gpu/' \
  --exclude='hf_models/' --exclude='logs/' \
  --exclude="${RESULTS_ROOT}/*/data/" --exclude="${RESULTS_ROOT}/*/samples/" \
  --exclude="${RESULTS_ROOT}/*/traces/" \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
  -e "${SSH}" "${REPO_ROOT}/" "${REMOTE}:${REMOTE_REPO}/"

echo "[sync] checkout ${SHA:0:12} + verify server commit consistency"
remote_sh "git checkout --quiet ${SHA} 2>/dev/null || git checkout --quiet -- . ; bash research/scripts/check_sync.sh ${SHA}"

echo "[sync] OK: server at ${SHA:0:12}, clean and consistent"
