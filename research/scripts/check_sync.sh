#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Verify the SERVER checkout is code-consistent with the local commit before
# launching a job. Code travels via version control; large weights / data via
# file sync. After syncing, the server must be at the SAME commit so a run's
# manifest git_sha is trustworthy and reproducible/rollback-able.
#
# Usage (server):
#   bash research/scripts/check_sync.sh <expected_sha>
#   EXPECTED_SHA=<sha> bash research/scripts/check_sync.sh
set -euo pipefail

EXPECTED_SHA="${1:-${EXPECTED_SHA:-}}"
if [[ -z "${EXPECTED_SHA}" ]]; then
  echo "usage: check_sync.sh <expected_sha>  (the local commit you synced from)" >&2
  exit 2
fi

HEAD_SHA="$(git rev-parse HEAD)"

# Allow short SHAs: compare on the common prefix length.
LEN="${#EXPECTED_SHA}"
if [[ "${HEAD_SHA:0:${LEN}}" != "${EXPECTED_SHA}" ]]; then
  echo "[check_sync] MISMATCH: server HEAD ${HEAD_SHA} != expected ${EXPECTED_SHA}" >&2
  echo "  sync the code (git fetch && git checkout ${EXPECTED_SHA}) before running." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[check_sync] WARNING: working tree is dirty; commit/stash for a clean run." >&2
  git status --short >&2
  exit 1
fi

echo "[check_sync] OK: server at ${HEAD_SHA}, clean tree, matches ${EXPECTED_SHA}"
