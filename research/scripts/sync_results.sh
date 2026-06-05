#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Sync one run's artifacts between the server and large-file storage. Skeleton:
# small structured results (manifest, metrics, plots) are committed to git;
# large artifacts (data / samples / traces / logs) go to object storage, never
# into version control (see repo .gitignore).
#
# Usage (server):
#   bash research/scripts/sync_results.sh <run_id> [push|pull]
set -euo pipefail

RUN_ID="${1:?usage: sync_results.sh <run_id> [push|pull]}"
DIRECTION="${2:-push}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"
REMOTE="${RESULTS_REMOTE:?set RESULTS_REMOTE (e.g. s3://bucket/cola-diag or rsync target)}"

BASE="${RESULTS_ROOT}/${RUN_ID}"
LARGE_KINDS=(data samples traces logs)   # kept out of git; synced to storage
SMALL_KINDS=(metrics plots)              # committed to git alongside manifest

case "${DIRECTION}" in
  push)
    for KIND in "${LARGE_KINDS[@]}"; do
      # TODO(server): replace with your transport (aws s3 sync / rsync / gsutil).
      echo "[sync] push ${BASE}/${KIND} -> ${REMOTE}/${RUN_ID}/${KIND}"
    done
    echo "[sync] commit small results to git: manifest + ${SMALL_KINDS[*]}"
    ;;
  pull)
    for KIND in "${LARGE_KINDS[@]}"; do
      echo "[sync] pull ${REMOTE}/${RUN_ID}/${KIND} -> ${BASE}/${KIND}"
    done
    ;;
  *)
    echo "unknown direction: ${DIRECTION} (expected push|pull)" >&2
    exit 2
    ;;
esac
