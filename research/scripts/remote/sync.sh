#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# sync.sh -- LOCAL trigger: push code to git, make the server check out the same
# commit, and VERIFY commit consistency before any job. Code travels via git;
# large weights/data via file sync. Returns immediately (no GPU work).
#
# Usage (local):
#   REMOTE=cola-gpu bash research/scripts/remote/sync.sh [branch]
#     branch : branch to push/checkout (default: current local branch)
#
# Auth: key-based ssh via ~/.ssh/config alias REMOTE. No passwords/tokens here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=research/scripts/remote/_common.sh
source "${SCRIPT_DIR}/_common.sh"

GIT="${GIT:-git}"
BRANCH="${1:-$($GIT rev-parse --abbrev-ref HEAD)}"
SHA="$($GIT rev-parse HEAD)"

echo "[sync] push ${BRANCH} (${SHA:0:12}) to origin"
$GIT push origin "${BRANCH}"

echo "[sync] server fetch + checkout ${SHA:0:12}"
remote_sh "git fetch --all --quiet && git checkout --quiet ${SHA}"

echo "[sync] verify server commit consistency"
remote_sh "bash research/scripts/check_sync.sh ${SHA}"

echo "[sync] OK: server at ${SHA:0:12}, clean and consistent"
