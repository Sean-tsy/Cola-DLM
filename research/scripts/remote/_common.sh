#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Shared config for the LOCAL-side remote-control helpers (B-mode: local
# triggers, server executes). Sourced by sync.sh / submit.sh / status.sh /
# fetch.sh. Contains NO passwords / tokens: auth is key-based ssh via an
# ~/.ssh/config alias.
#
# Required env:
#   REMOTE        : ssh alias from ~/.ssh/config (e.g. "cola-gpu"); key-based.
# Optional env (defaults):
#   REMOTE_REPO   : repo path on server      (default /data0/users/siyuan/projects/Cola-DLM)
#   RESULTS_ROOT  : results dir (local+remote, relative to repo) (default research/results)
#   LOG_DIR       : detached-job log dir on server, relative to repo (default logs)
#   SSH           : ssh command              (default "ssh"; override for tests)
#   RSYNC         : rsync command            (default "rsync -az"; override for tests)
#
# All commands here are "fire-and-return": never block on a foreground GPU job.
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE (ssh alias from ~/.ssh/config; key-based, no password)}"
REMOTE_REPO="${REMOTE_REPO:-/data0/users/siyuan/projects/Cola-DLM}"
RESULTS_ROOT="${RESULTS_ROOT:-research/results}"
LOG_DIR="${LOG_DIR:-logs}"
SSH="${SSH:-ssh}"
RSYNC="${RSYNC:-rsync -az}"

# Run a command on the server inside the repo. Returns when the *ssh* call
# returns; the command itself must be non-blocking (detached) for GPU jobs.
remote_sh() {
  # shellcheck disable=SC2086
  $SSH "${REMOTE}" "cd ${REMOTE_REPO} && $*"
}

log_path() { printf '%s/%s/%s.log' "${REMOTE_REPO}" "${LOG_DIR}" "$1"; }
done_path() { printf '%s/%s/%s.done' "${REMOTE_REPO}" "${LOG_DIR}" "$1"; }
