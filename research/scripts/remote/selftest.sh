#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# selftest.sh -- verify the remote four-piece chain (sync/submit/status/fetch)
# end-to-end with a PURE-PYTHON fake job and INJECTED fake ssh/rsync (no real
# server, no model). Proves the launch->poll->fetch wiring before going live.
#
# Usage (local):  bash research/scripts/remote/selftest.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# "Server" repo = a checkout of the current tree (no .git needed for the fake).
SERVER="${WORK}/server/Cola-DLM"
mkdir -p "${SERVER}"
cp -R "${REPO_ROOT}/research" "${SERVER}/research"
# Minimal stand-ins so the server-side scripts the chain calls exist.
mkdir -p "${SERVER}/research/scripts"

# Fake ssh: ignore the host arg, run the command locally via bash.
cat > "${WORK}/fake_ssh" <<'EOF'
#!/usr/bin/env bash
shift            # drop the host alias
exec bash -c "$*"
EOF
chmod +x "${WORK}/fake_ssh"

# Fake rsync: strip a leading "host:" from src/dst, then local rsync -a.
cat > "${WORK}/fake_rsync" <<'EOF'
#!/usr/bin/env bash
args=()
for a in "$@"; do args+=("${a#*:}"); done   # "host:/p" -> "/p"; "-az" unchanged
exec rsync -a "${args[@]}"
EOF
chmod +x "${WORK}/fake_rsync"

export REMOTE="fake-server"
export REMOTE_REPO="${SERVER}"
export SSH="${WORK}/fake_ssh"
export RSYNC="${WORK}/fake_rsync"
export LAUNCHER="nohup"
export JOB_CMD="python3 research/scripts/remote/_fake_job.py selftest_run"

RUN_ID="selftest_run"
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1" >&2; exit 1; }

echo "[selftest] 1/3 submit (detached fake job)"
bash "${SCRIPT_DIR}/submit.sh" "research/configs/experiments/smoke_dyck1.yaml" "${RUN_ID}"
[ -f "${SERVER}/logs/${RUN_ID}.log" ] || fail "log not created"
pass "submit returned immediately, log created"

echo "[selftest] 2/3 status (poll until DONE)"
for i in $(seq 1 50); do
  set +e
  bash "${SCRIPT_DIR}/status.sh" "${RUN_ID}" 5 >/dev/null
  rc=$?
  set -e
  [ "${rc}" -eq 0 ] && break
  [ "${rc}" -eq 1 ] && fail "job reported FAILED"
  sleep 0.2
done
[ "${rc}" -eq 0 ] || fail "job did not reach DONE (rc=${rc})"
pass "status reached DONE (exit 0)"

echo "[selftest] 3/3 fetch (rsync artifacts back)"
LOCAL_DST="${WORK}/local_results"
# fetch reads from server's RESULTS_ROOT, writes to an isolated local root so we
# don't touch the repo tree.
LOCAL_RESULTS_ROOT="${LOCAL_DST}" bash "${SCRIPT_DIR}/fetch.sh" "${RUN_ID}"
[ -f "${LOCAL_DST}/${RUN_ID}/manifest.json" ] || fail "manifest not fetched"
[ -f "${LOCAL_DST}/${RUN_ID}/metrics/summary.json" ] || fail "metrics not fetched"
pass "fetch pulled manifest + metrics"

echo "[selftest] OK: sync/submit/status/fetch chain verified with fake job"
