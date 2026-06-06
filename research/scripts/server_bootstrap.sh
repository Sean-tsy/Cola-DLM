#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SERVER-SIDE ONLY. Build the reproducible GPU runtime (venv + CUDA torch +
# requirements.lock) for model sampling / evaluation. NEVER run locally.
#
# Network note (cityu_tecent / Tencent China host): github.com and
# huggingface.co are GFW-blocked. PyPI, download.pytorch.org/whl/cu128 and
# hf-mirror.com ARE reachable -> code travels by rsync (not git fetch), weights
# by HF_ENDPOINT=https://hf-mirror.com. This script only needs PyPI + the torch
# CUDA index, both reachable.
#
# Usage (server, detached):
#   cd /data0/users/siyuan/projects/Cola-DLM
#   mkdir -p logs && nohup bash research/scripts/server_bootstrap.sh \
#       > logs/bootstrap.log 2>&1 &
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

# NOTE: requirements.lock pins torch==2.12.0 (a CPU build captured at lock time
# on the local box). The cu128 wheel index only publishes up to 2.11.0, so on
# the GPU server we install the newest cu128 build (>=2.1 is all cola_dlm needs)
# and filter the exact torch pin out of the lock install below.
VENV="${VENV:-.venv-gpu}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"

echo "[bootstrap] $(date -Is) repo=${REPO_DIR} venv=${VENV}"

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip -i "${PIP_INDEX}"

# 1) CUDA torch first (matching pinned version; cu128 index is reachable).
# Fast path: if TORCH_WHEEL_URL is set, curl the wheel from a reachable mirror
# (download.pytorch.org throttles to ~80KB/s from this host; the aliyun mirror
# serves the same wheel at ~5MB/s) and install the local file. Otherwise install
# from the cu128 simple index.
if [[ -n "${TORCH_WHEEL_URL:-}" ]]; then
  # pip validates wheel filenames, so keep the original (URL-decode %2B -> +).
  WHEEL_NAME="$(basename "${TORCH_WHEEL_URL}" | sed 's/%2[bB]/+/g')"
  echo "[bootstrap] curl torch wheel ${TORCH_WHEEL_URL} -> /tmp/${WHEEL_NAME}"
  curl -fL --retry 3 -o "/tmp/${WHEEL_NAME}" "${TORCH_WHEEL_URL}"
  pip install "/tmp/${WHEEL_NAME}"
  rm -f "/tmp/${WHEEL_NAME}"
else
  echo "[bootstrap] installing torch==${TORCH_VERSION} from ${TORCH_INDEX}"
  pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
fi

# 2) Everything else from the locked closure, with the exact torch==2.12.0 pin
#    filtered out so it does not clobber the cu128 GPU build just installed.
echo "[bootstrap] installing requirements.lock (torch pin filtered)"
grep -ivE '^torch==' requirements.lock > /tmp/cola_req.nolorch.txt
pip install -r /tmp/cola_req.nolorch.txt -i "${PIP_INDEX}"

# 3) Sanity: torch sees CUDA, key model deps import.
python - <<'PY'
import torch, transformers, tokenizers, huggingface_hub
print("[bootstrap] torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
print("[bootstrap] transformers", transformers.__version__,
      "tokenizers", tokenizers.__version__, "hub", huggingface_hub.__version__)
PY

echo "[bootstrap] DONE $(date -Is)"
