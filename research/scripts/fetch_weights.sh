#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SERVER-SIDE ONLY. Fetch / place model weights and record checksums. Prepared
# locally but NEVER run locally; weights are large and must not enter version
# control (see .gitignore: hf_models/ + *.safetensors/*.pt/...).
#
# Network: the GPU server (Tencent China) cannot reach huggingface.co (GFW), so
# downloads route through the mirror via HF_ENDPOINT=https://hf-mirror.com.
#
# Default target layout matches scripts/run_benchmark.sh:
#   hf_models/cola_dlm/cola_dit   hf_models/cola_dlm/cola_vae   hf_models/tokenizer.json
#
# Records sha256 of every downloaded weight file to ${DEST}/weights.sha256 so the
# run manifest can pin exactly which weights produced a result (env节九 9.0 gate).
#
# Usage (server):
#   bash research/scripts/fetch_weights.sh                       # Cola (default)
#   WEIGHTS_REPO=GSAI-ML/LLaDA-8B-Base WEIGHTS_DEST=hf_models/llada \
#       bash research/scripts/fetch_weights.sh                   # a baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# HF repo id (preferred) or a raw URL via WEIGHTS_URL for non-HF transports.
WEIGHTS_REPO="${WEIGHTS_REPO:-ByteDance-Seed/Cola-DLM}"
DEST="${WEIGHTS_DEST:-${REPO_DIR}/hf_models}"
SHA256_FILE="${SHA256_FILE:-${DEST}/weights.sha256}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${DEST}"

# 1) Download. Prefer the huggingface_hub CLI (`hf`, in requirements.lock).
echo "[fetch_weights] HF_ENDPOINT=${HF_ENDPOINT}"
if [[ -n "${WEIGHTS_URL:-}" ]]; then
  echo "[fetch_weights] download ${WEIGHTS_URL} -> ${DEST}"
  ( cd "${DEST}" && curl -fL --retry 3 -O "${WEIGHTS_URL}" )
elif command -v hf >/dev/null 2>&1; then
  echo "[fetch_weights] hf download ${WEIGHTS_REPO} -> ${DEST}"
  hf download "${WEIGHTS_REPO}" --local-dir "${DEST}"
else
  echo "[fetch_weights] hf CLI not found; falling back to huggingface-cli"
  huggingface-cli download "${WEIGHTS_REPO}" --local-dir "${DEST}"
fi

# 2) Record + verify sha256 of weight files (env节九 9.0: hashes -> manifest).
echo "[fetch_weights] recording sha256 -> ${SHA256_FILE}"
( cd "${DEST}" && find . -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name 'tokenizer.json' \) \
    -print0 | sort -z | xargs -0 sha256sum > "${SHA256_FILE}" )
echo "[fetch_weights] verifying"
( cd "${DEST}" && sha256sum -c "${SHA256_FILE}" >/dev/null )
echo "[fetch_weights] checksum OK; $(wc -l < "${SHA256_FILE}") files hashed"

echo "[fetch_weights] weights ready under ${DEST} (NOT tracked by git)"
echo "[fetch_weights] add ${SHA256_FILE} contents to the run manifest."
