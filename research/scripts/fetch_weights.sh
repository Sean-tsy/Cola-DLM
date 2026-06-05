#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SERVER-SIDE ONLY. Fetch / place the Cola DLM weights and verify checksums.
# Prepared locally but NEVER run locally; weights are large and must not enter
# version control (see repo .gitignore: hf_models/ + *.safetensors/*.pt/...).
#
# Source / target / hashes:
#   - WEIGHTS_URL    : where to pull from (HF repo tarball, internal mirror, ...)
#   - hf_models/     : target layout matching scripts/run_benchmark.sh defaults
#                      (cola_dlm/cola_dit, cola_dlm/cola_vae, tokenizer.json)
#   - SHA256_FILE    : sha256 manifest the downloaded files are verified against
#                      (template: research/scripts/weights.sha256.example)
#
# Usage (server):
#   WEIGHTS_URL=... bash research/scripts/fetch_weights.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEST="${WEIGHTS_DEST:-${REPO_DIR}/hf_models}"
SHA256_FILE="${SHA256_FILE:-${SCRIPT_DIR}/weights.sha256}"
: "${WEIGHTS_URL:?set WEIGHTS_URL (weight source; not committed to git)}"

mkdir -p "${DEST}"

# 1) Download into DEST. Replace with your transport (huggingface-cli / aws s3 /
#    wget). Skeleton keeps the contract explicit without hard-coding a source.
echo "[fetch_weights] download ${WEIGHTS_URL} -> ${DEST}"
# TODO(server): e.g.
#   huggingface-cli download "${WEIGHTS_URL}" --local-dir "${DEST}"
#   aws s3 sync "${WEIGHTS_URL}" "${DEST}"

# 2) Verify integrity against the sha256 manifest before any model use.
if [[ -f "${SHA256_FILE}" ]]; then
  echo "[fetch_weights] verifying sha256 against ${SHA256_FILE}"
  ( cd "${DEST}" && sha256sum -c "${SHA256_FILE}" )
  echo "[fetch_weights] checksum OK"
else
  echo "[fetch_weights] WARNING: no checksum manifest at ${SHA256_FILE};" >&2
  echo "  copy weights.sha256.example -> weights.sha256 and fill real hashes." >&2
  exit 1
fi

echo "[fetch_weights] weights ready under ${DEST} (NOT tracked by git)"
