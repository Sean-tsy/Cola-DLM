#!/usr/bin/env python
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SERVER-SIDE VAE-only GT encode->decode baseline.

Loads the Cola VAE and tokenizer, encodes each record's ground-truth structural
text, decodes it without the DiT prior, and writes Cola-style sample JSONL with
``generate``. This isolates decoder/VAE reconstruction errors from diffusion
prior errors for 9.4 H5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_text(record: dict) -> str:
    meta = record.get("meta") or {}
    if meta.get("probe") == "dyck" and meta.get("mode") == "completion":
        return f"{meta.get('prefix', '')}{record.get('ground_truth', '')}"
    return str(record.get("ground_truth", ""))


def _shape_tensor(torch, lens: list[int], device):
    return torch.tensor([[int(l)] for l in lens], dtype=torch.long, device=device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VAE-only ground-truth roundtrip (server-side).")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pad-token-id", type=int, default=100277)
    args = parser.parse_args(argv)

    import torch
    from tokenizers import Tokenizer

    from cola_dlm.modeling_cola_vae import ColaTextVAEModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    chunk = vae.patch_size * vae.block_size

    rows = _read_jsonl(args.input_jsonl)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    out_rows = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        ids_list = []
        real_lens = []
        targets = []
        for rec in batch:
            text = _target_text(rec)
            ids = tokenizer.encode(text).ids
            real_lens.append(len(ids))
            pad = (chunk - len(ids) % chunk) % chunk
            ids_list.append(torch.tensor(ids + [args.pad_token_id] * pad, dtype=torch.long, device=device))
            targets.append(text)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            enc = vae.encode(ids_list)
            latents = torch.cat(enc.latents_list, dim=0)
            shape = _shape_tensor(torch, [x.shape[0] for x in enc.latents_list], device)
            logits = vae.decode(latents, txt_shape=shape, txt_q_shape=shape)

        offset = 0
        for rec, lat, real_len, target in zip(batch, enc.latents_list, real_lens, targets):
            n_tokens = lat.shape[0] * vae.patch_size
            token_ids = logits[0, offset * vae.patch_size : offset * vae.patch_size + n_tokens].argmax(dim=-1)
            token_ids = token_ids[:real_len].detach().cpu().tolist()
            out = dict(rec)
            out["generate"] = tokenizer.decode(token_ids)
            out["ground_truth"] = target
            out["model_family"] = "cola_vae_only"
            out_rows.append(out)
            offset += lat.shape[0]

    with open(args.output_jsonl, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[vae_roundtrip] {len(out_rows)} records -> {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
