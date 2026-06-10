#!/usr/bin/env python
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SERVER-SIDE autoregressive baseline runner for the diagnostic probes.

Runs a plain HF causal LM (e.g. Qwen2.5-1.5B base) over the exact same probe
data, prompts, decoding budget, and verifiers as the Cola runs, so the
diagnostic numbers are apples-to-apples: prompt fed verbatim (no chat
template), greedy decoding, ``max_new_tokens`` from the experiment config,
samples written in the Cola JSONL convention (``generate`` = continuation
only). NEVER run locally (imports torch + loads weights).

Usage (server, one GPU)::

    python -m research.scripts.infer_ar_baseline \
        --config research/configs/experiments/dyck_L64_D8_k1_qwen15b.yaml \
        --input-jsonl  research/results/<run>/data/seed1234.jsonl \
        --output-jsonl research/results/<run>/samples/seed1234.jsonl \
        --model-path "$BASELINE_MODEL_PATH"
"""

from __future__ import annotations

import argparse
import json
import os

from research.experiment import load_experiment


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AR baseline inference (server-side, GPU).")
    parser.add_argument("--config", required=True, help="Experiment YAML (max_new_tokens budget).")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", required=True, help="HF model id or local path.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_experiment(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ar_baseline] loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16).to(device)
    model.eval()

    records = _read_jsonl(args.input_jsonl)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    results: list[dict] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = [str(r.get("prompt", "")) for r in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuations = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for rec, gen in zip(batch, continuations):
            results.append(
                {
                    "id": rec.get("id"),
                    "prompt": rec.get("prompt", ""),
                    "generate": gen,
                    "ground_truth": rec.get("ground_truth", ""),
                }
            )
        print(f"[ar_baseline] {len(results)}/{len(records)}")

    with open(args.output_jsonl, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ar_baseline] {len(results)} samples -> {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
