#!/usr/bin/env python
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SERVER-SIDE inference wrapper for diagnostic experiments (loads the model).

Thin research-layer wrapper around ``cola_dlm.inference`` that adds exactly the
three things env节四/九 need on top of the upstream CLI, WITHOUT editing core:

1. **Config-driven sampling knobs** — read directly from one experiment YAML
   (``timestep_num`` / ``guidance_scale`` / ``max_new_tokens`` / decode knobs)
   so a run is fully described by ``config + git commit``.
2. **block_size override (H3 sweep)** — the upstream CLI has no ``--block_size``
   flag; ``block_size`` is a model attribute (``dit.block_size``, read in
   ``generate_task_repaint_inference``). We set it on the loaded DiT (and VAE if
   present) per ``research/docs/change_map.md``. ``patch_size`` comes from the
   VAE weights and is NOT overridden here (changing it needs matching weights).
3. **prompt->question prep** — mirror ``prompt`` into ``question`` (see
   ``prep_input``) and drive ``apply_prompt_template`` with ``--task_name
   lambada`` so the synthetic prompt reaches the model verbatim.

Data-parallel sharding reuses the upstream stride convention
(``records[rank::world_size]`` -> ``<run_id>_rank<rank>.jsonl``); the shards are
merged by ``run_experiment.sh``.

NEVER run locally (imports torch + loads weights). Not imported by the local
pure-Python test suite.

Usage (server, one GPU rank)::

    CUDA_VISIBLE_DEVICES=0 python -m research.scripts.infer_cola \
        --config research/configs/experiments/smoke_dyck1.yaml \
        --input-jsonl  research/results/smoke_dyck1/data/seed0.jsonl \
        --output-jsonl research/results/smoke_dyck1/samples/seed0.jsonl \
        --dit-path "$DIT_PATH" --vae-path "$VAE_PATH" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --rank 0 --world-size 1
"""

from __future__ import annotations

import argparse
import json
import os

from research.experiment import load_experiment
from research.scripts.prep_input import VERBATIM_TASK_NAME, to_infer_record


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _shard_output_path(output_jsonl: str, rank: int, world_size: int) -> str:
    if world_size <= 1:
        return output_jsonl
    base, ext = os.path.splitext(output_jsonl)
    return f"{base}_rank{rank}{ext or '.jsonl'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cola diagnostic inference (server-side, GPU).")
    parser.add_argument("--config", required=True, help="Experiment YAML (sampling knobs + block_size).")
    parser.add_argument("--input-jsonl", required=True, help="Probe data JSONL for one seed.")
    parser.add_argument("--output-jsonl", required=True, help="Where to write this rank's samples.")
    parser.add_argument("--dit-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0, help="GPU rank for data-parallel sharding.")
    parser.add_argument("--world-size", type=int, default=1, help="Total GPUs for data-parallel sharding.")
    parser.add_argument("--pad-token-id", type=int, default=100277)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--im-end-token-id", type=int, default=None)
    args = parser.parse_args(argv)

    # Heavy imports kept inside main so `-h` / import-time stays torch-free.
    import torch
    from tokenizers import Tokenizer

    from cola_dlm.inference import generate_task_repaint_inference
    from cola_dlm.modeling_cola_dit import ColaDiTModel
    from cola_dlm.modeling_cola_vae import ColaTextVAEModel

    cfg = load_experiment(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[infer] loading DiT  {args.dit_path}")
    dit = ColaDiTModel.from_pretrained(args.dit_path).to(device)
    print(f"[infer] loading VAE  {args.vae_path}")
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # block_size override (H3). patch_size is left to the VAE weights.
    if cfg.block_size is not None:
        print(f"[infer] override block_size {dit.block_size} -> {cfg.block_size} (dit+vae)")
        dit.block_size = cfg.block_size
        if hasattr(vae, "block_size"):
            vae.block_size = cfg.block_size

    records = _read_jsonl(args.input_jsonl)
    records = [to_infer_record(r) for r in records]  # mirror prompt->question
    total = len(records)
    my_records = records[args.rank :: args.world_size]
    print(f"[infer] rank {args.rank}/{args.world_size}: {len(my_records)}/{total} records")

    out_path = _shard_output_path(args.output_jsonl, args.rank, args.world_size)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not my_records:
        open(out_path, "w").close()
        print(f"[infer] rank {args.rank}: no records, wrote empty {out_path}")
        return 0

    results: list[dict] = []
    for start in range(0, len(my_records), args.batch_size):
        batch = my_records[start : start + args.batch_size]
        print(f"[infer] rank {args.rank} batch {start // args.batch_size + 1} ({len(batch)})")
        results.extend(
            generate_task_repaint_inference(
                dit=dit,
                vae=vae,
                tokenizer=tokenizer,
                prompts=batch,
                task_name=VERBATIM_TASK_NAME,
                device=device,
                T=1000.0,
                timestep_num=cfg.timestep_num,
                guidance_scale=cfg.guidance_scale,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                pad_token_id=args.pad_token_id,
                eos_token_id=args.eos_token_id,
                im_end_token_id=args.im_end_token_id,
            )
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] rank {args.rank}: {len(results)} samples -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
