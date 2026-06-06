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

"""Make probe records inference-ready (pure-Python / CPU; safe to run locally).

Our data generators emit records with a ``prompt`` field, but
``cola_dlm.inference.apply_prompt_template`` only knows a fixed set of task
names. For an *unregistered* task it returns ``question`` verbatim; the
``lambada`` task does the same. So to feed an arbitrary synthetic prompt to the
model untouched we (a) mirror ``prompt`` into ``question`` and (b) drive
inference with ``--task_name lambada`` (see ``run_experiment.sh``). This keeps
the prompt exactly as generated without editing upstream core templates.

This step is pure data munging — no torch, no model — so it is unit-tested and
runs both locally and server-side.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# The task_name to pass to cola_dlm.inference so apply_prompt_template returns
# the question (== our prompt) verbatim, with no extra few-shot scaffolding.
VERBATIM_TASK_NAME = "lambada"


def to_infer_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``record`` with ``question`` mirrored from ``prompt``.

    An existing non-empty ``question`` is preserved. ``prompt`` and all other
    fields are kept so downstream eval can still join on ``id`` / ``ground_truth``.
    """
    if "prompt" not in record:
        raise KeyError("record has no 'prompt' field; cannot make it inference-ready")
    out = dict(record)
    if not out.get("question"):
        out["question"] = record["prompt"]
    return out


def transform_file(in_path: str, out_path: str) -> int:
    """Mirror prompt->question for every JSONL record. Returns record count."""
    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = to_infer_record(json.loads(line))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mirror prompt->question for inference (CPU only).")
    parser.add_argument("input_jsonl", help="Probe data JSONL (id/prompt/ground_truth/meta).")
    parser.add_argument("output_jsonl", help="Inference-ready JSONL (adds question=prompt).")
    args = parser.parse_args(argv)
    n = transform_file(args.input_jsonl, args.output_jsonl)
    print(f"[prep_input] {n} records -> {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
