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

"""Reconstruct calibration *inputs* from committed reference benchmark outputs.

环节九 · 9.2 复现校准. The upstream eval harness (``scripts/run_benchmark.sh``)
consumes pre-materialized ``generate_task_data/<task>.jsonl`` inputs that are
**not** shipped in the repo (gitignored), and the server cannot reach
HuggingFace (GFW). However, the repo *does* commit the reference benchmark
**outputs** under ``eval_output/tasks_default/<task>.jsonl`` together with the
measured accuracy in ``eval_output/accuracy_summary.csv`` (LAMBADA 50.80).

For ``lambada`` the prompt template returns ``question`` verbatim. For the other
seven official tasks, ``scripts/acc_calc.py`` has already stripped the committed
few-shot prefix and preserved ``choices`` / ``ground_truth`` in the reference
outputs. That is enough to invert the task-specific prompt suffixes back into
the upstream ``generate_task_data/<task>.jsonl`` input contract.

Pure data munging: no torch, no model, runs locally and is unit-tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SUPPORTED_TASKS = ("lambada", "mmlu", "obqa", "hellaswag", "race", "siqa", "squad", "story_cloze")


def _require_prompt(record: dict[str, Any]) -> str:
    if "prompt" not in record:
        raise KeyError("reference record has no 'prompt' field; cannot reconstruct input")
    return str(record["prompt"])


def _strip_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def _before_choices(prompt: str) -> str:
    for marker in ("\nOptions:", "\n(A)"):
        if marker in prompt:
            return prompt.split(marker, 1)[0]
    return _strip_suffix(prompt, "\nAnswer:")


def _split_squad(prompt: str) -> tuple[str, str]:
    prompt = _strip_suffix(prompt, "\nAnswer:")
    marker = "\nQuestion: "
    if marker not in prompt:
        return "", prompt
    context, question = prompt.rsplit(marker, 1)
    return context, question


def to_input_record(record: dict[str, Any], task: str = "lambada") -> dict[str, Any]:
    """Rebuild an inference *input* record from a reference *output* record.

    The output record is the committed, already-scored official benchmark row.
    We keep only fields consumed by ``cola_dlm.inference`` plus
    ``ground_truth`` for downstream scoring.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported calibration task: {task}")
    prompt = _require_prompt(record)
    gt = record.get("ground_truth", record.get("answer", ""))
    out = {
        "id": record.get("id"),
        "question": prompt,
        "ground_truth": gt,
    }
    if task == "lambada":
        return out
    if task == "squad":
        context, question = _split_squad(prompt)
        out["context"] = context
        out["question"] = question
        return out
    out["question"] = _before_choices(prompt)
    if "choices" in record:
        out["choices"] = record["choices"]
    return out


def transform_file(in_path: str, out_path: str, task: str = "lambada") -> int:
    """Reconstruct every input record from the reference output JSONL. Returns count."""
    n = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = to_input_record(json.loads(line), task)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct official calibration inputs from committed reference outputs (CPU only).",
    )
    parser.add_argument(
        "--task",
        default="lambada",
        choices=SUPPORTED_TASKS,
        help="Calibration task.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Committed reference output JSONL, e.g. eval_output/tasks_default/lambada.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Reconstructed input JSONL, e.g. generate_task_data/lambada.jsonl",
    )
    args = parser.parse_args(argv)
    n = transform_file(args.reference, args.output, args.task)
    print(f"[prep_calibration] {args.task}: {n} records -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
