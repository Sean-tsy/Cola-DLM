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

"""Reconstruct the LAMBADA calibration *input* from committed reference output.

环节九 · 9.2 复现校准. The upstream eval harness (``scripts/run_benchmark.sh``)
consumes pre-materialized ``generate_task_data/<task>.jsonl`` inputs that are
**not** shipped in the repo (gitignored), and the server cannot reach
HuggingFace (GFW). However, the repo *does* commit the reference benchmark
**outputs** under ``eval_output/tasks_default/<task>.jsonl`` together with the
measured accuracy in ``eval_output/accuracy_summary.csv`` (LAMBADA 50.80).

For the ``lambada`` task ``apply_prompt_template`` returns ``question``
verbatim, so the reference output's ``prompt`` field *is* exactly the model
input. We can therefore rebuild the input JSONL offline — no dataset download —
giving a bit-for-bit identical calibration target. (Other tasks embed a
few-shot prefix in ``prompt`` and are not losslessly invertible, so this script
is intentionally scoped to lambada.)

Pure data munging: no torch, no model, runs locally and is unit-tested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SUPPORTED_TASKS = ("lambada",)


def to_input_record(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a lambada inference *input* record from a reference *output* record.

    The reference output stores the verbatim model prompt under ``prompt``
    (lambada template == question), so we map it back to ``question`` and keep
    ``id`` / ``ground_truth`` for downstream joining and scoring.
    """
    if "prompt" not in record:
        raise KeyError("reference record has no 'prompt' field; cannot reconstruct input")
    gt = record.get("ground_truth", record.get("answer", ""))
    return {
        "id": record.get("id"),
        "question": record["prompt"],
        "ground_truth": gt,
    }


def transform_file(in_path: str, out_path: str) -> int:
    """Reconstruct every input record from the reference output JSONL. Returns count."""
    n = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = to_input_record(json.loads(line))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct lambada calibration input from committed reference output (CPU only).",
    )
    parser.add_argument(
        "--task",
        default="lambada",
        choices=SUPPORTED_TASKS,
        help="Calibration task (only lambada is losslessly reconstructable).",
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
    n = transform_file(args.reference, args.output)
    print(f"[prep_calibration] {args.task}: {n} records -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
