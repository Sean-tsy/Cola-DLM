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

"""Unit tests for research.scripts.prep_input (pure-Python; no model)."""

from __future__ import annotations

import json

from research.scripts.prep_input import to_infer_record, transform_file


def test_mirrors_prompt_into_question():
    rec = {"id": 3, "prompt": "()()", "ground_truth": "()()", "meta": {"probe": "dyck"}}
    out = to_infer_record(rec)
    assert out["question"] == "()()"
    # original fields are preserved for downstream eval join
    assert out["id"] == 3
    assert out["prompt"] == "()()"
    assert out["ground_truth"] == "()()"
    assert out["meta"] == {"probe": "dyck"}


def test_does_not_mutate_input():
    rec = {"id": 1, "prompt": "x"}
    to_infer_record(rec)
    assert "question" not in rec


def test_preserves_existing_question():
    rec = {"id": 1, "prompt": "x", "question": "already here"}
    assert to_infer_record(rec)["question"] == "already here"


def test_empty_question_is_overwritten():
    rec = {"id": 1, "prompt": "x", "question": ""}
    assert to_infer_record(rec)["question"] == "x"


def test_missing_prompt_raises():
    try:
        to_infer_record({"id": 1})
    except KeyError:
        return
    raise AssertionError("expected KeyError for record without 'prompt'")


def test_transform_file_roundtrip(tmp_path):
    src = tmp_path / "data.jsonl"
    dst = tmp_path / "infer.jsonl"
    rows = [{"id": 0, "prompt": "()"}, {"id": 1, "prompt": "(())"}]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    n = transform_file(str(src), str(dst))
    assert n == 2
    got = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in got] == ["()", "(())"]
