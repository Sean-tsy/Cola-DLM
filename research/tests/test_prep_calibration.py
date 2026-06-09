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

"""Unit tests for research.scripts.prep_calibration (pure-Python; no model)."""

from __future__ import annotations

import json

import pytest

from research.scripts.prep_calibration import to_input_record, transform_file


def test_reconstructs_lambada_input_from_reference_output():
    ref = {
        "id": 1,
        "prompt": "He looked at the",
        "generate": " sky and smiled.",
        "ground_truth": "stars",
        "others": "noise",
    }
    out = to_input_record(ref, "lambada")
    # prompt (verbatim lambada template) becomes the model question
    assert out["question"] == "He looked at the"
    assert out["id"] == 1
    assert out["ground_truth"] == "stars"
    # the reference 'generate'/'others' must NOT leak into the input
    assert "generate" not in out
    assert "others" not in out


def test_falls_back_to_answer_field_for_ground_truth():
    ref = {"id": 2, "prompt": "Q", "answer": "A"}
    out = to_input_record(ref, "lambada")
    assert out["ground_truth"] == "A"


def test_missing_prompt_raises():
    with pytest.raises(KeyError):
        to_input_record({"id": 5, "ground_truth": "x"}, "lambada")


def test_reconstructs_multiple_choice_question_and_choices():
    ref = {
        "id": 0,
        "prompt": "Find the answer.\n(A) red\n(B) blue\nAnswer:",
        "ground_truth": "blue",
        "choices": ["red", "blue"],
    }
    out = to_input_record(ref, "mmlu")
    assert out == {
        "id": 0,
        "question": "Find the answer.",
        "ground_truth": "blue",
        "choices": ["red", "blue"],
    }


def test_reconstructs_race_question_before_options():
    ref = {
        "id": 0,
        "prompt": "Article text plus question?\nOptions:\n(A) x\n(B) y\nAnswer:",
        "ground_truth": "y",
        "choices": ["x", "y"],
    }
    out = to_input_record(ref, "race")
    assert out["question"] == "Article text plus question?"
    assert out["choices"] == ["x", "y"]


def test_reconstructs_squad_context_and_question():
    ref = {
        "id": "abc",
        "prompt": "Context paragraph.\nQuestion: Who won?\nAnswer:",
        "ground_truth": "Ada",
    }
    out = to_input_record(ref, "squad")
    assert out["context"] == "Context paragraph."
    assert out["question"] == "Who won?"


def test_transform_file_roundtrip(tmp_path):
    ref_path = tmp_path / "lambada_ref.jsonl"
    out_path = tmp_path / "sub" / "lambada.jsonl"
    rows = [
        {"id": 1, "prompt": "alpha beta", "generate": " gamma", "ground_truth": "gamma"},
        {"id": 2, "prompt": "delta", "generate": " epsilon", "ground_truth": "epsilon"},
    ]
    ref_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    n = transform_file(str(ref_path), str(out_path), "lambada")
    assert n == 2
    assert out_path.exists()  # parent dir auto-created

    got = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in got] == ["alpha beta", "delta"]
    assert all(set(r.keys()) == {"id", "question", "ground_truth"} for r in got)
