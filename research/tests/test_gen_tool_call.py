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

"""Tests for the tool-call generator, cross-checked with the tool-call validator."""

from __future__ import annotations

from research.data_gen.tool_call import generate_tool_calls
from research.validators.tool_call import ToolCallStatus, check_tool_call


def test_generated_calls_are_valid() -> None:
    samples = generate_tool_calls(100, seed=0)
    assert len(samples) == 100
    for s in samples:
        result = check_tool_call(s.text, s.registry)
        assert result.status is ToolCallStatus.VALID, (s.text, s.registry)
        assert result.function_name == s.function_name


def test_required_arguments_always_present() -> None:
    samples = generate_tool_calls(100, seed=1)
    for s in samples:
        schema = s.registry[s.function_name]
        for req in schema.get("required", []):
            assert req in s.expected_call["arguments"]


def test_intent_is_nonempty() -> None:
    for s in generate_tool_calls(20, seed=2):
        assert isinstance(s.intent, str) and s.intent


def test_reproducible_same_seed() -> None:
    assert generate_tool_calls(30, seed=9) == generate_tool_calls(30, seed=9)


def test_different_seed_differs() -> None:
    assert generate_tool_calls(30, seed=1) != generate_tool_calls(30, seed=2)


def test_to_record_shape() -> None:
    rec = generate_tool_calls(1, seed=0)[0].to_record()
    assert set(rec) == {"id", "prompt", "ground_truth", "meta"}
    assert rec["meta"]["probe"] == "tool_call"
    # The ground-truth record validates against the registry it ships with.
    assert check_tool_call(rec["ground_truth"], rec["meta"]["registry"]).status is ToolCallStatus.VALID
