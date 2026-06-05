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

"""Unit + fuzz tests for the tool-call validator."""

from __future__ import annotations

import json
import random

from research.validators.tool_call import ToolCallStatus, check_tool_call

# Registry with a nested-structure argument schema to exercise nested checks.
_TOOLS = {
    "get_weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    },
    "send_email": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "body": {
                "type": "object",
                "properties": {"subject": {"type": "string"}, "text": {"type": "string"}},
                "required": ["subject", "text"],
            },
        },
        "required": ["to", "body"],
    },
}


def _call(name, arguments) -> str:
    return json.dumps({"name": name, "arguments": arguments})


# --------------------------------------------------------------------------- #
# States — crafted
# --------------------------------------------------------------------------- #


def test_valid_call() -> None:
    result = check_tool_call(_call("get_weather", {"city": "Paris", "days": 3}), _TOOLS)
    assert result.status is ToolCallStatus.VALID
    assert result.function_name == "get_weather"


def test_valid_nested_call() -> None:
    text = _call("send_email", {"to": "a@b.com", "body": {"subject": "hi", "text": "yo"}})
    assert check_tool_call(text, _TOOLS).status is ToolCallStatus.VALID


def test_parse_error() -> None:
    assert check_tool_call("{not json", _TOOLS).status is ToolCallStatus.PARSE_ERROR


def test_envelope_schema_error_missing_name() -> None:
    result = check_tool_call(json.dumps({"arguments": {}}), _TOOLS)
    assert result.status is ToolCallStatus.SCHEMA_ERROR


def test_envelope_schema_error_arguments_not_object() -> None:
    result = check_tool_call(json.dumps({"name": "get_weather", "arguments": []}), _TOOLS)
    assert result.status is ToolCallStatus.SCHEMA_ERROR


def test_unknown_function() -> None:
    result = check_tool_call(_call("teleport", {"city": "Paris"}), _TOOLS)
    assert result.status is ToolCallStatus.UNKNOWN_FUNCTION
    assert result.function_name == "teleport"


def test_missing_required_argument() -> None:
    result = check_tool_call(_call("get_weather", {"days": 3}), _TOOLS)
    assert result.status is ToolCallStatus.MISSING_ARGUMENT


def test_missing_nested_required_argument() -> None:
    text = _call("send_email", {"to": "a@b.com", "body": {"subject": "hi"}})  # no 'text'
    result = check_tool_call(text, _TOOLS)
    assert result.status is ToolCallStatus.MISSING_ARGUMENT
    assert "body" in result.error_path


def test_invalid_argument_type() -> None:
    result = check_tool_call(_call("get_weather", {"city": "Paris", "days": "three"}), _TOOLS)
    assert result.status is ToolCallStatus.INVALID_ARGUMENT
    assert result.error_path == ("days",)


def test_invalid_nested_argument_type() -> None:
    text = _call("send_email", {"to": "a@b.com", "body": {"subject": 1, "text": "yo"}})
    result = check_tool_call(text, _TOOLS)
    assert result.status is ToolCallStatus.INVALID_ARGUMENT
    assert result.error_path == ("body", "subject")


# --------------------------------------------------------------------------- #
# Fuzz / property tests
# --------------------------------------------------------------------------- #


def test_well_formed_random_calls_are_valid() -> None:
    rng = random.Random(0)
    for _ in range(500):
        args = {"city": rng.choice(["Paris", "Rome", ""])}
        if rng.random() < 0.5:
            args["days"] = rng.randint(0, 14)
        assert check_tool_call(_call("get_weather", args), _TOOLS).status is ToolCallStatus.VALID


def test_random_garbage_never_crashes() -> None:
    rng = random.Random(1)
    alphabet = '{}[]":,0123 nametargusciy'
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
        result = check_tool_call(text, _TOOLS)
        assert result.status in set(ToolCallStatus)
