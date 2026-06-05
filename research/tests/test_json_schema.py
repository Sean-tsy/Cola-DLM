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

"""Unit + fuzz tests for the JSON / JSON-Schema three-state validator."""

from __future__ import annotations

import json
import random

import pytest

from research.validators.json_schema import JsonStatus, check_json

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}


# --------------------------------------------------------------------------- #
# Three states — crafted
# --------------------------------------------------------------------------- #


def test_valid_without_schema() -> None:
    result = check_json('{"a": 1, "b": [1, 2, 3]}')
    assert result.status is JsonStatus.VALID
    assert result.value == {"a": 1, "b": [1, 2, 3]}


def test_valid_with_schema() -> None:
    result = check_json('{"name": "ada", "age": 36}', _SCHEMA)
    assert result.status is JsonStatus.VALID


def test_parse_error_reports_position() -> None:
    result = check_json('{"name": "ada",}')  # trailing comma
    assert result.status is JsonStatus.PARSE_ERROR
    assert result.error_position is not None
    assert result.error_message


def test_empty_string_is_parse_error() -> None:
    assert check_json("").status is JsonStatus.PARSE_ERROR


def test_schema_error_missing_required() -> None:
    result = check_json('{"age": 36}', _SCHEMA)
    assert result.status is JsonStatus.SCHEMA_ERROR
    assert result.error_message


def test_schema_error_wrong_type_reports_path() -> None:
    result = check_json('{"name": 5}', _SCHEMA)
    assert result.status is JsonStatus.SCHEMA_ERROR
    assert result.error_path == ("name",)


def test_parse_error_takes_priority_over_schema() -> None:
    # Invalid JSON is reported as a parse error even with a schema present.
    assert check_json("{not json}", _SCHEMA).status is JsonStatus.PARSE_ERROR


@pytest.mark.parametrize("text", ["null", "42", "3.14", '"hello"', "true", "[]", "{}"])
def test_bare_json_values_valid_without_schema(text: str) -> None:
    assert check_json(text).status is JsonStatus.VALID


# --------------------------------------------------------------------------- #
# Fuzz / property tests
# --------------------------------------------------------------------------- #


def _random_json_value(rng: random.Random, depth: int = 0):
    if depth >= 3 or rng.random() < 0.3:
        return rng.choice([rng.randint(-100, 100), rng.random(), rng.choice(["a", "bb", ""]), True, None])
    if rng.random() < 0.5:
        return [_random_json_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {f"k{i}": _random_json_value(rng, depth + 1) for i in range(rng.randint(0, 4))}


def test_roundtrip_serialized_json_always_valid() -> None:
    rng = random.Random(0)
    for _ in range(1000):
        text = json.dumps(_random_json_value(rng))
        assert check_json(text).status is JsonStatus.VALID, text


def test_random_garbage_never_crashes() -> None:
    rng = random.Random(1)
    alphabet = '{}[]":,0123 abcn'
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        result = check_json(text, _SCHEMA)
        assert result.status in (JsonStatus.PARSE_ERROR, JsonStatus.SCHEMA_ERROR, JsonStatus.VALID)
        # PARSE_ERROR always carries a position; the others never do.
        assert (result.error_position is not None) == (result.status is JsonStatus.PARSE_ERROR)
