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

"""Tool-call validator: function name, required arguments, nested structure.

Pure-Python, no model. Used for evaluation and diagnosis only (NOT as a
post-training reward at this stage).

Builds on :mod:`research.validators.json_schema`: a tool call must first be
valid JSON, then match the call *envelope* ``{"name": str, "arguments":
object}``, then the named function must exist and its arguments must satisfy the
function's JSON Schema (required fields and nested structure included).

States (:class:`ToolCallStatus`):

* ``PARSE_ERROR``      — not valid JSON.
* ``SCHEMA_ERROR``     — valid JSON but not a well-formed call envelope.
* ``UNKNOWN_FUNCTION`` — ``name`` is not in the supplied tool registry.
* ``MISSING_ARGUMENT`` — a required argument is absent.
* ``INVALID_ARGUMENT`` — an argument violates the schema (wrong type / nested).
* ``VALID``            — a well-formed call to a known tool with valid args.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from .json_schema import JsonStatus, check_json

# Call envelope: a JSON object with a string ``name`` and an object ``arguments``.
_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["name", "arguments"],
}


class ToolCallStatus(str, Enum):
    """Result state of :func:`check_tool_call`."""

    PARSE_ERROR = "parse_error"
    SCHEMA_ERROR = "schema_error"
    UNKNOWN_FUNCTION = "unknown_function"
    MISSING_ARGUMENT = "missing_argument"
    INVALID_ARGUMENT = "invalid_argument"
    VALID = "valid"


@dataclass(frozen=True)
class ToolCallResult:
    """Outcome of :func:`check_tool_call`.

    Attributes:
        status: One of :class:`ToolCallStatus`.
        function_name: The requested ``name`` when it could be read; else ``None``.
        error_message: Human-readable error for the non-valid states.
        error_path: JSON path (relative to ``arguments``) of the offending
            field for ``MISSING_ARGUMENT`` / ``INVALID_ARGUMENT``; ``None``
            otherwise.
    """

    status: ToolCallStatus
    function_name: str | None = None
    error_message: str | None = None
    error_path: tuple[Any, ...] | None = None


def check_tool_call(text: str, tools: dict[str, dict[str, Any]]) -> ToolCallResult:
    """Validate a single tool call against a registry of tool argument schemas.

    Args:
        text: The raw tool-call string (expected to be JSON).
        tools: Registry mapping ``function name -> JSON Schema`` for that
            function's ``arguments`` object.
    """
    # 1) Must be valid JSON and match the call envelope.
    envelope = check_json(text, _ENVELOPE_SCHEMA)
    if envelope.status is JsonStatus.PARSE_ERROR:
        return ToolCallResult(ToolCallStatus.PARSE_ERROR, error_message=envelope.error_message)
    if envelope.status is JsonStatus.SCHEMA_ERROR:
        return ToolCallResult(ToolCallStatus.SCHEMA_ERROR, error_message=envelope.error_message)

    call = envelope.value
    name = call["name"]
    arguments = call["arguments"]

    # 2) The function must be known.
    if name not in tools:
        return ToolCallResult(
            ToolCallStatus.UNKNOWN_FUNCTION,
            function_name=name,
            error_message=f"unknown function: {name!r}",
        )

    # 3) Arguments must satisfy the function schema (required + nested structure).
    error = best_match(Draft202012Validator(tools[name]).iter_errors(arguments))
    if error is not None:
        status = ToolCallStatus.MISSING_ARGUMENT if error.validator == "required" else ToolCallStatus.INVALID_ARGUMENT
        return ToolCallResult(
            status,
            function_name=name,
            error_message=error.message,
            error_path=tuple(error.absolute_path),
        )

    return ToolCallResult(ToolCallStatus.VALID, function_name=name)
