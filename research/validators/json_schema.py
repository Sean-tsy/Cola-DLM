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

"""JSON / JSON-Schema validator with an explicit three-state outcome.

Pure-Python, no model. Used for evaluation and diagnosis only (NOT as a
post-training reward at this stage).

Three states (:class:`JsonStatus`):

* ``PARSE_ERROR``  — the text is not valid JSON at all.
* ``SCHEMA_ERROR`` — valid JSON, but it violates the supplied schema.
* ``VALID``        — valid JSON (and, if a schema is given, conformant).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match


class JsonStatus(str, Enum):
    """Three-state result of :func:`check_json`."""

    PARSE_ERROR = "parse_error"
    SCHEMA_ERROR = "schema_error"
    VALID = "valid"


@dataclass(frozen=True)
class JsonResult:
    """Outcome of :func:`check_json`.

    Attributes:
        status: One of :class:`JsonStatus`.
        error_message: Human-readable error for the non-valid states.
        error_position: Character offset of the JSON syntax error
            (``PARSE_ERROR`` only); ``None`` otherwise. Doubles as a diagnostic
            signal for how far parsing progressed before failing.
        error_path: JSON path (tuple of keys/indices) to the schema violation
            (``SCHEMA_ERROR`` only); ``None`` otherwise.
        value: The parsed JSON value when parsing succeeded; ``None`` on
            ``PARSE_ERROR``.
    """

    status: JsonStatus
    error_message: str | None = None
    error_position: int | None = None
    error_path: tuple[Any, ...] | None = None
    value: Any | None = None


def check_json(text: str, schema: dict[str, Any] | None = None) -> JsonResult:
    """Validate JSON text, optionally against a JSON Schema.

    Returns a :class:`JsonResult` whose ``status`` distinguishes parse failure,
    schema violation, and full validity.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return JsonResult(JsonStatus.PARSE_ERROR, error_message=exc.msg, error_position=exc.pos)

    if schema is None:
        return JsonResult(JsonStatus.VALID, value=value)

    error = best_match(Draft202012Validator(schema).iter_errors(value))
    if error is not None:
        return JsonResult(
            JsonStatus.SCHEMA_ERROR,
            error_message=error.message,
            error_path=tuple(error.absolute_path),
            value=value,
        )

    return JsonResult(JsonStatus.VALID, value=value)
