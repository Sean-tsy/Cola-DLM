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

"""Tool-call generator (probe level 3): function signature + intent -> call.

Pure-Python, no model, reproducible (seeded). Each sample pairs a natural-ish
*intent* with the *expected tool call* (``{"name", "arguments"}``) for a known
function signature, plus a single-tool *registry* (function name -> arguments
JSON Schema) suitable for :func:`research.validators.tool_call.check_tool_call`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

# Function library: (name, [(param, type, required), ...]).
_FUNCTIONS: list[tuple[str, list[tuple[str, str, bool]]]] = [
    ("get_weather", [("city", "string", True), ("days", "integer", False)]),
    ("send_email", [("to", "string", True), ("subject", "string", True), ("body", "string", False)]),
    ("set_timer", [("duration_seconds", "integer", True), ("label", "string", False)]),
]

_CITIES = ["Paris", "Rome", "Tokyo", "Berlin", "Cairo"]
_WORDS = ["hello", "update", "reminder", "lunch", "report"]


def _args_schema(params: list[tuple[str, str, bool]]) -> dict:
    properties = {name: {"type": ptype} for name, ptype, _ in params}
    required = [name for name, _, req in params if req]
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _sample_value(rng: random.Random, name: str, ptype: str) -> Any:
    if ptype == "integer":
        return rng.randint(1, 600)
    if name == "city":
        return rng.choice(_CITIES)
    return rng.choice(_WORDS)


def _intent(name: str, args: dict) -> str:
    if name == "get_weather":
        base = f"What's the weather in {args['city']}"
        return base + (f" for the next {args['days']} days?" if "days" in args else "?")
    if name == "send_email":
        base = f"Email {args['to']} with subject '{args['subject']}'"
        return base + (f" and body '{args['body']}'." if "body" in args else ".")
    if name == "set_timer":
        base = f"Set a timer for {args['duration_seconds']} seconds"
        return base + (f" labeled '{args['label']}'." if "label" in args else ".")
    raise ValueError(f"unknown function: {name}")  # pragma: no cover


@dataclass
class ToolCallSample:
    """One tool-call probe sample."""

    id: int
    function_name: str
    registry: dict  # {name: arguments JSON Schema} for the validator
    intent: str
    expected_call: dict  # {"name", "arguments"}
    text: str  # JSON serialization of expected_call (the ground truth)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        signature = json.dumps(self.registry[self.function_name], ensure_ascii=False)
        prompt = (
            f"{self.intent}\n\n"
            f"Available function `{self.function_name}` with arguments schema {signature}. "
            f"Respond with a single JSON tool call of the form "
            f'{{"name": ..., "arguments": ...}}.'
        )
        return {
            "id": self.id,
            "prompt": prompt,
            "ground_truth": self.text,
            "meta": {"probe": "tool_call", "function_name": self.function_name, "registry": self.registry},
        }


def generate_tool_calls(n: int, seed: int) -> list[ToolCallSample]:
    """Generate ``n`` reproducible tool-call samples.

    Same ``(seed, n)`` yields identical output. Every sample's ``text`` is a
    valid call to its ``function_name`` under its ``registry``.
    """
    rng = random.Random(seed)
    samples: list[ToolCallSample] = []
    for i in range(n):
        name, params = rng.choice(_FUNCTIONS)
        args: dict[str, Any] = {}
        for pname, ptype, required in params:
            if required or rng.random() < 0.5:
                args[pname] = _sample_value(rng, pname, ptype)
        expected_call = {"name": name, "arguments": args}
        samples.append(
            ToolCallSample(
                id=i,
                function_name=name,
                registry={name: _args_schema(params)},
                intent=_intent(name, args),
                expected_call=expected_call,
                text=json.dumps(expected_call, ensure_ascii=False),
            )
        )
    return samples
