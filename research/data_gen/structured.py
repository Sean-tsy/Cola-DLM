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

"""JSON / XML structured-format generator (probe level 2).

Pure-Python, no model, reproducible (seeded). Generates a random JSON Schema
(object with typed, possibly nested, possibly required properties) together with
a conforming instance, serialized as JSON or XML.

The JSON instances validate against their schema via
:func:`research.validators.json_schema.check_json`; the XML serialization is a
well-formed element tree carrying the same required structure.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

_LEAF_TYPES = ["string", "integer", "number", "boolean"]
_WORDS = ["alpha", "beta", "gamma", "delta", "paris", "rome", "tokyo", "ab", "x"]


@dataclass
class StructuredSample:
    """One JSON/XML structured-format probe sample."""

    id: int
    fmt: str  # "json" | "xml"
    schema: dict  # logical JSON Schema describing the required structure
    value: Any  # the conforming Python value
    text: str  # serialized document (JSON or XML)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        prompt = f"Produce a {self.fmt.upper()} document conforming to this JSON Schema:\n" + json.dumps(
            self.schema, ensure_ascii=False
        )
        return {
            "id": self.id,
            "prompt": prompt,
            "ground_truth": self.text,
            "meta": {"probe": "structured", "fmt": self.fmt, "schema": self.schema},
        }


def _gen_schema(rng: random.Random, max_depth: int, max_props: int) -> dict:
    n = rng.randint(1, max_props)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for i in range(n):
        name = f"f{i}"
        allow_object = max_depth > 0
        choices = _LEAF_TYPES + (["object"] if allow_object else [])
        t = rng.choice(choices)
        if t == "object":
            properties[name] = _gen_schema(rng, max_depth - 1, max_props)
        else:
            properties[name] = {"type": t}
        if rng.random() < 0.6:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _gen_instance(rng: random.Random, schema: dict) -> Any:
    t = schema["type"]
    if t == "object":
        required = set(schema.get("required", []))
        obj: dict[str, Any] = {}
        for name, sub in schema["properties"].items():
            if name in required or rng.random() < 0.5:
                obj[name] = _gen_instance(rng, sub)
        return obj
    if t == "string":
        return rng.choice(_WORDS)
    if t == "integer":
        return rng.randint(-50, 50)
    if t == "number":
        return round(rng.uniform(-50, 50), 3)
    if t == "boolean":
        return rng.choice([True, False])
    raise ValueError(f"unsupported type: {t}")  # pragma: no cover


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_xml(value: Any, tag: str) -> ET.Element:
    elem = ET.Element(tag)
    if isinstance(value, dict):
        for key, sub in value.items():
            elem.append(_to_xml(sub, key))
    else:
        elem.text = _scalar_text(value)
    return elem


def generate_structured(
    n: int,
    seed: int,
    *,
    fmt: str = "json",
    max_depth: int = 2,
    max_props: int = 4,
) -> list[StructuredSample]:
    """Generate ``n`` reproducible JSON/XML schema-constrained samples.

    Args:
        n: Number of samples.
        seed: RNG seed; same ``(seed, params, n)`` yields identical output.
        fmt: ``"json"`` or ``"xml"``.
        max_depth: Maximum nesting depth of the schema (>= 0).
        max_props: Maximum properties per object (>= 1).
    """
    if fmt not in ("json", "xml"):
        raise ValueError("fmt must be 'json' or 'xml'")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if max_props < 1:
        raise ValueError("max_props must be >= 1")

    rng = random.Random(seed)
    samples: list[StructuredSample] = []
    for i in range(n):
        schema = _gen_schema(rng, max_depth, max_props)
        value = _gen_instance(rng, schema)
        if fmt == "json":
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = ET.tostring(_to_xml(value, "object"), encoding="unicode")
        samples.append(StructuredSample(id=i, fmt=fmt, schema=schema, value=value, text=text))
    return samples
