# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Adapters for published benchmarks used by the diagnostic study.

The conversion helpers accept plain dictionaries and are unit-testable without
``datasets``. ``load_hf_records`` imports Hugging Face datasets lazily and is
intended for the GPU server only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from research.data_gen.dyck import space_delimit


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _first(row: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def dycklanguage_to_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert a lighteval/DyckLanguage-style row into an inference record."""
    prompt = str(_first(row, ("input", "prompt", "question", "text"), "Complete the Dyck sequence:"))
    answer = str(_first(row, ("output", "target", "answer", "label", "ground_truth"), ""))
    spaced_answer = space_delimit("".join(ch for ch in answer if ch in "()[]{}<>"))
    return {
        "id": _first(row, ("id", "idx", "unique_id"), idx),
        "prompt": "Complete the following Dyck bracket prefix:\n" + prompt,
        "ground_truth": spaced_answer or answer,
        "meta": {
            "probe": "dyck",
            "source": "lighteval/DyckLanguage",
            "mode": "completion",
            "prefix": prompt,
            "raw_row": row,
        },
    }


def jsonschemabench_to_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert one JSONSchemaBench schema row into a structured-output prompt."""
    schema = _json_value(_first(row, ("json_schema", "schema"), {}))
    unique_id = _first(row, ("unique_id", "id"), idx)
    prompt = "Produce a JSON document conforming to this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
    return {
        "id": unique_id,
        "prompt": prompt,
        "ground_truth": "",
        "meta": {
            "probe": "structured",
            "fmt": "json",
            "source": "epfl-dlab/JSONSchemaBench",
            "schema": schema if isinstance(schema, dict) else {},
        },
    }


def _schema_from_parameters(params: Any) -> dict[str, Any]:
    params = _json_value(params)
    if isinstance(params, dict) and params.get("type") == "object":
        return params
    if isinstance(params, dict) and "properties" in params:
        out = {"type": "object", "properties": params.get("properties", {})}
        if "required" in params:
            out["required"] = params["required"]
        return out
    return {"type": "object"}


def _tools_from_bfcl(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_tools = _json_value(_first(row, ("function", "functions", "tools", "function_doc"), []))
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    tools: dict[str, dict[str, Any]] = {}
    if isinstance(raw_tools, list):
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue
            name = _first(tool, ("name", "function_name"), "")
            if not name and isinstance(tool.get("function"), dict):
                name = _first(tool["function"], ("name", "function_name"), "")
                params = tool["function"].get("parameters")
            else:
                params = _first(tool, ("parameters", "arguments", "schema"), {})
            if name:
                tools[str(name)] = _schema_from_parameters(params)
    return tools


def bfcl_to_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert one BFCL row into a tool-call formatting prompt."""
    tools = _tools_from_bfcl(row)
    prompt = str(_first(row, ("prompt", "question", "user_query", "instruction"), ""))
    if tools:
        prompt += "\nAvailable tools:\n" + json.dumps(tools, ensure_ascii=False)
    prompt += '\nReturn exactly one JSON object: {"name": <tool name>, "arguments": {...}}'
    return {
        "id": _first(row, ("id", "idx", "question_id"), idx),
        "prompt": prompt,
        "ground_truth": _first(row, ("ground_truth", "answer", "possible_answer"), ""),
        "meta": {
            "probe": "tool_call",
            "source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            "tools": tools,
            "raw_row": row,
        },
    }


def convert_rows(source: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converters = {
        "dycklanguage": dycklanguage_to_record,
        "jsonschemabench": jsonschemabench_to_record,
        "bfcl": bfcl_to_record,
    }
    key = source.lower()
    if key not in converters:
        raise ValueError(f"unknown external source: {source}")
    return [converters[key](dict(row), idx) for idx, row in enumerate(rows)]


def load_hf_records(
    dataset: str,
    *,
    source: str,
    name: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a Hugging Face dataset and convert rows to inference records."""
    from datasets import load_dataset

    ds = load_dataset(dataset, name, split=split) if name else load_dataset(dataset, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return convert_rows(source, ds)
