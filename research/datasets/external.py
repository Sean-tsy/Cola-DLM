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


# BFCL parameter docs use Python-ish type names; map them onto JSON Schema
# primitives so Draft 2020-12 validation does not hit UnknownType. ``any``
# maps to None, meaning the ``type`` constraint is dropped.
_BFCL_TYPE_MAP = {
    "dict": "object",
    "tuple": "array",
    "float": "number",
    "any": None,
}


def _sanitize_bfcl_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_sanitize_bfcl_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str) and value in _BFCL_TYPE_MAP:
            mapped = _BFCL_TYPE_MAP[value]
            if mapped is not None:
                out[key] = mapped
            continue
        out[key] = _sanitize_bfcl_schema(value)
    return out


def _schema_from_parameters(params: Any) -> dict[str, Any]:
    params = _sanitize_bfcl_schema(_json_value(params))
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


def _bfcl_prompt(question: Any) -> str:
    """Flatten BFCL v3 ``question`` ([[{role, content}, ...]] turns) to text."""
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        parts: list[str] = []
        for turn in question:
            messages = turn if isinstance(turn, list) else [turn]
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role", "user") == "user" and msg.get("content"):
                        parts.append(str(msg["content"]))
                elif isinstance(msg, str):
                    parts.append(msg)
        return "\n".join(parts)
    return str(question)


def bfcl_to_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert one BFCL row into a tool-call formatting prompt."""
    tools = _tools_from_bfcl(row)
    prompt = _bfcl_prompt(_first(row, ("prompt", "question", "user_query", "instruction"), ""))
    if tools:
        prompt += "\nAvailable tools:\n" + json.dumps(tools, ensure_ascii=False)
    prompt += '\nReturn exactly one JSON object: {"name": <tool name>, "arguments": {...}}'
    ground_truth = _first(row, ("ground_truth", "answer", "possible_answer"), "")
    if not isinstance(ground_truth, str):
        ground_truth = json.dumps(ground_truth, ensure_ascii=False)
    return {
        "id": _first(row, ("id", "idx", "question_id"), idx),
        "prompt": prompt,
        "ground_truth": ground_truth,
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


def load_bfcl_records(
    *,
    category: str = "simple",
    limit: int | None = None,
    repo: str = "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
) -> list[dict[str, Any]]:
    """Load one BFCL v3 category and convert rows to inference records.

    The BFCL repo stores raw JSONL files that ``datasets`` cannot auto-load
    (DataFilesNotFoundError), so this goes through ``hf_hub_download``. The
    matching ``possible_answer/`` file is joined by id into each row's
    ``possible_answer`` so the ground truth survives for later semantic
    scoring; categories without one (e.g. ``rest``) just skip the join.
    """
    from huggingface_hub import hf_hub_download

    def _read(path: str) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    data_path = hf_hub_download(repo, f"BFCL_v3_{category}.json", repo_type="dataset")
    rows = _read(data_path)
    try:
        answer_path = hf_hub_download(repo, f"possible_answer/BFCL_v3_{category}.json", repo_type="dataset")
        answers = {row.get("id"): row.get("ground_truth") for row in _read(answer_path)}
    except Exception:
        answers = {}
    for row in rows:
        if row.get("id") in answers:
            row["possible_answer"] = answers[row["id"]]
    if limit is not None:
        rows = rows[:limit]
    return convert_rows("bfcl", rows)
