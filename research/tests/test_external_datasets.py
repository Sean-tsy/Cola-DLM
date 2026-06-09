"""Tests for external benchmark row adapters (no network)."""

from __future__ import annotations

from research.datasets.external import bfcl_to_record, dycklanguage_to_record, jsonschemabench_to_record


def test_jsonschemabench_adapter_parses_schema_string():
    row = {"unique_id": "s1", "json_schema": '{"type":"object","properties":{"x":{"type":"string"}}}'}
    rec = jsonschemabench_to_record(row, 0)
    assert rec["id"] == "s1"
    assert rec["meta"]["schema"]["type"] == "object"
    assert "JSON Schema" in rec["prompt"]


def test_bfcl_adapter_extracts_tool_registry():
    row = {
        "id": "q1",
        "question": "What is the weather?",
        "function": [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }
    rec = bfcl_to_record(row, 0)
    assert "get_weather" in rec["meta"]["tools"]
    assert '"name"' in rec["prompt"]


def test_dycklanguage_adapter_space_delimits_ground_truth():
    row = {"id": 7, "input": "Complete:", "target": "(()())"}
    rec = dycklanguage_to_record(row, 0)
    assert rec["id"] == 7
    assert rec["ground_truth"] == "( ( ) ( ) )"
