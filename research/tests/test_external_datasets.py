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


def test_bfcl_adapter_handles_v3_row_shape():
    """Real BFCL_v3_simple.json row: nested question turns + python-ish types."""
    row = {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Find the area of a triangle."}]],
        "function": [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate the area of a triangle given its base and height.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "factor": {"type": "float"},
                        "options": {"type": "dict"},
                        "anything": {"type": "any"},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
        "possible_answer": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    }
    rec = bfcl_to_record(row, 0)
    assert rec["id"] == "simple_0"
    assert rec["prompt"].startswith("Find the area of a triangle.")
    schema = rec["meta"]["tools"]["calculate_triangle_area"]
    # Python-ish types are mapped to JSON Schema primitives ("any" drops type)
    # so Draft 2020-12 validation cannot hit UnknownType.
    assert schema["type"] == "object"
    assert schema["properties"]["factor"]["type"] == "number"
    assert schema["properties"]["options"]["type"] == "object"
    assert "type" not in schema["properties"]["anything"]
    assert schema["required"] == ["base", "height"]
    # joined possible_answer survives as a JSON string ground truth
    assert '"calculate_triangle_area"' in rec["ground_truth"]

    from research.validators.tool_call import ToolCallStatus, check_tool_call

    call = '{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}'
    assert check_tool_call(call, rec["meta"]["tools"]).status is ToolCallStatus.VALID


def test_dycklanguage_adapter_space_delimits_ground_truth():
    row = {"id": 7, "input": "[ [ ]", "output": " ]"}
    rec = dycklanguage_to_record(row, 0)
    assert rec["id"] == 7
    assert rec["ground_truth"] == "]"
    assert rec["meta"]["mode"] == "completion"
    assert rec["meta"]["prefix"] == "[ [ ]"
