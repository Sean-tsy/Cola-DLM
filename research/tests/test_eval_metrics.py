"""Tests for structural experiment metrics."""

from __future__ import annotations

from research.eval.metrics import evaluate_pair, summarize


def test_evaluates_valid_dyck_generation():
    inp = {"id": 1, "meta": {"probe": "dyck", "length": 4, "max_depth": 2, "max_pairing_distance": 3}}
    row = evaluate_pair(inp, {"id": 1, "generate": "(())"}, task="dyck", seed=0)
    assert row.valid
    assert row.repair_distance == 0


def test_evaluates_completion_with_prefix():
    inp = {"id": 2, "meta": {"probe": "dyck", "mode": "completion", "prefix": "(("}}
    row = evaluate_pair(inp, {"id": 2, "generate": "))"}, task="dyck", seed=0)
    assert row.valid


def test_evaluates_json_schema_status():
    inp = {
        "id": "j",
        "meta": {"schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}},
    }
    ok = evaluate_pair(inp, {"generate": '{"x":"a"}'}, task="structured", seed=0)
    bad = evaluate_pair(inp, {"generate": '{"x":1}'}, task="structured", seed=0)
    assert ok.valid
    assert not bad.valid
    assert bad.error_type == "schema_error"


def test_summary_has_bootstrap_ci_and_groups():
    rows = [
        evaluate_pair({"id": 1, "meta": {"length": 4}}, {"generate": "()()"}, task="dyck", seed=0),
        evaluate_pair({"id": 2, "meta": {"length": 4}}, {"generate": "(()"}, task="dyck", seed=1),
    ]
    summary = summarize(rows)
    assert summary["n"] == 2
    assert summary["valid_rate"] == 0.5
    assert summary["by_seed"]["0"]["valid_rate"] == 1.0
