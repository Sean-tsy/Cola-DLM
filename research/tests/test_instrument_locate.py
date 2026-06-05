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

"""Tests for block-wise mismatch localization."""

from __future__ import annotations

import pytest

from research.instrument.locate import BlockMismatchLocator


def test_valid_dyck_across_blocks() -> None:
    loc = BlockMismatchLocator("dyck")
    for frag in ["()", "[]", "{}"]:
        loc.add_block(frag)
    result = loc.locate()
    assert result.valid
    assert result.error_position is None


def test_interior_mismatch_attribution() -> None:
    # Blocks: "((" | "))" -> with a bad swap "(]" in block 1 interior.
    loc = BlockMismatchLocator("dyck")
    loc.add_block("([")  # block 0:  positions 0,1
    loc.add_block(")]")  # block 1:  positions 2,3  -> ")" mismatches "["
    result = loc.locate()
    assert not result.valid
    # First error is the ")" at global position 2, start of block 1 -> boundary.
    assert result.error_position == 2
    assert result.block_index == 1
    assert result.block_local_position == 0
    assert result.on_boundary is True


def test_boundary_vs_interior() -> None:
    # block0 = "(", block1 = "[)]" -> global "([)]" (positions 0,1,2,3).
    # ')' at global pos 2 mismatches '[' ; block1 covers positions 1,2,3 so the
    # error is local pos 1 of a length-3 block -> interior (not a seam).
    loc = BlockMismatchLocator("dyck")
    loc.add_block("(")
    loc.add_block("[)]")
    result = loc.locate()
    assert not result.valid
    assert result.error_position == 2
    assert result.block_index == 1
    assert result.block_local_position == 1
    assert result.on_boundary is False


def test_non_bracket_chars_are_valid() -> None:
    loc = BlockMismatchLocator("dyck")
    loc.add_block("xxx")  # bracket checker ignores non-bracket characters
    assert loc.locate().valid


def test_unclosed_attributes_to_last_block() -> None:
    loc = BlockMismatchLocator("dyck")
    loc.add_block("((")
    loc.add_block("[]")  # still one '(' unclosed
    result = loc.locate()
    assert not result.valid
    assert result.detail == "unclosed"
    # error_position points at the leftmost unclosed '(' (global pos 0, block 0).
    assert result.block_index == 0


def test_json_probe_valid_and_invalid() -> None:
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    good = BlockMismatchLocator("json", schema=schema)
    good.add_block('{"a":')
    good.add_block(" 1}")
    assert good.locate().valid

    bad = BlockMismatchLocator("json", schema=schema)
    bad.add_block('{"a":')
    bad.add_block(' "no"}')  # wrong type -> schema error (no char offset)
    res = bad.locate()
    assert not res.valid
    assert res.detail == "schema_error"


def test_tool_call_probe() -> None:
    tools = {"f": {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}}
    good = BlockMismatchLocator("tool_call", tools=tools)
    good.add_block('{"name": "f",')
    good.add_block(' "arguments": {"x": 1}}')
    assert good.locate().valid

    bad = BlockMismatchLocator("tool_call", tools=tools)
    bad.add_block('{"name": "f", "arguments": {}}')
    res = bad.locate()
    assert not res.valid
    assert res.detail == "missing_argument"


def test_unknown_probe_raises() -> None:
    with pytest.raises(ValueError):
        BlockMismatchLocator("bogus")


def test_text_property_reconstructs() -> None:
    loc = BlockMismatchLocator("dyck")
    loc.add_block("()")
    loc.add_block("[]")
    assert loc.text == "()[]"
