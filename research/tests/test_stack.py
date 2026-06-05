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

"""Unit + fuzz tests for the bracket / Dyck validator and its signals."""

from __future__ import annotations

import random

import pytest

from research.validators.stack import (
    BracketErrorType,
    check_brackets,
    longest_valid_prefix_len,
    repair_distance,
)

_ALPHABET = "()[]{}"


def _random_balanced(rng: random.Random, max_pairs: int = 12) -> str:
    """Generate a random balanced Dyck-3 string."""
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    out: list[str] = []
    stack: list[str] = []
    steps = rng.randint(0, max_pairs)
    for _ in range(steps):
        if stack and rng.random() < 0.5:
            out.append(stack.pop())
        else:
            opening, closing = rng.choice(pairs)
            out.append(opening)
            stack.append(closing)
    while stack:
        out.append(stack.pop())
    return "".join(out)


# --------------------------------------------------------------------------- #
# Boundary / crafted cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s", ["", "()", "[]", "{}", "([{}])", "()[]{}", "(()())"])
def test_balanced_is_valid(s: str) -> None:
    result = check_brackets(s)
    assert result.valid
    assert result.error_position is None
    assert result.error_type is None


def test_unexpected_closing_reports_position() -> None:
    result = check_brackets("ab)c")
    assert not result.valid
    assert result.error_type is BracketErrorType.UNEXPECTED_CLOSING
    assert result.error_position == 2


def test_mismatched_reports_position() -> None:
    result = check_brackets("(]")
    assert not result.valid
    assert result.error_type is BracketErrorType.MISMATCHED
    assert result.error_position == 1


def test_unclosed_points_at_first_open() -> None:
    result = check_brackets("a(b(c)")
    assert not result.valid
    assert result.error_type is BracketErrorType.UNCLOSED
    assert result.error_position == 1  # the outer '(' that never closes


def test_first_error_wins() -> None:
    # An unexpected ')' precedes a later unclosed '(' -> closing error reported.
    result = check_brackets(")(")
    assert result.error_type is BracketErrorType.UNEXPECTED_CLOSING
    assert result.error_position == 0


def test_non_bracket_characters_ignored() -> None:
    assert check_brackets("a = f(x[0]) + {y}").valid


# --------------------------------------------------------------------------- #
# Diagnostic signals — crafted
# --------------------------------------------------------------------------- #


def test_longest_valid_prefix_len_on_errors() -> None:
    assert longest_valid_prefix_len("()x)") == 3  # error at index 3
    assert longest_valid_prefix_len("(]") == 1  # mismatch at index 1
    # Unclosed is still a valid prefix of a longer valid string.
    assert longest_valid_prefix_len("(((") == 3
    assert longest_valid_prefix_len("") == 0


def test_repair_distance_crafted() -> None:
    assert repair_distance("") == 0
    assert repair_distance("()") == 0
    assert repair_distance("(") == 1
    assert repair_distance(")") == 1
    assert repair_distance("(]") == 2
    assert repair_distance("([)]") == 2
    assert repair_distance(")(") == 2


# --------------------------------------------------------------------------- #
# Fuzz / property tests
# --------------------------------------------------------------------------- #


def test_random_balanced_strings_are_valid() -> None:
    rng = random.Random(0)
    for _ in range(500):
        s = _random_balanced(rng)
        result = check_brackets(s)
        assert result.valid, s
        assert repair_distance(s) == 0, s
        assert longest_valid_prefix_len(s) == len(s), s


def test_random_strings_invariants() -> None:
    rng = random.Random(1)
    for _ in range(2000):
        s = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 16)))
        result = check_brackets(s)

        # validity <=> zero repair distance.
        assert result.valid == (repair_distance(s) == 0), s

        # longest valid prefix length is within bounds and consistent.
        lvp = longest_valid_prefix_len(s)
        assert 0 <= lvp <= len(s), s
        if result.error_type in (BracketErrorType.UNEXPECTED_CLOSING, BracketErrorType.MISMATCHED):
            assert lvp == result.error_position, s
        else:
            assert lvp == len(s), s

        # invalid strings always carry a position and a type.
        if not result.valid:
            assert result.error_position is not None
            assert result.error_type is not None


def test_repair_distance_single_type_matches_greedy_reference() -> None:
    """For a single bracket type, repair_distance equals the classic count."""
    rng = random.Random(2)
    for _ in range(1000):
        s = "".join(rng.choice("()") for _ in range(rng.randint(0, 20)))
        # Reference: min add/delete to balance "()" = unmatched ')' + leftover '('.
        open_count = 0
        unmatched_close = 0
        for ch in s:
            if ch == "(":
                open_count += 1
            elif open_count:
                open_count -= 1
            else:
                unmatched_close += 1
        assert repair_distance(s) == unmatched_close + open_count, s
