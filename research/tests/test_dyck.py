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

"""Tests for the Dyck-k generator, cross-checked with the bracket validator."""

from __future__ import annotations

import pytest

from research.data_gen.dyck import generate_dyck, space_delimit
from research.validators.stack import BracketErrorType, check_brackets, repair_distance

_PAIR_CHARS = set("()[]{}<>")


def _used_types(word: str) -> int:
    openings = [c for c in word if c in "([{<"]
    return len(set(openings))


# --------------------------------------------------------------------------- #
# Validity + knobs
# --------------------------------------------------------------------------- #


def test_unconditional_words_are_valid_and_respect_knobs() -> None:
    samples = generate_dyck(50, seed=0, length=16, max_depth=3, k=2)
    assert len(samples) == 50
    for s in samples:
        word = s.full()
        assert check_brackets(word).valid, word
        assert repair_distance(word) == 0
        assert len(word) == 16
        assert s.prefix == ""
        assert s.max_depth <= 3
        assert _used_types(word) <= 2


def test_max_pairing_distance_is_respected() -> None:
    for cap in (1, 2, 5):
        samples = generate_dyck(50, seed=cap, length=20, max_depth=6, k=3, max_pairing_distance=cap)
        for s in samples:
            assert check_brackets(s.full()).valid
            assert s.max_pairing_distance <= cap, (cap, s.full())


def test_distance_cap_one_forces_flat_sequences() -> None:
    # cap == 1 means only adjacent "()" pairs -> depth 1 everywhere.
    samples = generate_dyck(20, seed=1, length=12, max_depth=5, k=2, max_pairing_distance=1)
    for s in samples:
        assert s.max_depth == 1
        assert s.max_pairing_distance == 1


def test_completion_mode_prefix_is_unclosed_and_completes() -> None:
    samples = generate_dyck(50, seed=2, length=16, max_depth=4, k=2, mode="completion")
    for s in samples:
        assert s.prefix != ""
        assert s.completion != ""
        # The prefix is a valid Dyck prefix but has unclosed brackets.
        pre = check_brackets(s.prefix)
        assert not pre.valid
        assert pre.error_type is BracketErrorType.UNCLOSED
        # prefix + completion is a full valid word.
        assert check_brackets(s.full()).valid
        assert len(s.full()) == 16


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_reproducible_same_seed() -> None:
    a = generate_dyck(20, seed=7, length=14, max_depth=4, k=3)
    b = generate_dyck(20, seed=7, length=14, max_depth=4, k=3)
    assert a == b


def test_different_seed_differs() -> None:
    a = generate_dyck(20, seed=1, length=14, max_depth=4, k=3)
    b = generate_dyck(20, seed=2, length=14, max_depth=4, k=3)
    assert a != b


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 3},  # odd
        {"length": 0},  # too small
        {"length": 8, "max_depth": 0},
        {"length": 8, "k": 0},
        {"length": 8, "k": 5},
        {"length": 8, "max_pairing_distance": 0},
        {"length": 8, "mode": "bogus"},
    ],
)
def test_invalid_params_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        generate_dyck(1, seed=0, **kwargs)


def test_to_record_shape() -> None:
    rec = generate_dyck(1, seed=0, length=8, max_depth=3, k=2, mode="completion")[0].to_record()
    assert set(rec) == {"id", "prompt", "ground_truth", "meta"}
    assert rec["meta"]["probe"] == "dyck"
    assert rec["meta"]["mode"] == "completion"


def test_space_delimit_helper() -> None:
    assert space_delimit("(())") == "( ( ) )"
    assert space_delimit("(") == "("
    assert space_delimit("") == ""


def test_to_record_is_space_delimited() -> None:
    # env节九 9.1: structural symbols must be space-delimited (1 token/bracket).
    for mode in ("unconditional", "completion"):
        rec = generate_dyck(1, seed=1, length=8, max_depth=3, k=2, mode=mode)[0].to_record()
        gt = rec["ground_truth"]
        brackets = [c for c in gt if c in _PAIR_CHARS]
        # Every bracket is isolated by whitespace: no two brackets are adjacent.
        assert " ".join(brackets) == gt
        # Spacing does not break structural validity (validator ignores spaces).
        assert check_brackets(rec["meta"]["prefix"] + "".join(brackets)).valid
        assert rec["meta"]["spacing"] == "space-delimited"
