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

"""Stack / bracket (Dyck) validator and bracket diagnostic signals.

Pure-Python, no model. Used for evaluation and diagnosis only (NOT as a
post-training reward at this stage).

Public API:

* :func:`check_brackets` — is the sequence balanced, and if not, *where* and
  *what kind* of mismatch occurred.
* :func:`longest_valid_prefix_len` — length of the longest prefix that is still
  a valid Dyck prefix (no committed error).
* :func:`repair_distance` — minimum single-bracket insert/delete edits to
  balance the sequence (a "distance to the nearest valid string" signal).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Mapping ``closing -> opening`` for the bracket alphabet. Generalized Dyck-k.
DEFAULT_PAIRS: dict[str, str] = {")": "(", "]": "[", "}": "{"}


class BracketErrorType(str, Enum):
    """Kind of bracket mismatch reported by :func:`check_brackets`."""

    UNEXPECTED_CLOSING = "unexpected_closing"  # closing bracket with nothing open
    MISMATCHED = "mismatched"  # closing bracket does not match the latest opening
    UNCLOSED = "unclosed"  # input ended while bracket(s) were still open


@dataclass(frozen=True)
class BracketResult:
    """Outcome of :func:`check_brackets`.

    Attributes:
        valid: Whether the sequence is fully balanced.
        error_position: Index of the offending character. For
            ``UNEXPECTED_CLOSING`` / ``MISMATCHED`` it is the index of the
            closing bracket; for ``UNCLOSED`` it is the index of the first
            (left-most) opening bracket that was never closed. ``None`` when
            ``valid``.
        error_type: The :class:`BracketErrorType`, or ``None`` when ``valid``.
    """

    valid: bool
    error_position: int | None
    error_type: BracketErrorType | None


def check_brackets(s: str, pairs: dict[str, str] = DEFAULT_PAIRS) -> BracketResult:
    """Validate a bracket sequence, reporting the first error and its kind.

    Non-bracket characters are ignored, so the checker also works on mixed
    text/structure strings. The first error encountered (scanning left to
    right) is reported.
    """
    openings = set(pairs.values())
    closings = set(pairs.keys())

    stack: list[tuple[str, int]] = []  # (opening char, index)
    for i, ch in enumerate(s):
        if ch in openings:
            stack.append((ch, i))
        elif ch in closings:
            if not stack:
                return BracketResult(False, i, BracketErrorType.UNEXPECTED_CLOSING)
            open_ch, _ = stack.pop()
            if open_ch != pairs[ch]:
                return BracketResult(False, i, BracketErrorType.MISMATCHED)

    if stack:
        _, first_open_idx = stack[0]
        return BracketResult(False, first_open_idx, BracketErrorType.UNCLOSED)

    return BracketResult(True, None, None)


def longest_valid_prefix_len(s: str, pairs: dict[str, str] = DEFAULT_PAIRS) -> int:
    """Length of the longest prefix of ``s`` that is a valid Dyck *prefix*.

    A valid Dyck prefix is one that commits no error: it never closes a bracket
    that is not open and never mismatches. An unclosed-but-otherwise-consistent
    string is itself a valid prefix (it can still be completed), so this returns
    ``len(s)`` whenever the only problem is unclosed brackets.

    Concretely this equals ``error_position`` for ``UNEXPECTED_CLOSING`` /
    ``MISMATCHED`` errors, and ``len(s)`` otherwise.
    """
    result = check_brackets(s, pairs)
    if result.error_type in (BracketErrorType.UNEXPECTED_CLOSING, BracketErrorType.MISMATCHED):
        assert result.error_position is not None
        return result.error_position
    return len(s)


def repair_distance(s: str, pairs: dict[str, str] = DEFAULT_PAIRS) -> int:
    """Minimum single-bracket insert/delete edits to balance ``s``.

    Computed by stack repair: every closing bracket that cannot be matched and
    every opening bracket left unclosed counts as one edit. This is the **exact**
    minimum number of bracket insertions/deletions for a single bracket type,
    and a tight upper bound for mixed bracket types (substitutions not counted).
    Non-bracket characters are ignored. ``0`` iff the sequence is balanced.
    """
    openings = set(pairs.values())
    closings = set(pairs.keys())

    stack: list[str] = []
    edits = 0
    for ch in s:
        if ch in openings:
            stack.append(ch)
        elif ch in closings:
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
            else:
                # Unmatched or mismatched closing: one edit (delete it / insert a
                # matching opening). Keep the stack so a still-open bracket may
                # match a later closing.
                edits += 1

    return edits + len(stack)
