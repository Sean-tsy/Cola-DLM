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

"""Dyck-k bracket sequence generator (probe level 1).

Pure-Python, no model, reproducible (seeded). Produces balanced Dyck-k words
with controllable length, maximum nesting depth, number of bracket types, and
an optional cap on per-pair *pairing distance* (close index - open index).

Two modes:

* ``"unconditional"`` — the sample is a full valid Dyck word.
* ``"completion"``    — the sample is split into an unclosed-open ``prefix`` and
  a reference ``completion`` that balances it (conditional completion probe).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Bracket alphabet (closing matched to opening). Dyck-k uses the first k pairs.
PAIRS: list[tuple[str, str]] = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
MAX_K = len(PAIRS)


@dataclass
class DyckSample:
    """One Dyck-k probe sample.

    ``prefix + completion`` is always a full valid Dyck word; for the
    unconditional mode ``prefix`` is empty.
    """

    id: int
    mode: str  # "unconditional" | "completion"
    prefix: str
    completion: str
    k: int
    length: int  # length of the full word (prefix + completion)
    max_depth: int  # measured max nesting depth of the full word
    max_pairing_distance: int  # measured max (close index - open index)
    meta: dict = field(default_factory=dict)

    def full(self) -> str:
        return self.prefix + self.completion

    def to_record(self) -> dict:
        """Render to an inference-style JSONL record (id / prompt / ground_truth)."""
        if self.mode == "completion":
            prompt = (
                f"Continue the following bracket sequence so every bracket is correctly "
                f"closed (Dyck-{self.k}):\n{self.prefix}"
            )
        else:
            prompt = (
                f"Generate a valid Dyck-{self.k} bracket sequence of length {self.length} "
                f"with maximum nesting depth {self.max_depth}."
            )
        return {
            "id": self.id,
            "prompt": prompt,
            "ground_truth": self.completion,
            "meta": {
                "probe": "dyck",
                "mode": self.mode,
                "k": self.k,
                "length": self.length,
                "max_depth": self.max_depth,
                "max_pairing_distance": self.max_pairing_distance,
                "prefix": self.prefix,
            },
        }


def _measure(word: str) -> tuple[int, int]:
    """Return ``(max_depth, max_pairing_distance)`` of a valid Dyck word."""
    opens = {o for o, _ in PAIRS}
    stack: list[int] = []  # open indices
    max_depth = 0
    max_dist = 0
    for i, ch in enumerate(word):
        if ch in opens:
            stack.append(i)
            max_depth = max(max_depth, len(stack))
        else:
            open_i = stack.pop()
            max_dist = max(max_dist, i - open_i)
    return max_depth, max_dist


def _generate_word(
    rng: random.Random,
    n_pairs: int,
    max_depth: int,
    k: int,
    max_pairing_distance: int | None,
) -> str:
    """Generate one valid Dyck-k word with exactly ``n_pairs`` pairs."""
    out: list[str] = []
    stack: list[int] = []  # bracket-type index of each open bracket
    open_positions: list[int] = []  # positions, parallel to ``stack``
    opens_used = 0
    pos = 0
    total = 2 * n_pairs

    while len(out) < total:
        can_open = opens_used < n_pairs and len(stack) < max_depth

        # Pairing-distance feasibility: opening now must still let the oldest
        # open bracket close within the cap once we drain (close top-first).
        if can_open and max_pairing_distance is not None and stack:
            bottom = open_positions[0]
            predicted = (pos + 1 - bottom) + len(stack)  # distance of bottom after draining
            if predicted > max_pairing_distance:
                can_open = False

        # Must we start draining to keep the oldest bracket within the cap?
        must_close = False
        if max_pairing_distance is not None and stack:
            bottom = open_positions[0]
            if (pos - bottom) + (len(stack) - 1) >= max_pairing_distance:
                must_close = True

        can_close = len(stack) > 0
        open_now = can_open and not must_close and (not can_close or rng.random() < 0.6)

        if open_now:
            t = rng.randrange(k)
            out.append(PAIRS[t][0])
            stack.append(t)
            open_positions.append(pos)
            opens_used += 1
        else:
            t = stack.pop()
            open_positions.pop()
            out.append(PAIRS[t][1])
        pos += 1

    return "".join(out)


def _depth_profile(word: str) -> list[int]:
    """Running nesting depth after each character."""
    opens = {o for o, _ in PAIRS}
    depth = 0
    profile: list[int] = []
    for ch in word:
        depth += 1 if ch in opens else -1
        profile.append(depth)
    return profile


def generate_dyck(
    n: int,
    seed: int,
    *,
    length: int,
    max_depth: int = 4,
    k: int = 1,
    max_pairing_distance: int | None = None,
    mode: str = "unconditional",
) -> list[DyckSample]:
    """Generate ``n`` reproducible Dyck-k samples.

    Args:
        n: Number of samples.
        seed: RNG seed; same ``(seed, params, n)`` yields identical output.
        length: Length of the full valid word (must be even and >= 2).
        max_depth: Maximum nesting depth (cap, >= 1).
        k: Number of bracket types used (1..4).
        max_pairing_distance: Optional cap on ``close_index - open_index`` (>= 1).
        mode: ``"unconditional"`` or ``"completion"``.
    """
    if length < 2 or length % 2 != 0:
        raise ValueError("length must be an even integer >= 2")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be in 1..{MAX_K}")
    if max_pairing_distance is not None and max_pairing_distance < 1:
        raise ValueError("max_pairing_distance must be >= 1")
    if mode not in ("unconditional", "completion"):
        raise ValueError("mode must be 'unconditional' or 'completion'")

    rng = random.Random(seed)
    n_pairs = length // 2
    samples: list[DyckSample] = []

    for i in range(n):
        word = _generate_word(rng, n_pairs, max_depth, k, max_pairing_distance)
        depth, dist = _measure(word)

        if mode == "completion":
            profile = _depth_profile(word)
            # Valid cut points: 1..len-1 where the prefix still has open brackets.
            cuts = [c for c in range(1, len(word)) if profile[c - 1] > 0]
            cut = rng.choice(cuts)
            prefix, completion = word[:cut], word[cut:]
        else:
            prefix, completion = "", word

        samples.append(
            DyckSample(
                id=i,
                mode=mode,
                prefix=prefix,
                completion=completion,
                k=k,
                length=length,
                max_depth=depth,
                max_pairing_distance=dist,
            )
        )

    return samples
