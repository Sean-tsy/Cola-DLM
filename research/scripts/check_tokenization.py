#!/usr/bin/env python
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

"""SERVER-SIDE 9.1 tokenization prior-check (loads the real tokenizer).

Per env节九 9.1, this MUST run before trusting any structural metric: confirm the
structural symbols ``()[]{}`` (and, for L1 JSON, the quote / comma / colon) are
NOT merged by BPE into multi-character tokens under the OLMo 2 tokenizer. If they
are merged, structural edit-distance / mismatch counts become unreliable and we
must fall back to char/byte level (or force the symbols to standalone tokens),
applied identically to every compared model (§2.3 tokenization trap).

It mirrors exactly how ``cola_dlm.inference`` tokenizes
(``tokenizers.Tokenizer.from_file`` + ``.encode(s).ids``) so the verdict reflects
the production path, not a different HF wrapper.

Gate (exit code):
  * 0  -> PASS: the Dyck bracket set ()[]{} is never merged in context.
  * 1  -> FAIL: at least one Dyck bracket is BPE-merged with an adjacent char.
JSON quote/comma/colon merges are reported as warnings (mitigation deferred to
L1), they do not by themselves fail the Dyck gate.

NEVER run locally (imports the model tokenizer). Not imported by the local
pure-Python test suite.

Usage (server)::

    python -m research.scripts.check_tokenization \
        --tokenizer-path hf_models/tokenizer.json \
        --out research/results/_prelim/tokenization_check.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# OLMo 2 tokenizer vocab size stated by the Cola paper / env节九 9.0.
EXPECTED_VOCAB_SIZE = 100_278

# Brackets whose integrity decides the L0 (Dyck) structural metrics.
DYCK_BRACKETS = ["(", ")", "[", "]", "{", "}"]
# Extra symbols that matter for L1 strict-format (JSON); reported as warnings.
JSON_SYMBOLS = ['"', ",", ":"]
STRUCTURAL_CHARS = set(DYCK_BRACKETS) | set(JSON_SYMBOLS)

# Representative samples exercising the symbols in realistic adjacency (the only
# place BPE merges can appear). Kept tiny and synthetic (no weights needed).
SAMPLES = [
    "( ( ) ( ( ) ) )",  # spaced Dyck-1
    "(()(()))",  # dense Dyck-1 (worst case for bracket merging)
    "([{}])[](){}",  # mixed Dyck-3, adjacent closing/opening
    "{}{}{}",
    "]]][[[",
    '{"a": 1, "b": [2, 3], "c": {"d": "x"}}',  # JSON
    '{"name": "f", "args": {"k": [1, 2], "ok": true}}',  # tool-call-ish
]


def _strip_space_marker(tok: str) -> str:
    """Drop a leading byte-level BPE space marker so 'Ġ(' compares as '('."""
    return tok[1:] if tok[:1] in ("\u0120", "\u2581") and len(tok) > 1 else tok


def _merged_structural_tokens(tokens: list[str]) -> list[str]:
    """Return tokens that contain a structural char but are NOT exactly it.

    A standalone bracket tokenizes as the bare char (optionally with a leading
    space marker). Anything longer that still contains a structural char means
    BPE glued the symbol to a neighbour -> a merge we must flag.
    """
    flagged = []
    for tok in tokens:
        core = _strip_space_marker(tok)
        if len(core) <= 1:
            continue
        if any(ch in STRUCTURAL_CHARS for ch in core):
            flagged.append(tok)
    return flagged


def run_check(tokenizer_path: str) -> dict:
    from tokenizers import Tokenizer  # heavy dep, server-only

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # 1) Each structural char, in isolation, must be a single standalone token.
    single = {}
    single_ok = True
    for ch in sorted(STRUCTURAL_CHARS):
        toks = tokenizer.encode(ch).tokens
        cores = [_strip_space_marker(t) for t in toks]
        ok = toks and all(len(c) <= 1 for c in cores) and "".join(cores) == ch
        single[ch] = {"tokens": toks, "ok": bool(ok)}
        single_ok = single_ok and bool(ok)

    # 2) In realistic context, no structural char may be glued to a neighbour.
    per_sample = []
    merged_chars: set[str] = set()
    for text in SAMPLES:
        enc = tokenizer.encode(text)
        merged = _merged_structural_tokens(enc.tokens)
        for tok in merged:
            for ch in _strip_space_marker(tok):
                if ch in STRUCTURAL_CHARS:
                    merged_chars.add(ch)
        per_sample.append(
            {
                "text": text,
                "n_tokens": len(enc.tokens),
                "tokens": enc.tokens,
                "merged_structural_tokens": merged,
            }
        )

    dyck_merged = sorted(merged_chars & set(DYCK_BRACKETS))
    json_merged = sorted(merged_chars & set(JSON_SYMBOLS))
    dyck_pass = single_ok and not dyck_merged

    return {
        "tokenizer_path": os.path.abspath(tokenizer_path),
        "vocab_size": vocab_size,
        "vocab_size_expected": EXPECTED_VOCAB_SIZE,
        "vocab_size_ok": vocab_size == EXPECTED_VOCAB_SIZE,
        "single_char_tokens": single,
        "single_char_ok": single_ok,
        "samples": per_sample,
        "dyck_merged_chars": dyck_merged,
        "json_merged_chars": json_merged,
        "dyck_pass": dyck_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="env节九 9.1 tokenization prior-check")
    parser.add_argument(
        "--tokenizer-path",
        default="hf_models/tokenizer.json",
        help="Path to tokenizer.json (same file cola_dlm.inference loads).",
    )
    parser.add_argument(
        "--out",
        default="research/results/_prelim/tokenization_check.json",
        help="Where to write the JSON verdict (for the run manifest).",
    )
    args = parser.parse_args()

    report = run_check(args.tokenizer_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"[check_tokenization] tokenizer={report['tokenizer_path']}")
    print(
        f"[check_tokenization] vocab_size={report['vocab_size']} "
        f"(expected {EXPECTED_VOCAB_SIZE}, ok={report['vocab_size_ok']})"
    )
    print(f"[check_tokenization] single_char_ok={report['single_char_ok']}")
    if report["dyck_merged_chars"]:
        print("[check_tokenization] DYCK brackets BPE-merged in context: " f"{report['dyck_merged_chars']}")
    if report["json_merged_chars"]:
        print("[check_tokenization] WARN JSON symbols BPE-merged (L1 mitigation): " f"{report['json_merged_chars']}")
    verdict = "PASS" if report["dyck_pass"] else "FAIL"
    print(f"[check_tokenization] Dyck gate: {verdict} -> {out_path}")
    return 0 if report["dyck_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
