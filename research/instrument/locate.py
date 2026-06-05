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

"""Mismatch localization for the block-wise sampling loop.

Pure-Python, no model. The instrumented sampling loop decodes one block at a
time (``one_block_ids`` -> text). Feeding those per-block text fragments to a
:class:`BlockMismatchLocator` reconstructs the running output and, using the
Phase-2 validators, reports:

* whether the running output is structurally valid,
* the global character position of the first structural error,
* **which generation block** that error falls in, and
* whether the error sits on a **block boundary** (the first/last character of a
  block) versus a block **interior** position.

The block-vs-boundary attribution is what feeds H3 (块内 vs 块边界 归因): a
diffusion language model that mostly fails at block seams points to the
block-causal transport, whereas interior failures point to within-block decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research.validators.json_schema import JsonStatus, check_json
from research.validators.stack import check_brackets
from research.validators.tool_call import ToolCallStatus, check_tool_call

# Supported structural probes; mirrors research/data_gen probe names.
PROBE_DYCK = "dyck"
PROBE_JSON = "json"
PROBE_TOOL_CALL = "tool_call"


@dataclass
class MismatchLocation:
    """Where (and whether) the running output first violates its structure."""

    valid: bool
    error_position: int | None = None  # global char offset of first error
    block_index: int | None = None  # which generation block it falls in
    block_local_position: int | None = None  # offset within that block
    on_boundary: bool = False  # first/last char of the block
    detail: str | None = None  # validator-specific error label


@dataclass
class BlockMismatchLocator:
    """Accumulates decoded block fragments and localizes the first mismatch.

    Args:
        probe: one of ``"dyck"`` / ``"json"`` / ``"tool_call"``.
        schema: optional JSON Schema (probe ``"json"``).
        tools: optional tool registry ``{name: args_schema}`` (probe ``"tool_call"``).
    """

    probe: str
    schema: dict | None = None
    tools: dict | None = None
    _blocks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.probe not in (PROBE_DYCK, PROBE_JSON, PROBE_TOOL_CALL):
            raise ValueError(f"unknown probe: {self.probe!r}")

    def add_block(self, text: str) -> None:
        """Append one decoded generation block's text fragment."""
        self._blocks.append(text)

    @property
    def text(self) -> str:
        """The running output reconstructed from all blocks so far."""
        return "".join(self._blocks)

    def _block_of(self, position: int) -> tuple[int, int, bool]:
        """Map a global char position to ``(block_index, local_pos, on_boundary)``."""
        cursor = 0
        for idx, block in enumerate(self._blocks):
            block_len = len(block)
            if position < cursor + block_len:
                local = position - cursor
                on_boundary = local == 0 or local == block_len - 1
                return idx, local, on_boundary
            cursor += block_len
        # Position at or past the end (e.g. UNCLOSED on a truncated sequence):
        # attribute it to the last block's trailing boundary.
        last = len(self._blocks) - 1
        last_len = len(self._blocks[last]) if self._blocks else 0
        return last, max(0, last_len - 1), True

    def _raw_error_position(self) -> tuple[bool, int | None, str | None]:
        """Run the probe's validator; return ``(valid, error_position, detail)``."""
        text = self.text
        if self.probe == PROBE_DYCK:
            res = check_brackets(text)
            detail = None if res.valid else res.error_type.value
            return res.valid, res.error_position, detail
        if self.probe == PROBE_JSON:
            res = check_json(text, self.schema)
            valid = res.status is JsonStatus.VALID
            return valid, (None if valid else res.error_position), (None if valid else res.status.value)
        res = check_tool_call(text, self.tools or {})
        valid = res.status is ToolCallStatus.VALID
        return valid, None, (None if valid else res.status.value)

    def locate(self) -> MismatchLocation:
        """Localize the first structural mismatch in the running output."""
        valid, position, detail = self._raw_error_position()
        if valid:
            return MismatchLocation(valid=True)
        if position is None or not self._blocks:
            # Validator flagged an error without a character offset (e.g. a
            # tool-call schema violation). Report invalidity without block
            # attribution rather than guessing.
            return MismatchLocation(valid=False, detail=detail)
        block_index, local, on_boundary = self._block_of(position)
        return MismatchLocation(
            valid=False,
            error_position=position,
            block_index=block_index,
            block_local_position=local,
            on_boundary=on_boundary,
            detail=detail,
        )
