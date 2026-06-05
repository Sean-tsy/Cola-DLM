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

"""Sampling-loop instrumentation probe (the core-hook contract).

The upstream block-wise loop calls a *probe* at a few well-defined points so
the diagnostics can locate where structural mismatches appear without changing
any algorithm logic. The contract:

* :meth:`SamplingProbe.on_request_start` — once, with the resolved knobs.
* :meth:`SamplingProbe.on_block_start`   — once per generation block.
* :meth:`SamplingProbe.on_ode_step`      — once per ODE / diffusion step.
* :meth:`SamplingProbe.on_block_decoded` — once per block, with the decoded
  per-sample text (the hook that drives mismatch localization).
* :meth:`SamplingProbe.on_request_finish`— once, with the final results.

All hook payloads are plain Python (floats / ints / strings / lists), so the
core converts tensors to host values before calling. The default
:class:`SamplingProbe` is a **no-op**: when no probe is registered the loop
behaves byte-for-byte as upstream and pays zero cost. See
``research/docs/change_map.md`` (Phase-4 patch plan) for the guarded, default
-off insertion points.
"""

from __future__ import annotations

from typing import Any

from research.instrument.locate import BlockMismatchLocator
from research.instrument.trace import JsonlTraceWriter


class SamplingProbe:
    """No-op base probe. Subclass and override the hooks you need."""

    def on_request_start(self, meta: dict[str, Any]) -> None: ...

    def on_block_start(self, step: int, txt_shape_cum: list[int]) -> None: ...

    def on_ode_step(
        self,
        step: int,
        ode_index: int,
        t_curr: float,
        t_next: float,
        drift_norm: float | None = None,
        txt_norm: float | None = None,
    ) -> None: ...

    def on_block_decoded(self, step: int, block_text: list[str]) -> None: ...

    def on_request_finish(self, results: list[dict[str, Any]]) -> None: ...


# --------------------------------------------------------------------------- #
# Global registry: the core reads the active probe via get_probe().
# --------------------------------------------------------------------------- #

_active: SamplingProbe | None = None


def set_probe(probe: SamplingProbe | None) -> None:
    """Register (or clear, with ``None``) the active probe."""
    global _active
    _active = probe


def get_probe() -> SamplingProbe | None:
    """Return the active probe, or ``None`` when instrumentation is off."""
    return _active


def clear_probe() -> None:
    """Remove any active probe (restores zero-overhead default behavior)."""
    set_probe(None)


class TracingProbe(SamplingProbe):
    """Probe that writes a JSONL trace and localizes per-sample mismatches.

    One :class:`~research.instrument.locate.BlockMismatchLocator` is kept per
    sample so the running structural validity is recomputed each block. The
    final per-sample :class:`~research.instrument.locate.MismatchLocation` is
    emitted on :meth:`on_request_finish`.

    Args:
        writer: sink for trace records.
        probe_kind: structural probe name for the locators (``"dyck"`` / ...).
        schema / tools: optional validator context forwarded to each locator.
    """

    def __init__(
        self,
        writer: JsonlTraceWriter,
        probe_kind: str,
        *,
        schema: dict | None = None,
        tools: dict | None = None,
    ):
        self._w = writer
        self._kind = probe_kind
        self._schema = schema
        self._tools = tools
        self._locators: list[BlockMismatchLocator] = []

    def on_request_start(self, meta: dict[str, Any]) -> None:
        self._w.write({"event": "request_start", **meta})

    def on_block_start(self, step: int, txt_shape_cum: list[int]) -> None:
        self._w.write({"event": "block_start", "step": step, "txt_shape_cum": list(txt_shape_cum)})

    def on_ode_step(
        self,
        step: int,
        ode_index: int,
        t_curr: float,
        t_next: float,
        drift_norm: float | None = None,
        txt_norm: float | None = None,
    ) -> None:
        self._w.write(
            {
                "event": "ode_step",
                "step": step,
                "ode_index": ode_index,
                "t_curr": t_curr,
                "t_next": t_next,
                "drift_norm": drift_norm,
                "txt_norm": txt_norm,
            }
        )

    def on_block_decoded(self, step: int, block_text: list[str]) -> None:
        if not self._locators:
            self._locators = [
                BlockMismatchLocator(self._kind, schema=self._schema, tools=self._tools) for _ in block_text
            ]
        for sample_idx, text in enumerate(block_text):
            self._locators[sample_idx].add_block(text)
        self._w.write({"event": "block_decoded", "step": step, "block_text": list(block_text)})

    def on_request_finish(self, results: list[dict[str, Any]]) -> None:
        for sample_idx, locator in enumerate(self._locators):
            loc = locator.locate()
            self._w.write(
                {
                    "event": "mismatch",
                    "sample_index": sample_idx,
                    "valid": loc.valid,
                    "error_position": loc.error_position,
                    "block_index": loc.block_index,
                    "block_local_position": loc.block_local_position,
                    "on_boundary": loc.on_boundary,
                    "detail": loc.detail,
                }
            )
        self._w.write({"event": "request_finish", "n_results": len(results)})
