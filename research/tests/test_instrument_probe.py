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

"""Tests for the probe registry, tracing probe, and JSONL trace writer."""

from __future__ import annotations

import json
from pathlib import Path

from research.instrument.probe import (
    SamplingProbe,
    TracingProbe,
    clear_probe,
    get_probe,
    set_probe,
)
from research.instrument.trace import JsonlTraceWriter


def test_default_registry_is_empty() -> None:
    clear_probe()
    assert get_probe() is None


def test_set_and_clear_probe() -> None:
    p = SamplingProbe()
    set_probe(p)
    assert get_probe() is p
    clear_probe()
    assert get_probe() is None


def test_noop_probe_hooks_do_nothing() -> None:
    p = SamplingProbe()
    # None of these should raise.
    p.on_request_start({"block_size": 4})
    p.on_block_start(0, [4])
    p.on_ode_step(0, 0, 1000.0, 900.0, drift_norm=1.0, txt_norm=2.0)
    p.on_block_decoded(0, ["()"])
    p.on_request_finish([{"id": 1}])


def test_jsonl_writer_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "trace.jsonl"  # nested dir must be created
    with JsonlTraceWriter(path) as w:
        w.write({"event": "a", "x": 1})
        w.write({"event": "b", "y": "z"})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["a", "b"]


def test_tracing_probe_full_flow(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with JsonlTraceWriter(path) as w:
        probe = TracingProbe(w, "dyck")
        probe.on_request_start({"block_size": 2, "timestep_num": 16})
        probe.on_block_start(0, [0])
        probe.on_ode_step(0, 0, 1000.0, 875.0, drift_norm=1.5, txt_norm=3.0)
        # Two samples; sample 0 stays valid, sample 1 breaks at the block-1 seam.
        probe.on_block_decoded(0, ["()", "(("])
        probe.on_block_decoded(1, ["[]", "])"])
        probe.on_request_finish([{"id": "s0"}, {"id": "s7"}])

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [r["event"] for r in records]
    assert events.count("ode_step") == 1
    assert events.count("block_decoded") == 2

    mismatches = {r["sample_index"]: r for r in records if r["event"] == "mismatch"}
    # Mismatch events carry the result's id so eval can join across rank shards.
    assert mismatches[0]["sample_id"] == "s0"
    assert mismatches[1]["sample_id"] == "s7"
    assert mismatches[0]["valid"] is True
    assert mismatches[1]["valid"] is False
    # sample 1 text = "((" + "])" -> ']' at the block-1 seam breaks the sequence.
    assert mismatches[1]["block_index"] == 1
    assert mismatches[1]["on_boundary"] is True


def test_read_trace_locations_keys_by_sample_id(tmp_path: Path) -> None:
    """Eval joins traces by sample id: rank-strided batch order != input order."""
    from research.scripts.eval_experiment import _read_trace_locations

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    rows = [
        {"event": "request_start", "block_size": 2},
        {"event": "mismatch", "sample_index": 0, "sample_id": "s7", "valid": False},
        {"event": "mismatch", "sample_index": 1, "sample_id": "s0", "valid": True},
        {"event": "mismatch", "sample_index": 2, "valid": True},  # no id -> skipped
    ]
    (trace_dir / "seed1234_rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (trace_dir / "seed99_rank0.jsonl").write_text(
        json.dumps({"event": "mismatch", "sample_index": 0, "sample_id": "other-seed"}) + "\n",
        encoding="utf-8",
    )

    locations = _read_trace_locations(str(tmp_path), 1234)
    assert set(locations) == {"s7", "s0"}
    assert locations["s7"]["sample_index"] == 0


def test_tracing_probe_handles_no_decoded_blocks(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with JsonlTraceWriter(path) as w:
        probe = TracingProbe(w, "dyck")
        probe.on_request_start({})
        probe.on_request_finish([])  # no block_decoded calls -> no locators
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert any(r["event"] == "request_finish" for r in records)
    assert not any(r["event"] == "mismatch" for r in records)
