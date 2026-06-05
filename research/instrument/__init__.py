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

"""Diagnostics instrumentation layer for Cola DLM (overlay, model-free).

Config (sweep knobs) + a no-op sampling-loop probe contract + mismatch
localization. The actual guarded insertion points into the upstream loop are
documented in ``research/docs/change_map.md`` (Phase-4 patch plan); this package
provides the harness those hooks call.
"""

from research.instrument.config import (
    NOISE_SEED_ENV,
    TRACE_ENV,
    TRACE_PATH_ENV,
    SweepConfig,
    load_sweep,
    load_sweep_grid,
)
from research.instrument.locate import (
    PROBE_DYCK,
    PROBE_JSON,
    PROBE_TOOL_CALL,
    BlockMismatchLocator,
    MismatchLocation,
)
from research.instrument.probe import (
    SamplingProbe,
    TracingProbe,
    clear_probe,
    get_probe,
    set_probe,
)
from research.instrument.trace import JsonlTraceWriter

__all__ = [
    "NOISE_SEED_ENV",
    "TRACE_ENV",
    "TRACE_PATH_ENV",
    "SweepConfig",
    "load_sweep",
    "load_sweep_grid",
    "PROBE_DYCK",
    "PROBE_JSON",
    "PROBE_TOOL_CALL",
    "BlockMismatchLocator",
    "MismatchLocation",
    "SamplingProbe",
    "TracingProbe",
    "clear_probe",
    "get_probe",
    "set_probe",
    "JsonlTraceWriter",
]
