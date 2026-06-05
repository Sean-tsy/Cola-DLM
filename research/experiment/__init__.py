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

"""Experiment layer: one config file == one experiment, archived by run_id.

Pure-Python, model-free. Ties together the data generators, the Phase-4 sweep
config, and a run-id keyed results archive so every experiment is reproducible
from ``config + git commit`` alone.
"""

from research.experiment.config import (
    TASK_DYCK,
    TASK_STRUCTURED,
    TASK_TOOL_CALL,
    ExperimentConfig,
    load_experiment,
)
from research.experiment.manifest import KINDS, MANIFEST_NAME, ResultsArchive

__all__ = [
    "TASK_DYCK",
    "TASK_STRUCTURED",
    "TASK_TOOL_CALL",
    "ExperimentConfig",
    "load_experiment",
    "ResultsArchive",
    "KINDS",
    "MANIFEST_NAME",
]
