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

"""Unified experiment configuration: one config file == one experiment.

Pure-Python (no torch / no model). A single :class:`ExperimentConfig` is the
*one and only* description of an experiment, capturing everything the project
convention requires:

* ``run_id``        — unique, traceable experiment identifier.
* ``task`` + ``data`` — probe type and its data-generation knobs (Dyck
  ``length`` / ``max_depth`` / ``k`` / ..., structured ``fmt`` / ..., tool-call).
* ``block_size`` / ``timestep_num`` (+ decode knobs) — sampling configuration,
  composed into a Phase-4 :class:`~research.instrument.config.SweepConfig`.
* ``checkpoint``   — DiT / VAE / tokenizer paths as ``${VAR}`` placeholders
  (resolved server-side; never absolute paths in version control).
* ``seeds``        — the random-seed *set* the experiment is replicated over.

The same config drives both the (local, CPU) data generation and the
(server-side, GPU) sampling/eval, so an experiment is fully reproducible from
``config + git commit`` alone. Materializing data is pure-Python; running the
model is not done here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from research.instrument.config import SweepConfig

# Supported probe tasks and the data_gen entrypoint each maps to.
TASK_DYCK = "dyck"
TASK_STRUCTURED = "structured"
TASK_TOOL_CALL = "tool_call"
_TASKS = (TASK_DYCK, TASK_STRUCTURED, TASK_TOOL_CALL)


@dataclass
class ExperimentConfig:
    """The single source of truth for one diagnostic experiment."""

    run_id: str
    task: str
    seeds: list[int]
    n_samples: int = 256

    # Task-specific data-generation knobs (forwarded to the data_gen fn).
    # Dyck: length / max_depth / k / max_pairing_distance / mode.
    # Structured: fmt / max_depth / max_props.  Tool-call: (none).
    data: dict[str, Any] = field(default_factory=dict)

    # Sampling / model configuration (composed into a SweepConfig per seed).
    block_size: int | None = None
    patch_size: int | None = None
    timestep_num: int = 16
    guidance_scale: float = 7.0
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    trace: bool = True

    # Checkpoint placeholders (resolved server-side via env / ${VAR}).
    checkpoint: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string")
        if any(c in self.run_id for c in "/\\ "):
            raise ValueError("run_id must not contain path separators or spaces")
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}")
        if not self.seeds:
            raise ValueError("seeds must be a non-empty list")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any((not isinstance(s, int)) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be non-negative integers")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.task == TASK_TOOL_CALL and self.data:
            raise ValueError("tool_call task takes no data knobs")
        # Validate sampling/model knobs by constructing a probe SweepConfig.
        self.sweep_for(self.seeds[0])

    # ----------------------------------------------------------------- #
    # Composition
    # ----------------------------------------------------------------- #
    def run_id_for(self, seed: int) -> str:
        """Per-seed run identifier (the unit archived under results/)."""
        return f"{self.run_id}.s{seed}"

    def sweep_for(self, seed: int) -> SweepConfig:
        """Build the Phase-4 sweep config for one seed of this experiment."""
        return SweepConfig(
            name=self.run_id_for(seed),
            seed=seed,
            trace=self.trace,
            task_name=self.task,
            timestep_num=self.timestep_num,
            guidance_scale=self.guidance_scale,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            block_size=self.block_size,
            patch_size=self.patch_size,
        )

    def materialize_data(self, seed: int) -> list[dict[str, Any]]:
        """Generate this experiment's probe data for one seed (pure CPU).

        Returns inference-ready JSONL records (``id`` / ``prompt`` /
        ``ground_truth`` / ``meta``). Same ``(config, seed)`` -> identical data.
        """
        if self.task == TASK_DYCK:
            from research.data_gen.dyck import generate_dyck

            samples = generate_dyck(self.n_samples, seed, **self.data)
        elif self.task == TASK_STRUCTURED:
            from research.data_gen.structured import generate_structured

            samples = generate_structured(self.n_samples, seed, **self.data)
        else:  # TASK_TOOL_CALL
            from research.data_gen.tool_call import generate_tool_calls

            samples = generate_tool_calls(self.n_samples, seed)
        return [s.to_record() for s in samples]

    # ----------------------------------------------------------------- #
    # Serialization
    # ----------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Diffable description of the whole experiment (manifest payload)."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "seeds": list(self.seeds),
            "n_samples": self.n_samples,
            "data": dict(self.data),
            "sampling": {
                "block_size": self.block_size,
                "patch_size": self.patch_size,
                "timestep_num": self.timestep_num,
                "guidance_scale": self.guidance_scale,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "trace": self.trace,
            },
            "checkpoint": dict(self.checkpoint),
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> ExperimentConfig:
        """Build from a parsed YAML mapping (flat or ``sampling:``-nested)."""
        flat: dict[str, Any] = {}
        for key, value in doc.items():
            if key == "sampling" and isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        valid = set(cls.__dataclass_fields__)
        unknown = set(flat) - valid
        if unknown:
            raise ValueError(f"unknown experiment config keys: {sorted(unknown)}")
        return cls(**flat)


def load_experiment(path: str | os.PathLike) -> ExperimentConfig:
    """Load and validate a single experiment YAML."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"experiment config must be a mapping, got {type(doc).__name__}")
    return ExperimentConfig.from_dict(doc)
