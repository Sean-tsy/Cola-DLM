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

"""Sweep configuration layer for the Cola DLM diagnostics.

Pure-Python (no torch), so it loads and validates locally in the test env.
A :class:`SweepConfig` captures every knob the diagnostic sweeps need and
splits them into three buckets matching *where* they apply in the upstream
inference path:

* ``sampling`` — keyword arguments accepted directly by
  ``cola_dlm.inference.generate_task_repaint_inference`` (e.g. ``timestep_num``,
  ``guidance_scale``, decode temperature). Passed through unchanged.
* ``model_overrides`` — attributes that live on the *loaded model configs*
  rather than the inference call: ``block_size`` and ``patch_size`` (latent
  compression). These are **not** function kwargs and are **not** safe naive
  attribute writes: ``patch_size`` is baked into the VAE Conv/Linear shapes, so
  a sweep over it selects a *matching checkpoint*, not ``vae.patch_size = N``;
  ``block_size`` must be applied to **both** ``dit`` and ``vae`` (both build
  block-causal masks) with an equality assertion. See
  ``research/docs/change_map.md`` (Phase-4 patch plan) for the server-side
  application rules.
* ``env`` — opt-in environment switches the upstream loop already honours
  (e.g. ``COLA_INFER_PER_SAMPLE_NOISE_SEED``) plus the diagnostics trace switch.

Nothing here imports or runs the model: it only prepares a reproducible,
diffable description of one sweep point. Actually applying it (loading weights,
sampling) happens server-side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Environment switch that turns the sampling-loop instrumentation on. The
# upstream loop stays byte-for-byte identical when this is unset; see
# ``research/docs/change_map.md`` (Phase-4 patch plan) for the guarded hooks.
TRACE_ENV = "COLA_DIAG_TRACE"
# Path the instrumented loop writes its JSONL trace to when tracing is on.
TRACE_PATH_ENV = "COLA_DIAG_TRACE_PATH"
# Upstream per-sample deterministic noise seed (already supported in the loop).
NOISE_SEED_ENV = "COLA_INFER_PER_SAMPLE_NOISE_SEED"

# Knobs forwarded verbatim as inference kwargs, with upstream defaults.
_SAMPLING_DEFAULTS: dict[str, Any] = {
    "task_name": "lambada",
    "T": 1000.0,
    "timestep_num": 16,
    "guidance_scale": 7.0,
    "max_new_tokens": 32,
    "temperature": 0.0,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


@dataclass
class SweepConfig:
    """One reproducible sweep point.

    Reproducibility: a non-``None`` ``seed`` is surfaced through the upstream
    ``COLA_INFER_PER_SAMPLE_NOISE_SEED`` switch so the same ``(seed, knobs)``
    yields the same per-sample noise regardless of batch/world size.
    """

    name: str = "default"
    seed: int | None = None
    trace: bool = False

    # --- sampling kwargs (forwarded to generate_task_repaint_inference) ---
    task_name: str = "lambada"
    T: float = 1000.0
    timestep_num: int = 16  # ODE / diffusion integration steps  (step sweep)
    guidance_scale: float = 7.0
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # --- model-config overrides (applied to loaded dit / vae server-side) ---
    block_size: int | None = None  # dit.block_size            (H3 block sweep)
    patch_size: int | None = None  # vae.patch_size / latent compression (H5)

    def __post_init__(self) -> None:
        if self.timestep_num < 1:
            raise ValueError("timestep_num must be >= 1")
        if self.T <= 0:
            raise ValueError("T must be > 0")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.patch_size is not None and self.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0")

    def to_inference_kwargs(self) -> dict[str, Any]:
        """Return the subset forwarded verbatim to the inference entrypoint."""
        return {key: getattr(self, key) for key in _SAMPLING_DEFAULTS}

    def to_model_overrides(self) -> dict[str, int]:
        """Return config attributes the server applies to ``dit`` / ``vae``.

        Only includes knobs explicitly set (non-``None``); an empty dict means
        "use the loaded model's own defaults".
        """
        overrides: dict[str, int] = {}
        if self.block_size is not None:
            overrides["block_size"] = self.block_size
        if self.patch_size is not None:
            overrides["patch_size"] = self.patch_size
        return overrides

    def to_env(self) -> dict[str, str]:
        """Return environment switches to export before a server-side run."""
        env: dict[str, str] = {}
        if self.seed is not None:
            env[NOISE_SEED_ENV] = str(self.seed)
        if self.trace:
            env[TRACE_ENV] = "1"
        return env

    def to_dict(self) -> dict[str, Any]:
        """Flat, diffable description of the sweep point (for results archive)."""
        return {
            "name": self.name,
            "seed": self.seed,
            "trace": self.trace,
            "sampling": self.to_inference_kwargs(),
            "model_overrides": self.to_model_overrides(),
            "env": self.to_env(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepConfig:
        """Build from a (possibly nested) mapping such as a parsed YAML doc.

        Accepts either a flat mapping (keys == field names) or the nested
        ``{name, seed, trace, sampling: {...}, model_overrides: {...}}`` shape
        produced by :meth:`to_dict`. Unknown keys raise ``ValueError`` so typos
        in a config file fail loudly instead of being silently ignored.
        """
        flat: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("sampling", "model_overrides") and isinstance(value, dict):
                flat.update(value)
            elif key == "env":
                continue  # derived from seed/trace, not an input field
            else:
                flat[key] = value

        valid = {f for f in cls.__dataclass_fields__}
        unknown = set(flat) - valid
        if unknown:
            raise ValueError(f"unknown sweep config keys: {sorted(unknown)}")
        return cls(**flat)


def load_sweep(path: str | os.PathLike) -> SweepConfig:
    """Load and validate a single sweep YAML into a :class:`SweepConfig`."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"sweep config must be a mapping, got {type(data).__name__}")
    return SweepConfig.from_dict(data)


def load_sweep_grid(path: str | os.PathLike) -> list[SweepConfig]:
    """Load a sweep *grid* file: a top-level ``sweep:`` list of points.

    Optional top-level ``defaults:`` mapping is merged under every point so
    shared knobs are written once. Returns one validated config per point.
    """
    import yaml

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict) or "sweep" not in doc:
        raise ValueError("sweep grid file must have a top-level 'sweep:' list")
    defaults = doc.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping")

    configs: list[SweepConfig] = []
    for point in doc["sweep"]:
        if not isinstance(point, dict):
            raise ValueError("each sweep point must be a mapping")
        merged = {**defaults, **point}
        configs.append(SweepConfig.from_dict(merged))
    return configs
