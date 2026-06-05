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

"""Tests for the sweep configuration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.instrument.config import (
    NOISE_SEED_ENV,
    TRACE_ENV,
    SweepConfig,
    load_sweep,
    load_sweep_grid,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "sweep"


def test_defaults_match_upstream_inference_kwargs() -> None:
    cfg = SweepConfig()
    kwargs = cfg.to_inference_kwargs()
    assert kwargs["timestep_num"] == 16
    assert kwargs["guidance_scale"] == 7.0
    assert kwargs["top_k"] == 50
    # block_size / patch_size are NOT inference kwargs.
    assert "block_size" not in kwargs
    assert cfg.to_model_overrides() == {}


def test_model_overrides_only_when_set() -> None:
    cfg = SweepConfig(block_size=8, patch_size=2)
    assert cfg.to_model_overrides() == {"block_size": 8, "patch_size": 2}


def test_env_reflects_seed_and_trace() -> None:
    assert SweepConfig().to_env() == {}
    env = SweepConfig(seed=1234, trace=True).to_env()
    assert env[NOISE_SEED_ENV] == "1234"
    assert env[TRACE_ENV] == "1"


def test_roundtrip_dict() -> None:
    cfg = SweepConfig(name="x", seed=7, trace=True, timestep_num=8, block_size=2)
    assert SweepConfig.from_dict(cfg.to_dict()) == cfg


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        SweepConfig.from_dict({"bogus_knob": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestep_num": 0},
        {"T": 0},
        {"max_new_tokens": 0},
        {"block_size": 0},
        {"patch_size": 0},
        {"seed": -1},
    ],
)
def test_validation_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SweepConfig(**kwargs)


def test_load_sweep_grid_files_parse_and_merge_defaults() -> None:
    for fname, knob in [
        ("block_sweep.yaml", "block_size"),
        ("step_sweep.yaml", "timestep_num"),
        ("vae_compression.yaml", "patch_size"),
    ]:
        configs = load_sweep_grid(_CONFIG_DIR / fname)
        assert len(configs) >= 2
        # defaults merged: every point inherits the shared seed + trace.
        for c in configs:
            assert c.seed == 1234
            assert c.trace is True
        # each point varies the swept knob and has a unique name.
        names = [c.name for c in configs]
        assert len(set(names)) == len(names)
        assert any(getattr(c, knob) is not None for c in configs)


def test_load_single_sweep(tmp_path: Path) -> None:
    p = tmp_path / "one.yaml"
    p.write_text("name: solo\nseed: 5\ntimestep_num: 8\nblock_size: 2\n", encoding="utf-8")
    cfg = load_sweep(p)
    assert cfg.name == "solo"
    assert cfg.timestep_num == 8
    assert cfg.to_model_overrides() == {"block_size": 2}


def test_load_sweep_grid_requires_sweep_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("defaults:\n  seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sweep_grid(p)
