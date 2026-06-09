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

"""Tests for the unified ExperimentConfig (one config == one experiment)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.experiment.config import ExperimentConfig, load_experiment
from research.instrument.config import SweepConfig
from research.validators.stack import check_brackets

_EXP_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def _base(**over) -> ExperimentConfig:
    kwargs = dict(run_id="exp1", task="dyck", seeds=[1, 2], data={"length": 8, "k": 2})
    kwargs.update(over)
    return ExperimentConfig(**kwargs)


def test_example_configs_load() -> None:
    yamls = sorted(_EXP_DIR.glob("*.yaml"))
    assert yamls, "no experiment configs found"
    for path in yamls:
        cfg = load_experiment(path)
        assert cfg.run_id
        assert cfg.seeds
        # run_id must equal the file stem by convention (traceability).
        assert cfg.run_id == path.stem


def test_smoke_config_is_minimal_and_valid() -> None:
    cfg = load_experiment(_EXP_DIR / "smoke_dyck1.yaml")
    # Minimal end-to-end validation knobs: shortest Dyck-1, fewest steps.
    assert cfg.task == "dyck"
    assert cfg.data["k"] == 1
    assert cfg.timestep_num <= 4
    assert cfg.n_samples <= 16
    assert len(cfg.seeds) == 1
    # It must still produce valid balanced Dyck words.
    for rec in cfg.materialize_data(cfg.seeds[0]):
        word = rec["meta"]["prefix"] + rec["ground_truth"]
        assert check_brackets(word).valid


def test_sweep_for_composes_seed_and_name() -> None:
    cfg = _base(block_size=4, timestep_num=8)
    sw = cfg.sweep_for(2)
    assert isinstance(sw, SweepConfig)
    assert sw.seed == 2
    assert sw.name == "exp1.s2"
    assert sw.block_size == 4
    assert sw.timestep_num == 8
    assert sw.task_name == "dyck"
    # trace flag flows through to the env switch.
    assert sw.to_env()["COLA_DIAG_TRACE"] == "1"


def test_run_id_for() -> None:
    assert _base().run_id_for(7) == "exp1.s7"


def test_materialize_data_reproducible_and_valid() -> None:
    cfg = _base(data={"length": 16, "max_depth": 3, "k": 2})
    a = cfg.materialize_data(1)
    b = cfg.materialize_data(1)
    assert a == b
    assert len(a) == cfg.n_samples
    # cross-check with the validator: full Dyck word is balanced.
    for rec in a:
        word = rec["meta"]["prefix"] + rec["ground_truth"]
        assert check_brackets(word).valid


def test_materialize_structured_and_tool_call() -> None:
    s = ExperimentConfig(run_id="s", task="structured", seeds=[1], n_samples=5, data={"fmt": "xml"})
    assert len(s.materialize_data(1)) == 5
    t = ExperimentConfig(run_id="t", task="tool_call", seeds=[1], n_samples=5)
    assert len(t.materialize_data(1)) == 5


def test_roundtrip_dict() -> None:
    cfg = _base(block_size=8, checkpoint={"dit": "${DIT}"})
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "over",
    [
        {"run_id": ""},
        {"run_id": "has space"},
        {"run_id": "has/slash"},
        {"task": "bogus"},
        {"seeds": []},
        {"seeds": [1, 1]},
        {"seeds": [-1]},
        {"n_samples": 0},
        {"timestep_num": 0},
        {"block_size": 0},
    ],
)
def test_validation_raises(over: dict) -> None:
    with pytest.raises(ValueError):
        _base(**over)


def test_tool_call_rejects_data() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(run_id="t", task="tool_call", seeds=[1], data={"x": 1})


def test_tool_call_accepts_external_source_data() -> None:
    cfg = ExperimentConfig(run_id="t", task="tool_call", seeds=[1], data={"source": "bfcl", "split": "train"})
    assert cfg.data["source"] == "bfcl"


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"run_id": "x", "task": "dyck", "seeds": [1], "bogus": 1})
