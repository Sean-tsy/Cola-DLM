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

"""Scaffold smoke tests for the ``research/`` diagnostic overlay.

These run without any GPU / model checkpoint and only verify that the
research-layer package skeleton (data generators, validators, eval) is
importable and that the directory skeleton is present. They give the
overlay a green baseline before the data-gen / validator / eval modules
are filled in by later milestones.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = _REPO_ROOT / "research"

_EXPECTED_DIRS = [
    "configs",
    "data_gen",
    "validators",
    "eval",
    "scripts",
    "results",
    "docs",
]


@pytest.mark.parametrize("subdir", _EXPECTED_DIRS)
def test_research_skeleton_dirs_exist(subdir: str) -> None:
    assert (_RESEARCH_ROOT / subdir).is_dir(), f"missing research/{subdir}"


@pytest.mark.parametrize("module", ["research", "research.data_gen", "research.validators", "research.eval"])
def test_research_packages_importable(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_no_post_training_dirs_yet() -> None:
    """Current scope is diagnosis/eval only — no SFT/DPO/RL dirs should exist."""
    forbidden = {"sft", "dpo", "rl", "rlhf", "rlvr", "post_training", "posttraining", "train"}
    present = {p.name.lower() for p in _RESEARCH_ROOT.iterdir() if p.is_dir()}
    assert not (present & forbidden), f"unexpected post-training dirs: {sorted(present & forbidden)}"
