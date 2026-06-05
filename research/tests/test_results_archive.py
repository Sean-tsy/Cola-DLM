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

"""Tests for the run-id keyed results archive + manifest and gen_data driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.experiment.config import ExperimentConfig
from research.experiment.manifest import KINDS, ResultsArchive
from research.scripts.gen_data import generate


def test_archive_paths_and_kinds(tmp_path: Path) -> None:
    arc = ResultsArchive("run1", root=str(tmp_path))
    assert arc.base == str(tmp_path / "run1")
    assert arc.manifest_path == str(tmp_path / "run1" / "manifest.json")
    for kind in KINDS:
        assert arc.dir(kind) == str(tmp_path / "run1" / kind)
    assert arc.path("data", "seed1.jsonl") == str(tmp_path / "run1" / "data" / "seed1.jsonl")


def test_unknown_kind_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ResultsArchive("run1", root=str(tmp_path)).dir("bogus")


def test_ensure_creates_all_subdirs(tmp_path: Path) -> None:
    arc = ResultsArchive("run1", root=str(tmp_path)).ensure()
    for kind in KINDS:
        assert Path(arc.dir(kind)).is_dir()


def test_write_and_read_manifest(tmp_path: Path) -> None:
    cfg = ExperimentConfig(run_id="run1", task="dyck", seeds=[1, 2], data={"length": 8})
    arc = ResultsArchive("run1", root=str(tmp_path))
    path = arc.write_manifest(cfg.to_dict(), extra={"world_size": 8})
    assert Path(path).is_file()
    loaded = arc.read_manifest()
    assert loaded["run_id"] == "run1"
    assert loaded["config"]["task"] == "dyck"
    assert loaded["provenance"]["world_size"] == 8
    assert "created_at" in loaded
    assert "git_sha" in loaded  # present (value may be None off-repo)


def test_gen_data_writes_per_seed_and_manifest(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        run_id="dyckrun",
        task="dyck",
        seeds=[1, 2],
        n_samples=10,
        data={"length": 12, "k": 2},
    )
    base = generate(cfg, results_root=str(tmp_path))
    base_path = Path(base)
    for seed in (1, 2):
        f = base_path / "data" / f"seed{seed}.jsonl"
        assert f.is_file()
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 10
        json.loads(lines[0])  # valid JSONL
    assert (base_path / "manifest.json").is_file()
