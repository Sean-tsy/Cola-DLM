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

"""Tests for the JSON/XML generator, cross-checked with the JSON validator."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from research.data_gen.structured import generate_structured
from research.validators.json_schema import JsonStatus, check_json


def test_json_samples_validate_against_their_schema() -> None:
    samples = generate_structured(100, seed=0, fmt="json", max_depth=2, max_props=4)
    assert len(samples) == 100
    for s in samples:
        assert s.fmt == "json"
        result = check_json(s.text, s.schema)
        assert result.status is JsonStatus.VALID, (s.text, s.schema)


def test_xml_samples_are_well_formed_with_required_elements() -> None:
    samples = generate_structured(100, seed=0, fmt="xml", max_depth=2, max_props=4)
    for s in samples:
        root = ET.fromstring(s.text)  # well-formed -> no parse error
        for name in s.schema.get("required", []):
            assert root.find(name) is not None, (name, s.text)


def test_reproducible_same_seed() -> None:
    a = generate_structured(30, seed=5, fmt="json")
    b = generate_structured(30, seed=5, fmt="json")
    assert a == b


def test_different_seed_differs() -> None:
    a = generate_structured(30, seed=1, fmt="json")
    b = generate_structured(30, seed=2, fmt="json")
    assert a != b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fmt": "yaml"},
        {"fmt": "json", "max_depth": -1},
        {"fmt": "json", "max_props": 0},
    ],
)
def test_invalid_params_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        generate_structured(1, seed=0, **kwargs)


def test_to_record_shape() -> None:
    rec = generate_structured(1, seed=0, fmt="xml")[0].to_record()
    assert set(rec) == {"id", "prompt", "ground_truth", "meta"}
    assert rec["meta"]["probe"] == "structured"
    assert rec["meta"]["fmt"] == "xml"
