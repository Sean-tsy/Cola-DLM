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

"""Pytest bootstrap shared by the pure-Python and the model test suites.

The local / CI developer environment is pure-Python (lint + unit tests for the
``research/`` overlay) and deliberately has **no torch installed**. The upstream
``tests/`` smoke tests import the model package (``cola_dlm`` -> ``torch``), so
when torch is unavailable we skip collecting them rather than erroring out.

On the server (or any environment with torch installed) nothing is ignored and
the full suite — including the model import smoke tests — runs as before.
"""

from __future__ import annotations

from importlib.util import find_spec

# Upstream tests that require torch / the model package to import.
_TORCH_DEPENDENT_TESTS = [
    "tests/test_package.py",
    "tests/test_attention_utils.py",
]

collect_ignore: list[str] = []

if find_spec("torch") is None:
    collect_ignore.extend(_TORCH_DEPENDENT_TESTS)
