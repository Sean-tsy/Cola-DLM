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

"""Append-only JSONL trace writer for sampling-loop instrumentation."""

from __future__ import annotations

import json
import os
from typing import Any, TextIO


class JsonlTraceWriter:
    """Minimal append-only JSONL sink.

    Each :meth:`write` call serializes one record as a single line. Usable as a
    context manager. Pure-Python and model-free so traces can be inspected and
    unit-tested without GPU.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh: TextIO | None = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise ValueError("trace writer is closed")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> JsonlTraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
