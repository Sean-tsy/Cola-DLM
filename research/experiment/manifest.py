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

"""Run-id keyed results archive + manifest.

Pure-Python. Defines *where every artifact of an experiment lives* so that one
``run_id`` is fully traceable: a manifest snapshotting the config (plus
provenance: git sha, timestamp, tool versions) next to per-kind artifact
subdirectories. Large artifacts (raw sample JSONL, traces) are written here but
kept out of version control by the repo ``.gitignore``; only small structured
results (metrics CSV/JSON, plots) are committed.

This module does not run the model: it only manages the directory contract that
the (local) data generation and the (server-side) sampling/eval write into.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Artifact kinds and their subdirectory under ``results/<run_id>/``.
KINDS = ("data", "samples", "traces", "metrics", "plots", "logs")

MANIFEST_NAME = "manifest.json"


def _git_sha(cwd: str | os.PathLike | None = None) -> str | None:
    """Best-effort current commit SHA for provenance (None if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.fspath(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class ResultsArchive:
    """Filesystem contract for one experiment's artifacts, keyed by ``run_id``.

    Layout::

        <root>/<run_id>/
            manifest.json
            data/      generated probe data (jsonl, gitignored)
            samples/   model outputs from the server (jsonl, gitignored)
            traces/    diagnostics JSONL traces (gitignored)
            metrics/   aggregated structural-consistency metrics (committed)
            plots/     figures (committed if small)
            logs/      job logs (gitignored)
    """

    run_id: str
    root: str = "research/results"

    @property
    def base(self) -> str:
        return os.path.join(self.root, self.run_id)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.base, MANIFEST_NAME)

    def dir(self, kind: str) -> str:
        """Absolute-ish path to an artifact subdirectory (validated kind)."""
        if kind not in KINDS:
            raise ValueError(f"unknown artifact kind {kind!r}; expected one of {KINDS}")
        return os.path.join(self.base, kind)

    def path(self, kind: str, filename: str) -> str:
        """Path to a named artifact under a kind subdirectory."""
        return os.path.join(self.dir(kind), filename)

    def ensure(self) -> ResultsArchive:
        """Create the base + all artifact subdirectories (idempotent)."""
        for kind in KINDS:
            os.makedirs(self.dir(kind), exist_ok=True)
        return self

    def write_manifest(
        self,
        config: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
        git_cwd: str | os.PathLike | None = None,
    ) -> str:
        """Write ``manifest.json`` snapshotting the config + provenance.

        Args:
            config: the experiment description (e.g. ``ExperimentConfig.to_dict()``).
            extra: optional extra provenance (checkpoint hashes, world size, ...).
            git_cwd: repo dir for the git-sha lookup (defaults to cwd).

        Returns the manifest path.
        """
        self.ensure()
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(git_cwd),
            "config": config,
        }
        if extra:
            manifest["provenance"] = extra
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        return self.manifest_path

    def read_manifest(self) -> dict[str, Any]:
        """Load the manifest back (for eval / aggregation)."""
        with open(self.manifest_path, encoding="utf-8") as fh:
            return json.load(fh)
