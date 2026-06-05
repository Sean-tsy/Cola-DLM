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

"""Materialize an experiment's probe data and seed its results archive.

Pure-Python / CPU-only — safe to run locally. Reads one experiment config,
generates the probe data for every seed (reproducibly), writes it as JSONL into
``results/<run_id>/data/`` and writes the run manifest. Does NOT run the model;
sampling/eval happen server-side (see ``run_experiment.sh``).

Usage::

    python -m research.scripts.gen_data research/configs/experiments/dyck_L32_D6_k2_blk4.yaml
    python -m research.scripts.gen_data <config.yaml> --results-root research/results
"""

from __future__ import annotations

import argparse
import json
import sys

from research.experiment import ExperimentConfig, ResultsArchive, load_experiment


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate(config: ExperimentConfig, results_root: str) -> str:
    """Generate data for every seed and write the manifest. Returns base dir."""
    archive = ResultsArchive(config.run_id, root=results_root).ensure()
    for seed in config.seeds:
        records = config.materialize_data(seed)
        out = archive.path("data", f"seed{seed}.jsonl")
        write_jsonl(records, out)
        print(f"[gen_data] {config.run_id} seed={seed}: {len(records)} samples -> {out}")
    manifest = archive.write_manifest(config.to_dict())
    print(f"[gen_data] manifest -> {manifest}")
    return archive.base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize experiment probe data (CPU only).")
    parser.add_argument("config", help="Path to an experiment YAML config.")
    parser.add_argument(
        "--results-root",
        default="research/results",
        help="Root directory for the run-id keyed results archive.",
    )
    args = parser.parse_args(argv)

    config = load_experiment(args.config)
    generate(config, args.results_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
