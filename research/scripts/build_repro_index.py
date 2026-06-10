# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build the 7.10 reproducibility index over all archived runs (CPU-only).

Walks ``research/results/*/manifest.json`` and joins, per run: git sha,
creation time, seeds, a canonical sha256 of the config snapshot, the headline
metrics (``metrics/summary.json`` for diagnostic runs, ``calibration_verdict``
/ ``accuracy_summary.csv`` for upstream-harness calibration runs), and the
checkpoint fingerprints from ``research/results/weights.sha256``. Output is a
single small JSON committed next to the runs.

Usage::

    python -m research.scripts.build_repro_index
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any

_SUMMARY_KEYS = (
    "n",
    "valid_rate",
    "valid_rate_ci95_low",
    "valid_rate_ci95_high",
    "mean_repair_distance",
    "boundary_error_rate",
    "mean_vae_ppl",
)


def _config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _calibration_metrics(base: str) -> dict[str, Any] | None:
    """Headline metrics for the upstream-harness calibration archives."""
    out: dict[str, Any] = {}
    verdict = _read_json(os.path.join(base, "calibration_verdict.json"))
    if verdict:
        out["calibration_verdict"] = verdict
    acc_path = os.path.join(base, "accuracy_summary.csv")
    if os.path.isfile(acc_path):
        # Header is ``task,<run_id>``: the value column is named after the run.
        with open(acc_path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            out["accuracy"] = {row[0]: float(row[1]) for row in reader if len(row) > 1 and row[1].strip()}
    return out or None


def build_index(results_root: str) -> dict[str, Any]:
    weights_path = os.path.join(results_root, "weights.sha256")
    weights: dict[str, str] = {}
    if os.path.isfile(weights_path):
        with open(weights_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    digest, name = line.split(None, 1)
                    weights[name.strip()] = digest

    runs = []
    for entry in sorted(os.listdir(results_root)):
        base = os.path.join(results_root, entry)
        manifest = _read_json(os.path.join(base, "manifest.json"))
        if manifest is None:
            continue
        # Diagnostic manifests snapshot the experiment under "config";
        # calibration manifests carry "settings"/"tasks" at top level instead.
        config = manifest.get("config") or {k: manifest[k] for k in ("settings", "tasks", "harness") if k in manifest}
        run = {
            "run_id": manifest.get("run_id", entry),
            "git_sha": manifest.get("git_sha"),
            "created_at": manifest.get("created_at"),
            "seeds": config.get("seeds"),
            "config_sha256": _config_sha256(config),
        }
        summary = _read_json(os.path.join(base, "metrics", "summary.json"))
        if summary is not None:
            run["metrics"] = {k: summary.get(k) for k in _SUMMARY_KEYS}
            run["by_error_type"] = summary.get("by_error_type")
        else:
            calib = _calibration_metrics(base)
            if calib is not None:
                run["metrics"] = calib
        runs.append(run)

    return {"checkpoint_sha256": weights, "runs": runs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the reproducibility index over archived runs.")
    parser.add_argument("--results-root", default="research/results")
    parser.add_argument("--output", default=None, help="Default: <results-root>/reproducibility_index.json")
    args = parser.parse_args(argv)

    index = build_index(args.results_root)
    out_path = args.output or os.path.join(args.results_root, "reproducibility_index.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[build_repro_index] {len(index['runs'])} runs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
