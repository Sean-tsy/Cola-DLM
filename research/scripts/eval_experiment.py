# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evaluate one diagnostic experiment archive (CPU-only post-processing)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

from research.eval.metrics import EvalRecord, evaluate_pair, read_jsonl, summarize
from research.experiment import ResultsArchive, load_experiment


def _index_by_id(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row.get("id"): row for row in rows}


def _read_trace_locations(base: str, seed: int) -> dict[int, dict[str, Any]]:
    locations: dict[int, dict[str, Any]] = {}
    trace_dir = os.path.join(base, "traces")
    if not os.path.isdir(trace_dir):
        return locations
    for name in sorted(os.listdir(trace_dir)):
        if not (name.startswith(f"seed{seed}_") and name.endswith(".jsonl")):
            continue
        for row in read_jsonl(os.path.join(trace_dir, name)):
            if row.get("event") == "mismatch" and "sample_index" in row:
                locations[len(locations)] = row
    return locations


def evaluate_archive(config_path: str, results_root: str) -> tuple[list[EvalRecord], dict[str, Any]]:
    cfg = load_experiment(config_path)
    archive = ResultsArchive(cfg.run_id, root=results_root).ensure()
    rows: list[EvalRecord] = []

    for seed in cfg.seeds:
        inputs = read_jsonl(archive.path("data", f"seed{seed}.jsonl"))
        samples = read_jsonl(archive.path("samples", f"seed{seed}.jsonl"))
        by_id = _index_by_id(samples)
        traces = _read_trace_locations(archive.base, seed)
        for idx, input_record in enumerate(inputs):
            sample = by_id.get(input_record.get("id"))
            if sample is None:
                sample = {}
            rows.append(
                evaluate_pair(
                    input_record,
                    sample,
                    task=cfg.task,
                    seed=seed,
                    trace_location=traces.get(idx),
                )
            )

    return rows, summarize(rows)


def write_outputs(rows: list[EvalRecord], summary: dict[str, Any], archive: ResultsArchive) -> None:
    metrics_dir = archive.dir("metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)

    with open(os.path.join(metrics_dir, "per_sample.csv"), "w", encoding="utf-8", newline="") as fh:
        fieldnames = list(rows[0].to_dict()) if rows else list(EvalRecord(0, "", "", False, 0.0).to_dict())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one research experiment archive.")
    parser.add_argument("config", help="Experiment YAML.")
    parser.add_argument("--results-root", default="research/results")
    args = parser.parse_args(argv)

    cfg = load_experiment(args.config)
    archive = ResultsArchive(cfg.run_id, root=args.results_root)
    rows, summary = evaluate_archive(args.config, args.results_root)
    write_outputs(rows, summary, archive)
    print(
        f"[eval_experiment] {cfg.run_id}: n={summary['n']} "
        f"valid_rate={summary['valid_rate']:.4f} -> {archive.dir('metrics')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
