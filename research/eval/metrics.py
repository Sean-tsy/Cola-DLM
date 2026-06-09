# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Structural-consistency metrics for diagnostic experiment outputs.

Pure-Python post-processing over generated JSONL samples. This module does not
load model weights and is safe for local tests.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any

from research.validators.json_schema import JsonStatus, check_json
from research.validators.stack import check_brackets, repair_distance
from research.validators.tool_call import ToolCallStatus, check_tool_call


@dataclass(frozen=True)
class EvalRecord:
    """One per-sample structural evaluation row."""

    seed: int
    id: Any
    task: str
    valid: bool
    score: float
    repair_distance: int | None = None
    error_type: str | None = None
    error_position: int | None = None
    length: int | None = None
    max_depth: int | None = None
    max_pairing_distance: int | None = None
    vae_nll_per_token: float | None = None
    vae_ppl: float | None = None
    block_index: int | None = None
    on_boundary: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "id": self.id,
            "task": self.task,
            "valid": self.valid,
            "score": self.score,
            "repair_distance": self.repair_distance,
            "error_type": self.error_type,
            "error_position": self.error_position,
            "length": self.length,
            "max_depth": self.max_depth,
            "max_pairing_distance": self.max_pairing_distance,
            "vae_nll_per_token": self.vae_nll_per_token,
            "vae_ppl": self.vae_ppl,
            "block_index": self.block_index,
            "on_boundary": self.on_boundary,
        }


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generated_text(sample: dict[str, Any]) -> str:
    """Return the model output field used by Cola's inference scripts."""
    for key in ("generate", "generated", "output", "text"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
    return ""


def _dyck_candidate(input_record: dict[str, Any], sample: dict[str, Any]) -> str:
    meta = input_record.get("meta") or {}
    prefix = meta.get("prefix") or ""
    gen = generated_text(sample)
    if meta.get("mode") == "completion":
        return f"{prefix}{gen}"
    return gen


def _trace_locations(trace_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") == "mismatch" and "sample_index" in row:
            out[int(row["sample_index"])] = row
    return out


def evaluate_pair(
    input_record: dict[str, Any],
    sample: dict[str, Any],
    *,
    task: str,
    seed: int,
    trace_location: dict[str, Any] | None = None,
) -> EvalRecord:
    meta = input_record.get("meta") or {}
    record_id = input_record.get("id", sample.get("id"))

    if task == "dyck":
        candidate = _dyck_candidate(input_record, sample)
        result = check_brackets(candidate)
        dist = repair_distance(candidate)
        return EvalRecord(
            seed=seed,
            id=record_id,
            task=task,
            valid=result.valid,
            score=1.0 if result.valid else 0.0,
            repair_distance=dist,
            error_type=result.error_type.value if result.error_type else None,
            error_position=result.error_position,
            length=meta.get("length"),
            max_depth=meta.get("max_depth"),
            max_pairing_distance=meta.get("max_pairing_distance"),
            vae_nll_per_token=sample.get("vae_nll_per_token"),
            vae_ppl=sample.get("vae_ppl"),
            block_index=(trace_location or {}).get("block_index"),
            on_boundary=(trace_location or {}).get("on_boundary"),
        )

    if task == "structured":
        schema = meta.get("schema")
        result = check_json(generated_text(sample), schema)
        return EvalRecord(
            seed=seed,
            id=record_id,
            task=task,
            valid=result.status is JsonStatus.VALID,
            score=1.0 if result.status is JsonStatus.VALID else 0.0,
            error_type=result.status.value,
            error_position=result.error_position,
            vae_nll_per_token=sample.get("vae_nll_per_token"),
            vae_ppl=sample.get("vae_ppl"),
        )

    if task == "tool_call":
        tools = meta.get("tools") or {}
        result = check_tool_call(generated_text(sample), tools)
        return EvalRecord(
            seed=seed,
            id=record_id,
            task=task,
            valid=result.status is ToolCallStatus.VALID,
            score=1.0 if result.status is ToolCallStatus.VALID else 0.0,
            error_type=result.status.value,
            vae_nll_per_token=sample.get("vae_nll_per_token"),
            vae_ppl=sample.get("vae_ppl"),
        )

    raise ValueError(f"unsupported eval task: {task}")


def bootstrap_mean_ci(values: list[float], *, seed: int = 0, n_resamples: int = 2000) -> tuple[float, float]:
    """Return percentile 95% bootstrap CI for a mean."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_resamples):
        means.append(mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int(0.025 * (n_resamples - 1))]
    hi = means[int(0.975 * (n_resamples - 1))]
    return lo, hi


def summarize(rows: list[EvalRecord]) -> dict[str, Any]:
    scores = [r.score for r in rows]
    ci_lo, ci_hi = bootstrap_mean_ci(scores)
    distances = [r.repair_distance for r in rows if r.repair_distance is not None]
    nlls = [r.vae_nll_per_token for r in rows if r.vae_nll_per_token is not None]
    ppls = [r.vae_ppl for r in rows if r.vae_ppl is not None]
    boundary_rows = [r for r in rows if r.on_boundary is not None]
    return {
        "n": len(rows),
        "valid_rate": mean(scores) if scores else 0.0,
        "valid_rate_ci95_low": ci_lo,
        "valid_rate_ci95_high": ci_hi,
        "mean_repair_distance": mean(distances) if distances else None,
        "mean_vae_nll_per_token": mean(nlls) if nlls else None,
        "mean_vae_ppl": mean(ppls) if ppls else None,
        "boundary_error_rate": mean(1.0 if r.on_boundary else 0.0 for r in boundary_rows) if boundary_rows else None,
        "by_error_type": _count_by(rows, "error_type"),
        "by_seed": _group_mean(rows, "seed"),
        "by_length": _group_mean(rows, "length"),
        "by_depth": _group_mean(rows, "max_depth"),
        "by_pairing_distance": _group_mean(rows, "max_pairing_distance"),
    }


def _count_by(rows: list[EvalRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = getattr(row, field)
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _group_mean(rows: list[EvalRecord], field: str) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = getattr(row, field)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(row.score)
    return {k: {"n": len(v), "valid_rate": mean(v)} for k, v in sorted(buckets.items())}
