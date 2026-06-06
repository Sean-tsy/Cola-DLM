#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Pure-Python fake job (NO torch / NO model) used to verify the remote-control
# four-piece chain (sync/submit/status/fetch) end-to-end before wiring a real
# GPU job. Writes a minimal results/<run_id>/manifest.json and exits 0.
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESULTS_ROOT = "research/results"


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "fake_run"
    out = Path(RESULTS_ROOT) / run_id
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    print(f"[fake_job] start run_id={run_id}", flush=True)
    time.sleep(0.5)
    (out / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "fake": True}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "metrics" / "summary.json").write_text(
        json.dumps({"valid": 1, "total": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[fake_job] done run_id={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
