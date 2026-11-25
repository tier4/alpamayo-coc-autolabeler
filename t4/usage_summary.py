# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a CoC 2.0 run's transport receipts (every ``usage.jsonl`` under the run).

Prints request counts by status, latency and token percentiles, how many completions hit
their token cap, and the clip outcomes recorded in ``clips.jsonl``.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def pct(values: list[float]) -> str:
    if not values:
        return "-"
    a = np.asarray(values, dtype=float)
    return (f"mean {a.mean():.0f}  p50 {np.percentile(a, 50):.0f}  p90 {np.percentile(a, 90):.0f}  "
            f"max {a.max():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="the --out of the generation run")
    args = parser.parse_args()
    events = [json.loads(line) for path in sorted(args.run.rglob("usage.jsonl"))
              for line in path.read_text().splitlines() if line.strip()]
    print(f"requests: {len(events)}  status: {dict(Counter(e.get('status_code') for e in events))}  "
          f"errors: {dict(Counter(e.get('error_type') for e in events if e.get('error_type')))}")
    ok = [e for e in events if e.get("status_code") == 200 and e.get("usage")]
    print("elapsed s:        ", pct([e["elapsed_s"] for e in events if e.get("elapsed_s") is not None]))
    print("prompt tokens:    ", pct([e["usage"].get("prompt_tokens", 0) for e in ok]))
    print("completion tokens:", pct([e["usage"].get("completion_tokens", 0) for e in ok]))
    reasoning = [(e["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") for e in ok]
    reasoning = [r for r in reasoning if r is not None]
    if reasoning:
        print("reasoning tokens: ", pct(reasoning))
    capped = sum(1 for e in ok if e.get("max_tokens_requested")
                 and e["usage"].get("completion_tokens", 0) >= e["max_tokens_requested"])
    print(f"completions at their token cap: {capped} / {len(ok)}")
    clips = [json.loads(line) for path in sorted(args.run.rglob("clips.jsonl"))
             for line in path.read_text().splitlines() if line.strip()]
    if clips:
        print(f"clips: {len(clips)}  status: {dict(Counter(c.get('status') for c in clips))}")
        for clip in clips:
            if clip.get("status") != "success":
                print("  ", clip.get("clip_id"), clip.get("status"), str(clip.get("error", ""))[:200])


if __name__ == "__main__":
    main()
