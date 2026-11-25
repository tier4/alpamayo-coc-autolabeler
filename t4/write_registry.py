# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a CoC 2.0 endpoint registry for local vLLM replicas.

One ``r<i>.json`` per replica. The ``openai_reasoning`` profile is used because a record
without a profile gets no cost estimate, and the workflow stops after a request of unknown
cost. Prices are nominal (the profile requires positive rates); ``api_key_env`` must name a
set variable, which vLLM ignores without ``--api-key``.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--reasoning-effort", default="medium", choices=("low", "medium", "high"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("r*.json"):
        old.unlink()
    for i, port in enumerate(args.ports):
        record = {
            "status": "ready",
            "url": f"http://{args.host}:{port}/v1",
            "model": args.model,
            "api_key_env": "VLLM_API_KEY",
            "request_profile": "openai_reasoning",
            "reasoning_effort": args.reasoning_effort,
            "pricing": {"input_usd_per_million_tokens": 1e-6, "output_usd_per_million_tokens": 1e-6},
        }
        (args.out / f"r{i}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"registry": str(args.out), "replicas": len(args.ports)}))


if __name__ == "__main__":
    main()
