# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export every clip of a CoC 2.0 publication in one process.

The upstream CLI (`generation export-annotation` / `export-public`) loads the whole compact
publication for each clip, so exporting a 1,000-clip batch clip by clip takes hours. This loads it
once and calls the same writers. Existing export directories are kept (rerunning resumes).

Usage: export_batch.py PUBLICATION [PUBLICATION ...] --out OUT_DIR
  PUBLICATION: a `clips` directory (gen/clips, or gen/workers/*/run/clips of a stopped run)
  writes OUT_DIR/annotation/<clip>, OUT_DIR/public/<clip> and OUT_DIR/export_summary.json
"""

import argparse
import collections
import json
from pathlib import Path

from rsg_labeling.tasks.streaming_rsg.generation.export.publication import (
    write_annotation,
    write_public_store,
)
from rsg_labeling.tasks.streaming_rsg.generation.export.storage import load_clip_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("publications", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for publication in args.publications:
        if (publication / "manifest.json").exists():
            records.update(load_clip_records(publication))
    errors = {"annotation": collections.Counter(), "public": collections.Counter()}
    counts = collections.Counter()
    for cid, record in sorted(records.items()):
        counts[f"status:{record.get('status')}"] += 1
        if record.get("status") not in ("complete", "review_needed"):
            continue
        for kind, write in (
            ("annotation", lambda out: write_annotation(record, out)),
            ("public", lambda out: write_public_store(record["result"], out)),
        ):
            target = args.out / kind / cid
            if target.exists():
                counts[f"{kind}:kept"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                write(target)
                counts[f"{kind}:written"] += 1
            except Exception as error:  # one bad clip must not stop the batch
                errors[kind][f"{type(error).__name__}: {str(error)[:160]}"] += 1
                counts[f"{kind}:failed"] += 1
    summary = {
        "clips": len(records),
        "counts": dict(sorted(counts.items())),
        "errors": {k: v.most_common(10) for k, v in errors.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "export_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
