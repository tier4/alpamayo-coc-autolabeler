#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Label one prepared batch of T4 clips with local vLLM servers, then export the labels.
#
#   PREPARED=<prepare_t4.py prepare --out> OUT=<new dir> MODEL=<model> [PRESET=flash-next] \
#     t4/label.sh
#
# Steps: start the servers (t4/serve_vllm.sh), write the endpoint registry, refresh the budget
# fields the workflow checks at dispatch, run the parallel streaming RSG generation, stop the
# servers, and export every labelled clip with t4/export_batch.py.
#
# Environment (besides serve_vllm.sh's): PY (python with this repository installed,
# default python3), WORKERS=32 (clips in flight), RSG_REQUEST_TIMEOUT_S=900,
# RECOVERY_SOURCE=<earlier OUT>/gen to continue a stopped or cancelled run into a new OUT
# (the workflow's --recovery-source; finished clips are reused). Run it again with the same OUT
# and WORKERS to resume a run that was interrupted.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREPARED=${PREPARED:?set PREPARED}
OUT=${OUT:?set OUT}
PY=${PY:-python3}
WORKERS=${WORKERS:-32}
RECOVERY_SOURCE=${RECOVERY_SOURCE:-}
GEN=$PREPARED/sensor/generation.json
LOGDIR=$OUT/vllm
source "$HERE/serve_vllm.sh"

export VLLM_API_KEY=${VLLM_API_KEY:-local}
export RSG_REQUEST_TIMEOUT_S=${RSG_REQUEST_TIMEOUT_S:-900}
# A failed request to a local server costs nothing; without this one 400 stops the campaign.
export RSG_UNPRICED_ERROR_USD=0
export ALPAMAYO_INFERENCE_STATE_DIR=$OUT/inference-state
umask 077
mkdir -p "$ALPAMAYO_INFERENCE_STATE_DIR"
echo "=== $(date -Is) prepared=$PREPARED out=$OUT recovery=${RECOVERY_SOURCE:-none} workers=$WORKERS"
trap 'stop_vllm' EXIT
start_vllm || exit 1

REG=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["endpoint_registry"])' "$GEN")
$PY "$HERE/write_registry.py" --out "$REG" --host 127.0.0.1 --ports "${ports[@]}" --model "$SERVED"
$PY - "$GEN" <<'EOF'
import datetime, json, os, sys
path = sys.argv[1]
config = json.load(open(path))
config["allow_paid_dispatch"] = True
config["budget"]["accounting_verified_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
os.makedirs(os.path.dirname(config["budget"]["ledger"]), exist_ok=True)
tmp = path + ".tmp"
json.dump(config, open(tmp, "w"), indent=2)
os.replace(tmp, path)
EOF

recovery=()
[ -n "$RECOVERY_SOURCE" ] && recovery=(--recovery-source "$RECOVERY_SOURCE")
start=$SECONDS
$PY -m rsg_labeling.tasks.streaming_rsg.generation.parallel --config "$GEN" --out "$OUT/gen" \
  --workers "$WORKERS" "${recovery[@]}"
rc=$?
echo "=== $(date -Is) generation exit $rc after $((SECONDS - start)) s"
stop_vllm
trap - EXIT

pubs=("$OUT/gen/clips")
[ -e "$OUT/gen/clips/manifest.json" ] || pubs=("$OUT"/gen/workers/*/run/clips)
$PY "$HERE/export_batch.py" "${pubs[@]}" --out "$OUT/export" | head -c 2000
echo
echo "=== $(date -Is) done: $(ls "$OUT/export/annotation" 2>/dev/null | wc -l) annotations"
exit $rc
