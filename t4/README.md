<!--
SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CoC 2.0 on T4 scenes with local models

These scripts run the CoC 2.0 streaming RSG workflow on scenes in the
[T4 dataset format](https://github.com/tier4/t4-devkit/tree/main/docs/schema). They use the sensor
workflow (front camera video, ego poses, ego meta-actions). The model runs on your own GPUs
behind vLLM's OpenAI-compatible server, instead of the OpenAI API. The labels come out per
interval, as in `docs/rsg.md`; there is no navigation or route input.

| File | What it does |
|---|---|
| `prepare_t4.py tile` | Lists every 20 s window of the scenes under one top-level directory of the T4 root |
| `prepare_t4.py prepare` | Cuts each window's front camera video, writes the ego poses, meta-actions, sensor manifests and the generation config |
| `check_prepared.py` | Runs the workflow's own input checks on a prepared batch, without model calls |
| `serve_vllm.sh` | Starts vLLM replicas of a model and waits until they are healthy (sourced) |
| `write_registry.py` | Writes the endpoint registry for those replicas |
| `label.sh` | Serves the model, runs the parallel generation, stops the servers and exports the labels |
| `export_batch.py` | Exports every clip of a publication in one process |
| `usage_summary.py` | Request counts, latency and token percentiles of a run |

## Requirements

- This repository installed with the streaming extras, plus `ffmpeg`, `opencv-python`,
  `pandas` and `pyarrow`.
- A Python environment with vLLM for the model you serve. Qwen3.8-Flash-Next needs the vLLM
  version its recipe names.
- T4 scenes with `annotation/` (`sample.json`, `sample_data.json`, `sensor.json`,
  `calibrated_sensor.json`, `ego_pose.json`) and the front camera as `data/<CHANNEL>.mp4`.
  A scene's id is its path under the T4 root with `/` replaced by `__`.
- Ego meta-actions for each scene, as `<scene id>.txt` in the text format
  `meta_action.post_processing` writes. For T4 scenes, the `feat/t4-support` branch of this
  repository generates them with `meta_action.t4_labeling`.

## Run

```bash
# 1. Windows: every 20 s of the scenes under <top> (a list you can split into batches)
python t4/prepare_t4.py tile --t4-root /path/to/t4_dataset --top <top> \
  --meta-actions /path/to/meta_action_txt --out batch/windows.json

# 2. Inputs: video cuts, poses, meta-actions, manifests and the generation config
python t4/prepare_t4.py prepare --t4-root /path/to/t4_dataset \
  --meta-actions /path/to/meta_action_txt --camera CAM_FRONT \
  --windows batch/windows.json --out batch/prepared \
  --registry batch/registry --ledger batch/ledger/ledger.jsonl --workers 8

# 3. Checks without model calls (the frame check needs the source videos)
python t4/check_prepared.py batch/prepared --t4-root /path/to/t4_dataset --frames 2

# 4. Label with local servers and export
PREPARED=batch/prepared OUT=batch/out MODEL=Qwen/Qwen3.8-27B t4/label.sh
PREPARED=batch/prepared OUT=batch/out MODEL=Qwen/Qwen3.8-Flash-Next-FP8 PRESET=flash-next \
  VPY=/path/to/vllm-env/bin/python t4/label.sh
```

- `--camera` takes one channel for every scene, or `TOP=CHANNEL` for the scenes under one
  top-level directory. Use the camera that faces forward on each vehicle.
- A window's clip id is its scene id plus `__f<start frame>`. `tile` adds one extra window at
  the scene end when at least 8 s are left after the last full window.
- `--exclude` on `tile` skips scenes from earlier window lists, or from a list of scene ids or
  paths (for example an evaluation split).
- The labels are in `OUT/export/annotation/<clip id>/`. `OUT/export/public/` holds the public
  export, which accepts only the recipe's tag vocabulary; local models often use other tags,
  so many clips fail there while their annotation export is fine.

## Changes to the workflow

Three small changes in `src/` make local models usable. Each is off unless its variable is
set, or only adds prompt text.

| Change | Why |
|---|---|
| `RSG_REQUEST_TIMEOUT_S` raises the fixed 300 s per-request timeout (`label.sh` sets 900) | A local server can take longer than 300 s for a long thinking answer |
| `RSG_UNPRICED_ERROR_USD` prices a failed request that has no usage (`label.sh` sets 0) | A 400 or a dropped connection otherwise leaves an unknown cost, and the admission gate stops every worker of the campaign |
| The action-phase prompt explains `before_id`/`after_id`, and the repair turn says how to fix a bracket error | Local models tend to read the bracket as the phase's start and end, and the bare validation error does not tell them how to fix it |

`prepare_t4.py` also raises `max_output_tokens` to 24576, because the recipe's 8192 includes the
thinking tokens of a thinking model.

## Notes

- `serve_vllm.sh` passes `--interleave-mm-strings`. Without it vLLM moves every image ahead of
  the text for string-format chat templates, and the model can no longer tell which time each
  frame shows.
- The budget fields stay because the workflow requires them; the registry's prices are nominal
  for a local endpoint. `label.sh` refreshes `accounting_verified_at` before each run.
- After a stop (for example a node failure), run `label.sh` again with the same `OUT` and
  `WORKERS`, or set `RECOVERY_SOURCE=<old OUT>/gen` with a new `OUT`.
