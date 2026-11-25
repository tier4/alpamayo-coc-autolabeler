# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check prepared T4 inputs without any model call.

1. The workflow's own validation: compiler config, sensor manifest, native env, anchors and
   the generation config (``interface._validate_generation_inputs``).
2. The pose quaternion is (w, x, y, z): its yaw follows the direction of travel.
3. The cut video shows the source frames it claims (first and last frame against the
   source video read sequentially), for ``--frames N`` clips.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd



def workflow_checks(prepared: Path, clip_ids: list[str]) -> None:
    from rsg_labeling.streaming_rsg.env.env import build_env_native
    from rsg_labeling.streaming_rsg.env.sensor import load_sensor_data
    from rsg_labeling.streaming_rsg.runtime.config import RunConfig
    from rsg_labeling.tasks.streaming_rsg.generation import interface, runtime

    generation = json.loads((prepared / "sensor/generation.json").read_text())
    compiler = json.loads((prepared / "sensor/compiler.json").read_text())
    RunConfig.model_validate(compiler)
    env_cfg = compiler["playground"]["env"]
    items = {i["clip_id"]: i for i in json.loads((prepared / "sensor/inputs.json").read_text())["clips"]}
    for cid in clip_ids:
        load_sensor_data(env_cfg["sensor_manifest"], cid)
        env = build_env_native(
            None, None, cid, sensor_manifest=env_cfg["sensor_manifest"],
            meta_action_manifest=env_cfg["meta_action_manifest"], anchor_mode="discovery",
            include_external_labels=False, max_fps=env_cfg["max_fps"],
            max_frames=env_cfg["max_frames"], image_max_height_px=env_cfg["image_max_height_px"])
        runtime.validate_native_inputs(env, env.anchors[0], items[cid], generation["policy"])
        print(f"workflow ok: {cid} anchors={len(env.anchors)}")
    os.environ.setdefault("VLLM_API_KEY", "local")
    probe = dict(generation, allow_paid_dispatch=True,
                 budget=dict(generation["budget"], accounting_verified_at="2026-01-01T00:00:00+00:00"))
    interface._validate_generation_inputs(probe)
    print("generation config ok")


def yaw_check(sensor: dict) -> float:
    """Median |yaw(quaternion) - heading of travel| (deg) over moving poses."""
    poses = pd.read_parquet(sensor["ego"]["path"])
    w, x, y, z = (poses[c].to_numpy() for c in ("qw", "qx", "qy", "qz"))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dx, dy = np.diff(poses.x.to_numpy()), np.diff(poses.y.to_numpy())
    dt = np.diff(poses.t_us.to_numpy()) / 1e6
    moving = np.hypot(dx, dy) / np.maximum(dt, 1e-3) > 2.0
    if not moving.any():
        return float("nan")
    err = np.angle(np.exp(1j * (yaw[1:] - np.arctan2(dy, dx))))[moving]
    return float(np.degrees(np.median(np.abs(err))))


def frame_check(sensor: dict, root: Path) -> tuple[float, float]:
    """Mean |diff| of the cut's first/last frame against the source frames (0-255 scale)."""
    src = root / sensor["source"]["scene"] / "data" / f"{sensor['source']['camera']}.mp4"
    start, stop = sensor["source"]["video_frames"]
    cut = cv2.VideoCapture(sensor["camera"]["video"])
    ok, first_cut = cut.read()
    cut.set(cv2.CAP_PROP_POS_FRAMES, stop - start - 1)
    ok2, last_cut = cut.read()
    source = cv2.VideoCapture(str(src))
    wanted = {start: None, stop - 1: None}
    for index in range(stop):
        ok3, frame = source.read()
        if not ok3:
            raise ValueError(f"source ended at {index}")
        if index in wanted:
            wanted[index] = cv2.resize(frame, (first_cut.shape[1], first_cut.shape[0]),
                                       interpolation=cv2.INTER_AREA)
    diff = lambda a, b: float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    return diff(first_cut, wanted[start]), diff(last_cut, wanted[stop - 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--t4-root", type=Path, help="T4 dataset root, needed for --frames")
    parser.add_argument("--frames", type=int, default=2, help="clips to check frame by frame")
    args = parser.parse_args()
    sensors = json.loads((args.prepared / "sensors.json").read_text())["clips"]
    workflow_checks(args.prepared, list(sensors))
    for i, (cid, sensor) in enumerate(sensors.items()):
        line = f"{cid}: yaw err {yaw_check(sensor):.2f} deg"
        if i < args.frames and args.t4_root:
            first, last = frame_check(sensor, args.t4_root)
            line += f", frame diff first {first:.2f} last {last:.2f}"
        print(line)


if __name__ == "__main__":
    main()
