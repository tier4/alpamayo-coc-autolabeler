# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare streaming RSG inputs (the sensor workflow) from T4 dataset scenes.

This mirrors ``rsg_labeling.preparation.pai`` for the T4 dataset and makes no model calls.

* A scene is a directory under the T4 root that holds ``annotation/`` and ``data/``. Its id is
  its path relative to the root with ``/`` replaced by ``__``. A clip is a 20 s window of a
  scene (CoC 2.0 caps a clip at max_frames / 5 Hz = 25.6 s).
* Front camera: ``--camera CHANNEL``, or ``--camera TOP=CHANNEL`` for the scenes under the
  top-level directory ``TOP``. The window is cut from ``data/<CHANNEL>.mp4`` and re-encoded to
  H.264 without B-frames, and a sidecar gives each frame's capture time from
  ``sample_data.json``. Frame k of the video is the image ``compression_manifest.json`` lists at
  k when the scene has one, else the k-th image of that channel by file name.
* Ego poses: ``annotation/ego_pose.json`` (base_link in the map frame, quaternion w, x, y, z).
* Meta-actions: rule-based ego meta-actions, one ``<scene id>.txt`` per scene in
  ``--meta-actions``, in the text format ``meta_action.post_processing`` writes. Their ticks
  count the samples of ``sample.json`` sorted by time.
"""

import argparse
import glob
import hashlib
import json
import random
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rsg_labeling
import yaml
from coc_labeling.data_loader.meta_action_loader import (
    meta_action_lane,
    meta_action_lateral,
    meta_action_longitu,
    normalize_meta_action,
    parse_line,
)
from rsg_labeling.common.meta_actions import SCHEMA, coverage_diagnostics, validate_meta_actions
from rsg_labeling.infra.artifacts import write_json_atomic
from rsg_labeling.streaming_rsg.runtime.config import RunConfig
from rsg_labeling.tasks.streaming_rsg.generation.frame_sampling import validate_camera_brackets
from rsg_labeling.tasks.streaming_rsg.schema.contracts import ClipClock

RECIPES = Path(rsg_labeling.__file__).parent / "tasks/streaming_rsg/configs"
#: Samples in one clip (a 20 s window).
CLIP_FRAMES = 200
#: A window's keyframe sits 8 s after its start (2 s of history and 6 s of future fit around it).
LEAD_FRAMES = 80
VIDEO_HEIGHT = 620
#: Completion cap per request, thinking included. The recipe's 8192 is too small for a local
#: thinking model.
MAX_OUTPUT_TOKENS = 24576
#: Ego poses kept beyond the clip clock on each side (the workflow filters to the clock).
POSE_MARGIN_US = 1_000_000


def sha256(path: Path) -> str:
    """Hash a file without holding it in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def scene_id(relative: str) -> str:
    return "__".join(Path(relative).parts)


def camera_for(cameras: dict, scene: str) -> str:
    """The front camera channel of a scene id (``--camera`` values)."""
    return cameras.get(scene.split("__")[0], cameras[""])


def parse_cameras(values: list[str]) -> dict:
    cameras = {"": "CAM_FRONT"}
    for value in values:
        top, _, channel = value.rpartition("=")
        cameras[top] = channel
    return cameras


def excluded_scenes(paths: list[Path]) -> set[str]:
    """Scene ids from window lists (JSON) or plain lists of scene ids or relative paths."""
    out = set()
    for path in paths:
        if path.suffix == ".json":
            out |= {w["scene_clip_id"] for w in load(path)}
        else:
            out |= {scene_id(line.strip()) for line in path.read_text().splitlines() if line.strip()}
    return out


def tile_windows(root: Path, top: str, meta_actions: Path, limit: int | None, seed: int,
                 exclude: set[str]) -> list[dict]:
    """Every 20 s window of randomly drawn scenes under ``root/top``, for labelling whole scenes.

    Windows start every CLIP_FRAMES samples; a remainder of at least 8 s (2 s of history plus 6 s
    of future for one anchor) gets one more window ending at the scene end, overlapping the one
    before. ``keyframe`` is the window start plus LEAD_FRAMES.
    """
    # glob, unlike Path.rglob, follows symlinked scene directories.
    found = glob.glob(str(root / top / "**/annotation/sample.json"), recursive=True)
    scenes = sorted(scene_id(str(Path(p).parent.parent.relative_to(root))) for p in found)
    pool = [s for s in scenes if s not in exclude and (meta_actions / f"{s}.txt").is_file()]
    random.Random(seed).shuffle(pool)
    windows = []
    for sid in pool[:limit]:
        frames = len(load(root.joinpath(*sid.split("__")) / "annotation/sample.json"))
        starts = list(range(0, frames - CLIP_FRAMES + 1, CLIP_FRAMES))
        if starts and frames - (starts[-1] + CLIP_FRAMES) >= 80:
            starts.append(frames - CLIP_FRAMES)
        windows += [{"scene_clip_id": sid, "keyframe": s + LEAD_FRAMES, "meta_action": "tile"}
                    for s in starts]
    return windows


def camera_times(scene: Path, camera: str) -> np.ndarray:
    """Capture time (us) of each frame of ``data/<camera>.mp4``."""
    ann = scene / "annotation"
    channels = {s["token"]: s["channel"] for s in load(ann / "sensor.json")}
    calib = {c["token"]: channels[c["sensor_token"]] for c in load(ann / "calibrated_sensor.json")}
    times = {Path(r["filename"]).name: int(r["timestamp"]) for r in load(ann / "sample_data.json")
             if calib[r["calibrated_sensor_token"]] == camera}
    manifest = scene / "compression_manifest.json"
    cameras = load(manifest).get("cameras", {}) if manifest.is_file() else {}
    names = list(cameras.get(camera, {}).get("frames") or sorted(times))
    return np.array([times[name] for name in names], dtype=np.int64)


def lidar_times(scene: Path) -> np.ndarray:
    """Sample times; meta-action ticks index these."""
    return np.array(sorted(int(s["timestamp"]) for s in load(scene / "annotation/sample.json")),
                    dtype=np.int64)


def cut_video(source: Path, target: Path, start: int, count: int) -> None:
    """Re-encode frames [start, start + count) frame-accurately (no B-frames, GOP 10)."""
    select = f"select='between(n\\,{start}\\,{start + count - 1})',scale=-2:{VIDEO_HEIGHT}"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-threads", "4", "-i", str(source), "-vf", select,
         "-vsync", "0", "-frames:v", str(count), "-an", "-c:v", "libx264", "-preset", "fast",
         "-crf", "18", "-bf", "0", "-g", "10", "-pix_fmt", "yuv420p", str(target)],
        check=True)
    capture = cv2.VideoCapture(str(target))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if frames != count:
        raise ValueError(f"cut video has {frames} frames, expected {count}: {target}")


def ego_poses(scene: Path, lo: int, hi: int, target: Path) -> None:
    """Write the ego parquet the sensor manifest names (map frame, unit quaternions)."""
    poses = load(scene / "annotation/ego_pose.json")
    frame = pd.DataFrame({
        "t_us": np.array([p["timestamp"] for p in poses], dtype=np.int64),
        "x": [p["translation"][0] for p in poses],
        "y": [p["translation"][1] for p in poses],
        "z": [p["translation"][2] for p in poses],
        "qw": [p["rotation"][0] for p in poses],
        "qx": [p["rotation"][1] for p in poses],
        "qy": [p["rotation"][2] for p in poses],
        "qz": [p["rotation"][3] for p in poses],
    }).sort_values("t_us").drop_duplicates("t_us")
    frame = frame[(frame.t_us >= lo - POSE_MARGIN_US) & (frame.t_us <= hi + POSE_MARGIN_US)]
    quat = frame[["qw", "qx", "qy", "qz"]].to_numpy()
    frame[["qw", "qx", "qy", "qz"]] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    if len(frame) < 2 or frame.t_us.iloc[0] > lo or frame.t_us.iloc[-1] < hi - 200_000:
        raise ValueError(f"ego poses do not cover the clip clock: {scene}")
    frame.reset_index(drop=True).to_parquet(target, index=False)


GROUPS = {action: group for group, actions in (("longitudinal", meta_action_longitu),
                                               ("lateral", meta_action_lateral),
                                               ("lane", meta_action_lane))
          for action in actions}


def action_entry(path: Path, ticks_us: np.ndarray, clock: dict, source: dict) -> dict:
    """The production meta-action entry for one clip (half-open spans clipped to the clock)."""

    def at(tick: int) -> int:
        if tick < len(ticks_us):
            return int(ticks_us[tick])
        return int(ticks_us[-1]) + (tick - len(ticks_us) + 1) * 100_000

    actions = []
    for line in path.read_text().splitlines():
        if not line.strip() or "Agent:<ego>" not in line:
            continue
        parsed = parse_line(line)
        if parsed is None:
            raise ValueError(f"malformed ego action in {path}: {line!r}")
        name, start_tick, end_tick = parsed
        action = normalize_meta_action(name)
        if start_tick < 0 or start_tick >= end_tick or action not in GROUPS:
            raise ValueError(f"invalid ego action in {path}: {line!r}")
        start = max(clock["origin_us"], at(start_tick))
        end = min(clock["end_us"], at(end_tick))
        if start < end:
            actions.append({"action": action, "group": GROUPS[action], "start_us": start,
                            "end_us": end})
    actions.sort(key=lambda a: (a["group"], a["start_us"], a["end_us"], a["action"]))
    entry = {
        "available": bool(actions),
        "reason": "decoded_existing_t4_meta_actions" if actions else "no_decodable_ego_actions",
        "clock": clock,
        "actions": actions,
        "source": {**source, "kind": "t4_rule_generated_meta_actions", "sha256": sha256(path)},
    }
    validate_meta_actions(entry)
    entry.update(coverage_diagnostics(entry))
    return entry


def prepare_clip(window: dict, target: Path, root: Path, meta_actions: Path, cameras: dict) -> dict:
    """Write one clip's media, poses and manifest entries; return them for the manifests."""
    sid = window["scene_clip_id"]
    scene, camera = root.joinpath(*sid.split("__")), camera_for(cameras, sid)
    times = camera_times(scene, camera)
    if len(times) < CLIP_FRAMES:
        raise ValueError(f"{sid}: {len(times)} {camera} frames, need {CLIP_FRAMES}")
    start = min(max(window["keyframe"] - LEAD_FRAMES, 0), len(times) - CLIP_FRAMES)
    clip_id = f"{sid}__f{start:04d}"
    captures = times[start:start + CLIP_FRAMES]
    if np.any(np.diff(captures) <= 0):
        raise ValueError(f"{clip_id}: capture times are not increasing")
    validate_camera_brackets(captures)
    period = max(1, int(round(float(np.median(np.diff(captures))))))
    clock = {"origin_us": int(captures[0]), "end_us": int(captures[-1]) + period}

    clip_dir = target / "sensors" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    video = clip_dir / "camera.mp4"
    if not video.is_file():
        partial = clip_dir / "camera.partial.mp4"
        cut_video(scene / "data" / f"{camera}.mp4", partial, start, CLIP_FRAMES)
        partial.rename(video)
    stamps = clip_dir / "camera.timestamps"
    np.savetxt(stamps, np.column_stack((np.arange(CLIP_FRAMES), captures)), fmt="%d")
    poses = clip_dir / "ego.parquet"
    ego_poses(scene, clock["origin_us"], clock["end_us"], poses)

    source = {"dataset": "t4", "scene": str(scene.relative_to(root)), "camera": camera,
              "video_frames": [start, start + CLIP_FRAMES], "selected_keyframe": window["keyframe"],
              "selected_meta_action": window["meta_action"],
              "motion": "annotation/ego_pose.json (map frame)"}
    sensor = {
        "clock": clock,
        "ego": {"path": str(poses), "sha256": sha256(poses)},
        "camera": {"video": str(video), "video_sha256": sha256(video), "timestamps": str(stamps),
                   "timestamps_sha256": sha256(stamps)},
        "source": source,
    }
    actions = action_entry(meta_actions / f"{sid}.txt", lidar_times(scene), clock, source)
    return {"clip_id": clip_id, "sensor": sensor, "actions": actions, "clock": clock}


def _prepare(args: tuple) -> dict:
    window = args[0]
    try:
        return prepare_clip(*args)
    except Exception as error:  # recorded per clip, the batch continues
        return {"error": f"{type(error).__name__}: {error}", "window": window,
                "trace": traceback.format_exc(limit=3)}


def write_configs(target: Path, clips: list[dict], registry: Path, ledger: Path, owner: str,
                  campaign: str) -> None:
    """sensors.json, actions.json and sensor/{inputs,compiler,generation}.json."""
    write_json_atomic(target / "sensors.json", {"schema": "alparsg_sensor_inputs_v1",
                                                "clips": {c["clip_id"]: c["sensor"] for c in clips}})
    write_json_atomic(target / "actions.json", {"schema": SCHEMA,
                                                "clips": {c["clip_id"]: c["actions"] for c in clips}})
    generation = yaml.safe_load((RECIPES / "wl_readable_context.yaml").read_text())
    compiler = yaml.safe_load((RECIPES / "wl_readable_context.compiler.yaml").read_text())
    compiler["infra"].update(metadb=None, data_dir=None, endpoint_registry=str(registry))
    compiler["playground"]["env"].update(
        sensor_manifest=str(target / "sensors.json"), meta_action_manifest=str(target / "actions.json"),
        navigation_manifest=None, video_manifest=None, sources=["ego_pose", "production_meta_actions"])
    RunConfig.model_validate(compiler)
    stride = generation["policy"]["anchor_stride_us"]
    items = [{"clip_id": c["clip_id"], "clock": c["clock"], "clock_verified": True,
              "sample_anchors_us": ClipClock(**c["clock"]).sample_anchors(stride)} for c in clips]
    directory = target / "sensor"
    directory.mkdir(exist_ok=True)
    write_json_atomic(directory / "inputs.json", {"clips": items})
    write_json_atomic(directory / "compiler.json", compiler)
    generation.update(
        endpoint_registry=str(registry), input_manifest=str(directory / "inputs.json"),
        compiler_config=str(directory / "compiler.json"), max_output_tokens=MAX_OUTPUT_TOKENS)
    # A local endpoint still needs a budget; the registry prices are nominal. label.sh sets
    # allow_paid_dispatch and a fresh accounting_verified_at right before a run.
    generation["budget"].update(
        owner=owner, campaign=campaign, ledger=str(ledger), dispatch_envelope_usd=100.0,
        policy={"monthly_limit_usd": 1e6, "reserve_usd": 0, "campaign_limit_usd": 1e6,
                "daily_limit_usd": 1e6, "large_job_usd": 1e6})
    write_json_atomic(directory / "generation.json", generation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("tile", "every 20 s window of randomly drawn scenes into a JSON list"),
                       ("prepare", "write media, poses and manifests for a window list")):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--t4-root", type=Path, required=True, help="T4 dataset root")
        cmd.add_argument("--meta-actions", type=Path, required=True,
                         help="directory of <scene id>.txt ego meta-actions")
    tile, prep = sub.choices["tile"], sub.choices["prepare"]
    tile.add_argument("--top", required=True, help="top-level directory under the root to tile")
    tile.add_argument("--scenes", type=int, default=None, help="at most this many scenes")
    tile.add_argument("--seed", type=int, default=0)
    tile.add_argument("--exclude", type=Path, nargs="*", default=[],
                      help="window lists (.json) or scene lists (one id or path per line) to skip")
    tile.add_argument("--out", type=Path, required=True)
    prep.add_argument("--camera", action="append", default=[],
                      help="front camera channel, or TOP=CHANNEL per top-level directory "
                           "(default CAM_FRONT)")
    prep.add_argument("--windows", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--registry", type=Path, required=True, help="endpoint registry directory")
    prep.add_argument("--ledger", type=Path, required=True, help="budget ledger (jsonl)")
    prep.add_argument("--owner", default="local-vllm", help="budget owner")
    prep.add_argument("--campaign", default="t4", help="budget campaign")
    prep.add_argument("--limit", type=int, default=None)
    prep.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, meta_actions = args.t4_root.resolve(), args.meta_actions.resolve()

    if args.command == "tile":
        windows = tile_windows(root, args.top, meta_actions, args.scenes, args.seed,
                               excluded_scenes(args.exclude))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(windows, indent=1))
        print(json.dumps({"windows": len(windows), "scenes": len({w["scene_clip_id"] for w in windows}),
                          "out": str(args.out)}))
        return

    windows = load(args.windows)[: args.limit]
    target = args.out.resolve()
    target.mkdir(parents=True, exist_ok=True)
    cameras = parse_cameras(args.camera)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_prepare, [(w, target, root, meta_actions, cameras) for w in windows]))
    clips = [r for r in results if "clip_id" in r]
    failed = [r for r in results if "error" in r]
    write_json_atomic(target / "skipped.json", failed)
    if not clips:
        sys.exit(f"no clip prepared; see {target / 'skipped.json'}")
    write_configs(target, clips, args.registry.resolve(), args.ledger.resolve(), args.owner,
                  args.campaign)
    print(json.dumps({"prepared": len(clips), "skipped": len(failed), "out": str(target)}))


if __name__ == "__main__":
    main()
