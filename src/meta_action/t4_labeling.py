# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step 1 of the pipeline, run over TIER IV T4 scenes instead of Physical-AI AV.

``meta_action.data_labeling`` reaches the ego trajectory through trajdata's
``UnifiedDataset`` -> ``SceneBatch`` -> ``TemporalScenario``. T4 is not a trajdata
dataset and adding it as one would mean carrying a pinned NVlabs fork that
declares ``protobuf==3.19.4`` for the sake of one array of poses.

It is not needed. With ``use_lane=False`` the classifiers read exactly three
arrays -- ``agent_xyh``, ``agent_speed``, ``agent_speed_along_heading`` -- off
``scenario.agent_trajdata``, plus ``all_agents_names`` and a ``segment_cache``
dict. Nothing else on the scenario is touched, and ``SceneBatch`` survives only
as a type annotation. So this module builds those three arrays from the T4 ego
poses and hands over a duck-typed scenario; the segmentation and classification
code runs unmodified, and trajdata is never imported.

What comes out is what Step 2 expects: one ``<clip_id>.json`` per scene holding
the flat list of ``"<Name>Temporal(ego at <start>-<end>)"`` strings that
``meta_action.post_processing`` turns into ``.txt``, which
``coc_labeling.keyframe_auto_select`` turns into keyframes.

Lane meta-actions need a lane graph, which for T4 lives in a lanelet2 map keyed
by ``area_map_id``/``area_map_version_id`` rather than in the scene. Pass
``--maps-root`` to enable them: ``meta_action.t4_lane`` then supplies the small
part of the trajdata map API the lane processor uses. Without it the run stays
lane-free, exactly as before.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from meta_action.data_labeling import get_all_ego_meta_action_names, save_result_clip
from meta_action.data_structures.ego_meta_action_wrapper import META_ACTION_MAPPING
from meta_action.utils.constant import DELTA_TIMESTAMP


class T4Scenario:
    """The subset of ``TemporalScenario`` the lane-free classifiers actually read.

    Deliberately not a subclass: ``TemporalScenario.__init__`` takes a
    ``SceneBatch`` and imports trajdata at module scope, which is the whole thing
    being avoided. The attribute surface below is the complete one, established
    by reading every ``scenario.`` access the classifiers make.

    ``ego_lr``, ``scene_batch`` and ``get_world_states`` appear only when a
    ``vector_map`` is given, because the lane code is guarded by
    ``hasattr(scenario, "ego_lr")``: no map means the lane paths stay switched
    off rather than failing.
    """

    def __init__(
        self,
        clip_id: str,
        xy: np.ndarray,
        heading: np.ndarray,
        dt: float,
        vector_map: Optional[Any] = None,
    ) -> None:
        """
        :param clip_id: scene identifier, used in the output filename.
        :param xy: ``[T, 2]`` ego positions in the map frame, metres.
        :param heading: ``[T]`` ego yaw in the map frame, radians.
        :param dt: sample interval in seconds.
        :param vector_map: optional ``T4VectorMap`` enabling lane meta-actions.
        """
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must be [T, 2], got {xy.shape}")
        if heading.shape != (xy.shape[0],):
            raise ValueError(f"heading must be [{xy.shape[0]}], got {heading.shape}")

        # Velocity by central difference of the pose track rather than from the
        # ego_pose twist field: prd_jt and x2_dev store twist as null, so the
        # only velocity that exists for every subtree is the one differentiated
        # from position.
        vx = np.gradient(xy[:, 0], dt)
        vy = np.gradient(xy[:, 1], dt)
        speed = np.hypot(vx, vy)
        # Signed, so reversing is distinguishable from driving forward -- which
        # is the whole basis of the Reverse* captions.
        speed_along_heading = vx * np.cos(heading) + vy * np.sin(heading)

        self.clip_id = clip_id
        self.cfg: Dict[str, Any] | None = None
        self.all_agents_names: List[str] = ["ego"]
        self.segment_cache: Dict[str, Dict[str, Any]] = {}
        self.agent_trajdata: Dict[str, Any] = {
            # A torch tensor, not an array: the lateral path calls
            # .detach().cpu().numpy() on this one without a hasattr guard.
            "agent_xyh": torch.from_numpy(
                np.stack([np.stack([xy[:, 0], xy[:, 1], heading], axis=-1)])
            ).float(),
            "agent_speed": torch.from_numpy(speed[None]).float(),
            "agent_speed_along_heading": torch.from_numpy(speed_along_heading[None]).float(),
            "agent_names": ["ego"],
        }

        if vector_map is not None:
            from meta_action.t4_lane import T4SceneBatch  # noqa: PLC0415

            # T4 poses are already in the map frame, so the trajdata transform
            # from agent-local to world coordinates is the identity here.
            self.scene_batch = T4SceneBatch(vector_map)
            self.get_world_states = lambda agent_xyzh, scene_batch: agent_xyzh
            world_xyzh = np.concatenate(
                [xy, np.zeros((len(xy), 1)), heading[:, None]], axis=1
            )
            self.ego_lr = vector_map.ego_lane_relations(world_xyzh)


def ego_track_arrays(scene_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one T4 scene's ego path as ``(xy [T, 2], heading [T])`` in the map frame.

    Imported lazily so this module stays importable -- and unit-testable against
    synthetic arrays -- without the alpamayo2 reader on the path.
    """
    from alpamayo2_super.t4.scene import LIDAR_CHANNEL, T4Scene  # noqa: PLC0415

    scene = T4Scene(Path(scene_dir))
    track = scene.ego_track
    xy = np.asarray(track.xyz[:, :2], dtype=np.float64)
    # Yaw about the map z axis. 'zyx' puts yaw first, and the T4 rig is close
    # enough to level that the other two angles do not leak into it.
    heading = track.rotations.as_euler("zyx")[:, 0].astype(np.float64)
    if len(xy) != len(scene.channel(LIDAR_CHANNEL).timestamps_us):
        raise ValueError(f"{scene_dir}: ego track and LiDAR frames disagree in length")
    return xy, heading


def lane_free_names(action_names: Sequence[str]) -> List[str]:
    """``action_names`` minus the lane-level ones, keeping the caller's order.

    The lane-free taxonomy is the source of truth rather than a literal list
    here, so adding a meta-action upstream cannot silently misclassify it.
    """
    lane_free = set(get_all_ego_meta_action_names(use_lane=False))
    return [name for name in action_names if name in lane_free]


def scene_vector_map(scene_dir: str, maps_root: str | None) -> Any:
    """The lanelet2 map this scene was recorded against, or None.

    The scene names its area map in ``derived/meta.json``; a scene whose map is
    absent from ``maps_root`` is labelled without lane actions rather than
    failing, since the rest of its meta-actions are unaffected.
    """
    if not maps_root:
        return None
    from meta_action.t4_lane import load_vector_map, resolve_map_path  # noqa: PLC0415

    meta_path = Path(scene_dir) / "derived" / "meta.json"
    if not meta_path.is_file():
        return None
    osm_path = resolve_map_path(json.loads(meta_path.read_text()), maps_root)
    return load_vector_map(str(osm_path)) if osm_path is not None else None


def label_scene(
    clip_id: str,
    scene_dir: str,
    save_root: str,
    action_names: Sequence[str],
    dt: float = DELTA_TIMESTAMP,
    maps_root: str | None = None,
) -> tuple[str, str | None]:
    """Label one scene and write ``<save_root>/<clip_id>.json``.

    Returns the failure as a value rather than raising: an unreadable scene is a
    fact about the export, and a fleet run must not stop on one.
    """
    try:
        xy, heading = ego_track_arrays(scene_dir)
        vector_map = scene_vector_map(scene_dir, maps_root)
        scenario = T4Scenario(clip_id, xy, heading, dt, vector_map)
        # Per scene, not per run: with --maps-root the caller asks for lane
        # actions, but a scene whose map is missing -- some T4 scenes name
        # no area map at all -- has no lane graph to classify against, and the
        # lane classifiers reach scenario.get_world_states before the
        # hasattr(scenario, "ego_lr") guard can turn them off.
        names = action_names if vector_map is not None else lane_free_names(action_names)
        results: List[Any] = []
        for name in names:
            results.extend(META_ACTION_MAPPING[name].get_motion_for_scenario("ego", scenario))
        # save_result_clip writes nothing for an empty result, which is how a
        # scene with no classifiable motion drops out of Step 2 by itself.
        save_result_clip(os.path.join(save_root, f"{clip_id}.json"), results)
        return clip_id, None
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return clip_id, f"{type(error).__name__}: {error}"


def clip_id_for(scene_dir: str) -> str:
    """A filename-safe clip id for a T4 scene path.

    ``post_processing`` derives the clip id with ``basename(f).split(".")[0]``,
    so the id must contain no dot; T4 scene names are ``HH_MM_SS-HH_MM_SS`` and
    are only unique within a vehicle-day, so the path is flattened rather than
    truncated to its last component.
    """
    return "__".join(Path(scene_dir).parts[-4:]).replace(".", "_")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", required=True, help="alpamayo2.t4.scene_index JSON")
    parser.add_argument("--save-dir", required=True, help="output dir for raw <clip_id>.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--maps-root",
        default=None,
        help=(
            "Directory of lanelet2 maps laid out as "
            "<area_map_id>/<area_map_version_id>/lanelet2_map.osm. Given, lane "
            "meta-actions are emitted as well; omitted, the run is lane-free."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    )
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument(
        "--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )
    args = parser.parse_args(argv)

    index = json.loads(Path(args.index).read_text())
    root = Path(index["root"])
    scenes = index["scenes"]
    if args.limit is not None:
        scenes = scenes[: args.limit]

    base, extra = divmod(len(scenes), args.n_tasks)
    start = args.task_id * base + min(args.task_id, extra)
    mine = scenes[start : start + base + (1 if args.task_id < extra else 0)]

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    action_names = get_all_ego_meta_action_names(use_lane=bool(args.maps_root))
    print(
        f"task {args.task_id}/{args.n_tasks}: {len(mine)} scenes, "
        f"{len(action_names)} ego meta-actions, {args.workers} workers",
        flush=True,
    )

    jobs = [
        (
            clip_id_for(e["scene_dir"]),
            str(root / e["scene_dir"]),
            str(save_root),
            action_names,
            DELTA_TIMESTAMP,
            args.maps_root,
        )
        for e in mine
    ]
    done = failed = 0
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs) or 1))) as pool:
        futures = [pool.submit(label_scene, *job) for job in jobs]
        for n, future in enumerate(as_completed(futures), start=1):
            clip_id, reason = future.result()
            if reason is None:
                done += 1
            else:
                failed += 1
                print(f"FAIL {clip_id}: {reason}", flush=True)
            if n % 200 == 0 or n == len(jobs):
                rate = (time.perf_counter() - t_start) / n
                print(f"  {n}/{len(jobs)} scenes, {rate:.2f} s/scene", flush=True)

    print(f"task {args.task_id}: labelled {done}, failed {failed}")
    return 1 if done == 0 and jobs else 0


if __name__ == "__main__":
    sys.exit(main())
