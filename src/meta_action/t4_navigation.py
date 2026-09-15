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

"""Navigation instructions ("Turn left in 23m") for T4 scenes.

The route reading and the turn detection are `scripts/generate_navigation_labels.py`
on the `webdataset` branch, unchanged in substance: read the route polyline, ask a
lanelet2 map what `turn_direction` each point sits on, keep the runs that the
heading actually turns through, and name the distance to where such a run starts.

What changes is where the numbers come from. That script reads four zstd tensors
out of a WebDataset tar and picks the map from a hard-coded date table; the tars
are gone and the table covers ten dates. T4 carries the same quantities per scene:
`trajectory`, `turn` and `valid_mask` in `derived/scalars.npz` with identical
shapes, `route` in `derived/frames.pack` as `[segments, points, 33]` per frame
whose first two channels are the x and y the generator reads, and the map named
outright in `derived/meta.json` -- several maps across prd_jt rather than one.

It also only computes the frames it is asked for. The original derived a label for
every frame of a clip and then indexed the keyframes out of it; over a whole
subtree that is millions of frame reads to keep a small fraction of them, and `frames.pack` charges a
pread for each.

One label file covers one frame, and a frame carries a meta-action from each
independent taxonomy -- longitudinal, lateral and, with a map, lane. Some
frames hold two or three keyframe records. The original
wrote one file per record and so let them overwrite each other, losing records;
the navigation text was unaffected, being a property of the frame, but the
meta-action kept was whichever happened to be written last. Here every record for
a frame is merged into one file and `meta_actions` holds them all.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

#: Frames per second of the T4 sample grid; a keyframe index times this is its
#: timestamp in microseconds, which is how the label files are named.
US_PER_FRAME = 100_000

#: A run of turn-tagged route points shorter than this is noise, not a turn.
MIN_RUN_POINTS = 3
#: Heading change a run must actually turn through, in radians.
MIN_TURN_RAD = 0.30
#: Below this the turn is already happening, so the distance is not worth saying.
NEAR_TURN_M = 5.0
#: Route points needed before a frame is worth labelling at all.
MIN_ROUTE_POINTS = 10


class LaneletMapPolygonIndex:
    """Lanelet polygons tagged left/right/straight, indexed for point queries.

    Built by parsing the OSM directly rather than through lanelet2: the only
    thing needed here is which polygon a point falls in and what
    `turn_direction` that polygon carries, and the T4 maps use `local_x`/
    `local_y` tags that put the polygons in the same frame as the ego poses.
    """

    def __init__(self, osm_path: str | Path) -> None:
        from shapely.geometry import Polygon  # noqa: PLC0415
        from shapely.strtree import STRtree  # noqa: PLC0415

        logger.info("loading map %s", osm_path)
        root = ET.parse(str(osm_path)).getroot()

        nodes: Dict[str, Tuple[float, float]] = {}
        for node in root.findall("node"):
            x = y = None
            for tag in node.findall("tag"):
                if tag.get("k") == "local_x":
                    x = float(tag.get("v"))
                elif tag.get("k") == "local_y":
                    y = float(tag.get("v"))
            if x is None or y is None:
                x, y = float(node.get("lon")), float(node.get("lat"))
            nodes[node.get("id")] = (x, y)

        ways = {
            way.get("id"): [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]
            for way in root.findall("way")
        }

        self.polygons: List[Any] = []
        self.labels: List[str] = []
        for rel in root.findall("relation"):
            tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
            if tags.get("type") != "lanelet":
                continue
            turn = tags.get("turn_direction")
            if turn not in ("right", "left", "straight"):
                continue
            left: List[Tuple[float, float]] = []
            right: List[Tuple[float, float]] = []
            for member in rel.findall("member"):
                target = left if member.get("role") == "left" else (
                    right if member.get("role") == "right" else None
                )
                if target is not None:
                    target.extend(ways.get(member.get("ref"), []))
            if not left or not right:
                continue
            vertices = left + right[::-1]
            if len(vertices) < 3:
                continue
            polygon = Polygon(vertices)
            if polygon.is_valid:
                self.polygons.append(polygon)
                self.labels.append(turn)

        self.tree = STRtree(self.polygons) if self.polygons else None
        logger.info("  %d tagged lane polygons", len(self.polygons))

    def query_polygon(self, x: float, y: float) -> Optional[str]:
        """The turn direction of the lanelet containing `(x, y)`, if any."""
        if self.tree is None:
            return None
        from shapely.geometry import Point  # noqa: PLC0415

        point = Point(x, y)
        found = {self.labels[i] for i in self.tree.query(point) if self.polygons[i].contains(point)}
        for preferred in ("left", "right", "straight"):
            if preferred in found:
                return preferred
        return None


def nav_text_for_frame(
    route_xy: np.ndarray,
    ego_pose: np.ndarray,
    lanelet_map: LaneletMapPolygonIndex,
) -> str:
    """The instruction for one frame.

    :param route_xy: `[segments, points, 2]` route in the ego frame at this frame.
    :param ego_pose: `[4]` global `(x, y, cos, sin)` for the same frame.
    """
    ego_x, ego_y, ego_cos, ego_sin = (float(v) for v in ego_pose[:4])

    gx: List[float] = []
    gy: List[float] = []
    dist: List[float] = []
    tags: List[Optional[str]] = []
    for segment in route_xy:
        for rx, ry in segment:
            # An all-zero point is padding, not a point at the ego's own position.
            if rx == 0.0 and ry == 0.0:
                continue
            gx.append(ego_x + rx * ego_cos - ry * ego_sin)
            gy.append(ego_y + rx * ego_sin + ry * ego_cos)
            dist.append(math.hypot(float(rx), float(ry)))
            tags.append(lanelet_map.query_polygon(gx[-1], gy[-1]))

    if len(gx) < MIN_ROUTE_POINTS:
        return "Continue straight"

    # Contiguous runs of the same turn tag, straight and untagged points ending one.
    runs: List[Tuple[str, int, int]] = []
    current: Optional[str] = None
    start = 0
    for i, tag in enumerate(tags):
        if tag in ("left", "right"):
            if current != tag:
                if current is not None:
                    runs.append((current, start, i - 1))
                current, start = tag, i
        elif current is not None:
            runs.append((current, start, i - 1))
            current = None
    if current is not None:
        runs.append((current, start, len(tags) - 1))

    for tag, first, last in runs:
        if last - first < MIN_RUN_POINTS:
            continue
        # A lanelet tagged for a turn is not proof the route turns through it --
        # a route can clip the corner of one going straight on. Compare the
        # heading before the run with the heading after it.
        in_a = max(0, first - 10)
        in_b = first if in_a != first else min(len(gx) - 1, first + 5)
        out_a = last
        out_b = min(len(gx) - 1, last + 15)
        if out_a == out_b:
            out_a = max(0, last - 5)
        yaw_in = math.atan2(gy[in_b] - gy[in_a], gx[in_b] - gx[in_a])
        yaw_out = math.atan2(gy[out_b] - gy[out_a], gx[out_b] - gx[out_a])
        turned = math.atan2(math.sin(yaw_out - yaw_in), math.cos(yaw_out - yaw_in))
        if (tag == "left" and turned > MIN_TURN_RAD) or (tag == "right" and turned < -MIN_TURN_RAD):
            metres = dist[first]
            if first <= 2 or metres < NEAR_TURN_M:
                return f"Turn {tag}"
            return f"Turn {tag} in {int(round(metres))}m"

    return "Continue straight"


def scene_map_path(scene_dir: Path, maps_root: Path) -> Optional[Path]:
    """The lanelet2 map a scene was recorded against, or None when it names none.

    Some scenes carry no `area_map_id`. Those are labelled
    "Continue straight" rather than skipped, so a downstream index does not
    develop holes it has to special-case.
    """
    try:
        meta = json.loads((scene_dir / "derived" / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    map_id, version = meta.get("area_map_id"), meta.get("area_map_version_id")
    if not map_id or not version:
        return None
    path = maps_root / str(map_id) / str(version) / meta.get("map_osm_file", "lanelet2_map.osm")
    return path if path.is_file() else None


def label_scene(job: Tuple[str, str, str, List[dict], str]) -> Tuple[str, int, Optional[str]]:
    """Write one scene's navigation labels; returns `(clip_id, written, error)`."""
    clip_id, scene_dir, maps_root, keyframes, out_root = job
    try:
        from t4_e2e_devkit.dataset.scene import T4BundleReader  # noqa: PLC0415

        scene = Path(scene_dir)
        with np.load(scene / "derived" / "scalars.npz") as z:
            trajectory = z["trajectory"]
            valid_mask = z["valid_mask"] if "valid_mask" in z.files else None

        lanelet_map = _map_cache(scene_map_path(scene, Path(maps_root)))
        out_dir = Path(out_root) / clip_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # A frame appears once per taxonomy it was tagged under, and they all
        # describe the same instant, so they belong in one file rather than
        # three that overwrite each other.
        by_frame: Dict[int, List[str]] = defaultdict(list)
        for record in keyframes:
            action = record.get("meta_action", "")
            frame = int(record["event_start_frame"])
            if action and action not in by_frame[frame]:
                by_frame[frame].append(action)
            by_frame.setdefault(frame, [])

        bundle = None
        written = 0
        try:
            for frame in sorted(by_frame):
                if frame >= len(trajectory):
                    continue
                if lanelet_map is None or (valid_mask is not None and not bool(valid_mask[frame])):
                    text = "Continue straight"
                else:
                    if bundle is None:
                        bundle = T4BundleReader(scene / "derived" / "frames.pack")
                    # Channels 0 and 1 of the 33 are the x and y of the polyline.
                    route = np.asarray(bundle.frame(frame)["route"])[..., :2]
                    text = nav_text_for_frame(route, trajectory[frame], lanelet_map)

                actions = sorted(by_frame[frame])
                (out_dir / f"nav_{frame * US_PER_FRAME}.yaml").write_text(
                    yaml.dump(
                        {
                            "event_start_frame": frame,
                            "event_start_timestamp": frame * US_PER_FRAME,
                            # Singular kept for readers that expect the original
                            # field; plural is the complete answer.
                            "meta_action": actions[0] if actions else "",
                            "meta_actions": actions,
                            "navigation_text": text,
                        },
                        allow_unicode=True,
                        default_flow_style=False,
                    )
                )
                written += 1
        finally:
            if bundle is not None:
                bundle.close()
        return clip_id, written, None
    except Exception as error:  # noqa: BLE001 - one scene must not end the run
        return clip_id, 0, f"{type(error).__name__}: {error}"


#: Maps are shared by hundreds of scenes and cost seconds to parse, so each
#: worker keeps the ones it has already built.
_MAPS: Dict[str, LaneletMapPolygonIndex] = {}


def _map_cache(path: Optional[Path]) -> Optional[LaneletMapPolygonIndex]:
    if path is None:
        return None
    key = str(path)
    if key not in _MAPS:
        _MAPS[key] = LaneletMapPolygonIndex(path)
    return _MAPS[key]


def scene_dir_from_clip_id(clip_id: str) -> str:
    """Invert `meta_action.t4_labeling.clip_id_for`."""
    parts = clip_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"clip_id {clip_id!r} does not have four __-separated parts")
    return "/".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", required=True, help="alpamayo2.t4.scene_index JSON")
    parser.add_argument("--keyframes", required=True, help="segments_relative_timestamp_*.json")
    parser.add_argument("--save-dir", required=True, help="output root for nav_labels/<clip_id>/")
    parser.add_argument(
        "--maps-root", required=True, help="lanelet2 maps root"
    )
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = json.loads(Path(args.index).read_text())
    root = Path(index["root"])
    known = {entry["scene_dir"] for entry in index["scenes"]}

    keyframes = json.loads(Path(args.keyframes).read_text())
    by_clip: Dict[str, List[dict]] = defaultdict(list)
    unknown = 0
    for entries in keyframes.values():
        for record in entries:
            try:
                scene_dir = scene_dir_from_clip_id(record["clip_id"])
            except ValueError:
                unknown += 1
                continue
            if scene_dir not in known:
                unknown += 1
                continue
            by_clip[record["clip_id"]].append(record)

    jobs = [
        (clip_id, str(root / scene_dir_from_clip_id(clip_id)), args.maps_root, records,
         args.save_dir)
        for clip_id, records in sorted(by_clip.items())
    ]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    total_kf = sum(len(j[3]) for j in jobs)
    logger.info(
        "%d scenes, %d keyframes (%d keyframes had no scene in the index), %d workers",
        len(jobs), total_kf, unknown, args.workers,
    )

    written = failed = 0
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs) or 1))) as pool:
        for n, (clip_id, count, error) in enumerate(pool.map(label_scene, jobs), start=1):
            written += count
            if error:
                failed += 1
                logger.warning("  %s: %s", clip_id, error)
            if n % 200 == 0:
                logger.info("  %d/%d scenes, %d labels", n, len(jobs), written)

    logger.info("wrote %d navigation labels, %d scenes failed", written, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
