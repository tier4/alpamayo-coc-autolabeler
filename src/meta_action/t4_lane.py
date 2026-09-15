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

"""A lanelet2-backed stand-in for the trajdata vector map, for T4 scenes.

The lane meta-actions (``keep_lane``, the lane changes and shifts, and
``turn_left``/``turn_right``) are the one part of step 1 that needs a lane
graph. On T4 that graph is a lanelet2 map addressed by each scene's
``area_map_id``/``area_map_version_id``, not something carried in the scene.

The lane processor touches a very small part of the trajdata map API, so this
module reimplements exactly that surface on top of ``lanelet2``:

* ``vector_map.get_current_lane(xyzh, max_dist, max_heading_error)`` returning
  lanes ordered by distance, each with ``.id`` and ``.center.xy``;
* per-lane adjacency and successors, from which
  ``meta_action.utils.trajdata.lanegraph`` builds ``ego_lr``. That helper is
  reimplemented here rather than reused, because importing it pulls in trajdata.

Coordinates
-----------
T4 poses are in the map's MGRS frame, while ``lanelet2`` projects the OSM
lat/lon. Within one 100 km MGRS square the two differ by a constant, so the
first node's ``local_x``/``local_y`` tags give the offset outright: measured
across 400 nodes of one area map the residual spread is 0.0000 m.

``loadRobust`` is used rather than ``load``: Autoware maps reference regulatory
element ids that upstream lanelet2 rejects (thousands of such errors in that same map),
and refusing the file over them would cost the lane geometry and adjacency,
which parse fine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

#: Radius and heading tolerance the lane processor queries with, mirrored here so
#: the ego-relation pass matches it.
DEFAULT_MAX_DIST = 2.0
DEFAULT_MAX_HEADING_ERROR = np.pi / 3
#: Successor depth used when collecting the lanes ahead, as in the trajdata path.
NEXT_LANE_DEPTH = 5


class _Centerline:
    """The ``lane.center`` shape the lane processor reads, holding only ``.xy``."""

    __slots__ = ("xy",)

    def __init__(self, xy: np.ndarray) -> None:
        self.xy = xy


@dataclass(frozen=True)
class T4Lane:
    """One lanelet, exposing the attributes the lane processor uses."""

    id: str
    center: _Centerline


def read_local_offset(osm_path: Path) -> Tuple[float, float, float, float]:
    """Return ``(lat, lon, local_x, local_y)`` of the map's first tagged node.

    That node doubles as the projection origin and as the offset from projected
    metres back to the MGRS frame the T4 poses live in.

    Raises:
        ValueError: If no node carries both ``local_x`` and ``local_y``.
    """
    node_re = re.compile(r'<node id="[^"]+" lat="([^"]+)" lon="([^"]+)"')
    value_re = re.compile(r'v="([^"]+)"')
    lat = lon = local_x = local_y = None
    with open(osm_path) as handle:
        for line in handle:
            stripped = line.strip()
            match = node_re.match(stripped)
            if match is not None and lat is None:
                lat, lon = float(match.group(1)), float(match.group(2))
            elif lat is not None and 'k="local_x"' in stripped:
                local_x = float(value_re.search(stripped).group(1))
            elif lat is not None and 'k="local_y"' in stripped:
                local_y = float(value_re.search(stripped).group(1))
                break
    if lat is None or lon is None or local_x is None or local_y is None:
        raise ValueError(f"{osm_path}: no node carrying local_x/local_y tags")
    return lat, lon, local_x, local_y


class T4VectorMap:
    """Nearest-lane queries and lane adjacency over one lanelet2 map."""

    def __init__(self, osm_path: str | Path) -> None:
        """
        Args:
            osm_path: A ``lanelet2_map.osm`` file.
        """
        # Imported here rather than at module scope so that scenes without a map
        # -- and every lane-free run -- need neither lanelet2 nor scipy.
        import lanelet2  # noqa: PLC0415
        import lanelet2.routing  # noqa: PLC0415
        import lanelet2.traffic_rules  # noqa: PLC0415
        from lanelet2.io import Origin, loadRobust  # noqa: PLC0415
        from lanelet2.projection import UtmProjector  # noqa: PLC0415
        from scipy.spatial import cKDTree  # noqa: PLC0415

        osm_path = Path(osm_path)
        lat, lon, offset_x, offset_y = read_local_offset(osm_path)
        lanelet_map, errors = loadRobust(str(osm_path), UtmProjector(Origin(lat, lon)))
        if errors:
            logging.debug("%s: %d lanelet2 parse errors ignored", osm_path, len(errors))

        # Traffic rules only gate which neighbours count as lane-changeable; the
        # geometry and adjacency below do not depend on the locale.
        rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        graph = lanelet2.routing.RoutingGraph(lanelet_map, rules)

        centers: List[np.ndarray] = []
        headings: List[np.ndarray] = []
        lane_ids: List[np.ndarray] = []
        self._centers: Dict[str, np.ndarray] = {}
        self._left: Dict[str, Set[str]] = {}
        self._right: Dict[str, Set[str]] = {}
        self._next: Dict[str, Set[str]] = {}

        for lanelet in lanelet_map.laneletLayer:
            center = np.array(
                [[point.x + offset_x, point.y + offset_y] for point in lanelet.centerline]
            )
            if len(center) < 2:
                continue
            lane_id = str(lanelet.id)
            self._centers[lane_id] = center

            tangent = np.gradient(center, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            centers.append(center)
            headings.append(tangent)
            lane_ids.append(np.full(len(center), lane_id, dtype=object))

            # Both the routable neighbour and the merely adjacent one: trajdata's
            # adj_lanes_* mean "lane beside this one", which is what a lane change
            # is detected against, not whether the rules permit crossing.
            self._left[lane_id] = {
                str(neighbour.id)
                for neighbour in (graph.left(lanelet), graph.adjacentLeft(lanelet))
                if neighbour is not None
            }
            self._right[lane_id] = {
                str(neighbour.id)
                for neighbour in (graph.right(lanelet), graph.adjacentRight(lanelet))
                if neighbour is not None
            }
            self._next[lane_id] = {str(nxt.id) for nxt in graph.following(lanelet)}

        if not centers:
            raise ValueError(f"{osm_path}: no lanelet carries a usable centerline")

        self._points = np.vstack(centers)
        self._headings = np.vstack(headings)
        self._point_lane_ids = np.concatenate(lane_ids)
        self._tree = cKDTree(self._points)

    def get_current_lane(
        self,
        xyzh: Sequence[float],
        max_dist: float = DEFAULT_MAX_DIST,
        max_heading_error: float = DEFAULT_MAX_HEADING_ERROR,
    ) -> List[T4Lane]:
        """Lanes under a pose, nearest first.

        Args:
            xyzh: Pose as (x, y, z, heading) in the map frame.
            max_dist: Metres from a lane centerline to still count as on it.
            max_heading_error: Radians the pose may differ from the lane's
                direction, which is what keeps the opposing lane out.

        Returns:
            Matching lanes ordered by distance to their centerline.
        """
        position = np.asarray(xyzh, dtype=float)[:2]
        if not np.all(np.isfinite(position)):
            return []
        candidates = self._tree.query_ball_point(position, r=float(max_dist))
        if not candidates:
            return []

        heading = float(np.asarray(xyzh, dtype=float)[3])
        forward = np.array([np.cos(heading), np.sin(heading)])
        best: Dict[str, float] = {}
        for index in candidates:
            tangent = self._headings[index]
            error = abs(
                np.arctan2(float(np.cross(tangent, forward)), float(tangent @ forward))
            )
            if error > float(max_heading_error):
                continue
            lane_id = self._point_lane_ids[index]
            distance = float(np.linalg.norm(self._points[index] - position))
            if distance < best.get(lane_id, np.inf):
                best[lane_id] = distance

        return [
            T4Lane(id=lane_id, center=_Centerline(self._centers[lane_id]))
            for lane_id, _ in sorted(best.items(), key=lambda item: item[1])
        ]

    def _next_lanes(self, current: Set[str], depth: int = NEXT_LANE_DEPTH) -> Set[str]:
        """Lane ids reachable from current within depth successors."""
        reachable: Set[str] = set()
        frontier = set(current)
        for _ in range(depth):
            frontier = {
                successor
                for lane_id in frontier
                for successor in self._next.get(lane_id, ())
            } - reachable
            if not frontier:
                break
            reachable |= frontier
        return reachable

    def ego_lane_relations(self, world_xyzh: np.ndarray) -> List[Dict[str, Set[str]]]:
        """Per-timestep lane neighbourhood, matching the trajdata ego_lr.

        Mirrors meta_action.utils.trajdata.lanegraph.update_ego_lane_relation:
        the left/right sets cover the lanes beside the current lane *and* beside
        the lanes ahead of it, so a lane change is still recognised when the ego
        has already crossed into the neighbour.

        Args:
            world_xyzh: [T, 4] ego poses in the map frame.

        Returns:
            One dictionary per timestep with current_lanes, next_lanes,
            left_lanes and right_lanes.
        """
        relations: List[Dict[str, Set[str]]] = []
        for pose in world_xyzh:
            current = {lane.id for lane in self.get_current_lane(pose)}
            ahead = self._next_lanes(current)
            left: Set[str] = set()
            right: Set[str] = set()
            for lane_id in current | ahead:
                left |= self._left.get(lane_id, set())
                right |= self._right.get(lane_id, set())
            relations.append(
                {
                    "current_lanes": current,
                    "next_lanes": ahead,
                    "left_lanes": left,
                    "right_lanes": right,
                }
            )
        return relations


class T4SceneBatch:
    """The scene_batch attribute surface the lane processor reads."""

    def __init__(self, vector_map: T4VectorMap) -> None:
        self.vector_maps = [vector_map]


@lru_cache(maxsize=8)
def load_vector_map(osm_path: str) -> T4VectorMap:
    """Load and cache one map, since scenes in a run share a handful of areas."""
    return T4VectorMap(osm_path)


def resolve_map_path(scene_meta: Dict[str, Any], maps_root: str | Path) -> Optional[Path]:
    """Locate the lanelet2 map a T4 scene was recorded against.

    Args:
        scene_meta: Parsed derived/meta.json of the scene.
        maps_root: Directory holding <area_map_id>/<area_map_version_id>/.

    Returns:
        Path to the map, or None when the scene names no area map or the export
        does not carry it -- either way the scene simply gets no lane actions.
    """
    area_id = scene_meta.get("area_map_id")
    version_id = scene_meta.get("area_map_version_id")
    osm_name = scene_meta.get("map_osm_file") or "lanelet2_map.osm"
    if not area_id or not version_id:
        return None
    candidate = Path(maps_root) / str(area_id) / str(version_id) / osm_name
    return candidate if candidate.is_file() else None
