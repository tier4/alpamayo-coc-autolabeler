# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lanelet2 OSM adapter that provides a trajdata VectorMap-compatible interface.

Parses a lanelet2_map.osm file and builds duck-typed objects matching the
interfaces expected by meta_action.processor.lane and
meta_action.utils.trajdata.lanegraph:

  - Lanelet2VectorMap.get_current_lane(state, max_dist, max_heading_error)
  - Lanelet2VectorMap.elements[MapElementType.ROAD_LANE]  (dict of lanes)
  - Lane objects with .id, .center.xy, .next_lanes, .adj_lanes_left, .adj_lanes_right
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class _LaneCenterline:
    """Duck-typed centerline with .xy attribute matching trajdata's Polyline."""

    __slots__ = ("xy",)

    def __init__(self, xy: np.ndarray) -> None:
        self.xy = xy


class Lanelet2Lane:
    """Duck-typed lane object matching the interface expected by lane processors.

    Required attributes:
      .id            : str
      .center        : object with .xy -> np.ndarray (N,2)
      .next_lanes    : list[str]
      .adj_lanes_left : list[str]
      .adj_lanes_right: list[str]
    """

    __slots__ = ("id", "center", "next_lanes", "adj_lanes_left", "adj_lanes_right", "heading")

    def __init__(
        self,
        lane_id: str,
        center_xy: np.ndarray,
        heading: float,
    ) -> None:
        self.id = lane_id
        self.center = _LaneCenterline(center_xy)
        self.heading = heading
        self.next_lanes: List[str] = []
        self.adj_lanes_left: List[str] = []
        self.adj_lanes_right: List[str] = []


class Lanelet2VectorMap:
    """Duck-typed VectorMap built from a lanelet2 OSM file.

    Provides:
      .elements[MapElementType.ROAD_LANE] -> dict[str, Lanelet2Lane]
      .get_current_lane(state, max_dist, max_heading_error) -> list[Lanelet2Lane]
    """

    def __init__(self, osm_path: str) -> None:
        logger.info("Loading lanelet2 map: %s", osm_path)
        nodes, ways = _parse_nodes_and_ways(osm_path)
        lanes = _build_lanes(osm_path, nodes, ways)
        _build_connectivity(osm_path, lanes, nodes, ways)
        _build_adjacency(osm_path, lanes)

        self._lanes: Dict[str, Lanelet2Lane] = lanes

        # Build KDTree over ALL centerline points for accurate nearest-lane queries.
        all_points: List[np.ndarray] = []
        self._point_lane_ids: List[str] = []
        for lid, lane in lanes.items():
            for pt in lane.center.xy:
                all_points.append(pt)
                self._point_lane_ids.append(lid)
        self._all_points = np.array(all_points)
        self._kdtree = cKDTree(self._all_points)

        # Provide .elements dict-like access keyed by MapElementType.ROAD_LANE
        self.elements = _ElementsProxy(self._lanes)

        logger.info("Lanelet2 map loaded: %d road lanes, %d centerline points", len(self._lanes), len(all_points))

    def get_current_lane(
        self,
        state: np.ndarray,
        max_dist: float = 2.0,
        max_heading_error: float = 1.047,
    ) -> List[Lanelet2Lane]:
        """Find lanes near a query state [x, y, z, heading] or [x, y, heading]."""
        xy = state[:2]
        if len(state) >= 4:
            heading = float(state[3])
        else:
            heading = float(state[2])

        candidate_indices = self._kdtree.query_ball_point(xy, r=max_dist)
        if not candidate_indices:
            return []

        seen: Set[str] = set()
        results = []
        for idx in candidate_indices:
            lid = self._point_lane_ids[idx]
            if lid in seen:
                continue
            seen.add(lid)
            lane = self._lanes[lid]
            heading_diff = abs(_normalize_angle(heading - lane.heading))
            if heading_diff > max_heading_error:
                continue
            results.append(lane)

        return results


class _ElementsProxy:
    """Proxy to allow vector_map.elements[MapElementType.ROAD_LANE] access."""

    def __init__(self, lanes: Dict[str, Lanelet2Lane]) -> None:
        self._lanes = lanes

    def __getitem__(self, key):
        return self._lanes


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _parse_nodes_and_ways(
    osm_path: str,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]:
    """Parse nodes and ways from lanelet2 OSM."""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.findall("node"):
        nid = node.get("id")
        x, y = None, None
        for tag in node.findall("tag"):
            if tag.get("k") == "local_x":
                x = float(tag.get("v"))
            if tag.get("k") == "local_y":
                y = float(tag.get("v"))
        if x is not None and y is not None:
            nodes[nid] = (x, y)

    ways: Dict[str, List[Tuple[float, float]]] = {}
    for way in root.findall("way"):
        wid = way.get("id")
        pts = [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]
        if pts:
            ways[wid] = pts

    return nodes, ways


def _compute_centerline(
    left_pts: List[Tuple[float, float]], right_pts: List[Tuple[float, float]]
) -> np.ndarray:
    """Compute centerline by averaging left and right boundary, resampled to equal length."""
    left = np.array(left_pts)
    right = np.array(right_pts)

    n = max(len(left), len(right))
    if len(left) != n:
        left = _resample(left, n)
    if len(right) != n:
        right = _resample(right, n)

    return (left + right) / 2.0


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to n points via linear interpolation."""
    if len(pts) < 2 or n < 2:
        return pts
    dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    total = dists[-1]
    if total < 1e-9:
        return np.tile(pts[0], (n, 1))
    target_dists = np.linspace(0, total, n)
    resampled = np.zeros((n, 2))
    for dim in range(2):
        resampled[:, dim] = np.interp(target_dists, dists, pts[:, dim])
    return resampled


def _compute_heading(center_xy: np.ndarray) -> float:
    """Compute representative heading from centerline."""
    if len(center_xy) < 2:
        return 0.0
    dx = center_xy[-1, 0] - center_xy[0, 0]
    dy = center_xy[-1, 1] - center_xy[0, 1]
    return float(np.arctan2(dy, dx))


def _build_lanes(
    osm_path: str,
    nodes: Dict[str, Tuple[float, float]],
    ways: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, Lanelet2Lane]:
    """Build Lanelet2Lane objects from lanelet relations in the OSM."""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    lanes: Dict[str, Lanelet2Lane] = {}
    for rel in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "road":
            continue

        rid = rel.get("id")
        members = {}
        for m in rel.findall("member"):
            if m.get("type") == "way":
                role = m.get("role")
                if role in ("left", "right"):
                    members[role] = m.get("ref")

        left_wid = members.get("left")
        right_wid = members.get("right")
        if not left_wid or not right_wid:
            continue
        if left_wid not in ways or right_wid not in ways:
            continue

        center_xy = _compute_centerline(ways[left_wid], ways[right_wid])
        heading = _compute_heading(center_xy)
        lanes[rid] = Lanelet2Lane(lane_id=rid, center_xy=center_xy, heading=heading)

    return lanes


def _build_connectivity(
    osm_path: str,
    lanes: Dict[str, Lanelet2Lane],
    nodes: Dict[str, Tuple[float, float]],
    ways: Dict[str, List[Tuple[float, float]]],
) -> None:
    """Build next_lanes by matching endpoint proximity of left boundaries."""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    lane_left_way: Dict[str, str] = {}
    for rel in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "road":
            continue
        rid = rel.get("id")
        if rid not in lanes:
            continue
        for m in rel.findall("member"):
            if m.get("type") == "way" and m.get("role") == "left":
                lane_left_way[rid] = m.get("ref")
                break

    # Build endpoint -> lane mapping for fast lookup
    eps = 0.15  # tolerance in meters
    end_points: List[Tuple[str, np.ndarray]] = []
    start_points: List[Tuple[str, np.ndarray]] = []

    for lid, wid in lane_left_way.items():
        pts = ways.get(wid, [])
        if not pts:
            continue
        end_points.append((lid, np.array(pts[-1])))
        start_points.append((lid, np.array(pts[0])))

    if not end_points or not start_points:
        return

    start_coords = np.array([sp[1] for sp in start_points])
    start_tree = cKDTree(start_coords)

    for lid, end_pt in end_points:
        indices = start_tree.query_ball_point(end_pt, r=eps)
        for idx in indices:
            next_lid = start_points[idx][0]
            if next_lid != lid and next_lid in lanes:
                lanes[lid].next_lanes.append(next_lid)


def _build_adjacency(
    osm_path: str,
    lanes: Dict[str, Lanelet2Lane],
) -> None:
    """Build adj_lanes_left/right from shared boundary ways.

    If lanelet A's right boundary == lanelet B's left boundary,
    then B is to A's right and A is to B's left.
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    # Map way_id -> list of (lanelet_id, role)
    way_users: Dict[str, List[Tuple[str, str]]] = {}
    for rel in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in rel.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "road":
            continue
        rid = rel.get("id")
        if rid not in lanes:
            continue
        for m in rel.findall("member"):
            if m.get("type") == "way" and m.get("role") in ("left", "right"):
                wid = m.get("ref")
                way_users.setdefault(wid, []).append((rid, m.get("role")))

    for wid, users in way_users.items():
        if len(users) < 2:
            continue
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                lid_a, role_a = users[i]
                lid_b, role_b = users[j]
                if role_a == "right" and role_b == "left":
                    # B is to A's right, A is to B's left
                    if lid_b not in lanes[lid_a].adj_lanes_right:
                        lanes[lid_a].adj_lanes_right.append(lid_b)
                    if lid_a not in lanes[lid_b].adj_lanes_left:
                        lanes[lid_b].adj_lanes_left.append(lid_a)
                elif role_a == "left" and role_b == "right":
                    if lid_a not in lanes[lid_b].adj_lanes_right:
                        lanes[lid_b].adj_lanes_right.append(lid_a)
                    if lid_b not in lanes[lid_a].adj_lanes_left:
                        lanes[lid_a].adj_lanes_left.append(lid_b)


def load_lanelet2_vector_map(osm_path: str) -> Lanelet2VectorMap:
    """Load a lanelet2 OSM file and return a VectorMap-compatible adapter."""
    return Lanelet2VectorMap(osm_path)
