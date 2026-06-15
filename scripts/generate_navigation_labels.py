import io
import json
import logging
import math
import os
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List

import numpy as np
import yaml
import zstandard as zstd
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


class LaneletMapPolygonIndex:
    def __init__(self, osm_path: str):
        logging.info("Loading map data: %s", osm_path)
        tree = ET.parse(osm_path)
        root = tree.getroot()

        nodes = {}
        for node in root.findall('node'):
            nid = node.get('id')
            x, y = None, None
            for tag in node.findall('tag'):
                if tag.get('k') == 'local_x': x = float(tag.get('v'))
                if tag.get('k') == 'local_y': y = float(tag.get('v'))
            if x is None or y is None:
                x, y = float(node.get('lon')), float(node.get('lat'))
            nodes[nid] = (x, y)

        ways = {}
        for way in root.findall('way'):
            wid = way.get('id')
            ways[wid] = [nodes[nd.get('ref')] for nd in way.findall('nd') if nd.get('ref') in nodes]

        self.polygons = []
        self.labels = []
        self.raw_coords = []

        for rel in root.findall('relation'):
            tags = {tag.get('k'): tag.get('v') for tag in rel.findall('tag')}
            if tags.get('type') != 'lanelet':
                continue
            turn_dir = tags.get('turn_direction')
            if turn_dir not in ['right', 'left', 'straight']:
                continue
            left_ways = [m.get('ref') for m in rel.findall('member') if m.get('role') == 'left']
            right_ways = [m.get('ref') for m in rel.findall('member') if m.get('role') == 'right']
            left_path, right_path = [], []
            for wid in left_ways:
                if wid in ways: left_path.extend(ways[wid])
            for wid in right_ways:
                if wid in ways: right_path.extend(ways[wid])
            if not left_path or not right_path:
                continue
            poly_vertices = left_path + right_path[::-1]
            if len(poly_vertices) >= 3:
                shapely_poly = Polygon(poly_vertices)
                if shapely_poly.is_valid:
                    self.polygons.append(shapely_poly)
                    self.labels.append(turn_dir)
                    self.raw_coords.append(np.array(poly_vertices))

        self.tree = STRtree(self.polygons)
        logging.info(
            "Map loaded: built spatial index for %d lane polygons.", len(self.polygons)
        )

    def query_polygon(self, x: float, y: float) -> str:
        pt = Point(x, y)
        found_labels = []
        for idx in self.tree.query(pt):
            if self.polygons[idx].contains(pt):
                found_labels.append(self.labels[idx])
        if 'left' in found_labels: return 'left'
        if 'right' in found_labels: return 'right'
        if 'straight' in found_labels: return 'straight'
        return None


def generate_nav_text_from_map(
    route_tensor: np.ndarray,
    trajectory: np.ndarray,
    turn_tensor: np.ndarray,
    lanelet_map: LaneletMapPolygonIndex,
    valid_mask=None,
) -> List[str]:
    T, num_segments, num_points, _ = route_tensor.shape
    nav_texts = []
    for t in range(T):
        if valid_mask is not None and not valid_mask[t]:
            nav_texts.append("Continue straight")
            continue
        ego_global_x, ego_global_y = trajectory[t, 0], trajectory[t, 1]
        ego_cos, ego_sin = trajectory[t, 2], trajectory[t, 3]
        gx_list, gy_list, tags, dist_list = [], [], [], []
        for s in range(num_segments):
            for p in range(num_points):
                rx, ry = route_tensor[t, s, p, 0], route_tensor[t, s, p, 1]
                if rx == 0.0 and ry == 0.0:
                    continue
                gx = ego_global_x + rx * ego_cos - ry * ego_sin
                gy = ego_global_y + rx * ego_sin + ry * ego_cos
                gx_list.append(gx)
                gy_list.append(gy)
                dist_list.append(math.hypot(rx, ry))
                tags.append(lanelet_map.query_polygon(gx, gy))
        if len(gx_list) < 10:
            nav_texts.append("Continue straight")
            continue
        chunks = []
        current_tag = None
        start_idx = 0
        for i, tag in enumerate(tags):
            if tag in ['left', 'right']:
                if current_tag != tag:
                    if current_tag is not None:
                        chunks.append((current_tag, start_idx, i - 1))
                    current_tag = tag
                    start_idx = i
            else:
                if current_tag is not None:
                    chunks.append((current_tag, start_idx, i - 1))
                    current_tag = None
        if current_tag is not None:
            chunks.append((current_tag, start_idx, len(tags) - 1))
        final_text = "Continue straight"
        for tag, s_idx, e_idx in chunks:
            if e_idx - s_idx < 3:
                continue
            p_in_s = max(0, s_idx - 10)
            p_in_e = s_idx
            if p_in_s == p_in_e:
                p_in_e = min(len(gx_list) - 1, s_idx + 5)
            yaw_in = math.atan2(gy_list[p_in_e] - gy_list[p_in_s], gx_list[p_in_e] - gx_list[p_in_s])
            p_out_s = e_idx
            p_out_e = min(len(gx_list) - 1, e_idx + 15)
            if p_out_s == p_out_e:
                p_out_s = max(0, e_idx - 5)
            yaw_out = math.atan2(gy_list[p_out_e] - gy_list[p_out_s], gx_list[p_out_e] - gx_list[p_out_s])
            yaw_diff = math.atan2(math.sin(yaw_out - yaw_in), math.cos(yaw_out - yaw_in))
            is_valid_turn = False
            if tag == 'left' and yaw_diff > 0.30:
                is_valid_turn = True
            elif tag == 'right' and yaw_diff < -0.30:
                is_valid_turn = True
            if is_valid_turn:
                dist_m = dist_list[s_idx]
                if s_idx <= 2 or dist_m < 5:
                    final_text = f"Turn {tag}"
                else:
                    final_text = f"Turn {tag} in {int(round(dist_m))}m"
                break
        nav_texts.append(final_text)
    return nav_texts


logger = logging.getLogger(__name__)

MAP_BASE = "/mnt/storage_rdma/datasets/tier4/maps/1423"
DATE_TO_MAP_VERSION = {
    "2025-11-19": "1423-20250905061011941236",
    "2025-12-09": "1423-20251209045313582007",
    "2025-12-10": "1423-20250905061011941236",
    "2025-12-17": "1423-20251210022250515914",
    "2025-12-18": "1423-20251210022250515914",
    "2025-12-23": "1423-20251210022250515914",
    "2025-12-24": "1423-20251210022250515914",
    "2025-12-25": "1423-20251210022250515914",
    "2026-01-06": "1423-20251210022250515914",
    "2026-01-07": "1423-20251210022250515914",
}
TAR_BASE = "/mnt/share_drive/workspace-at/data/e2e-scene-shards-jpntaxi"
BATCH_OUTPUTS = os.path.join(os.path.dirname(__file__), "batch_outputs")

TENSOR_SUFFIXES = {
    ".route.npy.zst": "route",
    ".trajectory.npy.zst": "trajectory",
    ".turn.npy.zst": "turn",
    ".valid_mask.npy.zst": "valid_mask",
}


def decode_npy_zst(data: bytes) -> np.ndarray:
    dctx = zstd.ZstdDecompressor()
    return np.load(io.BytesIO(dctx.decompress(data)))


def load_tensors_from_tar(tar_path: str) -> dict[str, np.ndarray]:
    """Stream-read tar file and extract only the 4 needed tensors, with early exit."""
    found: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            for suffix, key in TENSOR_SUFFIXES.items():
                if member.name.endswith(suffix) and key not in found:
                    f = tar.extractfile(member)
                    if f is not None:
                        found[key] = decode_npy_zst(f.read())
                    break
            if len(found) == len(TENSOR_SUFFIXES):
                break
    return found


def load_keyframes(json_path: str) -> list[dict]:
    """Flatten all meta_action lists from the keyframe JSON into a single list."""
    with open(json_path) as f:
        data = json.load(f)
    keyframes = []
    for entries in data.values():
        keyframes.extend(entries)
    return keyframes


def process_date(date: str, lanelet_map: LaneletMapPolygonIndex) -> None:
    date_dir = os.path.join(BATCH_OUTPUTS, date)
    json_path = os.path.join(date_dir, "keyframes", "segments_relative_timestamp_sampled.json")
    if not os.path.exists(json_path):
        logger.warning("Keyframe JSON not found: %s", json_path)
        return

    keyframes = load_keyframes(json_path)
    if not keyframes:
        logger.info("%s: no keyframes, skipping", date)
        return

    # Group by clip_id so each tar is opened only once
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for kf in keyframes:
        by_clip[kf["clip_id"]].append(kf)

    tar_dir = os.path.join(TAR_BASE, date)
    nav_labels_dir = os.path.join(date_dir, "nav_labels")

    logger.info("%s: processing %d scenes", date, len(by_clip))

    for clip_id, clip_keyframes in sorted(by_clip.items()):
        tar_path = os.path.join(tar_dir, f"{clip_id}.tar")
        if not os.path.exists(tar_path):
            logger.warning("  tar not found: %s", tar_path)
            continue

        tensors = load_tensors_from_tar(tar_path)
        if "route" not in tensors or "trajectory" not in tensors or "turn" not in tensors:
            logger.warning("  required tensors missing for: %s", clip_id)
            continue

        nav_texts = generate_nav_text_from_map(
            tensors["route"],
            tensors["trajectory"],
            tensors["turn"],
            lanelet_map,
            tensors.get("valid_mask"),
        )
        total_frames = len(nav_texts)

        scene_out_dir = os.path.join(nav_labels_dir, clip_id)
        os.makedirs(scene_out_dir, exist_ok=True)

        for kf in clip_keyframes:
            frame_idx = kf["event_start_frame"]
            if frame_idx >= total_frames:
                logger.warning(
                    "  frame_idx=%d out of range (total=%d): %s", frame_idx, total_frames, clip_id
                )
                continue

            event_start_timestamp = frame_idx * 100_000
            nav_text = nav_texts[frame_idx]

            out_path = os.path.join(scene_out_dir, f"nav_{event_start_timestamp}.yaml")
            record = {
                "event_start_frame": frame_idx,
                "event_start_timestamp": event_start_timestamp,
                "meta_action": kf["meta_action"],
                "navigation_text": nav_text,
            }
            with open(out_path, "w") as f:
                yaml.dump(record, f, allow_unicode=True, default_flow_style=False)

        logger.info("  %s: %d keyframes done", clip_id, len(clip_keyframes))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    map_cache: dict[str, LaneletMapPolygonIndex] = {}

    dates = sorted(
        d for d in os.listdir(BATCH_OUTPUTS) if os.path.isdir(os.path.join(BATCH_OUTPUTS, d))
    )

    for date in dates:
        map_version = DATE_TO_MAP_VERSION.get(date)
        if map_version is None:
            logger.warning("No map version defined for date %s, skipping", date)
            continue

        if map_version not in map_cache:
            osm_path = os.path.join(MAP_BASE, map_version, "lanelet2_map.osm")
            map_cache[map_version] = LaneletMapPolygonIndex(osm_path)

        process_date(date, map_cache[map_version])

    logger.info("All dates processed.")


if __name__ == "__main__":
    main()
