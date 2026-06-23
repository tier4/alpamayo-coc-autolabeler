"""
Shared utilities for data curation scripts.

Common functions for parsing meta-actions, loading GT bounding boxes,
and constants used across curation/analysis/visualization scripts.
"""
import io
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import zstandard as zstd

ALL_ACTIONS = [
    # Longitudinal
    "Stop", "Reverse", "GentleAcceleration", "StrongAcceleration",
    "GentleDeceleration", "StrongDeceleration", "MaintainSpeed",
    # Lateral
    "GoStraight", "SteerLeft", "SteerRight",
    "SharpSteerLeft", "SharpSteerRight", "ReverseLeft", "ReverseRight",
    # Lane
    "LaneKeep", "LeftLaneChange", "RightLaneChange",
    "SlightlyShiftLeft", "SlightlyShiftRight", "TurnLeft", "TurnRight",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}

RARE_ACTIONS = {
    "StrongAcceleration", "StrongDeceleration",
    "LeftLaneChange", "RightLaneChange",
    "TurnLeft", "TurnRight",
    "SharpSteerLeft", "SharpSteerRight",
}

REDUNDANT_PATTERN_ACTIONS = {"Stop", "LaneKeep", "GoStraight", "MaintainSpeed"}

LABEL_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
    4: "pedestrian",
}

PEDESTRIAN_LABEL = 4


def load_zstd_npy(data: bytes) -> np.ndarray:
    dctx = zstd.ZstdDecompressor()
    return np.load(io.BytesIO(dctx.decompress(data)))


def parse_metaactions(filepath: str) -> list:
    segments = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r"(\w+)\s+-\s+Agent:<ego>,\s+Start:(\d+),\s+End:(\d+)", line
            )
            if m:
                segments.append({
                    "action": m.group(1),
                    "start": int(m.group(2)),
                    "end": int(m.group(3)),
                })
    return segments


def build_frame_action_map(segments: list, n_frames: int) -> dict:
    frame_map = defaultdict(set)
    for seg in segments:
        for f in range(seg["start"], min(seg["end"], n_frames)):
            frame_map[f].add(seg["action"])
    return frame_map


def build_rare_context_set(segments: list, context_frames: int, n_frames: int) -> set:
    rare_frames = set()
    for seg in segments:
        if seg["action"] in RARE_ACTIONS:
            start = max(0, seg["start"] - context_frames)
            end = min(n_frames, seg["end"] + context_frames)
            for f in range(start, end):
                rare_frames.add(f)
    return rare_frames


def load_gt_tar(tar_path: str) -> dict:
    import tarfile
    frames = {}
    boxes_buf = None
    boxes_frame = None
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            f = tar.extractfile(member)
            if f is None:
                continue
            arr = load_zstd_npy(f.read())
            fname = member.name.split("/")[-1]
            frame_idx = int(fname.split(".")[0])
            if "boxes" in fname:
                boxes_buf = arr
                boxes_frame = frame_idx
            elif "labels" in fname and boxes_buf is not None and boxes_frame == frame_idx:
                frames[frame_idx] = (boxes_buf, arr)
                boxes_buf = None
    return frames


def compute_frame_context(boxes: np.ndarray, labels: np.ndarray, radius: float = 20.0) -> dict:
    if len(boxes) == 0:
        return {"nearby_objects": 0, "nearby_pedestrians": 0}
    distances = np.sqrt(boxes[:, 0] ** 2 + boxes[:, 1] ** 2)
    nearby_mask = distances < radius
    nearby_ped = int(((labels == PEDESTRIAN_LABEL) & nearby_mask).sum())
    return {
        "nearby_objects": int(nearby_mask.sum()),
        "nearby_pedestrians": nearby_ped,
    }


def collect_clip_pairs(metaaction_root: str, data_root: str) -> list:
    pairs = []
    for date in sorted(os.listdir(metaaction_root)):
        final_dir = os.path.join(metaaction_root, date, "meta_actions", "final_outputs")
        if not os.path.isdir(final_dir):
            continue
        for fname in sorted(os.listdir(final_dir)):
            if not fname.endswith(".txt"):
                continue
            clip_id = fname.replace(".txt", "")
            meta_path = os.path.join(final_dir, fname)
            gt_tar_path = os.path.join(data_root, date, f"{clip_id}.gt.tar")
            clip_key = f"{date}/{clip_id}"
            pairs.append((meta_path, gt_tar_path, clip_key))
    return pairs
