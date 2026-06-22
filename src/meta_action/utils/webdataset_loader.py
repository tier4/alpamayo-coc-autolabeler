# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import os
import tarfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import zstandard as zstd

from meta_action.utils.constant import START_TS, STEP

logger = logging.getLogger(__name__)


def load_zstd_numpy(data: bytes) -> np.ndarray:
    """Decompress zstandard bytes to a numpy array."""
    dctx = zstd.ZstdDecompressor()
    return np.load(io.BytesIO(dctx.decompress(data)))


class _FakeSceneBatch:
    """Minimal duck-typed SceneBatch for lane-aware WebDataset processing.

    Provides:
      .vector_maps[0]               -> Lanelet2VectorMap
      .centered_world_from_agent_tf[0] -> 3x3 identity (coords already in world frame)
    """

    def __init__(self, vector_map: Any) -> None:
        self.vector_maps = [vector_map]
        self.centered_world_from_agent_tf = [np.eye(3, dtype=np.float64)]


class WebDatasetTemporalScenario:
    """A duck-typed TemporalScenario that reads directly from WebDataset extracted arrays."""

    def __init__(
        self,
        clip_id: str,
        data: Dict[str, Any],
        cfg: Optional[dict] = None,
        vector_map: Any = None,
    ) -> None:
        self.clip_id = clip_id
        self.cfg = cfg
        self.all_agents_names = ["ego"]

        trajectory = data["trajectory"]  # [T, 4] (x, y, cos_h, sin_h)
        velocity = data["velocity"]      # [T, 2] (vx, vy)

        self.scene_len = trajectory.shape[0]
        self.all_ts = list(range(START_TS, self.scene_len, STEP))
        self.segment_cache: Dict[str, Dict[str, Any]] = {}

        # Precompute kinematics
        speed = np.linalg.norm(velocity, axis=1)  # [T]
        cos_h = trajectory[:, 2]
        sin_h = trajectory[:, 3]
        speed_along_heading = velocity[:, 0] * cos_h + velocity[:, 1] * sin_h
        h = np.arctan2(sin_h, cos_h)

        xyh = np.zeros((self.scene_len, 3))
        xyh[:, 0] = trajectory[:, 0]
        xyh[:, 1] = trajectory[:, 1]
        xyh[:, 2] = h

        ego_xyzh = np.concatenate([xyh[:, :2], np.zeros((self.scene_len, 1)), h[:, None]], axis=1)

        self.agent_trajdata = {
            "agent_names": ["ego"],
            "agent_xyh": torch.from_numpy(xyh).unsqueeze(0).float(),
            "agent_speed": torch.from_numpy(speed).unsqueeze(0).float(),
            "agent_speed_along_heading": torch.from_numpy(speed_along_heading).unsqueeze(0).float(),
            "agent_type": [0],
            "ego_xyzh": torch.from_numpy(ego_xyzh).float()
        }

        # Lane-aware mode: set up scene_batch and ego_lr
        if vector_map is not None:
            self.scene_batch = _FakeSceneBatch(vector_map)
            from meta_action.utils.trajdata.lanegraph import update_ego_lane_relation
            self.ego_lr = update_ego_lane_relation(self.scene_batch, ego_xyzh)

    def get_world_states(self, states: np.ndarray, scene_batch: Any) -> np.ndarray:
        """Map agent-frame states to world-frame. Identity for WebDataset (already world coords)."""
        return states

    def get_tag_motions(self, motion_class: Any, ego_only: bool = True) -> List[Any]:
        motions = []
        for agent in self.all_agents_names:
            if (ego_only and agent == "ego") or (not ego_only):
                motions.extend(motion_class.get_motion_for_scenario(agent, self))
        return motions


def process_wds_clip(
    clip_id: str,
    save_root: str,
    tar_path: str,
    meta_action_classes: List[Any],
    vector_map: Any = None,
) -> str:
    """Read a WebDataset tar, construct a scenario, and process meta-actions."""
    save_file = os.path.join(save_root, f"{clip_id}.json")

    # Load required arrays from the tar file
    data = {}
    # Use streaming mode "r|" to avoid random seeks on network filesystems (Lustre etc.)
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if member.name.endswith(".npy.zst"):
                field_name = member.name.split(".")[-3] # scene_id.field.npy.zst
                if field_name in ["trajectory", "velocity"]:
                    f = tar.extractfile(member)
                    if f is not None:
                        data[field_name] = load_zstd_numpy(f.read())
            if len(data) == 2:
                break  # early exit once both arrays are found

    if "trajectory" not in data or "velocity" not in data:
        logger.error("Missing trajectory or velocity in %s", tar_path)
        return clip_id

    scenario = WebDatasetTemporalScenario(
        clip_id=clip_id, data=data, vector_map=vector_map,
    )

    results = []
    for meta_action in meta_action_classes:
        output = scenario.get_tag_motions(meta_action)
        results += output

    if len(results) > 0:
        res_strs = [str(ma) for ma in results]
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(res_strs, f)

    return clip_id
