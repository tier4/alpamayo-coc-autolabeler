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

"""Ego speed features for TIER IV T4 scenes, without trajdata.

VectorLoader reaches ego speed through trajdata's UnifiedDataset, which
T4 is not: adding it would mean carrying the pinned NVlabs fork for the sake of
one velocity array, the same trade meta_action.t4_labeling already declined
for step 1.

The VLM CoC agent reads exactly one field off the vector data -- ego_text --
and everything needed to build it sits in each T4 scene's
derived/scalars.npz, which loads in ~16 ms.

Two facts about that file are established by measurement, not documentation,
since it is produced by the T4 export rather than by any code in this repo:

* trajectory is [T, 4] holding (x, y, cos_heading, sin_heading) in
  the map frame. Columns 2 and 3 square-sum to 1.0, and projecting the
  position-differentiated velocity onto (cos, sin) recovers the full speed
  (mean v_long / |v| = 1.0000 over 11 scenes while moving), which the
  (sin, cos) reading does not (0.2228).
* velocity carries speed in column 0 and zeros in column 1, so it cannot
  supply lateral speed and is not used here.

Velocity comes from differentiating the pose track, matching how
meta_action.t4_labeling derives the speeds the meta-actions are cut from, so
the numbers in the prompt agree with the meta-action spans beside them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig

from coc_labeling.data_loader.vector_loader import format_ego_speed_text

#: Scene-relative path of the export's per-frame scalars.
SCALARS_RELPATH = Path("derived") / "scalars.npz"

#: Separator meta_action.t4_labeling.clip_id_for joins scene path parts with.
T4_CLIP_ID_SEPARATOR = "__"


def ego_speed_arrays(
    trajectory: np.ndarray, event_start: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ego velocity in the ego frame frozen at the event, as (v_long, v_lat).

    The frame is frozen at event_start rather than tracking the instantaneous
    heading: projected on its own heading a non-slipping vehicle has no lateral
    speed at all (measured v_long / |v| = 1.0), so a per-frame projection
    would report noise. Frozen at the event, a turn shows up as exactly what the
    prompt says lateral speed means -- sustained sideways motion, positive to the
    right. Validated against the step-1 meta-actions on 196 spans: SteerRight
    positive 85/85, SteerLeft negative 81/81, and the Sharp* variants about twice
    the magnitude.

    Args:
        trajectory: [T, 4] array of (x, y, cos_heading, sin_heading).
        event_start: Frame index defining the reference frame, t=0.
        dt: Seconds between frames.

    Returns:
        (v_long, v_lat), each [T], in metres per second.
    """
    x, y = trajectory[:, 0], trajectory[:, 1]
    cos_h, sin_h = trajectory[:, 2], trajectory[:, 3]
    v_x = np.gradient(x, dt)
    v_y = np.gradient(y, dt)
    reference = int(np.clip(event_start, 0, len(trajectory) - 1))
    cos_0, sin_0 = float(cos_h[reference]), float(sin_h[reference])
    v_long = v_x * cos_0 + v_y * sin_0
    v_lat = v_x * sin_0 - v_y * cos_0
    return v_long, v_lat


class T4VectorLoader:
    """Build ego_text for a T4 segment from the scene's derived scalars."""

    def __init__(self, vector_config: Optional[DictConfig], data_config: DictConfig) -> None:
        """
        Args:
            vector_config: Vector settings; only the temporal fields are read.
            data_config: Data paths; video_dir is the T4 dataset root, since a
                T4 clip id is its scene directory flattened.
        """
        self.vector_config = vector_config
        self.data_config = data_config
        # Kept for API parity with VectorLoader, whose trajdata handles the
        # segment generators consult. T4 has neither.
        self.dataset = None
        self.scene_ts_idx_map = None

    def _scene_dir(self, clip_id: str) -> Path:
        """Recover a scene directory from a flattened T4 clip id."""
        return Path(self.data_config.video_dir).joinpath(
            *clip_id.split(T4_CLIP_ID_SEPARATOR)
        )

    def load(self, clip_id: str, event_start: int) -> Dict[str, Any]:
        """Load ego speed text for one segment.

        Args:
            clip_id: Flattened T4 scene directory.
            event_start: Event start frame index, which for T4 is also the video
                frame index and the LiDAR frame index.

        Returns:
            Dictionary with the four ego speed arrays and the three text blocks,
            or an empty dictionary when the scene has no derived scalars, so a
            partial export degrades to a video-and-meta-action-only prompt
            instead of failing the run.
        """
        if self.vector_config is None:
            return {}

        scalars_path = self._scene_dir(clip_id) / SCALARS_RELPATH
        if not scalars_path.is_file():
            logging.warning("No T4 derived scalars for clip_id=%s at %s", clip_id, scalars_path)
            return {}

        fps = float(self.vector_config.fps)
        stride = max(1, int(round(fps * float(self.vector_config.time_interval))))
        hist_frames = int(float(self.vector_config.hist_length_sec) * fps)
        fut_frames = int(float(self.vector_config.fut_length_sec) * fps)

        with np.load(scalars_path) as scalars:
            trajectory = np.asarray(scalars["trajectory"], dtype=np.float64)

        v_long, v_lat = ego_speed_arrays(trajectory, event_start, 1.0 / fps)

        # Clamped the same way VideoLoader.calculate_segment_bounds clamps, so the
        # speed samples line up with the frames the model is shown.
        hist_start = max(event_start - hist_frames, 0)
        fut_end = min(event_start + fut_frames, len(trajectory))
        data_dict: Dict[str, Any] = {
            "ego_hist_longitudinal_speed": v_long[hist_start:event_start:stride],
            "ego_hist_lateral_speed": v_lat[hist_start:event_start:stride],
            "ego_fut_longitudinal_speed": v_long[event_start:fut_end:stride],
            "ego_fut_lateral_speed": v_lat[event_start:fut_end:stride],
        }
        ego_hist_text, ego_fut_text, ego_text = format_ego_speed_text(
            data_dict, float(self.vector_config.time_interval)
        )
        data_dict["ego_hist_text"] = ego_hist_text
        data_dict["ego_fut_text"] = ego_fut_text
        data_dict["ego_text"] = ego_text
        return data_dict
