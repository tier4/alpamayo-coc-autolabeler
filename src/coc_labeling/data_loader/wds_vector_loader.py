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

import glob
import io
import logging
import os
import tarfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zstandard as zstd
from omegaconf import DictConfig, open_dict


class WdsVectorLoader:
    """Load vector-state data from WebDataset format for CoC labeling."""

    def __init__(self, vector_config: Optional[DictConfig], data_config: DictConfig) -> None:
        self.data_config = data_config
        self.vector_config = None
        self.tar_dir = data_config.data_dir

        if vector_config is None:
            logging.info("WdsVectorLoader is not activated")
            return

        self.vector_config = self.expand_config(vector_config)
        logging.info("Initialized WdsVectorLoader with data_dir: %s", self.tar_dir)

    def _require_vector_config(self) -> DictConfig:
        if self.vector_config is None:
            raise ValueError("WdsVectorLoader is not activated (vector_config is None).")
        return self.vector_config

    @staticmethod
    def expand_config(vector_config: DictConfig) -> DictConfig:
        with open_dict(vector_config):
            if "trajdata" in vector_config:
                vector_config.trajdata.desired_dt = 1.0 / float(vector_config.fps)
                vector_config.trajdata.history_sec = float(vector_config.hist_length_sec)
                vector_config.trajdata.future_sec = float(vector_config.fut_length_sec)

            vector_config.hist_length_frame = int(vector_config.hist_length_sec * vector_config.fps)
            vector_config.fut_length_frame = int(vector_config.fut_length_sec * vector_config.fps)
            vector_config.freq_sample = int(1 / vector_config.time_interval)
            vector_config.hist_length_frame_sample = int(
                vector_config.hist_length_sec * vector_config.freq_sample
            )
            vector_config.fut_length_frame_sample = int(
                vector_config.fut_length_sec * vector_config.freq_sample
            )
            vector_config.sample_rate = int(vector_config.fps / vector_config.freq_sample)
        return vector_config

    def _load_zstd_numpy(self, data: bytes) -> np.ndarray:
        dctx = zstd.ZstdDecompressor()
        return np.load(io.BytesIO(dctx.decompress(data)))

    def _load_tar_data(self, clip_id: str) -> Dict[str, np.ndarray]:
        tar_path = os.path.join(self.tar_dir, f"{clip_id}.tar")
        if not os.path.exists(tar_path):
            matched = glob.glob(os.path.join(self.tar_dir, "**", f"{clip_id}.tar"), recursive=True)
            if not matched:
                raise FileNotFoundError(
                    f"WebDataset tar file not found: {clip_id}.tar in {self.tar_dir}"
                )
            tar_path = matched[0]

        data: Dict[str, np.ndarray] = {}
        with tarfile.open(tar_path, "r|") as tar:
            for member in tar:
                if member.name.endswith(".npy.zst"):
                    field_name = member.name.split(".")[-3]
                    if field_name in ["trajectory", "velocity"]:
                        f = tar.extractfile(member)
                        if f is not None:
                            data[field_name] = self._load_zstd_numpy(f.read())
                if len(data) == 2:
                    break
        return data

    def convert_speed_to_text(self, velocity: np.ndarray, event_start: int) -> Tuple[str, str, str]:
        cfg = self._require_vector_config()

        hist_start_frame = max(0, event_start - cfg.hist_length_frame)
        fut_end_frame = min(velocity.shape[0], event_start + cfg.fut_length_frame)

        ego_hist_vxvy = velocity[hist_start_frame:event_start:cfg.freq_sample]
        ego_fut_vxvy = velocity[event_start:fut_end_frame:cfg.freq_sample]

        dt = float(cfg.time_interval)
        hist_count = ego_hist_vxvy.shape[0]
        fut_count = ego_fut_vxvy.shape[0]

        hist_timestamps = [-(hist_count - 1 - i) * dt for i in range(hist_count)]
        fut_timestamps = [(i + 1) * dt for i in range(fut_count)]

        def _format_speed_series(timestamps: list, values: np.ndarray, axis_name: str) -> str:
            pairs = ", ".join(
                f"(t={ts:.1f}s, {axis_name}={float(v):.3f} m/s)"
                for ts, v in zip(timestamps, values)
            )
            return f"[{pairs}]"

        hist_long_series = _format_speed_series(
            hist_timestamps, ego_hist_vxvy[:, 0] if hist_count > 0 else np.array([]), "v_long"
        )
        hist_lat_series = _format_speed_series(
            hist_timestamps, ego_hist_vxvy[:, 1] if hist_count > 0 else np.array([]), "v_lat"
        )
        fut_long_series = _format_speed_series(
            fut_timestamps, ego_fut_vxvy[:, 0] if fut_count > 0 else np.array([]), "v_long"
        )
        fut_lat_series = _format_speed_series(
            fut_timestamps, ego_fut_vxvy[:, 1] if fut_count > 0 else np.array([]), "v_lat"
        )

        ego_hist_text = (
            "History speed samples:\n"
            f"- Longitudinal (m/s): {hist_long_series}\n"
            f"- Lateral (m/s): {hist_lat_series}"
        )
        ego_fut_text = (
            "Future speed samples:\n"
            f"- Longitudinal (m/s): {fut_long_series}\n"
            f"- Lateral (m/s): {fut_lat_series}"
        )

        ego_longitudinal_speed = np.concatenate(
            (ego_hist_vxvy[:, 0] if hist_count > 0 else [], ego_fut_vxvy[:, 0] if fut_count > 0 else [])
        )
        ego_lateral_speed = np.concatenate(
            (ego_hist_vxvy[:, 1] if hist_count > 0 else [], ego_fut_vxvy[:, 1] if fut_count > 0 else [])
        )
        all_timestamps = hist_timestamps + fut_timestamps
        all_long_series = _format_speed_series(all_timestamps, ego_longitudinal_speed, "v_long")
        all_lat_series = _format_speed_series(all_timestamps, ego_lateral_speed, "v_lat")

        ego_text = (
            "Ego speed time series (relative to event time t=0.0s):\n"
            f"- Longitudinal (m/s): {all_long_series}\n"
            f"- Lateral (m/s): {all_lat_series}\n"
            "Positive lateral speed means rightward motion, negative means leftward motion."
        )
        return ego_hist_text, ego_fut_text, ego_text

    def load(self, clip_id: str, event_start: int) -> Dict[str, Any]:
        if self.vector_config is None:
            return {}

        data = self._load_tar_data(clip_id)
        if "velocity" not in data:
            logging.warning("Velocity data not found for clip %s", clip_id)
            return {}

        velocity = data["velocity"]
        ego_hist_text, ego_fut_text, ego_text = self.convert_speed_to_text(velocity, event_start)

        return {
            "ego_hist_text": ego_hist_text,
            "ego_fut_text": ego_fut_text,
            "ego_text": ego_text,
        }
