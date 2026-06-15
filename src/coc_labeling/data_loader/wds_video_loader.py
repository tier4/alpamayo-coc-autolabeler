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

import logging
import os
import tarfile

import cv2
import numpy as np

from coc_labeling.data_loader.video_loader import ProcessedVideoData, VideoFrameData, VideoLoader

logger = logging.getLogger(__name__)


class WdsVideoLoader(VideoLoader):
    """VideoLoader variant that reads frames from WebDataset tar archives."""

    def _get_video_filepath(self, clip_id: str) -> str | None:
        if not self.video_config.use_fpv:
            return None
        return os.path.join(self.data_config.data_dir, f"{clip_id}.tar")

    def _get_video_timestamps_filepath(self, clip_id: str) -> str | None:
        return None

    def _load_timestamp_mapping(self, clip_id: str) -> list[tuple[int, int]]:
        if clip_id in self._timestamp_mappings:
            return self._timestamp_mappings[clip_id]

        tar_path = self._get_video_filepath(clip_id)
        count = 0
        # Streaming mode: no seeks, fast sequential scan of headers only
        with tarfile.open(tar_path, "r|") as tar:
            for member in tar:
                if member.name.lower().endswith("cam_front.jpg"):
                    count += 1

        mapping = [(i, i * 100000) for i in range(count)]
        self._timestamp_mappings[clip_id] = mapping
        return mapping

    def _load_video_frames(self, video_path: str, event_start: int | None) -> VideoFrameData:
        source_fps = 10
        event_start_new = event_start
        frame_offset = 0
        frame_interval = 1

        if source_fps != self.video_config.fps:
            if source_fps % self.video_config.fps != 0:
                frame_interval = max(1, int(source_fps / self.video_config.fps))
            else:
                frame_interval = int(source_fps / self.video_config.fps)
            if self.video_config.event_start_from_source_fps and event_start is not None:
                frame_offset = event_start % frame_interval
                event_start_new = int(event_start / frame_interval)

        # Pass 1 (streaming): collect sorted cam_front.jpg names — reads headers only, no JPEG data
        img_members: list[str] = []
        with tarfile.open(video_path, "r|") as tar:
            for member in tar:
                if member.name.lower().endswith("cam_front.jpg"):
                    img_members.append(member.name)
        img_members.sort()

        if not img_members:
            logger.warning("No cam_front.jpg found in %s", video_path)
            return VideoFrameData(frames=[], event_start=event_start_new, frame_indices=[])

        if event_start is not None and event_start_new is not None:
            hist_length_frame = int(self.video_config.hist_length_sec * self.video_config.fps)
            fut_length_frame = int(self.video_config.fut_length_sec * self.video_config.fps)
            sampled_start = max(0, int(event_start_new) - hist_length_frame)
            sampled_end = int(event_start_new) + fut_length_frame
            source_start = frame_offset + sampled_start * frame_interval
            source_end_exclusive = frame_offset + sampled_end * frame_interval
        else:
            source_start = frame_offset
            source_end_exclusive = len(img_members)

        target_indices = list(range(
            max(0, source_start),
            min(source_end_exclusive, len(img_members)),
            frame_interval,
        ))
        target_names = {img_members[i]: i for i in target_indices}
        remaining = len(target_names)

        # Pass 2 (streaming): extract JPEG data for target frames only, with early exit
        frames_bgr: list[tuple[int, np.ndarray]] = []
        with tarfile.open(video_path, "r|") as tar:
            for member in tar:
                if member.name in target_names:
                    f = tar.extractfile(member)
                    if f is not None:
                        img_array = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            frames_bgr.append((target_names[member.name], img))
                            remaining -= 1
                            if remaining == 0:
                                break  # all target frames collected

        frames_bgr.sort(key=lambda x: x[0])
        frame_indices = [x[0] for x in frames_bgr]
        frames_bgr_out = [x[1] for x in frames_bgr]

        event_start_local = None
        if event_start is not None and event_start_new is not None:
            hist_length_frame = int(self.video_config.hist_length_sec * self.video_config.fps)
            sampled_start = max(0, int(event_start_new) - hist_length_frame)
            event_start_local = int(event_start_new) - sampled_start

        return VideoFrameData(
            frames=frames_bgr_out,
            event_start=event_start_local,
            frame_indices=frame_indices,
        )

    def _load_bev_frames(self, clip_id: str, event_start: int | None) -> ProcessedVideoData:
        return ProcessedVideoData([], [], [], [], None)

    def _extract_source_fps_segment(
        self, video_path: str, start_idx: int, end_idx: int
    ) -> tuple[list[np.ndarray], float]:
        if start_idx < 0 or end_idx < start_idx:
            return [], float(self.video_config.fps)

        # Pass 1 (streaming): collect sorted names
        img_members: list[str] = []
        with tarfile.open(video_path, "r|") as tar:
            for member in tar:
                if member.name.lower().endswith("cam_front.jpg"):
                    img_members.append(member.name)
        img_members.sort()

        target_names = set(img_members[start_idx: end_idx + 1])
        remaining = len(target_names)

        # Pass 2 (streaming): extract frames with early exit
        frames_dict: dict[str, np.ndarray] = {}
        with tarfile.open(video_path, "r|") as tar:
            for member in tar:
                if member.name in target_names:
                    f = tar.extractfile(member)
                    if f is not None:
                        img_array = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            frames_dict[member.name] = img
                            remaining -= 1
                            if remaining == 0:
                                break

        frames_bgr = [frames_dict[name] for name in sorted(frames_dict.keys())]
        return frames_bgr, 10.0
