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

"""Batch Step 3 entrypoint: loads VLM weights once, then processes all dates sequentially.

Usage inside Docker:
    python -m coc_labeling.batch_data_labeling \
        --dates 2025-11-19 2025-12-09 ... \
        --data_dir_root /dataset_root \
        --batch_outputs_root /workspace/batch_outputs \
        --coc_cache_dir /coc_cache \
        --model_name qwen3.5_397b_fp8
"""

import argparse
import logging
import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from coc_labeling.agents.labeling_agent import LabelingAgent
from coc_labeling.model_clients.runtime_config import ModelRuntimeConfig
from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.utils import io as my_io_utils


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--data_dir_root", required=True)
    parser.add_argument("--batch_outputs_root", required=True)
    parser.add_argument("--coc_cache_dir", required=True)
    parser.add_argument("--model_name", default="qwen3.5_397b_fp8")
    args = parser.parse_args()

    first_date = args.dates[0]

    # Build Hydra config using first date's paths; these will be overridden per date below.
    GlobalHydra.instance().clear()
    # config_path is relative to this file's directory (src/coc_labeling/)
    initialize(config_path="config", version_base=None)
    cfg = compose(
        config_name="base_config_vlm_rel_ts",
        overrides=[
            "+data_format=webdataset",
            f"model_name={args.model_name}",
            "data_loader.video.save_segment_videos=false",
            f"data.data_dir={args.data_dir_root}/{first_date}",
            f"data.cache_dir={args.coc_cache_dir}",
            f"data.meta_action_dir={args.batch_outputs_root}/{first_date}/meta_actions/final_outputs",
            f"data.segment_config_path={args.batch_outputs_root}/{first_date}/keyframes/segments_relative_timestamp_sampled.json",
            f"exp_name=coc_{first_date}",
        ],
    )

    runtime_config = ModelRuntimeConfig(timeout_sec=TIMEOUT_MAX)

    # Create LabelingAgent ONCE — VLM weights are loaded here (~7 min)
    logger.info("Loading model weights (done only once for all %d dates)...", len(args.dates))
    first_save_root = os.path.join(args.batch_outputs_root, first_date, "coc_labels")
    my_io_utils.mkdir_if_missing(first_save_root)

    labeling_agent = LabelingAgent(
        cfg=cfg,
        save_root=first_save_root,
        verbose=cfg.verbose,
        mode=cfg.mode,
        runtime_config=runtime_config,
    )
    logger.info("Model loaded. Processing %d dates.", len(args.dates))

    for date in args.dates:
        logger.info("=" * 60)
        logger.info("Processing date: %s", date)

        segment_config_path = (
            f"{args.batch_outputs_root}/{date}/keyframes/segments_relative_timestamp_sampled.json"
        )
        if not os.path.isfile(segment_config_path):
            logger.warning("Segment config not found, skipping: %s", segment_config_path)
            continue

        # Update per-date paths. Existing keys can be set directly in struct mode.
        OmegaConf.update(cfg, "data.data_dir", f"{args.data_dir_root}/{date}")
        OmegaConf.update(cfg, "data.cache_dir", args.coc_cache_dir)
        OmegaConf.update(
            cfg, "data.meta_action_dir",
            f"{args.batch_outputs_root}/{date}/meta_actions/final_outputs",
        )
        OmegaConf.update(cfg, "data.segment_config_path", segment_config_path)
        OmegaConf.update(cfg, "exp_name", f"coc_{date}")

        save_root = os.path.join(args.batch_outputs_root, date, "coc_labels")
        my_io_utils.mkdir_if_missing(save_root)

        try:
            labeling_agent.parse_dataset(cfg, save_root)
            labeling_agent.run()
        except Exception:
            logger.exception("Error processing date %s — continuing with next date.", date)

    logger.info("Batch processing complete.")


if __name__ == "__main__":
    main()
