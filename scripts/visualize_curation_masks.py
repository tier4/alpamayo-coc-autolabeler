"""
Visualize curation mask categories with front camera images.

Samples a few frames per category (A/B/C/D) from selected clips and
generates a grid image showing what kind of frames are kept vs dropped.

Usage:
    python scripts/visualize_curation_masks.py \
        --curation_masks ~/alpamayo_labels_metaaction/train/curation_masks.json \
        --metaaction_root ~/alpamayo_labels_metaaction/train \
        --data_root /mnt/storage_rdma/datasets/tier4/e2e \
        --output_dir ~/alpamayo_labels_metaaction/train/curation_mask_samples
"""
import argparse
import json
import os
import tarfile
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from curation_utils import (
    RARE_ACTIONS,
    build_frame_action_map,
    build_rare_context_set,
    parse_metaactions,
)

CATEGORY_INFO = {
    "A": {"label": "A: High value (KEPT 100%)", "color": "#C44E52", "kept": True},
    "B": {"label": "B: Standard (KEPT 100%)", "color": "#4C72B0", "kept": True},
    "C": {"label": "C: Low density (DROPPED 80%)", "color": "#999999", "kept": False},
    "D": {"label": "D: Empty stop (DROPPED 90%)", "color": "#CCCCCC", "kept": False},
}


def build_frame_categories(clip_data, segments, context_frames=20):
    n_frames = clip_data["total_frames"]
    valid_set = set(clip_data["valid_frames"])
    frame_action_map = build_frame_action_map(segments, n_frames)
    rare_context = build_rare_context_set(segments, context_frames, n_frames)

    # We don't have GT bbox here, so infer category from valid_frames + actions
    categories = {}
    for f in range(n_frames):
        actions = frame_action_map.get(f, set())
        if actions & RARE_ACTIONS or f in rare_context:
            categories[f] = "A"
        elif f in valid_set:
            categories[f] = "B"
        elif "Stop" in actions and f not in valid_set:
            categories[f] = "D"
        else:
            categories[f] = "C"
    return categories


def extract_front_camera(tar_path, frame_indices):
    cam_names = []
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if member.name.endswith("_CAM_FRONT.jpg"):
                cam_names.append(member.name)
    cam_names.sort()

    needed = {cam_names[i]: i for i in frame_indices if i < len(cam_names)}
    images = {}
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if member.name in needed:
                f = tar.extractfile(member)
                if f:
                    img_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        images[needed[member.name]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(images) >= len(needed):
                break
    return images


def sample_frames_per_category(categories, n_per_cat=4):
    by_cat = defaultdict(list)
    for f, cat in sorted(categories.items()):
        by_cat[cat].append(f)

    sampled = {}
    rng = np.random.RandomState(42)
    for cat in ["A", "B", "C", "D"]:
        frames = by_cat.get(cat, [])
        if len(frames) == 0:
            continue
        # Sample evenly spaced through the clip
        if len(frames) <= n_per_cat:
            sampled[cat] = frames
        else:
            indices = np.linspace(0, len(frames) - 1, n_per_cat, dtype=int)
            sampled[cat] = [frames[i] for i in indices]
    return sampled


def render_clip_grid(clip_key, categories, images, frame_action_map, sampled, output_path):
    cats_present = [c for c in ["A", "B", "C", "D"] if c in sampled]
    n_cols = max(len(v) for v in sampled.values())
    n_rows = len(cats_present)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(f"{clip_key}", fontsize=14, fontweight="bold", y=1.02)

    for row_idx, cat in enumerate(cats_present):
        frames = sampled[cat]
        info = CATEGORY_INFO[cat]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(frames):
                f = frames[col_idx]
                actions = frame_action_map.get(f, set())
                action_str = ", ".join(sorted(actions)) if actions else "none"
                kept = "KEPT" if info["kept"] or (cat == "C" and f % 5 == 0) or (cat == "D" and f % 10 == 0) else "DROPPED"

                if f in images:
                    ax.imshow(images[f])
                else:
                    ax.text(0.5, 0.5, "No image", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12, color="gray")

                title_color = "green" if kept == "KEPT" else "red"
                ax.set_title(f"f={f} [{kept}]\n{action_str}", fontsize=7, color=title_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(info["color"])
                    spine.set_linewidth(3)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        # Row label
        axes[row_idx][0].set_ylabel(info["label"], fontsize=9, fontweight="bold",
                                     rotation=0, labelpad=120, va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def select_diverse_clips(masks, n_clips=3):
    """Select clips with diverse category distributions."""
    clips = masks["clips"]
    scored = []
    for clip_key, clip_data in clips.items():
        cats = clip_data["category_counts"]
        # Prefer clips that have all 4 categories
        n_cats = sum(1 for c in ["A", "B", "C", "D"] if cats.get(c, 0) > 0)
        # Prefer clips with some A (rare actions)
        a_ratio = cats.get("A", 0) / max(clip_data["total_frames"], 1)
        d_count = cats.get("D", 0)
        scored.append((clip_key, n_cats, a_ratio, d_count))

    # Pick: 1 with most categories, 1 with high A ratio, 1 with high D count
    scored_by_diversity = sorted(scored, key=lambda x: (x[1], x[2]), reverse=True)
    scored_by_rare = sorted(scored, key=lambda x: x[2], reverse=True)
    scored_by_stop = sorted(scored, key=lambda x: x[3], reverse=True)

    selected = []
    seen = set()
    for source in [scored_by_diversity, scored_by_rare, scored_by_stop]:
        for item in source:
            if item[0] not in seen:
                selected.append(item[0])
                seen.add(item[0])
                break
    return selected[:n_clips]


def main():
    parser = argparse.ArgumentParser(description="Visualize curation mask categories.")
    parser.add_argument("--curation_masks", required=True)
    parser.add_argument("--metaaction_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_clips", type=int, default=3)
    parser.add_argument("--n_per_cat", type=int, default=4)
    args = parser.parse_args()

    with open(args.curation_masks) as f:
        masks = json.load(f)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.curation_masks), "curation_mask_samples")
    os.makedirs(args.output_dir, exist_ok=True)

    selected_clips = select_diverse_clips(masks, args.n_clips)
    print(f"Selected {len(selected_clips)} clips for visualization:")

    for clip_key in selected_clips:
        clip_data = masks["clips"][clip_key]
        date, clip_id = clip_key.split("/")
        print(f"\n  {clip_key}: {clip_data['total_frames']} frames, "
              f"cats={clip_data['category_counts']}, rare={clip_data['rare_actions']}")

        # Parse meta-actions
        meta_path = os.path.join(args.metaaction_root, date, "meta_actions", "final_outputs", f"{clip_id}.txt")
        segments = parse_metaactions(meta_path)

        # Build categories
        categories = build_frame_categories(clip_data, segments)
        frame_action_map = build_frame_action_map(segments, clip_data["total_frames"])

        # Sample frames
        sampled = sample_frames_per_category(categories, args.n_per_cat)
        all_frames = []
        for frames in sampled.values():
            all_frames.extend(frames)
        all_frames = sorted(set(all_frames))

        # Extract camera images
        tar_path = os.path.join(args.data_root, date, f"{clip_id}.tar")
        if not os.path.exists(tar_path):
            print(f"    [SKIP] tar not found: {tar_path}")
            continue

        print(f"    Extracting {len(all_frames)} frames from tar...")
        images = extract_front_camera(tar_path, all_frames)
        print(f"    Got {len(images)} images")

        # Render grid
        out_path = os.path.join(args.output_dir, f"{date}_{clip_id}.png")
        render_clip_grid(clip_key, categories, images, frame_action_map, sampled, out_path)
        print(f"    Saved: {out_path}")

    print(f"\nDone! Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
