"""
Analyze meta-action segments with surrounding GT bbox context.

Combines per-frame GT bounding boxes with meta-action temporal segments to
produce joint statistics: what objects are present during each action type,
at what distances, and in what densities.

Usage:
    python scripts/analyze_metaaction_context.py \
        --metaaction_root ~/alpamayo_labels_metaaction/train \
        --data_root /mnt/storage_rdma/datasets/tier4/e2e \
        --output_dir ~/alpamayo_labels_metaaction/train/context_analysis
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm

from curation_utils import (
    ALL_ACTIONS,
    LABEL_NAMES,
    build_frame_action_map,
    collect_clip_pairs,
    load_gt_tar,
    parse_metaactions,
)


def compute_distances(boxes: np.ndarray) -> np.ndarray:
    return np.sqrt(boxes[:, 0] ** 2 + boxes[:, 1] ** 2)


def process_clip(meta_path: str, gt_tar_path: str) -> Optional[dict]:
    if not os.path.exists(gt_tar_path):
        return None
    segments = parse_metaactions(meta_path)
    if not segments:
        return None
    gt_frames = load_gt_tar(gt_tar_path)
    if not gt_frames:
        return None

    n_frames = max(gt_frames.keys()) + 1
    frame_action_map = build_frame_action_map(segments, n_frames)

    action_stats = {}
    for action in ALL_ACTIONS:
        action_stats[action] = {
            "n_frames": 0,
            "class_counts": Counter(),
            "class_nearby_counts": Counter(),
            "total_objects": 0,
            "total_nearby": 0,
            "distances": [],
        }

    for frame_idx, (boxes, labels) in gt_frames.items():
        active_actions = frame_action_map.get(frame_idx, set())
        if not active_actions or len(boxes) == 0:
            continue
        distances = compute_distances(boxes)
        for action in active_actions:
            if action not in action_stats:
                continue
            stats = action_stats[action]
            stats["n_frames"] += 1
            stats["total_objects"] += len(boxes)
            for label, dist in zip(labels, distances):
                cls_name = LABEL_NAMES.get(int(label), f"unk_{label}")
                stats["class_counts"][cls_name] += 1
                stats["distances"].append(float(dist))
                if dist < 20.0:
                    stats["class_nearby_counts"][cls_name] += 1
                    stats["total_nearby"] += 1

    return action_stats


def merge_stats(all_stats: list) -> dict:
    merged = {}
    for action in ALL_ACTIONS:
        merged[action] = {
            "n_frames": 0, "class_counts": Counter(), "class_nearby_counts": Counter(),
            "total_objects": 0, "total_nearby": 0, "distances": [],
        }
    for clip_stats in all_stats:
        if clip_stats is None:
            continue
        for action in ALL_ACTIONS:
            s = clip_stats[action]
            m = merged[action]
            m["n_frames"] += s["n_frames"]
            m["class_counts"] += s["class_counts"]
            m["class_nearby_counts"] += s["class_nearby_counts"]
            m["total_objects"] += s["total_objects"]
            m["total_nearby"] += s["total_nearby"]
            m["distances"].extend(s["distances"])
    return merged


def plot_objects_per_action(ax, stats):
    classes = list(LABEL_NAMES.values())
    actions = [a for a in ALL_ACTIONS if stats[a]["n_frames"] > 0]
    data = np.zeros((len(actions), len(classes)))
    for i, action in enumerate(actions):
        n = stats[action]["n_frames"]
        for j, cls in enumerate(classes):
            data[i, j] = stats[action]["class_counts"].get(cls, 0) / max(n, 1)
    x = np.arange(len(actions))
    bottom = np.zeros(len(actions))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for j, cls in enumerate(classes):
        ax.barh(x, data[:, j], left=bottom, label=cls, color=colors[j % len(colors)])
        bottom += data[:, j]
    ax.set_yticks(x)
    ax.set_yticklabels(actions, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Objects per Frame")
    ax.set_title("Average Object Count per Frame by Meta-Action", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)


def plot_nearby_objects_per_action(ax, stats):
    classes = list(LABEL_NAMES.values())
    actions = [a for a in ALL_ACTIONS if stats[a]["n_frames"] > 0]
    data = np.zeros((len(actions), len(classes)))
    for i, action in enumerate(actions):
        n = stats[action]["n_frames"]
        for j, cls in enumerate(classes):
            data[i, j] = stats[action]["class_nearby_counts"].get(cls, 0) / max(n, 1)
    x = np.arange(len(actions))
    bottom = np.zeros(len(actions))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for j, cls in enumerate(classes):
        ax.barh(x, data[:, j], left=bottom, label=cls, color=colors[j % len(colors)])
        bottom += data[:, j]
    ax.set_yticks(x)
    ax.set_yticklabels(actions, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Nearby Objects (<20m) per Frame")
    ax.set_title("Average Nearby Objects (<20m) per Frame by Meta-Action", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)


def plot_pedestrian_density_heatmap(ax, stats):
    actions = [a for a in ALL_ACTIONS if stats[a]["n_frames"] > 0]
    ped_density = [stats[a]["class_nearby_counts"].get("pedestrian", 0) / max(stats[a]["n_frames"], 1) for a in actions]
    colors = ["#C44E52" if d > 1.0 else "#DD8452" if d > 0.3 else "#4C72B0" for d in ped_density]
    bars = ax.barh(range(len(actions)), ped_density, color=colors)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Pedestrians (<20m) per Frame")
    ax.set_title("Pedestrian Density near Ego by Meta-Action", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, ped_density):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)


def plot_distance_distribution(ax, stats):
    focus = ["TurnLeft", "TurnRight", "LeftLaneChange", "RightLaneChange", "Stop", "StrongDeceleration", "GoStraight"]
    focus = [a for a in focus if stats[a]["distances"]]
    colors = plt.cm.tab10(np.linspace(0, 1, len(focus)))
    for i, action in enumerate(focus):
        dists = np.array(stats[action]["distances"])
        dists = dists[dists < 80]
        if len(dists) == 0:
            continue
        ax.hist(dists, bins=40, alpha=0.5, label=f"{action} (n={len(dists)})", color=colors[i], density=True)
    ax.set_xlabel("Distance from Ego (m)")
    ax.set_ylabel("Density")
    ax.set_title("Object Distance Distribution during Key Actions", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)


def plot_interaction_scenarios(ax, stats):
    actions = [a for a in ALL_ACTIONS if stats[a]["n_frames"] > 0]
    classes = list(LABEL_NAMES.values())
    scenarios = []
    for action in actions:
        n = stats[action]["n_frames"]
        if n == 0:
            continue
        for cls in classes:
            nearby = stats[action]["class_nearby_counts"].get(cls, 0)
            avg = nearby / n
            if avg > 0.1:
                scenarios.append((f"{action} + {cls}", avg, n))
    scenarios.sort(key=lambda x: x[1], reverse=True)
    top = scenarios[:25]
    labels = [s[0] for s in top]
    values = [s[1] for s in top]
    total_frames = [s[2] for s in top]
    bars = ax.barh(range(len(labels)), values, color="#55A868")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Nearby Count (<20m) per Frame")
    ax.set_title("Top Action + Object Interaction Scenarios", fontsize=11, fontweight="bold")
    for bar, val, nf in zip(bars, values, total_frames):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f} ({nf} frames)", va="center", fontsize=6)


def plot_action_complexity_score(ax, stats):
    actions = [a for a in ALL_ACTIONS if stats[a]["n_frames"] > 0]
    complexity = []
    for action in actions:
        n = stats[action]["n_frames"]
        if n == 0:
            complexity.append(0)
            continue
        avg_nearby = stats[action]["total_nearby"] / n
        n_classes = len([c for c in LABEL_NAMES.values() if stats[action]["class_nearby_counts"].get(c, 0) > 0])
        complexity.append(avg_nearby * (1 + 0.2 * n_classes))
    sort_idx = np.argsort(complexity)[::-1]
    sorted_actions = [actions[i] for i in sort_idx]
    sorted_complexity = [complexity[i] for i in sort_idx]
    colors = ["#C44E52" if c > 5 else "#DD8452" if c > 2 else "#4C72B0" for c in sorted_complexity]
    bars = ax.barh(range(len(sorted_actions)), sorted_complexity, color=colors)
    ax.set_yticks(range(len(sorted_actions)))
    ax.set_yticklabels(sorted_actions, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Complexity Score")
    ax.set_title("Action Complexity Score (higher = more complex surroundings)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, sorted_complexity):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze meta-action + GT bbox context.")
    parser.add_argument("--metaaction_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_clips", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.metaaction_root, "context_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    clip_pairs = collect_clip_pairs(args.metaaction_root, args.data_root)
    if args.max_clips:
        clip_pairs = clip_pairs[:args.max_clips]

    print(f"Processing {len(clip_pairs)} clips...")
    all_stats = []
    for i, (meta_path, gt_tar_path, clip_key) in enumerate(clip_pairs):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(clip_pairs)}] {clip_key}")
        stats = process_clip(meta_path, gt_tar_path)
        all_stats.append(stats)

    n_valid = sum(1 for s in all_stats if s is not None)
    print(f"  Valid clips: {n_valid}/{len(clip_pairs)}")

    print("Merging statistics...")
    merged = merge_stats(all_stats)

    print("\n=== Summary ===")
    for action in ALL_ACTIONS:
        m = merged[action]
        if m["n_frames"] == 0:
            continue
        avg_obj = m["total_objects"] / m["n_frames"]
        avg_nearby = m["total_nearby"] / m["n_frames"]
        ped_nearby = m["class_nearby_counts"].get("pedestrian", 0) / m["n_frames"]
        print(f"  {action:25s}: {m['n_frames']:7d} frames, avg_obj={avg_obj:.1f}, nearby={avg_nearby:.1f}, ped_nearby={ped_nearby:.2f}")

    print("\nGenerating visualizations...")
    pdf_path = os.path.join(args.output_dir, "context_analysis.pdf")
    png_dir = os.path.join(args.output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    plot_specs = [
        ("01_objects_per_action", (13, 9), lambda ax: plot_objects_per_action(ax, merged)),
        ("02_nearby_objects_per_action", (13, 9), lambda ax: plot_nearby_objects_per_action(ax, merged)),
        ("03_pedestrian_density", (12, 8), lambda ax: plot_pedestrian_density_heatmap(ax, merged)),
        ("04_distance_distribution", (12, 6), lambda ax: plot_distance_distribution(ax, merged)),
        ("05_interaction_scenarios", (14, 9), lambda ax: plot_interaction_scenarios(ax, merged)),
        ("06_action_complexity", (12, 8), lambda ax: plot_action_complexity_score(ax, merged)),
    ]

    with PdfPages(pdf_path) as pdf:
        for name, figsize, plot_fn in plot_specs:
            fig, ax = plt.subplots(figsize=figsize)
            plot_fn(ax)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(os.path.join(png_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"  PDF: {pdf_path}")
    print(f"  PNGs: {png_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
