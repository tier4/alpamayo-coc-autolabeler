"""
Detailed curation visualization: meta-action + nearby object context + mask status.

Shows what data was kept/dropped and why, combining ego meta-actions with
surrounding object context.

Usage:
    python scripts/visualize_curation_detail.py \
        --curation_masks ~/alpamayo_labels_metaaction/train/curation_masks.json \
        --metaaction_root ~/alpamayo_labels_metaaction/train \
        --data_root /mnt/storage_rdma/datasets/tier4/e2e \
        --output_dir ~/alpamayo_labels_metaaction/train/curation_detail
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm

from curation_utils import (
    ALL_ACTIONS,
    LABEL_NAMES,
    PEDESTRIAN_LABEL,
    RARE_ACTIONS,
    build_frame_action_map,
    build_rare_context_set,
    compute_frame_context,
    load_gt_tar,
    parse_metaactions,
)

CAT_COLORS = {"A": "#C44E52", "B": "#4C72B0", "C": "#AAAAAA", "D": "#DDDDDD"}


def process_clip(meta_path, gt_tar_path, clip_data, context_frames=20):
    segments = parse_metaactions(meta_path)
    if not segments:
        return None

    gt_frames = load_gt_tar(gt_tar_path) if os.path.exists(gt_tar_path) else {}

    n_frames = clip_data["total_frames"]
    valid_set = set(clip_data["valid_frames"])
    frame_action_map = build_frame_action_map(segments, n_frames)
    rare_context = build_rare_context_set(segments, context_frames, n_frames)

    records = []
    for f in range(n_frames):
        actions = frame_action_map.get(f, set())
        boxes_labels = gt_frames.get(f)
        if boxes_labels is not None:
            ctx = compute_frame_context(boxes_labels[0], boxes_labels[1])
            nearby_objects = ctx["nearby_objects"]
            nearby_ped = ctx["nearby_pedestrians"]
            nearby_car = int(((boxes_labels[1] == 0) & (np.sqrt(boxes_labels[0][:, 0]**2 + boxes_labels[0][:, 1]**2) < 20.0)).sum()) if len(boxes_labels[0]) > 0 else 0
        else:
            nearby_objects = 0
            nearby_ped = 0
            nearby_car = 0

        # Classify
        if actions & RARE_ACTIONS or f in rare_context:
            cat = "A"
        elif "Stop" in actions and nearby_objects < 1:
            cat = "D"
        elif nearby_objects >= 2:
            cat = "B"
        else:
            cat = "C"

        records.append({
            "actions": actions,
            "nearby_objects": nearby_objects,
            "nearby_ped": nearby_ped,
            "nearby_car": nearby_car,
            "category": cat,
            "is_valid": f in valid_set,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curation_masks", required=True)
    parser.add_argument("--metaaction_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.curation_masks) as f:
        masks = json.load(f)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.curation_masks), "curation_detail")
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect clip processing args
    tasks = []
    for clip_key, clip_data in masks["clips"].items():
        date, clip_id = clip_key.split("/")
        meta_path = os.path.join(args.metaaction_root, date, "meta_actions", "final_outputs", f"{clip_id}.txt")
        gt_tar_path = os.path.join(args.data_root, date, f"{clip_id}.gt.tar")
        tasks.append((meta_path, gt_tar_path, clip_data, clip_key))

    print(f"Processing {len(tasks)} clips...")

    all_records = []
    n_workers = min(args.num_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_clip, t[0], t[1], t[2]): t[3]
            for t in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                records = future.result()
                if records:
                    all_records.extend(records)
            except Exception as e:
                print(f"  [ERROR] {futures[future]}: {e}")

    print(f"Total frame records: {len(all_records):,}")

    # ==============================
    # Aggregate statistics
    # ==============================

    # Per action: total frames, kept frames, category breakdown, avg nearby
    action_stats = {}
    for action in ALL_ACTIONS:
        action_stats[action] = {
            "total": 0, "kept": 0, "dropped": 0,
            "cat_counts": Counter(),
            "nearby_kept": [], "nearby_dropped": [],
            "ped_kept": [], "ped_dropped": [],
        }

    # Global stats
    nearby_kept_all = []
    nearby_dropped_all = []

    for rec in all_records:
        for action in rec["actions"]:
            if action not in action_stats:
                continue
            s = action_stats[action]
            s["total"] += 1
            s["cat_counts"][rec["category"]] += 1
            if rec["is_valid"]:
                s["kept"] += 1
                s["nearby_kept"].append(rec["nearby_objects"])
                s["ped_kept"].append(rec["nearby_ped"])
            else:
                s["dropped"] += 1
                s["nearby_dropped"].append(rec["nearby_objects"])
                s["ped_dropped"].append(rec["nearby_ped"])

        if rec["is_valid"]:
            nearby_kept_all.append(rec["nearby_objects"])
        else:
            nearby_dropped_all.append(rec["nearby_objects"])

    # ==============================
    # Generate visualizations
    # ==============================
    print("Generating visualizations...")
    pdf_path = os.path.join(args.output_dir, "curation_detail.pdf")
    png_dir = os.path.join(args.output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    plot_specs = []

    # --- Plot 1: Meta-action frame count before/after ---
    def plot_action_before_after(ax):
        actions = [a for a in ALL_ACTIONS if action_stats[a]["total"] > 0]
        kept = [action_stats[a]["kept"] for a in actions]
        dropped = [action_stats[a]["dropped"] for a in actions]
        y = np.arange(len(actions))

        ax.barh(y, kept, label="Kept", color="#4C72B0")
        ax.barh(y, dropped, left=kept, label="Dropped", color="#DD8452", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(actions, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frame Count")
        ax.set_title("Meta-Action Frame Count: Kept vs Dropped", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")

        for i, a in enumerate(actions):
            total = action_stats[a]["total"]
            pct = action_stats[a]["kept"] / total * 100 if total > 0 else 0
            ax.text(total + total * 0.01, i, f"{pct:.0f}%", va="center", fontsize=7)
    plot_specs.append(("01_action_kept_dropped", (13, 9), plot_action_before_after))

    # --- Plot 2: Per action category breakdown ---
    def plot_action_category_breakdown(ax):
        actions = [a for a in ALL_ACTIONS if action_stats[a]["total"] > 0]
        y = np.arange(len(actions))
        cats = ["A", "B", "C", "D"]
        bottom = np.zeros(len(actions))
        for cat in cats:
            vals = [action_stats[a]["cat_counts"].get(cat, 0) / max(action_stats[a]["total"], 1) * 100
                    for a in actions]
            ax.barh(y, vals, left=bottom, label=f"Cat {cat}", color=CAT_COLORS[cat])
            bottom += np.array(vals)
        ax.set_yticks(y)
        ax.set_yticklabels(actions, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Percentage of Frames (%)")
        ax.set_title("Frame Category Breakdown per Meta-Action", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
    plot_specs.append(("02_action_category_breakdown", (13, 9), plot_action_category_breakdown))

    # --- Plot 3: Nearby objects histogram: kept vs dropped ---
    def plot_nearby_histogram(ax):
        bins = np.arange(0, 20, 1)
        ax.hist(nearby_kept_all, bins=bins, alpha=0.7, label=f"Kept ({len(nearby_kept_all):,})",
                color="#4C72B0", density=True)
        ax.hist(nearby_dropped_all, bins=bins, alpha=0.7, label=f"Dropped ({len(nearby_dropped_all):,})",
                color="#DD8452", density=True)
        ax.set_xlabel("Nearby Objects (<20m)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Nearby Objects: Kept vs Dropped Frames", fontsize=12, fontweight="bold")
        ax.legend()
        kept_mean = np.mean(nearby_kept_all) if nearby_kept_all else 0
        drop_mean = np.mean(nearby_dropped_all) if nearby_dropped_all else 0
        ax.axvline(kept_mean, color="#4C72B0", linestyle="--", linewidth=1.5)
        ax.axvline(drop_mean, color="#DD8452", linestyle="--", linewidth=1.5)
        ax.text(kept_mean + 0.2, ax.get_ylim()[1] * 0.9, f"kept mean={kept_mean:.1f}",
                color="#4C72B0", fontsize=9)
        ax.text(drop_mean + 0.2, ax.get_ylim()[1] * 0.8, f"dropped mean={drop_mean:.1f}",
                color="#DD8452", fontsize=9)
    plot_specs.append(("03_nearby_objects_kept_dropped", (10, 6), plot_nearby_histogram))

    # --- Plot 4: Avg nearby objects per action: kept vs dropped ---
    def plot_nearby_per_action(ax):
        actions = [a for a in ALL_ACTIONS if action_stats[a]["total"] > 0]
        y = np.arange(len(actions))
        w = 0.35

        kept_means = []
        dropped_means = []
        for a in actions:
            s = action_stats[a]
            kept_means.append(np.mean(s["nearby_kept"]) if s["nearby_kept"] else 0)
            dropped_means.append(np.mean(s["nearby_dropped"]) if s["nearby_dropped"] else 0)

        ax.barh(y - w / 2, kept_means, w, label="Kept", color="#4C72B0")
        ax.barh(y + w / 2, dropped_means, w, label="Dropped", color="#DD8452", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(actions, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Avg Nearby Objects (<20m) per Frame")
        ax.set_title("Average Nearby Objects: Kept vs Dropped per Meta-Action", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
    plot_specs.append(("04_avg_nearby_per_action", (13, 9), plot_nearby_per_action))

    # --- Plot 5: Avg nearby pedestrians per action: kept vs dropped ---
    def plot_ped_per_action(ax):
        actions = [a for a in ALL_ACTIONS if action_stats[a]["total"] > 0]
        y = np.arange(len(actions))
        w = 0.35

        kept_means = []
        dropped_means = []
        for a in actions:
            s = action_stats[a]
            kept_means.append(np.mean(s["ped_kept"]) if s["ped_kept"] else 0)
            dropped_means.append(np.mean(s["ped_dropped"]) if s["ped_dropped"] else 0)

        ax.barh(y - w / 2, kept_means, w, label="Kept", color="#4C72B0")
        ax.barh(y + w / 2, dropped_means, w, label="Dropped", color="#DD8452", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(actions, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Avg Nearby Pedestrians (<20m) per Frame")
        ax.set_title("Average Nearby Pedestrians: Kept vs Dropped per Meta-Action", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
    plot_specs.append(("05_avg_ped_per_action", (13, 9), plot_ped_per_action))

    # --- Plot 6: Scatter: kept ratio vs avg nearby objects per action ---
    def plot_kept_vs_nearby_scatter(ax):
        actions = [a for a in ALL_ACTIONS if action_stats[a]["total"] > 0]
        x_vals = []
        y_vals = []
        sizes = []
        colors = []
        for a in actions:
            s = action_stats[a]
            all_nearby = s["nearby_kept"] + s["nearby_dropped"]
            avg_nearby = np.mean(all_nearby) if all_nearby else 0
            kept_ratio = s["kept"] / max(s["total"], 1) * 100
            x_vals.append(avg_nearby)
            y_vals.append(kept_ratio)
            sizes.append(np.sqrt(s["total"]) * 2)
            colors.append("#C44E52" if a in RARE_ACTIONS else "#4C72B0")

        ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidths=0.5)
        for i, a in enumerate(actions):
            ax.annotate(a, (x_vals[i], y_vals[i]), fontsize=6, ha="center", va="bottom")

        ax.set_xlabel("Avg Nearby Objects (<20m)")
        ax.set_ylabel("Kept Ratio (%)")
        ax.set_title("Kept Ratio vs Surrounding Complexity per Meta-Action\n(size=total frames, red=rare action)",
                      fontsize=11, fontweight="bold")
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylim(50, 105)
    plot_specs.append(("06_kept_vs_nearby_scatter", (10, 7), plot_kept_vs_nearby_scatter))

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
