"""
Generate per-frame curation masks for E2E autonomous driving training.

Classifies each frame into 4 categories based on ego meta-action rarity
and surrounding object context, then generates valid frame indices with
category-specific sampling rates.

Categories:
  A - High value:  rare action or its context window  (100% kept)
  B - Standard:    normal action + nearby objects >= 2 (100% kept)
  C - Low density: normal action + nearby objects < 2  (20% kept)
  D - Empty stop:  Stop + nearby objects < 1           (10% kept)

Usage:
    python scripts/generate_curation_masks.py \
        --metaaction_root ~/alpamayo_labels_metaaction/train \
        --data_root /mnt/storage_rdma/datasets/tier4/e2e \
        --output ~/alpamayo_labels_metaaction/train/curation_masks.json \
        --visualize
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from tqdm import tqdm

from curation_utils import (
    RARE_ACTIONS,
    build_frame_action_map,
    build_rare_context_set,
    collect_clip_pairs,
    compute_frame_context,
    load_gt_tar,
    parse_metaactions,
)

DEFAULT_CONTEXT_FRAMES = 20
DEFAULT_NEARBY_THRESHOLD_B = 2.0
DEFAULT_NEARBY_THRESHOLD_D = 1.0


def classify_frame(
    active_actions: set,
    nearby_objects: int,
    is_in_rare_context: bool,
    nearby_threshold_b: float,
    nearby_threshold_d: float,
) -> str:
    if active_actions & RARE_ACTIONS or is_in_rare_context:
        return "A"
    if "Stop" in active_actions and nearby_objects < nearby_threshold_d:
        return "D"
    if nearby_objects >= nearby_threshold_b:
        return "B"
    return "C"


def is_frame_valid(category: str, frame_idx: int) -> bool:
    if category in ("A", "B"):
        return True
    if category == "C":
        return frame_idx % 5 == 0
    if category == "D":
        return frame_idx % 10 == 0
    return True


def process_clip(
    meta_path: str,
    gt_tar_path: str,
    context_frames: int,
    nearby_threshold_b: float,
    nearby_threshold_d: float,
) -> Optional[dict]:
    segments = parse_metaactions(meta_path)
    if not segments:
        return None

    if not os.path.exists(gt_tar_path):
        return None

    gt_frames = load_gt_tar(gt_tar_path)
    if not gt_frames:
        return None

    n_frames = max(gt_frames.keys()) + 1
    frame_action_map = build_frame_action_map(segments, n_frames)
    rare_context = build_rare_context_set(segments, context_frames, n_frames)

    rare_actions_in_clip = set()
    for seg in segments:
        if seg["action"] in RARE_ACTIONS:
            rare_actions_in_clip.add(seg["action"])

    category_counts = Counter()
    valid_frames = []
    total_nearby = 0
    n_ctx_frames = 0

    for frame_idx in range(n_frames):
        active_actions = frame_action_map.get(frame_idx, set())
        boxes_labels = gt_frames.get(frame_idx)
        if boxes_labels is not None:
            ctx = compute_frame_context(boxes_labels[0], boxes_labels[1])
        else:
            ctx = {"nearby_objects": 0, "nearby_pedestrians": 0}

        nearby = ctx["nearby_objects"]
        total_nearby += nearby
        n_ctx_frames += 1

        category = classify_frame(
            active_actions, nearby,
            frame_idx in rare_context,
            nearby_threshold_b, nearby_threshold_d,
        )
        category_counts[category] += 1

        if is_frame_valid(category, frame_idx):
            valid_frames.append(frame_idx)

    avg_nearby = total_nearby / max(n_ctx_frames, 1)

    return {
        "total_frames": n_frames,
        "valid_frames": valid_frames,
        "category_counts": dict(category_counts),
        "rare_actions": sorted(rare_actions_in_clip),
        "avg_nearby_objects": round(avg_nearby, 2),
    }


def generate_visualizations(result: dict, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    png_dir = os.path.join(output_dir, "curation_masks_png")
    os.makedirs(png_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "curation_masks.pdf")

    clips = result["clips"]
    meta = result["metadata"]

    total_cats = Counter()
    per_date_cats = {}
    valid_ratios = []
    for clip_key, clip_data in clips.items():
        date = clip_key.split("/")[0]
        if date not in per_date_cats:
            per_date_cats[date] = Counter()
        for cat, count in clip_data["category_counts"].items():
            total_cats[cat] += count
            per_date_cats[date][cat] += count
        ratio = len(clip_data["valid_frames"]) / max(clip_data["total_frames"], 1)
        valid_ratios.append(ratio)

    plot_specs = []

    def plot_pie(ax):
        cats = ["A", "B", "C", "D"]
        labels_map = {
            "A": f"A: High value ({total_cats.get('A', 0):,})",
            "B": f"B: Standard ({total_cats.get('B', 0):,})",
            "C": f"C: Low density ({total_cats.get('C', 0):,})",
            "D": f"D: Empty stop ({total_cats.get('D', 0):,})",
        }
        sizes = [total_cats.get(c, 0) for c in cats]
        colors = ["#C44E52", "#4C72B0", "#AAAAAA", "#DDDDDD"]
        labels = [labels_map[c] for c in cats]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Frame Category Distribution (all frames)", fontsize=12, fontweight="bold")
        total = sum(sizes)
        valid = meta["total_valid_frames"]
        ax.text(0, -1.3, f"Total: {total:,} frames -> Valid: {valid:,} frames ({valid/total*100:.1f}%)",
                ha="center", fontsize=10)
    plot_specs.append(("01_category_pie", (8, 8), plot_pie))

    def plot_per_date(ax):
        dates = sorted(per_date_cats.keys())
        cats = ["A", "B", "C", "D"]
        colors = ["#C44E52", "#4C72B0", "#AAAAAA", "#DDDDDD"]
        bottom = np.zeros(len(dates))
        for cat, color in zip(cats, colors):
            vals = [per_date_cats[d].get(cat, 0) for d in dates]
            ax.bar(range(len(dates)), vals, bottom=bottom, label=cat, color=color)
            bottom += np.array(vals)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Frame Count")
        ax.set_title("Frame Categories per Date", fontsize=12, fontweight="bold")
        ax.legend()
    plot_specs.append(("02_per_date_categories", (12, 6), plot_per_date))

    def plot_valid_ratio(ax):
        ax.hist(valid_ratios, bins=30, color="#4C72B0", edgecolor="white")
        ax.axvline(np.mean(valid_ratios), color="red", linestyle="--",
                   label=f"Mean: {np.mean(valid_ratios):.2f}")
        ax.set_xlabel("Valid Frame Ratio per Clip")
        ax.set_ylabel("Clip Count")
        ax.set_title("Distribution of Valid Frame Ratio across Clips", fontsize=12, fontweight="bold")
        ax.legend()
    plot_specs.append(("03_valid_ratio_histogram", (10, 5), plot_valid_ratio))

    def plot_before_after(ax):
        cats = ["A", "B", "C", "D"]
        cat_labels = ["A: High value", "B: Standard", "C: Low density", "D: Empty stop"]
        before = [total_cats.get(c, 0) for c in cats]
        rates = [1.0, 1.0, meta["sampling_rates"]["C"], meta["sampling_rates"]["D"]]
        after = [b * r for b, r in zip(before, rates)]
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w / 2, before, w, label="Before (all frames)", color="#4C72B0")
        ax.bar(x + w / 2, after, w, label="After (valid frames)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.set_ylabel("Frame Count")
        ax.set_title("Frame Count Before vs After Curation", fontsize=12, fontweight="bold")
        ax.legend()
        for i, (b, a) in enumerate(zip(before, after)):
            ax.text(i - w / 2, b, f"{b:,}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w / 2, a, f"{int(a):,}", ha="center", va="bottom", fontsize=7)
    plot_specs.append(("04_before_after", (10, 6), plot_before_after))

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-frame curation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metaaction_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--context_frames", type=int, default=DEFAULT_CONTEXT_FRAMES)
    parser.add_argument("--nearby_threshold_b", type=float, default=DEFAULT_NEARBY_THRESHOLD_B)
    parser.add_argument("--nearby_threshold_d", type=float, default=DEFAULT_NEARBY_THRESHOLD_D)
    parser.add_argument("--sampling_rate_c", type=float, default=0.2)
    parser.add_argument("--sampling_rate_d", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.metaaction_root, "curation_masks.json")

    clip_pairs = collect_clip_pairs(args.metaaction_root, args.data_root)
    print(f"Found {len(clip_pairs)} clips")

    print("Processing clips...")
    clips_result = {}
    total_frames = 0
    total_valid = 0
    n_workers = min(args.num_workers, len(clip_pairs))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_clip, meta_path, gt_path,
                args.context_frames, args.nearby_threshold_b, args.nearby_threshold_d,
            ): clip_key
            for meta_path, gt_path, clip_key in clip_pairs
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            clip_key = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  [ERROR] {clip_key}: {e}")
                continue
            if result is None:
                continue
            clips_result[clip_key] = result
            total_frames += result["total_frames"]
            total_valid += len(result["valid_frames"])

    output = {
        "metadata": {
            "version": 1,
            "sampling_rates": {
                "A": 1.0, "B": 1.0,
                "C": args.sampling_rate_c, "D": args.sampling_rate_d,
            },
            "context_frames": args.context_frames,
            "nearby_threshold_b": args.nearby_threshold_b,
            "nearby_threshold_d": args.nearby_threshold_d,
            "total_clips": len(clips_result),
            "total_frames": total_frames,
            "total_valid_frames": total_valid,
        },
        "clips": clips_result,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")

    reduction = (1 - total_valid / max(total_frames, 1)) * 100
    print(f"\n=== Summary ===")
    print(f"  Clips:        {len(clips_result)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Valid frames: {total_valid:,}")
    print(f"  Reduction:    {reduction:.1f}%")

    cat_totals = Counter()
    for clip_data in clips_result.values():
        for cat, count in clip_data["category_counts"].items():
            cat_totals[cat] += count
    for cat in ["A", "B", "C", "D"]:
        c = cat_totals.get(cat, 0)
        pct = c / max(total_frames, 1) * 100
        print(f"  Category {cat}: {c:>8,} frames ({pct:5.1f}%)")

    if args.visualize:
        print("\nGenerating visualizations...")
        generate_visualizations(output, os.path.dirname(args.output))

    print("Done!")


if __name__ == "__main__":
    main()
