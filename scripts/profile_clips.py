"""
Clip-level profiling and data curation for E2E autonomous driving training.

Reads meta-action final_outputs and produces:
  1. Per-clip 21-dim profile (frame occupancy ratio per action type)
  2. Clip clustering and redundancy analysis
  3. Rare-action flagging
  4. Curated clip list with sampling weights

Usage:
    python scripts/profile_clips.py \
        --labels_root ~/alpamayo_labels_metaaction/train \
        --output_dir ~/alpamayo_labels_metaaction/train/curation
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

ALL_ACTIONS = [
    # Longitudinal
    "Stop", "Reverse", "GentleAcceleration", "StrongAcceleration",
    "GentleDeceleration", "StrongDeceleration", "MaintainSpeed",
    # Lateral
    "GoStraight", "SteerLeft", "SteerRight",
    "SharpSteerLeft", "SharpSteerRight",
    "ReverseLeft", "ReverseRight",
    # Lane
    "LaneKeep", "LeftLaneChange", "RightLaneChange",
    "SlightlyShiftLeft", "SlightlyShiftRight",
    "TurnLeft", "TurnRight",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}

RARE_ACTIONS = {
    "StrongAcceleration", "StrongDeceleration",
    "LeftLaneChange", "RightLaneChange",
    "TurnLeft", "TurnRight",
    "SharpSteerLeft", "SharpSteerRight",
}

REDUNDANT_PATTERN_ACTIONS = {"Stop", "LaneKeep", "GoStraight", "MaintainSpeed"}


def parse_clip(filepath: str) -> dict:
    segments = []
    max_frame = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(\w+)\s+-\s+Agent:<ego>,\s+Start:(\d+),\s+End:(\d+)", line)
            if m:
                action = m.group(1)
                start = int(m.group(2))
                end = int(m.group(3))
                segments.append((action, start, end))
                max_frame = max(max_frame, end)
    return {"segments": segments, "max_frame": max_frame}


def compute_profile(segments: list, max_frame: int) -> np.ndarray:
    if max_frame <= 0:
        return np.zeros(len(ALL_ACTIONS))
    counts = np.zeros(len(ALL_ACTIONS))
    for action, start, end in segments:
        idx = ACTION_TO_IDX.get(action)
        if idx is not None:
            counts[idx] += (end - start)
    # Each axis (longitudinal/lateral/lane) covers the full clip independently,
    # so normalize within each axis group.
    longi_slice = slice(0, 7)
    lateral_slice = slice(7, 14)
    lane_slice = slice(14, 21)
    profile = np.zeros(len(ALL_ACTIONS))
    for s in [longi_slice, lateral_slice, lane_slice]:
        total = counts[s].sum()
        if total > 0:
            profile[s] = counts[s] / total
    return profile


def profile_all_clips(labels_root: str) -> list[dict]:
    clips = []
    for date in sorted(os.listdir(labels_root)):
        final_dir = os.path.join(labels_root, date, "meta_actions", "final_outputs")
        if not os.path.isdir(final_dir):
            continue
        for fname in sorted(os.listdir(final_dir)):
            if not fname.endswith(".txt"):
                continue
            clip_id = fname.replace(".txt", "")
            fpath = os.path.join(final_dir, fname)
            parsed = parse_clip(fpath)
            profile = compute_profile(parsed["segments"], parsed["max_frame"])

            actions_present = {seg[0] for seg in parsed["segments"]}
            rare_flags = actions_present & RARE_ACTIONS
            redundant_ratio = sum(
                profile[ACTION_TO_IDX[a]] for a in REDUNDANT_PATTERN_ACTIONS if a in ACTION_TO_IDX
            ) / 3.0  # average across 3 axes

            clips.append({
                "date": date,
                "clip_id": clip_id,
                "profile": profile,
                "max_frame": parsed["max_frame"],
                "actions_present": actions_present,
                "rare_actions": rare_flags,
                "has_rare": len(rare_flags) > 0,
                "redundant_ratio": redundant_ratio,
            })
    return clips


def assign_sampling_weights(clips: list, redundant_threshold: float = 0.7, downsample_rate: float = 0.3) -> list:
    for clip in clips:
        if clip["has_rare"]:
            clip["weight"] = 2.0
        elif clip["redundant_ratio"] > redundant_threshold:
            clip["weight"] = downsample_rate
        else:
            clip["weight"] = 1.0
    return clips


def plot_action_occupancy_heatmap(ax, clips: list, title: str) -> None:
    profiles = np.array([c["profile"] for c in clips])
    # Sort by redundant_ratio descending
    sort_idx = np.argsort([-c["redundant_ratio"] for c in clips])
    profiles = profiles[sort_idx]

    im = ax.imshow(profiles.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(ALL_ACTIONS)))
    ax.set_yticklabels(ALL_ACTIONS, fontsize=7)
    ax.set_xlabel("Clips (sorted by redundancy)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Frame Occupancy Ratio")

    # Draw axis group separators
    for y in [6.5, 13.5]:
        ax.axhline(y, color="white", linewidth=2)


def plot_redundancy_distribution(ax, clips: list) -> None:
    ratios = [c["redundant_ratio"] for c in clips]
    rare_mask = [c["has_rare"] for c in clips]
    normal = [r for r, m in zip(ratios, rare_mask) if not m]
    rare = [r for r, m in zip(ratios, rare_mask) if m]

    ax.hist(normal, bins=30, alpha=0.7, label=f"Normal ({len(normal)})", color="#4C72B0")
    ax.hist(rare, bins=30, alpha=0.7, label=f"Has rare action ({len(rare)})", color="#DD8452")
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1.5, label="Redundancy threshold")
    ax.set_xlabel("Redundant Action Ratio (Stop+LaneKeep+GoStraight+MaintainSpeed)")
    ax.set_ylabel("Clip Count")
    ax.set_title("Clip Redundancy Distribution", fontsize=11, fontweight="bold")
    ax.legend()


def plot_weight_summary(ax, clips: list) -> None:
    weight_counts = Counter()
    for c in clips:
        if c["weight"] == 2.0:
            weight_counts["Upweight (rare, 2.0x)"] += 1
        elif c["weight"] < 1.0:
            weight_counts[f"Downsample (redundant, {c['weight']:.1f}x)"] += 1
        else:
            weight_counts["Normal (1.0x)"] += 1

    labels = list(weight_counts.keys())
    counts = [weight_counts[l] for l in labels]
    colors = ["#DD8452", "#AAAAAA", "#4C72B0"][:len(labels)]

    bars = ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Clip Count")
    ax.set_title("Sampling Weight Assignment", fontsize=11, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)

    total = sum(counts)
    effective = sum(c["weight"] for c in clips)
    ax.text(0.95, 0.05, f"Total: {total} clips\nEffective: {effective:.0f} clips",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def plot_rare_action_counts(ax, clips: list) -> None:
    rare_counter = Counter()
    for c in clips:
        for ra in c["rare_actions"]:
            rare_counter[ra] += 1
    if not rare_counter:
        ax.text(0.5, 0.5, "No rare actions found", ha="center", va="center", transform=ax.transAxes)
        return

    actions = sorted(rare_counter.keys(), key=lambda a: rare_counter[a], reverse=True)
    counts = [rare_counter[a] for a in actions]

    bars = ax.barh(range(len(actions)), counts, color="#55A868")
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Clips Containing Action")
    ax.set_title("Rare Action Coverage (clips containing each rare action)", fontsize=11, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)


def plot_effective_distribution(ax, clips: list) -> None:
    """Compare original vs weighted meta-action distribution."""
    original = np.zeros(len(ALL_ACTIONS))
    weighted = np.zeros(len(ALL_ACTIONS))
    for c in clips:
        original += c["profile"]
        weighted += c["profile"] * c["weight"]

    x = np.arange(len(ALL_ACTIONS))
    w = 0.35
    ax.barh(x - w / 2, original, w, label="Original", color="#4C72B0", alpha=0.7)
    ax.barh(x + w / 2, weighted, w, label="Weighted", color="#DD8452", alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(ALL_ACTIONS, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of Occupancy Ratios across Clips")
    ax.set_title("Original vs Weighted Action Distribution", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    for y in [6.5, 13.5]:
        ax.axhline(y, color="gray", linewidth=0.5, linestyle="--")


def plot_cluster_dendrogram(ax, clips: list, max_display: int = 100) -> None:
    profiles = np.array([c["profile"] for c in clips])
    if len(profiles) > max_display:
        idx = np.random.RandomState(42).choice(len(profiles), max_display, replace=False)
        profiles = profiles[idx]

    if len(profiles) < 2:
        ax.text(0.5, 0.5, "Not enough clips", ha="center", va="center", transform=ax.transAxes)
        return

    dist = pdist(profiles, metric="cosine")
    Z = linkage(dist, method="ward")
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.5)
    ax.set_title(f"Clip Similarity Dendrogram (sample of {len(profiles)})", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance (cosine)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and curate clips for E2E training.")
    parser.add_argument("--labels_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--redundant_threshold", type=float, default=0.7)
    parser.add_argument("--downsample_rate", type=float, default=0.3)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.labels_root, "curation")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Profiling clips...")
    clips = profile_all_clips(args.labels_root)
    print(f"  {len(clips)} clips profiled")

    print("Assigning sampling weights...")
    clips = assign_sampling_weights(clips, args.redundant_threshold, args.downsample_rate)

    # Save clip list with weights
    clip_list = []
    for c in clips:
        clip_list.append({
            "date": c["date"],
            "clip_id": c["clip_id"],
            "weight": c["weight"],
            "has_rare": c["has_rare"],
            "rare_actions": sorted(c["rare_actions"]),
            "redundant_ratio": round(c["redundant_ratio"], 3),
            "max_frame": c["max_frame"],
        })
    json_path = os.path.join(args.output_dir, "clip_weights.json")
    with open(json_path, "w") as f:
        json.dump(clip_list, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary stats
    n_rare = sum(1 for c in clips if c["has_rare"])
    n_redundant = sum(1 for c in clips if c["weight"] < 1.0)
    n_normal = len(clips) - n_rare - n_redundant
    effective = sum(c["weight"] for c in clips)
    print(f"\n  Summary:")
    print(f"    Total clips:      {len(clips)}")
    print(f"    Rare-action:      {n_rare} (weight 2.0x)")
    print(f"    Redundant:        {n_redundant} (weight {args.downsample_rate}x)")
    print(f"    Normal:           {n_normal} (weight 1.0x)")
    print(f"    Effective clips:  {effective:.0f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    pdf_path = os.path.join(args.output_dir, "clip_curation.pdf")
    png_dir = os.path.join(args.output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    plot_specs = [
        ("01_occupancy_heatmap", (16, 8),
         lambda ax: plot_action_occupancy_heatmap(ax, clips, "Per-Clip Action Occupancy (sorted by redundancy)")),
        ("02_redundancy_distribution", (10, 5),
         lambda ax: plot_redundancy_distribution(ax, clips)),
        ("03_weight_summary", (10, 4),
         lambda ax: plot_weight_summary(ax, clips)),
        ("04_rare_action_counts", (10, 5),
         lambda ax: plot_rare_action_counts(ax, clips)),
        ("05_effective_distribution", (12, 8),
         lambda ax: plot_effective_distribution(ax, clips)),
        ("06_cluster_dendrogram", (14, 6),
         lambda ax: plot_cluster_dendrogram(ax, clips)),
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
