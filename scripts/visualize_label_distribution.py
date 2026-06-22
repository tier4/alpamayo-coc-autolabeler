"""
Visualize the distribution of generated labels in alpamayo_labels.

Produces a multi-page PDF with:
  1. Meta-action distribution (from final_outputs/*.txt) — segment-level counts
  2. Keyframe meta-action distribution (from labels/*.yaml) — per-keyframe
  3. Keyframe meta-action per date (stacked bar)
  4. Navigation text distribution (from labels/*.yaml)
  5. Segment duration distribution per meta-action type (box plot)
  6. Co-occurring meta-action pairs at keyframes (from comma-separated labels)
  7. Scenes / keyframes per date summary

Usage:
    python scripts/visualize_label_distribution.py \
        --labels_root ~/alpamayo_labels \
        --output ~/alpamayo_labels/label_distribution.pdf
"""
import argparse
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml


LONGITUDINAL_ACTIONS = [
    "Stop", "Reverse", "GentleAcceleration", "StrongAcceleration",
    "GentleDeceleration", "StrongDeceleration", "MaintainSpeed",
]
LATERAL_ACTIONS = [
    "GoStraight", "SteerLeft", "SteerRight",
    "SharpSteerLeft", "SharpSteerRight",
    "ReverseLeft", "ReverseRight",
]

CATEGORY_COLORS = {
    "longitudinal": "#4C72B0",
    "lateral": "#DD8452",
    "lane": "#55A868",
}


def categorize_action(action: str) -> str:
    if action in LONGITUDINAL_ACTIONS:
        return "longitudinal"
    if action in LATERAL_ACTIONS:
        return "lateral"
    return "lane"


def parse_final_outputs(labels_root: str) -> list[dict]:
    records = []
    for date in sorted(os.listdir(labels_root)):
        final_dir = os.path.join(labels_root, date, "meta_actions", "final_outputs")
        if not os.path.isdir(final_dir):
            continue
        for fname in os.listdir(final_dir):
            if not fname.endswith(".txt"):
                continue
            scene_id = fname.replace(".txt", "")
            fpath = os.path.join(final_dir, fname)
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(
                        r"(\w+)\s+-\s+Agent:<ego>,\s+Start:(\d+),\s+End:(\d+)", line
                    )
                    if m:
                        records.append({
                            "date": date,
                            "scene": scene_id,
                            "action": m.group(1),
                            "start": int(m.group(2)),
                            "end": int(m.group(3)),
                        })
    return records


def parse_keyframe_labels(labels_root: str) -> list[dict]:
    records = []
    for date in sorted(os.listdir(labels_root)):
        date_dir = os.path.join(labels_root, date)
        if not os.path.isdir(date_dir):
            continue

        # Try unified labels first (labels/*.yaml)
        labels_dir = os.path.join(date_dir, "labels")
        if os.path.isdir(labels_dir):
            for scene_id in sorted(os.listdir(labels_dir)):
                scene_dir = os.path.join(labels_dir, scene_id)
                if not os.path.isdir(scene_dir):
                    continue
                for fname in sorted(os.listdir(scene_dir)):
                    if not fname.endswith(".yaml"):
                        continue
                    with open(os.path.join(scene_dir, fname)) as f:
                        data = yaml.safe_load(f)
                    if data:
                        data["date"] = date
                        data["scene"] = scene_id
                        records.append(data)
            continue

        # Fallback: keyframes/segments_relative_timestamp_sampled.json
        kf_json = os.path.join(date_dir, "keyframes", "segments_relative_timestamp_sampled.json")
        if not os.path.isfile(kf_json):
            continue
        import json
        with open(kf_json) as f:
            kf_data = json.load(f)
        for action_name, segments in kf_data.items():
            for seg in segments:
                records.append({
                    "date": date,
                    "scene": seg.get("clip_id", ""),
                    "meta_action": seg.get("meta_action", action_name),
                    "start_frame": seg.get("event_start_frame", 0),
                })
    return records


def plot_segment_distribution(ax, records: list[dict]) -> None:
    counter = Counter(r["action"] for r in records)
    actions = sorted(counter.keys(), key=lambda a: counter[a], reverse=True)
    counts = [counter[a] for a in actions]
    colors = [CATEGORY_COLORS[categorize_action(a)] for a in actions]

    bars = ax.barh(range(len(actions)), counts, color=colors)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Segment Count")
    ax.set_title("Meta-Action Segment Distribution (from final_outputs)", fontsize=12, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["longitudinal"], label="Longitudinal"),
        Patch(facecolor=CATEGORY_COLORS["lateral"], label="Lateral"),
        Patch(facecolor=CATEGORY_COLORS["lane"], label="Lane"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)


def plot_keyframe_distribution(ax, records: list[dict]) -> None:
    counter = Counter()
    for r in records:
        meta = r.get("meta_action", "")
        for a in meta.split(","):
            a = a.strip()
            if a:
                counter[a] += 1

    actions = sorted(counter.keys(), key=lambda a: counter[a], reverse=True)
    counts = [counter[a] for a in actions]
    colors = [CATEGORY_COLORS[categorize_action(a)] for a in actions]

    bars = ax.barh(range(len(actions)), counts, color=colors)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Keyframe Count")
    ax.set_title("Keyframe Meta-Action Distribution (from labels/*.yaml)", fontsize=12, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)


def plot_keyframe_per_date(ax, records: list[dict]) -> None:
    date_action = defaultdict(Counter)
    all_actions = set()
    for r in records:
        meta = r.get("meta_action", "")
        for a in meta.split(","):
            a = a.strip()
            if a:
                date_action[r["date"]][a] += 1
                all_actions.add(a)

    dates = sorted(date_action.keys())
    total_per_action = Counter()
    for d in dates:
        for a, c in date_action[d].items():
            total_per_action[a] += c
    actions = sorted(all_actions, key=lambda a: total_per_action[a], reverse=True)

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(actions))
    bottom = np.zeros(len(dates))
    for i, action in enumerate(actions):
        vals = [date_action[d].get(action, 0) for d in dates]
        ax.bar(range(len(dates)), vals, bottom=bottom, label=action,
               color=cmap(i), edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Keyframe Count")
    ax.set_title("Keyframe Meta-Action per Date (Stacked)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def plot_navigation_distribution(ax, records: list[dict]) -> None:
    nav_counter = Counter()
    for r in records:
        nav = r.get("navigation_text", "")
        if nav:
            base = re.sub(r"\s+in\s+\d+m$", "", nav)
            nav_counter[base] += 1

    labels = sorted(nav_counter.keys(), key=lambda x: nav_counter[x], reverse=True)
    counts = [nav_counter[l] for l in labels]

    bars = ax.barh(range(len(labels)), counts, color="#7A76C2")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Keyframe Count")
    ax.set_title("Navigation Text Distribution (base command, distance removed)", fontsize=12, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)


def plot_segment_duration(ax, records: list[dict], delta_t: float = 0.1) -> None:
    duration_by_action = defaultdict(list)
    for r in records:
        dur = (r["end"] - r["start"]) * delta_t
        duration_by_action[r["action"]].append(dur)

    actions = sorted(duration_by_action.keys(),
                     key=lambda a: np.median(duration_by_action[a]), reverse=True)
    data = [duration_by_action[a] for a in actions]
    colors = [CATEGORY_COLORS[categorize_action(a)] for a in actions]

    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel("Duration (seconds)")
    ax.set_title("Segment Duration Distribution per Meta-Action (outliers hidden)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()


def plot_cooccurrence(ax, records: list[dict]) -> None:
    pair_counter = Counter()
    for r in records:
        meta = r.get("meta_action", "")
        parts = [a.strip() for a in meta.split(",") if a.strip()]
        if len(parts) >= 2:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    pair = tuple(sorted([parts[i], parts[j]]))
                    pair_counter[pair] += 1

    if not pair_counter:
        ax.text(0.5, 0.5, "No co-occurring pairs found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Co-occurring Meta-Action Pairs at Keyframes", fontsize=12, fontweight="bold")
        return

    top_pairs = pair_counter.most_common(20)
    labels = [f"{a} + {b}" for (a, b), _ in top_pairs]
    counts = [c for _, c in top_pairs]

    bars = ax.barh(range(len(labels)), counts, color="#E07B54")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top Co-occurring Meta-Action Pairs at Keyframes", fontsize=12, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)


def plot_date_summary(ax, seg_records: list[dict], kf_records: list[dict]) -> None:
    date_scenes_seg = defaultdict(set)
    date_scenes_kf = defaultdict(set)
    date_kf_count = Counter()

    for r in seg_records:
        date_scenes_seg[r["date"]].add(r["scene"])
    for r in kf_records:
        date_scenes_kf[r["date"]].add(r["scene"])
        date_kf_count[r["date"]] += 1

    dates = sorted(set(list(date_scenes_seg.keys()) + list(date_scenes_kf.keys())))
    scenes_counts = [len(date_scenes_kf.get(d, set())) for d in dates]
    kf_counts = [date_kf_count.get(d, 0) for d in dates]

    x = np.arange(len(dates))
    w = 0.35
    bars1 = ax.bar(x - w / 2, scenes_counts, w, label="Scenes", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, kf_counts, w, label="Keyframes", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Scenes / Keyframes per Date", fontsize=12, fontweight="bold")
    ax.legend()

    for bar, count in zip(bars1, scenes_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha="center", va="bottom", fontsize=7)
    for bar, count in zip(bars2, kf_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha="center", va="bottom", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize label distribution.")
    parser.add_argument("--labels_root", required=True, help="Root directory of alpamayo_labels")
    parser.add_argument("--output", default=None, help="Output PDF path (default: <labels_root>/label_distribution.pdf)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.labels_root, "label_distribution.pdf")

    print("Parsing final_outputs (segments)...")
    seg_records = parse_final_outputs(args.labels_root)
    print(f"  {len(seg_records)} segments found")

    print("Parsing keyframe labels...")
    kf_records = parse_keyframe_labels(args.labels_root)
    print(f"  {len(kf_records)} keyframes found")

    print(f"Generating plots -> {args.output}")

    plot_specs = [
        ("01_segment_distribution", (12, 7), plot_segment_distribution, [seg_records]),
        ("02_keyframe_distribution", (12, 8), plot_keyframe_distribution, [kf_records]),
        ("03_keyframe_per_date", (14, 8), plot_keyframe_per_date, [kf_records]),
        ("04_navigation_distribution", (10, 4), plot_navigation_distribution, [kf_records]),
        ("05_segment_duration", (12, 7), plot_segment_duration, [seg_records]),
        ("06_cooccurrence", (12, 7), plot_cooccurrence, [kf_records]),
        ("07_date_summary", (12, 6), plot_date_summary, [seg_records, kf_records]),
    ]

    png_dir = os.path.splitext(args.output)[0] + "_png"
    os.makedirs(png_dir, exist_ok=True)

    with PdfPages(args.output) as pdf:
        for name, figsize, plot_fn, plot_args in plot_specs:
            fig, ax = plt.subplots(figsize=figsize)
            plot_fn(ax, *plot_args)
            fig.tight_layout()
            pdf.savefig(fig)
            png_path = os.path.join(png_dir, f"{name}.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Done!")
    print(f"  PDF: {args.output}")
    print(f"  PNGs: {png_dir}/")


if __name__ == "__main__":
    main()
