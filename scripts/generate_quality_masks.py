"""
Generate per-frame quality masks based on frame_tags.npz and quality_ledger.json.

Excludes frames with driving quality issues (route deviation, localization jumps,
unjustified stops, etc.) and clips flagged as suspect route errors.

Outputs quality_masks.json, independent of curation_masks.json and camera_masks.json.
Combine via set intersection at training time.

Usage:
    python scripts/generate_quality_masks.py \
        --frame_tags /mnt/storage_rdma/datasets/tier4/e2e/frame_tags.npz \
        --output ~/alpamayo_labels_metaaction/train/quality_masks.json

    python scripts/generate_quality_masks.py \
        --frame_tags /mnt/storage_rdma/datasets/tier4/e2e-val/frame_tags.npz \
        --output ~/alpamayo_labels_metaaction/val/quality_masks.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

FLAG_BITS = {
    "PRE_ROUTE": 1,
    "POST_GOAL": 2,
    "OFF_ROUTE": 4,
    "LOC_JUMP": 8,
    "STATIONARY": 16,
    "IN_JUNCTION": 32,
    "UNJUSTIFIED_STOP": 64,
}

DEFAULT_EXCLUDE = ["OFF_ROUTE", "LOC_JUMP", "PRE_ROUTE", "POST_GOAL", "UNJUSTIFIED_STOP"]


def load_suspect_clips(ledger_path: str) -> set:
    if not ledger_path or not os.path.exists(ledger_path):
        return set()
    with open(ledger_path) as f:
        data = json.load(f)
    suspect = set()
    for scene in data.get("scenes", []):
        if scene.get("class") == "suspect_route_error":
            key = scene["scene"].replace(".tar", "")
            suspect.add(key)
    return suspect


def main():
    parser = argparse.ArgumentParser(
        description="Generate quality masks from frame_tags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frame_tags", required=True,
                        help="Path to frame_tags.npz")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--exclude", nargs="+", default=DEFAULT_EXCLUDE,
                        help="frame_tags flags to exclude")
    parser.add_argument("--quality_ledger", default=None,
                        help="Path to quality_ledger.json (optional)")
    parser.add_argument("--exclude_suspect_route", action="store_true", default=True,
                        help="Exclude suspect_route_error clips from quality_ledger")
    parser.add_argument("--no_exclude_suspect_route", dest="exclude_suspect_route",
                        action="store_false")
    args = parser.parse_args()

    # Validate exclude flags
    for flag in args.exclude:
        if flag not in FLAG_BITS:
            parser.error(f"Unknown flag: {flag}. Valid: {list(FLAG_BITS.keys())}")

    exclude_mask = 0
    for flag in args.exclude:
        exclude_mask |= FLAG_BITS[flag]

    print(f"Excluding flags: {args.exclude} (bitmask={exclude_mask})")

    # Auto-detect quality_ledger if not specified
    if args.quality_ledger is None:
        tags_dir = os.path.dirname(args.frame_tags)
        candidate = os.path.join(tags_dir, "quality_ledger.json")
        if os.path.exists(candidate):
            args.quality_ledger = candidate
            print(f"Auto-detected quality_ledger: {candidate}")

    suspect_clips = set()
    if args.exclude_suspect_route and args.quality_ledger:
        suspect_clips = load_suspect_clips(args.quality_ledger)
        print(f"suspect_route_error clips: {len(suspect_clips)}")

    # Process frame_tags
    print(f"Loading {args.frame_tags}...")
    tags_data = np.load(args.frame_tags, allow_pickle=True)

    clips = {}
    total_frames = 0
    total_excluded = 0
    total_valid = 0
    flag_frame_counts = Counter()
    suspect_excluded_frames = 0
    suspect_excluded_clips = 0

    for clip_key in sorted(tags_data.keys()):
        tags = tags_data[clip_key]
        n_frames = len(tags)
        total_frames += n_frames

        # Per-flag excluded frame lists
        excluded_per_flag = {}
        for flag in args.exclude:
            bit = FLAG_BITS[flag]
            flagged = np.where((tags & bit) > 0)[0].tolist()
            excluded_per_flag[flag] = flagged
            flag_frame_counts[flag] += len(flagged)

        # Combine all excluded frames
        all_excluded = set()
        for frames in excluded_per_flag.values():
            all_excluded.update(frames)

        # Suspect route error: exclude entire clip
        is_suspect = clip_key in suspect_clips
        if is_suspect:
            all_excluded = set(range(n_frames))
            suspect_excluded_frames += n_frames
            suspect_excluded_clips += 1

        valid_frames = sorted(set(range(n_frames)) - all_excluded)
        total_excluded += len(all_excluded)
        total_valid += len(valid_frames)

        clip_result = {
            "total_frames": n_frames,
            "valid_frames": valid_frames,
            "excluded_count": len(all_excluded),
            "excluded_frames_per_flag": excluded_per_flag,
        }
        if is_suspect:
            clip_result["suspect_route_error"] = True

        clips[clip_key] = clip_result

    output = {
        "metadata": {
            "version": 1,
            "source_frame_tags": args.frame_tags,
            "source_quality_ledger": args.quality_ledger,
            "excluded_flags": args.exclude,
            "exclude_suspect_route": args.exclude_suspect_route,
            "total_clips": len(clips),
            "total_frames": total_frames,
            "total_excluded_frames": total_excluded,
            "total_valid_frames": total_valid,
            "suspect_route_error_clips": suspect_excluded_clips,
            "suspect_route_error_frames": suspect_excluded_frames,
            "per_flag_excluded_frames": dict(flag_frame_counts),
        },
        "clips": clips,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"\n=== Summary ===")
    print(f"  Total clips:           {len(clips)}")
    print(f"  Total frames:          {total_frames:,}")
    print(f"  Excluded frames:       {total_excluded:,} ({total_excluded/total_frames*100:.2f}%)")
    print(f"  Valid frames:          {total_valid:,} ({total_valid/total_frames*100:.2f}%)")
    print(f"\n  Per-flag breakdown:")
    for flag in args.exclude:
        c = flag_frame_counts[flag]
        print(f"    {flag:<20} {c:>8,} frames ({c/total_frames*100:.2f}%)")
    if suspect_excluded_clips > 0:
        print(f"    {'SUSPECT_ROUTE':<20} {suspect_excluded_frames:>8,} frames "
              f"({suspect_excluded_clips} clips, all frames excluded)")


if __name__ == "__main__":
    main()
