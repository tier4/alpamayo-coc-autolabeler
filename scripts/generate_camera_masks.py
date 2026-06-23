"""
Generate per-clip camera availability masks.

Detects frames where the front camera image is missing or black (camera drop).
Black frames are identified by abnormally small JPEG file size compared to
the clip's median, using .tar.idx for fast scanning without image decoding.

Usage:
    python scripts/generate_camera_masks.py \
        --data_root /mnt/storage_rdma/datasets/tier4/e2e \
        --output ~/alpamayo_labels_metaaction/train/camera_masks.json
"""
import argparse
import glob
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

BLACK_SIZE_RATIO_THRESHOLD = 0.15  # frames < 15% of median size are black


def scan_clip(tar_path: str) -> dict:
    idx_path = tar_path + ".idx"

    # Collect front camera frame indices and sizes
    front_frames = []  # (frame_idx, file_size)

    if os.path.exists(idx_path):
        with open(idx_path) as f:
            entries = json.load(f)
        for entry in entries:
            name = entry["name"]
            if name.endswith("_CAM_FRONT.jpg"):
                fname = name.split("/")[-1]
                idx = int(fname.split("_")[0])
                front_frames.append((idx, entry["size"]))
    else:
        import tarfile
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("_CAM_FRONT.jpg"):
                    fname = member.name.split("/")[-1]
                    idx = int(fname.split("_")[0])
                    front_frames.append((idx, member.size))

    if not front_frames:
        return {
            "total_expected": 0,
            "total_present": 0,
            "missing_count": 0,
            "missing_frames": [],
            "black_count": 0,
            "black_frames": [],
            "valid_frames": [],
        }

    front_frames.sort()
    frame_indices = [f[0] for f in front_frames]
    sizes = np.array([f[1] for f in front_frames])

    # Detect missing frames (gaps in index sequence)
    expected = set(range(frame_indices[0], frame_indices[-1] + 1))
    present = set(frame_indices)
    missing = sorted(expected - present)

    # Detect black frames by file size
    median_size = np.median(sizes)
    threshold = median_size * BLACK_SIZE_RATIO_THRESHOLD
    black_mask = sizes < threshold
    black_frames = [frame_indices[i] for i in range(len(frame_indices)) if black_mask[i]]

    # Valid = present and not black
    invalid = set(missing) | set(black_frames)
    valid_frames = sorted(set(frame_indices) - set(black_frames))

    return {
        "total_expected": len(expected),
        "total_present": len(present),
        "missing_count": len(missing),
        "missing_frames": missing,
        "black_count": len(black_frames),
        "black_frames": black_frames,
        "black_threshold_bytes": int(threshold),
        "median_size_bytes": int(median_size),
        "valid_frames": valid_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate camera availability masks.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    tar_files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.tar"), recursive=True))
    tar_files = [t for t in tar_files if not t.endswith(".gt.tar")]
    print(f"Found {len(tar_files)} clip tar files")

    clips = {}
    total_missing_clips = 0
    total_black_clips = 0
    total_missing_frames = 0
    total_black_frames = 0
    total_frames = 0
    total_valid = 0

    n_workers = min(args.num_workers, len(tar_files))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for tar_path in tar_files:
            date = os.path.basename(os.path.dirname(tar_path))
            clip_id = os.path.basename(tar_path).replace(".tar", "")
            clip_key = f"{date}/{clip_id}"
            futures[executor.submit(scan_clip, tar_path)] = clip_key

        for future in tqdm(as_completed(futures), total=len(futures)):
            clip_key = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  [ERROR] {clip_key}: {e}")
                continue

            clips[clip_key] = result
            total_frames += result["total_expected"]
            total_valid += len(result["valid_frames"])
            if result["missing_count"] > 0:
                total_missing_clips += 1
                total_missing_frames += result["missing_count"]
            if result["black_count"] > 0:
                total_black_clips += 1
                total_black_frames += result["black_count"]

    output = {
        "metadata": {
            "total_clips": len(clips),
            "clips_with_missing_frames": total_missing_clips,
            "clips_with_black_frames": total_black_clips,
            "total_expected_frames": total_frames,
            "total_missing_frames": total_missing_frames,
            "total_black_frames": total_black_frames,
            "total_valid_frames": total_valid,
            "black_size_ratio_threshold": BLACK_SIZE_RATIO_THRESHOLD,
        },
        "clips": clips,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"\n=== Summary ===")
    print(f"  Total clips:               {len(clips)}")
    print(f"  Total expected frames:      {total_frames:,}")
    print(f"  Valid frames:               {total_valid:,}")
    print(f"  Missing frames (gaps):      {total_missing_frames:,} ({total_missing_clips} clips)")
    print(f"  Black frames (camera drop): {total_black_frames:,} ({total_black_clips} clips)")
    if total_frames > 0:
        invalid = total_missing_frames + total_black_frames
        print(f"  Invalid rate:               {invalid/total_frames*100:.2f}%")

    if total_black_clips > 0:
        print(f"\n  Clips with black frames:")
        for clip_key, data in sorted(clips.items()):
            if data["black_count"] > 0:
                preview = data["black_frames"][:5]
                suffix = "..." if len(data["black_frames"]) > 5 else ""
                print(f"    {clip_key}: {data['black_count']} black frames "
                      f"(threshold={data['black_threshold_bytes']} bytes): {preview}{suffix}")


if __name__ == "__main__":
    main()
