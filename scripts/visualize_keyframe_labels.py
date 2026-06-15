"""
Visualize keyframe labels: front camera + BEV + navigation_text + CoC label.

Usage:
    python visualize_keyframe_labels.py <DATE>
    e.g.: python visualize_keyframe_labels.py 2025-11-19

Output (flat, no per-scene hierarchy):
    batch_outputs/<DATE>/keyframe_viz/<clip_id>_f<frame:06d>_<meta_action>.png
"""
import os
import sys
import tarfile
import textwrap
from collections import defaultdict
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import yaml

from generate_navigation_labels import (
    BATCH_OUTPUTS,
    DATE_TO_MAP_VERSION,
    MAP_BASE,
    TAR_BASE,
    LaneletMapPolygonIndex,
    decode_npy_zst,
    load_keyframes,
)

# Only tensors needed for visualization (route + trajectory + valid_mask)
VIZ_TENSOR_SUFFIXES = {
    ".route.npy.zst": "route",
    ".trajectory.npy.zst": "trajectory",
    ".valid_mask.npy.zst": "valid_mask",
}


def extract_from_tar(
    tar_path: str, frame_indices: list[int]
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """
    Two-pass streaming extraction:
      Pass 1 – collect & sort cam_front.jpg names (header scan only)
      Pass 2 – extract tensors + camera frames for requested indices, early exit
    Returns (tensors, {frame_idx: BGR image}).
    """
    # Pass 1: collect camera frame names (pattern: cameras/NNNN_CAM_FRONT.jpg)
    cam_names: list[str] = []
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if member.name.lower().endswith("_cam_front.jpg"):
                cam_names.append(member.name)
    cam_names.sort()

    # Map cam_name -> frame_idx for needed frames
    needed_cams: dict[str, int] = {}
    for idx in frame_indices:
        if 0 <= idx < len(cam_names):
            needed_cams[cam_names[idx]] = idx

    tensors: dict[str, np.ndarray] = {}
    cam_frames: dict[int, np.ndarray] = {}
    n_tensors = len(VIZ_TENSOR_SUFFIXES)
    n_cams = len(needed_cams)

    # Pass 2: tensors + camera frames in one streaming pass
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            matched = False
            for suffix, key in VIZ_TENSOR_SUFFIXES.items():
                if member.name.endswith(suffix) and key not in tensors:
                    f = tar.extractfile(member)
                    if f:
                        tensors[key] = decode_npy_zst(f.read())
                    matched = True
                    break

            if not matched and member.name in needed_cams:
                frame_idx = needed_cams[member.name]
                if frame_idx not in cam_frames:
                    f = tar.extractfile(member)
                    if f:
                        img_arr = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            cam_frames[frame_idx] = img

            if len(tensors) >= n_tensors and len(cam_frames) >= n_cams:
                break

    return tensors, cam_frames


def load_coc_text(date_dir: str, clip_id: str, event_start_timestamp: int) -> str:
    path = os.path.join(date_dir, "coc_labels", clip_id, f"cot_{event_start_timestamp}.yaml")
    if not os.path.exists(path):
        return "N/A"
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return data["final_content"]["ego_behavior_schema"]["effect_on_ego_behavior"]
    except (KeyError, TypeError):
        return "N/A"


def load_nav_text(date_dir: str, clip_id: str, event_start_timestamp: int) -> str:
    path = os.path.join(date_dir, "nav_labels", clip_id, f"nav_{event_start_timestamp}.yaml")
    if not os.path.exists(path):
        return "N/A"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("navigation_text", "N/A")


def draw_bev(
    ax,
    trajectory: np.ndarray,
    route_tensor: Optional[np.ndarray],
    lanelet_map: LaneletMapPolygonIndex,
    frame_idx: int,
    nav_text: str,
) -> None:
    """Draw bird's eye view on ax (dark background, map polygons, trajectory, route, HERE marker)."""
    ax.set_facecolor("#1e1e1e")

    if trajectory is None or len(trajectory) == 0:
        ax.text(0.5, 0.5, "No trajectory", color="gray", ha="center", va="center",
                transform=ax.transAxes)
        return

    traj_x, traj_y = trajectory[:, 0], trajectory[:, 1]
    min_x, max_x = traj_x.min(), traj_x.max()
    min_y, max_y = traj_y.min(), traj_y.max()
    pad = 40.0

    # Map polygons (only those in view)
    for vertices, label in zip(lanelet_map.raw_coords, lanelet_map.labels):
        px, py = vertices[:, 0], vertices[:, 1]
        if px.max() < min_x - pad or px.min() > max_x + pad:
            continue
        if py.max() < min_y - pad or py.min() > max_y + pad:
            continue
        color = "dodgerblue" if label == "left" else "darkorange" if label == "right" else "dimgray"
        poly = MplPolygon(vertices, closed=True, facecolor=color, edgecolor=color,
                          alpha=0.3, zorder=2)
        ax.add_patch(poly)

    # Full scene trajectory (thin gray)
    ax.plot(traj_x, traj_y, color="lightgray", linewidth=0.8, alpha=0.5, zorder=3)

    # Route prediction at this frame projected to global coords
    if route_tensor is not None and frame_idx < len(route_tensor):
        ego_x = trajectory[frame_idx, 0]
        ego_y = trajectory[frame_idx, 1]
        ego_cos = trajectory[frame_idx, 2]
        ego_sin = trajectory[frame_idx, 3]
        rx_flat = route_tensor[frame_idx, :, :, 0].ravel()
        ry_flat = route_tensor[frame_idx, :, :, 1].ravel()
        valid = ~((rx_flat == 0.0) & (ry_flat == 0.0))
        gx = ego_x + rx_flat[valid] * ego_cos - ry_flat[valid] * ego_sin
        gy = ego_y + rx_flat[valid] * ego_sin + ry_flat[valid] * ego_cos
        ax.scatter(gx, gy, c="yellow", s=5, alpha=0.7, zorder=4, linewidths=0)

    # Current position (HERE marker)
    cx, cy = trajectory[frame_idx, 0], trajectory[frame_idx, 1]
    ax.scatter([cx], [cy], marker="*", s=300, c="white", edgecolors="black",
               linewidths=1.5, zorder=6)
    ax.text(cx, cy + (max_y - min_y) * 0.02 + 1, "HERE",
            color="white", fontsize=8, ha="center", va="bottom",
            fontweight="bold", zorder=7)

    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal")
    ax.tick_params(colors="gray", labelsize=7)
    ax.set_xlabel("x [m]", color="gray", fontsize=8)
    ax.set_ylabel("y [m]", color="gray", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
    ax.set_title(f"BEV  |  {nav_text}", color="white", fontsize=10, pad=4)


def save_composite(
    out_path: str,
    cam_bgr: Optional[np.ndarray],
    trajectory: np.ndarray,
    route_tensor: Optional[np.ndarray],
    lanelet_map: LaneletMapPolygonIndex,
    frame_idx: int,
    nav_text: str,
    coc_text: str,
    meta_action: str,
    clip_id: str,
) -> None:
    fig = plt.figure(figsize=(20, 12), facecolor="#1e1e1e")
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[5, 1],
        hspace=0.04, wspace=0.04,
        left=0.02, right=0.98, top=0.97, bottom=0.03,
    )

    ax_cam = fig.add_subplot(gs[0, 0])
    ax_bev = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])

    # Camera image
    ax_cam.set_facecolor("#1e1e1e")
    ax_cam.axis("off")
    if cam_bgr is not None:
        ax_cam.imshow(cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB))
    else:
        ax_cam.text(0.5, 0.5, "No camera image", color="gray",
                    ha="center", va="center", transform=ax_cam.transAxes, fontsize=14)
    ax_cam.set_title("Front Camera", color="white", fontsize=10, pad=4)

    # BEV
    draw_bev(ax_bev, trajectory, route_tensor, lanelet_map, frame_idx, nav_text)

    # Text panel
    ax_txt.set_facecolor("#111111")
    ax_txt.axis("off")
    # Wrap CoC text to fit width; indent continuation lines
    coc_lines = textwrap.wrap(coc_text or "N/A", width=130)
    coc_str = ("\n" + " " * 15).join(coc_lines)
    info = (
        f"[Navigation]   {nav_text or 'N/A'}\n"
        f"[CoC Label]    {coc_str}\n"
        f"[Meta]  {meta_action}   |   Scene: {clip_id}   |   Frame: {frame_idx}"
    )
    ax_txt.text(
        0.01, 0.5, info,
        transform=ax_txt.transAxes,
        color="white", fontsize=11, va="center", ha="left",
        fontfamily="monospace", linespacing=1.7,
    )

    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close(fig)


def process_date(date: str, lanelet_map: LaneletMapPolygonIndex) -> None:
    date_dir = os.path.join(BATCH_OUTPUTS, date)
    json_path = os.path.join(date_dir, "keyframes", "segments_relative_timestamp_sampled.json")

    if not os.path.exists(json_path):
        print(f"[SKIP] Keyframe JSON not found: {json_path}")
        return

    keyframes = load_keyframes(json_path)
    if not keyframes:
        print(f"[SKIP] {date}: no keyframes found")
        return

    out_dir = os.path.join(date_dir, "keyframe_viz")
    os.makedirs(out_dir, exist_ok=True)

    tar_dir = os.path.join(TAR_BASE, date)

    # Group by clip_id to open each tar only once
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for kf in keyframes:
        by_clip[kf["clip_id"]].append(kf)

    print(f"{date}: {len(by_clip)} scenes, {len(keyframes)} keyframes")

    for clip_id, clip_kfs in sorted(by_clip.items()):
        tar_path = os.path.join(tar_dir, f"{clip_id}.tar")
        if not os.path.exists(tar_path):
            print(f"  [WARN] tar not found: {tar_path}")
            continue

        frame_indices = [kf["event_start_frame"] for kf in clip_kfs]
        tensors, cam_frames = extract_from_tar(tar_path, frame_indices)

        if "trajectory" not in tensors:
            print(f"  [WARN] trajectory tensor missing: {clip_id}")
            continue

        trajectory = tensors["trajectory"]
        route_tensor = tensors.get("route")

        for kf in clip_kfs:
            frame_idx = kf["event_start_frame"]
            meta_action = kf["meta_action"]
            event_start_timestamp = frame_idx * 100_000

            nav_text = load_nav_text(date_dir, clip_id, event_start_timestamp)
            coc_text = load_coc_text(date_dir, clip_id, event_start_timestamp)
            cam_bgr = cam_frames.get(frame_idx)

            if frame_idx >= len(trajectory):
                print(f"  [WARN] frame_idx={frame_idx} out of trajectory range: {clip_id}")
                continue

            fname = f"{clip_id}_f{frame_idx:06d}_{meta_action}.png"
            out_path = os.path.join(out_dir, fname)

            save_composite(
                out_path=out_path,
                cam_bgr=cam_bgr,
                trajectory=trajectory,
                route_tensor=route_tensor,
                lanelet_map=lanelet_map,
                frame_idx=frame_idx,
                nav_text=nav_text,
                coc_text=coc_text,
                meta_action=meta_action,
                clip_id=clip_id,
            )

        print(f"  {clip_id}: {len(clip_kfs)} frames done")

    print(f"{date}: done -> {out_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python visualize_keyframe_labels.py <DATE>")
        print("  e.g.: python visualize_keyframe_labels.py 2025-11-19")
        sys.exit(1)

    date = sys.argv[1]
    map_version = DATE_TO_MAP_VERSION.get(date)
    if map_version is None:
        print(f"Error: no map version defined for date '{date}'.")
        sys.exit(1)

    osm_path = os.path.join(MAP_BASE, map_version, "lanelet2_map.osm")
    lanelet_map = LaneletMapPolygonIndex(osm_path)
    process_date(date, lanelet_map)


if __name__ == "__main__":
    main()
