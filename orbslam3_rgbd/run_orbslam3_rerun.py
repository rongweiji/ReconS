#!/usr/bin/env python3
"""
Replay ORB-SLAM3 RGBD results in the Rerun UI (RGB, depth, trajectory).

Requirements:
  python3 -m pip install rerun-sdk opencv-python pyyaml numpy

Example:
  python3 orbslam3_rgbd/run_orbslam3_rerun.py \
    --rgb-dir data/sample_20260119_125703/left \
    --depth-dir data/sample_20260119_125703/left_depth \
    --trajectory data/sample_20260119_125703/orbslam3_poses.tum \
    --calibration data/sample_20260119_125703/left_calibration.yaml \
    --timestamps data/sample_20260119_125703/timestamps.txt \
    --depth-scale 1000 \
    --fps 30

If timestamps are provided, playback uses them; otherwise falls back to a fixed FPS.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import yaml


def parse_calibration(calib_path: Path) -> Optional[Tuple[float, float, float, float]]:
    data = yaml.safe_load(calib_path.read_text())
    if not isinstance(data, dict):
        return None

    def flatten(val) -> List[float]:
        if isinstance(val, dict):
            arr = val.get("data") or val.get("Data") or val.get("values")
            return arr if isinstance(arr, list) else []
        if isinstance(val, list):
            flat: List[float] = []
            for row in val:
                if isinstance(row, list):
                    flat.extend(row)
                else:
                    flat.append(row)
            return flat
        return []

    k = []
    if "K" in data:
        k = flatten(data["K"])
    if not k and "camera_matrix" in data:
        k = flatten(data["camera_matrix"])
    if len(k) >= 6:
        fx = float(k[0])
        fy = float(k[4]) if len(k) > 4 else float(k[1])
        cx = float(k[2])
        cy = float(k[5]) if len(k) > 5 else float(k[3])
        return fx, fy, cx, cy
    return None


def load_poses(tum_path: Path) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    poses = []
    with tum_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            poses.append((t, np.array([tx, ty, tz], dtype=np.float32), np.array([qx, qy, qz, qw], dtype=np.float32)))
    if not poses:
        raise SystemExit(f"No poses found in {tum_path}")
    return poses


def load_timestamps(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    rows = list(csv.reader(path.read_text().splitlines()))
    if not rows or len(rows) < 2:
        return None
    header = [h.strip().lower() for h in rows[0]]
    frame_idx = header.index("frame") if "frame" in header else header.index("filename")
    ts_idx = header.index("timestamp_ns")
    mapping: Dict[str, float] = {}
    for row in rows[1:]:
        if not row or len(row) <= ts_idx:
            continue
        frame_id = row[frame_idx].strip()
        ts_ns = float(row[ts_idx])
        mapping[frame_id] = ts_ns * 1e-9
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ORB-SLAM3 RGBD output with Rerun.")
    parser.add_argument("--rgb-dir", required=True, type=Path, help="Directory with RGB frames")
    parser.add_argument("--depth-dir", required=True, type=Path, help="Directory with depth frames")
    parser.add_argument("--trajectory", required=True, type=Path, help="TUM trajectory file (orbslam3_poses.tum)")
    parser.add_argument("--calibration", type=Path, help="Calibration YAML with K matrix")
    parser.add_argument("--timestamps", type=Path, help="timestamps.txt (frame,timestamp_ns)")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale (value per meter, default 1000 for mm)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS if timestamps are missing")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit number of frames to visualize")
    parser.add_argument("--spawn", action="store_true", help="Spawn a Rerun viewer (default: connect to background viewer if running)")
    args = parser.parse_args()

    rgb_dir = args.rgb_dir.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    traj_path = args.trajectory.expanduser().resolve()

    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise SystemExit("RGB or depth directory not found.")
    if not traj_path.is_file():
        raise SystemExit(f"Trajectory file not found: {traj_path}")

    poses = load_poses(traj_path)
    timestamps = load_timestamps(args.timestamps) if args.timestamps else None

    # Assume filenames share numeric stem with poses order (first pose -> 0000001.*)
    frame_ids = [f"{i+1:07d}" for i in range(len(poses))]
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
        poses = poses[: args.max_frames]

    fx_fy_cx_cy = None
    if args.calibration:
        fx_fy_cx_cy = parse_calibration(args.calibration.expanduser().resolve())

    rr.init("ORB-SLAM3 RGBD (rerun)", spawn=args.spawn)

    # Prepare static camera intrinsics if available.
    sample_rgb = None
    for ext in (".png", ".jpg", ".jpeg"):
        sample_rgb = cv2.imread(str(rgb_dir / f"{frame_ids[0]}{ext}"))
        if sample_rgb is not None:
            break
    if sample_rgb is None:
        raise SystemExit(f"Failed to read sample RGB frame {frame_ids[0]}")
    h, w = sample_rgb.shape[:2]
    if fx_fy_cx_cy is None:
        fx = fy = 0.5 * max(w, h)
        cx = w / 2.0
        cy = h / 2.0
    else:
        fx, fy, cx, cy = fx_fy_cx_cy

    rr.log(
        "world/camera",
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=w,
            height=h,
        ),
    )

    # Layout: left = 3D trajectory, right column = RGB (top), Depth (bottom).
    rr.send_blueprint(
        rr.blueprint.Blueprint(
            rr.blueprint.Grid(
                contents=[
                    rr.blueprint.Spatial3DView(origin="world", name="Trajectory"),
                    rr.blueprint.Grid(
                        contents=[
                            rr.blueprint.Spatial2DView(origin="ui/rgb", name="RGB"),
                            rr.blueprint.Spatial2DView(origin="ui/depth", name="Depth"),
                        ],
                        grid_columns=1,
                        name="Images",
                    ),
                ],
                grid_columns=2,
                name="Layout",
            )
        )
    )

    path_points: List[List[float]] = []
    meter = 1.0 / args.depth_scale if args.depth_scale > 0 else None

    for idx, (frame_id, (t, pos, quat)) in enumerate(zip(frame_ids, poses)):
        has_ts = timestamps is not None and frame_id in timestamps
        if has_ts:
            rr.set_time_sequence("frame", idx)
            rr.set_time_seconds("time", timestamps[frame_id])
        else:
            rr.set_time_sequence("frame", idx)
            rr.set_time_seconds("time", idx / args.fps)

        rgb_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = rgb_dir / f"{frame_id}{ext}"
            if candidate.is_file():
                rgb_path = candidate
                break
        if rgb_path is None:
            continue
        depth_path = depth_dir / f"{frame_id}.png"

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rr.log("ui/rgb", rr.Image(rgb_rgb))
        depth_f = depth.astype(np.float32)
        rr.log("ui/depth", rr.DepthImage(depth_f, meter=meter))

        rr.log(
            "world/camera",
            rr.Transform3D(translation=pos.tolist(), quaternion=quat.tolist()),
        )

        path_points.append(pos.tolist())
        rr.log("world/trajectory", rr.LineStrips3D([path_points]))


if __name__ == "__main__":
    main()
