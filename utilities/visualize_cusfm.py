#!/usr/bin/env python3
"""
Quick visualization of cuSFM results.

Default paths point to the outputs produced in ReconS/data/cusfm_output:
- Trajectory: output_poses/merged_pose_file.tum
- Sparse points: sparse/points3D.txt (COLMAP text format)

Usage example:
  python tools/visualize_cusfm.py \
      --poses /mnt/g/GithubProject/ReconS/data/cusfm_output/output_poses/merged_pose_file.tum \
      --points /mnt/g/GithubProject/ReconS/data/cusfm_output/sparse/points3D.txt \
      --max_points 30000
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_tum_poses(path: Path) -> np.ndarray:
    poses: List[Tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        _, x, y, z, *_ = map(float, parts[:8])
        poses.append((x, y, z))
    return np.array(poses)


def load_colmap_points(path: Path, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            # id X Y Z R G B ...
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            points.append((x, y, z))
            colors.append((r / 255.0, g / 255.0, b / 255.0))
            if len(points) >= max_points:
                break
    if not points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.array(points), np.array(colors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cuSFM trajectory and sparse points.")
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("/mnt/g/GithubProject/ReconS/data/cusfm_output/output_poses/merged_pose_file.tum"),
        help="Path to TUM pose file.",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=Path("/mnt/g/GithubProject/ReconS/data/cusfm_output/sparse/points3D.txt"),
        help="Path to COLMAP points3D.txt.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=20000,
        help="Maximum number of 3D points to plot from points3D.txt.",
    )
    args = parser.parse_args()

    poses = load_tum_poses(args.poses)
    if poses.size == 0:
        raise SystemExit(f"No poses read from {args.poses}")

    points, colors = load_colmap_points(args.points, args.max_points)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], "-", color="C1", label="Trajectory")
    ax.scatter(poses[0, 0], poses[0, 1], poses[0, 2], color="green", s=40, label="Start")
    ax.scatter(poses[-1, 0], poses[-1, 1], poses[-1, 2], color="red", s=40, label="End")

    if points.size > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors if colors.size else "C0",
            s=0.5,
            alpha=0.6,
            label=f"Points (n={len(points)})",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    ax.set_title("cuSFM Trajectory and Sparse Points")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
