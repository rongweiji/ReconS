#!/usr/bin/env python3
"""
Compare two TUM pose files and visualize them in Rerun UI.

Usage:
    python utilities/compare_poses_rerun.py \
        --pose1 data/sample_xxx/cuvslam_poses_slam.tum \
        --pose2 data/sample_xxx/cusfm_output/output_poses/merged_pose_file.tum \
        --label1 "SLAM" \
        --label2 "SfM Refined"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation


@dataclass
class Pose:
    """Single pose with timestamp."""
    timestamp: float
    position: np.ndarray  # [x, y, z]
    quaternion: np.ndarray  # [qx, qy, qz, qw]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        return Rotation.from_quat(self.quaternion).as_matrix()

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T


def load_tum_poses(filepath: Path) -> List[Pose]:
    """Load poses from TUM format file.

    TUM format: timestamp tx ty tz qx qy qz qw
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            poses.append(Pose(
                timestamp=timestamp,
                position=np.array([tx, ty, tz]),
                quaternion=np.array([qx, qy, qz, qw])
            ))
    return poses


def create_axis_lines(pose: Pose, scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create axis visualization lines for a pose.

    Returns:
        Tuple of (x_axis_points, y_axis_points, z_axis_points)
        Each is a (2, 3) array representing start and end points
    """
    origin = pose.position
    R = pose.rotation_matrix

    x_axis = np.array([origin, origin + R[:, 0] * scale])
    y_axis = np.array([origin, origin + R[:, 1] * scale])
    z_axis = np.array([origin, origin + R[:, 2] * scale])

    return x_axis, y_axis, z_axis


def visualize_trajectory(
    poses: List[Pose],
    entity_path: str,
    color: Tuple[int, int, int],
    label: str,
    show_orientation: bool = True,
    orientation_interval: int = 10,
    axis_scale: float = 0.05,
) -> None:
    """Log a trajectory to Rerun.

    Args:
        poses: List of poses
        entity_path: Rerun entity path (e.g., "world/slam")
        color: RGB color tuple (0-255)
        label: Label for the trajectory
        show_orientation: Whether to show orientation axes
        orientation_interval: Show orientation every N poses
        axis_scale: Scale of orientation axes
    """
    if not poses:
        print(f"Warning: No poses to visualize for {label}")
        return

    # Extract positions for the path
    positions = np.array([p.position for p in poses])

    # Log the trajectory path as a line strip
    rr.log(
        f"{entity_path}/path",
        rr.LineStrips3D(
            [positions],
            colors=[color],
            labels=[label],
        ),
    )

    # Log positions as points
    rr.log(
        f"{entity_path}/points",
        rr.Points3D(
            positions,
            colors=[color] * len(positions),
            radii=[0.005] * len(positions),
        ),
    )

    # Log orientation axes at intervals
    if show_orientation:
        x_axes = []
        y_axes = []
        z_axes = []

        for i, pose in enumerate(poses):
            if i % orientation_interval == 0:
                x_axis, y_axis, z_axis = create_axis_lines(pose, scale=axis_scale)
                x_axes.append(x_axis)
                y_axes.append(y_axis)
                z_axes.append(z_axis)

        if x_axes:
            # X-axis (red)
            rr.log(
                f"{entity_path}/orientation/x",
                rr.LineStrips3D(x_axes, colors=[[255, 0, 0]] * len(x_axes)),
            )
            # Y-axis (green)
            rr.log(
                f"{entity_path}/orientation/y",
                rr.LineStrips3D(y_axes, colors=[[0, 255, 0]] * len(y_axes)),
            )
            # Z-axis (blue)
            rr.log(
                f"{entity_path}/orientation/z",
                rr.LineStrips3D(z_axes, colors=[[0, 0, 255]] * len(z_axes)),
            )


def compute_trajectory_stats(poses: List[Pose], label: str) -> None:
    """Print trajectory statistics."""
    if not poses:
        print(f"{label}: No poses")
        return

    positions = np.array([p.position for p in poses])

    # Compute total path length
    diffs = np.diff(positions, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    # Bounding box
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)

    print(f"\n{label}:")
    print(f"  Poses: {len(poses)}")
    print(f"  Path length: {path_length:.3f} m")
    print(f"  Bounding box: [{min_pos[0]:.3f}, {min_pos[1]:.3f}, {min_pos[2]:.3f}] to [{max_pos[0]:.3f}, {max_pos[1]:.3f}, {max_pos[2]:.3f}]")
    print(f"  Time range: {poses[0].timestamp:.3f} to {poses[-1].timestamp:.3f} s")


def compute_pose_differences(poses1: List[Pose], poses2: List[Pose]) -> None:
    """Compute and print differences between two trajectories (aligned by timestamp)."""
    if not poses1 or not poses2:
        return

    # Create timestamp lookup for poses2
    ts2_to_pose = {p.timestamp: p for p in poses2}

    position_errors = []
    rotation_errors = []
    matched_count = 0

    for p1 in poses1:
        # Find matching timestamp in poses2 (within tolerance)
        best_match = None
        best_diff = float('inf')
        for ts2 in ts2_to_pose:
            diff = abs(p1.timestamp - ts2)
            if diff < best_diff and diff < 0.001:  # 1ms tolerance
                best_diff = diff
                best_match = ts2_to_pose[ts2]

        if best_match is not None:
            matched_count += 1
            # Position error
            pos_err = np.linalg.norm(p1.position - best_match.position)
            position_errors.append(pos_err)

            # Rotation error (angle between rotations)
            R1 = Rotation.from_quat(p1.quaternion)
            R2 = Rotation.from_quat(best_match.quaternion)
            R_diff = R1.inv() * R2
            angle_err = np.abs(R_diff.magnitude())  # in radians
            rotation_errors.append(np.degrees(angle_err))

    if position_errors:
        pos_errors = np.array(position_errors)
        rot_errors = np.array(rotation_errors)
        print(f"\nPose Differences (matched {matched_count} poses):")
        print(f"  Position error - Mean: {pos_errors.mean():.4f} m, Max: {pos_errors.max():.4f} m, Std: {pos_errors.std():.4f} m")
        print(f"  Rotation error - Mean: {rot_errors.mean():.4f}°, Max: {rot_errors.max():.4f}°, Std: {rot_errors.std():.4f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two TUM pose files and visualize in Rerun UI"
    )
    parser.add_argument(
        "--pose1", "-p1",
        type=Path,
        required=True,
        help="First TUM pose file (e.g., SLAM poses)"
    )
    parser.add_argument(
        "--pose2", "-p2",
        type=Path,
        required=True,
        help="Second TUM pose file (e.g., SfM refined poses)"
    )
    parser.add_argument(
        "--label1", "-l1",
        type=str,
        default="Trajectory 1",
        help="Label for first trajectory"
    )
    parser.add_argument(
        "--label2", "-l2",
        type=str,
        default="Trajectory 2",
        help="Label for second trajectory"
    )
    parser.add_argument(
        "--color1",
        type=str,
        default="255,100,100",
        help="RGB color for trajectory 1 (comma-separated)"
    )
    parser.add_argument(
        "--color2",
        type=str,
        default="100,100,255",
        help="RGB color for trajectory 2 (comma-separated)"
    )
    parser.add_argument(
        "--orientation-interval",
        type=int,
        default=10,
        help="Show orientation axes every N poses"
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.05,
        help="Scale of orientation axes"
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Hide orientation axes"
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        default=True,
        help="Spawn Rerun viewer (default: True)"
    )
    parser.add_argument(
        "--connect",
        type=str,
        default=None,
        help="Connect to existing Rerun viewer at address (e.g., 127.0.0.1:9876)"
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save to .rrd file instead of viewing"
    )
    args = parser.parse_args()

    # Validate input files
    if not args.pose1.exists():
        raise FileNotFoundError(f"Pose file not found: {args.pose1}")
    if not args.pose2.exists():
        raise FileNotFoundError(f"Pose file not found: {args.pose2}")

    # Parse colors
    color1 = tuple(map(int, args.color1.split(',')))
    color2 = tuple(map(int, args.color2.split(',')))

    # Load poses
    print(f"Loading {args.pose1}...")
    poses1 = load_tum_poses(args.pose1)
    print(f"Loading {args.pose2}...")
    poses2 = load_tum_poses(args.pose2)

    # Print statistics
    compute_trajectory_stats(poses1, args.label1)
    compute_trajectory_stats(poses2, args.label2)
    compute_pose_differences(poses1, poses2)

    # Initialize Rerun
    rr.init("pose_comparison", spawn=False)

    if args.save:
        rr.save(str(args.save))
    elif args.connect:
        rr.connect(args.connect)
    else:
        rr.spawn()

    # Set up 3D view
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Visualize trajectories
    print(f"\nVisualizing trajectories...")

    visualize_trajectory(
        poses1,
        entity_path=f"world/{args.label1.lower().replace(' ', '_')}",
        color=color1,
        label=args.label1,
        show_orientation=not args.no_orientation,
        orientation_interval=args.orientation_interval,
        axis_scale=args.axis_scale,
    )

    visualize_trajectory(
        poses2,
        entity_path=f"world/{args.label2.lower().replace(' ', '_')}",
        color=color2,
        label=args.label2,
        show_orientation=not args.no_orientation,
        orientation_interval=args.orientation_interval,
        axis_scale=args.axis_scale,
    )

    # Add origin marker
    rr.log(
        "world/origin",
        rr.Points3D([[0, 0, 0]], colors=[[255, 255, 255]], radii=[0.02]),
    )

    print("\nVisualization complete. Check Rerun viewer.")

    if not args.save and not args.connect:
        # Keep script running so Rerun viewer stays open
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
