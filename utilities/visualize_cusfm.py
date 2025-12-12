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


def visualize_plotly(poses: np.ndarray, points: np.ndarray, colors: np.ndarray, title: str) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        raise SystemExit("Plotly is not installed. Install with: python -m pip install plotly")

    traces = []
    traces.append(
        go.Scatter3d(
            x=poses[:, 0],
            y=poses[:, 1],
            z=poses[:, 2],
            mode="lines",
            name="Trajectory",
            line=dict(width=4, color="orange"),
        )
    )
    traces.append(
        go.Scatter3d(
            x=[poses[0, 0]],
            y=[poses[0, 1]],
            z=[poses[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=6, color="green"),
        )
    )
    traces.append(
        go.Scatter3d(
            x=[poses[-1, 0]],
            y=[poses[-1, 1]],
            z=[poses[-1, 2]],
            mode="markers",
            name="End",
            marker=dict(size=6, color="red"),
        )
    )

    if points.size > 0:
        if colors.size:
            rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            color_str = [f"rgb({r},{g},{b})" for r, g, b in rgb]
            marker = dict(size=1.5, color=color_str, opacity=0.7)
        else:
            marker = dict(size=1.5, color="rgba(80,140,255,0.7)")

        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=f"Points (n={len(points)})",
                marker=marker,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()


def visualize_qt(poses: np.ndarray, points: np.ndarray, colors: np.ndarray, title: str) -> None:
    """Qt 3D viewer via pyqtgraph.opengl (wheel zoom + mouse pan/orbit)."""
    try:
        from pyqtgraph.Qt import QtGui, QtWidgets  # type: ignore
        import pyqtgraph.opengl as gl  # type: ignore
    except Exception:
        raise ImportError(
            "Qt backend requires PyQtGraph + a Qt binding. Install with: "
            "python -m pip install pyqtgraph PySide6 PyOpenGL"
        )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(title)
    view = gl.GLViewWidget()
    win.setCentralWidget(view)
    win.resize(1200, 800)

    # Add axes/grid for orientation
    axis = gl.GLAxisItem()
    view.addItem(axis)
    grid = gl.GLGridItem()
    grid.scale(1, 1, 1)
    view.addItem(grid)

    # Auto-center and set initial distance based on scene scale
    all_xyz = poses
    if points.size > 0:
        all_xyz = np.vstack([poses, points])
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    diag = float(np.linalg.norm(maxs - mins))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    # pyqtgraph expects a QVector3D for the center on most Qt bindings
    if hasattr(QtGui, "QVector3D"):
        view.opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))
    else:
        view.opts["center"] = (float(center[0]), float(center[1]), float(center[2]))
    view.setCameraPosition(distance=diag * 1.2)

    # Trajectory line
    traj = gl.GLLinePlotItem(pos=poses, color=(1.0, 0.6, 0.0, 1.0), width=2, antialias=True, mode="line_strip")
    view.addItem(traj)

    # Start/end markers
    start = gl.GLScatterPlotItem(pos=poses[[0]], color=(0.1, 0.9, 0.1, 1.0), size=10)
    end = gl.GLScatterPlotItem(pos=poses[[-1]], color=(0.9, 0.1, 0.1, 1.0), size=10)
    view.addItem(start)
    view.addItem(end)

    # Point cloud
    if points.size > 0:
        if colors.size:
            rgba = np.concatenate([colors, np.full((colors.shape[0], 1), 0.7, dtype=colors.dtype)], axis=1)
        else:
            rgba = (0.3, 0.55, 1.0, 0.7)
        pcd = gl.GLScatterPlotItem(pos=points, color=rgba, size=1)
        view.addItem(pcd)

    win.show()
    # Qt6 (PySide6/PyQt6) uses exec(); Qt5 used exec_().
    if hasattr(app, "exec"):
        app.exec()  # type: ignore[attr-defined]
    else:
        app.exec_()  # type: ignore[attr-defined]


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

    title = "cuSFM Trajectory and Sparse Points"
    try:
        visualize_qt(poses, points, colors, title)
    except ImportError as e:
        print(f"[warn] {e}\n[warn] Falling back to Plotly viewer.")
        visualize_plotly(poses, points, colors, title)


if __name__ == "__main__":
    main()
