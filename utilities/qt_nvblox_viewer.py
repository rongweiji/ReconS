#!/usr/bin/env python3
"""
Voxel grid viewer for nvblox occupancy/ESDF PLY or NPZ files.

Usage:
  python utilities/qt_mesh_viewer.py path/to/voxel_grid.ply
  python utilities/qt_mesh_viewer.py path/to/voxel_grid.npz
"""

import argparse
from pathlib import Path
import sys
import numpy as np

try:
    from nvblox_common.voxel_grid import VoxelGrid
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    nvblox_common_root = repo_root / "third_party" / "nvblox" / "python" / "common"
    sys.path.insert(0, str(nvblox_common_root))
    from nvblox_common.voxel_grid import VoxelGrid

try:
    from plyfile import PlyData  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("plyfile is required. Install with: python -m pip install plyfile") from exc
try:
    from PySide6 import QtGui, QtWidgets  # type: ignore
    import pyqtgraph.opengl as gl  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Qt viewer requires PySide6 and pyqtgraph. Install with: "
        "python -m pip install PySide6 pyqtgraph PyOpenGL"
    ) from exc


def _esdf_colors(values: "np.ndarray") -> "np.ndarray":
    import numpy as np

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.tile(np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32), (values.shape[0], 1))
    t = (values - vmin) / (vmax - vmin)
    colors = np.stack([t, 0.2 + 0.6 * (1.0 - t), 1.0 - t], axis=1)
    alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
    return np.concatenate([colors, alpha], axis=1).astype(np.float32)


class QtViewer(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=3)

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        self.view.addItem(grid)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.setWindowTitle(title)

    def add_scatter(self, points, colors):
        scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=2.0)
        self.view.addItem(scatter)
        if points.size > 0:
            center = points.mean(axis=0)
            extent = float((points.max(axis=0) - points.min(axis=0)).max())
            if extent > 0:
                self.view.opts["distance"] = extent * 1.5
            self.view.opts["center"] = QtGui.QVector3D(*center)

    def add_mesh(self, verts, faces, colors):
        mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=None,
            vertexColors=colors,
            smooth=True,
            computeNormals=True,
            drawEdges=False,
        )
        self.view.addItem(mesh_item)
        if verts.size > 0:
            center = verts.mean(axis=0)
            extent = float((verts.max(axis=0) - verts.min(axis=0)).max())
            if extent > 0:
                self.view.opts["distance"] = extent * 1.5
            self.view.opts["center"] = QtGui.QVector3D(*center)


def _detect_ply_kind(path: Path) -> str:
    ply = PlyData.read(str(path))
    elements = {el.name: el for el in ply.elements}
    if "face" in elements:
        return "mesh"
    vertex = elements.get("vertex")
    if vertex is not None and "intensity" in vertex.data.dtype.names:
        return "voxel"
    return "mesh"


def _load_mesh(mesh_path: Path):
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("Open3D is required for mesh preview. Install with: python -m pip install open3d") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh or mesh is empty: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vcolors = None
    if mesh.has_vertex_colors():
        vcolors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        if vcolors.max() > 1.0:
            vcolors = vcolors / 255.0
    return verts, faces, vcolors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a VoxelGrid stored as ply or npz file."
    )
    parser.add_argument("file_path", type=Path, help="Path to the file to visualize.")
    parser.add_argument(
        "--max_visualization_dist_vox",
        type=int,
        default=2,
        help=(
            "Max. distance in voxels to the surface to show a voxel in the ESDF "
            "pointcloud."
        ),
    )
    args = parser.parse_args()

    file_extension = args.file_path.suffix.lstrip(".").lower()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    if file_extension == "npz":
        print("Loading npz file:", args.file_path)
        voxel_grid = VoxelGrid.create_from_npz(args.file_path)
        if voxel_grid.is_occupancy_grid:
            print("Visualizing the voxel grid as occupancy point cloud.")
            centers = voxel_grid.get_valid_voxel_centers()
            values = voxel_grid.get_valid_voxel_values()
            mask = values.astype(bool)
            points = centers[mask]
            colors = np.tile(np.array([0.2, 0.9, 0.3, 1.0], dtype=np.float32), (points.shape[0], 1))
            title = "Occupancy Voxel Grid"
        else:
            print("Visualizing the voxel grid as esdf point cloud.")
            centers = voxel_grid.get_valid_voxel_centers()
            values = voxel_grid.get_valid_voxel_values()
            max_dist_m = float(args.max_visualization_dist_vox) * float(voxel_grid.get_voxel_size())
            mask = values < max_dist_m
            points = centers[mask]
            colors = _esdf_colors(values[mask].astype(np.float32))
            title = "ESDF Voxel Grid"
        win = QtViewer(title)
        win.add_scatter(points.astype("float32"), colors.astype("float32"))
    elif file_extension == "ply":
        kind = _detect_ply_kind(args.file_path)
        if kind == "voxel":
            print("Loading ply file as voxel grid:", args.file_path)
            voxel_grid = VoxelGrid.create_from_ply(args.file_path)
            name_hint = "occupancy" in args.file_path.name.lower()
            values = voxel_grid.get_valid_voxel_values()
            is_binary = np.all((values == 0) | (values == 1))
            voxel_grid.is_occupancy_grid = bool(name_hint or is_binary)
            if voxel_grid.is_occupancy_grid:
                print("Visualizing the voxel grid as occupancy point cloud.")
                centers = voxel_grid.get_valid_voxel_centers()
                mask = values.astype(bool)
                points = centers[mask]
                colors = np.tile(np.array([0.2, 0.9, 0.3, 1.0], dtype=np.float32), (points.shape[0], 1))
                title = "Occupancy Voxel Grid"
            else:
                print("Visualizing the voxel grid as esdf point cloud.")
                centers = voxel_grid.get_valid_voxel_centers()
                values = voxel_grid.get_valid_voxel_values()
                max_dist_m = float(args.max_visualization_dist_vox) * float(voxel_grid.get_voxel_size())
                mask = values < max_dist_m
                points = centers[mask]
                colors = _esdf_colors(values[mask].astype(np.float32))
                title = "ESDF Voxel Grid"
            win = QtViewer(title)
            win.add_scatter(points.astype("float32"), colors.astype("float32"))
        else:
            print("Loading ply file as mesh:", args.file_path)
            verts, faces, vcolors = _load_mesh(args.file_path)
            title = "Mesh Viewer"
            win = QtViewer(title)
            win.add_mesh(verts.astype("float32"), faces.astype("int32"), vcolors)
    else:
        raise ValueError("File extension not supported: " + file_extension)

    win.resize(1200, 900)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
