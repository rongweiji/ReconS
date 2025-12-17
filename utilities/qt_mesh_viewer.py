#!/usr/bin/env python3
"""
Qt + pyqtgraph mesh viewer to inspect nvblox outputs without GLFW.

Usage:
  python utilities/qt_mesh_viewer.py path/to/mesh.ply

Requires: PySide6, pyqtgraph, PyOpenGL, numpy, and Open3D (for mesh loading only).
Open3D is used headlessly to parse the mesh; rendering is via Qt/pyqtgraph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl


def load_mesh(mesh_path: Path):
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError:
        print("Open3D is required to load meshes. Install with: python -m pip install open3d", file=sys.stderr)
        sys.exit(1)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh or mesh is empty: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vnormals = np.asarray(mesh.vertex_normals, dtype=np.float32) if mesh.has_vertex_normals() else None
    vcolors = None
    if mesh.has_vertex_colors():
        vcolors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        if vcolors.max() > 1.0:
            vcolors = vcolors / 255.0
    return verts, faces, vnormals, vcolors


class MeshViewer(QtWidgets.QWidget):
    def __init__(self, verts: np.ndarray, faces: np.ndarray, normals=None, colors=None):
        super().__init__()
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=3)

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        self.view.addItem(grid)

        mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=None,
            vertexColors=colors,
            smooth=normals is not None,
            computeNormals=normals is None,
            drawEdges=False,
        )
        self.view.addItem(mesh_item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.setWindowTitle("Qt Mesh Viewer")

        # Center/scale view
        center = verts.mean(axis=0)
        extent = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
        if extent > 0:
            self.view.opts["distance"] = extent * 1.5
        self.view.opts["center"] = QtGui.QVector3D(*center)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Path to mesh file (ply/obj/etc.)")
    args = parser.parse_args()

    verts, faces, normals, colors = load_mesh(args.mesh)

    app = QtWidgets.QApplication(sys.argv)
    win = MeshViewer(verts, faces, normals, colors)
    win.resize(1200, 900)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
