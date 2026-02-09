"""Run nvblox_torch mapping on raw RGB + depth + calibration + poses.

Inputs (paths):
- --rgb-dir: RGB frames folder (e.g., iphone_mono)
- --depth-dir: depth frames folder (aligned to RGB)
- --calibration: calibration YAML with pinhole K matrix
- --poses: TUM pose file (timestamp tx ty tz qx qy qz qw)
- --timestamps: timestamps.txt (frame,timestamp_ns)

Outputs (default): mesh/voxel exports under --out_dir (defaults to the RGB folder parent).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

import cv2
import numpy as np
import torch
import sys
import time
import yaml


def _write_voxel_ply(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> None:
    if xyz.shape[0] != intensity.shape[0]:
        raise ValueError("xyz/intensity length mismatch.")
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        for (x, y, z), val in zip(xyz, intensity):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {float(val):.6f}\n")




def _export_tsdf_voxel_ply(mapper, out_path: Path, band_m: float, as_occupancy: bool) -> bool:
    if not hasattr(mapper, "tsdf_layer_view"):
        print("Skipping voxel PLY export: Mapper has no tsdf_layer_view.")
        return False
    tsdf_layer = mapper.tsdf_layer_view()
    if not hasattr(tsdf_layer, "get_all_blocks"):
        print("Skipping voxel PLY export: TsdfLayer.get_all_blocks is unavailable.")
        return False

    blocks, indices = tsdf_layer.get_all_blocks()
    block_dim = int(tsdf_layer.block_dim_in_voxels)
    voxel_size = float(tsdf_layer.voxel_size())

    if isinstance(indices, torch.Tensor):
        idxs = indices.to(device="cuda", dtype=torch.int32)
    elif isinstance(indices, (list, tuple)) and (len(indices) == 0 or isinstance(indices[0], torch.Tensor)):
        idxs = torch.stack([i.to(device="cuda", dtype=torch.int32) for i in indices], dim=0) if len(indices) > 0 else torch.empty((0, 3), device="cuda", dtype=torch.int32)
    else:
        idxs = torch.tensor(indices, device="cuda", dtype=torch.int32)

    blocks_iter = blocks if isinstance(blocks, (list, tuple)) else [b for b in blocks]

    pts_accum: list[torch.Tensor] = []
    val_accum: list[torch.Tensor] = []

    for block, bidx in zip(blocks_iter, idxs):
        block = block.to(device="cuda")
        if block.shape[-1] < 2:
            continue
        tsdf = block[..., 0]
        w = block[..., 1]
        mask = w > 0
        if band_m > 0:
            mask &= torch.abs(tsdf) < band_m
        if not torch.any(mask):
            continue

        coords = torch.nonzero(mask, as_tuple=False).to(torch.float32)
        bidx_f = bidx.to(torch.float32)
        gv = bidx_f[None, :] * block_dim + coords + 0.5
        centers = gv * voxel_size
        vals = tsdf[mask].reshape(-1)
        if as_occupancy:
            vals = (vals < 0).to(torch.float32)
        pts_accum.append(centers)
        val_accum.append(vals.to(torch.float32))

    if not pts_accum:
        print("Skipping voxel PLY export: no observed voxels.")
        return False

    xyz = torch.cat(pts_accum, dim=0).cpu().numpy().astype(np.float32)
    vals = torch.cat(val_accum, dim=0).cpu().numpy().astype(np.float32)
    _write_voxel_ply(out_path, xyz, vals)
    return True


def _parse_timestamps(path: Path) -> List[Tuple[str, float]]:
    rows = list(csv.reader(path.read_text().splitlines()))
    if not rows or len(rows) < 2:
        raise ValueError(f"No timestamp rows found in {path}")
    header = [h.strip().lower() for h in rows[0]]
    frame_idx = header.index("frame")
    ts_idx = header.index("timestamp_ns")
    out: list[tuple[str, float]] = []
    for row in rows[1:]:
        if len(row) <= ts_idx:
            continue
        frame_id = row[frame_idx].strip()
        ts_ns = float(row[ts_idx])
        out.append((frame_id, ts_ns * 1e-9))
    if not out:
        raise ValueError(f"No valid timestamp entries in {path}")
    return out


def _find_frame(path_base: Path, frame_id: str, exts: Iterable[str]) -> Path:
    for ext in exts:
        candidate = path_base / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frame {frame_id} not found in {path_base} (exts={list(exts)})")


def _detect_exts(rgb_dir: Path, depth_dir: Path, frame_id: str) -> Tuple[str, str]:
    rgb_ext = _find_frame(rgb_dir, frame_id, [".png", ".jpg", ".jpeg"]).suffix
    depth_ext = _find_frame(depth_dir, frame_id, [".png", ".exr", ".tiff", ".tif"]).suffix
    return rgb_ext, depth_ext


def _parse_k(calib_path: Path) -> Tuple[float, float, float, float]:
    data = yaml.safe_load(calib_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected calibration format in {calib_path}")

    def flatten(val) -> list[float]:
        if isinstance(val, dict):
            arr = val.get("data") or val.get("Data") or val.get("values") or val.get("K") or val.get("k")
            return arr if isinstance(arr, list) else []
        if isinstance(val, list):
            flat: list[float] = []
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
    if len(k) < 6:
        raise ValueError(f"Calibration K is incomplete in {calib_path}")
    fx = float(k[0])
    fy = float(k[4]) if len(k) > 4 else float(k[1])
    cx = float(k[2])
    cy = float(k[5]) if len(k) > 5 else float(k[3])
    return fx, fy, cx, cy


class QtMeshViewer:
    """Qt/pyqtgraph viewer with three panes: RGB, depth, and 3D."""

    def __init__(self, title: str = "nvblox mesh"):
        # Lazy imports so headless mode stays lightweight
        try:
            from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
            import pyqtgraph.opengl as gl  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Qt viewer requires PySide6 and pyqtgraph. Install with: "
                "python -m pip install PySide6 pyqtgraph PyOpenGL"
            ) from exc

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.QtGui = QtGui
        self.gl = gl
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        def img_to_qpixmap(img: np.ndarray) -> QtGui.QPixmap:
            """Convert a numpy image (H,W,3 RGB) or (H,W) grayscale to QPixmap."""
            img = np.ascontiguousarray(img)
            if img.ndim == 2:
                h, w = img.shape
                qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            else:
                h, w, ch = img.shape
                if ch != 3:
                    raise ValueError(f"Expected 3-channel RGB image, got shape {img.shape}")
                qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            return QtGui.QPixmap.fromImage(qimg.copy())

        class ImageView(QtWidgets.QLabel):
            def __init__(self, title: str):
                super().__init__()
                self._title = title
                self._pix: QtGui.QPixmap | None = None
                self.setAlignment(QtCore.Qt.AlignCenter)
                self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
                self.setMinimumSize(320, 240)
                self.setText(title)

            def set_image(self, img: np.ndarray | None):
                if img is None:
                    self._pix = None
                    self.setText(self._title)
                    self.setPixmap(QtGui.QPixmap())
                else:
                    self._pix = img_to_qpixmap(img)
                    self._update_scaled()

            def resizeEvent(self, event):
                super().resizeEvent(event)
                self._update_scaled()

            def _update_scaled(self):
                if self._pix is None:
                    return
                self.setPixmap(
                    self._pix.scaled(
                        self.size(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.FastTransformation,
                    )
                )

        self._img_to_qpixmap = img_to_qpixmap
        self._ImageView = ImageView

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=3)
        self.view.setBackgroundColor('k')
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        self.view.addItem(grid)

        # Image panes
        self.rgb_view = ImageView("RGB")
        self.depth_view = ImageView("Depth")

        self.mesh_item = None
        self.pose_items: list = []
        self.path_item = None
        self.field_item = None
        self.slice_plane_item = None
        self.light_pos: np.ndarray | None = None  # point light in world coords
        def make_small_title(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            # Point-size based scaling can look huge under Qt/WSL DPI settings.
            # Use a small pixel size and cap the label height to keep titles compact.
            f = lbl.font()
            f.setPixelSize(11)
            lbl.setFont(f)
            lbl.setMaximumHeight(16)
            lbl.setContentsMargins(0, 0, 0, 0)
            return lbl

        self.window = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Row 1: RGB + Depth (short row)
        top_row = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        rgb_col = QtWidgets.QVBoxLayout()
        rgb_col.setContentsMargins(0, 0, 0, 0)
        rgb_col.setSpacing(2)
        rgb_col.addWidget(make_small_title("RGB"))
        rgb_col.addWidget(self.rgb_view)
        rgb_widget = QtWidgets.QWidget()
        rgb_widget.setLayout(rgb_col)

        depth_col = QtWidgets.QVBoxLayout()
        depth_col.setContentsMargins(0, 0, 0, 0)
        depth_col.setSpacing(2)
        depth_col.addWidget(make_small_title("Depth"))
        depth_col.addWidget(self.depth_view)
        depth_widget = QtWidgets.QWidget()
        depth_widget.setLayout(depth_col)

        top_layout.addWidget(rgb_widget)
        top_layout.addWidget(depth_widget)
        top_row.setLayout(top_layout)

        # Row 2: 3D view (takes remaining space)
        view_col = QtWidgets.QVBoxLayout()
        view_col.setContentsMargins(0, 0, 0, 0)
        view_col.setSpacing(2)
        view_col.addWidget(make_small_title("3D"))
        view_col.addWidget(self.view)
        view_widget = QtWidgets.QWidget()
        view_widget.setLayout(view_col)

        main_layout.addWidget(top_row)
        main_layout.addWidget(view_widget)
        # Keep the top row at ~25% of the window height.
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

        self.window.setLayout(main_layout)

        # Enforce top row height ~= 1/4 window height even when resizing.
        class _ResizeFilter(QtCore.QObject):
            def __init__(self, host: QtWidgets.QWidget, top: QtWidgets.QWidget):
                super().__init__(host)
                self._top = top

            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Resize:
                    h = int(event.size().height())
                    top_h = max(120, int(h * 0.25))
                    self._top.setFixedHeight(top_h)
                return False

        self._resize_filter = _ResizeFilter(self.window, top_row)
        self.window.installEventFilter(self._resize_filter)
        self.window.setWindowTitle(title)
        self.window.resize(1600, 900)

    def update_rgb_frame(self, rgb_uint8: np.ndarray):
        """Update the RGB pane. Expects RGB uint8 image (H,W,3)."""
        if rgb_uint8 is None:
            return
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            raise ValueError(f"RGB frame must be (H,W,3), got {rgb_uint8.shape}")
        if rgb_uint8.dtype != np.uint8:
            rgb_uint8 = rgb_uint8.astype(np.uint8)
        self.rgb_view.set_image(rgb_uint8)
        self.process_events()

    def update_depth_frame(self, depth_m: np.ndarray, *, max_depth_m: float = 5.0):
        """Update the depth pane from metric depth (H,W) float32 in meters."""
        if depth_m is None:
            return
        if depth_m.ndim != 2:
            raise ValueError(f"Depth frame must be (H,W), got {depth_m.shape}")
        dm = depth_m.astype(np.float32)
        # Normalize: 0 => invalid; clip to max depth for visualization.
        valid = dm > 0
        dm_vis = np.zeros_like(dm, dtype=np.float32)
        dm_vis[valid] = np.clip(dm[valid], 0.0, float(max_depth_m)) / max(float(max_depth_m), 1e-6)
        norm_uint8 = (dm_vis * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_PLASMA)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        self.depth_view.set_image(colored_rgb)
        self.process_events()

    def show(self):
        self.window.show()
        self.process_events()

    def process_events(self):
        self.app.processEvents()

    def refresh(self):
        self.process_events()

    def set_light_position(self, pos: np.ndarray):
        self.light_pos = pos.astype(np.float32)

    @staticmethod
    def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(verts, dtype=np.float32)
        tris = verts[faces]
        face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)
        norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        normals /= norm
        return normals

    def update_mesh(self, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray | None, use_vertex_colors: bool = True):
        # Remove existing mesh item if present
        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)
        if self.field_item is not None:
            self.view.removeItem(self.field_item)
            self.field_item = None
        if self.slice_plane_item is not None:
            self.view.removeItem(self.slice_plane_item)
            self.slice_plane_item = None
        # Compute simple Lambertian shading for better shape perception.
        normals = self._compute_vertex_normals(verts, faces)
        if self.light_pos is not None:
            vec = self.light_pos[None, :] - verts
            vec_norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
            light_dir = vec / vec_norm
        else:
            light_dir = np.array([[0.3, 0.3, 0.9]], dtype=np.float32)
            light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-9)
        intensity = np.clip((normals * light_dir).sum(axis=1, keepdims=True) * 0.7 + 0.3, 0.1, 1.0)
        if use_vertex_colors and colors is not None:
            base_color = colors
            if base_color.shape[1] == 3:
                base_color = np.concatenate([base_color, np.ones((base_color.shape[0], 1), dtype=np.float32)], axis=1)
            shaded = np.clip(base_color * intensity, 0.0, 1.0)
        else:
            base_color = np.array([0.7, 0.8, 0.9], dtype=np.float32)
            shaded = np.clip(base_color * intensity, 0.0, 1.0)
            shaded = np.concatenate([shaded, np.ones((shaded.shape[0], 1), dtype=np.float32)], axis=1)
        colors_rgba = shaded

        mesh_kwargs = dict(
            vertexes=verts,
            faces=faces,
            smooth=True,
            drawEdges=True,
            edgeColor=(0.2, 0.2, 0.2, 0.6),
            computeNormals=False,
            vertexColors=colors_rgba,
            glOptions="opaque",
        )
        self.mesh_item = self.gl.GLMeshItem(**mesh_kwargs)
        self.view.addItem(self.mesh_item)
        self.process_events()

    def update_field_points(self, xyz: np.ndarray, rgba: np.ndarray):
        """Display a colored point cloud (for ESDF/TSDF/occupancy/pointcloud)."""
        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)
            self.mesh_item = None
        if self.slice_plane_item is not None:
            self.view.removeItem(self.slice_plane_item)
            self.slice_plane_item = None
        if self.field_item is None:
            self.field_item = self.gl.GLScatterPlotItem(pos=xyz, color=rgba, size=2.0, pxMode=True)
            self.view.addItem(self.field_item)
        else:
            self.field_item.setData(pos=xyz, color=rgba, size=2.0, pxMode=True)
        self.process_events()

    def update_slice_plane(
        self,
        origin: np.ndarray,
        forward_hint: np.ndarray,
        *,
        normal: np.ndarray,
        ahead_m: float,
        behind_m: float,
        half_width_m: float,
    ):
        """Draw a semi-transparent plane showing where ESDF/TSDF is queried."""
        if self.slice_plane_item is not None:
            self.view.removeItem(self.slice_plane_item)
            self.slice_plane_item = None

        f, r, _n_u = _plane_basis(forward_hint, normal=normal)
        o = origin.astype(np.float32)
        p0 = o + (-behind_m) * f + (-half_width_m) * r
        p1 = o + (ahead_m) * f + (-half_width_m) * r
        p2 = o + (ahead_m) * f + (half_width_m) * r
        p3 = o + (-behind_m) * f + (half_width_m) * r
        verts = np.vstack([p0, p1, p2, p3]).astype(np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        colors = np.tile(np.array([[0.2, 0.2, 0.2, 0.25]], dtype=np.float32), (4, 1))
        self.slice_plane_item = self.gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            vertexColors=colors,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.4, 0.4, 0.4, 0.5),
            glOptions="translucent",
            computeNormals=True,
        )
        self.view.addItem(self.slice_plane_item)
        self.process_events()

    def exec(self):
        return self.app.exec()

    def update_pose_axes(self, pose: np.ndarray, axis_len: float = 0.2):
        # Clear existing
        for item in self.pose_items:
            self.view.removeItem(item)
        self.pose_items = []
        t = pose[:3, 3]
        R = pose[:3, :3]
        axes = [
            ((1.0, 0.0, 0.0, 1.0), R @ (axis_len * np.array([1.0, 0.0, 0.0]))),  # X red
            ((0.0, 1.0, 0.0, 1.0), R @ (axis_len * np.array([0.0, 1.0, 0.0]))),  # Y green
            ((0.0, 0.0, 1.0, 1.0), R @ (axis_len * np.array([0.0, 0.0, 1.0]))),  # Z blue
        ]
        for color, direction in axes:
            pos = np.vstack([t, t + direction])
            item = self.gl.GLLinePlotItem(pos=pos, color=color, width=3, antialias=True)
            self.view.addItem(item)
            self.pose_items.append(item)

    def update_path(self, points: np.ndarray):
        if points.shape[0] < 2:
            return
        if self.path_item is None:
            # z=inf forces it to render on top (pyqtgraph convention).
            self.path_item = self.gl.GLLinePlotItem(
                pos=points,
                color=(1.0, 0.0, 0.0, 1.0),
                width=2,
                antialias=True,
                glOptions="opaque",
            )
            self.view.addItem(self.path_item)
        else:
            self.path_item.setData(pos=points)
        self.process_events()


def _colormap_plasma01(x: np.ndarray) -> np.ndarray:
    """Approximate 'plasma' colormap for x in [0, 1]. Returns RGB in [0,1]."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    # Anchor points sampled from a plasma-like palette.
    xp = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    cp = np.array(
        [
            [0.05, 0.03, 0.53],
            [0.45, 0.12, 0.68],
            [0.75, 0.27, 0.50],
            [0.95, 0.53, 0.25],
            [0.99, 0.91, 0.14],
        ],
        dtype=np.float32,
    )
    rgb = np.stack([np.interp(x, xp, cp[:, 0]), np.interp(x, xp, cp[:, 1]), np.interp(x, xp, cp[:, 2])], axis=1)
    return rgb


def _make_xy_slice(bounds_min: np.ndarray, bounds_max: np.ndarray, z: float, step: float) -> np.ndarray:
    x0, y0 = float(bounds_min[0]), float(bounds_min[1])
    x1, y1 = float(bounds_max[0]), float(bounds_max[1])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3), dtype=np.float32)
    xs = np.arange(x0, x1, step, dtype=np.float32)
    ys = np.arange(y0, y1, step, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    zg = np.full_like(xg, float(z), dtype=np.float32)
    pts = np.stack([xg, yg, zg], axis=-1).reshape(-1, 3)
    return pts


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _project_to_plane(v: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    return v - n_unit * float(np.dot(v, n_unit))


def _ground_basis(forward_hint: np.ndarray, *, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (forward, right, up_unit) for a ground-parallel plane."""
    up_u = _normalize(up.astype(np.float32))
    f = _project_to_plane(forward_hint.astype(np.float32), up_u)
    f = _normalize(f)
    if float(np.linalg.norm(f)) < 1e-6:
        f = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    r = _normalize(np.cross(up_u, f))
    return f, r, up_u


def _plane_basis(forward_hint: np.ndarray, *, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (forward, right, normal_unit) for an arbitrary plane."""
    n_u = _normalize(normal.astype(np.float32))
    f = _project_to_plane(forward_hint.astype(np.float32), n_u)
    f = _normalize(f)
    if float(np.linalg.norm(f)) < 1e-6:
        # Choose a stable fallback not parallel to the plane normal.
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, n_u))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        f = _normalize(_project_to_plane(fallback, n_u))
    r = _normalize(np.cross(n_u, f))
    return f, r, n_u


def _make_ground_aligned_slice(
    origin: np.ndarray,
    forward_hint: np.ndarray,
    *,
    up: np.ndarray,
    ahead_m: float,
    behind_m: float,
    half_width_m: float,
    step_m: float,
) -> np.ndarray:
    """Make a horizontal slice (ground-parallel) rotated to match a forward direction.

    Plane is parallel to the ground (normal=up). Axes are (forward,right) on that plane.
    """
    f, r, _up_u = _ground_basis(forward_hint, up=up)

    xs = np.arange(-behind_m, ahead_m, step_m, dtype=np.float32)
    ys = np.arange(-half_width_m, half_width_m, step_m, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")

    origin = origin.astype(np.float32)
    pts = origin[None, None, :] + xg[..., None] * f[None, None, :] + yg[..., None] * r[None, None, :]
    return pts.reshape(-1, 3).astype(np.float32)


def _downsample_points(xyz: np.ndarray, rgba: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= max_points:
        return xyz, rgba
    idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx], rgba[idx]


def _build_cube_mesh(centers: np.ndarray, voxel_size: float, colors: np.ndarray | None):
    if centers.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32), None

    centers = centers.astype(np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        centers = centers.reshape(-1, 3)

    hs = float(voxel_size) * 0.5
    offsets = np.array(
        [
            [-hs, -hs, -hs],
            [hs, -hs, -hs],
            [hs, hs, -hs],
            [-hs, hs, -hs],
            [-hs, -hs, hs],
            [hs, -hs, hs],
            [hs, hs, hs],
            [-hs, hs, hs],
        ],
        dtype=np.float32,
    )
    base_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )

    n = centers.shape[0]
    verts = centers[:, None, :] + offsets[None, :, :]
    verts = verts.reshape(-1, 3)

    faces = np.tile(base_faces, (n, 1))
    face_offsets = (np.arange(n, dtype=np.int32) * 8).repeat(12).reshape(-1, 1)
    faces = faces + face_offsets

    vcolors = None
    if colors is not None:
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.float32)], axis=1)
        vcolors = np.repeat(colors.astype(np.float32), 8, axis=0)
    return verts.astype(np.float32), faces.astype(np.int32), vcolors


def _depth_to_pointcloud(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    rgb_uint8: np.ndarray,
    *,
    stride: int,
    max_depth_m: float | None,
    pose_w_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    xv, yv = np.meshgrid(xs, ys)
    z = depth_m[yv, xv].astype(np.float32)
    valid = z > 0
    if max_depth_m is not None:
        valid &= z <= float(max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x = (xv[valid].astype(np.float32) - cx) * z[valid] / fx
    y = (yv[valid].astype(np.float32) - cy) * z[valid] / fy
    xyz_cam = np.stack([x, y, z[valid]], axis=1)

    rot = pose_w_c[:3, :3].astype(np.float32)
    trans = pose_w_c[:3, 3].astype(np.float32)
    xyz = (rot @ xyz_cam.T).T + trans[None, :]

    rgb = rgb_uint8[yv[valid], xv[valid]].astype(np.float32) / 255.0
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
    return xyz.astype(np.float32), rgba.astype(np.float32)




def _make_plane_slice(
    origin: np.ndarray,
    forward_hint: np.ndarray,
    *,
    normal: np.ndarray,
    ahead_m: float,
    behind_m: float,
    half_width_m: float,
    step_m: float,
) -> np.ndarray:
    """Make a planar slice (through origin) rotated to match a forward direction on that plane."""
    f, r, _n_u = _plane_basis(forward_hint, normal=normal)
    xs = np.arange(-behind_m, ahead_m, step_m, dtype=np.float32)
    ys = np.arange(-half_width_m, half_width_m, step_m, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    origin = origin.astype(np.float32)
    pts = origin[None, None, :] + xg[..., None] * f[None, None, :] + yg[..., None] * r[None, None, :]
    return pts.reshape(-1, 3).astype(np.float32)


def _read_tum_trajectory(path: Path) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Return mapping timestamp -> (t_xyz, q_xyzw) from a TUM file."""
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        ts = float(parts[0])
        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:8])
        out[ts] = (
            np.array([tx, ty, tz], dtype=np.float32),
            np.array([qx, qy, qz, qw], dtype=np.float32),
        )
    if not out:
        raise ValueError(f"No poses parsed from {path}")
    return out


def _build_pose_lookup(traj: Dict[float, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    ts_sorted = np.array(sorted(traj.keys()), dtype=np.float64)
    return ts_sorted, traj


def _lookup_pose(ts: float, ts_sorted: np.ndarray, traj: Dict[float, Tuple[np.ndarray, np.ndarray]], tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Find pose for timestamp; use nearest neighbor within tol seconds."""
    if ts in traj:
        return traj[ts]
    idx = np.searchsorted(ts_sorted, ts)
    candidates = []
    if idx > 0:
        candidates.append(ts_sorted[idx - 1])
    if idx < len(ts_sorted):
        candidates.append(ts_sorted[idx])
    if not candidates:
        raise KeyError(f"No pose for timestamp {ts}")
    nearest = min(candidates, key=lambda t: abs(t - ts))
    if abs(nearest - ts) > tol:
        raise KeyError(f"No pose for timestamp {ts} (nearest={nearest}, tol={tol})")
    return traj[float(nearest)]


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _make_pose_matrix(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Build 4x4 camera pose matrix from translation + quaternion.

    Assumes the trajectory provides camera-in-world (T_W_C). If your output looks
    wrong, try inverting this matrix before feeding it to nvblox.
    """
    r = _quat_xyzw_to_rotmat(q_xyzw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = r
    pose[:3, 3] = t
    return pose


def _pose_changed(prev_pose: np.ndarray | None, pose: np.ndarray, pos_eps: float = 1e-4, rot_eps_deg: float = 0.1) -> bool:
    """Check if pose changed beyond thresholds."""
    if prev_pose is None:
        return True
    dp = np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3])
    r_rel = pose[:3, :3] @ prev_pose[:3, :3].T
    cos_theta = np.clip((np.trace(r_rel) - 1) * 0.5, -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cos_theta))
    return dp > pos_eps or ang_deg > rot_eps_deg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path, required=True, help="RGB frames folder (e.g., iphone_mono)")
    parser.add_argument("--depth-dir", type=Path, required=True, help="Depth frames folder aligned to RGB")
    parser.add_argument("--calibration", type=Path, required=True, help="Calibration YAML with K matrix")
    parser.add_argument("--poses", type=Path, required=True, help="TUM pose file (timestamp tx ty tz qx qy qz qw)")
    parser.add_argument("--poses-compare", type=Path, help="Optional second TUM pose file to visualize for comparison (e.g., odometry vs SLAM)")
    parser.add_argument("--timestamps", type=Path, required=True, help="timestamps.txt with frame,timestamp_ns")
    parser.add_argument("--voxel_size_m", type=float, default=0.03)
    parser.add_argument("--max_integration_distance_m", type=float, default=5.0)
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Meters per depth unit (uint16 mm => 0.001)")
    parser.add_argument("--mesh_every", type=int, default=50, help="Update mesh every N frames")
    parser.add_argument("--out_dir", type=Path, help="Output folder (default: <rgb_dir>/../nvblox_out)")
    parser.add_argument("--invert_pose", action="store_true", help="Invert poses before integration (if your trajectory is T_C_W)")
    parser.add_argument("--ui", action="store_true", help="Show live UI (RGB + depth frames + 3D view)")
    parser.add_argument(
        "--mode",
        choices=["colormesh", "solidmesh", "esdf", "tsdf", "pointcloud"],
        default="colormesh",
        help="What to visualize in the UI",
    )
    parser.add_argument(
        "--field_step_m",
        type=float,
        default=0.0,
        help="ESDF/TSDF slice sampling step in meters (0 => use 2*voxel_size_m). Smaller => higher resolution, slower.",
    )
    parser.add_argument(
        "--voxel_band_m",
        type=float,
        default=0.03,
        help="Voxel mode: show voxels with |tsdf| < this band (meters).",
    )
    parser.add_argument(
        "--voxel_radius_m",
        type=float,
        default=6.0,
        help="Voxel mode: only visualize voxels within this radius of the current pose (meters).",
    )
    parser.add_argument(
        "--voxel_max_points",
        type=int,
        default=50000,
        help="Voxel mode: downsample to at most this many points for UI performance.",
    )
    parser.add_argument(
        "--cube",
        action="store_true",
        help="TSDF mode: render voxels as cubes instead of points (slower).",
    )
    parser.add_argument(
        "--pointcloud_stride",
        type=int,
        default=4,
        help="Pointcloud mode: pixel stride for depth projection (larger = fewer points).",
    )
    parser.add_argument(
        "--pointcloud_max_points",
        type=int,
        default=60000,
        help="Pointcloud mode: max points sent to the UI per frame.",
    )

    args = parser.parse_args()

    rgb_dir = args.rgb_dir.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    calib_path = args.calibration.expanduser().resolve()
    poses_path = args.poses.expanduser().resolve()
    ts_path = args.timestamps.expanduser().resolve()
    dataset_root = rgb_dir.parent
    out_dir = (args.out_dir or (dataset_root / "nvblox_out")).resolve()

    for path, desc in [
        (rgb_dir, "RGB dir"),
        (depth_dir, "Depth dir"),
        (calib_path, "Calibration"),
        (poses_path, "Poses file"),
        (ts_path, "Timestamps"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    timestamps = _parse_timestamps(ts_path)
    first_frame = timestamps[0][0]
    rgb_ext, depth_ext = _detect_exts(rgb_dir, depth_dir, first_frame)

    sample_rgb_path = _find_frame(rgb_dir, first_frame, [rgb_ext])
    sample_rgb = cv2.imread(str(sample_rgb_path), cv2.IMREAD_COLOR)
    if sample_rgb is None:
        raise RuntimeError(f"Failed to read sample rgb: {sample_rgb_path}")
    h, w = sample_rgb.shape[:2]
    fx, fy, cx, cy = _parse_k(calib_path)
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    traj = _read_tum_trajectory(poses_path)
    ts_sorted, traj_map = _build_pose_lookup(traj)

    # Optional comparison trajectory (e.g., odometry vs SLAM)
    traj_compare = None
    ts_sorted_compare = None
    traj_map_compare = None
    if args.poses_compare and args.poses_compare.exists():
        traj_compare = _read_tum_trajectory(args.poses_compare.expanduser().resolve())
        ts_sorted_compare, traj_map_compare = _build_pose_lookup(traj_compare)

    if not torch.cuda.is_available():
        raise RuntimeError("nvblox_torch requires a CUDA-capable GPU. torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"RGB ext: {rgb_ext}, depth ext: {depth_ext}, image size: {w}x{h}")

    # Import nvblox_torch lazily so the script can be inspected without it.
    # Note: Different nvblox_torch versions expose classes from different modules.
    try:
        from nvblox_torch.mapper import Mapper  # type: ignore
        from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams  # type: ignore
        from nvblox_torch.mapper import QueryType  # type: ignore
        from nvblox_torch.constants import constants  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Failed to import nvblox_torch Mapper APIs. "
            "Make sure you installed the correct nvblox_torch wheel for your Ubuntu/CUDA version. "
            "This script expects Mapper to be available as nvblox_torch.mapper.Mapper."
        ) from exc

    projective_params = ProjectiveIntegratorParams()
    projective_params.projective_integrator_max_integration_distance_m = float(args.max_integration_distance_m)
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_params)

    mapper = Mapper(
        voxel_sizes_m=float(args.voxel_size_m),
        mapper_parameters=mapper_params,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    rerun_enabled = False
    rr = None  # type: ignore
    def _rr_set_time(frame_idx: int, ts_sec: float) -> None:
        if rr is None:
            return
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("time", timestamp=ts_sec)
        else:
            # Fallback for older rerun SDKs.
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("frame", frame_idx)
            if hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("time", ts_sec)
    if args.ui:
        try:
            import rerun as rr  # type: ignore
            rerun_enabled = True
            rr.init("nvblox", spawn=True)
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log(
                "world/camera",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    width=w,
                    height=h,
                ),
            )
        except Exception as exc:
            raise RuntimeError("Requested --ui but failed to import/init rerun (pip install rerun-sdk).") from exc

    viewer: QtMeshViewer | None = None  # Qt UI disabled; using rerun when --ui
    prev_pose: np.ndarray | None = None
    path_points: list[np.ndarray] = []
    path_points_compare: list[np.ndarray] = []  # comparison trajectory (e.g., odometry)
    recent_points: list[np.ndarray] = []
    last_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    last_forward_on_plane = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if args.mode == "esdf":
        if not hasattr(mapper, "query_differentiable_layer"):
            raise RuntimeError("This nvblox_torch build does not expose query_differentiable_layer; cannot visualize ESDF.")
        if not hasattr(mapper, "update_esdf"):
            raise RuntimeError("This nvblox_torch build does not expose update_esdf; cannot visualize ESDF.")
    if args.mode == "tsdf":
        if not hasattr(mapper, "tsdf_layer_view"):
            raise RuntimeError("This nvblox_torch build does not expose tsdf_layer_view; cannot visualize voxels.")
    # Use median frame spacing as tolerance for nearest pose lookup.
    if len(ts_sorted) > 1:
        tol = float(np.median(np.diff(ts_sorted)) * 0.51)
    else:
        tol = 1e-3

    for idx, (frame_id, ts_sec) in enumerate(timestamps):
        rgb_path = rgb_dir / f"{frame_id}{rgb_ext}"
        depth_path = depth_dir / f"{frame_id}{depth_ext}"
        if not rgb_path.exists() or not depth_path.exists():
            print(f"Skipping frame {frame_id}: missing {rgb_path if not rgb_path.exists() else depth_path}")
            continue

        try:
            t, q = _lookup_pose(ts_sec, ts_sorted, traj_map, tol)
        except KeyError as exc:
            print(f"Skipping frame {frame_id} (t={ts_sec:.6f}): {exc}")
            continue

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read rgb: {rgb_path}")

        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")
        if depth_u16.ndim != 2 or depth_u16.dtype != np.uint16:
            raise ValueError(f"Expected uint16 single-channel depth, got {depth_u16.shape} {depth_u16.dtype} at {depth_path}")

        # Convert to meters float32 for nvblox.
        depth_m = depth_u16.astype(np.float32) * float(args.depth_scale)

        # nvblox_torch color expects RGB uint8 with 3 channels on GPU.
        rgb_uint8 = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        pose = _make_pose_matrix(t, q)
        if args.invert_pose:
            pose = np.linalg.inv(pose)

        # Always skip frames with unchanged poses (common in phone logs during tracking stalls).
        if not _pose_changed(prev_pose, pose):
            print(f"Skipped frame {frame_id} (t={ts_sec:.6f}) - static pose")
            continue
        prev_pose = pose
        path_points.append(pose[:3, 3].copy())
        recent_points.append(pose[:3, 3].copy())

        # Build comparison path if available
        if traj_map_compare is not None:
            try:
                t_cmp, q_cmp = _lookup_pose(ts_sec, ts_sorted_compare, traj_map_compare, tol)
                pose_cmp = _make_pose_matrix(t_cmp, q_cmp)
                if args.invert_pose:
                    pose_cmp = np.linalg.inv(pose_cmp)
                path_points_compare.append(pose_cmp[:3, 3].copy())
            except KeyError:
                pass  # No matching pose in comparison trajectory
        if len(recent_points) > 200:
            recent_points.pop(0)

        # Estimate a best-fit plane normal from recent trajectory points (PCA).
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if len(recent_points) >= 3:
            pts = np.vstack(recent_points).astype(np.float32)
            center = pts.mean(axis=0)
            X = pts - center[None, :]
            cov = (X.T @ X) / max(1, X.shape[0] - 1)
            evals, evecs = np.linalg.eigh(cov)
            normal = evecs[:, 0].astype(np.float32)  # smallest eigenvalue
            if float(np.dot(normal, world_up)) < 0:
                normal = -normal
            if float(np.linalg.norm(normal)) < 1e-6:
                normal = world_up
            last_plane_normal = _normalize(normal).astype(np.float32)
        else:
            last_plane_normal = world_up

        # Egocentric forward: camera +Z projected onto the estimated plane.
        cam_forward_w = pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fwd = _normalize(_project_to_plane(cam_forward_w, _normalize(last_plane_normal)))
        if float(np.linalg.norm(fwd)) > 1e-6:
            last_forward_on_plane = fwd.astype(np.float32)

        # Mapper expects depth/color on GPU, intrinsics and pose on CPU.
        depth_t = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)
        rgb_t = torch.from_numpy(rgb_uint8).to(device=device, dtype=torch.uint8)
        pose_t = torch.from_numpy(pose).to(device="cpu", dtype=torch.float32)
        intrinsics_t = torch.from_numpy(intrinsics).to(device="cpu", dtype=torch.float32)

        t_int0 = time.perf_counter()
        mapper.add_depth_frame(depth_t, pose_t, intrinsics_t)
        mapper.add_color_frame(rgb_t, pose_t, intrinsics_t)
        dt_int_ms = (time.perf_counter() - t_int0) * 1000.0

        if rerun_enabled:
            _rr_set_time(idx, ts_sec)
            rr.log(
                "world/camera_pose",
                rr.Transform3D(
                    translation=pose[:3, 3].tolist(),
                    rotation=pose[:3, :3].tolist(),
                ),
            )
            if path_points:
                rr.log("world/path_slam", rr.LineStrips3D([path_points], colors=[[0, 255, 0]]))  # green = SLAM/primary
            if path_points_compare:
                rr.log("world/path_odom", rr.LineStrips3D([path_points_compare], colors=[[255, 128, 0]]))  # orange = odometry/compare
            rr.log("world/rgb", rr.Image(rgb_uint8))
            rr.log("world/depth", rr.DepthImage(depth_m.astype(np.float32), meter=1.0))

        if args.mesh_every > 0 and (idx % int(args.mesh_every) == 0):
            t0 = time.perf_counter()
            if args.mode in ("colormesh", "solidmesh"):
                mapper.update_color_mesh()
                mesh = mapper.get_color_mesh().to_open3d()
                if rerun_enabled:
                    verts = np.asarray(mesh.vertices, dtype=np.float32)
                    faces = np.asarray(mesh.triangles, dtype=np.int32)
                    vcolors = None
                    if mesh.has_vertex_colors():
                        vcolors = np.asarray(mesh.vertex_colors, dtype=np.float32)
                        if vcolors.max() > 1.0:
                            vcolors = vcolors / 255.0
                    rr.log(
                        "world/mesh",
                        rr.Mesh3D(
                            vertex_positions=verts,
                            triangle_indices=faces,
                            vertex_colors=vcolors if mesh.has_vertex_colors() else None,
                        ),
                    )
            elif args.mode == "esdf":
                mapper.update_esdf()
                qtype = QueryType.ESDF

                # Egocentric navigation-style slice aligned to camera heading on the ground plane.
                forward_hint = last_forward_on_plane
                step = float(args.field_step_m) if float(args.field_step_m) > 0 else float(args.voxel_size_m) * 2.0
                origin = pose[:3, 3].copy()
                ahead_m = 8.0
                behind_m = 2.0
                half_width_m = 4.0
                query_xyz = _make_plane_slice(
                    origin,
                    forward_hint,
                    normal=last_plane_normal,
                    ahead_m=ahead_m,
                    behind_m=behind_m,
                    half_width_m=half_width_m,
                    step_m=step,
                )
                if query_xyz.shape[0] > 0:
                    query_t = torch.from_numpy(query_xyz).to(device=device, dtype=torch.float32)
                    sdf = mapper.query_differentiable_layer(qtype, query_t).detach()
                    sdf_np = sdf.float().reshape(-1).cpu().numpy()

                    unknown = float(constants.esdf_unknown_distance())
                    valid = sdf_np != unknown
                    sdf_vis = np.clip(sdf_np[valid], 0.0, 1.0)
                    colors_rgb = _colormap_plasma01((sdf_vis - 0.0) / (1.0 - 0.0))
                    xyz_vis = query_xyz[valid]

                    rgba = np.concatenate([colors_rgb, np.ones((colors_rgb.shape[0], 1), dtype=np.float32)], axis=1)
                    if rerun_enabled:
                        xyz_vis, rgba = _downsample_points(
                            xyz_vis.astype(np.float32), rgba.astype(np.float32), int(args.voxel_max_points)
                        )
                        rr.log("world/esdf_points", rr.Points3D(positions=xyz_vis, colors=rgba))
            elif args.mode == "tsdf":
                # Voxel mode: visualize a local band of TSDF voxels near the surface.
                tsdf_layer = mapper.tsdf_layer_view()
                if not hasattr(tsdf_layer, "get_all_blocks"):
                    raise RuntimeError("This nvblox_torch build does not expose TsdfLayer.get_all_blocks; cannot visualize voxels.")

                blocks, indices = tsdf_layer.get_all_blocks()
                block_dim = int(tsdf_layer.block_dim_in_voxels)
                voxel_size = float(tsdf_layer.voxel_size())
                band = float(args.voxel_band_m)
                radius = float(args.voxel_radius_m)
                radius2 = radius * radius

                pose_pos_t = torch.tensor(pose[:3, 3], device="cuda", dtype=torch.float32)

                # Indices returned by nvblox_torch are typically List[torch.Tensor] (each shape (3,)).
                if isinstance(indices, torch.Tensor):
                    idxs = indices.to(device="cuda", dtype=torch.int32)
                elif isinstance(indices, (list, tuple)) and (len(indices) == 0 or isinstance(indices[0], torch.Tensor)):
                    idxs = torch.stack(
                        [i.to(device="cuda", dtype=torch.int32) for i in indices],
                        dim=0,
                    ) if len(indices) > 0 else torch.empty((0, 3), device="cuda", dtype=torch.int32)
                else:
                    idxs = torch.tensor(indices, device="cuda", dtype=torch.int32)

                # Blocks returned by nvblox_torch are typically List[torch.Tensor].
                blocks_iter = blocks if isinstance(blocks, (list, tuple)) else [b for b in blocks]

                pts_accum: list[torch.Tensor] = []
                val_accum: list[torch.Tensor] = []

                block_size_m = block_dim * voxel_size
                block_radius_m = (3.0 ** 0.5) * 0.5 * block_size_m
                max_block_center_dist2 = (radius + block_radius_m) ** 2

                for block, bidx in zip(blocks_iter, idxs):
                    block = block.to(device="cuda")
                    if block.shape[-1] < 2:
                        continue

                    bidx_f = bidx.to(torch.float32)
                    block_center_vox = bidx_f * block_dim + (block_dim * 0.5)
                    block_center_m = (block_center_vox + 0.5) * voxel_size
                    if torch.sum((block_center_m - pose_pos_t) ** 2).item() > max_block_center_dist2:
                        continue

                    tsdf = block[..., 0]
                    w = block[..., 1]
                    mask = (w > 0) & (torch.abs(tsdf) < band)
                    if not torch.any(mask):
                        continue

                    coords = torch.nonzero(mask, as_tuple=False).to(torch.float32)  # (N,3)
                    gv = bidx_f[None, :] * block_dim + coords + 0.5
                    centers = gv * voxel_size
                    d2 = torch.sum((centers - pose_pos_t[None, :]) ** 2, dim=1)
                    keep = d2 < radius2
                    if not torch.any(keep):
                        continue

                    centers = centers[keep]
                    vals = tsdf[mask][keep].reshape(-1)
                    pts_accum.append(centers)
                    val_accum.append(vals.to(torch.float32))

                if pts_accum:
                    xyz_t = torch.cat(pts_accum, dim=0)
                    tsdf_t = torch.cat(val_accum, dim=0)
                    d2 = torch.sum((xyz_t - pose_pos_t[None, :]) ** 2, dim=1)
                    keep = d2 < radius2
                    xyz_t = xyz_t[keep]
                    tsdf_t = tsdf_t[keep]

                    # Simple occupancy coloring: inside=orange, outside=purple.
                    rgb = torch.zeros((xyz_t.shape[0], 3), device="cuda", dtype=torch.float32)
                    rgb[:, 0] = torch.where(tsdf_t < 0, 1.0, 0.3)
                    rgb[:, 1] = torch.where(tsdf_t < 0, 0.5, 0.0)
                    rgb[:, 2] = torch.where(tsdf_t < 0, 0.1, 1.0)
                    rgba_t = torch.cat([rgb, torch.ones((rgb.shape[0], 1), device="cuda", dtype=torch.float32)], dim=1)
                    xyz = xyz_t.detach().cpu().numpy().astype(np.float32)
                    rgba = rgba_t.detach().cpu().numpy().astype(np.float32)
                    xyz, rgba = _downsample_points(xyz, rgba, int(args.voxel_max_points))
                    if rerun_enabled:
                        if args.cube:
                            voxel_size = float(tsdf_layer.voxel_size())
                            verts, faces, vcolors = _build_cube_mesh(xyz, voxel_size, rgba)
                            rr.log(
                                "world/tsdf_mesh",
                                rr.Mesh3D(
                                    vertex_positions=verts,
                                    triangle_indices=faces,
                                    vertex_colors=vcolors,
                                ),
                            )
                        else:
                            rr.log("world/tsdf_points", rr.Points3D(positions=xyz, colors=rgba))
            dt_mesh = (time.perf_counter() - t0) * 1000.0
            print(f"Integrated frame {frame_id} (t={ts_sec:.6f}) - integ {dt_int_ms:.1f} ms, vis {dt_mesh:.1f} ms")
        else:
            print(f"Integrated frame {frame_id} (t={ts_sec:.6f}) - integ {dt_int_ms:.1f} ms")

    # Final mesh update and export if available.
    mapper.update_color_mesh()
    final_mesh = mapper.get_color_mesh().to_open3d()
    if rerun_enabled and args.mode in ("colormesh", "solidmesh"):
        verts = np.asarray(final_mesh.vertices, dtype=np.float32)
        faces = np.asarray(final_mesh.triangles, dtype=np.int32)
        vcolors = None
        if final_mesh.has_vertex_colors():
            vcolors = np.asarray(final_mesh.vertex_colors, dtype=np.float32)
            if vcolors.max() > 1.0:
                vcolors = vcolors / 255.0
        rr.log(
            "world/mesh_final",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces,
                vertex_colors=vcolors if final_mesh.has_vertex_colors() else None,
            ),
        )

    # Export: API name differs across versions. Try mapper methods first, then fall back to ColorMesh.save.
    out_mesh = out_dir / "mesh.ply"
    exported = False
    for method_name in ("save_mesh", "export_mesh", "write_mesh"):
        fn = getattr(mapper, method_name, None)
        if callable(fn):
            fn(str(out_mesh))
            exported = True
            break

    if not exported:
        try:
            mesh_obj = mapper.get_color_mesh()
            # ColorMesh.save uses Open3D to write the mesh.
            mesh_obj.save(str(out_mesh))
            exported = True
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"Mesh export via ColorMesh.save failed: {exc}")

    if exported:
        print(f"Wrote mesh: {out_mesh}")
    else:
        print("Mesh export method not found on Mapper. You can still visualize or access mesh/layers via the nvblox_torch API.")

    out_tsdf = out_dir / "tsdf_voxel_grid.ply"
    did_export_tsdf = _export_tsdf_voxel_ply(
        mapper,
        out_tsdf,
        band_m=float(args.voxel_band_m),
        as_occupancy=False,
    )
    if did_export_tsdf:
        print(f"Wrote voxel grid: {out_tsdf}")

    out_occ = out_dir / "occupancy_voxel_grid.ply"
    did_export_occ = _export_tsdf_voxel_ply(
        mapper,
        out_occ,
        band_m=float(args.voxel_band_m),
        as_occupancy=True,
    )
    if did_export_occ:
        print(f"Wrote voxel grid: {out_occ}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
