#!/usr/bin/env python3
"""
Stream depth images as a 3D point cloud using Qt 3D.

Dependencies (install in your venv):
  python3 -m pip install PyQt5 PyQt3D opencv-python pyyaml numpy

Example:
  python3 utilities/depth_stream_pointcloud_qt.py \
    --depth-dir data/sample_20260119/iphone_mono_depth \
    --calibration data/sample_20260119/iphone_calibration.yaml \
    --depth-scale 1000 \
    --fps 30 \
    --ext .png \
    --stride 2 \
    --loop
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Configure Qt plugin path/platform before Qt instantiation to avoid cv2's plugin dir.
def _configure_qt_env() -> None:
    # Prefer PyQt's own plugin path over OpenCV's.
    try:
        from PyQt5.QtCore import QLibraryInfo
        plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        if plugin_dir:
            os.environ["QT_PLUGIN_PATH"] = plugin_dir
    except Exception:
        pass

    if "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
        else:
            os.environ["QT_QPA_PLATFORM"] = "xcb"


_configure_qt_env()

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets  # type: ignore
from PyQt5 import Qt3DCore, Qt3DExtras, Qt3DRender  # type: ignore


def find_depth_files(depth_dir: Path, extensions: Iterable[str]) -> List[Path]:
    exts = {ext.lower() for ext in extensions}
    files = [p for p in sorted(depth_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]
    if not files:
        raise SystemExit(f"No depth images found in {depth_dir} with extensions {sorted(exts)}")
    return files


def parse_intrinsics(calib_path: Optional[Path]) -> Optional[Tuple[float, float, float, float]]:
    if calib_path is None:
        return None
    data = yaml.safe_load(calib_path.read_text())
    if not isinstance(data, dict):
        return None

    def flatten_k(val) -> Optional[List[float]]:
        if isinstance(val, dict):
            arr = val.get("data") or val.get("Data") or val.get("values")
            return arr if isinstance(arr, list) else None
        if isinstance(val, list):
            flat: List[float] = []
            for row in val:
                if isinstance(row, list):
                    flat.extend(row)
                else:
                    flat.append(row)
            return flat
        return None

    k = None
    if "K" in data:
        k = flatten_k(data["K"])
    if k is None and "camera_matrix" in data:
        k = flatten_k(data["camera_matrix"])
    if k and len(k) >= 6:
        fx = float(k[0])
        fy = float(k[4]) if len(k) > 4 else float(k[1])
        cx = float(k[2])
        cy = float(k[5]) if len(k) > 5 else float(k[3])
        return fx, fy, cx, cy
    return None


def depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int) -> np.ndarray:
    h, w = depth_m.shape
    xs = np.arange(0, w, stride, dtype=np.float32)
    ys = np.arange(0, h, stride, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth_m[::stride, ::stride]
    mask = z > 0
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)
    z = z[mask]
    uu = uu[mask]
    vv = vv[mask]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack((x, y, z), axis=1).astype(np.float32)


class PointCloudEntity(Qt3DCore.QEntity):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.geometry = Qt3DRender.QGeometry(self)
        self.position_buffer = Qt3DRender.QBuffer(self.geometry)
        self.color_buffer = Qt3DRender.QBuffer(self.geometry)

        self.position_attribute = Qt3DRender.QAttribute()
        self.position_attribute.setName(Qt3DRender.QAttribute.defaultPositionAttributeName())
        self.position_attribute.setAttributeType(Qt3DRender.QAttribute.VertexAttribute)
        self.position_attribute.setVertexBaseType(Qt3DRender.QAttribute.Float)
        self.position_attribute.setVertexSize(3)
        self.position_attribute.setByteStride(12)
        self.position_attribute.setBuffer(self.position_buffer)

        self.color_attribute = Qt3DRender.QAttribute()
        self.color_attribute.setName(Qt3DRender.QAttribute.defaultColorAttributeName())
        self.color_attribute.setAttributeType(Qt3DRender.QAttribute.VertexAttribute)
        self.color_attribute.setVertexBaseType(Qt3DRender.QAttribute.Float)
        self.color_attribute.setVertexSize(3)
        self.color_attribute.setByteStride(12)
        self.color_attribute.setBuffer(self.color_buffer)

        self.geometry.addAttribute(self.position_attribute)
        self.geometry.addAttribute(self.color_attribute)

        self.renderer = Qt3DRender.QGeometryRenderer()
        self.renderer.setGeometry(self.geometry)
        self.renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.Points)

        self.material = Qt3DExtras.QPerVertexColorMaterial(self)

        self.addComponent(self.renderer)
        self.addComponent(self.material)

    def update_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        count = positions.shape[0]
        pos_bytes = QtCore.QByteArray(positions.tobytes())
        col_bytes = QtCore.QByteArray(colors.tobytes())
        self.position_buffer.setData(pos_bytes)
        self.color_buffer.setData(col_bytes)
        self.position_attribute.setCount(count)
        self.color_attribute.setCount(count)
        self.geometry.setBoundingVolumePositionAttribute(self.position_attribute)


class Viewer(QtWidgets.QWidget):
    def __init__(self, files: List[Path], fx: float, fy: float, cx: float, cy: float, depth_scale: float, stride: int, fps: float, loop: bool):
        super().__init__()
        self.files = files
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.stride = max(1, stride)
        self.loop = loop
        self.idx = 0

        self.view = Qt3DExtras.Qt3DWindow()
        container = QtWidgets.QWidget.createWindowContainer(self.view)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)
        self.setLayout(layout)

        self.root = Qt3DCore.QEntity()
        self.view.setRootEntity(self.root)

        self.point_entity = PointCloudEntity(self.root)

        cam = self.view.camera()
        cam.lens().setPerspectiveProjection(45.0, 16 / 9, 0.01, 1000.0)
        cam.setPosition(QtGui.QVector3D(0.0, 0.0, 1.5))
        cam.setViewCenter(QtGui.QVector3D(0.0, 0.0, 0.0))

        controller = Qt3DExtras.QOrbitCameraController(self.root)
        controller.setLinearSpeed(5.0)
        controller.setLookSpeed(180.0)
        controller.setCamera(cam)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        interval_ms = max(1, int(1000.0 / max(fps, 1e-3)))
        self.timer.start(interval_ms)

        self.setWindowTitle("Depth point cloud (Qt3D)")
        self.resize(1280, 720)

    def next_frame(self) -> None:
        if self.idx >= len(self.files):
            if self.loop:
                self.idx = 0
            else:
                self.timer.stop()
                return

        path = self.files[self.idx]
        self.idx += 1

        depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            return
        if depth_raw.dtype == np.uint16 or depth_raw.dtype == np.uint32:
            depth_m = depth_raw.astype(np.float32) / float(self.depth_scale)
        else:
            depth_m = depth_raw.astype(np.float32)

        pts = depth_to_points(depth_m, self.fx, self.fy, self.cx, self.cy, self.stride)
        if pts.shape[0] == 0:
            return

        depths = pts[:, 2]
        d_min = depths.min()
        d_range = depths.max() - d_min + 1e-6
        norm = (depths - d_min) / d_range
        colors = np.stack([1.0 - norm, norm, 0.5 * (1.0 - norm)], axis=1).astype(np.float32)

        self.point_entity.update_points(pts, colors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qt3D viewer for streaming depth images as point clouds")
    parser.add_argument("--depth-dir", required=True, type=Path, help="Directory containing depth images")
    parser.add_argument("--calibration", type=Path, help="Calibration YAML with K matrix")
    parser.add_argument("--fx", type=float, help="Override fx")
    parser.add_argument("--fy", type=float, help="Override fy")
    parser.add_argument("--cx", type=float, help="Override cx")
    parser.add_argument("--cy", type=float, help="Override cy")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale (value per meter, default 1000 for mm)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    parser.add_argument("--stride", type=int, default=1, help="Subsample pixels to lighten rendering (default 1)")
    parser.add_argument("--ext", default=".png", help="Comma-separated depth extensions (default: .png)")
    parser.add_argument("--loop", action="store_true", help="Loop the sequence")
    args = parser.parse_args()

    depth_dir = args.depth_dir.expanduser().resolve()
    if not depth_dir.is_dir():
        raise SystemExit(f"Depth directory not found: {depth_dir}")

    extensions = [ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}" for ext in args.ext.split(",")]
    files = find_depth_files(depth_dir, extensions)

    intrinsics = parse_intrinsics(args.calibration)
    if args.fx and args.fy and args.cx and args.cy:
        intrinsics = (args.fx, args.fy, args.cx, args.cy)
    if intrinsics is None:
        sample = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
        if sample is None:
            raise SystemExit(f"Failed to read sample depth: {files[0]}")
        h, w = sample.shape[:2]
        fx = fy = 0.5 * max(w, h)
        cx = w / 2.0
        cy = h / 2.0
        intrinsics = (fx, fy, cx, cy)
        print("No intrinsics provided; using approximate pinhole from image size.")

    fx, fy, cx, cy = intrinsics

    app = QtWidgets.QApplication([])
    viewer = Viewer(files, fx, fy, cx, cy, args.depth_scale, args.stride, args.fps, args.loop)
    viewer.show()
    app.exec_()


if __name__ == "__main__":
    main()
