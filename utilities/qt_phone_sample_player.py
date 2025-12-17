import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl


@dataclass
class FrameRecord:
    timestamp: float
    rgb_path: Path
    depth_color_path: Optional[Path]
    depth_metric_path: Optional[Path]
    translation: Optional[np.ndarray] = None  # shape (3,)
    quaternion: Optional[np.ndarray] = None  # shape (4,) as (x, y, z, w)


@dataclass
class FramePixmaps:
    rgb: Optional[QtGui.QPixmap]
    depth_color: Optional[QtGui.QPixmap]
    depth_metric: Optional[QtGui.QPixmap]


class PixmapCache:
    """Small LRU cache for frame pixmaps."""

    def __init__(self, max_items: int = 200):
        from collections import OrderedDict

        self._store = OrderedDict()
        self.max_items = max_items

    def get(self, idx: int, loader) -> FramePixmaps:
        store = self._store
        if idx in store:
            val = store.pop(idx)
            store[idx] = val
            return val
        val = loader(idx)
        store[idx] = val
        if len(store) > self.max_items:
            store.popitem(last=False)
        return val


class PreloaderThread(QtCore.QThread):
    frame_ready = QtCore.Signal(int, object)  # idx, FramePixmaps

    def __init__(self, records: List[FrameRecord], loader, parent=None):
        super().__init__(parent)
        self.records = records
        self.loader = loader
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        for idx in range(len(self.records)):
            if self._stop:
                break
            pix = self.loader(idx)
            self.frame_ready.emit(idx, pix)


def load_associations(base_dir: Path) -> List[FrameRecord]:
    assoc_file = base_dir / "associations.txt"
    if not assoc_file.exists():
        raise FileNotFoundError(f"Missing associations.txt in {base_dir}")

    records: List[FrameRecord] = []
    with assoc_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ts_rgb = float(parts[0])
            rgb_rel = parts[1]
            ts_depth = float(parts[2])
            depth_metric_rel = parts[3]
            rgb_path = base_dir / rgb_rel
            depth_metric_path = base_dir / depth_metric_rel
            if not depth_metric_path.exists():
                depth_metric_path = None

            # Relative-depth folder is typically "<rgb_folder>d" (e.g., frames_0002 -> frames_0002d)
            rgb_folder = Path(rgb_rel).parent.name
            depth_color_candidate = base_dir / f"{rgb_folder}d" / Path(rgb_rel).name
            depth_color_path = depth_color_candidate if depth_color_candidate.exists() else None

            records.append(
                FrameRecord(
                    timestamp=ts_rgb,
                    rgb_path=rgb_path,
                    depth_color_path=depth_color_path,
                    depth_metric_path=depth_metric_path,
                )
            )
    return records


def load_poses(base_dir: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    pose_file = base_dir / "CameraTrajectory.csv"
    if not pose_file.exists():
        return {}

    poses = {}
    with pose_file.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts = float(row["timestamp"])
            t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float32)
            q = np.array(
                [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                dtype=np.float32,
            )
            poses[ts] = (t, q)
    return poses


def infer_missing_frames(base_dir: Path, records: List[FrameRecord]) -> List[FrameRecord]:
    """Add placeholder records for RGB frames that are not listed in associations.txt."""
    if not records:
        return records

    rgb_dir = records[0].rgb_path.parent
    if not rgb_dir.exists():
        return records

    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() == ".png"])
    if not rgb_files:
        return records

    # Estimate frame period from existing timestamps
    ts_sorted = sorted([r.timestamp for r in records])
    if len(ts_sorted) > 1:
        diffs = np.diff(ts_sorted)
        frame_dt = float(np.median(diffs))
    else:
        frame_dt = 1.0 / 30.0

    name_to_record = {r.rgb_path.name: r for r in records}
    new_records = list(records)

    for rgb_path in rgb_files:
        name = rgb_path.name
        if name in name_to_record:
            continue

        # Infer index from filename (e.g., frame_000123.png)
        stem = rgb_path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            idx = int(digits)
            ts = idx * frame_dt
        else:
            ts = ts_sorted[-1] + frame_dt

        rgb_folder = rgb_path.parent.name
        depth_color_candidate = base_dir / f"{rgb_folder}d" / rgb_path.name
        depth_color_path = depth_color_candidate if depth_color_candidate.exists() else None

        depth_metric_candidate = base_dir / f"{rgb_folder}_metric_d" / rgb_path.name
        depth_metric_path = depth_metric_candidate if depth_metric_candidate.exists() else None

        new_records.append(
            FrameRecord(
                timestamp=ts,
                rgb_path=rgb_path,
                depth_color_path=depth_color_path,
                depth_metric_path=depth_metric_path,
            )
        )

    # Sort by timestamp to keep playback order consistent
    new_records.sort(key=lambda r: r.timestamp)
    return new_records


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rot


def img_to_qpixmap(img: np.ndarray) -> QtGui.QPixmap:
    """Convert a numpy image (H,W,3) in RGB or (H,W) grayscale to QPixmap."""
    img = np.ascontiguousarray(img)  # ensure C-contiguous for Qt image buffer
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        assert ch == 3
        qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def colorize_depth(depth_metric: np.ndarray) -> Optional[np.ndarray]:
    """Create an RGB visualization for a single-channel metric depth image."""
    if depth_metric is None:
        return None
    if depth_metric.ndim == 3 and depth_metric.shape[2] == 3:
        return cv2.cvtColor(depth_metric, cv2.COLOR_BGR2RGB)
    if depth_metric.ndim != 2:
        return None
    valid = depth_metric > 0
    if np.any(valid):
        vmin, vmax = depth_metric[valid].min(), depth_metric[valid].max()
    else:
        vmin, vmax = depth_metric.min(), depth_metric.max()
    if math.isclose(vmax, vmin):
        vmax = vmin + 1.0
    norm = np.clip((depth_metric - vmin) / (vmax - vmin), 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


class ImageView(QtWidgets.QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        self.setMinimumSize(320, 240)
        self.title = title
        self.setScaledContents(False)

    def update_image(self, pix: QtGui.QPixmap):
        if pix is None:
            self.setText(f"No image\n{self.title}")
        else:
            self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


class TrajectoryView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.opts["distance"] = 3
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        self.addItem(grid)
        self.traj_item = None
        self.axis_items = []

    def set_trajectory(self, positions: np.ndarray):
        if positions.size == 0:
            return
        if self.traj_item is None:
            self.traj_item = gl.GLLinePlotItem(pos=positions, color=(1.0, 1.0, 0.0, 1.0), width=2, antialias=True)
            self.addItem(self.traj_item)
        else:
            self.traj_item.setData(pos=positions)

    def update_camera(self, t: np.ndarray, q: np.ndarray, axis_len: float = 0.2):
        for item in self.axis_items:
            self.removeItem(item)
        self.axis_items = []
        rot = quaternion_to_matrix(q)
        axes = {
            (1.0, 0.0, 0.0, 1.0): rot @ (axis_len * np.array([1.0, 0.0, 0.0])),
            (0.0, 1.0, 0.0, 1.0): rot @ (axis_len * np.array([0.0, 1.0, 0.0])),
            (0.0, 0.5, 1.0, 1.0): rot @ (axis_len * np.array([0.0, 0.0, 1.0])),
        }
        for color, direction in axes.items():
            pos = np.vstack([t, t + direction])
            item = gl.GLLinePlotItem(pos=pos, color=color, width=3, antialias=True)
            self.addItem(item)
            self.axis_items.append(item)


class PlayerWindow(QtWidgets.QWidget):
    def __init__(self, records: List[FrameRecord]):
        super().__init__()
        self.records = records
        self.timestamps = np.array([r.timestamp for r in records], dtype=float)
        self.current_idx = 0
        self.playing = False
        self.time_scale = 1.0
        self.play_start_wall = None
        self.play_start_ts = None
        self.cache = PixmapCache(max_items=200)
        self.preloaded: List[Optional[FramePixmaps]] = [None] * len(records)
        self.preloader = PreloaderThread(records, self.load_pixmaps_for_index)
        self.preloader.frame_ready.connect(self.on_frame_preloaded)
        self.preloader.start()

        self.rgb_view = ImageView("RGB")
        self.depth_color_view = ImageView("Depth Color")
        self.depth_metric_view = ImageView("Metric Depth")
        self.traj_view = TrajectoryView()

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, len(records) - 1)
        self.slider.valueChanged.connect(self.on_slider_changed)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.on_play_toggled)

        self.speed_box = QtWidgets.QComboBox()
        for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
            self.speed_box.addItem(f"{s}x", s)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.currentIndexChanged.connect(self.on_speed_changed)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(0)  # let Qt drive as fast as possible; we skip via timestamps

        self._build_layout()
        self._precompute_traj()
        self.update_frame(0)

    def _build_layout(self):
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.rgb_view, 0, 0)
        grid.addWidget(self.depth_color_view, 0, 1)
        grid.addWidget(self.depth_metric_view, 0, 2)
        grid.addWidget(self.traj_view, 1, 0, 1, 3)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.play_btn)
        controls.addWidget(QtWidgets.QLabel("Frame"))
        controls.addWidget(self.slider, 1)
        controls.addWidget(QtWidgets.QLabel("Speed"))
        controls.addWidget(self.speed_box)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(grid)
        layout.addLayout(controls)
        self.setLayout(layout)
        self.setWindowTitle("Phone sample player")
        self.resize(1200, 800)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.preloader.isRunning():
            self.preloader.stop()
            self.preloader.wait(2000)
        return super().closeEvent(event)

    def _precompute_traj(self):
        positions = []
        for rec in self.records:
            if rec.translation is not None:
                positions.append(rec.translation)
        if positions:
            pos_arr = np.vstack(positions)
            self.traj_view.set_trajectory(pos_arr)

    def load_pixmaps_for_index(self, idx: int) -> FramePixmaps:
        # If already preloaded, return it
        existing = self.preloaded[idx] if idx < len(self.preloaded) else None
        if existing is not None:
            return existing

        rec = self.records[idx]
        rgb_pix = None
        depth_color_pix = None
        depth_metric_pix = None

        rgb = cv2.imread(str(rec.rgb_path), cv2.IMREAD_COLOR)
        if rgb is not None:
            rgb_pix = img_to_qpixmap(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        if rec.depth_color_path is not None and rec.depth_color_path.exists():
            depth_color = cv2.imread(str(rec.depth_color_path), cv2.IMREAD_COLOR)
            if depth_color is not None:
                depth_color_pix = img_to_qpixmap(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))

        if rec.depth_metric_path is not None and rec.depth_metric_path.exists():
            depth_metric = cv2.imread(str(rec.depth_metric_path), cv2.IMREAD_UNCHANGED)
            depth_metric_rgb = colorize_depth(depth_metric)
            if depth_metric_rgb is not None:
                depth_metric_pix = img_to_qpixmap(depth_metric_rgb)

        fp = FramePixmaps(rgb=rgb_pix, depth_color=depth_color_pix, depth_metric=depth_metric_pix)
        if idx < len(self.preloaded):
            self.preloaded[idx] = fp
        return fp

    def on_speed_changed(self, idx: int):
        self.time_scale = float(self.speed_box.currentData())
        if self.playing:
            self.play_start_wall = time.perf_counter()
            self.play_start_ts = self.timestamps[self.current_idx]

    def on_play_toggled(self, checked: bool):
        self.playing = checked
        self.play_btn.setText("Pause" if checked else "Play")
        if checked:
            self.play_start_wall = time.perf_counter()
            self.play_start_ts = self.timestamps[self.current_idx]

    def on_slider_changed(self, value: int):
        self.update_frame(value)

    def on_tick(self):
        if not self.playing or len(self.records) < 2:
            return
        now = time.perf_counter()
        elapsed = (now - self.play_start_wall) * self.time_scale
        target_ts = self.play_start_ts + elapsed

        idx = int(np.searchsorted(self.timestamps, target_ts, side="right") - 1)
        idx = max(0, min(idx, len(self.records) - 1))

        if idx != self.current_idx:
            self.update_frame(idx)
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.prefetch(idx)

        # Stop at end
        if idx >= len(self.records) - 1 and target_ts > self.timestamps[-1]:
            self.playing = False
            self.play_btn.setChecked(False)
            self.play_btn.setText("Play")

    def prefetch(self, idx: int, ahead: int = 10):
        # Preload a few upcoming frames into cache to reduce stalls
        for i in range(idx + 1, min(idx + 1 + ahead, len(self.records))):
            if i not in self.cache._store:  # direct access OK for hinting
                self.cache.get(i, self.load_pixmaps_for_index)

    @QtCore.Slot(int, object)
    def on_frame_preloaded(self, idx: int, pix: FramePixmaps):
        # Store in cache and preloaded list for instant access
        if idx < len(self.preloaded):
            self.preloaded[idx] = pix
        self.cache.get(idx, lambda _: pix)

    def update_frame(self, idx: int):
        idx = max(0, min(idx, len(self.records) - 1))
        self.current_idx = idx
        rec = self.records[idx]
        pix = self.cache.get(idx, self.load_pixmaps_for_index)

        self.rgb_view.update_image(pix.rgb)
        self.depth_color_view.update_image(pix.depth_color)
        self.depth_metric_view.update_image(pix.depth_metric)

        if rec.translation is not None and rec.quaternion is not None:
            self.traj_view.update_camera(rec.translation, rec.quaternion)


def build_records(base_dir: Path) -> List[FrameRecord]:
    records = load_associations(base_dir)
    records = infer_missing_frames(base_dir, records)
    poses = load_poses(base_dir)
    # Merge pose info into frames using exact timestamp match first, then nearest neighbor
    pose_times = sorted(poses.keys())
    for rec in records:
        if rec.timestamp in poses:
            t, q = poses[rec.timestamp]
        else:
            # nearest timestamp
            idx = np.searchsorted(pose_times, rec.timestamp)
            candidates = []
            if idx > 0:
                candidates.append(pose_times[idx - 1])
            if idx < len(pose_times):
                candidates.append(pose_times[idx])
            if not candidates:
                continue
            best_ts = min(candidates, key=lambda ts: abs(ts - rec.timestamp))
            t, q = poses[best_ts]
        rec.translation = t
        rec.quaternion = q / np.linalg.norm(q)
    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: python utilities/qt_phone_sample_player.py <path-to-phone_sample3>")
        sys.exit(1)
    base_dir = Path(sys.argv[1]).expanduser()
    if not base_dir.exists():
        print(f"Path not found: {base_dir}")
        sys.exit(1)

    records = build_records(base_dir)
    if not records:
        print("No frames found.")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    win = PlayerWindow(records)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
