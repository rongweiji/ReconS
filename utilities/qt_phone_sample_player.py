import csv
import os
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
    matched_pose_timestamp: Optional[float] = None
    matched_pose_abs_dt: Optional[float] = None
    pose_match_exact: bool = False
    inferred: bool = False


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
                inferred=True,
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
    def __init__(self, records: List[FrameRecord], alignment_report: Optional["AlignmentReport"] = None):
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

        small_font = QtGui.QFont()
        small_font.setPointSize(9)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setFont(small_font)
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.align_label = QtWidgets.QLabel("")
        self.align_label.setFont(small_font)
        self.align_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.check_align_btn = QtWidgets.QPushButton("Check alignment")
        self.check_align_btn.setFont(small_font)
        self.check_align_btn.setFixedHeight(22)
        self.check_align_btn.clicked.connect(self.on_check_alignment)

        self.alignment_report = alignment_report

        self._build_layout()
        self._precompute_traj()
        self.update_frame(0)

        # Print + show summary once on startup.
        self._apply_alignment_summary_to_ui(print_to_console=True)

    def _build_layout(self):
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.rgb_view, 0, 0)
        grid.addWidget(self.depth_color_view, 0, 1)
        grid.addWidget(self.depth_metric_view, 0, 2)
        grid.addWidget(self.traj_view, 1, 0, 1, 3)

        # Compact info panel directly under the 3D view
        info_panel = QtWidgets.QWidget()
        info_panel.setFixedHeight(24)
        footer = QtWidgets.QHBoxLayout(info_panel)
        footer.setContentsMargins(6, 0, 6, 0)
        footer.setSpacing(8)
        footer.addWidget(self.info_label, 1)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(6, 0, 6, 0)
        controls.setSpacing(8)
        controls.addWidget(self.play_btn)
        controls.addWidget(QtWidgets.QLabel("Frame"))
        controls.addWidget(self.slider, 1)
        controls.addWidget(QtWidgets.QLabel("Speed"))
        controls.addWidget(self.speed_box)
        controls.addStretch(1)
        controls.addWidget(self.align_label)
        controls.addWidget(self.check_align_btn)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(grid)
        layout.addWidget(info_panel)
        layout.addLayout(controls)
        self.setLayout(layout)
        self.setWindowTitle("Phone sample player")
        self.resize(1200, 800)

    def _apply_alignment_summary_to_ui(self, print_to_console: bool = False):
        rpt = self.alignment_report
        if rpt is None:
            self.align_label.setText("Alignment: (not checked)")
            return

        self.align_label.setText(rpt.summary_text())
        if print_to_console:
            print(rpt.to_console_report())

    def on_check_alignment(self):
        self._apply_alignment_summary_to_ui(print_to_console=True)

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

        # Per-frame status
        pose_state = "MISSING"
        pose_extra = ""
        if rec.translation is not None and rec.quaternion is not None:
            pose_state = "OK"
            if rec.matched_pose_abs_dt is not None:
                pose_extra = f" (Δt={rec.matched_pose_abs_dt * 1000.0:.1f}ms)"
        inferred = " inferred" if rec.inferred else ""
        self.info_label.setText(
            f"Frame {idx + 1}/{len(self.records)}  t={rec.timestamp:.6f}s{inferred}  pose={pose_state}{pose_extra}"
        )


@dataclass
class AlignmentReport:
    image_label: str
    pose_label: str
    image_count: int
    pose_count: int
    image_fps: Optional[float]
    pose_fps: Optional[float]
    image_dt_median: Optional[float]
    pose_dt_median: Optional[float]
    fps_diff_pct: Optional[float]
    match_tolerance_s: float
    matched: int
    unmatched: int
    abs_dt_median: Optional[float]
    abs_dt_p95: Optional[float]
    abs_dt_max: Optional[float]
    image_gaps: int
    pose_gaps: int
    missing_images_in_associations: int
    ok: bool
    warnings: List[str]

    def summary_text(self) -> str:
        if self.pose_count == 0:
            return "Alignment: NO POSES"
        if self.image_count == 0:
            return "Alignment: NO IMAGES"
        if self.ok:
            return "Alignment: OK"
        return "Alignment: ISSUES (see console)"

    def to_console_report(self) -> str:
        lines: List[str] = []
        lines.append("=== Timestamp Alignment Report ===")
        lines.append(f"Images: {self.image_label}")
        lines.append(f"Poses:  {self.pose_label}")
        lines.append(f"Counts: images={self.image_count}, poses={self.pose_count}")
        if self.image_fps is not None and self.pose_fps is not None:
            lines.append(f"FPS:    images≈{self.image_fps:.3f}, poses≈{self.pose_fps:.3f}")
        if self.fps_diff_pct is not None:
            lines.append(f"FPS Δ:  {self.fps_diff_pct:.2f}%")
        lines.append(f"Match tolerance: ±{self.match_tolerance_s * 1000.0:.1f}ms")
        lines.append(f"Matches: {self.matched}/{self.image_count}  (unmatched={self.unmatched})")
        if self.abs_dt_median is not None:
            lines.append(
                f"|Δt| stats: median={self.abs_dt_median * 1000.0:.2f}ms, "
                f"p95={self.abs_dt_p95 * 1000.0:.2f}ms, max={self.abs_dt_max * 1000.0:.2f}ms"
            )
        lines.append(f"Gaps:  images={self.image_gaps}, poses={self.pose_gaps}")
        if self.missing_images_in_associations:
            lines.append(f"Missing images in associations.txt: {self.missing_images_in_associations}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"- {w}")
        lines.append(f"Overall: {'OK' if self.ok else 'NOT OK'}")
        return "\n".join(lines)


def _robust_median_dt(timestamps: np.ndarray) -> Optional[float]:
    if timestamps.size < 2:
        return None
    ts = np.sort(timestamps.astype(float))
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _estimate_fps(timestamps: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    dt = _robust_median_dt(timestamps)
    if dt is None or dt <= 0:
        return None, dt
    return 1.0 / dt, dt


def _count_gaps(timestamps: np.ndarray, dt_median: Optional[float], gap_factor: float = 1.5) -> int:
    if dt_median is None or timestamps.size < 2:
        return 0
    ts = np.sort(timestamps.astype(float))
    diffs = np.diff(ts)
    return int(np.sum(diffs > gap_factor * dt_median))


def _nearest_abs_dts(query_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, compute abs(query - nearest(ref))."""
    if query_ts.size == 0 or ref_ts.size == 0:
        return np.array([], dtype=float)
    q = np.sort(query_ts.astype(float))
    r = np.sort(ref_ts.astype(float))
    idxs = np.searchsorted(r, q)
    out = np.empty_like(q, dtype=float)
    for i, (ts, idx) in enumerate(zip(q, idxs)):
        candidates = []
        if idx > 0:
            candidates.append(r[idx - 1])
        if idx < r.size:
            candidates.append(r[idx])
        if not candidates:
            out[i] = np.inf
            continue
        best = min(candidates, key=lambda x: abs(x - ts))
        out[i] = abs(best - ts)
    return out


def diagnose_timestamp_alignment(
    *,
    image_ts: List[float],
    pose_ts: List[float],
    image_label: str,
    pose_label: str,
    missing_images_in_associations: int = 0,
) -> AlignmentReport:
    img = np.array(image_ts, dtype=float)
    pose = np.array(pose_ts, dtype=float)

    image_fps, image_dt = _estimate_fps(img)
    pose_fps, pose_dt = _estimate_fps(pose)

    fps_diff_pct = None
    if image_fps is not None and pose_fps is not None and image_fps > 0:
        fps_diff_pct = abs(pose_fps - image_fps) / image_fps * 100.0

    # Use image cadence to set matching tolerance.
    # If timestamps are aligned, nearest pose should be within ~half a frame.
    if image_dt is None:
        match_tol = 0.05
    else:
        match_tol = max(0.02, 0.5 * image_dt)

    abs_dts = _nearest_abs_dts(img, pose)
    if abs_dts.size:
        abs_dt_median = float(np.median(abs_dts))
        abs_dt_p95 = float(np.percentile(abs_dts, 95))
        abs_dt_max = float(np.max(abs_dts))
        matched = int(np.sum(abs_dts <= match_tol))
        unmatched = int(np.sum(abs_dts > match_tol))
    else:
        abs_dt_median = abs_dt_p95 = abs_dt_max = None
        matched = 0
        unmatched = int(img.size) if img.size else 0

    image_gaps = _count_gaps(img, image_dt)
    pose_gaps = _count_gaps(pose, pose_dt)

    warnings: List[str] = []
    if pose.size == 0:
        warnings.append("No pose timestamps found.")
    if img.size == 0:
        warnings.append("No image timestamps found.")
    if fps_diff_pct is not None and fps_diff_pct > 5.0:
        warnings.append(f"FPS mismatch looks high (Δ={fps_diff_pct:.2f}%).")
    if unmatched and img.size:
        warnings.append(f"{unmatched}/{img.size} images have no pose within tolerance.")
    if image_gaps:
        warnings.append(f"Detected {image_gaps} image timestamp gaps (likely missing frames/data).")
    if pose_gaps:
        warnings.append(f"Detected {pose_gaps} pose timestamp gaps (likely missing poses).")
    if missing_images_in_associations:
        warnings.append("Some RGB PNGs are missing from associations.txt.")

    ok = True
    if pose.size == 0 or img.size == 0:
        ok = False
    if fps_diff_pct is not None and fps_diff_pct > 5.0:
        ok = False
    if img.size and (unmatched / img.size) > 0.02:
        ok = False
    if abs_dt_p95 is not None and abs_dt_p95 > match_tol:
        ok = False

    return AlignmentReport(
        image_label=image_label,
        pose_label=pose_label,
        image_count=int(img.size),
        pose_count=int(pose.size),
        image_fps=image_fps,
        pose_fps=pose_fps,
        image_dt_median=image_dt,
        pose_dt_median=pose_dt,
        fps_diff_pct=fps_diff_pct,
        match_tolerance_s=match_tol,
        matched=matched,
        unmatched=unmatched,
        abs_dt_median=abs_dt_median,
        abs_dt_p95=abs_dt_p95,
        abs_dt_max=abs_dt_max,
        image_gaps=image_gaps,
        pose_gaps=pose_gaps,
        missing_images_in_associations=int(missing_images_in_associations),
        ok=ok,
        warnings=warnings,
    )


def _count_png_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    try:
        # Fast path on Windows
        return sum(1 for name in os.listdir(folder) if name.lower().endswith(".png"))
    except OSError:
        return 0


def build_records(base_dir: Path) -> tuple[List[FrameRecord], AlignmentReport]:
    assoc_records = load_associations(base_dir)
    poses = load_poses(base_dir)

    rgb_dir = assoc_records[0].rgb_path.parent if assoc_records else (base_dir / "")
    rgb_png_count = _count_png_files(rgb_dir) if assoc_records else 0
    missing_in_assoc = max(0, rgb_png_count - len(assoc_records)) if rgb_png_count else 0

    report = diagnose_timestamp_alignment(
        image_ts=[r.timestamp for r in assoc_records],
        pose_ts=sorted(poses.keys()),
        image_label="images (associations.txt)",
        pose_label="poses (CameraTrajectory.csv)",
        missing_images_in_associations=missing_in_assoc,
    )

    records = infer_missing_frames(base_dir, assoc_records)

    # Merge pose info into frames using exact match first, then nearest neighbor within tolerance.
    pose_times = np.array(sorted(poses.keys()), dtype=float)
    for rec in records:
        if pose_times.size == 0:
            continue

        if rec.timestamp in poses:
            best_ts = rec.timestamp
            rec.pose_match_exact = True
        else:
            idx = int(np.searchsorted(pose_times, rec.timestamp))
            candidates = []
            if idx > 0:
                candidates.append(float(pose_times[idx - 1]))
            if idx < int(pose_times.size):
                candidates.append(float(pose_times[idx]))
            if not candidates:
                continue
            best_ts = min(candidates, key=lambda ts: abs(ts - rec.timestamp))

        abs_dt = abs(best_ts - rec.timestamp)
        rec.matched_pose_timestamp = best_ts
        rec.matched_pose_abs_dt = abs_dt

        if abs_dt > report.match_tolerance_s:
            # Too far: treat as missing pose for this frame.
            continue

        t, q = poses[best_ts]
        rec.translation = t
        rec.quaternion = q / np.linalg.norm(q)

    return records, report


def main():
    if len(sys.argv) < 2:
        print("Usage: python utilities/qt_phone_sample_player.py <path-to-phone_sample3>")
        sys.exit(1)
    base_dir = Path(sys.argv[1]).expanduser()
    if not base_dir.exists():
        print(f"Path not found: {base_dir}")
        sys.exit(1)

    records, report = build_records(base_dir)
    if not records:
        print("No frames found.")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    win = PlayerWindow(records, alignment_report=report)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
