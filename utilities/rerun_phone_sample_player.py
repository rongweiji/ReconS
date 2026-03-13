#!/usr/bin/env python3
"""
Rerun-based preview tool for a phone sample folder.

Expected inputs inside the sample folder:
  - associations.txt
  - CameraTrajectory.csv (optional)
  - RGB images referenced by associations.txt
  - depth images referenced by associations.txt

Usage:
  python utilities/rerun_phone_sample_player.py data/sample_xxx
  python utilities/rerun_phone_sample_player.py data/sample_xxx --save /tmp/sample_preview.rrd --no-spawn
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rerun as rr


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class FrameRecord:
    timestamp: float
    rgb_path: Path
    depth_color_path: Optional[Path]
    depth_metric_path: Optional[Path]
    translation: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None  # xyzw
    matched_pose_timestamp: Optional[float] = None
    matched_pose_abs_dt: Optional[float] = None
    pose_match_exact: bool = False
    inferred: bool = False


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
        return "Alignment: OK" if self.ok else "Alignment: ISSUES"

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
        lines.append(f"Matches: {self.matched}/{self.image_count} (unmatched={self.unmatched})")
        if self.abs_dt_median is not None:
            lines.append(
                f"|Δt| stats: median={self.abs_dt_median * 1000.0:.2f}ms, "
                f"p95={self.abs_dt_p95 * 1000.0:.2f}ms, max={self.abs_dt_max * 1000.0:.2f}ms"
            )
        lines.append(f"Gaps: images={self.image_gaps}, poses={self.pose_gaps}")
        if self.missing_images_in_associations:
            lines.append(f"Missing images in associations.txt: {self.missing_images_in_associations}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"- {warning}")
        lines.append(f"Overall: {'OK' if self.ok else 'NOT OK'}")
        return "\n".join(lines)


def load_associations(base_dir: Path) -> List[FrameRecord]:
    assoc_file = base_dir / "associations.txt"
    if not assoc_file.exists():
        raise FileNotFoundError(f"Missing associations.txt in {base_dir}")

    records: List[FrameRecord] = []
    with assoc_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ts_rgb = float(parts[0])
            rgb_rel = parts[1]
            depth_metric_rel = parts[3]
            rgb_path = base_dir / rgb_rel
            depth_metric_path = base_dir / depth_metric_rel
            if not depth_metric_path.exists():
                depth_metric_path = None

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

    poses: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    with pose_file.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts = float(row["timestamp"])
            t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float32)
            q = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float32)
            poses[ts] = (t, q)
    return poses


def infer_missing_frames(base_dir: Path, records: List[FrameRecord]) -> List[FrameRecord]:
    if not records:
        return records

    rgb_dir = records[0].rgb_path.parent
    if not rgb_dir.exists():
        return records

    rgb_files = sorted(
        [p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    )
    if not rgb_files:
        return records

    ts_sorted = sorted(record.timestamp for record in records)
    if len(ts_sorted) > 1:
        frame_dt = float(np.median(np.diff(np.array(ts_sorted, dtype=float))))
    else:
        frame_dt = 1.0 / 30.0

    name_to_record = {record.rgb_path.name: record for record in records}
    new_records = list(records)

    for rgb_path in rgb_files:
        if rgb_path.name in name_to_record:
            continue

        stem = rgb_path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        timestamp = int(digits) * frame_dt if digits else ts_sorted[-1] + frame_dt

        rgb_folder = rgb_path.parent.name
        depth_color_candidate = base_dir / f"{rgb_folder}d" / rgb_path.name
        depth_metric_candidate = base_dir / f"{rgb_folder}_metric_d" / rgb_path.name

        new_records.append(
            FrameRecord(
                timestamp=float(timestamp),
                rgb_path=rgb_path,
                depth_color_path=depth_color_candidate if depth_color_candidate.exists() else None,
                depth_metric_path=depth_metric_candidate if depth_metric_candidate.exists() else None,
                inferred=True,
            )
        )

    new_records.sort(key=lambda record: record.timestamp)
    return new_records


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def colorize_depth(depth_metric: np.ndarray) -> Optional[np.ndarray]:
    if depth_metric is None:
        return None
    if depth_metric.ndim == 3 and depth_metric.shape[2] == 3:
        return cv2.cvtColor(depth_metric, cv2.COLOR_BGR2RGB)
    if depth_metric.ndim != 2:
        return None
    valid = depth_metric > 0
    if np.any(valid):
        vmin = float(depth_metric[valid].min())
        vmax = float(depth_metric[valid].max())
    else:
        vmin = float(depth_metric.min())
        vmax = float(depth_metric.max())
    if np.isclose(vmax, vmin):
        vmax = vmin + 1.0
    norm = np.clip((depth_metric.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    norm_uint8 = (norm * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


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
        out[i] = abs(min(candidates, key=lambda value: abs(value - ts)) - ts) if candidates else np.inf
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

    match_tol = 0.05 if image_dt is None else max(0.02, 0.5 * image_dt)
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
        warnings.append(f"Detected {image_gaps} image timestamp gaps.")
    if pose_gaps:
        warnings.append(f"Detected {pose_gaps} pose timestamp gaps.")
    if missing_images_in_associations:
        warnings.append("Some RGB images are missing from associations.txt.")

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


def _count_image_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    try:
        return sum(1 for name in os.listdir(folder) if Path(name).suffix.lower() in SUPPORTED_IMAGE_EXTS)
    except OSError:
        return 0


def build_records(base_dir: Path) -> tuple[List[FrameRecord], AlignmentReport]:
    assoc_records = load_associations(base_dir)
    poses = load_poses(base_dir)

    rgb_dir = assoc_records[0].rgb_path.parent if assoc_records else base_dir
    rgb_count = _count_image_files(rgb_dir)
    missing_in_assoc = max(0, rgb_count - len(assoc_records)) if rgb_count else 0

    report = diagnose_timestamp_alignment(
        image_ts=[record.timestamp for record in assoc_records],
        pose_ts=sorted(poses.keys()),
        image_label="images (associations.txt)",
        pose_label="poses (CameraTrajectory.csv)",
        missing_images_in_associations=missing_in_assoc,
    )

    records = infer_missing_frames(base_dir, assoc_records)
    pose_times = np.array(sorted(poses.keys()), dtype=float)

    for record in records:
        if pose_times.size == 0:
            continue

        best_ts = None
        if record.timestamp in poses:
            best_ts = record.timestamp
            record.pose_match_exact = True
        else:
            idx = int(np.searchsorted(pose_times, record.timestamp))
            candidates = []
            if idx > 0:
                candidates.append(float(pose_times[idx - 1]))
            if idx < int(pose_times.size):
                candidates.append(float(pose_times[idx]))
            if candidates:
                best_ts = min(candidates, key=lambda ts: abs(ts - record.timestamp))

        if best_ts is None:
            continue

        abs_dt = abs(best_ts - record.timestamp)
        record.matched_pose_timestamp = best_ts
        record.matched_pose_abs_dt = abs_dt
        if abs_dt > report.match_tolerance_s:
            continue

        t, q = poses[best_ts]
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0:
            continue
        record.translation = t
        record.quaternion = q / q_norm

    return records, report


def _init_rerun(*, app_id: str, spawn: bool, connect: str | None, save: Path | None) -> None:
    if connect and save:
        raise ValueError("Use only one of --connect or --save.")

    rr.init(app_id, spawn=spawn and connect is None and save is None)
    if connect:
        rr.connect_tcp(connect)
    if save:
        rr.save(save)


def _rgb_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _current_axes(record: FrameRecord, axis_len: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if record.translation is None or record.quaternion is None:
        empty = np.zeros((0, 2, 3), dtype=np.float32)
        return empty, empty, empty
    origin = record.translation.astype(np.float32)
    rot = quaternion_to_matrix(record.quaternion.astype(np.float32))
    x_axis = np.stack([origin, origin + rot[:, 0] * axis_len], axis=0)[None, :, :]
    y_axis = np.stack([origin, origin + rot[:, 1] * axis_len], axis=0)[None, :, :]
    z_axis = np.stack([origin, origin + rot[:, 2] * axis_len], axis=0)[None, :, :]
    return x_axis.astype(np.float32), y_axis.astype(np.float32), z_axis.astype(np.float32)


def _log_axis(entity_path: str, strips: np.ndarray, color: list[int]) -> None:
    if strips.shape[0] == 0:
        rr.log(entity_path, rr.LineStrips3D([]))
        return
    rr.log(entity_path, rr.LineStrips3D(strips, colors=[color] * strips.shape[0]))


def _log_static_context(base_dir: Path, records: List[FrameRecord], report: AlignmentReport) -> None:
    posed = [record.translation for record in records if record.translation is not None]
    if posed:
        positions = np.vstack(posed).astype(np.float32)
        rr.log(
            "preview/trajectory/path",
            rr.LineStrips3D([positions], colors=[[255, 255, 0, 255]]),
            static=True,
        )
        rr.log(
            "preview/trajectory/points",
            rr.Points3D(
                positions,
                colors=np.tile(np.array([255, 255, 0, 255], dtype=np.uint8), (positions.shape[0], 1)),
                radii=np.full(positions.shape[0], 0.01, dtype=np.float32),
            ),
            static=True,
        )

    info = [
        f"sample={base_dir}",
        f"frames={len(records)}",
        f"posed_frames={sum(record.translation is not None for record in records)}",
        report.summary_text(),
    ]
    rr.log("preview/info", rr.TextDocument("\n".join(info)), static=True)
    rr.log("preview/alignment_report", rr.TextDocument(report.to_console_report()), static=True)


def stream_to_rerun(base_dir: Path, records: List[FrameRecord], report: AlignmentReport, args) -> None:
    if not records:
        raise ValueError("No frames found.")

    _init_rerun(
        app_id="rerun_phone_sample_player",
        spawn=args.spawn,
        connect=args.connect,
        save=args.save.expanduser().resolve() if args.save else None,
    )
    _log_static_context(base_dir, records, report)

    first_ts = records[0].timestamp
    effective_records = list(enumerate(records))[:: args.stride]
    if args.max_frames > 0:
        effective_records = effective_records[: args.max_frames]

    for original_idx, record in effective_records:
        rr.set_time_sequence("frame", original_idx)
        rr.set_time_seconds("elapsed", record.timestamp - first_ts)

        rgb = _rgb_image(record.rgb_path)
        if rgb is not None:
            rr.log("preview/images/rgb", rr.Image(rgb))

        if record.depth_color_path is not None and record.depth_color_path.exists():
            depth_color = _rgb_image(record.depth_color_path)
            if depth_color is not None:
                rr.log("preview/images/depth_color", rr.Image(depth_color))

        if record.depth_metric_path is not None and record.depth_metric_path.exists():
            depth_metric = cv2.imread(str(record.depth_metric_path), cv2.IMREAD_UNCHANGED)
            if depth_metric is not None:
                if depth_metric.ndim == 2:
                    if depth_metric.dtype == np.uint16:
                        rr.log("preview/images/depth_metric_raw", rr.DepthImage(depth_metric, meter=1000.0))
                    else:
                        rr.log("preview/images/depth_metric_raw", rr.DepthImage(depth_metric.astype(np.float32)))
                depth_metric_vis = colorize_depth(depth_metric)
                if depth_metric_vis is not None:
                    rr.log("preview/images/depth_metric_vis", rr.Image(depth_metric_vis))

        if record.translation is not None:
            rr.log(
                "preview/current/position",
                rr.Points3D(
                    np.asarray([record.translation], dtype=np.float32),
                    colors=np.asarray([[255, 255, 0, 255]], dtype=np.uint8),
                    radii=np.asarray([0.03], dtype=np.float32),
                ),
            )
        else:
            rr.log("preview/current/position", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))

        x_axis, y_axis, z_axis = _current_axes(record)
        _log_axis("preview/current/orientation/x", x_axis, [255, 0, 0, 255])
        _log_axis("preview/current/orientation/y", y_axis, [0, 255, 0, 255])
        _log_axis("preview/current/orientation/z", z_axis, [0, 128, 255, 255])

        pose_state = "MISSING"
        if record.translation is not None and record.quaternion is not None:
            pose_state = "OK"
            if record.matched_pose_abs_dt is not None:
                pose_state += f" (Δt={record.matched_pose_abs_dt * 1000.0:.1f}ms)"

        status_lines = [
            f"frame={original_idx + 1}/{len(records)}",
            f"timestamp={record.timestamp:.6f}s",
            f"rgb={record.rgb_path.name}",
            f"pose={pose_state}",
        ]
        if record.inferred:
            status_lines.append("inferred_from_rgb_dir=true")
        rr.log("preview/frame_status", rr.TextDocument("\n".join(status_lines)))

    print(report.to_console_report())
    print(f"Logged {len(effective_records)} frames to Rerun.")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Preview a phone sample folder in Rerun.")
    parser.add_argument("sample_dir", type=Path, help="Sample folder containing associations.txt.")
    parser.add_argument("--stride", type=int, default=1, help="Log every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit the number of logged frames (0 = all).")
    parser.add_argument("--connect", type=str, help="Connect to an existing Rerun viewer at host:port.")
    parser.add_argument("--save", type=Path, help="Save the recording to an .rrd file.")
    parser.add_argument("--no-spawn", dest="spawn", action="store_false", help="Do not spawn a local Rerun viewer.")
    parser.set_defaults(spawn=True)
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    base_dir = args.sample_dir.expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Path not found: {base_dir}")

    records, report = build_records(base_dir)
    if not records:
        print("No frames found.")
        return 1

    stream_to_rerun(base_dir, records, report, args)
    print("Visualization complete. Check Rerun viewer or saved recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
