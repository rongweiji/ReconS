from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps_reported: float
    frame_count_reported: int
    extracted_frames: int
    duration_sec_from_timestamps: float
    duration_sec_from_reported_fps: float
    dt_median_sec: float
    dt_std_sec: float


@dataclass(frozen=True)
class ArInfo:
    path: str
    samples: int
    duration_sec: float
    dt_median_sec: float
    dt_std_sec: float
    rate_hz_median: float


@dataclass(frozen=True)
class AlignmentInfo:
    video: VideoInfo
    ar: ArInfo
    duration_offset_sec: float
    duration_ratio: float
    residual_ms_p50: float
    residual_ms_p95: float
    residual_ms_max_abs: float


def _find_first_file(folder: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_ar_csv_timestamps(ar_csv_path: Path) -> np.ndarray:
    with ar_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "timestamp_ref" not in reader.fieldnames:
            raise ValueError(
                f"{ar_csv_path} missing 'timestamp_ref' column; got columns: {reader.fieldnames}"
            )
        timestamps: list[float] = []
        for row in reader:
            ts = row.get("timestamp_ref")
            if ts is None or ts == "":
                continue
            timestamps.append(float(ts))
    if not timestamps:
        raise ValueError(f"{ar_csv_path} contains no timestamps")
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    if not np.all(np.isfinite(ts_arr)):
        raise ValueError(f"{ar_csv_path} contains non-finite timestamps")
    return ts_arr


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv_rows(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(list(row))


def extract_video_frames(
    video_path: Path,
    out_dir: Path,
    *,
    ext: str = ".png",
    overwrite: bool = False,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
) -> tuple[VideoInfo, np.ndarray]:
    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if ext and not ext.startswith("."):
        ext = "." + ext

    _ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps_reported) or fps_reported <= 1e-6:
        fps_reported = 30.0

    frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    timestamps_sec: list[float] = []
    written = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if not math.isfinite(t_sec) or t_sec <= 0.0:
            t_sec = idx / fps_reported
        timestamps_sec.append(t_sec)

        if idx % frame_step == 0:
            out_path = out_dir / f"frame_{idx:06d}{ext}"
            if overwrite or not out_path.exists():
                ok_write = cv2.imwrite(str(out_path), frame)
                if not ok_write:
                    raise RuntimeError(f"Failed to write frame: {out_path}")
            written += 1

        idx += 1
        if max_frames is not None and idx >= max_frames:
            break

    cap.release()

    ts = np.asarray(timestamps_sec, dtype=np.float64)
    if ts.size == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    ts = ts - ts[0]
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        dt_median = 1.0 / fps_reported
        dt_std = 0.0
    else:
        dt_median = float(np.median(dts))
        dt_std = float(np.std(dts))

    duration_ts = float(ts[-1])
    duration_fps = float(ts.size / fps_reported)

    info = VideoInfo(
        path=str(video_path),
        fps_reported=fps_reported,
        frame_count_reported=frame_count_reported,
        extracted_frames=written,
        duration_sec_from_timestamps=duration_ts,
        duration_sec_from_reported_fps=duration_fps,
        dt_median_sec=dt_median,
        dt_std_sec=dt_std,
    )

    _write_csv_rows(
        out_dir / "video_frames.csv",
        ["frame_index", "t_sec"],
        ((i, float(t)) for i, t in enumerate(ts)),
    )
    with (out_dir / "video_info.json").open("w") as f:
        json.dump(asdict(info), f, indent=2)

    return info, ts


def analyze_ar(ar_csv_path: Path) -> tuple[ArInfo, np.ndarray]:
    ts_abs = _read_ar_csv_timestamps(ar_csv_path)
    ts = ts_abs - ts_abs[0]
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        dt_median = 0.01
        dt_std = 0.0
    else:
        dt_median = float(np.median(dts))
        dt_std = float(np.std(dts))

    rate_hz = float(1.0 / dt_median) if dt_median > 0 else float("inf")
    info = ArInfo(
        path=str(ar_csv_path),
        samples=int(ts.size),
        duration_sec=float(ts[-1]),
        dt_median_sec=dt_median,
        dt_std_sec=dt_std,
        rate_hz_median=rate_hz,
    )
    return info, ts


def _nearest_residuals(ar_t: np.ndarray, video_t: np.ndarray) -> np.ndarray:
    if ar_t.size < 2 or video_t.size < 1:
        return np.zeros((0,), dtype=np.float64)
    idx = np.searchsorted(ar_t, video_t, side="left")
    idx = np.clip(idx, 1, ar_t.size - 1)
    left = idx - 1
    right = idx
    choose_right = (ar_t[right] - video_t) < (video_t - ar_t[left])
    nearest = np.where(choose_right, right, left)
    return ar_t[nearest] - video_t


def visualize_alignment(
    *,
    ar_t: np.ndarray,
    video_t: np.ndarray,
    video_info: VideoInfo,
    ar_info: ArInfo,
    out_png: Path,
    out_json: Path,
) -> AlignmentInfo:
    max_t = float(max(ar_t[-1] if ar_t.size else 0.0, video_t[-1] if video_t.size else 0.0))
    if not math.isfinite(max_t) or max_t <= 0:
        max_t = 1.0

    residuals = _nearest_residuals(ar_t, video_t)
    residuals_ms = residuals * 1000.0
    if residuals_ms.size:
        p50 = float(np.percentile(np.abs(residuals_ms), 50))
        p95 = float(np.percentile(np.abs(residuals_ms), 95))
        max_abs = float(np.max(np.abs(residuals_ms)))
    else:
        p50 = p95 = max_abs = 0.0

    duration_offset = float(ar_info.duration_sec - video_info.duration_sec_from_timestamps)
    duration_ratio = float(
        (ar_info.duration_sec / video_info.duration_sec_from_timestamps)
        if video_info.duration_sec_from_timestamps > 0
        else float("inf")
    )

    alignment = AlignmentInfo(
        video=video_info,
        ar=ar_info,
        duration_offset_sec=duration_offset,
        duration_ratio=duration_ratio,
        residual_ms_p50=p50,
        residual_ms_p95=p95,
        residual_ms_max_abs=max_abs,
    )

    sec_bins = np.arange(0.0, max_t + 1.0, 1.0, dtype=np.float64)
    if sec_bins.size < 2:
        sec_bins = np.array([0.0, max_t], dtype=np.float64)

    video_counts, _ = np.histogram(video_t, bins=sec_bins)
    ar_counts, _ = np.histogram(ar_t, bins=sec_bins)
    bin_centers = (sec_bins[:-1] + sec_bins[1:]) * 0.5

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].step(bin_centers, video_counts, where="mid", label="video frames / sec", linewidth=1.5)
    axes[0].step(bin_centers, ar_counts, where="mid", label="AR samples / sec", linewidth=1.5)
    axes[0].axhline(video_info.fps_reported, linestyle="--", linewidth=1, label="video fps (reported)")
    axes[0].axhline(ar_info.rate_hz_median, linestyle="--", linewidth=1, label="AR rate (median)")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Per-second sampling density (duration alignment + stability)")

    if video_t.size >= 2:
        axes[1].plot(video_t[1:], np.diff(video_t) * 1000.0, label="video dt (ms)", linewidth=1)
    if ar_t.size >= 2:
        axes[1].plot(ar_t[1:], np.diff(ar_t) * 1000.0, label="AR dt (ms)", linewidth=1)
    axes[1].axhline((1.0 / video_info.fps_reported) * 1000.0, linestyle="--", linewidth=1)
    axes[1].axhline(ar_info.dt_median_sec * 1000.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("dt (ms)")
    axes[1].legend(loc="upper right")

    if residuals_ms.size:
        axes[2].plot(video_t, residuals_ms, label="nearest AR - video (ms)", linewidth=1)
        axes[2].axhline(0.0, color="black", linewidth=1)
        axes[2].set_ylabel("residual (ms)")
        axes[2].legend(loc="upper right")
    else:
        axes[2].text(0.5, 0.5, "No residuals computed", ha="center", va="center")

    axes[2].set_xlabel("time since start (s)")
    subtitle = (
        f"video_duration={video_info.duration_sec_from_timestamps:.3f}s, "
        f"ar_duration={ar_info.duration_sec:.3f}s, "
        f"offset(ar-video)={duration_offset:+.3f}s, "
        f"|residual| p50={p50:.2f}ms p95={p95:.2f}ms"
    )
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    with out_json.open("w") as f:
        json.dump(asdict(alignment), f, indent=2)

    return alignment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert phone sample folder: extract video frames and visualize alignment with *_ar.csv."
        )
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing video (e.g. .mov) and AR csv (e.g. *_ar.csv).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional video filename within input_folder; otherwise autodetected.",
    )
    parser.add_argument(
        "--ar",
        type=str,
        default=None,
        help="Optional ar csv filename within input_folder; otherwise autodetected (*_ar.csv or ar.csv).",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Output frames directory name (created inside input_folder). Default: frames_<video_stem>.",
    )
    parser.add_argument("--ext", type=str, default=".png", help="Frame file extension (.png or .jpg).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames.")
    parser.add_argument("--frame-step", type=int, default=1, help="Write every Nth frame (default: 1).")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Decode at most this many frames (debug).",
    )

    args = parser.parse_args()
    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    video_path: Optional[Path]
    if args.video:
        video_path = (input_folder / args.video).resolve()
    else:
        video_path = _find_first_file(input_folder, [f"*{ext}" for ext in VIDEO_EXTS])
    if not video_path or not video_path.exists():
        raise FileNotFoundError(f"No video found in {input_folder} (searched: {VIDEO_EXTS})")

    ar_path: Optional[Path]
    if args.ar:
        ar_path = (input_folder / args.ar).resolve()
    else:
        ar_path = _find_first_file(input_folder, ["*_ar.csv", "ar.csv"])
    if not ar_path or not ar_path.exists():
        raise FileNotFoundError(f"No AR csv found in {input_folder} (searched: *_ar.csv, ar.csv)")

    video_stem = video_path.stem
    frames_dir_name = args.frames_dir or f"frames_{video_stem}"
    frames_dir = (input_folder / frames_dir_name).resolve()

    video_info, video_t = extract_video_frames(
        video_path,
        frames_dir,
        ext=args.ext,
        overwrite=args.overwrite,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    ar_info, ar_t = analyze_ar(ar_path)

    alignment_png = input_folder / f"alignment_{video_stem}.png"
    alignment_json = input_folder / f"alignment_{video_stem}.json"
    alignment = visualize_alignment(
        ar_t=ar_t,
        video_t=video_t,
        video_info=video_info,
        ar_info=ar_info,
        out_png=alignment_png,
        out_json=alignment_json,
    )

    print("Input folder:", input_folder)
    print("Video:", video_info.path)
    print(
        f"  fps_reported={video_info.fps_reported:.3f} extracted_frames={video_info.extracted_frames} "
        f"duration_ts={video_info.duration_sec_from_timestamps:.3f}s dt_median={video_info.dt_median_sec*1000:.3f}ms "
        f"dt_std={video_info.dt_std_sec*1000:.3f}ms"
    )
    print("AR:", ar_info.path)
    print(
        f"  samples={ar_info.samples} rate_median={ar_info.rate_hz_median:.2f}Hz duration={ar_info.duration_sec:.3f}s "
        f"dt_median={ar_info.dt_median_sec*1000:.3f}ms dt_std={ar_info.dt_std_sec*1000:.3f}ms"
    )
    print(
        f"Duration offset (ar-video): {alignment.duration_offset_sec:+.3f}s "
        f"(ratio {alignment.duration_ratio:.6f})"
    )
    print(
        f"Nearest residual |ms|: p50={alignment.residual_ms_p50:.2f} p95={alignment.residual_ms_p95:.2f} "
        f"max={alignment.residual_ms_max_abs:.2f}"
    )
    print("Frames saved to:", frames_dir)
    print("Alignment plot:", alignment_png)
    print("Alignment summary:", alignment_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
