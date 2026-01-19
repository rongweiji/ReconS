#!/usr/bin/env python3
"""
Extract frames from a video and generate per-frame timestamps.

Frames are written to a sibling folder (default: <video_stem>_frames) and a
timestamps.txt is written next to the video with the header:
frame,timestamp_ns

The first timestamp is aligned to the video's creation time metadata when
available (falling back to the file's mtime), and subsequent timestamps are
computed from the video FPS.

Rotation:
- `--rotate auto` (default) respects video metadata (rotate/displaymatrix tags).
- `--rotate 0|90|180|270` forces a rotation.
- `--landscape` additionally rotates 90° CCW if the (rotated) frame is portrait.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2  # type: ignore


def fraction_to_float(value: Optional[str]) -> Optional[float]:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_rotation(deg: Optional[Union[int, float, str]]) -> Optional[int]:
    """Return a signed rotation in degrees restricted to multiples of 90."""
    if deg is None:
        return None
    try:
        val = int(round(float(deg)))
    except Exception:
        return None
    if val % 90 != 0:
        return None
    # Clamp into [-270, 270] range while preserving sign direction.
    val_mod = val % 360
    if val_mod == 0:
        return 0
    if val_mod == 90:
        return 90 if val >= 0 else -270
    if val_mod == 180:
        return 180
    if val_mod == 270:
        return -90 if val < 0 else 270
    return None


def probe_video(video_path: Path) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """Return (fps, creation_time_iso8601, rotation_deg) from ffprobe if available."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,side_data_list,stream_tags=rotate:format_tags=creation_time",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    stream = streams[0] if streams else {}
    fps = fraction_to_float(stream.get("avg_frame_rate")) or fraction_to_float(
        stream.get("r_frame_rate")
    )
    rotation = None
    creation_time = None
    fmt = data.get("format", {})
    if fmt:
        tags = fmt.get("tags") or {}
        creation_time = tags.get("creation_time")
    tags = stream.get("tags") or {}
    rotation = normalize_rotation(tags.get("rotate"))
    if rotation is None:
        for sd in stream.get("side_data_list", []):
            if isinstance(sd, dict) and "rotation" in sd:
                rotation = normalize_rotation(sd.get("rotation"))
                break
    return fps, creation_time, rotation


def resolve_start_ns(video_path: Path, creation_time: Optional[str]) -> int:
    """Convert creation_time to nanoseconds since epoch; fall back to mtime."""
    if creation_time:
        try:
            # ffprobe uses ISO8601 with trailing Z
            parsed = datetime.fromisoformat(
                creation_time.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            return int(parsed.timestamp() * 1_000_000_000)
        except ValueError:
            pass
    stat = video_path.stat()
    return int(stat.st_mtime * 1_000_000_000)


def apply_rotation(frame, rotation_deg: int):
    if rotation_deg == 0:
        return frame
    deg = abs(rotation_deg) % 360
    sign = 1 if rotation_deg >= 0 else -1
    if deg == 90:
        # Positive: CCW, Negative: CW
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE if sign > 0 else cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        # 270 CCW is 90 CW; 270 CW is 90 CCW
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE if sign > 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_frames(
    video_path: Path,
    output_dir: Path,
    timestamps_path: Path,
    overwrite: bool,
    rotate_mode: Union[str, int],
    force_landscape: bool,
) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamps_path.exists() and not overwrite:
        raise SystemExit(f"Timestamps file already exists: {timestamps_path}")

    probed_fps, creation_time, meta_rotation = probe_video(video_path)
    start_ns = resolve_start_ns(video_path, creation_time)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = probed_fps or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise SystemExit("Could not determine FPS from video metadata.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pad = max(7, len(str(total_frames))) if total_frames else 7

    lines = ["frame,timestamp_ns"]
    frame_idx = 0
    ns_per_frame = 1_000_000_000 / fps

    if isinstance(rotate_mode, str) and rotate_mode.lower() == "auto":
        base_rotation = meta_rotation or 0
    else:
        base_rotation = normalize_rotation(rotate_mode) or 0

    landscape_extra = None  # computed on first frame if needed

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        if base_rotation:
            frame = apply_rotation(frame, base_rotation)

        if force_landscape:
            if landscape_extra is None:
                landscape_extra = 0
                if frame.shape[0] > frame.shape[1]:
                    landscape_extra = 90
                    frame = apply_rotation(frame, landscape_extra)
            elif landscape_extra:
                frame = apply_rotation(frame, landscape_extra)

        frame_id = f"{frame_idx:0{pad}d}"
        filename = f"{frame_id}.png"
        frame_path = output_dir / filename
        if not cv2.imwrite(str(frame_path), frame):
            raise SystemExit(f"Failed to write frame {frame_idx} to {frame_path}")
        timestamp_ns = start_ns + int(round((frame_idx - 1) * ns_per_frame))
        lines.append(f"{frame_id},{timestamp_ns}")

    cap.release()

    if frame_idx == 0:
        raise SystemExit("No frames were read from the video.")

    timestamps_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {frame_idx} frames to {output_dir}\n"
        f"Timestamps: {timestamps_path}\n"
        f"FPS used: {fps:.3f} | Start time (ns): {start_ns}\n"
        f"Rotation applied: base {base_rotation} deg"
        + (f" + 90 deg for landscape" if landscape_extra else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a video into frames and generate timestamps.txt"
    )
    parser.add_argument("video", type=Path, help="Path to the input video (e.g. .mov)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save frames (default: <video_stem>_frames next to video)",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        help="Path for timestamps.txt (default: next to the video)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory/timestamps file",
    )
    parser.add_argument(
        "--rotate",
        default="auto",
        help="Rotation in degrees (0/90/180/270) or 'auto' to honor video metadata (default: auto)",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="If the rotated frame is portrait, rotate an additional 90 deg CCW to make it landscape",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else video_path.parent / f"{video_path.stem}_frames"
    )
    timestamps_path = (
        args.timestamps.expanduser().resolve()
        if args.timestamps
        else video_path.parent / "timestamps.txt"
    )

    extract_frames(
        video_path,
        output_dir,
        timestamps_path,
        args.overwrite,
        args.rotate,
        args.landscape,
    )


if __name__ == "__main__":
    main()
