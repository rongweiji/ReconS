#!/usr/bin/env python3
"""
Extract frames from a video and generate per-frame timestamps.

Frames are written to a sibling folder (default: <video_stem>_frames) and a
timestamps.txt is written next to the video with the header:
frame,timestamp_ns

The first timestamp is aligned to the video's creation time metadata when
available (falling back to the file's mtime), and subsequent timestamps are
computed from the video FPS.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

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


def probe_video(video_path: Path) -> Tuple[Optional[float], Optional[str]]:
    """Return (fps, creation_time_iso8601) from ffprobe if available."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames:format_tags=creation_time",
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
    creation_time = None
    fmt = data.get("format", {})
    if fmt:
        tags = fmt.get("tags") or {}
        creation_time = tags.get("creation_time")
    return fps, creation_time


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


def extract_frames(
    video_path: Path, output_dir: Path, timestamps_path: Path, overwrite: bool
) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamps_path.exists() and not overwrite:
        raise SystemExit(f"Timestamps file already exists: {timestamps_path}")

    probed_fps, creation_time = probe_video(video_path)
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

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        filename = f"{frame_idx:0{pad}d}.png"
        frame_path = output_dir / filename
        if not cv2.imwrite(str(frame_path), frame):
            raise SystemExit(f"Failed to write frame {frame_idx} to {frame_path}")
        timestamp_ns = start_ns + int(round((frame_idx - 1) * ns_per_frame))
        lines.append(f"{filename},{timestamp_ns}")

    cap.release()

    if frame_idx == 0:
        raise SystemExit("No frames were read from the video.")

    timestamps_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {frame_idx} frames to {output_dir}\n"
        f"Timestamps: {timestamps_path}\n"
        f"FPS used: {fps:.3f} | Start time (ns): {start_ns}"
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

    extract_frames(video_path, output_dir, timestamps_path, args.overwrite)


if __name__ == "__main__":
    main()
