#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

try:
    import cv2  # type: ignore
except Exception as e:
    print("Error: OpenCV (cv2) is required to read images.")
    print("Install with: pip install opencv-python")
    raise

import numpy as np  # type: ignore

ROS1_AVAILABLE = True
ROSBAGS_AVAILABLE = False
try:
    import rosbag  # type: ignore
    from sensor_msgs.msg import Image as RosImage  # type: ignore
    from std_msgs.msg import Header as RosHeader  # type: ignore
    from genpy import Time  # type: ignore
except Exception:
    ROS1_AVAILABLE = False
    try:
        from rosbags.rosbag1 import Writer as Rosbag1Writer  # type: ignore
        ROSBAGS_AVAILABLE = True
    except Exception:
        ROSBAGS_AVAILABLE = False


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def natural_key(s: str):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def collect_image_paths(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    images.sort(key=lambda p: natural_key(p.name))
    return images


def make_image_message_ros1(img, frame_id: str, stamp: "Time", encoding: str = "bgr8") -> "RosImage":
    msg = RosImage()
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]

    msg.header = RosHeader()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.height = h
    msg.width = w
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = w * channels
    msg.data = img.tobytes()
    return msg


def make_image_message_rosbags(img, frame_id: str, stamp_ns: int, seq: int, encoding: str = "bgr8"):
    # Types are registered at runtime; import after registration
    from rosbags.typesys.types import (  # type: ignore
        sensor_msgs__msg__Image as RbImage,
        std_msgs__msg__Header as RbHeader,
    )
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    header = RbHeader(seq=seq, stamp=stamp_ns, frame_id=frame_id)
    return RbImage(
        header=header,
        height=h,
        width=w,
        encoding=encoding,
        is_bigendian=0,
        step=w * channels,
        data=img.tobytes(),
    )


def write_stereo_bag_ros1(
    left_paths: List[Path],
    right_paths: List[Path],
    fps: float,
    out_bag_path: Path,
    left_topic: str,
    right_topic: str,
    left_frame_id: str,
    right_frame_id: str,
    encoding: str = "bgr8",
) -> Tuple[int, Path]:
    if fps <= 0:
        raise ValueError("FPS must be > 0")

    n = min(len(left_paths), len(right_paths))
    if n == 0:
        raise ValueError("No images found in one or both folders.")

    # Start timestamps at current epoch and increment by 1/fps
    start = time.time()
    dt = 1.0 / fps

    out_bag_path.parent.mkdir(parents=True, exist_ok=True)
    with rosbag.Bag(str(out_bag_path), "w") as bag:
        for i in range(n):
            ts = Time.from_sec(start + i * dt)

            # Read images
            left_img = cv2.imread(str(left_paths[i]), cv2.IMREAD_COLOR)
            right_img = cv2.imread(str(right_paths[i]), cv2.IMREAD_COLOR)

            if left_img is None:
                raise RuntimeError(f"Failed to read left image: {left_paths[i]}")
            if right_img is None:
                raise RuntimeError(f"Failed to read right image: {right_paths[i]}")

            # Ensure both images have same size
            if left_img.shape[:2] != right_img.shape[:2]:
                raise RuntimeError(
                    f"Mismatched image sizes at index {i}: "
                    f"left {left_img.shape[:2]} vs right {right_img.shape[:2]}"
                )

            left_msg = make_image_message_ros1(left_img, left_frame_id, ts, encoding)
            right_msg = make_image_message_ros1(right_img, right_frame_id, ts, encoding)

            bag.write(left_topic, left_msg, ts)
            bag.write(right_topic, right_msg, ts)

    return n, out_bag_path


def write_stereo_bag_rosbags(
    left_paths: List[Path],
    right_paths: List[Path],
    fps: float,
    out_bag_path: Path,
    left_topic: str,
    right_topic: str,
    left_frame_id: str,
    right_frame_id: str,
    encoding: str = "bgr8",
) -> Tuple[int, Path]:
    if not ROSBAGS_AVAILABLE:
        raise RuntimeError("rosbags library not available.")
    if fps <= 0:
        raise ValueError("FPS must be > 0")

    n = min(len(left_paths), len(right_paths))
    if n == 0:
        raise ValueError("No images found in one or both folders.")

    start = time.time()
    dt = 1.0 / fps

    # Use ROS1 Noetic typestore and built-in message types
    from rosbags.typesys import Stores, get_typestore  # type: ignore
    from rosbags.typesys.stores.ros1_noetic import (  # type: ignore
        sensor_msgs__msg__Image as RImage,
        std_msgs__msg__Header as RHeader,
        builtin_interfaces__msg__Time as RTime,
    )

    typestore = get_typestore(Stores.ROS1_NOETIC)

    out_bag_path.parent.mkdir(parents=True, exist_ok=True)
    with Rosbag1Writer(str(out_bag_path)) as writer:
        left_conn = writer.add_connection(left_topic, RImage.__msgtype__, typestore=typestore)
        right_conn = writer.add_connection(right_topic, RImage.__msgtype__, typestore=typestore)

        for i in range(n):
            stamp_ns = int((start + i * dt) * 1e9)

            left_img = cv2.imread(str(left_paths[i]), cv2.IMREAD_COLOR)
            right_img = cv2.imread(str(right_paths[i]), cv2.IMREAD_COLOR)

            if left_img is None:
                raise RuntimeError(f"Failed to read left image: {left_paths[i]}")
            if right_img is None:
                raise RuntimeError(f"Failed to read right image: {right_paths[i]}")

            if left_img.shape[:2] != right_img.shape[:2]:
                raise RuntimeError(
                    f"Mismatched image sizes at index {i}: "
                    f"left {left_img.shape[:2]} vs right {right_img.shape[:2]}"
                )

            # Build ROS1-normalized messages
            h, w = left_img.shape[:2]
            channels = left_img.shape[2] if left_img.ndim == 3 else 1
            if encoding == "rgb8":
                left_arr = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_arr = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            else:
                left_arr = left_img
                right_arr = right_img

            # Ensure uint8 1-D arrays for rosbags serializer
            if left_arr.dtype != np.uint8:
                left_arr = left_arr.astype(np.uint8)
            if right_arr.dtype != np.uint8:
                right_arr = right_arr.astype(np.uint8)
            left_flat = left_arr.reshape(-1)
            right_flat = right_arr.reshape(-1)

            stamp_time = RTime(sec=int(stamp_ns // 1_000_000_000), nanosec=int(stamp_ns % 1_000_000_000))

            left_msg = RImage(
                header=RHeader(seq=i, stamp=stamp_time, frame_id=left_frame_id),
                height=h,
                width=w,
                encoding=encoding,
                is_bigendian=0,
                step=w * channels,
                data=left_flat,
            )
            right_msg = RImage(
                header=RHeader(seq=i, stamp=stamp_time, frame_id=right_frame_id),
                height=h,
                width=w,
                encoding=encoding,
                is_bigendian=0,
                step=w * channels,
                data=right_flat,
            )

            writer.write(left_conn, stamp_ns, typestore.serialize_ros1(left_msg, RImage.__msgtype__))
            writer.write(right_conn, stamp_ns, typestore.serialize_ros1(right_msg, RImage.__msgtype__))

    return n, out_bag_path


def find_common_root(left_dir: Path, right_dir: Path) -> Path:
    # Common parent directory for both left and right
    left_parts = left_dir.resolve().parts
    right_parts = right_dir.resolve().parts
    common = []
    for a, b in zip(left_parts, right_parts):
        if a == b:
            common.append(a)
        else:
            break
    if not common:
        # Fallback to left parent if no commonality
        return left_dir.parent
    return Path(*common)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert two image folders (left/right) to a ROS1 stereo rosbag with generated timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--left", required=True, help="Path to left camera image folder")
    p.add_argument("--right", required=True, help="Path to right camera image folder")
    p.add_argument("--fps", type=float, required=True, help="Playback frame rate used to generate timestamps")
    p.add_argument("--left-topic", default="/camera/left/image_raw", help="ROS topic for left images")
    p.add_argument("--right-topic", default="/camera/right/image_raw", help="ROS topic for right images")
    p.add_argument("--left-frame-id", default="left_camera", help="Frame ID for left images")
    p.add_argument("--right-frame-id", default="right_camera", help="Frame ID for right images")
    p.add_argument(
        "--encoding",
        default="bgr8",
        choices=["bgr8", "rgb8"],
        help="Image encoding to declare in sensor_msgs/Image",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Optional explicit output .bag path. If omitted, the bag is written to the common root of the two folders"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    left_dir = Path(args.left)
    right_dir = Path(args.right)

    left_paths = collect_image_paths(left_dir)
    right_paths = collect_image_paths(right_dir)

    if args.output:
        out_bag = Path(args.output)
        if out_bag.suffix.lower() != ".bag":
            out_bag = out_bag.with_suffix(".bag")
        out_root = out_bag.parent
    else:
        # Use the common root directory; if both folders share a typical parent like workspace9/left & right,
        # we will place the bag under that parent.
        common_root = find_common_root(left_dir.parent, right_dir.parent)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        name = f"stereo_{args.fps:.2f}fps_{min(len(left_paths), len(right_paths))}frames_{ts_str}.bag"
        out_bag = common_root / name
        out_root = common_root

    print(f"Left images:  {len(left_paths)} from {left_dir}")
    print(f"Right images: {len(right_paths)} from {right_dir}")
    print(f"Writing rosbag to: {out_bag}")

    if ROS1_AVAILABLE:
        written, path = write_stereo_bag_ros1(
            left_paths,
            right_paths,
            args.fps,
            out_bag,
            args.left_topic,
            args.right_topic,
            args.left_frame_id,
            args.right_frame_id,
            args.encoding,
        )
    elif ROSBAGS_AVAILABLE:
        written, path = write_stereo_bag_rosbags(
            left_paths,
            right_paths,
            args.fps,
            out_bag,
            args.left_topic,
            args.right_topic,
            args.left_frame_id,
            args.right_frame_id,
            args.encoding,
        )
    else:
        print("Neither ROS1 Python (rosbag) nor rosbags library found.")
        print("Install one of:")
        print("  - WSL/Ubuntu + ROS Noetic (rosbag, sensor_msgs, std_msgs, genpy)")
        print("  - pip install rosbags")
        return 1
    print(f"Done. Wrote {written} synchronized stereo frames.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
