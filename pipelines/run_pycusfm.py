#!/usr/bin/env python3
"""
Run cuSFM on a single RGB sequence using existing poses.

Inputs:
  - RGB folder (e.g., data/sample_xxx/iphone_mono)
  - Calibration YAML (fx, fy, cx, cy)
  - timestamps.txt (frame,timestamp_ns)
  - Poses in TUM format (timestamp tx ty tz qx qy qz qw), e.g., cuvslam_poses_slam.tum

Outputs (default next to the RGB folder):
  - frames_meta.json generated from the inputs
  - cusfm_output/ (cuSFM workspace with refined poses and sparse COLMAP files)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml
from PIL import Image


def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, env=dict(env) if env else None, check=True)


def _env_with_conda_lib(env: Mapping[str, str]) -> dict[str, str]:
    """Ensure native deps see correct CUDA/conda libs (prefer WSL shim + CUDA13)."""
    merged = dict(env)
    ld_parts: list[str] = []
    wsl_lib = Path("/usr/lib/wsl/lib")
    if wsl_lib.is_dir():
        ld_parts.append(str(wsl_lib))
    # Add pyCuSFM bundled libraries (libcvcuda, etc.)
    pycusfm_lib = Path(__file__).resolve().parent.parent / "third_party" / "pyCuSFM" / "pycusfm" / "lib"
    if pycusfm_lib.is_dir():
        ld_parts.append(str(pycusfm_lib))
    conda_prefix = merged.get("CONDA_PREFIX")
    if conda_prefix:
        ld_parts.append(str(Path(conda_prefix) / "lib"))

    def _add_cuda_paths(*bases: Path) -> None:
        for base in bases:
            for sub in ("targets/x86_64-linux/lib", "lib64", "lib"):
                candidate = base / sub
                if candidate.is_dir():
                    ld_parts.append(str(candidate))

    # Force CUDA 13.0 first (matches driver), then allow 13.1/generic if 13.0 missing.
    cuda_13 = Path("/usr/local/cuda-13.0")
    cuda_131 = Path("/usr/local/cuda-13.1")
    cuda_generic = Path("/usr/local/cuda")
    if cuda_13.exists():
        _add_cuda_paths(cuda_13)
    elif cuda_131.exists():
        _add_cuda_paths(cuda_131)
    elif cuda_generic.exists():
        _add_cuda_paths(cuda_generic)

    if merged.get("LD_LIBRARY_PATH"):
        current = []
        for p in merged["LD_LIBRARY_PATH"].split(":"):
            if not p:
                continue
            # Drop older /usr/local/cuda-* entries to avoid picking stale compat stubs.
            if p.startswith("/usr/local/cuda") and "cuda-13" not in p:
                continue
            current.append(p)
        ld_parts.extend(current)
    if ld_parts:
        merged["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return merged


def _parse_timestamps(path: Path) -> List[Tuple[str, int]]:
    rows = list(csv.reader(path.read_text().splitlines()))
    if not rows or len(rows) < 2:
        raise ValueError(f"No timestamp rows found in {path}")
    header = [h.strip().lower() for h in rows[0]]
    frame_idx = header.index("frame")
    ts_idx = header.index("timestamp_ns")
    out: list[tuple[str, int]] = []
    for row in rows[1:]:
        if len(row) <= ts_idx:
            continue
        frame_id = row[frame_idx].strip()
        ts_ns = int(float(row[ts_idx]))
        out.append((frame_id, ts_ns))
    if not out:
        raise ValueError(f"No valid timestamp entries in {path}")
    return out


def _find_frame(path_base: Path, frame_id: str, exts: Iterable[str]) -> Path | None:
    for ext in exts:
        candidate = path_base / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_calibration(calib_path: Path) -> Tuple[float, float, float, float]:
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


def _load_tum_poses(path: Path) -> Dict[int, Tuple[float, float, float, float, float, float, float]]:
    """Map timestamp_ns -> pose (tx, ty, tz, qx, qy, qz, qw)."""
    poses: dict[int, tuple[float, float, float, float, float, float, float]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        ts_sec = float(parts[0])
        ts_ns = int(round(ts_sec * 1e9))
        tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:])
        poses[ts_ns] = (tx, ty, tz, qx, qy, qz, qw)
    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return poses


def _quat_to_axis_angle_deg(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float, float]:
    """Return axis (x,y,z) and angle in degrees."""
    qw = max(min(qw, 1.0), -1.0)
    angle = 2.0 * math.acos(qw)
    sin_half = math.sqrt(1.0 - qw * qw)
    if sin_half < 1e-8:
        return 1.0, 0.0, 0.0, math.degrees(angle)
    return qx / sin_half, qy / sin_half, qz / sin_half, math.degrees(angle)


def _estimate_frequency_hz(timestamps: Sequence[int]) -> float:
    if len(timestamps) < 2:
        return 30.0
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not deltas:
        return 30.0
    avg_ns = sum(deltas) / len(deltas)
    return float(1e9 / avg_ns)


def _build_frames_meta(
    rgb_dir: Path,
    timestamps: Sequence[Tuple[str, int]],
    poses_by_ns: Mapping[int, Tuple[float, float, float, float, float, float, float]],
    calib_path: Path,
    out_path: Path,
) -> Tuple[int, int]:
    fx, fy, cx, cy = _parse_calibration(calib_path)
    first_frame = _find_frame(rgb_dir, timestamps[0][0], [".png", ".jpg", ".jpeg"])
    if not first_frame:
        raise FileNotFoundError(f"No RGB frame found for {timestamps[0][0]} in {rgb_dir}")
    width, height = Image.open(first_frame).size

    freq_hz = _estimate_frequency_hz([ts for _, ts in timestamps])
    cam_params = {
        "sensor_meta_data": {
            "sensor_id": 0,
            "sensor_type": "CAMERA",
            "sensor_name": "rgb",
            "frequency": freq_hz,
            "sensor_to_vehicle_transform": {
                "axis_angle": {"x": 0, "y": 0, "z": 0, "angle_degrees": 0},
                "translation": {"x": 0, "y": 0, "z": 0},
            },
        },
        "calibration_parameters": {
            "image_width": width,
            "image_height": height,
            "camera_matrix": {
                "data": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                "row_count": 3,
                "column_count": 3,
            },
            "distortion_coefficients": {
                "data": [0.0, 0.0, 0.0, 0.0, 0.0],
                "row_count": 1,
                "column_count": 5,
            },
            "rectification_matrix": {
                "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "row_count": 3,
                "column_count": 3,
            },
            "projection_matrix": {
                "data": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
                "row_count": 3,
                "column_count": 4,
            },
        },
    }

    keyframes: list[dict] = []
    used = 0
    skipped = 0
    for frame_id, ts_ns in timestamps:
        pose = poses_by_ns.get(ts_ns)
        if pose is None:
            skipped += 1
            continue
        rgb_path = _find_frame(rgb_dir, frame_id, [".png", ".jpg", ".jpeg"])
        if not rgb_path:
            skipped += 1
            continue
        tx, ty, tz, qx, qy, qz, qw = pose
        ax, ay, az, ang_deg = _quat_to_axis_angle_deg(qx, qy, qz, qw)
        keyframes.append(
            {
                "id": frame_id,
                "camera_params_id": "0",
                "timestamp_microseconds": str(int(round(ts_ns / 1000))),
                "image_name": str(rgb_path.relative_to(out_path.parent)),
                "camera_to_world": {
                    "axis_angle": {"x": ax, "y": ay, "z": az, "angle_degrees": ang_deg},
                    "translation": {"x": tx, "y": ty, "z": tz},
                },
                "synced_sample_id": frame_id,
            }
        )
        used += 1

    payload = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {"0": "session_0"},
        "camera_params_id_to_camera_params": {"0": cam_params},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return used, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cuSFM on a single RGB sequence using precomputed poses.")
    parser.add_argument("--rgb-dir", required=True, type=Path, help="Folder containing RGB images.")
    parser.add_argument("--calibration", required=True, type=Path, help="Calibration YAML for the RGB camera.")
    parser.add_argument("--timestamps", required=True, type=Path, help="timestamps.txt with frame,timestamp_ns.")
    parser.add_argument("--poses", required=True, type=Path, help="TUM trajectory file (e.g., cuvslam_poses_slam.tum).")
    parser.add_argument("--out-dir", type=Path, help="cuSFM output folder (default: <rgb-dir>/../cusfm_output).")
    parser.add_argument("--frames-meta-out", type=Path, help="frames_meta.json output (default: <base>/frames_meta_cusfm.json).")
    args = parser.parse_args()

    rgb_dir = args.rgb_dir.expanduser().resolve()
    calib_path = args.calibration.expanduser().resolve()
    ts_path = args.timestamps.expanduser().resolve()
    poses_path = args.poses.expanduser().resolve()
    if not rgb_dir.is_dir():
        raise SystemExit(f"RGB folder not found: {rgb_dir}")
    for p in [calib_path, ts_path, poses_path]:
        if not p.exists():
            raise SystemExit(f"Required file not found: {p}")

    base_dir = rgb_dir.parent
    cusfm_out = (args.out_dir or base_dir / "cusfm_output").expanduser().resolve()
    frames_meta_path = (args.frames_meta_out or base_dir / "frames_meta_cusfm.json").expanduser().resolve()

    timestamps = _parse_timestamps(ts_path)
    poses_by_ns = _load_tum_poses(poses_path)
    frames_meta_path.parent.mkdir(parents=True, exist_ok=True)
    used, skipped = _build_frames_meta(rgb_dir, timestamps, poses_by_ns, calib_path, frames_meta_path)
    print(f"[info] frames_meta.json written to {frames_meta_path} (used {used}, skipped {skipped})")

    cusfm_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pycusfm.cusfm_cli",
        "--input_dir",
        str(base_dir),
        "--cusfm_base_dir",
        str(cusfm_out),
        "--override_frames_meta_file",
        str(frames_meta_path),
        "--skip_cuvslam",
    ]
    env = _env_with_conda_lib(os.environ)
    _run(cmd, env=env)

    print("Done.")
    print(f"cuSFM workspace: {cusfm_out}")
    print(f"frames_meta used: {frames_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
