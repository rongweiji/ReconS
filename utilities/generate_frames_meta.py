#!/usr/bin/env python3
"""
Generate a pyCuSFM-compatible frames_meta.json from:
1) A newline-delimited JSON timestamp file (frames_time.json)
2) A stereo calibration YAML (K1/D1/R1/P1/K2/D2/R2/P2/R/T)

Output is written next to the timestamp file.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required to run this script. pip install pyyaml") from exc


LEFT_CAMERA_PARAMS_ID = "0"
RIGHT_CAMERA_PARAMS_ID = "1"


def read_timestamps(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
            if "filename" not in obj or "timestamp_ns" not in obj:
                raise ValueError(f"Missing keys on line {line_no}; need 'filename' and 'timestamp_ns'")
            entries.append(
                {"filename": str(obj["filename"]), "timestamp_ns": int(obj["timestamp_ns"])}
            )
    if not entries:
        raise ValueError("No timestamp rows found")
    return entries


def infer_camera_dirs(
    dataset_dir: Path, left_name: Optional[str], right_name: Optional[str]
) -> Tuple[Path, Path]:
    if left_name and right_name:
        left = dataset_dir / left_name
        right = dataset_dir / right_name
    else:
        dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        left = next((p for p in dirs if "left" in p.name.lower()), None)
        right = next((p for p in dirs if "right" in p.name.lower()), None)
    if left is None or right is None:
        raise ValueError("Could not infer left/right image folders. Pass --left-dir-name/--right-dir-name.")
    if not left.is_dir() or not right.is_dir():
        raise ValueError("Provided left/right paths are not directories.")
    return left, right


def rotation_matrix_to_axis_angle_deg(R: List[List[float]]) -> Tuple[float, float, float, float]:
    trace = R[0][0] + R[1][1] + R[2][2]
    cos_theta = max(min((trace - 1.0) / 2.0, 1.0), -1.0)
    angle = math.acos(cos_theta)
    if angle < 1e-9:
        return 0.0, 0.0, 0.0, 0.0
    sin_theta = math.sin(angle)
    x = (R[2][1] - R[1][2]) / (2.0 * sin_theta)
    y = (R[0][2] - R[2][0]) / (2.0 * sin_theta)
    z = (R[1][0] - R[0][1]) / (2.0 * sin_theta)
    return x, y, z, math.degrees(angle)


def invert_rotation(R: List[List[float]]) -> List[List[float]]:
    return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]


def mat_vec_mul(R: List[List[float]], v: List[float]) -> List[float]:
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def load_calibration(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    required = ["K1", "D1", "R1", "P1", "K2", "D2", "R2", "P2", "R", "T"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Calibration file missing keys: {missing}")
    return data


def read_image_size(img_path: Path) -> Tuple[int, int]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency check
        raise SystemExit("OpenCV (cv2) is required to read image dimensions.") from exc
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    h, w = img.shape[:2]
    return w, h


def flatten_matrix(mat: List[List[float]]) -> List[float]:
    return [float(v) for row in mat for v in row]


def build_camera_params(
    camera_id: str,
    sensor_id: int,
    sensor_name: str,
    image_size: Tuple[int, int],
    K: List[List[float]],
    D: Union[List[List[float]], List[float]],
    R_rect: List[List[float]],
    P: List[List[float]],
    R_cam_to_world: List[List[float]],
    t_cam_to_world: List[float],
    frequency: int,
) -> Dict[str, Any]:
    axis_x, axis_y, axis_z, angle_deg = rotation_matrix_to_axis_angle_deg(R_cam_to_world)
    dist_flat = flatten_matrix(D if isinstance(D[0], list) else [D])
    return {
        camera_id: {
            "sensor_meta_data": {
                "sensor_id": sensor_id,
                "sensor_type": "CAMERA",
                "sensor_name": sensor_name,
                "frequency": frequency,
                "sensor_to_vehicle_transform": {
                    "axis_angle": {
                        "x": axis_x,
                        "y": axis_y,
                        "z": axis_z,
                        "angle_degrees": angle_deg,
                    },
                    "translation": {
                        "x": t_cam_to_world[0],
                        "y": t_cam_to_world[1],
                        "z": t_cam_to_world[2],
                    },
                },
            },
            "calibration_parameters": {
                "image_width": image_size[0],
                "image_height": image_size[1],
                "camera_matrix": {
                    "data": flatten_matrix(K),
                    "row_count": 3,
                    "column_count": 3,
                },
                "distortion_coefficients": {
                    "data": [float(v) for v in dist_flat],
                    "row_count": 1,
                    "column_count": len(dist_flat),
                },
                "rectification_matrix": {
                    "data": flatten_matrix(R_rect),
                    "row_count": 3,
                    "column_count": 3,
                },
                "projection_matrix": {
                    "data": flatten_matrix(P),
                    "row_count": 3,
                    "column_count": 4,
                },
            },
        }
    }


def build_keyframes(
    timestamps: List[Dict[str, Any]],
    left_dir_name: str,
    right_dir_name: str,
    right_R_cam_to_world: List[List[float]],
    right_t_cam_to_world: List[float],
) -> List[Dict[str, Any]]:
    keyframes: List[Dict[str, Any]] = []
    right_axis = rotation_matrix_to_axis_angle_deg(right_R_cam_to_world)
    base_pose = {
        "axis_angle": {"x": 0.0, "y": 0.0, "z": 0.0, "angle_degrees": 0.0},
        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    right_pose = {
        "axis_angle": {
            "x": right_axis[0],
            "y": right_axis[1],
            "z": right_axis[2],
            "angle_degrees": right_axis[3],
        },
        "translation": {
            "x": right_t_cam_to_world[0],
            "y": right_t_cam_to_world[1],
            "z": right_t_cam_to_world[2],
        },
    }

    next_id = 0
    for idx, row in enumerate(timestamps):
        ts_us = str(int(round(row["timestamp_ns"] / 1000.0)))
        synced_id = str(idx)
        filename = row["filename"]
        keyframes.append(
            {
                "id": str(next_id),
                "camera_params_id": LEFT_CAMERA_PARAMS_ID,
                "timestamp_microseconds": ts_us,
                "image_name": f"{left_dir_name}/{filename}",
                "camera_to_world": base_pose,
                "synced_sample_id": synced_id,
            }
        )
        next_id += 1
        keyframes.append(
            {
                "id": str(next_id),
                "camera_params_id": RIGHT_CAMERA_PARAMS_ID,
                "timestamp_microseconds": ts_us,
                "image_name": f"{right_dir_name}/{filename}",
                "camera_to_world": right_pose,
                "synced_sample_id": synced_id,
            }
        )
        next_id += 1
    return keyframes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert frames_time.json + stereo calibration YAML into frames_meta.json for pyCuSFM"
    )
    parser.add_argument("timestamps", help="Path to newline-delimited JSON timestamps file (frames_time.json)")
    parser.add_argument("calibration", help="Path to stereo calibration YAML (stereo_result.yaml)")
    parser.add_argument("--left-dir-name", help="Relative folder name for left images (default: auto-detect contains 'left')")
    parser.add_argument("--right-dir-name", help="Relative folder name for right images (default: auto-detect contains 'right')")
    parser.add_argument("--sensor-frequency", type=int, default=30, help="Sensor frequency Hz (default: 30)")
    args = parser.parse_args()

    ts_path = Path(args.timestamps).resolve()
    calib_path = Path(args.calibration).resolve()
    dataset_dir = ts_path.parent

    timestamps = read_timestamps(ts_path)
    calib = load_calibration(calib_path)

    left_dir, right_dir = infer_camera_dirs(dataset_dir, args.left_dir_name, args.right_dir_name)

    sample_image = left_dir / timestamps[0]["filename"]
    image_size = read_image_size(sample_image)

    R_lr = calib["R"]
    T_lr = [float(v) for v in flatten_matrix(calib["T"])]
    R_right_to_world = invert_rotation(R_lr)
    t_right_to_world = [-v for v in mat_vec_mul(R_right_to_world, T_lr)]
    baseline_m = math.sqrt(sum(v * v for v in T_lr))

    camera_params: Dict[str, Any] = {}
    camera_params.update(
        build_camera_params(
            camera_id=LEFT_CAMERA_PARAMS_ID,
            sensor_id=0,
            sensor_name=left_dir.name,
            image_size=image_size,
            K=calib["K1"],
            D=calib["D1"],
            R_rect=calib["R1"],
            P=calib["P1"],
            R_cam_to_world=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_cam_to_world=[0.0, 0.0, 0.0],
            frequency=args.sensor_frequency,
        )
    )
    camera_params.update(
        build_camera_params(
            camera_id=RIGHT_CAMERA_PARAMS_ID,
            sensor_id=1,
            sensor_name=right_dir.name,
            image_size=image_size,
            K=calib["K2"],
            D=calib["D2"],
            R_rect=calib["R2"],
            P=calib["P2"],
            R_cam_to_world=R_right_to_world,
            t_cam_to_world=t_right_to_world,
            frequency=args.sensor_frequency,
        )
    )

    keyframes = build_keyframes(
        timestamps,
        left_dir_name=left_dir.name,
        right_dir_name=right_dir.name,
        right_R_cam_to_world=R_right_to_world,
        right_t_cam_to_world=t_right_to_world,
    )

    meta = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {cam_id: "0" for cam_id in camera_params.keys()},
        "camera_params_id_to_camera_params": camera_params,
        "stereo_pair": [
            {
                "left_camera_param_id": LEFT_CAMERA_PARAMS_ID,
                "right_camera_param_id": RIGHT_CAMERA_PARAMS_ID,
                "baseline_meters": baseline_m,
            }
        ],
    }

    out_path = dataset_dir / "frames_meta.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
