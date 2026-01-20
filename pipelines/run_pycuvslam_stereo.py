#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import cuvslam
except ImportError:
    print("Failed to import cuvslam. Activate the PyCuVSLAM environment first.", file=sys.stderr)
    raise


def read_timestamps(path: Path) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "frame":
                continue
            if len(row) < 2:
                continue
            frame = row[0].strip()
            try:
                ts_ns = int(row[1].strip())
            except ValueError:
                continue
            entries.append((frame, ts_ns))
    return entries


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    frame = np.array(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {frame.shape} for {path}")
    frame = np.ascontiguousarray(frame[:, :, ::-1])
    return frame


def _matrix_from_yaml(data: dict, key: str) -> list[list[float]] | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return None


def _flatten_list(value: list[list[float]] | list[float]) -> list[float]:
    if not value:
        return []
    if isinstance(value[0], list):
        return [float(x) for row in value for x in row]
    return [float(x) for x in value]


def load_stereo_calibration(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    k1 = _matrix_from_yaml(data, "K1")
    d1 = _matrix_from_yaml(data, "D1")
    k2 = _matrix_from_yaml(data, "K2")
    d2 = _matrix_from_yaml(data, "D2")
    r = _matrix_from_yaml(data, "R")
    t = _matrix_from_yaml(data, "T")
    r1 = _matrix_from_yaml(data, "R1")
    r2 = _matrix_from_yaml(data, "R2")
    p1 = _matrix_from_yaml(data, "P1")
    p2 = _matrix_from_yaml(data, "P2")
    if not (k1 and d1 and k2 and d2):
        raise ValueError(f"Missing K1/D1/K2/D2 in {path}")
    return {
        "K1": k1,
        "D1": d1,
        "K2": k2,
        "D2": d2,
        "R": r,
        "T": t,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
    }


def build_rectify_maps(
    size: tuple[int, int],
    k: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    w, h = size
    if model == "fisheye":
        if d.size != 4:
            raise ValueError(f"Fisheye undistort expects 4 parameters, got {d.size}")
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k, d, r, p[:3, :3], (w, h), cv2.CV_32FC1
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            k, d, r, p[:3, :3], (w, h), cv2.CV_32FC1
        )
    return map1, map2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyCuVSLAM stereo on left/right RGB folders.")
    parser.add_argument("--left-dir", required=True, type=Path, help="Folder containing left RGB images.")
    parser.add_argument("--right-dir", required=True, type=Path, help="Folder containing right RGB images.")
    parser.add_argument("--calibration", required=True, type=Path, help="Stereo calibration YAML (K1/D1/K2/D2/etc).")
    parser.add_argument("--timestamps", required=True, type=Path, help="timestamps.txt with frame,timestamp_ns.")
    parser.add_argument("--out", type=Path, default=Path("outputs/pycuvslam_stereo/poses.tum"), help="Output TUM pose file.")
    parser.add_argument("--slam-out", type=Path, default=Path("outputs/pycuvslam_stereo/poses_slam.tum"), help="Output TUM pose file for SLAM (when enabled).")
    parser.add_argument("--left-ext", default=".jpg", help="Left image extension.")
    parser.add_argument("--right-ext", default=".jpg", help="Right image extension.")
    parser.add_argument("--undistort", action="store_true", default=True, help="Undistort/rectify images before tracking.")
    parser.add_argument("--no-undistort", action="store_false", dest="undistort", help="Disable undistort/rectify.")
    parser.add_argument("--distortion-model", choices=["brown", "fisheye"], default="fisheye", help="Distortion model for calibration parameters.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 = all).")
    parser.add_argument("--skip-missing", action="store_true", help="Skip frames with missing left/right files.")
    parser.add_argument("--enable-slam", action="store_true", help="Enable SLAM backend (pose graph + loop closure).")
    parser.add_argument("--slam-max-map-size", type=int, default=300, help="SLAM max pose graph size (0 = unlimited).")
    parser.add_argument("--slam-planar", action="store_true", help="Constrain motion to a horizontal plane in SLAM.")
    parser.add_argument("--slam-throttle-ms", type=int, default=0, help="Minimum time between loop closures (ms).")
    parser.add_argument("--preview", action="store_true", help="Show live Rerun preview.")
    parser.add_argument("--preview-interval", type=int, default=1, help="Log every N frames when previewing.")
    parser.add_argument("--preview-jpeg-quality", type=int, default=80, help="JPEG quality for RGB preview.")
    parser.add_argument("--show-features", action="store_true", help="Overlay tracked feature observations in preview.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_ext = args.left_ext if args.left_ext.startswith(".") else f".{args.left_ext}"
    right_ext = args.right_ext if args.right_ext.startswith(".") else f".{args.right_ext}"

    timestamps = read_timestamps(args.timestamps)
    if not timestamps:
        print(f"No timestamps found in {args.timestamps}", file=sys.stderr)
        return 1

    first_left = args.left_dir / f"{timestamps[0][0]}{left_ext}"
    if not first_left.exists():
        print(f"First left frame not found: {first_left}", file=sys.stderr)
        return 1
    size = Image.open(first_left).size  # (width, height)

    try:
        import cv2
    except ImportError:
        if args.undistort:
            print("OpenCV (cv2) is required for --undistort.", file=sys.stderr)
            return 1
        cv2 = None

    calib = load_stereo_calibration(args.calibration)
    k1 = np.array(calib["K1"], dtype=np.float64)
    d1 = np.array(_flatten_list(calib["D1"]), dtype=np.float64).reshape(-1, 1)
    k2 = np.array(calib["K2"], dtype=np.float64)
    d2 = np.array(_flatten_list(calib["D2"]), dtype=np.float64).reshape(-1, 1)

    r1 = calib["R1"]
    r2 = calib["R2"]
    p1 = calib["P1"]
    p2 = calib["P2"]

    if args.undistort and (r1 is None or r2 is None or p1 is None or p2 is None):
        if calib["R"] is None or calib["T"] is None:
            print("Calibration must include R/T or R1/R2/P1/P2 for rectification.", file=sys.stderr)
            return 1
        if args.distortion_model == "fisheye":
            r1, r2, p1, p2, _, _ = cv2.fisheye.stereoRectify(
                k1, d1, k2, d2, size, np.array(calib["R"]), np.array(calib["T"])
            )
        else:
            r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                k1, d1, k2, d2, size, np.array(calib["R"]), np.array(calib["T"])
            )

    if r1 is None or r2 is None or p1 is None or p2 is None:
        print("Calibration missing rectification matrices (R1/R2/P1/P2).", file=sys.stderr)
        return 1

    r1 = np.array(r1, dtype=np.float64)
    r2 = np.array(r2, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    undistort_maps = None
    if args.undistort:
        undistort_maps = (
            build_rectify_maps(size, k1, d1, r1, p1, args.distortion_model),
            build_rectify_maps(size, k2, d2, r2, p2, args.distortion_model),
        )

    left_cam = cuvslam.Camera()
    right_cam = cuvslam.Camera()
    left_cam.size = size
    right_cam.size = size
    left_cam.focal = [float(p1[0, 0]), float(p1[1, 1])]
    right_cam.focal = [float(p2[0, 0]), float(p2[1, 1])]
    left_cam.principal = [float(p1[0, 2]), float(p1[1, 2])]
    right_cam.principal = [float(p2[0, 2]), float(p2[1, 2])]
    right_cam.rig_from_camera.translation[0] = -float(p2[0, 3]) / float(p2[0, 0])

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=True,
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
    )
    slam_cfg = None
    if args.enable_slam:
        slam_cfg = cuvslam.Tracker.SlamConfig()
        slam_cfg.max_map_size = args.slam_max_map_size
        slam_cfg.planar_constraints = args.slam_planar
        slam_cfg.throttling_time_ms = args.slam_throttle_ms

    tracker = cuvslam.Tracker(cuvslam.Rig([left_cam, right_cam]), cfg, slam_cfg)

    rr = None
    if args.preview:
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            print("Rerun is not installed. Install with: pip install rerun-sdk", file=sys.stderr)
            return 1
        rr.init("pycuvslam_stereo", strict=True, spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(contents=[
                rrb.Vertical(contents=[
                    rrb.Spatial2DView(origin="world/camera_left/image", name="Left"),
                    rrb.Spatial2DView(origin="world/camera_right/image", name="Right"),
                ]),
                rrb.Spatial3DView(name="3D"),
            ]),
        ), make_active=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")
    out_f.write("# timestamp tx ty tz qx qy qz qw\n")
    slam_f = None
    if args.enable_slam:
        args.slam_out.parent.mkdir(parents=True, exist_ok=True)
        slam_f = args.slam_out.open("w")
        slam_f.write("# timestamp tx ty tz qx qy qz qw\n")

    processed = 0
    tracked = 0
    slam_tracked = 0
    trajectory: list[list[float]] = []
    slam_trajectory: list[list[float]] = []

    for frame_id, ts_ns in timestamps:
        left_path = args.left_dir / f"{frame_id}{left_ext}"
        right_path = args.right_dir / f"{frame_id}{right_ext}"
        if not left_path.exists() or not right_path.exists():
            if args.skip_missing:
                continue
            print(f"Missing frame: {left_path} or {right_path}", file=sys.stderr)
            return 1

        left = load_rgb(left_path)
        right = load_rgb(right_path)
        if undistort_maps:
            (map1_l, map2_l), (map1_r, map2_r) = undistort_maps
            left = cv2.remap(left, map1_l, map2_l, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right = cv2.remap(right, map1_r, map2_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        pose_est, slam_pose = tracker.track(ts_ns, images=[left, right])
        if pose_est.world_from_rig is not None:
            pose = pose_est.world_from_rig.pose
            tx, ty, tz = pose.translation
            qx, qy, qz, qw = pose.rotation
            out_f.write(
                f"{ts_ns / 1e9:.9f} {tx:.6f} {ty:.6f} {tz:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
            tracked += 1
            trajectory.append([float(tx), float(ty), float(tz)])

        if slam_f and slam_pose is not None:
            stx, sty, stz = slam_pose.translation
            sqx, sqy, sqz, sqw = slam_pose.rotation
            slam_f.write(
                f"{ts_ns / 1e9:.9f} {stx:.6f} {sty:.6f} {stz:.6f} "
                f"{sqx:.6f} {sqy:.6f} {sqz:.6f} {sqw:.6f}\n"
            )
            slam_tracked += 1
            slam_trajectory.append([float(stx), float(sty), float(stz)])

        processed += 1
        if rr and (processed % max(args.preview_interval, 1) == 0):
            rr.set_time_sequence("frame", processed)
            rr.log("world/camera_left/image", rr.Image(left[:, :, ::-1]).compress(jpeg_quality=args.preview_jpeg_quality))
            rr.log("world/camera_right/image", rr.Image(right[:, :, ::-1]).compress(jpeg_quality=args.preview_jpeg_quality))
            if args.show_features:
                observations = tracker.get_last_observations(0)
                if observations:
                    obs_uv = [[obs.u, obs.v] for obs in observations]
                    rr.log(
                        "world/camera_left/image/observations",
                        rr.Points2D(obs_uv, radii=3, colors=[255, 128, 0]),
                    )
            if pose_est.world_from_rig is not None:
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=pose.translation,
                        quaternion=pose.rotation,
                    ),
                )
                if trajectory:
                    rr.log("trajectory", rr.LineStrips3D(trajectory))
            if slam_pose is not None:
                rr.log(
                    "world/camera_slam",
                    rr.Transform3D(
                        translation=slam_pose.translation,
                        quaternion=slam_pose.rotation,
                    ),
                )
                if slam_trajectory:
                    rr.log("trajectory_slam", rr.LineStrips3D(slam_trajectory))

        if args.max_frames and processed >= args.max_frames:
            break
        if processed % 50 == 0:
            print(f"Processed {processed} frames, tracked {tracked}")

    out_f.close()
    if slam_f:
        slam_f.close()
    print(f"Done. Tracked {tracked}/{processed} frames.")
    if args.enable_slam:
        print(f"SLAM poses: {slam_tracked}/{processed}")
    print(f"Saved poses to {args.out}")
    if args.enable_slam:
        print(f"Saved SLAM poses to {args.slam_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
