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
except ImportError as exc:
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


def load_depth(path: Path) -> np.ndarray:
    image = Image.open(path)
    frame = np.array(image)
    if frame.ndim != 2:
        raise ValueError(f"Expected depth image with shape [H W], got {frame.shape} for {path}")
    if frame.dtype != np.uint16:
        frame = frame.astype(np.uint16)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame


def _matrix_from_yaml(data: dict, key: str) -> list[list[float]] | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return None


def _matrix_from_ros(data: dict, key: str) -> list[list[float]] | None:
    block = data.get(key)
    if not isinstance(block, dict):
        return None
    raw = block.get("data")
    if not isinstance(raw, list) or len(raw) != 9:
        return None
    return [raw[0:3], raw[3:6], raw[6:9]]


def _vector_from_ros(data: dict, key: str) -> list[float] | None:
    block = data.get(key)
    if not isinstance(block, dict):
        return None
    raw = block.get("data")
    if not isinstance(raw, list):
        return None
    return [float(x) for x in raw]


def load_calibration(path: Path) -> tuple[float, float, float, float, list[float] | None]:
    data = yaml.safe_load(path.read_text())
    k = (
        _matrix_from_yaml(data, "K")
        or _matrix_from_yaml(data, "K1")
        or _matrix_from_ros(data, "camera_matrix")
    )
    if k is None:
        raise ValueError(f"Missing intrinsics K in {path}")
    fx = float(k[0][0])
    fy = float(k[1][1])
    cx = float(k[0][2])
    cy = float(k[1][2])
    d = (
        _matrix_from_yaml(data, "D")
        or _matrix_from_yaml(data, "D1")
        or _vector_from_ros(data, "distortion_coefficients")
    )
    if isinstance(d, list) and d and isinstance(d[0], list):
        d = [float(x) for row in d for x in row]
    if isinstance(d, list):
        d = [float(x) for x in d]
    return fx, fy, cx, cy, d


def setup_camera(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    size: tuple[int, int],
    d: list[float] | None,
    use_distortion: bool,
    distortion_model: str,
) -> cuvslam.Camera:
    cam = cuvslam.Camera()
    cam.size = size
    cam.focal = [fx, fy]
    cam.principal = [cx, cy]
    if use_distortion and d:
        params = d[:]
        if distortion_model == "fisheye":
            if len(params) != 4:
                print(f"Warning: fisheye expects 4 parameters, got {len(params)}; ignoring distortion.")
            else:
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, params)
        else:
            if len(params) == 4:
                params.append(0.0)
            if len(params) in (4, 5):
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Brown, params)
            else:
                print(f"Warning: unsupported distortion length {len(params)}; ignoring distortion.")
    return cam


def build_undistort_maps(
    size: tuple[int, int],
    k: np.ndarray,
    d: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    w, h = size
    if model == "fisheye":
        if d.size != 4:
            raise ValueError(f"Fisheye undistort expects 4 parameters, got {d.size}")
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k, d, np.eye(3), k, (w, h), cv2.CV_32FC1
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            k, d, np.eye(3), k, (w, h), cv2.CV_32FC1
        )
    return map1, map2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyCuVSLAM RGBD on a left RGB + left_depth dataset.")
    parser.add_argument("--rgb-dir", required=True, type=Path, help="Folder containing RGB images (e.g. left).")
    parser.add_argument("--depth-dir", required=True, type=Path, help="Folder containing depth images (e.g. left_depth).")
    parser.add_argument("--calibration", required=True, type=Path, help="Calibration YAML for the RGB camera.")
    parser.add_argument("--timestamps", required=True, type=Path, help="timestamps.txt with frame,timestamp_ns.")
    parser.add_argument("--out", type=Path, default=Path("outputs/pycuvslam_rgbd/poses.tum"), help="Output TUM pose file.")
    parser.add_argument("--slam-out", type=Path, default=Path("outputs/pycuvslam_rgbd/poses_slam.tum"), help="Output TUM pose file for SLAM (when enabled).")
    parser.add_argument("--rgb-ext", default=".jpg", help="RGB image extension.")
    parser.add_argument("--depth-ext", default=".png", help="Depth image extension.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale factor (value per meter).")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 = all).")
    parser.add_argument("--skip-missing", action="store_true", help="Skip frames with missing rgb/depth files.")
    parser.add_argument("--use-distortion", action="store_true", help="Apply Brown distortion coefficients from calibration.")
    parser.add_argument("--undistort", action="store_true", help="Undistort RGB and depth images before tracking.")
    parser.add_argument("--distortion-model", choices=["brown", "fisheye"], default="brown", help="Distortion model for calibration parameters.")
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
    rgb_ext = args.rgb_ext if args.rgb_ext.startswith(".") else f".{args.rgb_ext}"
    depth_ext = args.depth_ext if args.depth_ext.startswith(".") else f".{args.depth_ext}"

    timestamps = read_timestamps(args.timestamps)
    if not timestamps:
        print(f"No timestamps found in {args.timestamps}", file=sys.stderr)
        return 1

    # Resolve RGB extension: try requested, then fallbacks (.png/.jpg/.jpeg).
    first_rgb = args.rgb_dir / f"{timestamps[0][0]}{rgb_ext}"
    if not first_rgb.exists():
        for alt in [".png", ".jpg", ".jpeg"]:
            candidate = args.rgb_dir / f"{timestamps[0][0]}{alt}"
            if candidate.exists():
                rgb_ext = alt
                first_rgb = candidate
                print(f"First RGB frame not found with {args.rgb_ext}; using '{alt}' instead.", file=sys.stderr)
                break
    if not first_rgb.exists():
        print(f"First RGB frame not found: {first_rgb}", file=sys.stderr)
        return 1
    size = Image.open(first_rgb).size  # (width, height)

    fx, fy, cx, cy, d = load_calibration(args.calibration)
    if args.undistort and args.use_distortion:
        print("Warning: --undistort disables --use-distortion (images are already undistorted).")

    undistort_maps = None
    if args.undistort:
        try:
            import cv2
        except ImportError:
            print("OpenCV (cv2) is required for --undistort.", file=sys.stderr)
            return 1
        if not d:
            print("Calibration distortion coefficients are required for --undistort.", file=sys.stderr)
            return 1
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        d_arr = np.array(d, dtype=np.float64).reshape(-1, 1)
        undistort_maps = build_undistort_maps(size, k, d_arr, args.distortion_model)

    cam = setup_camera(
        fx,
        fy,
        cx,
        cy,
        size,
        d,
        args.use_distortion and not args.undistort,
        args.distortion_model,
    )

    rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = args.depth_scale
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    cfg = cuvslam.Tracker.OdometryConfig(
        odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings,
        enable_final_landmarks_export=True,
    )
    slam_cfg = None
    if args.enable_slam:
        slam_cfg = cuvslam.Tracker.SlamConfig()
        slam_cfg.max_map_size = args.slam_max_map_size
        slam_cfg.planar_constraints = args.slam_planar
        slam_cfg.throttling_time_ms = args.slam_throttle_ms
    tracker = cuvslam.Tracker(cuvslam.Rig([cam]), cfg, slam_cfg)

    rr = None
    if args.preview:
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            print("Rerun is not installed. Install with: pip install rerun-sdk", file=sys.stderr)
            return 1
        rr.init("pycuvslam_rgbd", strict=True, spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(contents=[
                rrb.Vertical(contents=[
                    rrb.Spatial2DView(origin="world/camera/image", name="RGB"),
                    rrb.Spatial2DView(origin="world/camera/depth", name="Depth"),
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
        rgb_path = args.rgb_dir / f"{frame_id}{rgb_ext}"
        depth_path = args.depth_dir / f"{frame_id}{depth_ext}"
        if not rgb_path.exists() or not depth_path.exists():
            if args.skip_missing:
                continue
            print(f"Missing frame: {rgb_path} or {depth_path}", file=sys.stderr)
            return 1

        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path)
        if undistort_maps:
            map1, map2 = undistort_maps
            rgb = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            depth = cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pose_est, slam_pose = tracker.track(ts_ns, images=[rgb], depths=[depth])
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
            rgb_preview = np.ascontiguousarray(rgb[:, :, ::-1])
            rr.log("world/camera/image", rr.Image(rgb_preview).compress(jpeg_quality=args.preview_jpeg_quality))
            rr.log("world/camera/depth", rr.Image(depth))
            if args.show_features:
                observations = tracker.get_last_observations(0)
                if observations:
                    obs_uv = [[obs.u, obs.v] for obs in observations]
                    rr.log(
                        "world/camera/image/observations",
                        rr.Points2D(obs_uv, radii=3, colors=[255, 128, 0]),
                    )
                    rr.log(
                        "world/camera/depth/observations",
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
