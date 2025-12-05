#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files[p.name] = p
    return files


def match_pairs(left_dir: Path, right_dir: Path) -> List[Tuple[Path, Path]]:
    left_files = list_images(left_dir)
    right_files = list_images(right_dir)
    common = sorted(set(left_files.keys()) & set(right_files.keys()))
    return [(left_files[name], right_files[name]) for name in common]


def find_corners(img_gray: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, np.ndarray]:
    # board_size = (rows, cols) of inner corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(img_gray, board_size, flags)
    if not ret:
        return False, None
    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners_refined


def build_object_points(board_size: Tuple[int, int], square_size_m: float) -> np.ndarray:
    rows, cols = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    # Chessboard points laid out in grid
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    return objp


def calibrate_single(object_points: List[np.ndarray],
                     image_points: List[np.ndarray],
                     image_size: Tuple[int, int],
                     *,
                     use_fisheye: bool,
                     use_rational: bool):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    if use_fisheye:
        # Get an initial guess from pinhole calibration to stabilize fisheye.
        K_init = np.zeros((3, 3), dtype=np.float64)
        D_init = np.zeros((5, 1), dtype=np.float64)
        obj_pinhole = [p.reshape(-1, 3).astype(np.float32) for p in object_points]
        img_pinhole = [p.astype(np.float32) for p in image_points]
        _, K_init, D_init, _, _ = cv2.calibrateCamera(
            obj_pinhole,
            img_pinhole,
            image_size,
            K_init,
            D_init,
            flags=0,
            criteria=criteria
        )
        K = K_init.copy()
        D = np.zeros((4, 1), dtype=np.float64)
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                 cv2.fisheye.CALIB_FIX_SKEW |
                 cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
        try:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K,
                D,
                flags=flags,
                criteria=criteria
            )
        except cv2.error:
            # Retry without recomputing extrinsics and let intrinsics adjust
            flags = (cv2.fisheye.CALIB_FIX_SKEW |
                     cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K,
                D,
                flags=flags,
                criteria=criteria
            )
    else:
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((8 if use_rational else 5, 1), dtype=np.float64)
        flags = cv2.CALIB_RATIONAL_MODEL if use_rational else 0
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            K,
            D,
            flags=flags,
            criteria=criteria
        )
    return ret, K, D, rvecs, tvecs


def stereo_calibrate(object_points, image_points_left, image_points_right, K1, D1, K2, D2, image_size, *, use_fisheye: bool):
    if use_fisheye:
        flags = (cv2.fisheye.CALIB_FIX_INTRINSIC |
                 cv2.fisheye.CALIB_FIX_SKEW |
                 cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        try:
            retval, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
                object_points,
                image_points_left,
                image_points_right,
                K1, D1,
                K2, D2,
                image_size,
                flags=flags,
                criteria=criteria
            )
        except cv2.error:
            # Retry letting intrinsics adjust a bit; still fix skew to reduce degeneracy
            flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
                     cv2.fisheye.CALIB_FIX_SKEW)
            retval, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
                object_points,
                image_points_left,
                image_points_right,
                K1, D1,
                K2, D2,
                image_size,
                flags=flags,
                criteria=criteria
            )
        E, F = np.zeros((3, 3)), np.zeros((3, 3))
    else:
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            object_points,
            image_points_left,
            image_points_right,
            K1, D1,
            K2, D2,
            image_size,
            criteria=criteria,
            flags=flags
        )
    return retval, K1, D1, K2, D2, R, T, E, F


def stereo_rectify(K1, D1, K2, D2, image_size, R, T, *, use_fisheye: bool):
    if use_fisheye:
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
        roi1 = roi2 = (0, 0, image_size[0], image_size[1])
    else:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, R2, P1, P2, Q, roi1, roi2


def save_yaml(path: Path, data: dict):
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is None:
        raise ImportError("PyYAML not installed. Install with 'pip install PyYAML'")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def to_list(mat: np.ndarray):
    return mat.tolist()


def preview_detected(img_color, corners, board_size):
    vis = img_color.copy()
    cv2.drawChessboardCorners(vis, board_size, corners, True)
    cv2.imshow("Corners", vis)
    cv2.waitKey(50)


def preview_rectification(left_img, right_img, map1x, map1y, map2x, map2y):
    left_rect = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    h1, w1 = left_rect.shape[:2]
    combined = np.hstack([left_rect, right_rect])
    for y in range(0, h1, 40):
        cv2.line(combined, (0, y), (combined.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.imshow("Rectified preview", combined)
    cv2.waitKey(50)


def quick_orientation_probe(pairs: List[Tuple[Path, Path]], board_size: Tuple[int, int]) -> Tuple[int, int]:
    """Try both (rows, cols) and (cols, rows) on a few pairs, pick the one with more detections."""
    if not pairs:
        return board_size
    sample = pairs[:min(10, len(pairs))]
    rc = board_size
    cr = (board_size[1], board_size[0])
    def count_ok(sz):
        cnt = 0
        for lp, rp in sample:
            li = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            ri = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if li is None or ri is None:
                continue
            ok_l, _ = find_corners(li, sz)
            ok_r, _ = find_corners(ri, sz)
            if ok_l and ok_r:
                cnt += 1
        return cnt
    a = count_ok(rc)
    b = count_ok(cr)
    return rc if a >= b else cr


def main():
    parser = argparse.ArgumentParser(description="OpenCV Stereo Calibration from folders (no ROS).")
    parser.add_argument("--left", required=True, help="Path to left camera image folder")
    parser.add_argument("--right", required=True, help="Path to right camera image folder")
    parser.add_argument("--pattern-size", nargs=2, type=int, required=True, metavar=("ROWS", "COLS"),
                        help="Chessboard inner corners: rows cols (e.g., 6 8)")
    parser.add_argument("--square-size", type=float, required=True, help="Square size in meters (e.g., 0.108)")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit number of pairs used (0 = all)")
    parser.add_argument("--best-pairs", type=int, default=0, help="Quality-based selection: keep top N pairs by detection quality (0 = disabled)")
    parser.add_argument("--preview", action="store_true", help="Show corner detection and rectification previews")
    parser.add_argument("--save-prefix", default="stereo", help="Prefix for output YAML files")
    parser.add_argument("--fisheye", action="store_true", help="Use OpenCV fisheye model (better for >120° FOV lenses)")
    parser.add_argument("--rational-model", action="store_true", help="Use 8-coefficient distortion model (helps strong distortion if not using --fisheye)")
    parser.add_argument("--analyze", action="store_true", help="Print input coverage/sharpness stats to diagnose bad datasets")
    args = parser.parse_args()
    t0 = time.time()

    left_dir = Path(args.left)
    right_dir = Path(args.right)
    if not left_dir.is_dir() or not right_dir.is_dir():
        print("Error: both --left and --right must be existing directories.", file=sys.stderr)
        sys.exit(1)

    pairs = match_pairs(left_dir, right_dir)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    if not pairs:
        print("No matching image pairs found.", file=sys.stderr)
        sys.exit(1)

    t_pairing = time.time()
    print(f"Found {len(pairs)} matching pairs. Pairing time: {(t_pairing - t0):.2f}s")

    board_size = (args.pattern_size[0], args.pattern_size[1])  # (rows, cols)
    # Auto-orientation probe: try both orientations and pick the better one
    board_size = quick_orientation_probe(pairs, board_size)
    print(f"Using pattern size (rows, cols): {board_size[0]} {board_size[1]}")
    if args.fisheye:
        print("Using fisheye distortion model (recommended for >=120 deg lenses).")
    elif args.rational_model:
        print("Using rational 8-coefficient distortion model.")
    objp = build_object_points(board_size, args.square_size)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_size = None
    detect_stats = []
    coverage_heat = np.zeros((4, 4), dtype=np.int32)  # coarse grid to see coverage spread

    t_detect_start = time.time()
    # Optional: quality-based selection pre-pass
    if args.best_pairs and args.best_pairs > 0:
        t_q_start = time.time()
        scored: List[Tuple[float, Tuple[Path, Path]]] = []
        for lp, rp in pairs:
            li = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            ri = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if li is None or ri is None:
                continue
            ok_l, cl = find_corners(li, board_size)
            ok_r, cr = find_corners(ri, board_size)
            if not (ok_l and ok_r):
                continue
            # Sharpness via Laplacian variance
            sharp_l = cv2.Laplacian(li, cv2.CV_64F).var()
            sharp_r = cv2.Laplacian(ri, cv2.CV_64F).var()
            # Corner spread: bounding box area to prefer wider coverage
            def spread_score(corners: np.ndarray) -> float:
                xs = corners[:, 0, 0]
                ys = corners[:, 0, 1]
                return float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            spread_l = spread_score(cl)
            spread_r = spread_score(cr)
            # Total score: weighted sum
            score = 0.5 * (sharp_l + sharp_r) + 0.5 * (spread_l + spread_r)
            scored.append((score, (lp, rp)))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [pair for _, pair in scored[:args.best_pairs]]
        pairs = kept if kept else pairs
        t_q_end = time.time()
        print(f"Quality selection: kept {len(pairs)} pairs (scored {len(scored)}). Time: {(t_q_end - t_q_start):.2f}s")

    for i, (lp, rp) in enumerate(pairs, 1):
        left_img = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            print(f"Skipping pair {lp.name}: failed to read images")
            continue

        if image_size is None:
            image_size = (left_img.shape[1], left_img.shape[0])

        gl = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        ok_l, corners_l = find_corners(gl, board_size)
        ok_r, corners_r = find_corners(gr, board_size)
        if not (ok_l and ok_r):
            print(f"[{i}/{len(pairs)}] No corners found in pair {lp.name}")
            continue

        objpoints.append(objp)
        if args.fisheye:
            imgpoints_left.append(np.ascontiguousarray(corners_l, dtype=np.float64))
            imgpoints_right.append(np.ascontiguousarray(corners_r, dtype=np.float64))
        else:
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)

        # Diagnostics: sharpness and coverage
        lap_l = cv2.Laplacian(gl, cv2.CV_64F).var()
        lap_r = cv2.Laplacian(gr, cv2.CV_64F).var()
        def bbox_stats(c: np.ndarray, w: int, h: int):
            xs = c[:, 0, 0]
            ys = c[:, 0, 1]
            dx = float(xs.max() - xs.min())
            dy = float(ys.max() - ys.min())
            cx = float(xs.mean())
            cy = float(ys.mean())
            return dx, dy, cx, cy
        dx_l, dy_l, cx_l, cy_l = bbox_stats(corners_l, image_size[0], image_size[1])
        dx_r, dy_r, cx_r, cy_r = bbox_stats(corners_r, image_size[0], image_size[1])
        norm_cov_l = (dx_l / image_size[0]) * (dy_l / image_size[1])
        norm_cov_r = (dx_r / image_size[0]) * (dy_r / image_size[1])
        center_dist = np.hypot(cx_l - cx_r, cy_l - cy_r)
        scale_ratio = (dx_l * dy_l + 1e-9) / (dx_r * dy_r + 1e-9)
        detect_stats.append({
            "name": lp.name,
            "lap_l": lap_l,
            "lap_r": lap_r,
            "cov_l": norm_cov_l,
            "cov_r": norm_cov_r,
            "center_dist": center_dist,
            "scale_ratio": scale_ratio
        })
        # Heatmap (use left corners)
        gx = np.clip((corners_l[:, 0, 0] / image_size[0] * coverage_heat.shape[1]).astype(int), 0, coverage_heat.shape[1] - 1)
        gy = np.clip((corners_l[:, 0, 1] / image_size[1] * coverage_heat.shape[0]).astype(int), 0, coverage_heat.shape[0] - 1)
        for xh, yh in zip(gx, gy):
            coverage_heat[yh, xh] += 1

        if args.preview:
            preview_detected(left_img, corners_l, board_size)
            preview_detected(right_img, corners_r, board_size)

    t_detect_end = time.time()
    valid_count = len(objpoints)
    if valid_count < 5:
        print(f"Insufficient valid pairs with detected corners ({valid_count}). Need at least 5.", file=sys.stderr)
        sys.exit(1)

    print(f"Corner detection: {valid_count} valid pairs out of {len(pairs)} "
          f"({(t_detect_end - t_detect_start):.2f}s total, "
          f"{((t_detect_end - t_detect_start) / max(1, len(pairs))):.3f}s/pair avg)")
    print(f"Using {valid_count} valid pairs for calibration.")

    if args.analyze and detect_stats:
        def stats(arr):
            return np.min(arr), np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75), np.max(arr)
        lap_l = np.array([d["lap_l"] for d in detect_stats])
        lap_r = np.array([d["lap_r"] for d in detect_stats])
        cov_l = np.array([d["cov_l"] for d in detect_stats])
        cov_r = np.array([d["cov_r"] for d in detect_stats])
        cdist = np.array([d["center_dist"] for d in detect_stats])
        sratio = np.array([d["scale_ratio"] for d in detect_stats])
        print("\n=== Input diagnostics (use to spot bad pairs) ===")
        print("Sharpness (Laplacian var) Left  min/p25/med/p75/max:", " / ".join(f"{v:.1f}" for v in stats(lap_l)))
        print("Sharpness (Laplacian var) Right min/p25/med/p75/max:", " / ".join(f"{v:.1f}" for v in stats(lap_r)))
        print("Coverage fraction (area/img) Left  min/p25/med/p75/max:", " / ".join(f"{v:.3f}" for v in stats(cov_l)))
        print("Coverage fraction (area/img) Right min/p25/med/p75/max:", " / ".join(f"{v:.3f}" for v in stats(cov_r)))
        print("Center distance between L/R (px) min/p25/med/p75/max:", " / ".join(f"{v:.1f}" for v in stats(cdist)))
        print("Scale ratio L/R (area) min/p25/med/p75/max:", " / ".join(f"{v:.2f}" for v in stats(sratio)))
        print("Heatmap (rows=top->bottom, cols=left->right) counts of corners on left image:")
        with np.printoptions(formatter={"int": lambda x: f"{x:3d}"}):
            print(coverage_heat)
        print("Heuristics:")
        print("- Coverage fraction med should be reasonably high (e.g., >0.10). Low values mean board is tiny/centered.")
        print("- Center distance should vary; all near-zero suggests almost no parallax or identical frames.")
        print("- Scale ratio far from 1.0 or wildly varying can indicate mismatched pairs or different zoom.")
        print("- Heatmap should show counts across the grid; if clustered, add poses near corners/edges.")

    # Fisheye expects (N,1,3) object points; standard model accepts (N,3)
    objpoints_use = [p.reshape(-1, 1, 3).astype(np.float64) for p in objpoints] if args.fisheye else objpoints

    t_single_start = time.time()
    err_l, K1, D1, rvecs1, tvecs1 = calibrate_single(
        objpoints_use, imgpoints_left, image_size, use_fisheye=args.fisheye, use_rational=args.rational_model
    )
    err_r, K2, D2, rvecs2, tvecs2 = calibrate_single(
        objpoints_use, imgpoints_right, image_size, use_fisheye=args.fisheye, use_rational=args.rational_model
    )
    t_single_end = time.time()
    print(f"Single-eye calibration time: {(t_single_end - t_single_start):.2f}s")
    print(f"Single-eye reprojection RMS: left={err_l:.4f}, right={err_r:.4f}")

    t_stereo_start = time.time()
    s_err, K1, D1, K2, D2, R, T, E, F = stereo_calibrate(
        objpoints_use, imgpoints_left, imgpoints_right, K1, D1, K2, D2, image_size, use_fisheye=args.fisheye
    )
    t_stereo_end = time.time()
    print(f"Stereo calibration time: {(t_stereo_end - t_stereo_start):.2f}s")
    print(f"Stereo reprojection RMS: {s_err:.4f}")
    print(f"Baseline (m): {np.linalg.norm(T):.6f}")

    t_rect_start = time.time()
    R1, R2, P1, P2, Q, roi1, roi2 = stereo_rectify(K1, D1, K2, D2, image_size, R, T, use_fisheye=args.fisheye)
    if args.fisheye:
        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    else:
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    t_rect_end = time.time()
    print(f"Rectification + maps time: {(t_rect_end - t_rect_start):.2f}s")

    if args.preview:
        for lp, rp in pairs:
            left_img = cv2.imread(str(lp), cv2.IMREAD_COLOR)
            right_img = cv2.imread(str(rp), cv2.IMREAD_COLOR)
            if left_img is None or right_img is None:
                continue
            preview_rectification(left_img, right_img, map1x, map1y, map2x, map2y)
            break
        print("Press any key in the preview windows to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save YAMLs to the common root of the input folders
    common_root = Path(os.path.commonpath([left_dir, right_dir]))
    out_dir = common_root
    left_yaml = out_dir / f"{args.save_prefix}_left.yaml"
    right_yaml = out_dir / f"{args.save_prefix}_right.yaml"
    stereo_yaml = out_dir / f"{args.save_prefix}_stereo.yaml"

    left_data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": {"rows": 3, "cols": 3, "data": to_list(K1)},
        "distortion_coefficients": {"rows": int(D1.shape[0]), "cols": 1, "data": to_list(D1)},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": to_list(R1)},
        "projection_matrix": {"rows": 3, "cols": 4, "data": to_list(P1)},
        "rms": float(err_l),
        "roi": {"x": int(roi1[0]), "y": int(roi1[1]), "width": int(roi1[2]), "height": int(roi1[3])},
    }
    right_data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": {"rows": 3, "cols": 3, "data": to_list(K2)},
        "distortion_coefficients": {"rows": int(D2.shape[0]), "cols": 1, "data": to_list(D2)},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": to_list(R2)},
        "projection_matrix": {"rows": 3, "cols": 4, "data": to_list(P2)},
        "rms": float(err_r),
        "roi": {"x": int(roi2[0]), "y": int(roi2[1]), "width": int(roi2[2]), "height": int(roi2[3])},
    }
    stereo_data = {
        "stereo_rms": float(s_err),
        "R": {"rows": 3, "cols": 3, "data": to_list(R)},
        "T": {"rows": 3, "cols": 1, "data": to_list(T)},
        "E": {"rows": 3, "cols": 3, "data": to_list(E)},
        "F": {"rows": 3, "cols": 3, "data": to_list(F)},
        "Q": {"rows": 4, "cols": 4, "data": to_list(Q)},
        "baseline_m": float(np.linalg.norm(T)),
    }

    t_save_start = time.time()
    try:
        save_yaml(left_yaml, left_data)
        save_yaml(right_yaml, right_data)
        save_yaml(stereo_yaml, stereo_data)
        print(f"Saved:\n  {left_yaml}\n  {right_yaml}\n  {stereo_yaml}")
    except ImportError:
        np.savez(out_dir / f"{args.save_prefix}_calib_npz.npz",
                 K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
        print("PyYAML not installed; saved NumPy .npz instead.")
    t_save_end = time.time()
    print(f"Saving outputs time: {(t_save_end - t_save_start):.2f}s")
    print(f"Total time: {(t_save_end - t0):.2f}s")


if __name__ == "__main__":
    main()
