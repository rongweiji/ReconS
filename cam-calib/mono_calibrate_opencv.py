#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# --- CONFIGURATION MATCHING C++ ---
SUBPIX_WIN_SIZE = (5, 5)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files[p.name] = p
    return files


def find_corners(img_gray: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, np.ndarray]:
    ret, corners = cv2.findChessboardCorners(img_gray, board_size, FIND_FLAGS)
    if not ret:
        return False, None

    corners_refined = cv2.cornerSubPix(img_gray, corners, SUBPIX_WIN_SIZE, (-1, -1), SUBPIX_CRITERIA)
    return True, corners_refined


def build_object_points_cpp_style(rows: int, cols: int, square_size: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 1, 3), dtype=np.float64)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            objp[idx, 0, 0] = float(j) * square_size
            objp[idx, 0, 1] = float(i) * square_size
            objp[idx, 0, 2] = 0.0
            idx += 1
    return objp


def mono_calibrate_fisheye(object_points, image_points, image_size):
    w, h = image_size
    K_guess = np.array([[w, 0, w / 2.0],
                        [0, w, h / 2.0],
                        [0, 0, 1.0]], dtype=np.float64)

    K = K_guess.copy()
    D = np.zeros((4, 1), dtype=np.float64)

    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | \
            cv2.fisheye.CALIB_CHECK_COND | \
            cv2.fisheye.CALIB_FIX_SKEW | \
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-6)

    print("Running calibration solver...")
    try:
        result = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            K, D,
            None, None,
            flags=flags,
            criteria=criteria
        )

        retval, K, D, rvecs, tvecs = result[:5]
    except cv2.error as e:
        print("\n[ERROR] Calibration failed inside OpenCV.")
        print(e)
        raise e

    return retval, K, D, rvecs, tvecs


def save_yaml(path: Path, data: dict):
    try:
        import yaml
    except ImportError:
        print("PyYAML not found. Printing to console instead.")
        print(data)
        return

    clean_data = {}
    for k, v in data.items():
        if hasattr(v, "tolist"):
            clean_data[k] = v.tolist()
        else:
            clean_data[k] = v

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean_data, f, sort_keys=False)


def preview_detected(img_color, corners, board_size):
    vis = img_color.copy()
    cv2.drawChessboardCorners(vis, board_size, corners, True)
    cv2.imshow("Corners (Press Key)", vis)
    cv2.waitKey(100)


def build_undistort_maps(K, D, image_size):
    R = np.eye(3, dtype=np.float64)
    new_K = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, new_K, image_size, cv2.CV_16SC2
    )
    return map1, map2


def preview_undistortion_sequence(images: List[Path], map1, map2):
    idx = 0
    total = len(images)
    if total == 0:
        return

    while True:
        ip = images[idx]
        img = cv2.imread(str(ip))
        if img is None:
            text = f"[{idx+1}/{total}] Failed to read {ip.name}"
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Undistorted Preview (n/p to navigate, q to quit)", blank)
        else:
            undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([img, undist])
            h, w = combined.shape[:2]
            for y in range(0, h, 30):
                cv2.line(combined, (0, y), (w - 1, y), (0, 255, 0), 1)
            cv2.putText(combined, f"{idx+1}/{total}: {ip.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Undistorted Preview (n/p to navigate, q to quit)", combined)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('n'):
            idx = (idx + 1) % total
        elif key == ord('p'):
            idx = (idx - 1 + total) % total
        else:
            continue
    cv2.destroyAllWindows()


def compute_coverage_metrics(corners: np.ndarray, image_size: Tuple[int, int]) -> Tuple[float, float, float]:
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    w, h = image_size
    coverage_x = float(xs.max() - xs.min()) / float(w)
    coverage_y = float(ys.max() - ys.min()) / float(h)
    coverage_area = coverage_x * coverage_y
    return coverage_x, coverage_y, coverage_area


def compute_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def format_stats(values: List[float], percent: bool = False) -> str:
    if not values:
        return "n/a"
    arr = np.array(values, dtype=np.float64)
    if percent:
        arr = arr * 100.0
        return f"min {arr.min():.1f}%, mean {arr.mean():.1f}%, max {arr.max():.1f}%"
    return f"min {arr.min():.3f}, mean {arr.mean():.3f}, max {arr.max():.3f}"


def main():
    parser = argparse.ArgumentParser(description="Single Camera Fisheye Calibration (OpenCV)")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--pattern-size", nargs=2, type=int, required=True, metavar=("ROWS", "COLS"))
    parser.add_argument("--square-size", type=float, required=True, help="Meters (e.g., 0.108)")
    parser.add_argument("--preview", action="store_true", help="Visualize corners and undistortion")
    parser.add_argument("--analyze", action="store_true", help="Print coverage and sharpness summary")
    parser.add_argument("--save-prefix", default=None, help="Output filename prefix (default: images folder name)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Folder not found: {images_dir}")
        sys.exit(1)

    image_files = list(list_images(images_dir).values())
    if not image_files:
        print("No images found.")
        sys.exit(1)

    print(f"Found {len(image_files)} images.")

    rows = args.pattern_size[0]
    cols = args.pattern_size[1]
    board_size_cv = (cols, rows)

    objp = build_object_points_cpp_style(rows, cols, args.square_size)
    objpoints = []
    imgpoints = []
    image_size = None
    valid_images = []
    coverage_x_values = []
    coverage_y_values = []
    coverage_area_values = []
    sharpness_values = []

    print("Detecting corners...")
    for ip in image_files:
        img = cv2.imread(str(ip))
        if img is None:
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != image_size:
            print(f" - Skipping (size mismatch): {ip.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, board_size_cv)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners.astype(np.float64))
            valid_images.append(ip)

            if args.analyze:
                coverage_x, coverage_y, coverage_area = compute_coverage_metrics(corners, image_size)
                coverage_x_values.append(coverage_x)
                coverage_y_values.append(coverage_y)
                coverage_area_values.append(coverage_area)
                sharpness_values.append(compute_sharpness(gray))

            if args.preview:
                preview_detected(img, corners, board_size_cv)
        else:
            print(f" - Failed detection: {ip.name}")

    cv2.destroyAllWindows()

    if args.analyze:
        print("\nAnalysis summary (valid images only):")
        print(f"Coverage X (% of width):  {format_stats(coverage_x_values, percent=True)}")
        print(f"Coverage Y (% of height): {format_stats(coverage_y_values, percent=True)}")
        print(f"Coverage area (%):        {format_stats(coverage_area_values, percent=True)}")
        print(f"Sharpness (Laplacian var): {format_stats(sharpness_values)}")

    if len(valid_images) < 5:
        print("Error: Not enough valid images (<5).")
        sys.exit(1)

    print(f"Calibrating with {len(valid_images)} images...")
    retval, K, D, rvecs, tvecs = mono_calibrate_fisheye(objpoints, imgpoints, image_size)

    print("\nCalibration Success!")
    print(f"RMS: {retval:.4f}")

    if args.preview:
        print("Preview undistortion: press 'n'/'p' to navigate, 'q' or ESC to quit.")
        map1, map2 = build_undistort_maps(K, D, image_size)
        preview_undistortion_sequence(valid_images, map1, map2)

    save_prefix = args.save_prefix or images_dir.name
    out_file = images_dir.parent / f"{save_prefix}_result.yaml"
    data = {
        "K": K,
        "D": D,
        "image_size": list(image_size),
        "rms": retval
    }
    save_yaml(out_file, data)
    print(f"Saved calibration to {out_file}")


if __name__ == "__main__":
    main()
