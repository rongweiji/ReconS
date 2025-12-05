#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

import cv2
import numpy as np

# --- CONFIGURATION MATCHING C++ ---
# C++: cv::Size(5, 5) -> Python: (5, 5)
SUBPIX_WIN_SIZE = (5, 5) 
# C++: TermCriteria(EPS | MAX_ITER, 30, 0.1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# C++: CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS

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
    # Using FILTER_QUADS to match C++
    ret, corners = cv2.findChessboardCorners(img_gray, board_size, FIND_FLAGS)
    if not ret:
        return False, None
    
    # Subpixel refinement
    corners_refined = cv2.cornerSubPix(img_gray, corners, SUBPIX_WIN_SIZE, (-1, -1), SUBPIX_CRITERIA)
    return True, corners_refined

def build_object_points_cpp_style(rows: int, cols: int, square_size: float) -> np.ndarray:
    """
    Replicates the exact loop order of the C++ code:
    for( int i = 0; i < board_height; ++i )
      for( int j = 0; j < board_width; ++j )
         push_back(Point3d(j*size, i*size, 0))
    """
    objp = np.zeros((rows * cols, 1, 3), dtype=np.float64)
    idx = 0
    for i in range(rows):      # Height / Y
        for j in range(cols):  # Width / X
            objp[idx, 0, 0] = float(j) * square_size
            objp[idx, 0, 1] = float(i) * square_size
            objp[idx, 0, 2] = 0.0
            idx += 1
    return objp

def stereo_calibrate_fisheye(object_points, image_points_left, image_points_right, image_size):
    # --- CRITICAL FIX 1: Initial Guess for K ---
    w, h = image_size
    K_guess = np.array([[w, 0, w/2.0], 
                        [0, w, h/2.0], 
                        [0, 0, 1.0]], dtype=np.float64)
    
    K1 = K_guess.copy()
    K2 = K_guess.copy()
    D1 = np.zeros((4, 1), dtype=np.float64)
    D2 = np.zeros((4, 1), dtype=np.float64)
    
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | \
            cv2.fisheye.CALIB_CHECK_COND | \
            cv2.fisheye.CALIB_FIX_SKEW | \
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-6)
    
    print("Running calibration solver...")
    try:
        # --- CRITICAL FIX 2: Handle Variable Return Values ---
        # OpenCV 4.12+ returns (retval, K1, D1, K2, D2, R, T, rvecs, tvecs)
        # Older OpenCV returns (retval, K1, D1, K2, D2, R, T)
        # this original source basded on the linux c++ opencv version : https://github.com/sourishg/fisheye-stereo-calibration is 4.6 based
        # We capture everything into a tuple 'result' and slice the first 7 items.
        result = cv2.fisheye.stereoCalibrate(
            object_points,
            image_points_left,
            image_points_right,
            K1, D1,
            K2, D2,
            image_size,
            flags=flags,
            criteria=criteria
        )
        
        # Unpack only the first 7 values
        retval, K1, D1, K2, D2, R, T = result[:7]

    except cv2.error as e:
        print("\n[ERROR] Calibration failed inside OpenCV.")
        print(e)
        raise e

    return retval, K1, D1, K2, D2, R, T

def stereo_rectify(K1, D1, K2, D2, image_size, R, T):
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, 
        flags=cv2.CALIB_ZERO_DISPARITY, 
        balance=0.0, 
        fov_scale=1.1,
        newImageSize=image_size
    )
    roi1 = roi2 = (0, 0, image_size[0], image_size[1])
    return R1, R2, P1, P2, Q, roi1, roi2

def save_yaml(path: Path, data: dict):
    try:
        import yaml
    except ImportError:
        # Fallback to manual string writing if PyYAML isn't there, 
        # but usually CV2 users have it. 
        print("PyYAML not found. Printing to console instead.")
        print(data)
        return

    # Convert numpy arrays to lists for Clean YAML
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

def preview_rectification(left_img, right_img, map1x, map1y, map2x, map2y):
    left_rect = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    
    combined = np.hstack([left_rect, right_rect])
    h, w = combined.shape[:2]
    # Draw green lines
    for y in range(0, h, 30):
        cv2.line(combined, (0, y), (w - 1, y), (0, 255, 0), 1)
        
    cv2.imshow("Rectified Preview (Green lines should be horizontal)", combined)
    cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser(description="Updated Python Stereo Calib (Fixing Zero-Division)")
    parser.add_argument("--left", required=True, help="Path to left images")
    parser.add_argument("--right", required=True, help="Path to right images")
    parser.add_argument("--pattern-size", nargs=2, type=int, required=True, metavar=("ROWS", "COLS"))
    parser.add_argument("--square-size", type=float, required=True, help="Meters (e.g., 0.108)")
    parser.add_argument("--preview", action="store_true", help="Visualize corners")
    parser.add_argument("--save-prefix", default="stereo", help="Output filename prefix")
    args = parser.parse_args()

    left_dir = Path(args.left)
    right_dir = Path(args.right)
    
    pairs = match_pairs(left_dir, right_dir)
    if not pairs:
        print("No matching images found.")
        sys.exit(1)
        
    print(f"Found {len(pairs)} pairs.")
    
    # NOTE: User input is typically "Rows Cols" (e.g., 8 6)
    # OpenCV convention is (Cols, Rows) for findChessboardCorners
    rows = args.pattern_size[0]
    cols = args.pattern_size[1]
    board_size_cv = (cols, rows) 
    
    # Create Object Points exactly like C++ loop
    objp = build_object_points_cpp_style(rows, cols, args.square_size)
    
    objpoints = [] 
    imgpoints_left = []
    imgpoints_right = []
    image_size = None
    
    valid_pairs = 0
    
    print("Detecting corners...")
    for lp, rp in pairs:
        img_l = cv2.imread(str(lp))
        img_r = cv2.imread(str(rp))
        if img_l is None or img_r is None: continue
        
        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])
            
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        found_l, corners_l = find_corners(gray_l, board_size_cv)
        found_r, corners_r = find_corners(gray_r, board_size_cv)
        
        if found_l and found_r:
            valid_pairs += 1
            # Force Strictly Typed Float64 (Double) to match C++ Point2d
            objpoints.append(objp) 
            imgpoints_left.append(corners_l.astype(np.float64))
            imgpoints_right.append(corners_r.astype(np.float64))
            
            if args.preview:
                preview_detected(img_l, corners_l, board_size_cv)
        else:
            print(f" - Failed detection: {lp.name}")

    cv2.destroyAllWindows()

    if valid_pairs < 5:
        print("Error: Not enough valid pairs (<5).")
        sys.exit(1)

    print(f"Calibrating with {valid_pairs} pairs...")
    
    retval, K1, D1, K2, D2, R, T = stereo_calibrate_fisheye(
        objpoints, imgpoints_left, imgpoints_right, image_size
    )
    
    print(f"\nCalibration Success!")
    print(f"Stereo RMS: {retval:.4f}")
    print(f"Baseline: {np.linalg.norm(T):.4f} units")
    
    # Rectification
    print("Computing rectification maps...")
    R1, R2, P1, P2, Q, roi1, roi2 = stereo_rectify(K1, D1, K2, D2, image_size, R, T)
    
    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)
    
    if args.preview:
        print("Showing rectified result. Press any key to close.")
        img_l = cv2.imread(str(pairs[0][0]))
        img_r = cv2.imread(str(pairs[0][1]))
        preview_rectification(img_l, img_r, map1x, map1y, map2x, map2y)
        cv2.destroyAllWindows()

    # Save Results
    out_dir = Path(os.path.commonpath([left_dir, right_dir]))
    out_file = out_dir / f"{args.save_prefix}_result.yaml"
    
    data = {
        "K1": K1, "D1": D1,
        "K2": K2, "D2": D2,
        "R": R, "T": T,
        "R1": R1, "R2": R2,
        "P1": P1, "P2": P2,
        "Q": Q,
        "rms": retval
    }
    
    save_yaml(out_file, data)
    print(f"Saved calibration to {out_file}")

if __name__ == "__main__":
    main()