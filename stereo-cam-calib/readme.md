Stereo Calibration (OpenCV, no ROS)

Checkerboard
- Inner corners: 8 rows x 6 cols (cross points). Use 6 8 if your board is rotated.
- Square size: 27 mm (0.027 m)
- Printed on letter paper; original 108 mm squares scaled to 25% → 27 mm.

Prepare Data
- Place synchronized image pairs in two folders (left/right) with matching filenames.
- Example paths:
	- `stereo-cam-calib/cali_data1/workspace_cali/front_stereo_cam_left`
	- `stereo-cam-calib/cali_data1/workspace_cali/front_stereo_cam_right`

Dependencies
- Python 3.9+ recommended
- `opencv-python`
- `PyYAML` (for YAML output)
Install via pip or conda, e.g.:
```powershell
python -m pip install opencv-python PyYAML
# or
conda install -n recons -c conda-forge opencv pyyaml
```

Run Calibration (Windows PowerShell)
```powershell
# Activate your environment if needed
# conda activate recons

# Install dependencies (if not installed yet)
# conda install -n recons -c conda-forge opencv pyyaml
# or
# python -m pip install opencv-python PyYAML

# Fisheye-only stereo calibration
python "g:\GithubProject\ReconS\stereo-cam-calib\stereo_calibrate_opencv.py" `
		--left  "g:\GithubProject\ReconS\stereo-cam-calib\cali_data1\workspace_cali\front_stereo_cam_left" `
		--right "g:\GithubProject\ReconS\stereo-cam-calib\cali_data1\workspace_cali\front_stereo_cam_right" `
		--pattern-size 8 6 `   # inner corners (rows cols); use 6 8 if rotated
		--square-size 0.027 `  # meters
		--max-pairs 0 `        # 0 = use all matched pairs
		--preview              # optional; shows detection/rectification windows
		--analyze              # optional; prints coverage/sharpness stats
```

Output
- One YAML saved alongside the left/right folders: `{save_prefix}_result.yaml` (default `stereo_result.yaml`).
- Fields inside the YAML (float64):
	- `K1`, `D1`, `K2`, `D2`: intrinsics and fisheye distortion for left/right.
	- `R`, `T`: stereo rotation and translation (baseline = `norm(T)`, meters).
	- `R1`, `R2`, `P1`, `P2`, `Q`: rectification, projection, and disparity-to-depth mapping.
	- `rms`: stereo reprojection RMS reported by OpenCV.

Console Notes
- Shows matched/valid pairs, detection time, RMS, baseline, and rectification time.
- Pairs with failed corner detection are skipped; at least 5 valid pairs are required.

Tips
- Keep the board large and move it to edges/corners with varied tilt/depth for better fisheye conditioning.
- Ensure left/right images are synchronized and the same resolution.
- If detection fails, double-check `--pattern-size` and `--square-size`.
- Use `--max-pairs` to limit to a clean subset while debugging.

Camera used: 1080p USB2.0 UVC camera with ~130° wide angle (purchase link: https://www.amazon.com/dp/B0CNCSFQC1?ref=ppx_yo2ov_dt_b_fed_asin_title).