Stereo + Mono Calibration (OpenCV, no ROS)

Checkerboard
- Inner corners: 8 rows x 6 cols (cross points). Use 6 8 if your board is rotated.
- Square size: 27 mm (0.027 m)
- Printed on letter paper; original 108 mm squares scaled to 25% → 27 mm.

Prepare Data
- Stereo: synchronized image pairs in two folders (left/right) with matching filenames.
- Mono: a single folder of images.
- Example paths:
	- `cam-calib/cali_data1/workspace_cali/front_stereo_cam_left`
	- `cam-calib/cali_data1/workspace_cali/front_stereo_cam_right`
	- `cam-calib/cali_data1/workspace_cali/front_stereo_cam_left`

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
python "g:\GithubProject\ReconS\cam-calib\stereo_calibrate_opencv.py" `
		--left  "g:\GithubProject\ReconS\cam-calib\cali_data1\workspace_cali\front_stereo_cam_left" `
		--right "g:\GithubProject\ReconS\cam-calib\cali_data1\workspace_cali\front_stereo_cam_right" `
		--pattern-size 8 6 `   # inner corners (rows cols); use 6 8 if rotated
		--square-size 0.027 `  # meters
		--preview              # optional; shows detection/rectification windows

# Fisheye-only mono calibration
python "g:\GithubProject\ReconS\cam-calib\mono_calibrate_opencv.py" `
		--images "g:\GithubProject\ReconS\cam-calib\cali_data1\workspace_cali\front_stereo_cam_left" `
		--pattern-size 8 6 `   # inner corners (rows cols); use 6 8 if rotated
		--square-size 0.027 `  # meters
		--preview              # optional; shows detection/undistortion windows
		--analyze              # optional; prints coverage/sharpness summary
```

Output
Stereo output
- One YAML saved alongside the left/right folders: `{save_prefix}_result.yaml` (default `stereo_result.yaml`).
- Intrinsics (per camera):
	- `K1`, `K2`: 3x3 camera matrices `[fx 0 cx; 0 fy cy; 0 0 1]`.
	- `D1`, `D2`: fisheye distortion coefficients `[k1, k2, k3, k4]`.
- Extrinsics (stereo relationship):
	- `R`: 3x3 rotation from cam1 to cam2.
	- `T`: 3x1 translation from cam1 to cam2 (meters); baseline = `norm(T)`.
- Rectification and projection (used for undistort/rectify + depth):
	- `R1`, `R2`: rectification rotations applied to each camera.
	- `P1`, `P2`: 3x4 projection matrices after rectification (contain adjusted focal lengths, principals, and baseline shift in `P2[0,3]`).
	- `Q`: 4x4 reprojection matrix to convert disparity to 3D points.
- Quality metric:
	- `rms`: stereo reprojection RMS reported by OpenCV (lower is better).

Mono output
- One YAML saved alongside the images folder: `{images_folder}_result.yaml` (or `{save_prefix}_result.yaml` if provided).
- Intrinsics:
	- `K`: 3x3 camera matrix `[fx 0 cx; 0 fy cy; 0 0 1]`.
	- `D`: fisheye distortion coefficients `[k1, k2, k3, k4]`.
- Metadata:
	- `image_size`: `[width, height]`.
	- `rms`: mono reprojection RMS reported by OpenCV (lower is better).

Console Notes
- Shows matched/valid images, RMS, and baseline (stereo).
- Images with failed corner detection are skipped; at least 5 valid images/pairs are required.
- Mono: use --analyze to print coverage and sharpness summary for valid images.

Tips
- Keep the board large and move it to edges/corners with varied tilt/depth for better fisheye conditioning.
- Ensure left/right images are synchronized and the same resolution.
- If detection fails, double-check `--pattern-size` and `--square-size`.

Camera used: 1080p USB2.0 UVC camera with ~130° wide angle (purchase link: https://www.amazon.com/dp/B0CNCSFQC1?ref=ppx_yo2ov_dt_b_fed_asin_title).

sample include the 640x480 size and 20-30. support camera 130 degree width
