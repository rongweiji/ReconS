Stereo Calibration (OpenCV, no ROS)

Checkerboard
- Inner corners: 8 rows x 6 cols (cross point count, not squares)
- Square size: 27 mm (0.027 m)
- Printed on letter size paper (original 108 mm squares, scaled to 25% → 27 mm)

Prepare Data
- Place synchronized image pairs in two folders (left/right) with matching filenames.
- Example paths:
	- `stereo-cam-calib/cali_data1/workspace3/front_stereo_cam_left`
	- `stereo-cam-calib/cali_data1/workspace3/front_stereo_cam_right`

Run Calibration (Windows PowerShell)
```powershell
# Activate your environment if needed
# conda activate recons

# Install dependencies (if not installed yet)
# conda install -n recons -c conda-forge opencv pyyaml
# or
# python -m pip install opencv-python PyYAML

# Run with correct checkerboard spec
python "g:\GithubProject\ReconS\stereo-cam-calib\stereo_calibrate_opencv.py" `
	--left  "g:\GithubProject\ReconS\stereo-cam-calib\cali_data1\workspace3\front_stereo_cam_left" `
	--right "g:\GithubProject\ReconS\stereo-cam-calib\cali_data1\workspace3\front_stereo_cam_right" `
	--pattern-size 8 6 `
	--square-size 0.027
```

Outputs
- YAML files saved to the common root of the input folders (e.g., `workspace3`):
	- `stereo_left.yaml` — left camera intrinsics, distortion, rectification, projection
	- `stereo_right.yaml` — right camera intrinsics, distortion, rectification, projection
	- `stereo_stereo.yaml` — stereo R/T/E/F, Q, baseline

Console Metrics
- Pairing time: time to match filenames across folders
- Corner detection: number of valid pairs, total and per-pair time
- Single-eye reprojection RMS: pixel error per camera (lower is better)
- Stereo reprojection RMS: pixel error across both cameras (should be similar magnitude to single-eye RMS)
- Baseline (m): distance between camera centers (norm of T)
- Rectification + maps time: generating undistort/rectify maps
- Saving outputs time and Total time

Tips
- Use full board visibility, sharp focus, varied poses/tilts.
- Ensure identical resolution between left/right images.
- If detection fails, verify `--pattern-size` (8 6) and `--square-size` (0.027).
- For faster runs, you can limit pairs (e.g., `--max-pairs 150`).