# ORB-SLAM3 RGB-D Wrapper

Run ORB-SLAM3 on RGB + depth frame folders using the `run_orbslam3_rgbd.sh`
helper (builds ORB-SLAM3 and this wrapper if needed).

## Requirements
- `third_party/ORB_SLAM3` present (with `Vocabulary/ORBvoc.txt` inside).
- OpenCV, Eigen, Pangolin (handled by the build), C++17 toolchain.
- Depth images scaled in meters (`--depth-scale` adjusts if they are in mm).

## Quick start (sample iPhone data)
From repo root:
```bash
bash orbslam3_rgbd/run_orbslam3_rgbd.sh \
  --rgb-dir data/sample_20260119/iphone_mono \
  --depth-dir data/sample_20260119/iphone_mono_depth \
  --calibration data/sample_20260119/iphone_calibration.yaml \
  --timestamps data/sample_20260119/timestamps.txt \
  --depth-scale 1000 \
  --distortion-model brown \
  --viewer
```
Outputs go next to the RGB folder: `orbslam3_poses.tum`, `orbslam3_runtime.yaml`
and (if `--save-keyframes`) `orbslam3_keyframes.tum`.

### Rerun visualization (RGB + depth + trajectory)
Replay an existing ORB-SLAM3 run in the Rerun UI:
```bash
python3 orbslam3_rgbd/run_orbslam3_rerun.py \
  --rgb-dir data/sample_20260119_125703/left \
  --depth-dir data/sample_20260119_125703/left_depth \
  --trajectory data/sample_20260119_125703/orbslam3_poses.tum \
  --calibration data/sample_20260119_125703/left_calibration.yaml \
  --timestamps data/sample_20260119_125703/timestamps.txt \
  --depth-scale 1000 \
  --fps 30 \
  --spawn
```
Requirements: `python3 -m pip install rerun-sdk opencv-python pyyaml numpy`.
If you omit `--timestamps`, playback uses a fixed FPS; with timestamps it uses
recorded times. `--spawn` starts a viewer; otherwise it connects to an existing
Rerun viewer.

## Common options
- `--distortion-model brown|fisheye|equidistant` overrides the YAML model.
- `--rgb-ext/--depth-ext` change expected extensions (default `.jpg`/`.png`).
- `--camera-fps` forces FPS in the generated config (otherwise inferred).
- `--undistort` (default) / `--no-undistort` to toggle undistortion.
- `--viewer` to show Pangolin visualization.
- `--rerun` enables Rerun logging (C++ SDK); add `--rerun-spawn` to auto-launch a viewer or `--rerun-addr` to connect to an existing one.

## Notes
- The script auto-patches ORB-SLAM3 to build with C++14+ if needed.
- If the vocabulary is elsewhere, pass `--vocab /path/to/ORBvoc.txt`.
- Timestamps must be CSV with header `frame,timestamp_ns` where `frame` is the
  basename (without extension) matching both RGB and depth files.


command example : 
bash orbslam3_rgbd/run_orbslam3_rgbd.sh     --rgb-dir data/sample_20260119/iphone_mono     --depth-dir data/sample_20260119/iphone_mono_depth     --calibration data/sample_20260119/iphone_calibration.yaml     --timestamps data/sample_20260119/timestamps.txt     --depth-scale 1000     --distortion-model brown     --rgb-ext .png     --depth-ext .png     --viewer
