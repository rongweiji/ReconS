# Depth Feature (Depth Anything TensorRT)

This module provides an offline depth inference tool using the Depth Anything v3
TensorRT engine and the C++ wrapper used in `/mnt/g/GithubProject/xlernav/non_ros`.
It takes a folder of RGB frames and writes depth images to an output folder.

## Requirements

- ROS 2 (for `sensor_msgs`/`std_msgs` headers and runtime libs)
- CUDA + TensorRT
- OpenCV 4
- The Depth Anything TensorRT engine and shared library (vendored here)

If ROS 2 is installed in a non-default location, set `ROS_PREFIX` before running
or pass `-DCMAKE_PREFIX_PATH` when configuring.

## Build + Run

```bash
bash depth_feature/run_depth_from_rgb.sh \
  --rgb-dir data/sample_20260117_205753/left \
  --out-dir data/sample_20260117_205753/left_depth \
  --calibration data/sample_20260117_205753/left_calibration.yaml
```

Common options:

```bash
--engine PATH        TensorRT engine file (default: depth_feature/models/DA3METRIC-LARGE.trt10.engine)
--calibration PATH   Calibration YAML (camera_info or K1/D1/R1/P1) (required)
--camera-info PATH   Alias for --calibration
--depth-scale VALUE  Scale meters to uint16 (default: 1000.0)
--save-exr           Save float32 depth .exr instead of 16-bit PNG
--skip-existing      Skip frames with existing outputs
--max-frames N       Process only first N frames
--ext LIST           Comma-separated extensions (default jpg,jpeg,png,bmp,tif,tiff)
--preview            Show RGB + depth preview while processing
```

If you are running inside WSL and encounter CUDA driver issues, the script
adds `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH` automatically.

If `--out-dir` is omitted, outputs go to a sibling folder named
`<rgb_dir>_depth` (for example, `left_depth`).

command example 
bash depth_feature/run_depth_from_rgb.sh \
    --rgb-dir data/sample_20260119/iphone_mono \
    --out-dir data/sample_20260119/iphone_mono_depth \
    --calibration data/sample_20260119/iphone_calibration