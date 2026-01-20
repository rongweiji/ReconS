# ReconS

End-to-end RGBD pipeline: depth → PyCuVSLAM → nvblox.

References:
- nvblox: https://nvidia-isaac.github.io/nvblox/index.html
- Neural reconstruction stereo (NuRec): https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html
- Depth generation (Depth Anything TRT): https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt

## Environment
- Use the `pycuvslam` conda env (has `cuvslam`). Install the rest into the same env:
  ```bash
  conda activate pycuvslam
  python -m pip install "torch==<cu12x build>" -f https://download.pytorch.org/whl/torch_stable.html
  python -m pip install <nvblox_torch_wheel>  # matching CUDA build
  python -m pip install rerun-sdk opencv-python numpy
  ```
  Replace `<cu12x build>` and wheel path with the CUDA build you have (cu12 wheels run on CUDA 13 drivers).
- On WSL add GUI libs if needed and keep `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH` for cuvslam.

## Data layout (per sample)
```
data/sample_xxx/
  iphone_mono/             # RGB frames (0000001.png ...)
  iphone_mono_depth/       # Depth frames aligned to RGB (uint16, mm)
  iphone_calibration.yaml  # Pinhole K
  timestamps.txt           # frame,timestamp_ns
```

Outputs land alongside the sample:
- `cuvslam_poses.tum` (and `cuvslam_poses_slam.tum` if SLAM enabled)
- `nvblox_out/mesh.ply` (+ voxel exports)
- depth folder is regenerated if missing.

## One-shot pipeline
```bash
python3 run_full_pipeline.py \
  --dataset data/sample_20260119_i4 \
  --nvblox-mode colormesh \
  --nvblox-ui   # drop if headless
```
This runs depth generation, PyCuVSLAM, builds nvblox artifacts, and runs nvblox with Rerun UI when `--nvblox-ui` is set.

## Individual runners
See `pipelines/README.md` for per-step commands (`run_pycuvslam_rgbd.py`, `run_nvblox.py`, stereo variant, etc.).
