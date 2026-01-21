# ReconS

End-to-end RGBD pipeline: depth → PyCuVSLAM → nvblox.

Demo : 



https://github.com/user-attachments/assets/c316bae7-069f-442d-bf2f-6c98325dd0ef





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
- NVIDIA driver: PyCuVSLAM requires a driver exposing CUDA ≥12.6 (R560+; CUDA 13 is OK). Check with `nvidia-smi`.
- On WSL add GUI libs if needed and keep `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH` so cuvslam loads the WSL libcuda shim instead of older toolkit stubs.
- If `conda` is missing after install, run `~/miniconda3/bin/conda init bash`, then `source ~/.bashrc` (or reopen a shell) so `conda` is on your PATH.

## Data layout (per sample)
```
data/sample_xxx/
  iphone_mono/             # RGB frames (0000001.png ...)
  iphone_mono_depth/       # Depth frames aligned to RGB (uint16, mm)
  iphone_calibration.yaml  # Pinhole K
  timestamps.txt           # frame,timestamp_ns
```

Outputs land alongside the sample (or next to your provided RGB folder):
- `cuvslam_poses.tum` and `cuvslam_poses_slam.tum` (SLAM enabled by default)
- `nvblox_out/mesh.ply` (+ voxel exports)
- depth folder is regenerated if missing.

## One-shot pipeline
```bash
python3 run_full_pipeline.py \
  --rgb-dir data/sample_20260119_i4/iphone_mono \
  --calibration data/sample_20260119_i4/iphone_calibration.yaml \
  --timestamps data/sample_20260119_i4/timestamps.txt \
  --nvblox-mode colormesh \
  --nvblox-ui   # drop if headless
```
This runs depth generation, PyCuVSLAM, builds nvblox artifacts, and runs nvblox with Rerun UI when `--nvblox-ui` is set. You can still pass `--dataset <folder>` to use the default filenames (`iphone_mono`, `iphone_calibration.yaml`, `timestamps.txt`), but `--dataset` is optional—any folder names work when you provide the three paths.
SLAM runs by default: both odometry (`pycuvslam_poses.tum`) and SLAM (`pycuvslam_poses_slam.tum`, `CameraTrajectory_slam.csv`) are saved, and nvblox consumes the SLAM trajectory unless you pass `--use-odom-poses` (or disable SLAM with `--disable-slam`).

## Individual runners
See `pipelines/README.md` for per-step commands (`run_pycuvslam_rgbd.py`, `run_nvblox.py`, stereo variant, etc.).
