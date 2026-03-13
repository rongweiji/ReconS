# ReconS

End-to-end pipeline: RGB-> depth → PyCuVSLAM ->SFM → nvblox -> 3D synthesis and visualization

Demo : 



https://github.com/user-attachments/assets/c316bae7-069f-442d-bf2f-6c98325dd0ef





References:
- nvblox: https://nvidia-isaac.github.io/nvblox/index.html
- Neural reconstruction stereo (NuRec): https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html
- Depth generation (Depth Anything TRT): https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt

## Environment
Use two conda environments:
- `pycuvslam`: depth + PyCuVSLAM + cuSFM + nvblox pipeline (this repo's current one-shot flow).
- `3dgrut`: NURec Step 5 neural reconstruction and USD/USDZ export.

Why two envs:
- `cuvslam` is pinned to Python `3.10.*`.
- `3dgrut` uses a different Python/Torch/CUDA stack (its installer creates a dedicated env).
- Mixing both stacks in one env is likely to cause dependency conflicts.

### 1) Setup `pycuvslam` (ReconS pipeline)
```bash
conda create -n pycuvslam python=3.10 -y
conda activate pycuvslam
# Optional (system deps used by Open3D/nvblox visualization):
# sudo apt-get install -y python3-pip libglib2.0-0 libgl1
python -m pip install --upgrade pip
python -m pip install ./third_party/PyCuVSLAM/bin/x86_64
python -m pip install ./third_party/pyCuSFM
# Install a CUDA-matched PyTorch build (choose one that matches your CUDA stack):
# CUDA 12.x example
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.8 example
# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install ONE nvblox_torch wheel matching your Ubuntu + CUDA stack:
# Ubuntu 24.04 + CUDA 12.8
# python -m pip install "https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.9/nvblox_torch-0.0.9+cu12ubuntu24-py3-none-linux_x86_64.whl"
# Ubuntu 22.04 + CUDA 12.6
python -m pip install "https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.9/nvblox_torch-0.0.9+cu12ubuntu22-py3-none-linux_x86_64.whl"
# Ubuntu 22.04 + CUDA 11.8
# python -m pip install "https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.9/nvblox_torch-0.0.9+cu11ubuntu22-py3-none-linux_x86_64.whl"
# Ubuntu 24.04 + CUDA 13.0
# python -m pip install "https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.9/nvblox_torch-0.0.9+cu13ubuntu24-py3-none-linux_x86_64.whl"
python -m pip install -r requirements.txt rerun-sdk open3d
```
Install these once when creating the `pycuvslam` env (or any time you recreate it), before running `run_full_pipeline.py` or any `pipelines/run_nvblox*.py` script.
If your machine is `aarch64` (Jetson), do not use the x86_64 wheel above; build `nvblox_torch` from source.

### 2) Setup `3dgrut` (NURec Step 5)
```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git third_party/3dgrut
cd third_party/3dgrut
./install_env.sh 3dgrut
conda activate 3dgrut
```
`install_env.sh` supports CUDA `11.8.0` and `12.8.1` via `CUDA_VERSION`.

### 3) Which env to use when
- Run ReconS scripts (`run_full_pipeline.py`, `pipelines/run_pycuvslam_*.py`, `pipelines/run_nvblox*.py`) in `pycuvslam`.
- Run `3dgrut` training/export in `3dgrut`.

### 4) Runtime notes
- NVIDIA driver: PyCuVSLAM requires a driver exposing CUDA >= 12.6 (R560+; CUDA 13 drivers are OK).
- On WSL, keep:
  ```bash
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  ```
  so `cuvslam` resolves the correct CUDA/libpython libraries.
- If `conda` is missing after install, run `~/miniconda3/bin/conda init bash`, then `source ~/.bashrc` (or reopen a shell).

## Data layout (per sample)
```
data/sample_xxx/
  iphone_mono/             # RGB frames (0000001.png ...)
  iphone_mono_depth/       # Depth frames aligned to RGB (uint16, mm)
  iphone_calibration.yaml  # Pinhole K
  timestamps.txt           # frame,timestamp_ns
```

Outputs land alongside the sample:
- `iphone_mono_depth/` - generated depth maps
- `pycuvslam_poses.tum`, `pycuvslam_poses_slam.tum` - trajectories
- `nvblox_out/` - mesh from all frames
- `cusfm_output/` - sparse reconstruction + `keyframes/`
- `nvblox_sfm_out/` - refined mesh from SFM keyframes
- `sharp_pre_cusfm/` - optional filtered subset + filtered cuSFM/nvblox-sfm branch
- `3dgrut_branches/` - branch manifests for comparing 3dgrut inputs

## Full Pipeline

`run_full_pipeline.py` executes 6 base steps end-to-end:

1. **Depth** - Generate depth maps via Depth Anything TensorRT
2. **PyCuVSLAM** - Visual odometry + SLAM poses
3. **Dataset prep** - Build nvblox artifacts (associations, trajectory CSV, intrinsics)
4. **nvblox** - Dense mesh from all frames
5. **cuSFM** - Sparse reconstruction with keyframe selection
6. **nvblox-sfm** - Refined mesh from SFM keyframes (cleaner for 3dgrut)

7. **sharp-pre-cusfm** - Filter RGB frames with `sharp-frames-python`, rebuild a subset dataset, then run filtered cuSFM + filtered nvblox-sfm as a third 3dgrut branch by default

```bash
# Using dataset folder (recommended)
python3 run_full_pipeline.py --dataset data/sample_20260208_i1

# Or explicit paths
python3 run_full_pipeline.py \
  --rgb-dir data/sample_xxx/iphone_mono \
  --calibration data/sample_xxx/iphone_calibration.yaml \
  --timestamps data/sample_xxx/timestamps.txt

# The filtered pre-cuSFM comparison branch now runs by default
python3 run_full_pipeline.py --dataset data/sample_20260208_i1
```

**Outputs** (in dataset folder):
- `iphone_mono_depth/` - depth maps
- `pycuvslam_poses.tum`, `pycuvslam_poses_slam.tum` - trajectories
- `nvblox_out/` - mesh from all frames
- `cusfm_output/` - sparse reconstruction + keyframes
- `nvblox_sfm_out/nvblox_mesh.ply` - refined mesh for 3dgrut
- `sharp_pre_cusfm/iphone_mono` - filtered RGB subset selected by sharp-frames
- `sharp_pre_cusfm/cusfm_output/` - filtered cuSFM sparse reconstruction
- `sharp_pre_cusfm/nvblox_sfm_out/nvblox_mesh.ply` - filtered refined mesh for branch C
- `3dgrut_branches/branch_*.json` - manifests for branch A/B/C comparison

**Options:**
- `--skip-cusfm` / `--skip-nvblox-sfm` - skip SFM steps
- `--disable-slam` - use odometry only
- `--nvblox-ui` / `--nvblox-sfm-ui` - show visualization
- `--sharp-selection-method batched|outlier-removal|best-n` - choose frame filtering strategy for the third branch (default: batched with batch size 3 and buffer 0)

## Individual runners
See `pipelines/README.md` for per-step commands (`run_pycuvslam_rgbd.py`, `run_nvblox.py`, stereo variant, etc.).


## Common command: 

- Visualize nvblox meshes with the rerun viewer (after running `run_full_pipeline.py` or `run_nvblox.py`):
```bash
conda activate pycuvslam
python utilities/rerun_phone_sample_player.py data/sample_20260302_i10_1080
```

```bash
conda activate pycuvslam
python utilities/rerun_nvblox_viewer.py data/sample_20260302_i10_1080/nvblox_out/mesh.ply
```