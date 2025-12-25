## Phone sample converter

This folder contains a small CLI to:

1. Extract a phone video (e.g. `0001.mov`) into an image-frame folder.
2. Read the corresponding `*_ar.csv` (100 Hz) and visualize time alignment vs the video (≈30 fps).

### Usage

Create a venv (recommended) and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r ../requirements.txt
```

Run the converter (input is the folder containing the video + csv files):

```bash
. .venv/bin/activate
python data_prepare.py data/phone_sample1
```

### Outputs (written inside the input folder)

- `frames_<video_stem>/frame_000000.png` ... extracted frames
- `frames_<video_stem>/video_frames.csv` ... per-frame timestamps (seconds since start)
- `frames_<video_stem>/video_info.json` ... video timing summary
- `alignment_<video_stem>.png` ... alignment visualization
- `alignment_<video_stem>.json` ... alignment summary metrics

## nvblox_torch Runner (phone_sample*)

`run_nvblox_phone_sample3.py` replays a prepared `phone_sample*` dataset into `nvblox_torch` and exports a reconstructed mesh.

### Requirements

- NVIDIA GPU + CUDA working inside WSL/Linux (the installed `nvblox_torch` wheel is CUDA-only).
- Python deps in your venv:
  - `nvblox_torch` (your CUDA/Ubuntu-matched wheel)
  - `opencv-python`, `numpy`, `torch`
  - Optional UI: `PySide6`, `pyqtgraph`, `PyOpenGL`

### Basic usage (headless)

```bash
python3 nvblox_ex/run_nvblox_phone_sample3.py \
  --dataset nvblox_ex/data/phone_sample3 \
  --out_dir nvblox_ex/data/phone_sample3/nvblox_out
```

This writes `mesh.ply` into `--out_dir` (the script falls back to `mapper.get_color_mesh().save(...)` if the mapper does not expose a direct export method).

### Live Qt UI (mesh + pose + path)

```bash
python3 nvblox_ex/run_nvblox_phone_sample3.py \
  --dataset nvblox_ex/data/phone_sample3 \
  --out_dir nvblox_ex/data/phone_sample3/nvblox_out \
  --ui
```

### Key parameters

- `--dataset`: Path to a `phone_sample*` folder containing `associations.txt`, `CameraTrajectory.csv`, and frame subfolders.
- `--out_dir`: Output folder for reconstructed artifacts (e.g., `mesh.ply`).
- `--intrinsics_json`: Intrinsics file (defaults to `nvblox_ex/iphone_intrinsics.json`).
- `--voxel_size_m`: Voxel resolution (smaller = more detail, slower/more memory).
- `--max_integration_distance_m`: Depth truncation / max integration distance.
- `--depth_scale`: Meters per depth unit (phone samples use uint16 millimeters → `0.001`).
- `--mesh_every`: Update mesh every N frames (affects UI refresh cadence).
- `--invert_pose`: Invert each pose before integration (use if your trajectory is `T_C_W` instead of `T_W_C`).
- `--ui`: Show live Qt mesh viewer while integrating.
- `--mode`: UI visualization mode: `mesh`, `esdf`, or `voxel` (ESDF/voxel require nvblox_torch layer/query APIs).
- `--color_mode`: UI mesh coloring, `mesh` (fused vertex colors) or `solid` (fixed shaded color).
- `--field_step_m`: ESDF slice sampling step (smaller = higher resolution, slower).
- `--voxel_band_m`: Voxel mode TSDF band (|tsdf| < band).
- `--voxel_radius_m`: Voxel mode local radius around current pose.
- `--voxel_max_points`: Voxel mode downsample cap for UI speed.
This script always exports `tsdf_voxel_grid.ply` and `occupancy_voxel_grid.ply` into `--out_dir`.
