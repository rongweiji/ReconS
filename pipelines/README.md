## Pipelines

Entry-point scripts that stitch together depth → PyCuVSLAM → nvblox for our RGBD datasets.

Environments:
- PyCuVSLAM: use the `pycuvslam` conda env (activates cuvslam).
- nvblox: use the `.venv` with `nvblox_torch`, `torch`, `opencv-python`, `numpy` (UI uses `rerun-sdk`).

Env setup (example):
- PyCuVSLAM (conda): `conda env create -f environment.yml` (or activate your existing `pycuvslam`), then `conda activate pycuvslam`.
- nvblox (.venv): `python3 -m venv .venv && . .venv/bin/activate && pip install -r ../requirements.txt` and install a CUDA-matched `torch` plus your `nvblox_torch` wheel. For UI: `pip install rerun-sdk`.

- `run_pycuvslam_rgbd.py`: run PyCuVSLAM on RGB + depth + calibration + timestamps. Outputs TUM poses to `<rgb-dir>/cuvslam_poses.tum` by default (SLAM poses to `<rgb-dir>/cuvslam_poses_slam.tum` when `--enable-slam`).
- `run_nvblox.py`: feed RGB + depth + calibration + TUM poses + timestamps into `nvblox_torch` to produce a mesh and optional UI visualization.
- `run_pycuvslam_stereo.py`: stereo variant for PyCuVSLAM (left/right + depths).

### PyCuVSLAM RGBD 

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
python3 pipelines/run_pycuvslam_rgbd.py \
  --rgb-dir data/sample_20260119_i4/iphone_mono \
  --depth-dir data/sample_20260119_i4/iphone_mono_depth \
  --calibration data/sample_20260119_i4/iphone_calibration.yaml \
  --timestamps data/sample_20260119_i4/timestamps.txt \
  --enable-slam \
  --preview
```

Defaults write poses to `data/sample_20260119_i4/cuvslam_poses.tum`. Also write `cuvslam_poses_slam.tum`.

### nvblox 

```bash
python3 pipelines/run_nvblox.py \
  --rgb-dir data/sample_20260119_i4/iphone_mono \
  --depth-dir data/sample_20260119_i4/iphone_mono_depth \
  --calibration data/sample_20260119_i4/iphone_calibration.yaml \
  --poses data/sample_20260119_i4/cuvslam_poses.tum \
  --timestamps data/sample_20260119_i4/timestamps.txt \
  --out_dir data/sample_20260119_i4/nvblox_out \
  --ui
```

Depth is assumed uint16 millimeters (`--depth_scale 0.001` by default). 
### Full chain (depth → poses → nvblox)

Use the repo root `run_full_pipeline.py` to automate all steps; see its help for flags.
