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
- `run_nvblox_sfm.py`: fuse cuSFM keyframes (frames_meta.json) + RGB + depth directly into nvblox_torch using refined SfM poses, with rerun UI.

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

### nvblox (cuSFM keyframes only)

```
python3 pipelines/run_nvblox_sfm.py \
  --frames-meta data/sample_20260119_i4/cusfm_output/keyframes/frames_meta.json \
  --rgb-dir data/sample_20260119_i4/iphone_mono \
  --depth-dir data/sample_20260119_i4/iphone_mono_depth \
  --calibration data/sample_20260119_i4/iphone_calibration.yaml \
  --out-dir data/sample_20260119_i4/nvblox_sfm_out
```

Uses refined poses from `frames_meta.json` and logs RGB/depth/mesh to rerun by default. Depth is assumed uint16 millimeters unless `--depth-scale` is set otherwise (`--no-ui` disables rerun).
### Full chain (depth → poses → nvblox)

Use the repo root `run_full_pipeline.py` to automate all steps; see its help for flags.

### 3dgrut (cuSFM + nvblox init)

Use `run_3dgrut.py` to:
- prepare a COLMAP-style 3dgrut dataset layout from a ReconS sample
- run `third_party/3dgrut/train.py` with `apps/cusfm_3dgut.yaml`
- save outputs under the same sample folder

Preparation only (safe first check):

```bash
python3 pipelines/run_3dgrut.py \
  --sample-dir data/sample_20260119_i4 \
  --prepare-only \
  --overwrite
```

Run training in the dedicated `3dgrut` conda env:

```bash
python3 pipelines/run_3dgrut.py \
  --sample-dir data/sample_20260119_i4 \
  --conda-env 3dgrut \
  --max-steps 3000 \
  --overwrite
```

Run with separate folders (without `--sample-dir`):

```bash
python3 pipelines/run_3dgrut.py \
  --images-dir data/sample_20260119_i4/iphone_mono \
  --sparse-dir data/sample_20260119_i4/cusfm_output/sparse \
  --fused-point-cloud data/sample_20260119_i4/nvblox_sfm_out/nvblox_mesh.ply \
  --work-dir /tmp/recons_i4_3dgrut_data \
  --out-dir data/sample_20260119_i4/3dgrut_out \
  --conda-env 3dgrut \
  --max-steps 3000 \
  --overwrite
```

By default this script:
- uses `data/sample_20260119_i4/cusfm_output/sparse` as COLMAP sparse input
- uses `data/sample_20260119_i4/nvblox_sfm_out/nvblox_mesh.ply` as fused point cloud init
- writes prepared data to `<sample>/3dgrut_data`
- writes training outputs to `<sample>/3dgrut_out`
- writes run metadata to `<sample>/3dgrut_manifest.json` and `<sample>/3dgrut_result.json`
