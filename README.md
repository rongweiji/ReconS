# ReconS

[Nurec Reconstruction](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html#)

```wsl
source .venv/bin/activate
```
## Outcome 
1. opencv 4.12 version python based calibration for stereo fisheye cameras. 
 in folder cam-calib

2. voltage affect the camera recording stall 5v 2.5-3A better for 


# using the conda to run the pycuslam: 
export PATH=/home/wayne/miniconda3/bin:$PATH   # if conda not on PATH
  source /home/wayne/miniconda3/etc/profile.d/conda.sh
  conda activate pycuvslam   # replace with the env that has cuvslam
## pycuslam for rgbd
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
  python3 run_pycuvslam_rgbd.py \
    --rgb-dir data/sample_20260117_205753/left \
    --depth-dir data/sample_20260117_205753/left_depth \
    --calibration data/sample_20260117_205753/left_calibration.yaml \
    --timestamps data/sample_20260117_205753/timestamps.txt \
    --depth-scale 1000 \
    --enable-slam \
    --undistort \
    --distortion-model fisheye \
    --preview \
    --show-features


## orbslam for rgbd 
bash orbslam3_rgbd/run_orbslam3_rgbd.sh \
    --rgb-dir data/sample_20260117_205753/left \
    --depth-dir data/sample_20260117_205753/left_depth \
    --calibration data/sample_20260117_205753/left_calibration.yaml \
    --timestamps data/sample_20260117_205753/timestamps.txt \
    --depth-scale 1000 \
    --distortion-model fisheye \
    --viewer


## ORBSLAM will faild when fast rotation motion 