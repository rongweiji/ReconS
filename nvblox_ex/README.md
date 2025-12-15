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
