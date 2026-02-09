#!/usr/bin/env python3
"""
End-to-end pipeline:
1) Generate depth from mono frames (Depth Anything TensorRT wrapper).
2) Run PyCuVSLAM RGBD to produce poses.tum.
3) Build nvblox dataset artifacts (associations.txt, CameraTrajectory.csv, intrinsics JSON).
4) Run nvblox mapper to export a mesh.
5) Run cuSFM sparse reconstruction (skip with --skip-cusfm).
6) Run nvblox with SFM keyframes for refined mesh (skip with --skip-nvblox-sfm).

Inputs: point to RGB frames, calibration YAML, and timestamps.txt. You can still
pass --dataset to use its default layout, but --dataset is no longer required.
When --dataset is omitted, defaults live next to --rgb-dir (e.g.,
<rgb-dir>/../iphone_calibration.yaml).

Outputs (default locations inside --dataset or parent of --rgb-dir):
  - iphone_mono_depth/            Generated depth PNGs (mm, uint16)
  - pycuvslam_poses.tum           TUM trajectory from PyCuVSLAM
  - CameraTrajectory.csv          Trajectory for nvblox
  - associations.txt              RGB/depth pairing for nvblox
  - intrinsics_auto.json          Intrinsics matching the RGB resolution
  - nvblox_out/                   Mesh/voxel exports from nvblox
  - cusfm_output/                 [Optional] cuSFM sparse reconstruction
  - frames_meta_cusfm.json        [Optional] cuSFM frames metadata
  - nvblox_sfm_out/               [Optional] nvblox mesh from SFM keyframes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import yaml
from PIL import Image


def _run(cmd: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=dict(env) if env else None, check=True)


def _parse_timestamps(path: Path) -> List[Tuple[str, int]]:
    rows = list(csv.reader(path.read_text().splitlines()))
    if not rows or len(rows) < 2:
        raise ValueError(f"No timestamp rows found in {path}")
    header = [h.strip().lower() for h in rows[0]]
    frame_idx = header.index("frame")
    ts_idx = header.index("timestamp_ns")
    out: list[tuple[str, int]] = []
    for row in rows[1:]:
        if len(row) <= ts_idx:
            continue
        frame_id = row[frame_idx].strip()
        ts_ns = int(float(row[ts_idx]))
        out.append((frame_id, ts_ns))
    if not out:
        raise ValueError(f"No valid timestamp entries in {path}")
    return out


def _find_frame(path_base: Path, frame_id: str, exts: Iterable[str]) -> Path:
    for ext in exts:
        candidate = path_base / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frame {frame_id} not found in {path_base} with extensions {list(exts)}")


def _detect_exts(rgb_dir: Path, depth_dir: Path, frame_id: str) -> Tuple[str, str]:
    rgb_ext = _find_frame(rgb_dir, frame_id, [".png", ".jpg", ".jpeg"]).suffix
    depth_ext = _find_frame(depth_dir, frame_id, [".png", ".exr", ".tiff", ".tif"]).suffix
    return rgb_ext, depth_ext


def _tum_to_csv(tum_path: Path, csv_path: Path) -> None:
    lines = tum_path.read_text().splitlines()
    rows: list[list[str]] = [["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        rows.append(parts)
    if len(rows) <= 1:
        raise ValueError(f"No pose rows found in {tum_path}")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _write_associations(
    timestamps: Sequence[Tuple[str, int]],
    rgb_dir: Path,
    depth_dir: Path,
    base_dir: Path,
    rgb_ext: str,
    depth_ext: str,
    out_path: Path,
) -> None:
    lines: list[str] = []
    for frame_id, ts_ns in timestamps:
        ts_sec = ts_ns * 1e-9
        rgb = rgb_dir / f"{frame_id}{rgb_ext}"
        depth = depth_dir / f"{frame_id}{depth_ext}"
        if not rgb.exists() or not depth.exists():
            continue
        rgb_rel = os.path.relpath(rgb, base_dir)
        depth_rel = os.path.relpath(depth, base_dir)
        lines.append(f"{ts_sec:.6f} {rgb_rel} {ts_sec:.6f} {depth_rel}")
    if not lines:
        raise ValueError("No associations written (missing frames?).")
    out_path.write_text("\n".join(lines) + "\n")


def _parse_k(calib_path: Path) -> Tuple[float, float, float, float]:
    data = yaml.safe_load(calib_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected calibration format in {calib_path}")

    def flatten(val) -> list[float]:
        if isinstance(val, dict):
            arr = val.get("data") or val.get("Data") or val.get("values") or val.get("K") or val.get("k")
            return arr if isinstance(arr, list) else []
        if isinstance(val, list):
            flat: list[float] = []
            for row in val:
                if isinstance(row, list):
                    flat.extend(row)
                else:
                    flat.append(row)
            return flat
        return []

    k = []
    if "K" in data:
        k = flatten(data["K"])
    if not k and "camera_matrix" in data:
        k = flatten(data["camera_matrix"])
    if len(k) < 6:
        raise ValueError(f"Calibration K is incomplete in {calib_path}")
    fx = float(k[0])
    fy = float(k[4]) if len(k) > 4 else float(k[1])
    cx = float(k[2])
    cy = float(k[5]) if len(k) > 5 else float(k[3])
    return fx, fy, cx, cy


def _write_intrinsics_json(in_path: Path, calib_path: Path, rgb_sample: Path) -> None:
    fx, fy, cx, cy = _parse_k(calib_path)
    w, h = Image.open(rgb_sample).size
    payload = {
        "intrinsics": [
            {
                "resolution": {"width": w, "height": h},
                "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "distortion_model": None,
                "D": None,
            }
        ]
    }
    in_path.write_text(json.dumps(payload, indent=2))


def _env_with_conda_lib(env: Mapping[str, str]) -> dict[str, str]:
    """Ensure native deps see conda lib/ first and WSL's libcuda shim if present."""
    merged = dict(env)
    ld_parts: list[str] = []

    conda_prefix = merged.get("CONDA_PREFIX")
    if conda_prefix:
        ld_parts.append(str(Path(conda_prefix) / "lib"))

    wsl_lib = Path("/usr/lib/wsl/lib")
    if wsl_lib.is_dir():
        ld_parts.append(str(wsl_lib))

    if merged.get("LD_LIBRARY_PATH"):
        ld_parts.append(merged["LD_LIBRARY_PATH"])

    if ld_parts:
        merged["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return merged


def _env_with_cusfm_libs(env: Mapping[str, str], repo_root: Path) -> dict[str, str]:
    """Ensure cuSFM native deps see correct CUDA/conda libs (prefer WSL shim + CUDA13)."""
    merged = dict(env)
    ld_parts: list[str] = []

    wsl_lib = Path("/usr/lib/wsl/lib")
    if wsl_lib.is_dir():
        ld_parts.append(str(wsl_lib))

    # Add pyCuSFM bundled libraries (libcvcuda, etc.)
    pycusfm_lib = repo_root / "third_party" / "pyCuSFM" / "pycusfm" / "lib"
    if pycusfm_lib.is_dir():
        ld_parts.append(str(pycusfm_lib))

    conda_prefix = merged.get("CONDA_PREFIX")
    if conda_prefix:
        ld_parts.append(str(Path(conda_prefix) / "lib"))

    def _add_cuda_paths(*bases: Path) -> None:
        for base in bases:
            for sub in ("targets/x86_64-linux/lib", "lib64", "lib"):
                candidate = base / sub
                if candidate.is_dir():
                    ld_parts.append(str(candidate))

    # Force CUDA 13.0 first (matches driver), then allow 13.1/generic if 13.0 missing.
    cuda_13 = Path("/usr/local/cuda-13.0")
    cuda_131 = Path("/usr/local/cuda-13.1")
    cuda_generic = Path("/usr/local/cuda")
    if cuda_13.exists():
        _add_cuda_paths(cuda_13)
    elif cuda_131.exists():
        _add_cuda_paths(cuda_131)
    elif cuda_generic.exists():
        _add_cuda_paths(cuda_generic)

    if merged.get("LD_LIBRARY_PATH"):
        current = []
        for p in merged["LD_LIBRARY_PATH"].split(":"):
            if not p:
                continue
            # Drop older /usr/local/cuda-* entries to avoid picking stale compat stubs.
            if p.startswith("/usr/local/cuda") and "cuda-13" not in p:
                continue
            current.append(p)
        ld_parts.extend(current)

    if ld_parts:
        merged["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return merged


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="End-to-end: depth -> PyCuVSLAM -> nvblox -> [cuSFM] -> [nvblox-sfm].")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional dataset folder. Defaults for rgb/calibration/timestamps/depth are derived from this if provided.",
    )
    parser.add_argument("--rgb-dir", type=Path, help="RGB folder (default: <dataset>/iphone_mono; required if --dataset is omitted)")
    parser.add_argument("--depth-dir", type=Path, help="Depth output folder (default: <base>/iphone_mono_depth; base is dataset or RGB parent)")
    parser.add_argument("--calibration", type=Path, help="Calibration YAML (default: <base>/iphone_calibration.yaml)")
    parser.add_argument("--timestamps", type=Path, help="timestamps.txt (default: <base>/timestamps.txt)")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale used for PNG export (value per meter).")
    parser.add_argument("--force-poses", action="store_true", help="Regenerate poses even if TUM file exists.")
    parser.add_argument("--depth-engine", type=Path, help="Optional override for Depth Anything TensorRT engine.")
    parser.add_argument("--disable-slam", action="store_true", help="Skip PyCuVSLAM SLAM backend (pose graph + loop closure).")
    parser.add_argument("--use-odom-poses", action="store_true", help="Use odometry poses for downstream artifacts/nvblox instead of SLAM.")
    parser.add_argument("--nvblox-mode", choices=["colormesh", "solidmesh", "esdf", "tsdf", "pointcloud"], default="colormesh")
    parser.add_argument("--nvblox-ui", action="store_true", help="Show nvblox Qt UI.")
    parser.add_argument("--nvblox-out", type=Path, help="Output folder for nvblox (default: <base>/nvblox_out)")
    parser.add_argument("--skip-nvblox", action="store_true", help="Run depth + poses + dataset prep, skip nvblox.")
    # cuSFM options (enabled by default)
    parser.add_argument("--skip-cusfm", action="store_true", help="Skip cuSFM sparse reconstruction.")
    parser.add_argument("--cusfm-out", type=Path, help="Output folder for cuSFM (default: <base>/cusfm_output)")
    parser.add_argument("--frames-meta-out", type=Path, help="frames_meta.json output for cuSFM (default: <base>/frames_meta_cusfm.json)")
    # nvblox-sfm options (enabled by default)
    parser.add_argument("--skip-nvblox-sfm", action="store_true", help="Skip nvblox with SFM keyframes for refined mesh.")
    parser.add_argument("--nvblox-sfm-out", type=Path, help="Output folder for nvblox-sfm (default: <base>/nvblox_sfm_out)")
    parser.add_argument("--nvblox-sfm-ui", action="store_true", help="Show rerun UI for nvblox-sfm.")
    args = parser.parse_args()

    # Defaults: run SLAM and feed SLAM poses to downstream unless explicitly disabled.
    args.enable_slam = not args.disable_slam
    args.use_slam_poses = not args.use_odom_poses and args.enable_slam

    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset and not dataset.is_dir():
        raise SystemExit(f"Dataset not found: {dataset}")

    default_rgb = dataset / "iphone_mono" if dataset else None
    rgb_dir_input = args.rgb_dir or default_rgb
    if not rgb_dir_input:
        raise SystemExit("Provide --rgb-dir when --dataset is not set.")
    rgb_dir = rgb_dir_input.expanduser().resolve()
    if not rgb_dir.is_dir():
        raise SystemExit(f"RGB folder not found: {rgb_dir}")

    base_dir = dataset if dataset else rgb_dir.parent

    depth_dir = (args.depth_dir or base_dir / "iphone_mono_depth").expanduser().resolve()
    calib_path = (args.calibration or base_dir / "iphone_calibration.yaml").expanduser().resolve()
    ts_path = (args.timestamps or base_dir / "timestamps.txt").expanduser().resolve()
    tum_out = base_dir / "pycuvslam_poses.tum"
    slam_tum_out = base_dir / "pycuvslam_poses_slam.tum"
    associations_path = base_dir / "associations.txt"
    cam_traj_path = base_dir / "CameraTrajectory.csv"
    cam_traj_slam_path = base_dir / "CameraTrajectory_slam.csv"
    intrinsics_json = base_dir / "intrinsics_auto.json"
    nvblox_out = (args.nvblox_out or base_dir / "nvblox_out").expanduser().resolve()

    # Step 1: Depth generation
    base_env = _env_with_conda_lib(os.environ)

    depth_cmd: list[str] = [str(repo_root / "depth_feature" / "run_depth_from_rgb.sh"), "--rgb-dir", str(rgb_dir), "--calibration", str(calib_path), "--out-dir", str(depth_dir), "--depth-scale", str(args.depth_scale)]
    if args.depth_engine:
        depth_cmd.extend(["--engine", str(args.depth_engine)])
    _run(depth_cmd, env=base_env)

    # Step 2: PyCuVSLAM RGBD
    need_poses = args.force_poses or not tum_out.exists() or (args.enable_slam and not slam_tum_out.exists())
    if need_poses:
        slam_cmd = [
            sys.executable,
            str(repo_root / "pipelines" / "run_pycuvslam_rgbd.py"),
            "--rgb-dir",
            str(rgb_dir),
            "--depth-dir",
            str(depth_dir),
            "--calibration",
            str(calib_path),
            "--timestamps",
            str(ts_path),
            "--depth-scale",
            str(args.depth_scale),
            "--out",
            str(tum_out),
        ]
        if args.enable_slam:
            slam_cmd.append("--enable-slam")
            slam_cmd.extend(["--slam-out", str(slam_tum_out)])
        _run(slam_cmd, env=base_env)
    else:
        print(f"[skip] Pose generation (reuse existing {tum_out})")

    timestamps = _parse_timestamps(ts_path)
    first_frame = timestamps[0][0]
    rgb_ext, depth_ext = _detect_exts(rgb_dir, depth_dir, first_frame)

    # Step 3: Dataset artifacts for nvblox
    _tum_to_csv(tum_out, cam_traj_path)
    if args.enable_slam and slam_tum_out.exists():
        _tum_to_csv(slam_tum_out, cam_traj_slam_path)
    _write_associations(timestamps, rgb_dir, depth_dir, base_dir, rgb_ext, depth_ext, associations_path)
    sample_rgb = _find_frame(rgb_dir, first_frame, [rgb_ext])
    _write_intrinsics_json(intrinsics_json, calib_path, sample_rgb)

    poses_for_nvblox = slam_tum_out if args.use_slam_poses else tum_out
    cam_csv_for_nvblox = cam_traj_slam_path if args.use_slam_poses else cam_traj_path
    if args.use_slam_poses and not slam_tum_out.exists():
        raise SystemExit(f"Requested SLAM poses for nvblox but missing {slam_tum_out}")

    if args.skip_nvblox:
        print("[skip] nvblox run (skip requested)")
        return 0

    m_per_unit = 1.0 / float(args.depth_scale) if float(args.depth_scale) != 0 else 0.001
    nvblox_cmd: list[str] = [
        sys.executable,
        str(repo_root / "pipelines" / "run_nvblox.py"),
        "--rgb-dir",
        str(rgb_dir),
        "--depth-dir",
        str(depth_dir),
        "--calibration",
        str(calib_path),
        "--poses",
        str(poses_for_nvblox),
        "--timestamps",
        str(ts_path),
        "--depth_scale",
        f"{m_per_unit}",
        "--out_dir",
        str(nvblox_out),
        "--mode",
        args.nvblox_mode,
    ]
    if args.nvblox_ui:
        nvblox_cmd.append("--ui")
        # Pass both trajectories for comparison visualization (SLAM vs odometry)
        if args.use_slam_poses and tum_out.exists():
            nvblox_cmd.extend(["--poses-compare", str(tum_out)])
        elif not args.use_slam_poses and slam_tum_out.exists():
            nvblox_cmd.extend(["--poses-compare", str(slam_tum_out)])
    _run(nvblox_cmd, env=base_env)

    # Step 5: cuSFM sparse reconstruction (enabled by default)
    cusfm_out = (args.cusfm_out or base_dir / "cusfm_output").expanduser().resolve()
    frames_meta_path = (args.frames_meta_out or base_dir / "frames_meta_cusfm.json").expanduser().resolve()
    run_cusfm = not args.skip_cusfm

    if run_cusfm:
        print("\n" + "=" * 60)
        print("Step 5: cuSFM sparse reconstruction")
        print("=" * 60)

        cusfm_cmd = [
            sys.executable,
            str(repo_root / "pipelines" / "run_pycusfm.py"),
            "--rgb-dir",
            str(rgb_dir),
            "--calibration",
            str(calib_path),
            "--timestamps",
            str(ts_path),
            "--poses",
            str(poses_for_nvblox),
            "--out-dir",
            str(cusfm_out),
            "--frames-meta-out",
            str(frames_meta_path),
        ]
        cusfm_env = _env_with_cusfm_libs(os.environ, repo_root)
        _run(cusfm_cmd, env=cusfm_env)

    # Step 6: nvblox with SFM keyframes (enabled by default)
    nvblox_sfm_out = (args.nvblox_sfm_out or base_dir / "nvblox_sfm_out").expanduser().resolve()
    run_nvblox_sfm = not args.skip_nvblox_sfm

    if run_nvblox_sfm:
        if not run_cusfm:
            print("[warning] nvblox-sfm requires cuSFM (don't use --skip-cusfm); skipping nvblox-sfm.")
        else:
            print("\n" + "=" * 60)
            print("Step 6: nvblox with SFM keyframes")
            print("=" * 60)

            # Find the cuSFM keyframes frames_meta.json
            cusfm_keyframes_meta = cusfm_out / "keyframes" / "frames_meta.json"
            if not cusfm_keyframes_meta.exists():
                print(f"[warning] cuSFM keyframes not found at {cusfm_keyframes_meta}; skipping nvblox-sfm.")
            else:
                nvblox_sfm_cmd = [
                    sys.executable,
                    str(repo_root / "pipelines" / "run_nvblox_sfm.py"),
                    "--frames-meta",
                    str(cusfm_keyframes_meta),
                    "--rgb-dir",
                    str(rgb_dir),
                    "--depth-dir",
                    str(depth_dir),
                    "--calibration",
                    str(calib_path),
                    "--depth-scale",
                    str(1.0 / args.depth_scale),
                    "--mesh-path",
                    str(nvblox_sfm_out / "nvblox_mesh.ply"),
                ]
                if args.nvblox_sfm_ui:
                    # UI is enabled by default in run_nvblox_sfm.py
                    # Pass SLAM/odometry trajectory for comparison with SFM poses
                    if poses_for_nvblox.exists():
                        nvblox_sfm_cmd.extend(["--poses-compare", str(poses_for_nvblox)])
                else:
                    nvblox_sfm_cmd.append("--no-ui")
                _run(nvblox_sfm_cmd, env=base_env)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Depth dir: {depth_dir}")
    print(f"PyCuVSLAM poses: {tum_out}")
    if args.enable_slam:
        print(f"PyCuVSLAM SLAM poses: {slam_tum_out}")
    print(f"Nvblox dataset: {associations_path}, {cam_traj_path}, {intrinsics_json}")
    if args.enable_slam:
        print(f"Nvblox dataset (SLAM CSV): {cam_traj_slam_path}")
    poses_label = "SLAM" if args.use_slam_poses else "odometry"
    print(f"Nvblox poses source ({poses_label}): {poses_for_nvblox}")
    print(f"Nvblox outputs: {nvblox_out}")
    if run_cusfm:
        print(f"cuSFM outputs: {cusfm_out}")
        print(f"cuSFM frames_meta: {frames_meta_path}")
    if run_nvblox_sfm and run_cusfm:
        print(f"nvblox-sfm outputs: {nvblox_sfm_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
