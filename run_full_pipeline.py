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
  - sharp_pre_cusfm/              [Optional] filtered subset + cuSFM/nvblox-sfm branch
  - 3dgrut_branches/              Branch manifests for comparing 3dgrut inputs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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


def _write_timestamps(path: Path, timestamps: Sequence[Tuple[str, int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_ns"])
        for frame_id, ts_ns in timestamps:
            writer.writerow([frame_id, str(ts_ns)])


def _find_frame(path_base: Path, frame_id: str, exts: Iterable[str]) -> Path:
    for ext in exts:
        candidate = path_base / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frame {frame_id} not found in {path_base} with extensions {list(exts)}")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


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


def _write_filtered_tum(src_path: Path, out_path: Path, selected_ts_ns: set[int]) -> int:
    if not src_path.exists():
        return 0
    kept_lines: list[str] = []
    for raw in src_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        ts_ns = int(round(float(parts[0]) * 1e9))
        if ts_ns in selected_ts_ns:
            kept_lines.append(line)
    out_path.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "")
    return len(kept_lines)


def _env_with_pythonpath(env: Mapping[str, str], *extra_paths: Path) -> dict[str, str]:
    merged = dict(env)
    py_parts = [str(path) for path in extra_paths if path]
    current = merged.get("PYTHONPATH")
    if current:
        py_parts.append(current)
    if py_parts:
        merged["PYTHONPATH"] = ":".join(py_parts)
    return merged


def _load_selected_frame_ids(selected_metadata_path: Path) -> set[str]:
    data = json.loads(selected_metadata_path.read_text())
    selected_frames = data.get("selected_frames")
    if not isinstance(selected_frames, list) or not selected_frames:
        selected_frames = data.get("selected_items")
    if not isinstance(selected_frames, list) or not selected_frames:
        raise ValueError(f"No selected_frames/selected_items entries in {selected_metadata_path}")
    frame_ids: set[str] = set()
    for entry in selected_frames:
        if not isinstance(entry, dict):
            continue
        original_path = entry.get("original_path")
        if original_path:
            frame_ids.add(Path(str(original_path)).stem)
            continue
        original_id = entry.get("original_id")
        if original_id:
            frame_ids.add(Path(str(original_id)).stem)
            continue
        output_filename = entry.get("output_filename")
        if output_filename:
            frame_ids.add(Path(str(output_filename)).stem)
    if not frame_ids:
        raise ValueError(f"No original_path/original_id/output_filename entries found in {selected_metadata_path}")
    return frame_ids


def _materialize_filtered_subset(
    *,
    branch_root: Path,
    selected_metadata_path: Path,
    rgb_dir: Path,
    depth_dir: Path,
    calib_path: Path,
    timestamps_path: Path,
    timestamps: Sequence[Tuple[str, int]],
    tum_out: Path,
    slam_tum_out: Path,
    enable_slam: bool,
) -> dict[str, object]:
    selected_ids = _load_selected_frame_ids(selected_metadata_path)
    filtered_timestamps = [(frame_id, ts_ns) for frame_id, ts_ns in timestamps if frame_id in selected_ids]
    if not filtered_timestamps:
        raise ValueError("sharp-frames selected no frames that match timestamps.txt")

    rgb_subset_dir = branch_root / rgb_dir.name
    depth_subset_dir = branch_root / depth_dir.name
    calib_subset_path = branch_root / calib_path.name
    timestamps_subset_path = branch_root / timestamps_path.name
    tum_subset_path = branch_root / tum_out.name
    slam_subset_path = branch_root / slam_tum_out.name
    associations_subset_path = branch_root / "associations.txt"
    cam_traj_subset_path = branch_root / "CameraTrajectory.csv"
    cam_traj_slam_subset_path = branch_root / "CameraTrajectory_slam.csv"
    intrinsics_subset_path = branch_root / "intrinsics_auto.json"

    rgb_subset_dir.mkdir(parents=True, exist_ok=True)
    depth_subset_dir.mkdir(parents=True, exist_ok=True)

    mount_modes: set[str] = set()
    for frame_id, _ in filtered_timestamps:
        rgb_src = _find_frame(rgb_dir, frame_id, [".png", ".jpg", ".jpeg"])
        depth_src = _find_frame(depth_dir, frame_id, [".png", ".exr", ".tiff", ".tif"])
        mount_modes.add(_link_or_copy_file(rgb_src, rgb_subset_dir / rgb_src.name))
        mount_modes.add(_link_or_copy_file(depth_src, depth_subset_dir / depth_src.name))

    mount_modes.add(_link_or_copy_file(calib_path, calib_subset_path))
    _write_timestamps(timestamps_subset_path, filtered_timestamps)

    selected_ts_ns = {ts_ns for _, ts_ns in filtered_timestamps}
    tum_rows = _write_filtered_tum(tum_out, tum_subset_path, selected_ts_ns)
    if tum_rows:
        _tum_to_csv(tum_subset_path, cam_traj_subset_path)
    slam_rows = 0
    if enable_slam and slam_tum_out.exists():
        slam_rows = _write_filtered_tum(slam_tum_out, slam_subset_path, selected_ts_ns)
        if slam_rows:
            _tum_to_csv(slam_subset_path, cam_traj_slam_subset_path)

    first_frame = filtered_timestamps[0][0]
    rgb_ext, depth_ext = _detect_exts(rgb_subset_dir, depth_subset_dir, first_frame)
    _write_associations(
        filtered_timestamps,
        rgb_subset_dir,
        depth_subset_dir,
        branch_root,
        rgb_ext,
        depth_ext,
        associations_subset_path,
    )
    sample_rgb = _find_frame(rgb_subset_dir, first_frame, [rgb_ext])
    _write_intrinsics_json(intrinsics_subset_path, calib_subset_path, sample_rgb)

    subset_manifest = {
        "selected_metadata_path": str(selected_metadata_path),
        "selected_input_frames": len(selected_ids),
        "subset_frames_with_timestamps": len(filtered_timestamps),
        "subset_pose_rows": tum_rows,
        "subset_slam_pose_rows": slam_rows,
        "mount_modes": sorted(mount_modes),
        "rgb_dir": str(rgb_subset_dir),
        "depth_dir": str(depth_subset_dir),
        "calibration": str(calib_subset_path),
        "timestamps": str(timestamps_subset_path),
        "poses": str(tum_subset_path),
        "poses_slam": str(slam_subset_path) if enable_slam and slam_rows else None,
    }
    (branch_root / "subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2) + "\n")
    return subset_manifest


def _detect_fused_point_cloud(base_dir: Path) -> Path | None:
    candidates = [
        base_dir / "nvblox_out" / "mesh.ply",
        base_dir / "nvblox_out" / "tsdf_voxel_grid.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_3dgrut_branch_manifests(
    *,
    base_dir: Path,
    rgb_dir: Path,
    branch_manifest_dir: Path,
    branch_root_sharp: Path | None,
) -> dict[str, Path]:
    branch_manifest_dir.mkdir(parents=True, exist_ok=True)

    branches = [
        {
            "branch_id": "branch_a_allframes_init",
            "description": "Original RGB frames + full-frame cuSFM sparse + full-frame nvblox mesh init.",
            "images_dir": rgb_dir,
            "sparse_dir": base_dir / "cusfm_output" / "sparse",
            "fused_point_cloud": None,
        },
        {
            "branch_id": "branch_b_cusfm_init",
            "description": "Original RGB frames + full-frame cuSFM sparse + nvblox_sfm refined mesh init.",
            "images_dir": rgb_dir,
            "sparse_dir": base_dir / "cusfm_output" / "sparse",
            "fused_point_cloud": base_dir / "nvblox_sfm_out" / "nvblox_mesh.ply",
        },
        {
            "branch_id": "branch_c_sharp_precusfm",
            "description": "Sharp-frame-filtered RGB subset + filtered cuSFM sparse + filtered nvblox_sfm refined mesh init.",
            "images_dir": (branch_root_sharp / rgb_dir.name) if branch_root_sharp else None,
            "sparse_dir": (branch_root_sharp / "cusfm_output" / "sparse") if branch_root_sharp else None,
            "fused_point_cloud": (branch_root_sharp / "nvblox_sfm_out" / "nvblox_mesh.ply") if branch_root_sharp else None,
        },
    ]

    branches[0]["fused_point_cloud"] = _detect_fused_point_cloud(base_dir)

    written: dict[str, Path] = {}
    index_payload = {"base_dir": str(base_dir), "branches": []}

    for branch in branches:
        branch_id = str(branch["branch_id"])
        images_dir = branch["images_dir"]
        sparse_dir = branch["sparse_dir"]
        fused_point_cloud = branch["fused_point_cloud"]
        missing_inputs: list[str] = []
        if not (isinstance(images_dir, Path) and images_dir.is_dir()):
            missing_inputs.append("images_dir")
        if not (isinstance(sparse_dir, Path) and sparse_dir.is_dir()):
            missing_inputs.append("sparse_dir")
        if not (isinstance(fused_point_cloud, Path) and fused_point_cloud.exists()):
            missing_inputs.append("fused_point_cloud")
        ready = (
            isinstance(images_dir, Path)
            and images_dir.is_dir()
            and isinstance(sparse_dir, Path)
            and sparse_dir.is_dir()
            and isinstance(fused_point_cloud, Path)
            and fused_point_cloud.exists()
        )
        compare_root = base_dir / "3dgrut_compare" / branch_id
        prepare_command = None
        train_command = None
        if ready:
            prepare_command = [
                "python3",
                "pipelines/run_3dgrut.py",
                "--images-dir",
                str(images_dir),
                "--sparse-dir",
                str(sparse_dir),
                "--fused-point-cloud",
                str(fused_point_cloud),
                "--work-dir",
                str(compare_root / "3dgrut_data"),
                "--out-dir",
                str(compare_root / "3dgrut_out"),
                "--experiment-name",
                f"{base_dir.name}_{branch_id}",
                "--prepare-only",
                "--overwrite",
            ]
            train_command = [
                "python3",
                "pipelines/run_3dgrut.py",
                "--images-dir",
                str(images_dir),
                "--sparse-dir",
                str(sparse_dir),
                "--fused-point-cloud",
                str(fused_point_cloud),
                "--work-dir",
                str(compare_root / "3dgrut_data"),
                "--out-dir",
                str(compare_root / "3dgrut_out"),
                "--experiment-name",
                f"{base_dir.name}_{branch_id}",
                "--overwrite",
            ]
        payload = {
            "branch_id": branch_id,
            "description": branch["description"],
            "ready_for_3dgrut": ready,
            "missing_inputs": missing_inputs,
            "images_dir": str(images_dir) if isinstance(images_dir, Path) else None,
            "sparse_dir": str(sparse_dir) if isinstance(sparse_dir, Path) else None,
            "fused_point_cloud": str(fused_point_cloud) if isinstance(fused_point_cloud, Path) else None,
            "suggested_work_dir": str(compare_root / "3dgrut_data"),
            "suggested_out_dir": str(compare_root / "3dgrut_out"),
            "suggested_experiment_name": f"{base_dir.name}_{branch_id}",
            "prepare_command": prepare_command,
            "train_command": train_command,
        }
        manifest_path = branch_manifest_dir / f"{branch_id}.json"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        written[branch_id] = manifest_path
        index_payload["branches"].append(
            {
                "branch_id": branch_id,
                "ready_for_3dgrut": ready,
                "missing_inputs": missing_inputs,
                "manifest_path": str(manifest_path),
            }
        )

    (branch_manifest_dir / "index.json").write_text(json.dumps(index_payload, indent=2) + "\n")
    return written


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
    parser.add_argument(
        "--cusfm-min-inter-frame-distance",
        type=float,
        default=0.06,
        help="cuSFM keyframe translation threshold in meters (denser than cuSFM default 0.5).",
    )
    parser.add_argument(
        "--cusfm-min-inter-frame-rotation-degrees",
        type=float,
        default=1.5,
        help="cuSFM keyframe rotation threshold in degrees (denser than cuSFM default 5.0).",
    )
    # nvblox-sfm options (enabled by default)
    parser.add_argument("--skip-nvblox-sfm", action="store_true", help="Skip nvblox with SFM keyframes for refined mesh.")
    parser.add_argument("--nvblox-sfm-out", type=Path, help="Output folder for nvblox-sfm (default: <base>/nvblox_sfm_out)")
    parser.add_argument("--nvblox-sfm-ui", action="store_true", help="Show rerun UI for nvblox-sfm.")
    # Default branch: sharp-frames filtering before cuSFM.
    parser.add_argument(
        "--sharp-pre-cusfm-root",
        type=Path,
        help="Output root for the filtered branch (default: <base>/sharp_pre_cusfm). Rebuilt on each run.",
    )
    parser.add_argument(
        "--sharp-selection-method",
        choices=["best-n", "batched", "outlier-removal"],
        default="batched",
        help="sharp-frames selection method for the pre-cuSFM branch.",
    )
    parser.add_argument("--sharp-num-frames", type=int, default=300, help="Target frame count for sharp best-n selection.")
    parser.add_argument("--sharp-min-buffer", type=int, default=3, help="Minimum gap for sharp best-n selection.")
    parser.add_argument("--sharp-batch-size", type=int, default=3, help="Batch size for sharp batched selection.")
    parser.add_argument("--sharp-batch-buffer", type=int, default=0, help="Batch gap for sharp batched selection.")
    parser.add_argument(
        "--sharp-outlier-window-size",
        type=int,
        default=15,
        help="Neighbor window for sharp outlier-removal selection.",
    )
    parser.add_argument(
        "--sharp-outlier-sensitivity",
        type=int,
        default=50,
        help="Sensitivity (0-100) for sharp outlier-removal selection.",
    )
    parser.add_argument(
        "--branch-manifest-dir",
        type=Path,
        help="Output folder for 3dgrut comparison manifests (default: <base>/3dgrut_branches).",
    )
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
    sharp_branch_root = (args.sharp_pre_cusfm_root or base_dir / "sharp_pre_cusfm").expanduser().resolve()
    branch_manifest_dir = (args.branch_manifest_dir or base_dir / "3dgrut_branches").expanduser().resolve()

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
            "--min-inter-frame-distance",
            str(args.cusfm_min_inter_frame_distance),
            "--min-inter-frame-rotation-degrees",
            str(args.cusfm_min_inter_frame_rotation_degrees),
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

    sharp_subset_manifest: dict[str, object] | None = None
    sharp_selected_metadata: Path | None = None
    sharp_cusfm_out = sharp_branch_root / "cusfm_output"
    sharp_frames_meta_path = sharp_branch_root / "frames_meta_cusfm.json"
    sharp_nvblox_sfm_out = sharp_branch_root / "nvblox_sfm_out"
    sharp_rgb_dir = sharp_branch_root / rgb_dir.name
    sharp_depth_dir = sharp_branch_root / depth_dir.name
    sharp_calib_path = sharp_branch_root / calib_path.name
    sharp_ts_path = sharp_branch_root / ts_path.name
    sharp_tum_out = sharp_branch_root / tum_out.name
    sharp_slam_tum_out = sharp_branch_root / slam_tum_out.name
    run_sharp_cusfm = False
    run_sharp_nvblox_sfm = False

    print("\n" + "=" * 60)
    print("Branch C: sharp-frames pre-cuSFM subset")
    print("=" * 60)

    if sharp_branch_root.exists():
        _remove_path(sharp_branch_root)
    sharp_branch_root.mkdir(parents=True, exist_ok=True)

    sharp_frames_out = sharp_branch_root / "sharp_frames"
    sharp_env = _env_with_pythonpath(base_env, repo_root / "third_party" / "sharp-frames-python")
    sharp_cmd = [
        sys.executable,
        "-m",
        "sharp_frames.sharp_frames",
        str(rgb_dir),
        str(sharp_frames_out),
        "--selection-method",
        args.sharp_selection_method,
        "--force-overwrite",
    ]
    if args.sharp_selection_method == "best-n":
        sharp_cmd.extend(["--num-frames", str(args.sharp_num_frames), "--min-buffer", str(args.sharp_min_buffer)])
    elif args.sharp_selection_method == "batched":
        sharp_cmd.extend(["--batch-size", str(args.sharp_batch_size), "--batch-buffer", str(args.sharp_batch_buffer)])
    else:
        sharp_cmd.extend(
            [
                "--outlier-window-size",
                str(args.sharp_outlier_window_size),
                "--outlier-sensitivity",
                str(args.sharp_outlier_sensitivity),
            ]
        )
    _run(sharp_cmd, env=sharp_env)

    sharp_selected_metadata = sharp_frames_out / "selected_metadata.json"
    if not sharp_selected_metadata.exists():
        raise SystemExit(f"sharp-frames metadata not found: {sharp_selected_metadata}")

    sharp_subset_manifest = _materialize_filtered_subset(
        branch_root=sharp_branch_root,
        selected_metadata_path=sharp_selected_metadata,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        calib_path=calib_path,
        timestamps_path=ts_path,
        timestamps=timestamps,
        tum_out=tum_out,
        slam_tum_out=slam_tum_out,
        enable_slam=args.enable_slam,
    )
    print(f"[info] sharp_pre_cusfm subset prepared at {sharp_branch_root}")

    sharp_poses_for_nvblox = sharp_slam_tum_out if args.use_slam_poses else sharp_tum_out
    sharp_pose_rows = int(sharp_subset_manifest.get("subset_slam_pose_rows", 0)) if args.use_slam_poses else int(sharp_subset_manifest.get("subset_pose_rows", 0))
    if sharp_pose_rows <= 0 or not sharp_poses_for_nvblox.exists():
        raise SystemExit(
            "sharp_pre_cusfm branch has no filtered poses for the selected frame subset. "
            "Adjust the sharp-frames settings or switch pose source."
        )

    run_sharp_cusfm = not args.skip_cusfm
    if run_sharp_cusfm:
        print("\n" + "=" * 60)
        print("Branch C: filtered cuSFM sparse reconstruction")
        print("=" * 60)

        sharp_cusfm_cmd = [
            sys.executable,
            str(repo_root / "pipelines" / "run_pycusfm.py"),
            "--rgb-dir",
            str(sharp_rgb_dir),
            "--calibration",
            str(sharp_calib_path),
            "--timestamps",
            str(sharp_ts_path),
            "--poses",
            str(sharp_poses_for_nvblox),
            "--out-dir",
            str(sharp_cusfm_out),
            "--frames-meta-out",
            str(sharp_frames_meta_path),
            "--min-inter-frame-distance",
            str(args.cusfm_min_inter_frame_distance),
            "--min-inter-frame-rotation-degrees",
            str(args.cusfm_min_inter_frame_rotation_degrees),
        ]
        cusfm_env = _env_with_cusfm_libs(os.environ, repo_root)
        _run(sharp_cusfm_cmd, env=cusfm_env)
    else:
        print("[warning] --skip-cusfm requested; sharp_pre_cusfm branch stops after subset generation.")

    run_sharp_nvblox_sfm = not args.skip_nvblox_sfm
    if run_sharp_nvblox_sfm:
        if not run_sharp_cusfm:
            print("[warning] sharp_pre_cusfm nvblox-sfm requires cuSFM; skipping filtered nvblox-sfm.")
        else:
            print("\n" + "=" * 60)
            print("Branch C: filtered nvblox with SFM keyframes")
            print("=" * 60)

            sharp_cusfm_keyframes_meta = sharp_cusfm_out / "keyframes" / "frames_meta.json"
            if not sharp_cusfm_keyframes_meta.exists():
                print(f"[warning] filtered cuSFM keyframes not found at {sharp_cusfm_keyframes_meta}; skipping filtered nvblox-sfm.")
            else:
                sharp_nvblox_sfm_cmd = [
                    sys.executable,
                    str(repo_root / "pipelines" / "run_nvblox_sfm.py"),
                    "--frames-meta",
                    str(sharp_cusfm_keyframes_meta),
                    "--rgb-dir",
                    str(sharp_rgb_dir),
                    "--depth-dir",
                    str(sharp_depth_dir),
                    "--calibration",
                    str(sharp_calib_path),
                    "--depth-scale",
                    str(1.0 / args.depth_scale),
                    "--mesh-path",
                    str(sharp_nvblox_sfm_out / "nvblox_mesh.ply"),
                ]
                if args.nvblox_sfm_ui:
                    if sharp_poses_for_nvblox.exists():
                        sharp_nvblox_sfm_cmd.extend(["--poses-compare", str(sharp_poses_for_nvblox)])
                else:
                    sharp_nvblox_sfm_cmd.append("--no-ui")
                _run(sharp_nvblox_sfm_cmd, env=base_env)
    elif run_sharp_cusfm:
        print("[warning] --skip-nvblox-sfm requested; sharp_pre_cusfm branch will not produce a fused mesh for 3dgrut.")

    branch_manifests = _write_3dgrut_branch_manifests(
        base_dir=base_dir,
        rgb_dir=rgb_dir,
        branch_manifest_dir=branch_manifest_dir,
        branch_root_sharp=sharp_branch_root,
    )

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
    print(f"sharp_pre_cusfm subset: {sharp_branch_root}")
    if sharp_selected_metadata:
        print(f"sharp-frames metadata: {sharp_selected_metadata}")
    if run_sharp_cusfm:
        print(f"filtered cuSFM outputs: {sharp_cusfm_out}")
        print(f"filtered cuSFM frames_meta: {sharp_frames_meta_path}")
    if run_sharp_nvblox_sfm and run_sharp_cusfm:
        print(f"filtered nvblox-sfm outputs: {sharp_nvblox_sfm_out}")
    print(f"3dgrut branch manifests: {branch_manifest_dir}")
    for branch_id, manifest_path in sorted(branch_manifests.items()):
        print(f"  {branch_id}: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
