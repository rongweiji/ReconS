#!/usr/bin/env python3
"""
Fuse cuSFM-refined keyframes (frames_meta.json) with nvblox_torch and visualize in rerun.

Inputs:
  - --frames-meta: cuSFM frames_meta.json (e.g., data/.../cusfm_output/keyframes/frames_meta.json)
  - --rgb-dir: RGB frames folder (e.g., data/.../iphone_mono)
  - --depth-dir: depth frames aligned to RGB (same filename stems)
  - --calibration: optional YAML with pinhole K; falls back to frames_meta camera params

Outputs:
  - Mesh logged to rerun; optionally written to --mesh-path (default: <rgb-dir>/../nvblox_sfm_out/nvblox_mesh.ply)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple

import cv2
import numpy as np
import torch
import yaml


@dataclass
class Frame:
    ts_sec: float
    rgb_path: Path
    depth_path: Path
    pose_w_c: np.ndarray  # 4x4
    frame_id: str


def _axis_angle_deg_to_rot(axis_angle: Mapping[str, float]) -> np.ndarray:
    x = float(axis_angle.get("x", 0.0))
    y = float(axis_angle.get("y", 0.0))
    z = float(axis_angle.get("z", 0.0))
    ang_deg = float(axis_angle.get("angle_degrees", 0.0))
    axis = np.array([x, y, z], dtype=np.float64)
    theta = np.deg2rad(ang_deg)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or abs(theta) < 1e-9:
        return np.eye(3, dtype=np.float32)
    axis /= norm
    ux, uy, uz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    rot = np.array(
        [
            [c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
            [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
            [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C],
        ],
        dtype=np.float32,
    )
    return rot


def _pose_from_metadata(entry: Mapping[str, object]) -> np.ndarray:
    ctw = entry.get("camera_to_world") or entry.get("pose") or {}
    axis_angle = {}
    translation = {}
    if isinstance(ctw, Mapping):
        axis_angle = ctw.get("axis_angle", {}) if isinstance(ctw.get("axis_angle"), Mapping) else {}
        translation = ctw.get("translation", {}) if isinstance(ctw.get("translation"), Mapping) else {}
    rot = _axis_angle_deg_to_rot(axis_angle)
    tx = float(translation.get("x", 0.0)) if translation else 0.0
    ty = float(translation.get("y", 0.0)) if translation else 0.0
    tz = float(translation.get("z", 0.0)) if translation else 0.0
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def _parse_calibration_yaml(calib_path: Path) -> Tuple[float, float, float, float]:
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


def _intrinsics_from_frames_meta(meta: Mapping[str, object]) -> Tuple[float, float, float, float] | None:
    cam_map = meta.get("camera_params_id_to_camera_params")
    if not isinstance(cam_map, Mapping) or not cam_map:
        return None
    first_key = sorted(cam_map.keys())[0]
    cam_params = cam_map[first_key]
    if not isinstance(cam_params, Mapping):
        return None
    calib = cam_params.get("calibration_parameters", {})
    if not isinstance(calib, Mapping):
        return None
    cm = calib.get("camera_matrix", {})
    data = cm.get("data") if isinstance(cm, Mapping) else None
    if not isinstance(data, list) or len(data) < 6:
        return None
    fx, fy, cx, cy = float(data[0]), float(data[4]), float(data[2]), float(data[5])
    return fx, fy, cx, cy


def _resolve_path(base_dir: Path, frames_meta_dir: Path, rel: str, *, extra_exts: Iterable[str] | None = None) -> Path | None:
    rel_path = Path(rel)
    candidates: List[Path] = []
    if rel_path.suffix:
        candidates.extend(
            [
                frames_meta_dir / rel_path,
                base_dir / rel_path,
                base_dir / rel_path.name,
            ]
        )
    else:
        stems = [rel_path.name]
        exts = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr"]
        if extra_exts:
            exts = list(dict.fromkeys(list(exts) + list(extra_exts)))
        for ext in exts:
            candidates.extend(
                [
                    frames_meta_dir / f"{rel_path}{ext}",
                    base_dir / f"{rel_path}{ext}",
                    base_dir / f"{rel_path.name}{ext}",
                ]
            )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_frames(frames_meta_path: Path, rgb_dir: Path, depth_dir: Path) -> list[Frame]:
    meta_dir = frames_meta_path.parent
    data = json.loads(frames_meta_path.read_text())
    kfs = data.get("keyframes_metadata") or []
    if not isinstance(kfs, list) or not kfs:
        raise ValueError(f"No keyframes_metadata entries in {frames_meta_path}")

    frames: list[Frame] = []
    for entry in kfs:
        if not isinstance(entry, Mapping):
            continue
        image_name = entry.get("image_name") or entry.get("relative_image_path")
        if not image_name:
            continue
        ts_us_raw = entry.get("timestamp_microseconds") or entry.get("timestamp")
        if ts_us_raw is None:
            continue
        ts_us = int(ts_us_raw)
        ts_sec = float(ts_us) * 1e-6

        pose = _pose_from_metadata(entry)

        rgb_path = _resolve_path(rgb_dir, meta_dir, str(image_name))
        if rgb_path is None:
            print(f"[warn] RGB not found for {image_name}, skipping frame", file=sys.stderr)
            continue
        base_name = Path(rgb_path).stem

        depth_rel = entry.get("depth_path") or f"{base_name}.png"
        depth_path = _resolve_path(depth_dir, meta_dir, str(depth_rel))
        if depth_path is None:
            print(f"[warn] depth not found for {base_name}, skipping frame", file=sys.stderr)
            continue

        frames.append(Frame(ts_sec=ts_sec, rgb_path=rgb_path, depth_path=depth_path, pose_w_c=pose, frame_id=base_name))

    frames.sort(key=lambda f: f.ts_sec)
    if not frames:
        raise ValueError(f"No usable frames resolved from {frames_meta_path}")
    return frames


def _rr_set_time(rr, frame_idx: int, ts_sec: float) -> None:
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=ts_sec)
    else:
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame", frame_idx)
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", ts_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse cuSFM keyframes into nvblox_torch with rerun UI.")
    parser.add_argument("--frames-meta", type=Path, required=True, help="Path to cuSFM frames_meta.json (keyframes).")
    parser.add_argument("--rgb-dir", type=Path, required=True, help="RGB frames folder (e.g., iphone_mono).")
    parser.add_argument("--depth-dir", type=Path, required=True, help="Depth frames folder aligned to RGB.")
    parser.add_argument("--calibration", type=Path, help="Optional calibration YAML (fx, fy, cx, cy).")
    parser.add_argument("--voxel-size-m", type=float, default=0.03)
    parser.add_argument("--max-integration-distance-m", type=float, default=5.0)
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Meters per depth unit when depth is uint16.")
    parser.add_argument("--mesh-every", type=int, default=20, help="Update mesh every N frames.")
    parser.add_argument("--out-dir", type=Path, help="Output folder (default: <rgb-dir>/../nvblox_sfm_out).")
    parser.add_argument("--mesh-path", type=Path, help="PLY output path (default: <out-dir>/nvblox_mesh.ply).")
    parser.add_argument("--invert-pose", action="store_true", help="Invert poses if frames_meta stores T_C_W instead of T_W_C.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames to fuse (0 = all).")
    parser.add_argument("--no-ui", dest="ui", action="store_false", help="Disable rerun UI logging.")
    parser.set_defaults(ui=True)
    args = parser.parse_args()

    frames_meta_path = args.frames_meta.expanduser().resolve()
    rgb_dir = args.rgb_dir.expanduser().resolve()
    depth_dir = args.depth_dir.expanduser().resolve()
    if not frames_meta_path.exists():
        raise FileNotFoundError(frames_meta_path)
    for p, desc in [(rgb_dir, "RGB dir"), (depth_dir, "Depth dir")]:
        if not p.exists():
            raise FileNotFoundError(f"{desc} not found: {p}")

    frames = _load_frames(frames_meta_path, rgb_dir, depth_dir)
    data = json.loads(frames_meta_path.read_text())
    intrinsics = None
    if args.calibration:
        fx, fy, cx, cy = _parse_calibration_yaml(args.calibration.expanduser().resolve())
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        vals = _intrinsics_from_frames_meta(data)
        if vals:
            fx, fy, cx, cy = vals
            intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        else:
            raise RuntimeError("No calibration provided and frames_meta lacks calibration_parameters.")

    sample_rgb = cv2.imread(str(frames[0].rgb_path), cv2.IMREAD_COLOR)
    if sample_rgb is None:
        raise RuntimeError(f"Failed to read sample RGB: {frames[0].rgb_path}")
    h, w = sample_rgb.shape[:2]
    print(f"[info] Using intrinsics {intrinsics.tolist()} size {w}x{h}")

    out_dir = (args.out_dir or (rgb_dir.parent / "nvblox_sfm_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = (args.mesh_path or (out_dir / "nvblox_mesh.ply")).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("nvblox_torch requires CUDA; torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    print(f"[info] Frames: {len(frames)}, device: {device}")

    try:
        from nvblox_torch.mapper import Mapper  # type: ignore
        from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Failed to import nvblox_torch. Install the CUDA-matched nvblox_torch wheel first."
        ) from exc

    projective_params = ProjectiveIntegratorParams()
    projective_params.projective_integrator_max_integration_distance_m = float(args.max_integration_distance_m)
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_params)
    mapper = Mapper(voxel_sizes_m=float(args.voxel_size_m), mapper_parameters=mapper_params)

    rr = None
    rerun_enabled = bool(args.ui)
    if rerun_enabled:
        try:
            import rerun as rr  # type: ignore
            rr.init("nvblox_sfm", spawn=True)
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log(
                "world/camera",
                rr.Pinhole(
                    focal_length=[float(intrinsics[0, 0]), float(intrinsics[1, 1])],
                    principal_point=[float(intrinsics[0, 2]), float(intrinsics[1, 2])],
                    width=w,
                    height=h,
                ),
            )
        except Exception as exc:
            raise RuntimeError("Requested UI but failed to init rerun (pip install rerun-sdk).") from exc

    path_points: list[np.ndarray] = []
    start_time = time.perf_counter()
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else len(frames)

    for idx, frame in enumerate(frames[:max_frames]):
        rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[warn] skip frame {frame.frame_id}: failed to read RGB", file=sys.stderr)
            continue
        depth_raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"[warn] skip frame {frame.frame_id}: failed to read depth", file=sys.stderr)
            continue

        if depth_raw.dtype == np.uint16:
            depth_m = depth_raw.astype(np.float32) * float(args.depth_scale)
        elif depth_raw.dtype in (np.float32, np.float64):
            depth_m = depth_raw.astype(np.float32)
        else:
            raise ValueError(f"Depth must be uint16 or float; got {depth_raw.dtype} at {frame.depth_path}")

        rgb_uint8 = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        pose = frame.pose_w_c.copy()
        if args.invert_pose:
            pose = np.linalg.inv(pose)

        depth_t = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)
        rgb_t = torch.from_numpy(rgb_uint8).to(device=device, dtype=torch.uint8)
        pose_t = torch.from_numpy(pose).to(device="cpu", dtype=torch.float32)
        intrinsics_t = torch.from_numpy(intrinsics).to(device="cpu", dtype=torch.float32)

        t0 = time.perf_counter()
        mapper.add_depth_frame(depth_t, pose_t, intrinsics_t)
        mapper.add_color_frame(rgb_t, pose_t, intrinsics_t)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        path_points.append(pose[:3, 3].copy())

        if rerun_enabled:
            _rr_set_time(rr, idx, frame.ts_sec)
            rr.log(
                "world/camera_pose",
                rr.Transform3D(translation=pose[:3, 3].tolist(), rotation=pose[:3, :3].tolist()),
            )
            if path_points:
                rr.log("world/path", rr.LineStrips3D([path_points]))
            rr.log("world/rgb", rr.Image(rgb_uint8))
            rr.log("world/depth", rr.DepthImage(depth_m.astype(np.float32), meter=1.0))

        if args.mesh_every > 0 and idx % int(args.mesh_every) == 0:
            try:
                mapper.update_color_mesh()
                mesh = mapper.get_color_mesh().to_open3d()
                if rerun_enabled:
                    import numpy as _np

                    verts = _np.asarray(mesh.vertices, dtype=_np.float32)
                    faces = _np.asarray(mesh.triangles, dtype=_np.int32)
                    vcolors = None
                    if mesh.has_vertex_colors():
                        vcolors = _np.asarray(mesh.vertex_colors, dtype=_np.float32)
                        if vcolors.max() > 1.0:
                            vcolors = vcolors / 255.0
                    rr.log(
                        "world/mesh",
                        rr.Mesh3D(
                            vertex_positions=verts,
                            triangle_indices=faces,
                            vertex_colors=vcolors if mesh.has_vertex_colors() else None,
                        ),
                    )
            except Exception as exc:
                print(f"[warn] mesh update failed at frame {frame.frame_id}: {exc}", file=sys.stderr)

        print(f"[info] fused frame {idx+1}/{max_frames} ({frame.frame_id}) in {dt_ms:.1f} ms")

    # Final mesh export
    try:
        mapper.update_color_mesh()
        mesh = mapper.get_color_mesh().to_open3d()
        mesh_dir = mesh_path.parent
        mesh_dir.mkdir(parents=True, exist_ok=True)
        try:
            import open3d as o3d  # type: ignore

            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            print(f"[info] mesh written to {mesh_path}")
        except ModuleNotFoundError:
            print("[warn] open3d not installed; skipping mesh file write", file=sys.stderr)
        if rerun_enabled:
            import numpy as _np

            verts = _np.asarray(mesh.vertices, dtype=_np.float32)
            faces = _np.asarray(mesh.triangles, dtype=_np.int32)
            vcolors = None
            if mesh.has_vertex_colors():
                vcolors = _np.asarray(mesh.vertex_colors, dtype=_np.float32)
                if vcolors.max() > 1.0:
                    vcolors = vcolors / 255.0
            rr.log(
                "world/mesh_final",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    vertex_colors=vcolors if mesh.has_vertex_colors() else None,
                ),
            )
    except Exception as exc:
        print(f"[warn] final mesh export failed: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - start_time
    print(f"[done] fused {min(max_frames, len(frames))} frames in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
