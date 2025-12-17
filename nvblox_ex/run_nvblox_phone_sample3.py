"""Run nvblox_torch mapping on the prepared phone_sample3 dataset.

This script is intended to be run inside WSL (Ubuntu) with an NVIDIA GPU.
It reads:
- associations.txt: timestamp, rgb_path, timestamp, depth_path
- CameraTrajectory.csv: per-frame poses (timestamp, tx, ty, tz, qx, qy, qz, qw)
- iphone_intrinsics.json: camera intrinsics (fx, fy, cx, cy)

Outputs:
- A mesh export written to --out_dir (if export is supported by installed nvblox_torch).

Note: The nvblox_torch Python API evolves; this runner targets the documented
Mapper interface (add_depth_frame/add_color_frame/update_color_mesh).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch


def _read_intrinsics_json(path: Path, *, width: int, height: int) -> np.ndarray:
    data = json.loads(path.read_text())
    entries = data.get("intrinsics") or []
    for entry in entries:
        res = entry.get("resolution") or {}
        if int(res.get("width", -1)) == int(width) and int(res.get("height", -1)) == int(height):
            fx = float(entry["fx"])
            fy = float(entry["fy"])
            cx = float(entry["cx"])
            cy = float(entry["cy"])
            # nvblox_torch expects 3x3 intrinsics matrix.
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"No matching intrinsics for {width}x{height} in {path}")


class LiveMeshViewer:
    """Minimal Open3D-based live mesh viewer."""

    def __init__(self, title: str = "nvblox mesh", width: int = 960, height: int = 720):
        try:
            import open3d as o3d  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "Open3D is required for --ui. Install with: python -m pip install open3d"
            ) from exc

        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=width, height=height)
        self.mesh_handle = None

    def update(self, mesh_o3d):
        if mesh_o3d is None:
            return
        if self.mesh_handle is None:
            self.mesh_handle = mesh_o3d
            self.vis.add_geometry(self.mesh_handle)
        else:
            # Replace geometry to avoid mutable references issues
            self.vis.remove_geometry(self.mesh_handle, reset_bounding_box=False)
            self.mesh_handle = mesh_o3d
            self.vis.add_geometry(self.mesh_handle)
        self.vis.poll_events()
        self.vis.update_renderer()

    def refresh(self):
        if self.mesh_handle is None:
            return
        self.vis.poll_events()
        self.vis.update_renderer()

    def run_forever(self):
        while self.vis.poll_events():
            self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


def _read_camera_trajectory_csv(path: Path) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Return mapping timestamp -> (t_xyz, q_xyzw)."""
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} missing required columns; got: {reader.fieldnames}")
        for row in reader:
            ts = float(row["timestamp"])
            t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float32)
            q = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float32)
            out[ts] = (t, q)
    return out


def _build_pose_lookup(traj: Dict[float, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    ts_sorted = np.array(sorted(traj.keys()), dtype=np.float64)
    return ts_sorted, traj


def _lookup_pose(ts: float, ts_sorted: np.ndarray, traj: Dict[float, Tuple[np.ndarray, np.ndarray]], tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Find pose for timestamp; use nearest neighbor within tol seconds."""
    if ts in traj:
        return traj[ts]
    idx = np.searchsorted(ts_sorted, ts)
    candidates = []
    if idx > 0:
        candidates.append(ts_sorted[idx - 1])
    if idx < len(ts_sorted):
        candidates.append(ts_sorted[idx])
    if not candidates:
        raise KeyError(f"No pose for timestamp {ts}")
    nearest = min(candidates, key=lambda t: abs(t - ts))
    if abs(nearest - ts) > tol:
        raise KeyError(f"No pose for timestamp {ts} (nearest={nearest}, tol={tol})")
    return traj[float(nearest)]


def _read_associations(path: Path) -> Iterable[Tuple[float, Path, Path]]:
    """Yield (timestamp, rgb_path, depth_path)."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad associations line: {line}")
        ts = float(parts[0])
        rgb_path = Path(parts[1])
        depth_path = Path(parts[3])
        yield ts, rgb_path, depth_path


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _make_pose_matrix(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Build 4x4 camera pose matrix from translation + quaternion.

    Assumes the trajectory provides camera-in-world (T_W_C). If your output looks
    wrong, try inverting this matrix before feeding it to nvblox.
    """
    r = _quat_xyzw_to_rotmat(q_xyzw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = r
    pose[:3, 3] = t
    return pose


def _pose_changed(prev_pose: np.ndarray | None, pose: np.ndarray, pos_eps: float = 1e-4, rot_eps_deg: float = 0.1) -> bool:
    """Check if pose changed beyond thresholds."""
    if prev_pose is None:
        return True
    dp = np.linalg.norm(pose[:3, 3] - prev_pose[:3, 3])
    r_rel = pose[:3, :3] @ prev_pose[:3, :3].T
    cos_theta = np.clip((np.trace(r_rel) - 1) * 0.5, -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cos_theta))
    return dp > pos_eps or ang_deg > rot_eps_deg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Path to phone_sample3 folder")
    parser.add_argument(
        "--intrinsics_json",
        type=Path,
        default=Path(__file__).resolve().parent / "iphone_intrinsics.json",
        help="Path to iphone_intrinsics.json",
    )
    parser.add_argument("--voxel_size_m", type=float, default=0.03)
    parser.add_argument("--max_integration_distance_m", type=float, default=5.0)
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Meters per depth unit (uint16 mm => 0.001)")
    parser.add_argument("--mesh_every", type=int, default=50, help="Update mesh every N frames")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_nvblox"))
    parser.add_argument("--invert_pose", action="store_true", help="Invert poses before integration (if your trajectory is T_C_W)")
    parser.add_argument("--ui", action="store_true", help="Show live mesh viewer (requires open3d)")
    parser.add_argument(
        "--skip_static_poses",
        action="store_true",
        help="Skip frames whose pose is effectively identical to the previous frame (helps when trajectory has stalls)",
    )

    args = parser.parse_args()

    dataset = args.dataset
    associations_path = dataset / "associations.txt"
    traj_path = dataset / "CameraTrajectory.csv"

    if not associations_path.exists():
        raise FileNotFoundError(associations_path)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    traj = _read_camera_trajectory_csv(traj_path)
    ts_sorted, traj_map = _build_pose_lookup(traj)

    if not torch.cuda.is_available():
        raise RuntimeError("nvblox_torch requires a CUDA-capable GPU. torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Import nvblox_torch lazily so the script can be inspected without it.
    # Note: Different nvblox_torch versions expose classes from different modules.
    try:
        from nvblox_torch.mapper import Mapper  # type: ignore
        from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Failed to import nvblox_torch Mapper APIs. "
            "Make sure you installed the correct nvblox_torch wheel for your Ubuntu/CUDA version. "
            "This script expects Mapper to be available as nvblox_torch.mapper.Mapper."
        ) from exc

    projective_params = ProjectiveIntegratorParams()
    projective_params.projective_integrator_max_integration_distance_m = float(args.max_integration_distance_m)
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_params)

    mapper = Mapper(
        voxel_sizes_m=float(args.voxel_size_m),
        mapper_parameters=mapper_params,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics: np.ndarray | None = None
    viewer: LiveMeshViewer | None = LiveMeshViewer() if args.ui else None
    prev_pose: np.ndarray | None = None
    # Use median frame spacing as tolerance for nearest pose lookup.
    if len(ts_sorted) > 1:
        tol = float(np.median(np.diff(ts_sorted)) * 0.51)
    else:
        tol = 1e-3

    for idx, (ts, rgb_rel, depth_rel) in enumerate(_read_associations(associations_path)):
        rgb_path = (dataset / rgb_rel).resolve()
        depth_path = (dataset / depth_rel).resolve()

        try:
            t, q = _lookup_pose(ts, ts_sorted, traj_map, tol)
        except KeyError as exc:
            print(f"Skipping frame {idx} (t={ts:.6f}): {exc}")
            if viewer:
                viewer.refresh()
            continue

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read rgb: {rgb_path}")

        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            raise RuntimeError(f"Failed to read depth: {depth_path}")
        if depth_u16.ndim != 2 or depth_u16.dtype != np.uint16:
            raise ValueError(f"Expected uint16 single-channel depth, got {depth_u16.shape} {depth_u16.dtype} at {depth_path}")

        h, w = rgb.shape[:2]
        if intrinsics is None:
            intrinsics = _read_intrinsics_json(args.intrinsics_json, width=w, height=h)

        # Convert to meters float32 for nvblox.
        depth_m = depth_u16.astype(np.float32) * float(args.depth_scale)

        # nvblox_torch color expects RGB uint8 with 3 channels on GPU.
        rgb_uint8 = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        pose = _make_pose_matrix(t, q)
        if args.invert_pose:
            pose = np.linalg.inv(pose)

        if args.skip_static_poses and not _pose_changed(prev_pose, pose):
            if viewer:
                viewer.refresh()
            print(f"Skipped frame {idx} (t={ts:.6f}) - static pose")
            continue
        prev_pose = pose

        # Mapper expects depth/color on GPU, intrinsics and pose on CPU.
        depth_t = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)
        rgb_t = torch.from_numpy(rgb_uint8).to(device=device, dtype=torch.uint8)
        pose_t = torch.from_numpy(pose).to(device="cpu", dtype=torch.float32)
        intrinsics_t = torch.from_numpy(intrinsics).to(device="cpu", dtype=torch.float32)

        mapper.add_depth_frame(depth_t, pose_t, intrinsics_t)
        mapper.add_color_frame(rgb_t, pose_t, intrinsics_t)

        if args.mesh_every > 0 and (idx % int(args.mesh_every) == 0):
            mapper.update_color_mesh()
            mesh = mapper.get_color_mesh().to_open3d()
            if viewer:
                viewer.update(mesh)
            print(f"Integrated frame {idx} (t={ts:.6f}) - updated mesh")
        else:
            if viewer:
                viewer.refresh()
            print(f"Integrated frame {idx} (t={ts:.6f})")

    # Final mesh update and export if available.
    mapper.update_color_mesh()
    final_mesh = mapper.get_color_mesh().to_open3d()
    if viewer:
        viewer.update(final_mesh)

    # Export: API name differs across versions. Try mapper methods first, then fall back to ColorMesh.save.
    out_mesh = args.out_dir / "mesh.ply"
    exported = False
    for method_name in ("save_mesh", "export_mesh", "write_mesh"):
        fn = getattr(mapper, method_name, None)
        if callable(fn):
            fn(str(out_mesh))
            exported = True
            break

    if not exported:
        try:
            mesh_obj = mapper.get_color_mesh()
            # ColorMesh.save uses Open3D to write the mesh.
            mesh_obj.save(str(out_mesh))
            exported = True
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"Mesh export via ColorMesh.save failed: {exc}")

    if exported:
        print(f"Wrote mesh: {out_mesh}")
    else:
        print("Mesh export method not found on Mapper. You can still visualize or access mesh/layers via the nvblox_torch API.")

    if viewer:
        print("Live viewer running; close the window to exit.")
        viewer.run_forever()
        viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
