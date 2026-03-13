#!/usr/bin/env python3
"""
Rerun-based viewer for nvblox occupancy/ESDF voxel grids or mesh PLY files.

Usage:
  python utilities/rerun_nvblox_viewer.py path/to/voxel_grid.ply
  python utilities/rerun_nvblox_viewer.py path/to/voxel_grid.npz
  python utilities/rerun_nvblox_viewer.py path/to/mesh.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

UNOBSERVED_SENTINEL = -1000.0


def _to_rgba_u8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.ndim != 2:
        raise ValueError(f"Expected colors with shape [N,C], got {colors.shape}")
    if colors.shape[1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    elif colors.shape[1] == 4:
        alpha = None
    else:
        raise ValueError(f"Expected RGB/RGBA colors, got shape {colors.shape}")

    if np.issubdtype(colors.dtype, np.floating):
        rgba = np.clip(colors, 0.0, 1.0)
        rgba = (rgba * 255.0).astype(np.uint8)
    else:
        rgba = np.clip(colors, 0, 255).astype(np.uint8)

    if alpha is not None:
        rgba = np.concatenate([rgba, alpha], axis=1)
    return rgba


def _esdf_colors(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.tile(np.array([51, 153, 255, 255], dtype=np.uint8), (values.shape[0], 1))
    t = (values - vmin) / (vmax - vmin)
    rgb = np.stack([t, 0.2 + 0.6 * (1.0 - t), 1.0 - t], axis=1)
    alpha = np.ones((rgb.shape[0], 1), dtype=np.float32)
    return _to_rgba_u8(np.concatenate([rgb, alpha], axis=1))


def _infer_voxel_size(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.05
    diffs = np.diff(points.astype(np.float32), axis=0).reshape(-1)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.05
    return float(np.min(diffs))


def _load_voxel_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float, bool]:
    data = np.load(str(npz_path))
    required = {"voxels", "min_indices", "voxel_size", "is_occupancy_grid"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {', '.join(missing)}")

    voxels = np.asarray(data["voxels"])
    min_indices = np.asarray(data["min_indices"], dtype=np.float32).reshape(3)
    voxel_size = float(np.asarray(data["voxel_size"]).reshape(()))
    is_occupancy = bool(np.asarray(data["is_occupancy_grid"]).reshape(()))

    valid_mask = voxels != UNOBSERVED_SENTINEL
    voxel_indices = np.argwhere(valid_mask).astype(np.float32)
    centers = (voxel_indices + min_indices + 0.5) * voxel_size
    values = voxels[valid_mask].astype(np.float32)
    return centers, values, voxel_size, is_occupancy


def _parse_ply_header(path: Path) -> tuple[str, int, list[tuple[str, str]], int]:
    with path.open("rb") as f:
        magic = f.readline().decode("ascii", errors="strict").strip()
        if magic != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt: str | None = None
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element: str | None = None
        header_lines = 1

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Malformed PLY header: {path}")
            header_lines += 1
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                _, current_element, count = line.split()
                if current_element == "vertex":
                    vertex_count = int(count)
            elif line.startswith("property ") and current_element == "vertex":
                parts = line.split()
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"Unsupported vertex property in {path}: {line}")
                vertex_properties.append((parts[2], parts[1]))
            elif line == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise ValueError(f"Incomplete PLY header: {path}")
        return fmt, vertex_count, vertex_properties, header_lines


def _load_voxel_ply(path: Path) -> tuple[np.ndarray, np.ndarray, float, bool]:
    fmt, vertex_count, vertex_properties, header_lines = _parse_ply_header(path)
    prop_names = [name for name, _ in vertex_properties]
    required = {"x", "y", "z", "intensity"}
    if not required.issubset(prop_names):
        raise ValueError(f"PLY voxel grid is missing required properties: {required - set(prop_names)}")

    if fmt == "ascii":
        table = np.loadtxt(str(path), dtype=np.float32, skiprows=header_lines, max_rows=vertex_count)
        if table.ndim == 1:
            table = table[None, :]
    elif fmt == "binary_little_endian":
        dtype_map = {
            "char": "<i1",
            "int8": "<i1",
            "uchar": "<u1",
            "uint8": "<u1",
            "short": "<i2",
            "int16": "<i2",
            "ushort": "<u2",
            "uint16": "<u2",
            "int": "<i4",
            "int32": "<i4",
            "uint": "<u4",
            "uint32": "<u4",
            "float": "<f4",
            "float32": "<f4",
            "double": "<f8",
            "float64": "<f8",
        }
        dtype_fields: list[tuple[str, Any]] = []
        for name, prop_type in vertex_properties:
            if prop_type not in dtype_map:
                raise ValueError(f"Unsupported PLY property type {prop_type} in {path}")
            dtype_fields.append((name, dtype_map[prop_type]))
        with path.open("rb") as f:
            for _ in range(header_lines):
                f.readline()
            structured = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
        table = np.column_stack([structured[name] for name in prop_names]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PLY format in {path}: {fmt}")

    index = {name: idx for idx, name in enumerate(prop_names)}
    centers = table[:, [index["x"], index["y"], index["z"]]].astype(np.float32)
    values = table[:, index["intensity"]].astype(np.float32)
    voxel_size = _infer_voxel_size(centers)
    name_hint = path.name.lower()
    is_occupancy = bool(
        "occupancy" in name_hint or np.all(np.isclose(values, 0.0) | np.isclose(values, 1.0))
    )
    return centers, values, voxel_size, is_occupancy


def _detect_ply_kind(path: Path) -> str:
    lower_name = path.name.lower()
    if "voxel" in lower_name or "occupancy" in lower_name or "tsdf" in lower_name or "esdf" in lower_name:
        return "voxel"
    try:
        verts, faces, _ = _load_mesh(path)
        if verts.size and faces.size:
            return "mesh"
    except Exception:
        pass
    return "voxel"


def _load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("Open3D is required for mesh preview. Install with: python -m pip install open3d") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh or mesh is empty: {mesh_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vertex_colors = None
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
        vertex_colors = _to_rgba_u8(vertex_colors)
    return verts, faces, vertex_colors


def _init_rerun(*, app_id: str, spawn: bool, connect: str | None, save: Path | None) -> None:
    if connect and save:
        raise ValueError("Use only one of --connect or --save.")

    rr.init(app_id, spawn=spawn and connect is None and save is None)
    if connect:
        rr.connect_tcp(connect)
    if save:
        rr.save(save)


def _log_voxel_points(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    entity_root: str,
    cube: bool,
) -> None:
    if cube:
        half_sizes = np.full((points.shape[0], 3), voxel_size * 0.5, dtype=np.float32)
        rr.log(
            f"{entity_root}/boxes",
            rr.Boxes3D(
                centers=points.astype(np.float32),
                half_sizes=half_sizes,
                colors=colors,
            ),
            static=True,
        )
    else:
        rr.log(
            f"{entity_root}/points",
            rr.Points3D(
                points.astype(np.float32),
                colors=colors,
                radii=np.full(points.shape[0], max(voxel_size * 0.35, 0.002), dtype=np.float32),
            ),
            static=True,
        )


def _log_text(entity_path: str, text: str) -> None:
    rr.log(entity_path, rr.TextDocument(text), static=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize an nvblox voxel grid or mesh in Rerun.")
    parser.add_argument("file_path", type=Path, help="Path to the file to visualize.")
    parser.add_argument(
        "--max_visualization_dist_vox",
        type=int,
        default=2,
        help="Max. ESDF distance in voxels to include in the visualization.",
    )
    parser.add_argument(
        "--cube",
        action="store_true",
        help="Render voxel grids as 3D boxes instead of points.",
    )
    parser.add_argument(
        "--entity-root",
        default="nvblox",
        help="Root entity path inside Rerun.",
    )
    parser.add_argument(
        "--connect",
        type=str,
        help="Connect to an existing Rerun viewer at host:port instead of spawning one.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the recording to an .rrd file instead of only streaming it live.",
    )
    parser.add_argument(
        "--no-spawn",
        dest="spawn",
        action="store_false",
        help="Do not spawn a local Rerun viewer.",
    )
    parser.set_defaults(spawn=True)
    args = parser.parse_args()

    file_path = args.file_path.expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    _init_rerun(
        app_id="rerun_nvblox_viewer",
        spawn=args.spawn,
        connect=args.connect,
        save=args.save.expanduser().resolve() if args.save else None,
    )

    entity_root = args.entity_root.strip("/") or "nvblox"
    file_extension = file_path.suffix.lstrip(".").lower()

    if file_extension == "npz":
        centers, values, voxel_size, is_occupancy_grid = _load_voxel_npz(file_path)

        if is_occupancy_grid:
            mask = values.astype(bool)
            points = centers[mask]
            colors = np.tile(np.array([51, 230, 77, 255], dtype=np.uint8), (points.shape[0], 1))
            title = "Occupancy Voxel Grid"
        else:
            max_dist_m = float(args.max_visualization_dist_vox) * voxel_size
            mask = values < max_dist_m
            points = centers[mask]
            colors = _esdf_colors(values[mask])
            title = "ESDF Voxel Grid"

        _log_voxel_points(
            points=points,
            colors=colors,
            voxel_size=voxel_size,
            entity_root=entity_root,
            cube=args.cube,
        )
        _log_text(
            f"{entity_root}/info",
            f"{title}\nsource={file_path}\npoints={points.shape[0]}\nvoxel_size={voxel_size:.6f}m",
        )
    elif file_extension == "ply":
        kind = _detect_ply_kind(file_path)
        if kind == "voxel":
            centers, values, voxel_size, is_occupancy_grid = _load_voxel_ply(file_path)

            if is_occupancy_grid:
                mask = values.astype(bool)
                points = centers[mask]
                colors = np.tile(np.array([51, 230, 77, 255], dtype=np.uint8), (points.shape[0], 1))
                title = "Occupancy Voxel Grid"
            else:
                max_dist_m = float(args.max_visualization_dist_vox) * voxel_size
                mask = values < max_dist_m
                points = centers[mask]
                colors = _esdf_colors(values[mask].astype(np.float32))
                title = "ESDF Voxel Grid"

            _log_voxel_points(
                points=points,
                colors=colors,
                voxel_size=voxel_size,
                entity_root=entity_root,
                cube=args.cube,
            )
            _log_text(
                f"{entity_root}/info",
                f"{title}\nsource={file_path}\npoints={points.shape[0]}\nvoxel_size={voxel_size:.6f}m",
            )
        else:
            verts, faces, vertex_colors = _load_mesh(file_path)
            rr.log(
                f"{entity_root}/mesh",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    vertex_colors=vertex_colors,
                ),
                static=True,
            )
            _log_text(
                f"{entity_root}/info",
                f"Mesh Viewer\nsource={file_path}\nvertices={verts.shape[0]}\ntriangles={faces.shape[0]}",
            )
    else:
        raise ValueError(f"File extension not supported: {file_extension}")

    print("Visualization complete. Check Rerun viewer or saved recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
