#!/usr/bin/env python3
r"""
Isaac Sim loader for ReconS sample outputs.

Given explicit paths to the two assets, this script:
1) Takes the 3dgrut USDZ (`export_last.usdz`)
2) Takes an nvblox mesh (`.usd` or `.ply`)
3) Builds a combined USD stage with both assets
4) Opens that stage in Isaac Sim

Run with Isaac Sim's Python on Windows, e.g.:
    C:\isaac-sim\python.bat G:\GithubProject\recons\utilities\isaacsim_load_sample.py ^
        --usdz-path G:\GithubProject\recons\data\sample_20260302_i9_1080\3dgrut_out\run\export_last.usdz ^
        --mesh-path G:\GithubProject\recons\data\sample_20260302_i9_1080\nvblox_out\mesh.usd
"""

from __future__ import annotations

import argparse
import os
import site
import struct
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load 3dgrut USDZ + nvblox mesh into Isaac Sim.")
    parser.add_argument(
        "--usdz-path",
        required=True,
        help="Path to the 3dgrut USDZ file (e.g. .../3dgrut_out/run/export_last.usdz).",
    )
    parser.add_argument(
        "--mesh-path",
        required=True,
        help="Path to the nvblox mesh file (.usd/.usda/.usdc/.usdz/.ply).",
    )
    parser.add_argument(
        "--output-stage",
        help="Combined stage output path (default: <usdz parent dir>/isaac_combined_stage.usda).",
    )
    parser.add_argument(
        "--no-apply-volume-transform",
        action="store_true",
        help="Do not apply the volume transform matrix to the mesh.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Create combined stage and exit without opening interactive viewer loop.",
    )
    parser.add_argument(
        "--headless-seconds",
        type=float,
        default=2.0,
        help="When --headless and not --stage-only, keep app alive this many seconds (default: 2.0).",
    )
    return parser.parse_args()


def _create_simulation_app(headless: bool):
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore
    return SimulationApp({"headless": headless})


def _sanitize_python_env_for_isaac() -> None:
    """
    Remove user-site package paths that can shadow Isaac Sim bundled deps.

    This specifically prevents accidental imports from:
      %APPDATA%\\Python\\Python311\\site-packages
    where incompatible torch wheels often live.
    """
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)

    candidates: List[str] = []
    try:
        candidates.append(str(Path(site.getusersitepackages()).resolve()).lower())
    except Exception:
        pass
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(str((Path(appdata) / "Python").resolve()).lower())

    if not candidates:
        return

    cleaned: List[str] = []
    removed: List[str] = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)
            continue
        try:
            resolved = str(Path(entry).resolve()).lower()
        except Exception:
            resolved = entry.lower()
        if any(c in resolved for c in candidates):
            removed.append(entry)
        else:
            cleaned.append(entry)
    sys.path[:] = cleaned

    # If torch was pre-imported from user-site, drop it so Isaac can load bundled torch.
    mod = sys.modules.get("torch")
    if mod is not None:
        mod_file = str(getattr(mod, "__file__", "")).lower()
        if any(c in mod_file for c in candidates):
            del sys.modules["torch"]

    if removed:
        print("[info] Removed user-site paths from sys.path:")
        for p in removed:
            print(f"  - {p}")



@dataclass
class _PlyProperty:
    name: str
    dtype: str
    is_list: bool = False
    count_dtype: str | None = None
    item_dtype: str | None = None


@dataclass
class _PlyElement:
    name: str
    count: int
    properties: List[_PlyProperty]


_PLY_TO_STRUCT = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _parse_ply_header(path: Path) -> Tuple[str, List[_PlyElement], int]:
    with path.open("rb") as f:
        first = f.readline().decode("ascii", errors="ignore").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")

        fmt = ""
        elements: List[_PlyElement] = []
        current: _PlyElement | None = None

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY header in {path}: missing end_header.")
            line = raw.decode("ascii", errors="ignore").strip()
            if not line or line.startswith("comment"):
                continue
            if line == "end_header":
                return fmt, elements, f.tell()
            parts = line.split()
            key = parts[0]
            if key == "format":
                if len(parts) < 2:
                    raise ValueError(f"Invalid format line in {path}: {line}")
                fmt = parts[1]
            elif key == "element":
                if len(parts) != 3:
                    raise ValueError(f"Invalid element line in {path}: {line}")
                current = _PlyElement(name=parts[1], count=int(parts[2]), properties=[])
                elements.append(current)
            elif key == "property":
                if current is None:
                    raise ValueError(f"Property without element in {path}: {line}")
                if len(parts) >= 5 and parts[1] == "list":
                    current.properties.append(
                        _PlyProperty(
                            name=parts[4],
                            dtype="list",
                            is_list=True,
                            count_dtype=parts[2],
                            item_dtype=parts[3],
                        )
                    )
                elif len(parts) == 3:
                    current.properties.append(_PlyProperty(name=parts[2], dtype=parts[1]))
                else:
                    raise ValueError(f"Invalid property line in {path}: {line}")


def _read_scalar_binary(f, dtype: str, endian: str) -> float | int:
    fmt_char = _PLY_TO_STRUCT.get(dtype)
    if fmt_char is None:
        raise ValueError(f"Unsupported PLY dtype: {dtype}")
    size = struct.calcsize(fmt_char)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Unexpected EOF while reading binary PLY.")
    return struct.unpack(endian + fmt_char, data)[0]


def _parse_ply(path: Path) -> Tuple[List[Tuple[float, float, float]], List[int], List[int]]:
    fmt, elements, data_offset = _parse_ply_header(path)
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format in {path}: {fmt}")

    points: List[Tuple[float, float, float]] = []
    face_counts: List[int] = []
    face_indices: List[int] = []

    if fmt == "ascii":
        with path.open("r", encoding="ascii", errors="ignore") as f:
            f.seek(data_offset)
            for elem in elements:
                for _ in range(elem.count):
                    line = f.readline()
                    if not line:
                        raise ValueError("Unexpected EOF while reading ascii PLY.")
                    tokens = line.strip().split()
                    cursor = 0
                    values: Dict[str, Any] = {}
                    for prop in elem.properties:
                        if prop.is_list:
                            if prop.count_dtype is None:
                                raise ValueError("Invalid list property in ascii PLY.")
                            n = int(tokens[cursor])
                            cursor += 1
                            arr = [int(tokens[cursor + i]) for i in range(n)]
                            cursor += n
                            values[prop.name] = arr
                        else:
                            values[prop.name] = float(tokens[cursor])
                            cursor += 1

                    if elem.name == "vertex":
                        x = float(values.get("x", 0.0))
                        y = float(values.get("y", 0.0))
                        z = float(values.get("z", 0.0))
                        points.append((x, y, z))
                    elif elem.name == "face":
                        idx = values.get("vertex_indices") or values.get("vertex_index")
                        if isinstance(idx, list) and idx:
                            face_counts.append(len(idx))
                            face_indices.extend(int(v) for v in idx)
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        with path.open("rb") as f:
            f.seek(data_offset)
            for elem in elements:
                for _ in range(elem.count):
                    values: Dict[str, Any] = {}
                    for prop in elem.properties:
                        if prop.is_list:
                            if prop.count_dtype is None or prop.item_dtype is None:
                                raise ValueError("Invalid list property in binary PLY.")
                            n = int(_read_scalar_binary(f, prop.count_dtype, endian))
                            arr = [_read_scalar_binary(f, prop.item_dtype, endian) for _ in range(n)]
                            values[prop.name] = arr
                        else:
                            values[prop.name] = _read_scalar_binary(f, prop.dtype, endian)

                    if elem.name == "vertex":
                        x = float(values.get("x", 0.0))
                        y = float(values.get("y", 0.0))
                        z = float(values.get("z", 0.0))
                        points.append((x, y, z))
                    elif elem.name == "face":
                        idx = values.get("vertex_indices") or values.get("vertex_index")
                        if isinstance(idx, list) and idx:
                            face_counts.append(len(idx))
                            face_indices.extend(int(v) for v in idx)

    if not points:
        raise ValueError(f"No vertices found in mesh file: {path}")
    if not face_counts:
        raise ValueError(f"No faces found in mesh file: {path}")

    return points, face_counts, face_indices


def _extract_transform_from_usdz(usdz_path: Path, Usd, UsdGeom, Gf):
    identity = Gf.Matrix4d(1.0)
    try:
        with zipfile.ZipFile(usdz_path, "r") as zf:
            names = zf.namelist()
            preferred = [n for n in names if n.lower().endswith("gauss.usda")]
            fallback = [n for n in names if n.lower().endswith(".usda")]
            target = preferred[0] if preferred else (fallback[0] if fallback else None)
            if target is None:
                return identity
            with tempfile.TemporaryDirectory() as tmp:
                zf.extract(target, tmp)
                tmp_file = Path(tmp) / target
                stage = Usd.Stage.Open(str(tmp_file))
                if stage is None:
                    return identity
                for prim in stage.Traverse():
                    if not prim.IsA(UsdGeom.Xformable):
                        continue
                    xformable = UsdGeom.Xformable(prim)
                    for op in xformable.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                            value = op.Get()
                            return value if isinstance(value, Gf.Matrix4d) else Gf.Matrix4d(value)
    except Exception:
        return identity
    return identity


def _build_combined_stage(
    output_stage: Path,
    volume_usdz: Path,
    mesh_path: Path,
    apply_transform: bool,
    Usd,
    UsdGeom,
    Gf,
) -> None:
    output_stage.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_stage))
    if stage is None:
        raise RuntimeError(f"Failed to create stage: {output_stage}")

    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetDefaultPrim(world.GetPrim())

    mesh_xform = UsdGeom.Xform.Define(stage, "/World/mesh")
    volume_xform = UsdGeom.Xform.Define(stage, "/World/volume")
    volume_xform.GetPrim().GetReferences().AddReference(volume_usdz.resolve().as_posix())

    mesh_suffix = mesh_path.suffix.lower()
    if mesh_suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        mesh_xform.GetPrim().GetReferences().AddReference(mesh_path.resolve().as_posix())
    elif mesh_suffix == ".ply":
        points, face_counts, face_indices = _parse_ply(mesh_path)
        mesh_prim = UsdGeom.Mesh.Define(stage, "/World/mesh/nvblox_mesh")
        mesh_prim.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
        mesh_prim.CreateFaceVertexCountsAttr(face_counts)
        mesh_prim.CreateFaceVertexIndicesAttr(face_indices)
        mesh_prim.CreateSubdivisionSchemeAttr().Set("none")
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    if apply_transform:
        xform = _extract_transform_from_usdz(volume_usdz, Usd, UsdGeom, Gf)
        xformable = UsdGeom.Xformable(mesh_xform.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(xform)

    stage.GetRootLayer().Save()


def main() -> int:
    args = _parse_args()

    _sanitize_python_env_for_isaac()
    simulation_app = _create_simulation_app(args.headless)
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, Usd, UsdGeom  # type: ignore

        usdz_path = Path(args.usdz_path).expanduser().resolve()
        mesh_path = Path(args.mesh_path).expanduser().resolve()
        output_stage = (
            Path(args.output_stage).expanduser().resolve()
            if args.output_stage
            else (usdz_path.parent / "isaac_combined_stage.usda").resolve()
        )
        apply_transform = not args.no_apply_volume_transform

        if not usdz_path.is_file():
            raise FileNotFoundError(f"USDZ file not found: {usdz_path}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        _build_combined_stage(
            output_stage=output_stage,
            volume_usdz=usdz_path,
            mesh_path=mesh_path,
            apply_transform=apply_transform,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Gf=Gf,
        )

        print(f"[info] volume_usdz: {usdz_path}")
        print(f"[info] mesh_path: {mesh_path}")
        print(f"[info] combined_stage: {output_stage}")

        if args.stage_only:
            return 0

        ctx = omni.usd.get_context()
        if not ctx.open_stage(output_stage.resolve().as_posix()):
            raise RuntimeError(f"Failed to open stage in Isaac Sim: {output_stage}")

        for _ in range(10):
            simulation_app.update()

        if args.headless:
            end_t = time.time() + max(0.1, float(args.headless_seconds))
            while simulation_app.is_running() and time.time() < end_t:
                simulation_app.update()
        else:
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
