#!/usr/bin/env python3
"""
Prepare and run 3dgrut on either:
1) a ReconS sample folder, or
2) explicitly provided image/sparse/init paths.

This script adapts our sample layout to COLMAP-style structure expected by 3dgrut:
  <work_dir>/
    images/<prefix>/...
    sparse/0/{cameras.txt,images.txt,points3D.txt,...}

Then it launches third_party/3dgrut/train.py with apps/cusfm_3dgut.yaml and writes
outputs under <sample_dir>/3dgrut_out by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _read_first_colmap_image_name(images_txt: Path) -> str:
    for raw in images_txt.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            return parts[9]
    raise ValueError(f"Could not parse first image name from {images_txt}")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_dir(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
        return "symlink"
    except OSError:
        shutil.copytree(src, dst)
        return "copy"


def _prepare_dataset_layout(
    work_dir: Path,
    images_dir: Path,
    sparse_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    images_txt = sparse_dir / "images.txt"
    first_rel_image = _read_first_colmap_image_name(images_txt)
    prefix = str(Path(first_rel_image).parent).strip(".")

    images_root = work_dir / "images"
    sparse_root = work_dir / "sparse"
    sparse_zero = sparse_root / "0"

    if overwrite and work_dir.exists():
        _remove_path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if sparse_zero.exists() or sparse_zero.is_symlink():
        if overwrite:
            _remove_path(sparse_zero)
        else:
            raise FileExistsError(f"{sparse_zero} already exists. Pass --overwrite to rebuild.")

    sparse_mode = _link_or_copy_dir(sparse_dir, sparse_zero)

    if prefix:
        target_images_dir = images_root / prefix
    else:
        target_images_dir = images_root

    if target_images_dir.exists() or target_images_dir.is_symlink():
        if overwrite:
            _remove_path(target_images_dir)
        else:
            raise FileExistsError(f"{target_images_dir} already exists. Pass --overwrite to rebuild.")

    images_mode = _link_or_copy_dir(images_dir, target_images_dir)

    return {
        "work_dir": str(work_dir),
        "images_dir": str(images_dir),
        "images_layout_prefix": prefix,
        "images_mount_mode": images_mode,
        "sparse_dir": str(sparse_dir),
        "sparse_mount_mode": sparse_mode,
    }


def _detect_fused_point_cloud(sample_dir: Path) -> Path:
    candidates = [
        sample_dir / "nvblox_sfm_out" / "nvblox_mesh.ply",
        sample_dir / "nvblox_out" / "mesh.ply",
        sample_dir / "nvblox_out" / "tsdf_voxel_grid.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No fused point cloud candidate found. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _latest_child_dir(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Prepare and run 3dgrut for a ReconS sample or explicit folders.")
    parser.add_argument(
        "--sample-dir",
        type=Path,
        help="Sample folder (e.g. data/sample_20260119_i4). Optional if --images-dir/--sparse-dir are provided.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="RGB image folder. Required when --sample-dir is not set (default: <sample-dir>/iphone_mono).",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        help="cuSFM/COLMAP sparse folder with cameras/images/points files. Required when --sample-dir is not set (default: <sample-dir>/cusfm_output/sparse).",
    )
    parser.add_argument(
        "--fused-point-cloud",
        type=Path,
        help="Initialization PLY path (default: auto-detect nvblox_sfm_out/nvblox_mesh.ply).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Prepared COLMAP-style dataset root. Required when --sample-dir is not set (default: <sample-dir>/3dgrut_data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="3dgrut output directory root. Required when --sample-dir is not set (default: <sample-dir>/3dgrut_out).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="3dgrut experiment group name (default: <sample-dir-name>_3dgut).",
    )
    parser.add_argument("--config-name", type=str, default="apps/cusfm_3dgut.yaml")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("third_party/3dgrut/train.py"),
        help="Path to 3dgrut train.py relative to repo root or absolute.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        help="Optional conda env to run with, via `conda run -n <env> ...`.",
    )
    parser.add_argument("--max-steps", type=int, help="Optional override for max_steps.")
    parser.add_argument("--downsample-factor", type=int, help="Optional override for dataset.downsample_factor.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Extra Hydra override(s).")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare dataset and write manifest, skip training.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate prepared dataset layout if it already exists.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional path for preparation manifest JSON (default: <sample-dir>/3dgrut_manifest.json or <out-dir>/3dgrut_manifest.json).",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        help="Optional path for run summary JSON (default: <sample-dir>/3dgrut_result.json or <out-dir>/3dgrut_result.json).",
    )
    parser.add_argument("--no-export-usdz", dest="export_usdz", action="store_false")
    parser.add_argument("--export-usdz", dest="export_usdz", action="store_true")
    parser.set_defaults(export_usdz=True)
    parser.add_argument(
        "--apply-normalizing-transform",
        action="store_true",
        help="Enable export_usdz.apply_normalizing_transform (default: false for coordinate consistency).",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve() if args.sample_dir else None
    if sample_dir and not sample_dir.is_dir():
        raise SystemExit(f"Sample directory not found: {sample_dir}")

    if not sample_dir and not args.images_dir:
        raise SystemExit("Provide --images-dir when --sample-dir is not set.")
    if not sample_dir and not args.sparse_dir:
        raise SystemExit("Provide --sparse-dir when --sample-dir is not set.")
    if not sample_dir and not args.work_dir:
        raise SystemExit("Provide --work-dir when --sample-dir is not set.")
    if not sample_dir and not args.out_dir:
        raise SystemExit("Provide --out-dir when --sample-dir is not set.")

    default_images_dir = (sample_dir / "iphone_mono") if sample_dir else None
    images_dir = (args.images_dir or default_images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    default_sparse_dir = (sample_dir / "cusfm_output" / "sparse") if sample_dir else None
    sparse_dir = (args.sparse_dir or default_sparse_dir).expanduser().resolve()
    required_sparse = ["cameras.txt", "images.txt", "points3D.txt"]
    for filename in required_sparse:
        p = sparse_dir / filename
        if not p.exists():
            raise SystemExit(f"Missing sparse file: {p}")

    if args.fused_point_cloud:
        fused_point_cloud = args.fused_point_cloud.expanduser().resolve()
    elif sample_dir:
        fused_point_cloud = _detect_fused_point_cloud(sample_dir)
    else:
        fused_point_cloud = None

    if not args.prepare_only and fused_point_cloud is None:
        raise SystemExit("Provide --fused-point-cloud when --sample-dir is not set and training is requested.")
    if fused_point_cloud is not None and not fused_point_cloud.exists():
        raise SystemExit(f"Fused point cloud file not found: {fused_point_cloud}")

    default_work_dir = (sample_dir / "3dgrut_data") if sample_dir else None
    default_out_dir = (sample_dir / "3dgrut_out") if sample_dir else None
    work_dir = (args.work_dir or default_work_dir).expanduser().resolve()
    out_dir = (args.out_dir or default_out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    default_experiment = f"{sample_dir.name}_3dgut" if sample_dir else f"{images_dir.name}_3dgut"
    experiment_name = args.experiment_name or default_experiment

    train_script = args.train_script if args.train_script.is_absolute() else (repo_root / args.train_script)
    train_script = train_script.resolve()
    if not train_script.exists():
        raise SystemExit(f"3dgrut train script not found: {train_script}")

    prep_info = _prepare_dataset_layout(
        work_dir=work_dir,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
        overwrite=args.overwrite,
    )

    if args.manifest_path:
        manifest_path = args.manifest_path.expanduser().resolve()
    elif sample_dir:
        manifest_path = sample_dir / "3dgrut_manifest.json"
    else:
        manifest_path = out_dir / "3dgrut_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        **prep_info,
        "sample_dir": str(sample_dir) if sample_dir else None,
        "config_name": args.config_name,
        "fused_point_cloud": str(fused_point_cloud) if fused_point_cloud else None,
        "out_dir": str(out_dir),
        "experiment_name": experiment_name,
        "export_usdz": args.export_usdz,
        "apply_normalizing_transform": args.apply_normalizing_transform,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[info] Wrote manifest: {manifest_path}")

    if args.prepare_only:
        print("[info] Preparation complete (--prepare-only).")
        return 0

    hydra_overrides = [
        f"path={work_dir}",
        f"out_dir={out_dir}",
        f"experiment_name={experiment_name}",
        f"initialization.fused_point_cloud_path={fused_point_cloud}",
        f"export_usdz.enabled={'true' if args.export_usdz else 'false'}",
        f"export_usdz.apply_normalizing_transform={'true' if args.apply_normalizing_transform else 'false'}",
    ]
    if args.max_steps is not None:
        hydra_overrides.append(f"n_iterations={int(args.max_steps)}")
    if args.downsample_factor is not None:
        hydra_overrides.append(f"dataset.downsample_factor={int(args.downsample_factor)}")
    hydra_overrides.extend(args.set)

    train_cmd: list[str] = []
    if args.conda_env:
        train_cmd.extend(["conda", "run", "-n", args.conda_env, "python"])
    else:
        train_cmd.append(sys.executable)
    train_cmd.extend([str(train_script), "--config-name", args.config_name])
    train_cmd.extend(hydra_overrides)

    _run(train_cmd, cwd=train_script.parent)

    exp_root = out_dir / experiment_name
    latest_run_dir = _latest_child_dir(exp_root)
    summary = {
        "experiment_root": str(exp_root),
        "latest_run_dir": str(latest_run_dir) if latest_run_dir else None,
    }
    if latest_run_dir:
        summary["artifacts"] = {
            "checkpoint": str(latest_run_dir / "ckpt_last.pt"),
            "usd": str(latest_run_dir / "export_last.usdz"),
            "ply": str(latest_run_dir / "export_last.ply"),
            "ingp": str(latest_run_dir / "export_last.ingp"),
            "parsed_config": str(latest_run_dir / "parsed.yaml"),
        }
    if args.result_path:
        summary_path = args.result_path.expanduser().resolve()
    elif sample_dir:
        summary_path = sample_dir / "3dgrut_result.json"
    else:
        summary_path = out_dir / "3dgrut_result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[info] Wrote run summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
