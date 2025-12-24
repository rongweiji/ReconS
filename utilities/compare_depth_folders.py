#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _collect_pairs(
    dir_a: Path,
    dir_b: Path,
    pattern: str,
    limit: int | None,
    sample: int | None,
    rng: np.random.Generator,
):
    files_a = {p.name: p for p in dir_a.glob(pattern) if p.is_file()}
    files_b = {p.name: p for p in dir_b.glob(pattern) if p.is_file()}
    common = sorted(set(files_a) & set(files_b))
    if limit is not None:
        common = common[:limit]
    if sample is not None and sample < len(common):
        idx = rng.choice(len(common), size=sample, replace=False)
        common = [common[i] for i in sorted(idx)]
    pairs = [(files_a[name], files_b[name]) for name in common]
    return pairs


def _read_depth(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"failed to read {path}")
    if img.ndim == 3:
        # Some pipelines store depth in a single channel of RGB; fall back to first channel.
        img = img[:, :, 0]
    return img.astype(np.float32)


def _downsample(values: np.ndarray, max_samples: int, rng: np.random.Generator):
    if values.size <= max_samples:
        return values
    idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Compare two depth folders with matching filenames."
    )
    parser.add_argument("folder_a", type=Path, help="First depth folder")
    parser.add_argument("folder_b", type=Path, help="Second depth folder")
    parser.add_argument(
        "--pattern", default="*.png", help="Glob pattern for depth files (default: *.png)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("utilities/depth_compare_out"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to raw depth values (e.g., 0.001 for mm->m)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional max depth for plotting (after scaling)",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Limit number of frame pairs to process",
    )
    parser.add_argument(
        "--random-frames",
        type=int,
        default=None,
        help="Randomly sample this many matching frames",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Compare only one random matching frame",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="Maximum number of pixels to keep for plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG seed for random frame sampling and downsampling",
    )
    args = parser.parse_args()

    if not args.folder_a.is_dir() or not args.folder_b.is_dir():
        raise SystemExit("Both inputs must be directories.")

    rng = np.random.default_rng(args.seed)
    pairs = _collect_pairs(
        args.folder_a,
        args.folder_b,
        args.pattern,
        args.limit_frames,
        args.random_frames,
        rng,
    )
    if not pairs:
        raise SystemExit("No matching files found.")
    if args.single:
        pairs = [pairs[rng.integers(len(pairs))]]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_a = []
    all_b = []
    diffs = []
    skipped = 0

    for path_a, path_b in pairs:
        try:
            depth_a = _read_depth(path_a) * args.scale
            depth_b = _read_depth(path_b) * args.scale
        except ValueError:
            skipped += 1
            continue

        if depth_a.shape != depth_b.shape:
            skipped += 1
            continue

        valid = (depth_a > 0) & (depth_b > 0)
        if args.max_depth is not None:
            valid &= (depth_a <= args.max_depth) & (depth_b <= args.max_depth)

        if not np.any(valid):
            continue

        a_vals = depth_a[valid].ravel()
        b_vals = depth_b[valid].ravel()
        all_a.append(a_vals)
        all_b.append(b_vals)
        diffs.append((a_vals - b_vals))

    if not all_a:
        raise SystemExit("No valid depth pixels found.")

    all_a = np.concatenate(all_a)
    all_b = np.concatenate(all_b)
    diffs = np.concatenate(diffs)

    all_a = _downsample(all_a, args.max_samples, rng)
    all_b = _downsample(all_b, args.max_samples, rng)
    diffs = _downsample(diffs, args.max_samples, rng)

    def _stats(arr: np.ndarray):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }

    stats_a = _stats(all_a)
    stats_b = _stats(all_b)
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))

    print(f"Pairs processed: {len(pairs) - skipped}/{len(pairs)}")
    print(f"Valid samples: {all_a.size}")
    print(f"A stats: {stats_a}")
    print(f"B stats: {stats_b}")
    print(f"MAE: {mae:.6f}  RMSE: {rmse:.6f}")

    # Plot 1: Histogram
    plt.figure(figsize=(9, 5))
    bins = 200
    plt.hist(all_a, bins=bins, alpha=0.55, label="A", color="#2a6f97")
    plt.hist(all_b, bins=bins, alpha=0.55, label="B", color="#f4a261")
    plt.title("Depth Distribution")
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "depth_hist.png", dpi=160)
    plt.close()

    # Plot 2: CDF
    plt.figure(figsize=(9, 5))
    for arr, label, color in [
        (all_a, "A", "#2a6f97"),
        (all_b, "B", "#f4a261"),
    ]:
        sorted_arr = np.sort(arr)
        cdf = np.linspace(0, 1, sorted_arr.size, endpoint=False)
        plt.plot(sorted_arr, cdf, label=label, color=color)
    plt.title("Depth CDF")
    plt.xlabel("Depth")
    plt.ylabel("Cumulative probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "depth_cdf.png", dpi=160)
    plt.close()

    # Plot 3: Scatter (hexbin) A vs B
    plt.figure(figsize=(6, 6))
    plt.hexbin(
        all_a,
        all_b,
        gridsize=120,
        cmap="viridis",
        mincnt=1,
    )
    max_val = max(all_a.max(), all_b.max())
    plt.plot([0, max_val], [0, max_val], color="white", linewidth=1)
    plt.xlabel("Depth A")
    plt.ylabel("Depth B")
    plt.title("A vs B (hexbin)")
    cb = plt.colorbar()
    cb.set_label("Count")
    plt.tight_layout()
    plt.savefig(args.out_dir / "depth_hexbin.png", dpi=160)
    plt.close()

    # Plot 4: Difference distribution
    plt.figure(figsize=(9, 5))
    plt.hist(diffs, bins=200, color="#6a4c93", alpha=0.8)
    plt.title("Depth Difference (A - B)")
    plt.xlabel("Depth difference")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(args.out_dir / "depth_diff_hist.png", dpi=160)
    plt.close()

    print(f"Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
