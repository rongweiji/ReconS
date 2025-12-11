#!/usr/bin/env python3
# CLI tool to visualize frame timestamps and FPS stability.
#
# Input: newline-delimited JSON with keys {"filename": str, "timestamp_ns": int}
# Example line:
# {"filename": "0000001.jpg", "timestamp_ns": 1764888062711310848}
#
# Features:
# - Parses timestamps (ns) and converts to seconds relative to first frame
# - Computes per-frame intervals and instantaneous FPS = 1 / delta_t
# - Plots FPS vs time (seconds) with moving average option
# - Shows markers/labels for min, max, and average timestamp (absolute in ns)
# - Prints summary stats to console
#
# Usage:
# python utilities/visualize_timestamps.py --file <path/to/frames_time.json> [--smooth 5]
#
# On Windows PowerShell:
# python "g:\GithubProject\ReconS\utilities\visualize_timestamps.py" --file "g:\GithubProject\ReconS\steor-cam-calib\cali_data1\workspace2\frames_time.json"

import argparse
import json
import math
import os
import statistics
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_timestamps_ns(path: str) -> Tuple[List[str], List[int]]:
    filenames: List[str] = []
    ts_ns: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
            if "timestamp_ns" not in entry:
                raise ValueError(f"Missing 'timestamp_ns' on line {line_no}")
            filenames.append(entry.get("filename", str(line_no)))
            ts_ns.append(int(entry["timestamp_ns"]))
    if not ts_ns:
        raise ValueError("No timestamps found in file")
    return filenames, ts_ns


def compute_intervals_and_fps(ts_ns: List[int]) -> Tuple[List[float], List[float], List[float]]:
    # Convert to seconds relative to first frame for plotting
    base = ts_ns[0]
    times_s = [(t - base) / 1e9 for t in ts_ns]
    # Instantaneous FPS from per-frame deltas; first frame has no delta, so set NaN
    fps: List[float] = [math.nan]
    intervals_s: List[float] = [math.nan]
    for i in range(1, len(ts_ns)):
        dt_s = (ts_ns[i] - ts_ns[i - 1]) / 1e9
        intervals_s.append(dt_s if dt_s > 0 else math.nan)
        if dt_s <= 0:
            fps.append(math.nan)
        else:
            fps.append(1.0 / dt_s)
    return times_s, fps, intervals_s


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    out: List[float] = []
    acc = 0.0
    count = 0
    buf: List[float] = []
    for v in values:
        if math.isnan(v):
            buf.append(v)
            out.append(math.nan)
            continue
        buf.append(v)
        acc += v
        count += 1
        if len(buf) > window:
            old = buf.pop(0)
            if not math.isnan(old):
                acc -= old
                count -= 1
        out.append(acc / count if count > 0 else math.nan)
    return out


def format_ns(ns: int) -> str:
    # Show human friendly ns, ms, s.
    if ns >= 1_000_000_000:
        return f"{ns/1e9:.3f}s ({ns} ns)"
    elif ns >= 1_000_000:
        return f"{ns/1e6:.3f}ms ({ns} ns)"
    elif ns >= 1_000:
        return f"{ns/1e3:.3f}us ({ns} ns)"
    else:
        return f"{ns} ns"


def percentile(values: List[float], pct: float) -> float:
    """Return percentile (0-100) using linear interpolation of the sorted list."""
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    vals = sorted(values)
    k = (len(vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[int(f)] * (c - k)
    d1 = vals[int(c)] * (k - f)
    return d0 + d1


def ecdf(values: List[float]) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    vals = sorted(values)
    n = len(vals)
    xs = vals
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="Visualize timestamps and FPS stability")
    parser.add_argument("--file", "-f", required=True, help="Path to newline-delimited JSON timestamps file")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window for FPS (frames). Default 1 (no smoothing)")
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    filenames, ts_ns = read_timestamps_ns(file_path)

    # Stats
    ts_min = min(ts_ns)
    ts_max = max(ts_ns)
    ts_avg = sum(ts_ns) // len(ts_ns)

    # Compute times, fps, and intervals
    times_s, fps, intervals_s = compute_intervals_and_fps(ts_ns)
    fps_smoothed = moving_average(fps, args.smooth)
    intervals_ms = [v * 1e3 for v in intervals_s if not math.isnan(v)]

    # Console summary
    duration_s = (ts_max - ts_min) / 1e9
    valid_fps = [v for v in fps if not math.isnan(v)]
    avg_fps_inst = sum(valid_fps) / len(valid_fps) if valid_fps else float("nan")
    median_interval_ms = statistics.median(intervals_ms) if intervals_ms else float("nan")
    mean_interval_ms = statistics.mean(intervals_ms) if intervals_ms else float("nan")
    max_interval_ms = max(intervals_ms) if intervals_ms else float("nan")
    p90_ms = percentile(intervals_ms, 90.0)
    p95_ms = percentile(intervals_ms, 95.0)
    p99_ms = percentile(intervals_ms, 99.0)
    stall_threshold_ms = p95_ms if not math.isnan(p95_ms) else median_interval_ms * 1.5
    stall_intervals = [v for v in intervals_ms if not math.isnan(stall_threshold_ms) and v > stall_threshold_ms]
    ecdf_x, ecdf_y = ecdf(intervals_ms)

    print("File:", file_path)
    print(f"Frames: {len(ts_ns)}")
    print(f"Duration: {duration_s:.3f} s")
    print(f"Timestamp min: {ts_min} ({format_ns(ts_min)})")
    print(f"Timestamp max: {ts_max} ({format_ns(ts_max)})")
    print(f"Timestamp avg: {ts_avg} ({format_ns(ts_avg)})")
    print(f"Instantaneous FPS avg: {avg_fps_inst:.3f}")
    print(
        "Interval mean: {:.3f} ms, median: {:.3f} ms, max: {:.3f} ms, p90: {:.3f} ms, p95: {:.3f} ms, p99: {:.3f} ms".format(
            mean_interval_ms,
            median_interval_ms,
            max_interval_ms,
            p90_ms,
            p95_ms,
            p99_ms,
        )
    )
    print(f"Stalls > {stall_threshold_ms:.3f} ms: {len(stall_intervals)}")

    # Plot FPS vs time
    plt.figure(figsize=(12, 6))
    plt.plot(times_s, fps, label="FPS (instant)", alpha=0.4, color="#1f77b4")
    if args.smooth and args.smooth > 1:
        plt.plot(times_s, fps_smoothed, label=f"FPS MA({args.smooth})", color="#ff7f0e")

    plt.xlabel("Time since start (s)")
    plt.ylabel("Frames per second (FPS)")
    title = args.title or f"FPS over Time ({os.path.basename(file_path)})"
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()

    # Annotate min/max/avg timestamp positions on the time axis
    base = ts_ns[0]
    t_min = (ts_min - base) / 1e9
    t_max = (ts_max - base) / 1e9
    t_avg = (ts_avg - base) / 1e9

    # Draw vertical lines with labels
    for t, label, color in [
        (t_min, f"min ts\n{format_ns(ts_min)}", "#2ca02c"),
        (t_avg, f"avg ts\n{format_ns(ts_avg)}", "#9467bd"),
        (t_max, f"max ts\n{format_ns(ts_max)}", "#d62728"),
    ]:
        plt.axvline(x=t, color=color, linestyle="--", alpha=0.8)
        plt.text(t, plt.ylim()[1]*0.95, label, rotation=90, va="top", ha="right", color=color, fontsize=9)

    plt.tight_layout()
    plt.show()

    # Plot interval box/strip to highlight outliers and stalls (log X to show long tail)
    plt.figure(figsize=(12, 6))
    if intervals_ms:
        # Boxplot summarizes median/IQR; strip shows individual intervals; red = stalls
        plt.boxplot(intervals_ms, vert=False, widths=0.3, patch_artist=True, boxprops=dict(facecolor="#c6dbef"))
        jitter_y = [0.75 + (0.1 * ((i % 5) - 2) / 5) for i in range(len(intervals_ms))]
        colors = ["#d62728" if (not math.isnan(stall_threshold_ms) and v > stall_threshold_ms) else "#1f77b4" for v in intervals_ms]
        plt.scatter(intervals_ms, jitter_y, s=18, alpha=0.6, color=colors, label="intervals")
        for x, label, color in [
            (mean_interval_ms, "mean", "#ff7f0e"),
            (median_interval_ms, "median", "#2ca02c"),
            (p95_ms, "p95", "#8c564b"),
            (stall_threshold_ms, "stall threshold", "#d62728"),
        ]:
            plt.axvline(x=x, color=color, linestyle="--", linewidth=1.2, label=f"{label}: {x:.3f} ms")
        if not math.isnan(stall_threshold_ms):
            plt.axvspan(stall_threshold_ms, max_interval_ms, color="#d62728", alpha=0.08, label="stall region")

    plt.xscale("log")
    plt.xlabel("Frame interval (ms) [log scale]")
    plt.yticks([])
    plt.title(args.title or f"Interval Box/Strip ({os.path.basename(file_path)})")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
