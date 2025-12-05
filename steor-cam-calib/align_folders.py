#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Set, Tuple

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_files(folder: str) -> List[str]:
    try:
        entries = os.listdir(folder)
    except OSError as e:
        raise RuntimeError(f"Failed to list folder '{folder}': {e}")
    files = []
    for name in entries:
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            files.append(name)
    return files


def filter_supported(files: List[str]) -> List[str]:
    return [f for f in files if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]


def align_folders(src_folder: str, dst_folder: str, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Make dst_folder contain only files that exist in src_folder.
    Removes files in dst_folder that are not present in src_folder (by filename).

    Returns: (initial_dst_count, removed_count, final_dst_count)
    """
    src_files = filter_supported(list_files(src_folder))
    dst_files = filter_supported(list_files(dst_folder))

    src_set: Set[str] = set(src_files)
    dst_set: Set[str] = set(dst_files)

    to_remove = sorted(dst_set - src_set)

    removed = 0
    for name in to_remove:
        path = os.path.join(dst_folder, name)
        if dry_run:
            print(f"[DRY-RUN] Would remove: {path}")
        else:
            try:
                os.remove(path)
                removed += 1
                print(f"Removed: {path}")
            except OSError as e:
                print(f"Failed to remove '{path}': {e}", file=sys.stderr)

    initial = len(dst_files)
    final = initial - removed
    return initial, removed, final


def main():
    parser = argparse.ArgumentParser(description="Align folder2 to match folder1 by removing unmatched files in folder2")
    parser.add_argument("folder1", help="Reference folder (kept contents)")
    parser.add_argument("folder2", help="Folder to clean (will remove files not present in folder1)")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be removed, without deleting")
    args = parser.parse_args()

    f1 = os.path.abspath(args.folder1)
    f2 = os.path.abspath(args.folder2)

    if not os.path.isdir(f1):
        print(f"Error: folder1 '{f1}' is not a directory", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(f2):
        print(f"Error: folder2 '{f2}' is not a directory", file=sys.stderr)
        sys.exit(2)

    # Show initial counts (supported types only)
    f1_files = filter_supported(list_files(f1))
    f2_files = filter_supported(list_files(f2))
    print(f"Folder1: {f1}")
    print(f" - Supported files: {len(f1_files)}")
    print(f"Folder2: {f2}")
    print(f" - Supported files: {len(f2_files)}")

    initial, removed, final = align_folders(f1, f2, dry_run=args.dry_run)

    print("Summary:")
    print(f" - Folder1 (reference) files: {len(f1_files)}")
    print(f" - Folder2 initial files: {initial}")
    print(f" - Removed from folder2: {removed}")
    print(f" - Folder2 final files: {final}")


if __name__ == "__main__":
    main()
