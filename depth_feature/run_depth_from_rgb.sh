#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
BUILD_DIR="${SRC_DIR}/build"
BIN="${BUILD_DIR}/depth_from_rgb"
VENDOR_LIB="${SRC_DIR}/vendor/depth_anything_v3/lib"

ROS_PREFIX_DEFAULT="/opt/ros/jazzy"
ROS_PREFIX="${ROS_PREFIX:-${ROS_PREFIX_DEFAULT}}"

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  mkdir -p "${BUILD_DIR}"
  CMAKE_PREFIX_PATH_ARG=""
  if [[ -d "${ROS_PREFIX}" ]]; then
    CMAKE_PREFIX_PATH_ARG="-DCMAKE_PREFIX_PATH=${ROS_PREFIX}"
  fi
  cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" ${CMAKE_PREFIX_PATH_ARG}
fi

cmake --build "${BUILD_DIR}" -j"$(nproc)"

LD_LIBRARY_PATH="${VENDOR_LIB}:${LD_LIBRARY_PATH:-}"
if [[ -d "${ROS_PREFIX}/lib" ]]; then
  LD_LIBRARY_PATH="${ROS_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi
if [[ -d "/usr/lib/wsl/lib" ]]; then
  LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi

export LD_LIBRARY_PATH

exec "${BIN}" "$@"
