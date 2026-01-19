#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN="${BUILD_DIR}/orbslam3_rgbd"
ORB_DIR="${ROOT_DIR}/third_party/ORB_SLAM3"

if [[ ! -d "${ORB_DIR}" ]]; then
  echo "Missing ORB_SLAM3 in ${ORB_DIR}"
  echo "Add it as a submodule under third_party/ORB_SLAM3 and rerun."
  exit 1
fi

ORB_CMAKE="${ORB_DIR}/CMakeLists.txt"
if [[ -f "${ORB_CMAKE}" ]]; then
  if ! grep -q "std=c\\+\\+14" "${ORB_CMAKE}"; then
    sed -i 's/std=c++11/std=c++14/g' "${ORB_CMAKE}"
    echo "Patched ORB-SLAM3 to use C++14."
  fi
fi

ORB_CACHE="${ORB_DIR}/build/CMakeCache.txt"
if [[ -f "${ORB_CACHE}" ]] && grep -q "std=c\\+\\+11" "${ORB_CACHE}"; then
  rm -rf "${ORB_DIR}/build"
fi

if [[ ! -f "${ORB_DIR}/lib/libORB_SLAM3.so" ]]; then
  (cd "${ORB_DIR}" && ./build.sh)
fi

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]] || [[ "${SCRIPT_DIR}/CMakeLists.txt" -nt "${BUILD_DIR}/CMakeCache.txt" ]]; then
  mkdir -p "${BUILD_DIR}"
  cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}"
fi
cmake --build "${BUILD_DIR}" -j"$(nproc)"

export LD_LIBRARY_PATH="${ORB_DIR}/lib:${ORB_DIR}/Thirdparty/DBoW2/lib:${ORB_DIR}/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"

exec "${BIN}" "$@"
