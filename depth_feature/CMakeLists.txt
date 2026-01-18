cmake_minimum_required(VERSION 3.16)
project(depth_feature_cpp LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr /usr/include /usr/include/x86_64-linux-gnu
  PATH_SUFFIXES include)

set(DEPTH_ANYTHING_DIR_DEFAULT "${CMAKE_CURRENT_LIST_DIR}/vendor/depth_anything_v3")
set(DEPTH_ANYTHING_DIR "${DEPTH_ANYTHING_DIR_DEFAULT}" CACHE PATH "Depth Anything vendor directory")
set(DEPTH_ANYTHING_INCLUDE_DIR "${DEPTH_ANYTHING_DIR}/include")
set(DEPTH_ANYTHING_LIB "${DEPTH_ANYTHING_DIR}/lib/libtensorrt_depth_anything.so")

add_library(depth_feature_common STATIC
  src/camera_info.cpp
  src/depth_estimator.cpp
)

target_compile_features(depth_feature_common PUBLIC cxx_std_17)

target_include_directories(depth_feature_common PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${DEPTH_ANYTHING_INCLUDE_DIR}
)

target_compile_definitions(depth_feature_common PUBLIC DEPTH_ANYTHING_STANDALONE)

target_link_libraries(depth_feature_common PUBLIC
  ${OpenCV_LIBRARIES}
  ${DEPTH_ANYTHING_LIB}
  ${CUDA_LIBRARIES}
)

ament_target_dependencies(depth_feature_common PUBLIC
  sensor_msgs
  std_msgs
  builtin_interfaces
  rosidl_runtime_cpp
)

add_executable(depth_from_rgb
  src/depth_from_rgb.cpp
)

target_link_libraries(depth_from_rgb PRIVATE depth_feature_common)
