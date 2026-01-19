cmake_minimum_required(VERSION 3.16)
project(orbslam3_rgbd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Pangolin REQUIRED)

set(RERUN_CPP_DIR "${CMAKE_SOURCE_DIR}/../third_party/rerun_cpp")
set(RERUN_CPP_LIB "${RERUN_CPP_DIR}/librerun_c__linux_x64.a")
file(GLOB_RECURSE RERUN_CPP_SOURCES "${RERUN_CPP_DIR}/include/rerun/*.cpp")

add_library(rerun_cpp STATIC ${RERUN_CPP_SOURCES})
target_include_directories(rerun_cpp PUBLIC ${RERUN_CPP_DIR}/include)
target_link_libraries(rerun_cpp PUBLIC ${RERUN_CPP_LIB} pthread)

set(ORB_SLAM3_DIR "${CMAKE_SOURCE_DIR}/../third_party/ORB_SLAM3")
set(ORB_SLAM3_LIB "${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so")

add_executable(orbslam3_rgbd
  main.cpp
  calibration.cpp
)

target_include_directories(orbslam3_rgbd PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2
  ${ORB_SLAM3_DIR}/Thirdparty/g2o
  ${RERUN_CPP_DIR}/include
)

target_compile_definitions(orbslam3_rgbd PRIVATE HAVE_EPOXY HAVE_EIGEN)

target_link_libraries(orbslam3_rgbd
  ${OpenCV_LIBRARIES}
  ${ORB_SLAM3_LIB}
  ${Pangolin_LIBRARIES}
  rerun_cpp
)

set_target_properties(orbslam3_rgbd PROPERTIES
  BUILD_RPATH "${ORB_SLAM3_DIR}/lib;${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib;${ORB_SLAM3_DIR}/Thirdparty/g2o/lib"
)
