cmake_minimum_required(VERSION 3.5)
project(cyberdog_camera)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PKG_VER "4.2.0")
option(XIAOMI_VISION "Build with Xiaomi Vision Package" OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(live_stream REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interaction_msgs REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(OpenCV ${PKG_VER} QUIET)
find_package(CUDA QUIET)
find_package(ARGUS)

if(XIAOMI_VISION)
  find_package(cyberdog_vision QUIET)
endif()

include(ExternalProject)
set(EXTPRJ_NAME nv_argus)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/nv_argus.tar.xz
  URL_MD5 973b7f0ab4464e84f29af363ed80bb2a
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 10
)

if(NOT CUDA_FOUND)
  message(WARNING "CUDA NOT found.")
  return()
endif()

if(NOT ARGUS_FOUND)
  message(WARNING "ARGUS NOT found.")
  return()
endif()

set(ARGUS_INC_DIR ${MYSYSROOT}/usr/src/jetson_multimedia_api/argus/samples/utils ${MYSYSROOT}/usr/src/jetson_multimedia_api/include/libjpeg-8b)
set(ARGUS_SRC_DIR ${MYSYSROOT}/usr/src/jetson_multimedia_api/samples/common/classes)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ARGUS_INCLUDE_DIR} ${ARGUS_INC_DIR})

set(dependencies
  rclcpp
  rclcpp_action
  std_msgs
  sensor_msgs
  interaction_msgs
  ception_msgs
  cyberdog_utils
  OpenCV
  live_stream
)

add_library(argussampleutils STATIC  IMPORTED)
set_property(TARGET argussampleutils PROPERTY IMPORTED_LOCATION ${PREFIX_DIR}/lib/libargussampleutils.a)
add_dependencies(argussampleutils ${EXTPRJ_NAME})

link_directories(${MYSYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)

add_compile_options(-Wno-pedantic -Wno-deprecated-declarations -Wno-unused-parameter)

cuda_add_library(color_convert
  src/camera_base/cuda/CUDAConvert.cu
  src/camera_base/cuda/CUDACrop.cu)

set(ARGUS_SRCS
  ${ARGUS_SRC_DIR}/NvJpegEncoder.cpp
  ${ARGUS_SRC_DIR}/NvElement.cpp
  ${ARGUS_SRC_DIR}/NvElementProfiler.cpp
  ${ARGUS_SRC_DIR}/NvVideoEncoder.cpp
  ${ARGUS_SRC_DIR}/NvBuffer.cpp
  ${ARGUS_SRC_DIR}/NvV4l2Element.cpp
  ${ARGUS_SRC_DIR}/NvV4l2ElementPlane.cpp
  ${ARGUS_SRC_DIR}/NvLogging.cpp
)

aux_source_directory(src SRCS)

add_executable(maincamera
  src/camera_base/camera_dispatcher.cpp
  src/camera_base/color_convert.cpp
  src/camera_base/jpeg_encoder.cpp
  src/camera_base/stream_consumer.cpp
  src/camera_base/video_encoder.cpp
  src/camera_base/video_muxer.cpp
  src/camera_algo/algo_dispatcher.cpp
  src/camera_algo/algorithm_base.cpp
  src/camera_algo/body_detect.cpp
  src/camera_algo/body_detector.cpp
  src/camera_algo/face_detect.cpp
  src/camera_algo/face_detector.cpp
  src/camera_algo/feature_detector.cpp
  src/camera_utils/utils.cpp
  src/camera_service/algo_stream_consumer.cpp
  src/camera_service/argus_camera_context.cpp
  src/camera_service/camera_manager.cpp
  src/camera_service/face_manager.cpp
  src/camera_service/face_manager_node.cpp
  src/camera_service/h264_stream_consumer.cpp
  src/camera_service/main_camera_node.cpp
  src/camera_service/ncs_client.cpp
  src/camera_service/rgb_stream_consumer.cpp
  src/camera_service/video_stream_consumer.cpp
  ${ARGUS_SRCS}
  src/main_camera/main.cpp
)

add_executable(stereo_camera
  src/camera_base/camera_dispatcher.cpp
  src/camera_base/stream_consumer.cpp
  src/stereo_camera/mono_stream_consumer.cpp
  src/stereo_camera/stereo_camera.cpp
  ${ARGUS_SRCS}
)

add_executable(showimage src/camera_test/showimage.cpp)

if(cyberdog_vision_FOUND AND XIAOMI_VISION)
  add_definitions(-DHAVE_VISION_LIB)
  set(dependencies
    ${dependencies}
    cyberdog_vision
  )
  ament_target_dependencies(maincamera cyberdog_vision)
else()
    message(WARNING ">>> Build camera without Xiaomi Vision.")
endif()

target_link_libraries(maincamera rt cuda EGL nvargus nvjpeg v4l2 avformat avcodec avutil argussampleutils pthread nvbuf_utils color_convert nvosd)
target_link_libraries(stereo_camera nvargus cuda EGL argussampleutils pthread nvbuf_utils nvjpeg v4l2)

ament_target_dependencies(maincamera ${dependencies})
ament_target_dependencies(showimage ${dependencies})
ament_target_dependencies(stereo_camera rclcpp std_msgs sensor_msgs)

install(TARGETS
  maincamera
  showimage
  stereo_camera
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_package()
