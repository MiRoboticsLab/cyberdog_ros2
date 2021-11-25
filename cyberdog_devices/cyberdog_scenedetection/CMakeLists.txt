cmake_minimum_required(VERSION 3.7)
project(cyberdog_scene_detection CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(motion_msgs REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(cyberdog_utils REQUIRED)

include(ExternalProject)
set(EXTPRJ_NAME bream)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

message(WARNING "Not support yet.")
return()

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/bream.tar.xz
  URL_MD5 bb6ffc6a8c271c0346d43ca309c9accd
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 10
)

set(dependencies
  rclcpp
  ception_msgs
  motion_msgs
  cyberdog_utils
)

include_directories(${PREFIX_DIR}/include)

add_library(Hal STATIC  IMPORTED)
set_property(TARGET Hal PROPERTY IMPORTED_LOCATION ${PREFIX_DIR}/lib/LibHal.a)
add_library(Bream STATIC  IMPORTED)
set_property(TARGET Bream PROPERTY IMPORTED_LOCATION ${PREFIX_DIR}/lib/LibBream.a)
add_library(PatchDownloader  STATIC  IMPORTED)
set_property(TARGET PatchDownloader PROPERTY IMPORTED_LOCATION ${PREFIX_DIR}/lib/LibPatchDownloader.a)
add_library(CallBack STATIC  IMPORTED)
set_property(TARGET CallBack PROPERTY IMPORTED_LOCATION ${PREFIX_DIR}/lib/LibCallBack.a)

add_dependencies(Hal Bream PachDownloader CallBack ${EXTPRJ_NAME})
set(executable_name service_scene_detection)

add_executable(${executable_name} src/scene_detection.cpp)
ament_target_dependencies(${executable_name} ${dependencies})
target_link_libraries(${executable_name})

install(TARGETS ${executable_name}
  DESTINATION lib/${PROJECT_NAME})

target_link_libraries(${executable_name}
  ${CMAKE_THREAD_LIBS_INIT}
  Hal
  Bream
  PatchDownloader
  CallBack
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()