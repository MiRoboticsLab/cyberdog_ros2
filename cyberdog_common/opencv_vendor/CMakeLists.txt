cmake_minimum_required(VERSION 3.8)
project(opencv_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PKG_VER "4.2.0")

find_package(ament_cmake REQUIRED)
find_package(OpenCV ${PKG_VER} QUIET)

option(XIAOMI_VISION "Build with Xiaomi Vision Package" OFF)

if(NOT XIAOMI_VISION)
  message(WARNING ">>> Skip building Xiaomi Vision Package")
  return()
endif()

if(OpenCV_FOUND)
  if(OpenCV_VERSION STREQUAL ${PKG_VER})
    message("OpenCV ${PKG_VER} found")
    return()
  endif()
endif()

set(GITHUB_URL "github.com")
option(BUILD_INSIDE_GFW "Build this package in PRC" OFF)

if(BUILD_INSIDE_GFW)
  message(">>> Replace github.com into hub.fastgit.org for PRC users!")
  set(GITHUB_URL "hub.fastgit.org")
endif()

set(EXTPRJ_NAME opencv)
set(EXTPRJ_DEP opencv-contrib)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}/src/${EXTPRJ_NAME}")
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(REPLACE_GIT sed -i "s/github.com/${GITHUB_URL}/g" `grep github.com -rl * .*`)
set(EXTERNAL_DOWNLOAD_LOCATION ${CMAKE_BINARY_DIR}/Download)

include(ExternalProject)

externalproject_add(${EXTPRJ_DEP}
  URL https://${GITHUB_URL}/opencv/opencv_contrib/archive/refs/tags/${PKG_VER}.tar.gz
  URL_MD5 7f8111deb2ce3ed6c87ede8b3bf82031
  SOURCE_DIR "${EXTERNAL_DOWNLOAD_LOCATION}/opencv-contrib"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://${GITHUB_URL}/opencv/opencv/archive/refs/tags/${PKG_VER}.tar.gz
  URL_MD5 e8cb208ce2723481408b604b480183b6
  SOURCE_DIR "${EXTERNAL_DOWNLOAD_LOCATION}/opencv"
  PATCH_COMMAND eval ${REPLACE_GIT}
  CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DBUILD_DOCS:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_PACKAGE:BOOL=OFF
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DBUILD_TESTS:BOOL=OFF
      -DOPENCV_EXTRA_MODULES_PATH=${EXTERNAL_DOWNLOAD_LOCATION}/opencv-contrib/modules
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  TIMEOUT 1200
)
add_dependencies(${EXTPRJ_NAME} ${EXTPRJ_DEP})

ament_package()
