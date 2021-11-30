cmake_minimum_required(VERSION 3.5)
project(toml11_vendor)

find_package(ament_cmake REQUIRED)

set(GITHUB_URL "github.com")
option(BUILD_INSIDE_GFW "Build this package in PRC" OFF)

if(BUILD_INSIDE_GFW)
  message(">>> Replace github.com into hub.fastgit.org for PRC users!")
  set(GITHUB_URL "hub.fastgit.org")
endif()

set(EXTPRJ_NAME toml11)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")
set(OUT_DIR "${CMAKE_BINARY_DIR}/${EXTPRJ_NAME}/include")
set(PKG_VER 3.7.0)

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://${GITHUB_URL}/ToruNiina/toml11/archive/refs/tags/v${PKG_VER}.tar.gz
  URL_MD5 5d4573d44ffc28b6387d46fc972cf286
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 1200
)

include_directories(${OUT_DIR}/)

install(DIRECTORY ${OUT_DIR}/
  DESTINATION include/${EXTPRJ_NAME}
)

ament_export_include_directories(include)
ament_package()