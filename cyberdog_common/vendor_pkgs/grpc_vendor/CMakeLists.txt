cmake_minimum_required(VERSION 3.8)
project(grpc_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Protobuf QUIET)
find_package(gRPC CONFIG QUIET)

if(Protobuf_FOUND AND gRPC_FOUND)
  message("protobuf & gRPC found, skip building from source")
  return()
endif()

set(GITHUB_URL "github.com")
option(BUILD_INSIDE_GFW "Build this package in PRC" OFF)

if(BUILD_INSIDE_GFW)
  message(">>> Replace github.com into hub.fastgit.org for PRC users!")
  set(GITHUB_URL "hub.fastgit.org")
endif()

set(EXTPRJ_NAME grpc)
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(GIT_URL "https://${GITHUB_URL}/grpc/grpc.git")
set(PKG_VER "v1.36.3")
set(REPLACE_GIT sed -i "s/github.com/${GITHUB_URL}/g" `grep github.com -rl * .*`)

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  GIT_REPOSITORY ${GIT_URL}
  GIT_TAG ${PKG_VER}
  PATCH_COMMAND eval ${REPLACE_GIT}
  CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  TIMEOUT 1200
)

ament_package()
