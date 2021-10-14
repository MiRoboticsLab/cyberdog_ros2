cmake_minimum_required(VERSION 3.8)
project(behaviortreecppv3_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

set(GITHUB_URL "github.com")
option(BUILD_INSIDE_GFW "Build this package in PRC" OFF)

if(BUILD_INSIDE_GFW)
  message(">>> Replace github.com into hub.fastgit.org for PRC users!")
  set(GITHUB_URL "hub.fastgit.org")
endif()

set(EXTPRJ_NAME behaviortreecppv3)
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(GIT_URL "https://${GITHUB_URL}/BehaviorTree/BehaviorTree.CPP.git")
set(PKG_VER "3.5.6")

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  GIT_REPOSITORY ${GIT_URL}
  GIT_TAG ${PKG_VER}
  CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DBUILD_EXAMPLES=OFF
      -DBUILD_UNIT_TESTS=OFF
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  TIMEOUT 1200
)

ament_package()
