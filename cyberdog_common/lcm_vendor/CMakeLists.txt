cmake_minimum_required(VERSION 3.8)
project(lcm_vendor)

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

set(EXTPRJ_NAME mpg123)
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(PKG_VER "v1.4.0")

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://${GITHUB_URL}/lcm-proj/lcm/archive/refs/tags/${PKG_VER}.tar.gz
  URL_MD5 f26ac001d8b4fa321edcf87c3b5657cc
  CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${OUT_DIR}
      -DLCM_ENABLE_JAVA=OFF
      -DLCM_ENABLE_PYTHON=OFF
      -DLCM_ENABLE_LUA=OFF
      -DLCM_ENABLE_TESTS=OFF
  INSTALL_DIR ${OUT_DIR}
  TIMEOUT 1200
)

set(lcm_libs
  lcm
)

include_directories(${OUT_DIR}/include)

install(DIRECTORY ${OUT_DIR}/bin/
  DESTINATION bin/
  PATTERN "*"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
)

install(DIRECTORY ${OUT_DIR}/include/
  DESTINATION include/
)

install(DIRECTORY ${OUT_DIR}/lib/
  DESTINATION lib/
  REGEX ".?so.?"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
)

install(DIRECTORY ${OUT_DIR}/share/
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${lcm_libs})
ament_package()
