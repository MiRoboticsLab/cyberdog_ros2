cmake_minimum_required(VERSION 3.5)
project(toml11_vendor)

find_package(ament_cmake REQUIRED)

set(EXTPRJ_NAME toml11)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")
set(OUT_DIR "${CMAKE_BINARY_DIR}/${EXTPRJ_NAME}/include")

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/toml11-3.7.0.tar.gz
  URL_MD5 5d4573d44ffc28b6387d46fc972cf286
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 10
)

install(DIRECTORY ${OUT_DIR}/
  DESTINATION include/${EXTPRJ_NAME}
)

ament_export_include_directories(include)
ament_package()