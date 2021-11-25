cmake_minimum_required(VERSION 3.5)

project(xiaoai_sdk_vendor)

option(XIAOMI_XIAOAI "Build with Xiaomi Xiaoai Package" OFF)

if(NOT XIAOMI_XIAOAI)
  message(WARNING ">>> Skip building Xiaomi Xiaoai Package")
  return()
endif()

find_package(ament_cmake REQUIRED)

include(ExternalProject)
set(EXTPRJ_NAME xiaoai_sdk)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

ExternalProject_Add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/xiaoai_sdk_1.0.0.8.2.tar.xz
  URL_MD5 1566ab64e17e3f35d453473647d58ca1
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 60
)

include_directories(${PREFIX_DIR}/include)

set(sdk_libs
  vpm
  paddy
  minerva_engine
  aivs_sdk
  audio_config
)

install(DIRECTORY ${PREFIX_DIR}/ai_conf/
  DESTINATION ai_conf/
)

install(DIRECTORY ${PREFIX_DIR}/include/
  DESTINATION include/
)

install(DIRECTORY ${PREFIX_DIR}/lib/
  DESTINATION lib/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${sdk_libs})
ament_package()
