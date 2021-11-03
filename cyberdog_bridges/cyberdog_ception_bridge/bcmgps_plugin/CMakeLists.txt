cmake_minimum_required(VERSION 3.5)
project(bcmgps_plugin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(ception_base REQUIRED)

set(EXTPRJ_NAME bcmgps_plugin_vendor)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

include(ExternalProject)

set(PKG_VER 1.0.0)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/bcmgps_plugin_vendor-${PKG_VER}.tar.xz
  URL_MD5 2759222fb323570e4b07997d3a4f83f7
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 1200
)

include_directories(${PREFIX_DIR}/include)

pluginlib_export_plugin_description_file(ception_base plugin.xml)

install(DIRECTORY ${PREFIX_DIR}/include/
  DESTINATION include/
)

install(DIRECTORY ${PREFIX_DIR}/lib/
  DESTINATION lib/
)

install(
  DIRECTORY logs/
  DESTINATION share/${PROJECT_NAME}/logs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
