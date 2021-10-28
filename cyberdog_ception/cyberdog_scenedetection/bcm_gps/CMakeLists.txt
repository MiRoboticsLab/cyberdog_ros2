cmake_minimum_required(VERSION 3.5)
project(bcm_gps LANGUAGES C CXX)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include)

# find dependencies
find_package(ament_cmake REQUIRED)

set(EXTPRJ_NAME bcmgps_vendor)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

include(ExternalProject)

set(PKG_VER 1.0.0)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/bcmgps_vendor-${PKG_VER}.tar.xz
  URL_MD5 ba2fd10682480fa5a870ff1e784997e6
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 1200
)

include_directories(${PREFIX_DIR}/include)

set(sdk_libs bcm_gps)

install(DIRECTORY include/
  DESTINATION include/
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
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_export_include_directories(include)
ament_export_libraries(${sdk_libs})
ament_package()
