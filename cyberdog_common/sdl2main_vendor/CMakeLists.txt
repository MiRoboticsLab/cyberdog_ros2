cmake_minimum_required(VERSION 3.8)
project(sdl2main_vendor)

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

set(EXTPRJ_NAME sdl2main)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}/src/${EXTPRJ_NAME}")
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(PKG_VER 2.0.16)

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://${GITHUB_URL}/libsdl-org/SDL/archive/refs/tags/release-${PKG_VER}.tar.gz
  URL_MD5 b52877e5e92d80a20a4be4c868aa1f18
  CMAKE_ARGS
      -DBUILD_SHARED_LIBS=ON
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${OUT_DIR}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DSDL_ATOMIC=OFF
      -DSDL_AUDIO=ON
      -DSDL_VIDEO=ON
      -DSDL_RENDER=ON
      -DSDL_EVENTS=ON
      -DSDL_JOYSTICK=ON
      -DSDL_HAPTIC=ON
      -DSDL_POWER=ON
      -DSDL_THREADS=ON
      -DSDL_TIMERS=ON
      -DSDL_FILE=ON
      -DSDL_LOADSO=ON
      -DSDL_CPUINFO=OFF
      -DSDL_FILESYSTEM=OFF
      -DSDL_DLOPEN=ON
      -DSDL_SENSOR=ON
      -DVIDEO_X11=OFF
  INSTALL_DIR ${OUT_DIR}
  TIMEOUT 300
)

set(sdl2main_libs
  SDL2
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
ament_export_libraries(${sdl2main_libs})
ament_package()
