cmake_minimum_required(VERSION 3.8)
project(sdl2_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(mpg123_vendor REQUIRED)

set(GITHUB_URL "github.com")
option(BUILD_INSIDE_GFW "Build this package in PRC" OFF)

if(BUILD_INSIDE_GFW)
  message(">>> Replace github.com into hub.fastgit.org for PRC users!")
  set(GITHUB_URL "hub.fastgit.org")
endif()

set(MAIN_NAME sdl2main)
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(PKG_VER 2.0.16)

include(ExternalProject)

externalproject_add(
  ${MAIN_NAME}
  PREFIX ${MAIN_NAME}
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
      -DSDL_VIDEO=OFF
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
  TIMEOUT 1200
)

set(MIXER_NAME sdl2mixer)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${MIXER_NAME}/src/${MIXER_NAME}")
set(PKG_VER 2.0.4)
set(mpg123_INCLUDEDIR ${mpg123_vendor_DIR}/../../../include)
set(mpg123_LIBDIR ${mpg123_vendor_DIR}/../../../lib)

externalproject_add(
  ${MIXER_NAME}
  PREFIX ${MIXER_NAME}
  URL https://${GITHUB_URL}/libsdl-org/SDL_mixer/archive/refs/tags/release-${PKG_VER}.tar.gz
  URL_MD5 7e9f54fd4379426eea14a0285ea61e71
  CONFIGURE_COMMAND eval ${PREFIX_DIR}/configure --prefix=${OUT_DIR}
    --with-sdl-prefix=${OUT_DIR} --enable-music-mp3-mpg123=yes
    CFLAGS="-I${mpg123_INCLUDEDIR}" LDFLAGS="-L${mpg123_LIBDIR}"
  BUILD_COMMAND "make"
  INSTALL_DIR ${OUT_DIR}
  TIMEOUT 1200
)
add_dependencies(${MIXER_NAME} ${MAIN_NAME})

set(sdl2_libs
  SDL2
  SDL2_mixer
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
ament_export_libraries(${sdl2_libs})
ament_package()
