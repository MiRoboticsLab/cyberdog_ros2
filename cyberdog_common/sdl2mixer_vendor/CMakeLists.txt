cmake_minimum_required(VERSION 3.8)
project(sdl2mixer_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(sdl2main_vendor REQUIRED)
find_package(mpg123_vendor REQUIRED)

set(GITHUB_URL "github.com")
option(BUILD_INSIDE_GFW "Build this package in PRC" OFF)

if(BUILD_INSIDE_GFW)
  message(">>> Replace github.com into hub.fastgit.org for PRC users!")
  set(GITHUB_URL "hub.fastgit.org")
endif()

set(EXTPRJ_NAME sdl2mixer)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}/src/${EXTPRJ_NAME}")
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")
set(PKG_VER 2.0.4)

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://${GITHUB_URL}/libsdl-org/SDL_mixer/archive/refs/tags/release-${PKG_VER}.tar.gz
  URL_MD5 7e9f54fd4379426eea14a0285ea61e71
  CONFIGURE_COMMAND eval ${PREFIX_DIR}/configure --prefix=${OUT_DIR} --with-sdl-prefix=${CMAKE_INSTALL_PREFIX} --enable-music-mp3-mpg123=yes
  BUILD_COMMAND "make"
  INSTALL_DIR ${OUT_DIR}
  TIMEOUT 300
)

set(sdl2mixer_libs
  SDL2_mixer
)

include_directories(${OUT_DIR}/include)

install(DIRECTORY ${OUT_DIR}/include/
  DESTINATION include/
)

install(DIRECTORY ${OUT_DIR}/lib/
  DESTINATION lib/
  REGEX ".?so.?"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
)

ament_export_include_directories(include)
ament_export_libraries(${sdl2mixer_libs})
ament_package()
