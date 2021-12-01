cmake_minimum_required(VERSION 3.8)
project(mpg123_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

set(EXTPRJ_NAME mpg123)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}/src/${EXTPRJ_NAME}")
set(OUT_DIR "${CMAKE_BINARY_DIR}/install")

include(ExternalProject)

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://www.mpg123.de/download/mpg123-1.29.2.tar.bz2
  URL_MD5 05137a60b40d66bc185b1e106815aec7
  CONFIGURE_COMMAND eval ${PREFIX_DIR}/configure --prefix=${OUT_DIR}
  BUILD_COMMAND "make"
  INSTALL_DIR ${OUT_DIR}
  TIMEOUT 1200
)

set(mpg123_libs
  mpg123
  out123
  syn123
)

include_directories(${OUT_DIR}/include)

install(DIRECTORY ${OUT_DIR}/bin/
  DESTINATION bin/
  PATTERN "*"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
)

install(DIRECTORY ${OUT_DIR}/include/
  DESTINATION include/${EXTPRJ_NAME}
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
ament_export_libraries(${mpg123_libs})
ament_package()
