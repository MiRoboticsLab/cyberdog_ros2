cmake_minimum_required(VERSION 3.5)

project(lcm_translate_msgs)

if (NOT MOTION)
  find_package(ament_cmake REQUIRED)
endif()

find_package(lcm_vendor REQUIRED)

set(LCMTYPE_VERSION master)
set(EXTPRJ_NAME cyberdog_lcm)
set(PREFIX_DIR "${CMAKE_BINARY_DIR}/lcm_type")
set(FILE_DIR "${PREFIX_DIR}/src/${EXTPRJ_NAME}/*.lcm")
set(OUT_DIR "${CMAKE_BINARY_DIR}/header/${PROJECT_NAME}")
set(GEN_CMD ${CMAKE_CURRENT_SOURCE_DIR}/gen_hpp.sh ${OUT_DIR} ${FILE_DIR})

include(ExternalProject)

ExternalProject_Add(
  ${EXTPRJ_NAME}
  PREFIX ${PREFIX_DIR}
  URL ${CMAKE_CURRENT_SOURCE_DIR}/lcm_type
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  COMMAND "${GEN_CMD}"
  TIMEOUT 10
)

if (NOT MOTION)
  install(DIRECTORY ${OUT_DIR}/
  DESTINATION include/${PROJECT_NAME}
  )

  ament_export_include_directories(include)
  ament_package()
endif()
