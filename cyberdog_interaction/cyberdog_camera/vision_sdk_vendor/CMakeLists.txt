cmake_minimum_required(VERSION 3.5)

project(vision_sdk_vendor)

find_package(ament_cmake REQUIRED)

include(ExternalProject)
set(EXTPRJ_NAME vision_sdk)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/vision_sdk_1.0.0.tar.xz
  URL_MD5 26f681fed0616012231b05db548c5b99
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 120
)

set(prebuilt_libs
  ContentMotionAPI
  ReIDTools
  XMFaceAPI
  resizeconvertion
)

include_directories(${PREFIX_DIR}/include)

set(MODEL_INSTALL_PATH lib/${PROJECT_NAME}/model)
set(AI_MODEL_PATH "\"${CMAKE_INSTALL_PREFIX}/${MODEL_INSTALL_PATH}\"")
configure_file(config/model_path.h.in ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/model_path.h)

install(DIRECTORY ${PREFIX_DIR}/model/
  DESTINATION ${MODEL_INSTALL_PATH}
)

install(DIRECTORY ${PREFIX_DIR}/include/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY ${PREFIX_DIR}/lib/
  DESTINATION lib/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${prebuilt_libs})
ament_package()
