cmake_minimum_required(VERSION 3.5)
project(cyberdog_vision LANGUAGES C CXX)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#dependencies
find_package(ament_cmake REQUIRED)
find_package(vision_sdk_vendor REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(libraries
  face_detect_api
  body_detect_api
  reid_api)

add_library(face_detect_api SHARED
  src/face_detect_api.cpp)

add_library(body_detect_api SHARED
  src/body_detect_api.cpp)

add_library(reid_api SHARED
  src/reid_api.cpp)

ament_target_dependencies(face_detect_api
  vision_sdk_vendor
)
ament_target_dependencies(body_detect_api
  vision_sdk_vendor
)
ament_target_dependencies(reid_api
  vision_sdk_vendor
)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS ${libraries}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_include_directories(include)
ament_export_libraries(${libraries})

if(BUILD_TESTING)
  set(ament_cmake_cppcheck_LANGUAGE c++)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_package()
