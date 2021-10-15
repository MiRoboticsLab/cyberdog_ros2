cmake_minimum_required(VERSION 3.5)
project(sdl_joy)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(LIBRARY_NAME ${PROJECT_NAME}_core)

find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sdl2_vendor REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(toml11_vendor REQUIRED)

include_directories(include)

add_library(${LIBRARY_NAME} SHARED
  src/sdl_joy.cpp
)

# define project name to get relative share path
target_compile_definitions(${LIBRARY_NAME} PRIVATE "PACKAGE_NAME=\"${PROJECT_NAME}\"")

ament_target_dependencies(${LIBRARY_NAME}
  rclcpp_components
  rclcpp
  sensor_msgs
  sdl2_vendor
  toml11_vendor
)

target_link_libraries(${LIBRARY_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS ${LIBRARY_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_export_include_directories(include)
ament_export_libraries(${LIBRARY_NAME})
ament_package()
