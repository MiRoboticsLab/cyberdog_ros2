cmake_minimum_required(VERSION 3.5)
project(cyberdog_joy)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(motion_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(sdl_joy REQUIRED)
find_package(toml11_vendor REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} SHARED src/cyberdog_joy.cpp)

# define project name to get relative share path
target_compile_definitions(${PROJECT_NAME} PRIVATE "PACKAGE_NAME=\"${PROJECT_NAME}\"")

ament_target_dependencies(${PROJECT_NAME}
  cyberdog_utils
  rclcpp
  rclcpp_action
  rclcpp_components
  sensor_msgs
  motion_msgs
  sdl_joy
  toml11_vendor
)

rclcpp_components_register_nodes(${PROJECT_NAME}
  "cyberdog_joy::CyberdogJoy"
)

add_executable(${PROJECT_NAME}_node src/joy_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_node
  PROPERTIES OUTPUT_NAME joy_node PREFIX ""
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
