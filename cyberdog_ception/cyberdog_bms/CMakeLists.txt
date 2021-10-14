cmake_minimum_required(VERSION 3.5)
project(cyberdog_bms)

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

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(lcm_vendor REQUIRED)
find_package(lcm_translate_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(interaction_msgs REQUIRED)
include_directories(
./include
${catkin_INCLUDE_DIRS}
)

add_executable(bms_recv src/bms_logger.cpp src/bms_recv.cpp)
ament_target_dependencies(bms_recv rclcpp std_msgs ception_msgs lcm_vendor rclcpp_action interaction_msgs
  lcm_translate_msgs)
target_link_libraries(bms_recv)

add_executable(bms_ctrl src/bms_ctrl.cpp)
ament_target_dependencies(bms_ctrl rclcpp std_msgs ception_msgs lcm_vendor
  lcm_translate_msgs)
target_link_libraries(bms_ctrl)

## Install headers

install(TARGETS
  bms_ctrl
  bms_recv
  DESTINATION
  lib/${PROJECT_NAME})

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
