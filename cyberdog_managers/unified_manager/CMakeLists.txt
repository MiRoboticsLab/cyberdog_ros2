cmake_minimum_required(VERSION 3.5)
project(cyberdog_unified_manager)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(automation_msgs REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cascade_lifecycle_msgs REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(cyberdog_motion_bridge REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(interaction_msgs REQUIRED)
find_package(lcm_vendor REQUIRED)
find_package(lcm_translate_msgs REQUIRED)
find_package(motion_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(toml11_vendor REQUIRED)

include_directories(include)

set(executable_name unified_manager)
set(library_name ${executable_name}_core)
set(debug_all_file ${CMAKE_SOURCE_DIR}/.debug_all)
set(debug_motion_file ${CMAKE_SOURCE_DIR}/.debug_motion)
set(debug_mock_file ${CMAKE_SOURCE_DIR}/.debug_mock)

set(dependencies
  automation_msgs
  behaviortree_cpp_v3
  builtin_interfaces
  cascade_lifecycle_msgs
  ception_msgs
  cyberdog_motion_bridge
  cyberdog_utils
  interaction_msgs
  lcm_translate_msgs
  lcm_vendor
  motion_msgs
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  toml11_vendor
)

add_library(${library_name} SHARED
  src/manager_utils/bt_engine.cpp
  src/manager_utils/cascade_manager.cpp
  src/managers/automation_manager.cpp
  src/managers/motion_manager.cpp
  src/managers/ception_manager.cpp
  src/managers/interaction_manager.cpp
)

if(EXISTS ${debug_all_file})
  add_definitions(-DDEBUG_ALL)
  message("Enable debug mode [ALL]")
  if(EXISTS ${debug_motion_file})
    add_definitions(-DDEBUG_MOTION)
    message("Enable debug mode [MOTION]")
  endif()
  if(EXISTS ${debug_mock_file})
    add_definitions(-DDEBUG_MOCK)
    message("Enable debug mode [MOCK]")
  endif()
endif()

# define project name to get relative share path
target_compile_definitions(${library_name} PRIVATE "PACKAGE_NAME=\"${PROJECT_NAME}\"")

ament_target_dependencies(${library_name}
  ${dependencies}
)

add_executable(${executable_name}
  src/main.cpp
)

ament_target_dependencies(${executable_name}
  ${dependencies}
)

target_link_libraries(${executable_name}
  ${library_name}
  stdc++fs
)

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${executable_name}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params/
)

install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})

ament_package()
