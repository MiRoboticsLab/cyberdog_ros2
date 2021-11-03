cmake_minimum_required(VERSION 3.8)
project(cyberdog_scene_detection)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(Threads REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(motion_msgs REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(pluginlib REQUIRED)
find_package(ception_base REQUIRED)

set(dependencies
  rclcpp
  ception_msgs
  motion_msgs
  cyberdog_utils
  pluginlib
  ception_base
)

set(executable_name service_scene_detection)

add_executable(${executable_name} src/scene_detection.cpp)
target_compile_features(${executable_name} PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(${executable_name} ${dependencies})
target_link_libraries(${executable_name})

install(TARGETS ${executable_name}
  DESTINATION lib/${PROJECT_NAME})

target_link_libraries(${executable_name}
  ${CMAKE_THREAD_LIBS_INIT}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_package()
