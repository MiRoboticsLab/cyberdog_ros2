cmake_minimum_required(VERSION 3.8)
project(cyberdog_gps)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include)

# find dependencies
find_package(Threads REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(pluginlib REQUIRED)
find_package(ception_base REQUIRED)

set(dependencies
  rclcpp
  ception_msgs
  cyberdog_utils
  pluginlib
  ception_base
)

add_library(${PROJECT_NAME} SHARED src/cyberdog_gps.cpp)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})
target_compile_features(${PROJECT_NAME} PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

add_executable(${PROJECT_NAME}_node src/gps_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_package()
