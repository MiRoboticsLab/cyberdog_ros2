cmake_minimum_required(VERSION 3.8)
project(cyberdog_ception_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params/
)

install(DIRECTORY logs/
  DESTINATION share/${PROJECT_NAME}/logs/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
