cmake_minimum_required(VERSION 3.8)
project(cyberdog_motion_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cyberdog_bt_engine REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(toml11_vendor REQUIRED)

include_directories(include)

set(library_name motion_bridge)
set(dependencies
  cyberdog_bt_engine
  cyberdog_utils
  toml11_vendor
)

add_library(${library_name} SHARED
  src/gait_interface.cpp
)

ament_target_dependencies(${library_name}
  ${dependencies}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_package()
