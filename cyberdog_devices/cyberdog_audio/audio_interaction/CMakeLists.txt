cmake_minimum_required(VERSION 3.5)
project(audio_interaction LANGUAGES C CXX)

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

include_directories(include)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(audio_base REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(ALSA REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ception_msgs REQUIRED)
find_package(interaction_msgs REQUIRED)
find_package(motion_msgs REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(toml11_vendor REQUIRED)

set(library_name audio_interaction)

set(dependencies
  audio_base
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  ALSA
  ception_msgs
  interaction_msgs
  motion_msgs
  cyberdog_utils
  rclcpp_components
  builtin_interfaces
  toml11_vendor
)

add_library(${library_name} SHARED
  src/voice_cmd.cpp
  src/md5.cpp
)

ament_target_dependencies(${library_name}
  ${dependencies}
)

target_link_libraries(${library_name}
  OpenSSL::Crypto
  OpenSSL::SSL
)

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)


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

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})
ament_package()
