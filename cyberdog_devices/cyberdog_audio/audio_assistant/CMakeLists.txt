cmake_minimum_required(VERSION 3.5)
project(audio_assistant LANGUAGES C CXX)

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

option(XIAOMI_XIAOAI "Build with Xiaomi Xiaoai Package" OFF)

if(NOT XIAOMI_XIAOAI)
  message(WARNING ">>> Skip building Xiaomi Xiaoai Package")
  return()
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interaction_msgs REQUIRED)
find_package(cyberdog_utils REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(audio_base REQUIRED)
find_package(xiaoai_sdk_vendor REQUIRED)
find_package(toml11_vendor REQUIRED)
find_package(mpg123_vendor REQUIRED)

include_directories(include)

set(library_name audio_assistant)

set(dependencies
  audio_base
  interaction_msgs
  cyberdog_utils
  builtin_interfaces
  xiaoai_sdk_vendor
  toml11_vendor
  mpg123_vendor
)

add_library(${library_name} SHARED
  src/audio_assistant.cpp
  src/audio_token.cpp
  src/ai_keyword.cpp
  src/ai_asr_native.cpp
  src/ai_nlp_native.cpp
  src/ai_online/AivsMain.cpp
  src/ai_online/AuthCapabilityImpl.cpp
  src/ai_online/ClientLoggerHooker.cpp
  src/ai_online/ConnectionCapabilityImpl.cpp
  src/ai_online/ErrorCapabilityImpl.cpp
  src/ai_online/StorageCapabilityImpl.cpp
  src/combiner.cpp
  src/mp3decoder.cpp
)

ament_target_dependencies(${library_name}
  ${dependencies}
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
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
endif()

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})
ament_package()
