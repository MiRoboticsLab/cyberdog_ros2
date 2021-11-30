cmake_minimum_required(VERSION 3.5)
project(cyberdog_audio LANGUAGES C CXX)

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

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(audio_base REQUIRED)
find_package(audio_interaction REQUIRED)
find_package(audio_assistant QUIET)

include_directories(include)

set(dependencies
  rclcpp
  audio_base
  audio_interaction
)

set(executable_name cyberdog_audio)

add_executable(${executable_name}
  src/main.cpp
)

target_compile_definitions(${executable_name} PRIVATE "ENABLE_BACKTRACE")

if(audio_assistant_FOUND AND XIAOMI_XIAOAI)
  target_compile_definitions(${executable_name} PRIVATE "ENABLE_ASSISTANT")
  set(dependencies
    ${dependencies}
    audio_assistant
  )
else()
  message(WARNING ">>> Build audio without Xiaomi Xiaoai.")
endif()

ament_target_dependencies(${executable_name}
  ${dependencies}
)

include(ExternalProject)
set(EXTPRJ_NAME wav_files)
set(PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXTPRJ_NAME}")

externalproject_add(
  ${EXTPRJ_NAME}
  PREFIX ${EXTPRJ_NAME}
  URL https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/CyberDog_wavs.tar.xz
  URL_MD5 07b843179ec7c77028642695f020cadd
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${PREFIX_DIR}
  INSTALL_DIR "${PREFIX_DIR}"
  TIMEOUT 30
)

install(DIRECTORY ${PREFIX_DIR}/wav
  DESTINATION data/
)

install(DIRECTORY data/
  DESTINATION data/
)

install(TARGETS ${executable_name}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_dependencies(${dependencies})
ament_package()
