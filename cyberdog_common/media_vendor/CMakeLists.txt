cmake_minimum_required(VERSION 3.5)
project(media_vendor)

find_package(ament_cmake REQUIRED)

install(DIRECTORY cmake DESTINATION share/${PROJECT_NAME})

ament_package(
  CONFIG_EXTRAS "media_vendor-extras.cmake"
)
