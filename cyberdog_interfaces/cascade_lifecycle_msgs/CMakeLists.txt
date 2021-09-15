cmake_minimum_required(VERSION 3.5)
project(cascade_lifecycle_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Activation.msg" 
  "msg/State.msg" 
 DEPENDENCIES builtin_interfaces lifecycle_msgs
)

ament_export_dependencies(rosidl_default_runtime lifecycle_msgs builtin_interfaces)

ament_package()
