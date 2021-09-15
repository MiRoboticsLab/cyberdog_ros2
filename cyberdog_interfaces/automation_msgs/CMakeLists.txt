cmake_minimum_required(VERSION 3.5)
project(automation_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(motion_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # action
  #
  # msg
  "msg/BehaviorTreeLog.msg"
  "msg/BehaviorTreeStatusChange.msg"
  "msg/Caution.msg"
  "msg/TrackingStatus.msg"
  "msg/NavStatus.msg"
  # srvs
  "srv/NavigateToPose.srv"
  "srv/NavMode.srv"
  "srv/RestrictedArea.srv"
  "srv/Target.srv"
 DEPENDENCIES builtin_interfaces motion_msgs nav_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime builtin_interfaces motion_msgs nav_msgs std_msgs)

ament_package()
