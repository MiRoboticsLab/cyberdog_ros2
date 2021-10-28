cmake_minimum_required(VERSION 3.5)
project(motion_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # action
  "action/ChangeGait.action"
  "action/ChangeMode.action"
  "action/ExtMonOrder.action"
  # msg
  "msg/ActionRequest.msg"
  "msg/ActionRespond.msg"
  "msg/BehaviorTreeLog.msg"
  "msg/BehaviorTreeStatusChange.msg"
  "msg/ControlState.msg"
  "msg/ErrorFlag.msg"
  "msg/Force.msg"
  "msg/Frameid.msg"
  "msg/Gait.msg"
  "msg/Mode.msg"
  "msg/MonOrder.msg"
  "msg/Parameters.msg"
  "msg/Safety.msg"
  "msg/Scene.msg"
  "msg/SE3Pose.msg"
  "msg/SE3Velocity.msg"
  "msg/SE3VelocityCMD.msg"
  # srv
  "srv/MotionTest.srv"
 DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs)

ament_package()
