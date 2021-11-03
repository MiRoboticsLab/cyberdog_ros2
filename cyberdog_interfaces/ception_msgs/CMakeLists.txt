cmake_minimum_required(VERSION 3.5)
project(ception_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # action
  #
  # msg
  "msg/Around.msg"
  "msg/Bms.msg"
  "msg/BtRemoteEvent.msg"
  "msg/BodyState.msg"
  "msg/GpsPayload.msg"
  "msg/Light.msg"
  "msg/Speed3.msg"
  "msg/Ultrasonic.msg"
  "msg/ShutDown.msg"
  # srv
  "srv/BmsInfo.srv"
  "srv/BtRemoteCommand.srv"
  "srv/CheckPower.srv"
  "srv/GpsSceneNode.srv"
  "srv/SensorDetectionNode.srv"
 DEPENDENCIES geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime geometry_msgs sensor_msgs std_msgs)

ament_package()
