cmake_minimum_required(VERSION 3.5)
project(interaction_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # action
  "action/AudioPlay.action"
  # msg
  "msg/AudioFeedback.msg"
  "msg/AudioOrder.msg"
  "msg/AudioResult.msg"
  "msg/AudioSongName.msg"
  "msg/AudioUser.msg"
  "msg/Body.msg"
  "msg/BodyInfo.msg"
  "msg/CompressedImage.msg"
  "msg/Face.msg"
  "msg/FaceInfo.msg"
  "msg/FaceResult.msg"
  "msg/Gesture.msg"
  "msg/GestureInfo.msg"
  "msg/Led.msg"
  "msg/Owner.msg"
  "msg/Token.msg"
  "msg/Touch.msg"
  "msg/VoiceprintEntry.msg"
  "msg/VoiceprintResult.msg"
  # srv
  "srv/BodyRegion.srv"
  "srv/CameraService.srv"
  "srv/FaceManager.srv"
  "srv/TokenNew.srv"
  "srv/TokenPass.srv"
  "srv/Voiceprint.srv"
  "srv/AskAssistant.srv"
 DEPENDENCIES sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime sensor_msgs std_msgs)

ament_package()
