// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "cyberdog_app_server.hpp"
#include <string>
#include <memory>

using SE3VelocityCMD_T = motion_msgs::msg::SE3VelocityCMD;
CyberdogAppImpl::CyberdogAppImpl(const std::string & db)
: decision_(NULL)
{
}
::grpc::Status CyberdogAppImpl::setMode(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::CheckoutMode_request * request,
  ::grpc::ServerWriter<::cyberdogapp::CheckoutMode_respond> * writer)
{
  std::cout << "Server get mode: " << request->next_mode().mode().control_mode() << std::endl;
  if (decision_) {
    decision_->publishMode(request, writer);
  }
  return Status::OK;
}
::grpc::Status CyberdogAppImpl::setPattern(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::CheckoutPattern_request * request,
  ::grpc::ServerWriter<::cyberdogapp::CheckoutPattern_respond> * writer)
{
  std::cout << "Server get pattern: " << request->patternstamped().pattern().gait_pattern() <<
    std::endl;
  if (decision_) {
    decision_->publishPattern(request, writer);
  }
  return Status::OK;
}

Status CyberdogAppImpl::requestCamera(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::CameraService_request * request,
  ::grpc::ServerWriter<::cyberdogapp::CameraService_respond> * writer)
{
  uint32_t command = request->command();
  std::string args = request->args();
  ::cyberdogapp::CameraService_respond respond;
  if (decision_) {
    decision_->callCameraService(command, args, &respond);
    respond.set_command(command);
    writer->Write(respond);
  }
  return Status::OK;
}

Status CyberdogAppImpl::sendAppDecision(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::Decissage * request,
  ::cyberdogapp::Result * response)
{
  SE3VelocityCMD_T decissage_out;
  if (decision_) {
    // decissage_out.header.stamp = decision_->get_clock()->now();
  }
  decissage_out.sourceid = motion_msgs::msg::SE3VelocityCMD::REMOTEC;
  decissage_out.velocity.frameid.id = motion_msgs::msg::Frameid::BODY_FRAME;
  decissage_out.velocity.timestamp = decision_->get_clock()->now();
  decissage_out.velocity.linear_x = request->twist().linear().x();
  decissage_out.velocity.linear_y = request->twist().linear().y();
  decissage_out.velocity.linear_z = request->twist().linear().z();

  decissage_out.velocity.angular_x = request->twist().angular().x();
  decissage_out.velocity.angular_y = request->twist().angular().y();
  decissage_out.velocity.angular_z = request->twist().angular().z();
  if (decision_) {
    decision_->publishMotion(decissage_out);
  }
  std::cout << "Server get SendAppDecision: " << std::endl;

  return Status::OK;
}

::grpc::Status CyberdogAppImpl::setFollowRegion(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::BodyRegion_Request * request,
  ::grpc::ServerWriter<::cyberdogapp::BodyRegion_Respond> * writer)
{
  ::cyberdogapp::BodyRegion_Respond respond;
  if (decision_) {
    decision_->setFollowRegion(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}

::grpc::Status CyberdogAppImpl::requestVoice(
  ::grpc::ServerContext * context, const ::cyberdogapp::Voiceprint_Request * request,
  ::grpc::ServerWriter<::cyberdogapp::Voiceprint_Response> * writer)
{
  ::cyberdogapp::Voiceprint_Response respond;
  if (decision_) {
    decision_->requestVoice(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}
::grpc::Status CyberdogAppImpl::requestFaceManager(
  ::grpc::ServerContext * context, const ::cyberdogapp::FaceManager_Request * request,
  ::grpc::ServerWriter<::cyberdogapp::FaceManager_Response> * writer)
{
  ::cyberdogapp::FaceManager_Response respond;
  if (decision_) {
    decision_->requestFaceManager(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}
::grpc::Status CyberdogAppImpl::sendAiToken(
  ::grpc::ServerContext * context, const ::cyberdogapp::TokenPass_Request * request,
  ::grpc::ServerWriter<::cyberdogapp::TokenPass_Response> * writer)
{
  ::cyberdogapp::TokenPass_Response respond;
  if (decision_) {
    decision_->sendAiToken(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}
::grpc::Status CyberdogAppImpl::setNavPosition(
  ::grpc::ServerContext * context, const ::cyberdogapp::Target_Request * request,
  ::grpc::ServerWriter<::cyberdogapp::Target_Response> * writer)
{
  ::cyberdogapp::Target_Response respond;
  if (decision_) {
    decision_->setNavPosition(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}

::grpc::Status CyberdogAppImpl::disconnect(
  ::grpc::ServerContext * context, const ::cyberdogapp::Disconnect * request,
  ::grpc::ServerWriter<::cyberdogapp::Result> * writer)
{
  ::cyberdogapp::Result respond;
  if (decision_) {
    decision_->disconnect(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}

::grpc::Status CyberdogAppImpl::setBtRemoteCmd(
  ::grpc::ServerContext * context, const ::cyberdogapp::BtRemoteCommand_Request * request,
  ::grpc::ServerWriter<::cyberdogapp::BtRemoteCommand_Respond> * writer)
{
  ::cyberdogapp::BtRemoteCommand_Respond respond;
  if (decision_) {
    decision_->setBtRemoteCmd(request, &respond);
    writer->Write(respond);
  }
  return Status::OK;
}

::grpc::Status CyberdogAppImpl::setBodyPara(
  ::grpc::ServerContext * context, const ::cyberdogapp::Parameters * request,
  ::cyberdogapp::Result * response)
{
  Parameters_T parameter;
  parameter.body_height = request->body_height();
  parameter.gait_height = request->gait_height();
  if (decision_) {
    decision_->setBodyPara(parameter);
  }
  return Status::OK;
}

void CyberdogAppImpl::process_motion(const ::cyberdogapp::MotionCommand * request)
{
  switch (request->command()) {
    case ::cyberdogapp::MotionCommand::TEST_INIT:
      system("ps -ef | grep decision | grep -v grep | awk '{print $2}' | xargs kill -9");
      system("/opt/ros2/foxy/bin/ros2 run cyberdog_motion_test motion_test_server");
      break;

    case ::cyberdogapp::MotionCommand::TEST_DEINIT:
      system("echo '123' | sudo systemctl restart cyberdog_ros2.service");
      break;

    case ::cyberdogapp::MotionCommand::TEST_START:
      RCLCPP_INFO(decision_->get_logger(), "START");
      system("/opt/ros2/cyberdog/bin/send_msg_ COMMAND:START");
      break;

    case ::cyberdogapp::MotionCommand::TEST_STOP:
      RCLCPP_INFO(decision_->get_logger(), "STOP");
      system("/opt/ros2/cyberdog/bin/send_msg_  COMMAND:STOP");
      break;

    case ::cyberdogapp::MotionCommand::TURN_LEFT:
      RCLCPP_INFO(decision_->get_logger(), "BBBB");
      system("/opt/ros2/cyberdog/bin/send_msg_  CONTROL:BBBB");
      break;

    case ::cyberdogapp::MotionCommand::TURN_RIGHT:
      RCLCPP_INFO(decision_->get_logger(), "OOOO");
      system("/opt/ros2/cyberdog/bin/send_msg_  CONTROL:OOOO");
      break;

    case ::cyberdogapp::MotionCommand::GO_AHEAD:
      RCLCPP_INFO(decision_->get_logger(), "WWWW");
      system("/opt/ros2/cyberdog/bin/send_msg_  CONTROL:WWWW");
      break;

    case ::cyberdogapp::MotionCommand::GO_BACK:
      RCLCPP_INFO(decision_->get_logger(), "SSSS");
      system("/opt/ros2/cyberdog/bin/send_msg_  CONTROL:SSSS");
      break;

    case ::cyberdogapp::MotionCommand::GO_LEFT:
      RCLCPP_INFO(decision_->get_logger(), "AAAA");
      system("/opt/ros2/cyberdog/bin/send_msg_  CONTROL:AAAA");
      break;

    case ::cyberdogapp::MotionCommand::GO_RIGHT:
      RCLCPP_INFO(decision_->get_logger(), "DDDD");
      system("/opt/ros2/cyberdog/bin/send_msg_  CONTROL:DDDD");
      break;

    default:
      break;
  }
}

Status CyberdogAppImpl::sendMotionTestRequest(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::MotionCommand * request,
  ::cyberdogapp::Result * response)
{
  std::shared_ptr<std::thread> feedback_thread = std::make_shared<std::thread>(
    &CyberdogAppImpl::process_motion, this, request);
  feedback_thread->detach();
  return Status::OK;
}
::grpc::Status CyberdogAppImpl::getOffsetData(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::OffsetRequest * request,
  ::grpc::ServerWriter< ::cyberdogapp::OffsetCalibationData> * writer)
{
  RCLCPP_INFO(decision_->get_logger(), "getOffsetData");
  if (decision_) {
    decision_->getOffsetData(request, writer);
  }
  return Status::OK;
}

::grpc::Status CyberdogAppImpl::setOffsetData(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::OffsetCalibationData * request,
  ::grpc::ServerWriter< ::cyberdogapp::OffsetRequest_result> * writer)
{
  RCLCPP_INFO(decision_->get_logger(), "setOffsetData");
  if (decision_) {
    decision_->setOffsetData(request, writer);
  }
  return Status::OK;
}

::grpc::Status CyberdogAppImpl::setExtmonOrder(
  ::grpc::ServerContext * context,
  const ::cyberdogapp::ExtMonOrder_Request * request,
  ::grpc::ServerWriter< ::cyberdogapp::ExtMonOrder_Respond> * writer)
{
  RCLCPP_INFO(decision_->get_logger(), "setExtmonOrder");
  if (decision_) {
    decision_->setExtmonOrder(request, writer);
  }
  return Status::OK;
}
