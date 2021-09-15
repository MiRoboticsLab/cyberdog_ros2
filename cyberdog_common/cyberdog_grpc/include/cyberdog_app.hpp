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

#ifndef CYBERDOG_APP_HPP_
#define CYBERDOG_APP_HPP_
#include <rclcpp_action/rclcpp_action.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <utility>

// Interfaces
#include "motion_msgs/msg/mode.hpp"
#include "ception_msgs/msg/bms.hpp"
#include "ception_msgs/msg/bt_remote_event.hpp"
#include "interaction_msgs/msg/body_info.hpp"
#include "interaction_msgs/srv/camera_service.hpp"
#include "interaction_msgs/srv/body_region.hpp"
#include "interaction_msgs/srv/face_manager.hpp"
#include "interaction_msgs/srv/voiceprint.hpp"
#include "interaction_msgs/srv/token_pass.hpp"
#include "automation_msgs/srv/target.hpp"
#include "automation_msgs/msg/caution.hpp"
#include "motion_msgs/msg/action_request.hpp"
#include "motion_msgs/msg/action_respond.hpp"
#include "lcm/lcm-cpp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/create_timer_ros.h"

#include "cyberdog_app_client.hpp"
#include "motion_msgs/action/change_mode.hpp"
#include "motion_msgs/action/change_gait.hpp"

#include "threadsafe_queue.hpp"
#include "ception_msgs/msg/around.hpp"
#include "msgdispatcher.hpp"
#include "lcm_translate_msgs/control_parameter_request_lcmt.hpp"
#include "lcm_translate_msgs/control_parameter_respones_lcmt.hpp"
#include "motion_msgs/msg/se3_velocity_cmd.hpp"
#include "motion_msgs/msg/se3_velocity.hpp"
#include "motion_msgs/msg/se3_pose.hpp"
#include "cyberdog_utils/Enums.hpp"
#include "motion_msgs/msg/scene.hpp"
#include "net_avalible.hpp"
#include "ception_msgs/srv/bt_remote_command.hpp"
#include "nav_msgs/msg/path.hpp"

using ControlState_T = motion_msgs::msg::ControlState;
using SE3VelocityCMD_T = motion_msgs::msg::SE3VelocityCMD;
using ChangeMode = motion_msgs::action::ChangeMode;
using GoalHandleChangeMode = rclcpp_action::ClientGoalHandle<ChangeMode>;
using ChangeGait = motion_msgs::action::ChangeGait;
using GoalHandleChangeGait = rclcpp_action::ClientGoalHandle<ChangeGait>;
using string = std::string;
namespace cyberdog_cyberdog_app
{
class Cyberdog_app : public rclcpp::Node
{
public:
  Cyberdog_app();
  void publishMotion(const SE3VelocityCMD_T & decissage_out);
  void publishPattern(
    const ::cyberdogapp::CheckoutPattern_request * request,
    ::grpc::ServerWriter<::cyberdogapp::CheckoutPattern_respond> * writer);
  void publishStatus(const ControlState_T & status_out);
  void publishMode(
    const ::cyberdogapp::CheckoutMode_request * request,
    ::grpc::ServerWriter<::cyberdogapp::CheckoutMode_respond> * writer);
  void callCameraService(
    int command, std::string args,
    ::cyberdogapp::CameraService_respond * respond);
  void setFollowRegion(
    const ::cyberdogapp::BodyRegion_Request * request,
    ::cyberdogapp::BodyRegion_Respond * respond);
  void requestFaceManager(
    const ::cyberdogapp::FaceManager_Request * request,
    ::cyberdogapp::FaceManager_Response * respond);
  void sendAiToken(
    const ::cyberdogapp::TokenPass_Request * request,
    ::cyberdogapp::TokenPass_Response * respond);
  void setNavPosition(
    const ::cyberdogapp::Target_Request * request,
    ::cyberdogapp::Target_Response * respond);
  void requestVoice(
    const ::cyberdogapp::Voiceprint_Request * request,
    ::cyberdogapp::Voiceprint_Response * respond);

  void getOffsetData(
    const ::cyberdogapp::OffsetRequest * request,
    ::grpc::ServerWriter< ::cyberdogapp::OffsetCalibationData> * writer);

  void setOffsetData(
    const ::cyberdogapp::OffsetCalibationData * request,
    ::grpc::ServerWriter< ::cyberdogapp::OffsetRequest_result> * writer);

  void setExtmonOrder(
    const ::cyberdogapp::ExtMonOrder_Request * request,
    ::grpc::ServerWriter< ::cyberdogapp::ExtMonOrder_Respond> * writer);

  void disconnect(
    const ::cyberdogapp::Disconnect * request,
    ::cyberdogapp::Result * respond);

  void setBtRemoteCmd(
    const ::cyberdogapp::BtRemoteCommand_Request * request,
    ::cyberdogapp::BtRemoteCommand_Respond * respond);

  void setBodyPara(const Parameters_T & para);

private:
  rclcpp::Publisher<motion_msgs::msg::ActionRequest>::SharedPtr action_request_pub;
  rclcpp::Publisher<SE3VelocityCMD_T>::SharedPtr decision_pub;
  rclcpp::Publisher<Parameters_T>::SharedPtr para_pub;
  rclcpp::Subscription<motion_msgs::msg::ActionRespond>::SharedPtr action_result_sub;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  void RunServer();
  rclcpp_action::Client<motion_msgs::action::ChangeMode>::SharedPtr ModeCheckout_client_;
  rclcpp_action::Client<motion_msgs::action::ChangeGait>::SharedPtr patternCheckout_client_;
  rclcpp::Client<interaction_msgs::srv::CameraService>::SharedPtr camera_client_;
  rclcpp::Client<interaction_msgs::srv::BodyRegion>::SharedPtr tracking_object_client_;
  rclcpp::Client<interaction_msgs::srv::Voiceprint>::SharedPtr voice_print_client_;
  rclcpp::Client<automation_msgs::srv::Target>::SharedPtr nav_position_client_;
  rclcpp::Client<interaction_msgs::srv::TokenPass>::SharedPtr ai_token_client_;
  rclcpp::Client<interaction_msgs::srv::FaceManager>::SharedPtr face_manager_client_;
  rclcpp::Client<ception_msgs::srv::BtRemoteCommand>::SharedPtr remote_manager_client_;

  std_msgs::msg::Header returnHeader();
  std::shared_ptr<std::thread> app_server_thread_;
  std::shared_ptr<std::thread> heart_beat_thread_;
  std::shared_ptr<std::thread> destory_grpc_server_thread_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ip_subscriber;
  rclcpp::Subscription<ception_msgs::msg::Bms>::SharedPtr Bms_subscriber;
  rclcpp::Subscription<ControlState_T>::SharedPtr Status_subscriber;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr wifi_rssi;
  rclcpp::Subscription<interaction_msgs::msg::BodyInfo>::SharedPtr body_select;
  rclcpp::Subscription<interaction_msgs::msg::BodyInfo>::SharedPtr tracking;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map;
  rclcpp::Subscription<SE3VelocityCMD_T>::SharedPtr nav_odom;
  rclcpp::Subscription<interaction_msgs::msg::VoiceprintResult>::SharedPtr voice_print_result;
  rclcpp::Subscription<interaction_msgs::msg::FaceResult>::SharedPtr face_result;
  rclcpp::Subscription<automation_msgs::msg::TrackingStatus>::SharedPtr tracking_status;
  rclcpp::Subscription<automation_msgs::msg::Caution>::SharedPtr nav_status;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_out;
  rclcpp::Subscription<ception_msgs::msg::Around>::SharedPtr obstacle_detection;
  rclcpp::Subscription<motion_msgs::msg::SE3Pose>::SharedPtr dog_pose;
  rclcpp::Subscription<motion_msgs::msg::Scene>::SharedPtr scene_detection;
  rclcpp::Subscription<ception_msgs::msg::BtRemoteEvent>::SharedPtr remote_status;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscriber;

  void subscribeVoiceprintResult(const interaction_msgs::msg::VoiceprintResult::SharedPtr msg);
  void subscribeFaceResult(const interaction_msgs::msg::FaceResult::SharedPtr msg);
  void subscribeMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void set_bms(const ception_msgs::msg::Bms::SharedPtr msg);
  void set_status(const ControlState_T::SharedPtr msg);
  void set_rssi(const std_msgs::msg::String::SharedPtr msg);
  void set_body_select(const interaction_msgs::msg::BodyInfo::SharedPtr msg);
  void set_tracking(const interaction_msgs::msg::BodyInfo::SharedPtr msg);
  void subscribePosition(const SE3VelocityCMD_T::SharedPtr msg);
  void subscribeIp(const std_msgs::msg::String::SharedPtr msg);
  void createGrpc();
  void destroyGrpc();
  void subscribeActionResult(const motion_msgs::msg::ActionRespond::SharedPtr msg);
  void subscribeTrackingStatus(const automation_msgs::msg::TrackingStatus::SharedPtr msg);
  void subscribeNavStatus(const automation_msgs::msg::Caution::SharedPtr msg);
  void subscribeOdomOut(const nav_msgs::msg::Odometry::SharedPtr msg);
  void subscribeObstacleDetection(const ception_msgs::msg::Around::SharedPtr msg);
  void subscribeGpsScene(const motion_msgs::msg::Scene::SharedPtr msg);
  void subscribePath(const nav_msgs::msg::Path::SharedPtr msg);
  void receive_lcm_data(
    const lcm::ReceiveBuffer * rbuf,
    const std::string & chan,
    const control_parameter_respones_lcmt * lcm_data);
  void subscribeDogPose(const motion_msgs::msg::SE3Pose::SharedPtr msg);
  void subscribeRemoteEvent(const ception_msgs::msg::BtRemoteEvent::SharedPtr msg);

  void destroyGrpcServer();
  std::string getDogIp(const string str, const string & split);
  std::string getPhoneIp(const string str, const string & split);
  std::shared_ptr<threadsafe_queue<std::shared_ptr<::cyberdogapp::CheckoutMode_respond>>>
  modeRespond_queue;
  std::shared_ptr<threadsafe_queue<std::shared_ptr<::cyberdogapp::CheckoutPattern_respond>>>
  patternRespond_queue;
  std::shared_ptr<threadsafe_queue<std::shared_ptr<::cyberdogapp::ExtMonOrder_Respond>>>
  extMonOrderRespond_queue;
  std::shared_ptr<Cyberdog_App_Client> app_stub;
  std::shared_ptr<std::string> server_ip;
  std::shared_ptr<grpc::Server> server_;
  bool is_mode_check;
  bool can_process_messages;
  void mode_feed_back(const std::shared_ptr<const ChangeMode::Feedback> feedback);
  void mode_result_back(const GoalHandleChangeMode::WrappedResult & result);
  uint32_t ticks_;
  void HeartBeat();
  std::shared_ptr<lcm::LCM> offset_response;
  std::shared_ptr<lcm::LCM> offset_request;
  std::string getLcmUrl(int ttl);
  bool wait_for_offset_data;
  bool wait_for_set_data_result;
  std::shared_ptr<threadsafe_queue<std::shared_ptr<::cyberdogapp::OffsetCalibationData>>>
  offsetCalibationData_queue;
  std::shared_ptr<threadsafe_queue<std::shared_ptr<::cyberdogapp::OffsetRequest_result>>>
  offsetRequestResult_queue;
  void recv_lcm_handle();
  std::shared_ptr<std::thread> lcm_handle_thread_;
  NetChecker net_checker;
  uint32_t heartbeat_err_cnt;
  bool app_disconnected;
  std::string local_ip;
  int change_mode_id;
  int change_gait_id;
  int ext_mon_id;
  bool isMatch(const char * source, const char * dest);
};
}  // namespace cyberdog_cyberdog_app

#endif  // CYBERDOG_APP_HPP_
