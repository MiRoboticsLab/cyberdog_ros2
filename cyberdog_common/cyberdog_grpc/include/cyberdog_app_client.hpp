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


#ifndef CYBERDOG_APP_CLIENT_HPP_
#define CYBERDOG_APP_CLIENT_HPP_
#include "./cyberdog_app.grpc.pb.h"
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include "ception_msgs/msg/bms.hpp"
#include "ception_msgs/msg/around.hpp"
#include "ception_msgs/msg/bt_remote_event.hpp"
#include "motion_msgs/msg/se3_velocity_cmd.hpp"
#include "motion_msgs/msg/se3_velocity.hpp"
#include "motion_msgs/msg/control_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "automation_msgs/msg/tracking_status.hpp"
#include "automation_msgs/msg/caution.hpp"
#include "interaction_msgs/msg/body_info.hpp"
#include "interaction_msgs/msg/voiceprint_result.hpp"
#include "interaction_msgs/msg/face_result.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "msgdispatcher.hpp"
#include "motion_msgs/msg/se3_pose.hpp"
#include "motion_msgs/msg/scene.hpp"

using cyberdogapp::Bms;
using cyberdogapp::Decissage;
using cyberdogapp::Mode;
using cyberdogapp::Pattern;
using cyberdogapp::RawStatus;
using cyberdogapp::Result;
using cyberdogapp::StatusStamped;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using ControlState_T = motion_msgs::msg::ControlState;
using SE3VelocityCMD_T = motion_msgs::msg::SE3VelocityCMD;
using Parameters_T = motion_msgs::msg::Parameters;
class Cyberdog_App_Client
{
public:
  explicit Cyberdog_App_Client(std::shared_ptr<Channel> channel);
  ~Cyberdog_App_Client();
  bool SetBms(const ception_msgs::msg::Bms::SharedPtr bms);
  bool SetStatus(const ControlState_T::SharedPtr status);
  bool set_rssi(const std_msgs::msg::String::SharedPtr msg);
  bool subscribeTrackingStatus(const automation_msgs::msg::TrackingStatus::SharedPtr status);
  bool subscribeNavStatus(const automation_msgs::msg::Caution::SharedPtr status);
  bool set_BodySelect(const interaction_msgs::msg::BodyInfo::SharedPtr msg);
  bool set_Tracking(const interaction_msgs::msg::BodyInfo::SharedPtr msg);
  bool subscribeMap(const nav_msgs::msg::OccupancyGrid::SharedPtr occupancy_grid);
  bool subscribePosition(const SE3VelocityCMD_T::SharedPtr msg);
  bool subscribeVoiceprintResult(const interaction_msgs::msg::VoiceprintResult::SharedPtr msg);
  bool subscribeFaceResult(const interaction_msgs::msg::FaceResult::SharedPtr msg);
  bool SetHeartBeat(std::string ip);
  bool subscribeOdomOut(const nav_msgs::msg::Odometry::SharedPtr msg);
  bool subscribeObstacleDetection(const ception_msgs::msg::Around::SharedPtr msg);
  bool subscribeDogPose(const motion_msgs::msg::SE3Pose::SharedPtr msg);
  bool subscribeGpsScene(const motion_msgs::msg::Scene::SharedPtr msg);
  bool subscribeRemoteEvent(const ception_msgs::msg::BtRemoteEvent::SharedPtr msg);
  bool subscribePath(const nav_msgs::msg::Path::SharedPtr msg);

private:
  void set_bms_(const std::shared_ptr<ception_msgs::msg::Bms::SharedPtr> msg);
  void SetStatus_(const std::shared_ptr<ControlState_T::SharedPtr> msg);
  void set_rssi_(const std::shared_ptr<std_msgs::msg::String::SharedPtr> msg);
  void subscribeTrackingStatus_(
    const std::shared_ptr<automation_msgs::msg::TrackingStatus::SharedPtr> msg);
  void subscribeNavStatus_(const std::shared_ptr<automation_msgs::msg::Caution::SharedPtr> status);
  void set_BodySelect_(const std::shared_ptr<interaction_msgs::msg::BodyInfo::SharedPtr> msg);
  void set_Tracking_(const std::shared_ptr<interaction_msgs::msg::BodyInfo::SharedPtr> msg);
  void subscribeMap_(const std::shared_ptr<nav_msgs::msg::OccupancyGrid::SharedPtr> occupancy_grid);
  void subscribePosition_(const std::shared_ptr<SE3VelocityCMD_T::SharedPtr> msg);
  void subscribeVoiceprintResult_(
    const std::shared_ptr<interaction_msgs::msg::VoiceprintResult::SharedPtr> msg);
  void subscribeFaceResult_(
    const std::shared_ptr<interaction_msgs::msg::FaceResult::SharedPtr> msg);
  void subscribeOdomOut_(const std::shared_ptr<nav_msgs::msg::Odometry::SharedPtr> msg);
  void subscribeObstacleDetection_(const std::shared_ptr<ception_msgs::msg::Around::SharedPtr> msg);
  void subscribeDogPose_(const std::shared_ptr<motion_msgs::msg::SE3Pose::SharedPtr> msg);
  void subscribeGpsScene_(const std::shared_ptr<motion_msgs::msg::Scene::SharedPtr> msg);
  void subscribeRemoteEvent_(
    const std::shared_ptr<ception_msgs::msg::BtRemoteEvent::SharedPtr> msg);
  void subscribePath_(const std::shared_ptr<nav_msgs::msg::Path::SharedPtr> msg);
  void generate_grpc_Header(
    cyberdogapp::Header & grpc_header,
    const std_msgs::msg::Header & std_header);

private:
  std::unique_ptr<cyberdogapp::CyberdogApp::Stub> stub_;
  std::shared_ptr<grpc::Channel> channel_;
  LatestMsgDispather<std_msgs::msg::String::SharedPtr> rssi_dispatcher;
  LatestMsgDispather<ception_msgs::msg::Bms::SharedPtr> bms_dispatcher;
  LatestMsgDispather<ControlState_T::SharedPtr> status_dispatcher;
  LatestMsgDispather<automation_msgs::msg::TrackingStatus::SharedPtr> TrackingStatus_dispatcher;
  LatestMsgDispather<automation_msgs::msg::Caution::SharedPtr> NavStatus_dispatcher;
  LatestMsgDispather<interaction_msgs::msg::BodyInfo::SharedPtr> bodySelect_dispatcher;
  LatestMsgDispather<interaction_msgs::msg::BodyInfo::SharedPtr> Tracking_dispatcher;
  LatestMsgDispather<nav_msgs::msg::OccupancyGrid::SharedPtr> map_dispatcher;

  LatestMsgDispather<SE3VelocityCMD_T::SharedPtr> subscribePosition_dispatcher;
  LatestMsgDispather<interaction_msgs::msg::VoiceprintResult::SharedPtr>
  subscribeVoiceprintResult_dispatcher;
  LatestMsgDispather<interaction_msgs::msg::FaceResult::SharedPtr> subscribeFaceResult_dispatcher;
  LatestMsgDispather<nav_msgs::msg::Odometry::SharedPtr> subscribeOdomOut_dispatcher;
  LatestMsgDispather<ception_msgs::msg::Around::SharedPtr> subscribeObstacleDetection_dispatcher;
  LatestMsgDispather<motion_msgs::msg::SE3Pose::SharedPtr> subscribeDogPose_dispatcher;
  LatestMsgDispather<motion_msgs::msg::Scene::SharedPtr> subscribeGpsScene_dispatcher;
  LatestMsgDispather<ception_msgs::msg::BtRemoteEvent::SharedPtr> remoteEvent_dispatcher;
  LatestMsgDispather<nav_msgs::msg::Path::SharedPtr> path_dispatcher;
};

#endif  // CYBERDOG_APP_CLIENT_HPP_
