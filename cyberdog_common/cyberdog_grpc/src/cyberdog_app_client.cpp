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


#include "cyberdog_app_client.hpp"
#include <memory>
#include <string>
#include <vector>

using cyberdogapp::Ticks;
using cyberdogapp::ErrorFlag;
using cyberdogapp::WifiRssi;
using cyberdogapp::TrackingStatus;
using cyberdogapp::Caution;
using cyberdogapp::BodyInfo;
using cyberdogapp::RegionOfInterest;
using cyberdogapp::Header;
using cyberdogapp::OccupancyGrid;
using cyberdogapp::Timestamp;
using cyberdogapp::MapMetaData;
using cyberdogapp::Pose;
using cyberdogapp::Quaternion;
using cyberdogapp::Point;
using cyberdogapp::DecisionStamped;
using cyberdogapp::Twist;
using cyberdogapp::Safety;
using cyberdogapp::Ultrasonic;
using cyberdogapp::Range;
using cyberdogapp::DogPose;
using cyberdogapp::Timestamp;
using cyberdogapp::RemoteEvent;
using cyberdogapp::Body;
using cyberdogapp::Vector3;
using cyberdogapp::FaceResult;
using cyberdogapp::VoiceprintResult;
using cyberdogapp::Odometry;
using cyberdogapp::Freameid;
using cyberdogapp::Around;
using cyberdogapp::Path;
using cyberdogapp::Scene;
using cyberdogapp::PoseStamped;

using std::placeholders::_1;
Cyberdog_App_Client::Cyberdog_App_Client(std::shared_ptr<Channel> channel)
: stub_(cyberdogapp::CyberdogApp::NewStub(channel))
{
  rssi_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::set_rssi_, this, _1));
  bms_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::set_bms_, this, _1));
  status_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::SetStatus_, this, _1));
  TrackingStatus_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribeTrackingStatus_,
      this, _1));
  NavStatus_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::subscribeNavStatus_, this, _1));
  bodySelect_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::set_BodySelect_, this, _1));
  Tracking_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::set_Tracking_, this, _1));
  map_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::subscribeMap_, this, _1));
  subscribePosition_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribePosition_, this,
      _1));
  subscribeVoiceprintResult_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::
      subscribeVoiceprintResult_, this, _1));
  subscribeFaceResult_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribeFaceResult_,
      this, _1));
  subscribeOdomOut_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribeOdomOut_, this,
      _1));
  subscribeObstacleDetection_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::
      subscribeObstacleDetection_, this, _1));
  subscribeDogPose_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribeDogPose_, this,
      _1));
  subscribeGpsScene_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribeGpsScene_, this,
      _1));
  remoteEvent_dispatcher.setCallback(
    std::bind(
      &Cyberdog_App_Client::subscribeRemoteEvent_, this,
      _1));
  path_dispatcher.setCallback(std::bind(&Cyberdog_App_Client::subscribePath_, this, _1));
}
Cyberdog_App_Client::~Cyberdog_App_Client()
{}

bool Cyberdog_App_Client::SetHeartBeat(std::string ip)
{
  ClientContext context;
  gpr_timespec timespec;
  timespec.tv_sec = 1;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Result result;
  Ticks ticks_;
  // std::cout << "before SetHeartBeat." << std::endl
  ticks_.set_ip(ip);
  Status status = stub_->heartbeat(&context, ticks_, &result);
  if (!status.ok()) {
    std::cout << "SetHeartBeat error code: " << status.error_code() << std::endl;
    return false;
  }
  std::cout << "SetHeartBeat rpc success." << std::endl;
  return true;
}

bool Cyberdog_App_Client::SetBms(const ception_msgs::msg::Bms::SharedPtr msg)
{
  bms_dispatcher.push(msg);
  return true;
}
bool Cyberdog_App_Client::SetStatus(const ControlState_T::SharedPtr msg)
{
  status_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::set_rssi(const std_msgs::msg::String::SharedPtr msg)
{
  rssi_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeTrackingStatus(
  const automation_msgs::msg::TrackingStatus::SharedPtr msg)
{
  TrackingStatus_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeNavStatus(const automation_msgs::msg::Caution::SharedPtr msg)
{
  NavStatus_dispatcher.push(msg);
  return true;
}
bool Cyberdog_App_Client::set_BodySelect(const interaction_msgs::msg::BodyInfo::SharedPtr msg)
{
  bodySelect_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::set_Tracking(const interaction_msgs::msg::BodyInfo::SharedPtr msg)
{
  Tracking_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  map_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribePosition(const SE3VelocityCMD_T::SharedPtr msg)
{
  subscribePosition_dispatcher.push(msg);
  return true;
}


bool Cyberdog_App_Client::subscribeVoiceprintResult(
  const interaction_msgs::msg::VoiceprintResult::SharedPtr msg)
{
  subscribeVoiceprintResult_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeFaceResult(
  const interaction_msgs::msg::FaceResult::SharedPtr msg)
{
  subscribeFaceResult_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeOdomOut(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  subscribeOdomOut_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeObstacleDetection(const ception_msgs::msg::Around::SharedPtr msg)
{
  subscribeObstacleDetection_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeDogPose(const motion_msgs::msg::SE3Pose::SharedPtr msg)
{
  subscribeDogPose_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeGpsScene(const motion_msgs::msg::Scene::SharedPtr msg)
{
  subscribeGpsScene_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribeRemoteEvent(
  const ception_msgs::msg::BtRemoteEvent::SharedPtr msg)
{
  remoteEvent_dispatcher.push(msg);
  return true;
}

bool Cyberdog_App_Client::subscribePath(const nav_msgs::msg::Path::SharedPtr msg)
{
  path_dispatcher.push(msg);
  return true;
}
// ========================
void Cyberdog_App_Client::set_bms_(const std::shared_ptr<ception_msgs::msg::Bms::SharedPtr> bms)
{
  Bms bms_send;
  bms_send.set_batt_soc(bms->get()->batt_soc);
  bms_send.set_status(bms->get()->status);
  ClientContext context;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Result result;
  Status status = stub_->subscribeBms(&context, bms_send, &result);
  if (!status.ok()) {
    // std::cout << "set_bms_ rpc failed." << std::endl;
    return;
  }
  // std::cout << "set_bms_ rpc success: " << bms->get()->batt_soc << std::endl;
}

void Cyberdog_App_Client::SetStatus_(const std::shared_ptr<ControlState_T::SharedPtr> msg)
{
  StatusStamped status_send;
  cyberdogapp::RawStatus rawstatus;
  cyberdogapp::Mode mode;
  cyberdogapp::Pattern pattern;
  cyberdogapp::Pattern cached_pattern;
  cyberdogapp::TwistWithCovariance twistC;
  cyberdogapp::Twist twist;
  cyberdogapp::Vector3 liner;
  cyberdogapp::Vector3 anger;

  cyberdogapp::PoseWithCovariance poseC;
  cyberdogapp::Pose pose;
  cyberdogapp::Point point;
  cyberdogapp::Quaternion orientation;

  cyberdogapp::Parameters parameter;
  cyberdogapp::Safety safety;
  cyberdogapp::Scene scene;
  cyberdogapp::MonOrder order;

  ErrorFlag error_flag;

  mode.set_control_mode(msg->get()->modestamped.control_mode);
  mode.set_mode_type(msg->get()->modestamped.mode_type);
  rawstatus.mutable_mode()->CopyFrom(mode);

  pattern.set_gait_pattern(msg->get()->gaitstamped.gait);

  cached_pattern.set_gait_pattern(msg->get()->cached_gait.gait);
  rawstatus.mutable_pattern()->CopyFrom(pattern);
  rawstatus.mutable_cached_pattern()->CopyFrom(cached_pattern);
  anger.set_x(msg->get()->velocitystamped.angular_x);
  anger.set_y(msg->get()->velocitystamped.angular_y);
  anger.set_z(msg->get()->velocitystamped.angular_z);
  liner.set_x(msg->get()->velocitystamped.linear_x);
  liner.set_y(msg->get()->velocitystamped.linear_y);
  liner.set_z(msg->get()->velocitystamped.linear_z);

  twist.mutable_angular()->CopyFrom(anger);
  twist.mutable_linear()->CopyFrom(liner);
  twistC.mutable_twist()->CopyFrom(twist);
  /*
  for (int i = 0; i < msg->get()->status.twist.covariance.size(); i++)
  {
    twistC.add_covariance(msg->get()->status.twist.covariance[i]);
  }
  */
  rawstatus.mutable_twist()->CopyFrom(twistC);

  point.set_x(msg->get()->posestamped.position_x);
  point.set_y(msg->get()->posestamped.position_y);
  point.set_z(msg->get()->posestamped.position_z);

  orientation.set_w(msg->get()->posestamped.rotation_w);
  orientation.set_x(msg->get()->posestamped.rotation_x);
  orientation.set_y(msg->get()->posestamped.rotation_y);
  orientation.set_z(msg->get()->posestamped.rotation_z);

  pose.mutable_position()->CopyFrom(point);
  pose.mutable_orientation()->CopyFrom(orientation);
  poseC.mutable_pose()->CopyFrom(pose);
  /*
  for (int i = 0; i < msg->get()->status.pose.covariance.size(); i++)
  {
    poseC.add_covariance(msg->get()->status.pose.covariance[i]);
  }
  */
  rawstatus.mutable_pose()->CopyFrom(poseC);

  parameter.set_body_height(msg->get()->parastamped.body_height);
  parameter.set_gait_height(msg->get()->parastamped.gait_height);
  rawstatus.mutable_para()->CopyFrom(parameter);

  safety.set_status(msg->get()->safety.status);
  rawstatus.mutable_safety()->CopyFrom(safety);

  scene.set_lat(msg->get()->scene.lat);
  scene.set_lon(msg->get()->scene.lon);
  scene.set_type(msg->get()->scene.type);
  rawstatus.mutable_scene()->CopyFrom(scene);

  order.set_id(msg->get()->orderstamped.id);
  order.set_para(msg->get()->orderstamped.para);
  rawstatus.set_foot_contact(msg->get()->foot_contact);
  rawstatus.mutable_order()->CopyFrom(order);

  error_flag.set_exist_error(msg->get()->error_flag.exist_error);
  error_flag.set_ori_error(msg->get()->error_flag.ori_error);
  error_flag.set_footpos_error(msg->get()->error_flag.footpos_error);
  int cnt = msg->get()->error_flag.motor_error.size();
  for (int i = 0; i < cnt; i++) {
    error_flag.add_motor_error(msg->get()->error_flag.motor_error[i]);
  }
  rawstatus.mutable_error_flag()->CopyFrom(error_flag);

  status_send.mutable_status()->CopyFrom(rawstatus);


  ClientContext context;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Result result;
  Status status = stub_->subscribeStatus(&context, status_send, &result);
  if (!status.ok()) {
    // std::cout << "SetStatus rpc failed." << std::endl;
    return;
  }
  // std::cout << "SetStatus rpc success." << std::endl;
}

void Cyberdog_App_Client::set_rssi_(std::shared_ptr<std_msgs::msg::String::SharedPtr> msg)
{
  ClientContext context;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Result result;
  WifiRssi rssi;
  rssi.set_rssi(msg->get()->data);
  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeWifiRssi(&context, rssi, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "set_rssi rpc failed." << std::endl;
    return;
  }
  // std::cout << "set_rssi rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeTrackingStatus_(
  const std::shared_ptr<automation_msgs::msg::TrackingStatus::SharedPtr> msg)
{
  ClientContext context;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Result result;
  TrackingStatus trackingstate;
  trackingstate.set_status(msg->get()->status);
  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeTrackingStatus(&context, trackingstate, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "set_TrackingStatus rpc failed." << std::endl;
    return;
  }
  // std::cout << "set_TrackingStatus rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeNavStatus_(
  const std::shared_ptr<automation_msgs::msg::Caution::SharedPtr> msg)
{
  ClientContext context;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Result result;
  Caution caution;
  caution.set_error_type(msg->get()->error_type);
  caution.set_robot_mode(msg->get()->robot_mode);
  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeNavStatus(&context, caution, &result);
  std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeNavStatus rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeNavStatus rpc success." << std::endl;
}
void Cyberdog_App_Client::set_BodySelect_(
  const std::shared_ptr<interaction_msgs::msg::BodyInfo::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  BodyInfo bodyinfo;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  bodyinfo.set_count(msg->get()->count);
  for (int i = 0; i < msg->get()->count; ++i) {
    Body * body = bodyinfo.add_infos();
    RegionOfInterest roi;
    roi.set_height(msg->get()->infos[i].roi.height);
    roi.set_width(msg->get()->infos[i].roi.width);
    roi.set_x_offset(msg->get()->infos[i].roi.x_offset);
    roi.set_y_offset(msg->get()->infos[i].roi.y_offset);
    roi.set_do_rectify(msg->get()->infos[i].roi.do_rectify);
    body->mutable_roi()->CopyFrom(roi);
  }

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeBodySelect(&context, bodyinfo, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "set_BodySelect rpc failed." << std::endl;
    return;
  }
  // std::cout << "set_BodySelect rpc success." << std::endl;
}

void Cyberdog_App_Client::set_Tracking_(
  const std::shared_ptr<interaction_msgs::msg::BodyInfo::SharedPtr> msg)
{
  ClientContext context;
  Header header;
  Result result;
  BodyInfo bodyinfo;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  bodyinfo.set_count(msg->get()->count);

  for (int i = 0; i < msg->get()->count; ++i) {
    Body * body = bodyinfo.add_infos();
    RegionOfInterest roi;
    roi.set_height(msg->get()->infos[i].roi.height);
    roi.set_width(msg->get()->infos[i].roi.width);
    roi.set_x_offset(msg->get()->infos[i].roi.x_offset);
    roi.set_y_offset(msg->get()->infos[i].roi.y_offset);
    roi.set_do_rectify(msg->get()->infos[i].roi.do_rectify);
    body->mutable_roi()->CopyFrom(roi);
  }

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeTracking(&context, bodyinfo, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "set_Tracking rpc failed." << std::endl;
    return;
  }
  // std::cout << "set_Tracking rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeMap_(
  const std::shared_ptr<nav_msgs::msg::OccupancyGrid::SharedPtr> occupancy_grid)
{
  ClientContext context;
  Header header;
  Timestamp stamp;
  OccupancyGrid occupacy;
  Result result;
  //  header
  generate_grpc_Header(header, occupancy_grid->get()->header);

  // info
  MapMetaData info;
  Timestamp map_load_time;
  Pose pose;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);

  map_load_time.set_sec(occupancy_grid->get()->info.map_load_time.sec);
  map_load_time.set_nanosec(occupancy_grid->get()->info.map_load_time.nanosec);

  Point point;
  Quaternion orientation;
  point.set_x(occupancy_grid->get()->info.origin.position.x);
  point.set_y(occupancy_grid->get()->info.origin.position.y);
  point.set_z(occupancy_grid->get()->info.origin.position.z);

  orientation.set_x(occupancy_grid->get()->info.origin.orientation.x);
  orientation.set_y(occupancy_grid->get()->info.origin.orientation.y);
  orientation.set_z(occupancy_grid->get()->info.origin.orientation.z);
  orientation.set_w(occupancy_grid->get()->info.origin.orientation.w);
  pose.mutable_orientation()->CopyFrom(orientation);
  pose.mutable_position()->CopyFrom(point);

  info.set_height(occupancy_grid->get()->info.height);
  info.set_width(occupancy_grid->get()->info.width);
  info.set_resolution(occupancy_grid->get()->info.resolution);
  info.mutable_map_load_time()->CopyFrom(map_load_time);
  info.mutable_origin()->CopyFrom(pose);
  occupacy.mutable_info()->CopyFrom(info);

  //  data
  int cnt = occupancy_grid->get()->data.size();
  for (int i = 0; i < cnt; i++) {
    occupacy.add_data(occupancy_grid->get()->data[i]);
  }
  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeMap(&context, occupacy, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeMap rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeMap rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribePosition_(const std::shared_ptr<SE3VelocityCMD_T::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  Header header;
  DecisionStamped decision_stamped;
  Decissage decissage;
  Twist twist;
  Vector3 linear;
  Vector3 angular;
  Pose pose;
  Point position;
  Quaternion orientation;
  Safety safety;

  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  //  header
/*
  generate_grpc_Header(header, msg->get()->header);
*/
  //  twist
  linear.set_x(msg->get()->velocity.linear_x);
  linear.set_y(msg->get()->velocity.linear_y);
  linear.set_z(msg->get()->velocity.linear_z);

  angular.set_x(msg->get()->velocity.angular_x);
  angular.set_y(msg->get()->velocity.angular_y);
  angular.set_z(msg->get()->velocity.angular_z);

  twist.mutable_linear()->CopyFrom(linear);
  twist.mutable_angular()->CopyFrom(angular);
/*
  pose.mutable_orientation()->CopyFrom(orientation);
  pose.mutable_position()->CopyFrom(position);

  // safety
  safety.set_status(msg->get()->decissage.safety.status);

*/
  decissage.mutable_twist()->CopyFrom(twist);
/*
  decissage.mutable_pose()->CopyFrom(pose);
  decissage.mutable_safety()->CopyFrom(safety);

  decision_stamped.mutable_header()->CopyFrom(header);
*/
  decision_stamped.mutable_decissage()->CopyFrom(decissage);
  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribePosition(&context, decision_stamped, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribePosition rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribePosition rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeVoiceprintResult_(
  const std::shared_ptr<interaction_msgs::msg::VoiceprintResult::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  Header header;
  VoiceprintResult vpr;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  generate_grpc_Header(header, msg->get()->header);
  vpr.mutable_header()->CopyFrom(header);
  vpr.set_error(msg->get()->error);
  vpr.set_succeed(msg->get()->succeed);
  vpr.set_type(msg->get()->type);

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeVoiceprintResult(&context, vpr, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeVoiceprintResult rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeVoiceprintResult rpc success." << std::endl;
}
void Cyberdog_App_Client::subscribeFaceResult_(
  const std::shared_ptr<interaction_msgs::msg::FaceResult::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  FaceResult faceresult;
  Header header;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  //  generate_grpc_Header(header, msg->get()->header);
  faceresult.set_msg(msg->get()->msg);
  faceresult.set_result(msg->get()->result);

  //  constuct face_images
  std::vector<interaction_msgs::msg::CompressedImage> face_images = msg->get()->face_images;
  int cnt = face_images.size();
  for (int i = 0; i < cnt; ++i) {
    ::cyberdogapp::CompressedImage * img = faceresult.add_face_images();
    ::cyberdogapp::Header header;
    ::cyberdogapp::Timestamp time;

    //  construct format
    img->set_format(face_images[i].format);

    //  construct header
    time.set_sec(face_images[i].header.stamp.sec);
    time.set_nanosec(face_images[i].header.stamp.nanosec);
    header.set_frame_id(face_images[i].header.frame_id);
    header.mutable_stamp()->CopyFrom(time);
    img->mutable_header()->CopyFrom(header);

    // construct data
    for (int j = 0; j < face_images[i].data.size(); j++) {
      img->add_data(face_images[i].data[j]);
    }
  }

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeFaceResult(&context, faceresult, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeFaceResult rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeFaceResult rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeOdomOut_(
  const std::shared_ptr<nav_msgs::msg::Odometry::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  Header header;
  Odometry odo;
  generate_grpc_Header(header, msg->get()->header);
  odo.set_child_frame_id(msg->get()->child_frame_id);
  odo.mutable_header()->CopyFrom(header);
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  // pose
  cyberdogapp::PoseWithCovariance poseC;
  cyberdogapp::Pose pose;
  cyberdogapp::Quaternion orientation;
  cyberdogapp::Point point;

  point.set_x(msg->get()->pose.pose.position.x);
  point.set_y(msg->get()->pose.pose.position.y);
  point.set_z(msg->get()->pose.pose.position.z);

  orientation.set_w(msg->get()->pose.pose.orientation.w);
  orientation.set_x(msg->get()->pose.pose.orientation.x);
  orientation.set_y(msg->get()->pose.pose.orientation.y);
  orientation.set_z(msg->get()->pose.pose.orientation.z);

  pose.mutable_position()->CopyFrom(point);
  pose.mutable_orientation()->CopyFrom(orientation);
  poseC.mutable_pose()->CopyFrom(pose);

  for (int i = 0; i < msg->get()->pose.covariance.size(); i++) {
    poseC.add_covariance(msg->get()->pose.covariance[i]);
  }
  odo.mutable_pose()->CopyFrom(poseC);
  // twist
  cyberdogapp::TwistWithCovariance twistC;
  cyberdogapp::Twist twist;
  cyberdogapp::Vector3 liner;
  cyberdogapp::Vector3 anger;
  anger.set_x(msg->get()->twist.twist.angular.x);
  anger.set_y(msg->get()->twist.twist.angular.y);
  anger.set_z(msg->get()->twist.twist.angular.z);
  liner.set_x(msg->get()->twist.twist.linear.x);
  liner.set_y(msg->get()->twist.twist.linear.y);
  liner.set_z(msg->get()->twist.twist.linear.z);

  twist.mutable_angular()->CopyFrom(anger);
  twist.mutable_linear()->CopyFrom(liner);
  twistC.mutable_twist()->CopyFrom(twist);

  for (int i = 0; i < msg->get()->twist.covariance.size(); i++) {
    twistC.add_covariance(msg->get()->twist.covariance[i]);
  }
  odo.mutable_twist()->CopyFrom(twistC);

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeOdomOut(&context, odo, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeOdomOut rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeOdomOut rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeObstacleDetection_(
  const std::shared_ptr<ception_msgs::msg::Around::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  Around around;
  Ultrasonic ultrasoinc;
  Range range;
  Header header;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  //  construct range
  generate_grpc_Header(header, msg->get()->front_distance.range_info.header);
  range.mutable_header()->CopyFrom(header);
  range.set_radiation_type(msg->get()->front_distance.range_info.radiation_type);
  range.set_field_of_view(msg->get()->front_distance.range_info.field_of_view);
  range.set_min_range(msg->get()->front_distance.range_info.min_range);
  range.set_max_range(msg->get()->front_distance.range_info.max_range);
  range.set_range(msg->get()->front_distance.range_info.range);

  ultrasoinc.mutable_range_info()->CopyFrom(range);
  around.mutable_front_distance()->CopyFrom(ultrasoinc);

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeObstacleDetection(&context, around, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeObstacleDetection rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeObstacleDetection rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeDogPose_(
  const std::shared_ptr<motion_msgs::msg::SE3Pose::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Freameid freameid;
  DogPose dog_pose;
  Timestamp timestamp;


  freameid.set_id(msg->get()->frameid.id);
  timestamp.set_sec(msg->get()->timestamp.sec);
  timestamp.set_nanosec(msg->get()->timestamp.nanosec);

  dog_pose.set_position_x(msg->get()->position_x);
  dog_pose.set_position_y(msg->get()->position_y);
  dog_pose.set_position_z(msg->get()->position_z);

  dog_pose.set_rotation_w(msg->get()->rotation_w);
  dog_pose.set_rotation_x(msg->get()->rotation_x);
  dog_pose.set_rotation_y(msg->get()->rotation_y);
  dog_pose.set_rotation_z(msg->get()->rotation_z);

  dog_pose.mutable_timestamp()->CopyFrom(timestamp);
  dog_pose.mutable_frameid()->CopyFrom(freameid);

  // std::cout << "before sub" << __func__ << std::endl;
  Status status = stub_->subscribeDogPose(&context, dog_pose, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeDogPose rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeDogPose rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeGpsScene_(
  const std::shared_ptr<motion_msgs::msg::Scene::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  Scene scene;

  scene.set_type(msg->get()->type);
  scene.set_lat(msg->get()->lat);
  scene.set_lon(msg->get()->lon);
  scene.set_if_danger(msg->get()->if_danger);

  Status status = stub_->subscribeGpsScene(&context, scene, &result);
  // std::cout << "after sub" << __func__ << std::endl;
  if (!status.ok()) {
    // std::cout << "subscribeGpsScene rpc failed." << std::endl;
    return;
  }
  // std::cout << "subscribeGpsScene rpc success." << std::endl;
}

void Cyberdog_App_Client::subscribeRemoteEvent_(
  const std::shared_ptr<ception_msgs::msg::BtRemoteEvent::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);
  RemoteEvent event;

  event.set_scan_status(msg->get()->scan_status);
  event.set_remote_status(msg->get()->remote_status);
  event.set_address(msg->get()->address);
  event.set_scan_device_info(msg->get()->scan_device_info);
  event.set_error(msg->get()->error);

  Status status = stub_->subscribeRemoteEvent(&context, event, &result);
}

void Cyberdog_App_Client::subscribePath_(const std::shared_ptr<nav_msgs::msg::Path::SharedPtr> msg)
{
  ClientContext context;
  Result result;
  gpr_timespec timespec;
  timespec.tv_sec = 2;
  timespec.tv_nsec = 0;
  timespec.clock_type = GPR_TIMESPAN;
  context.set_deadline(timespec);

  Path path;
  Header header;
  generate_grpc_Header(header, msg->get()->header);
  int cnt = msg->get()->poses.size();
  for (int i = 0; i < cnt; i++) {
    Header header;
    Point position;
    Quaternion orientation;
    Pose pose;
    PoseStamped * posestamped = path.add_posestamped();
    generate_grpc_Header(header, msg->get()->poses[i].header);

    position.set_x(msg->get()->poses[i].pose.position.x);
    position.set_y(msg->get()->poses[i].pose.position.y);
    position.set_z(msg->get()->poses[i].pose.position.z);

    orientation.set_w(msg->get()->poses[i].pose.orientation.w);
    orientation.set_x(msg->get()->poses[i].pose.orientation.x);
    orientation.set_y(msg->get()->poses[i].pose.orientation.y);
    orientation.set_z(msg->get()->poses[i].pose.orientation.z);


    pose.mutable_position()->CopyFrom(position);
    pose.mutable_orientation()->CopyFrom(orientation);

    posestamped->mutable_pose()->CopyFrom(pose);
    posestamped->mutable_header()->CopyFrom(header);
  }

  Status status = stub_->subscribePath(&context, path, &result);
}

void Cyberdog_App_Client::generate_grpc_Header(
  Header & grpc_header,
  const std_msgs::msg::Header & std_header)
{
  Timestamp stamp;
  stamp.set_sec(std_header.stamp.sec);
  stamp.set_nanosec(std_header.stamp.nanosec);
  grpc_header.mutable_stamp()->CopyFrom(stamp);
  grpc_header.set_frame_id(std_header.frame_id);
}
