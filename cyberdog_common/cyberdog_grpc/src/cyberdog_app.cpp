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


#include "cyberdog_app.hpp"
#include <sys/types.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <linux/if.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "cyberdog_app_server.hpp"
#include "std_msgs/msg/string.hpp"

#define gettid() syscall(SYS_gettid)

using grpc::ServerWriter;
#define RECV_URL "udpm://239.255.76.67:7667?ttl=1"
#define SEND_URL "udpm://239.255.76.67:7667?ttl=2"
#define DES_V_MAX 3.0
#define DES_V_MIN -3.0
#define DES_ANG_R_MAX 0.5
#define DES_ANG_R_MIN -0.5
#define DES_ANG_P_MAX 0.5
#define DES_ANG_P_MIN -0.5
#define DES_ANG_Y_MAX 3.0
#define DES_ANG_Y_MIN -3.0
#define DES_BODY_H_MAX 0.5
#define DES_BODY_H_MIN 0.0
#define DES_GAIT_H_MAX 0.2
#define DES_GAIT_H_MIN 0.0
#define SEND_CMD_TTL 2
#define RECV_CMD_TTL 12
#define APP_CONNECTED_FAIL_CNT 3
using std::placeholders::_1;
using namespace std::chrono_literals;
using SE3VelocityCMD_T = motion_msgs::msg::SE3VelocityCMD;
#define CON_TO_CHAR(a) (reinterpret_cast<const char *>(a))
namespace cyberdog_cyberdog_app
{
static int64_t requestNumber;
Cyberdog_app::Cyberdog_app()
: Node("motion_test_server"), is_mode_check(false), patternCheckout_client_(NULL),
  ModeCheckout_client_(NULL), ticks_(0),
  can_process_messages(false), wait_for_set_data_result(false), wait_for_offset_data(false),
  heartbeat_err_cnt(0), heart_beat_thread_(nullptr), app_server_thread_(nullptr), server_(nullptr),
  app_stub(nullptr), app_disconnected(false), destory_grpc_server_thread_(nullptr),
  change_mode_id(0), change_gait_id(0), ext_mon_id(0)
{
  RCLCPP_INFO(get_logger(), "Cyberdog_app Configuring");
  server_ip = std::make_shared<std::string>("0.0.0.0");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  modeRespond_queue =
    std::make_shared<threadsafe_queue<std::shared_ptr<::cyberdogapp::CheckoutMode_respond>>>();
  patternRespond_queue =
    std::make_shared<threadsafe_queue<std::shared_ptr<::cyberdogapp::CheckoutPattern_respond>>>();
  extMonOrderRespond_queue =
    std::make_shared<threadsafe_queue<std::shared_ptr<::cyberdogapp::ExtMonOrder_Respond>>>();

  offsetCalibationData_queue =
    std::make_shared<threadsafe_queue<std::shared_ptr<::cyberdogapp::OffsetCalibationData>>>();
  offsetRequestResult_queue =
    std::make_shared<threadsafe_queue<std::shared_ptr<::cyberdogapp::OffsetRequest_result>>>();
  offsetCalibationData_queue->set_time_out(3);
  offsetRequestResult_queue->set_time_out(3);

  camera_client_ = this->create_client<interaction_msgs::srv::CameraService>(
    "camera_service",
    rmw_qos_profile_services_default,
    callback_group_);
  tracking_object_client_ = this->create_client<interaction_msgs::srv::BodyRegion>(
    "tracking_object", rmw_qos_profile_services_default, callback_group_);
  voice_print_client_ = this->create_client<interaction_msgs::srv::Voiceprint>(
    "voiceprint",
    rmw_qos_profile_services_default,
    callback_group_);
  nav_position_client_ = this->create_client<automation_msgs::srv::Target>(
    "nav_target",
    rmw_qos_profile_services_default,
    callback_group_);
  ai_token_client_ = this->create_client<interaction_msgs::srv::TokenPass>(
    "token_update",
    rmw_qos_profile_services_default,
    callback_group_);
  face_manager_client_ = this->create_client<interaction_msgs::srv::FaceManager>(
    "face_manager",
    rmw_qos_profile_services_default,
    callback_group_);

  remote_manager_client_ = this->create_client<ception_msgs::srv::BtRemoteCommand>(
    "btRemoteCommand",
    rmw_qos_profile_services_default,
    callback_group_);

  para_pub = this->create_publisher<Parameters_T>("para_change", rclcpp::SensorDataQoS());
  decision_pub = this->create_publisher<SE3VelocityCMD_T>("body_cmd", rclcpp::SystemDefaultsQoS());
  action_request_pub = this->create_publisher<motion_msgs::msg::ActionRequest>(
    "cyberdog_action",
    rclcpp::SystemDefaultsQoS());
  action_result_sub = this->create_subscription<motion_msgs::msg::ActionRespond>(
    "cyberdog_action_result",
    rclcpp::SystemDefaultsQoS(),
    std::bind(&Cyberdog_app::subscribeActionResult, this, _1));
  rclcpp::ServicesQoS pub_qos;
  pub_qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);

  ip_subscriber = this->create_subscription<std_msgs::msg::String>(
    "ip_notify", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribeIp, this, _1));

  Bms_subscriber = this->create_subscription<ception_msgs::msg::Bms>(
    "bms_recv", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::set_bms, this, _1));

  Status_subscriber = this->create_subscription<ControlState_T>(
    "status_out", pub_qos, std::bind(&Cyberdog_app::set_status, this, _1));

  wifi_rssi = this->create_subscription<std_msgs::msg::String>(
    "wifi_rssi", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::set_rssi, this, _1));

  body_select = this->create_subscription<interaction_msgs::msg::BodyInfo>(
    "body", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::set_body_select, this, _1));

  tracking = this->create_subscription<interaction_msgs::msg::BodyInfo>(
    "tracking_result", pub_qos, std::bind(&Cyberdog_app::set_tracking, this, _1));

  map = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    "map", 1, std::bind(&Cyberdog_app::subscribeMap, this, _1));

  nav_odom = this->create_subscription<SE3VelocityCMD_T>(
    "body_cmd", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribePosition, this, _1));

  voice_print_result = this->create_subscription<interaction_msgs::msg::VoiceprintResult>(
    "voiceprint_result", rclcpp::SystemDefaultsQoS(),
    std::bind(&Cyberdog_app::subscribeVoiceprintResult, this, _1));

  face_result = this->create_subscription<interaction_msgs::msg::FaceResult>(
    "/face_manager/result", rclcpp::SystemDefaultsQoS(),
    std::bind(&Cyberdog_app::subscribeFaceResult, this, _1));

  tracking_status = this->create_subscription<automation_msgs::msg::TrackingStatus>(
    "tracking_status", pub_qos, std::bind(&Cyberdog_app::subscribeTrackingStatus, this, _1));

  nav_status = this->create_subscription<automation_msgs::msg::Caution>(
    "nav_status", rclcpp::SystemDefaultsQoS(), std::bind(
      &Cyberdog_app::subscribeNavStatus, this,
      _1));

  odom_out = this->create_subscription<nav_msgs::msg::Odometry>(
    "odom_out", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribeOdomOut, this, _1));

  obstacle_detection = this->create_subscription<ception_msgs::msg::Around>(
    "ObstacleDetection", rclcpp::SystemDefaultsQoS(),
    std::bind(&Cyberdog_app::subscribeObstacleDetection, this, _1));

  dog_pose = this->create_subscription<motion_msgs::msg::SE3Pose>(
    "dog_pose", rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribeDogPose, this, _1));

  scene_detection = this->create_subscription<motion_msgs::msg::Scene>(
    "SceneDetection",
    rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribeGpsScene, this, _1));

  remote_status = this->create_subscription<ception_msgs::msg::BtRemoteEvent>(
    "remoteEvent",
    rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribeRemoteEvent, this, _1));

  path_subscriber = this->create_subscription<nav_msgs::msg::Path>(
    "plan",
    rclcpp::SystemDefaultsQoS(), std::bind(&Cyberdog_app::subscribePath, this, _1));

  heart_beat_thread_ = std::make_shared<std::thread>(&Cyberdog_app::HeartBeat, this);

  offset_request = std::make_shared<lcm::LCM>(getLcmUrl(SEND_CMD_TTL));
  offset_response = std::make_shared<lcm::LCM>(getLcmUrl(RECV_CMD_TTL));
  offset_response->subscribe("interface_response", &Cyberdog_app::receive_lcm_data, this);
  lcm_handle_thread_ = std::make_shared<std::thread>(&Cyberdog_app::recv_lcm_handle, this);
}
void Cyberdog_app::recv_lcm_handle()
{
  rclcpp::WallRate r(50);
  while (0 == offset_response->handle()) {
    r.sleep();
  }
}

void Cyberdog_app::HeartBeat()
{
  rclcpp::WallRate r(500ms);
  std::string ipv4;
  while (true) {
    if (can_process_messages && app_stub) {
      if (!app_stub->SetHeartBeat(local_ip)) {
        if (heartbeat_err_cnt++ >= APP_CONNECTED_FAIL_CNT) {
          if (!app_disconnected) {
            destroyGrpc();
            createGrpc();
          }
        }
      }
    }
    r.sleep();
  }
}
void Cyberdog_app::RunServer()
{
  RCLCPP_INFO(get_logger(), "run_server thread id is %d", gettid());
  std::string server_address("0.0.0.0:50051");
  CyberdogAppImpl service(server_address);
  service.SetRequesProcess(this);
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 1000);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 1000);
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 500);
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_SENT_PING_INTERVAL_WITHOUT_DATA_MS, 1000);
  builder.RegisterService(&service);
  server_ = std::move(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  RCLCPP_INFO(get_logger(), "server thread id is %d", gettid());
  server_->Wait();
  RCLCPP_INFO(get_logger(), "after wait");
}

void Cyberdog_app::publishPattern(
  const ::cyberdogapp::CheckoutPattern_request * request,
  ::grpc::ServerWriter<::cyberdogapp::CheckoutPattern_respond> * writer)
{
  RCLCPP_INFO(get_logger(), "publishPattern");
  using namespace std::placeholders;
  motion_msgs::msg::ActionRequest request_;
  request_.type = motion_msgs::msg::ActionRequest::CHECKOUT_PATTERN;
  request_.gait.gait = request->patternstamped().pattern().gait_pattern();
  RCLCPP_INFO(get_logger(), "get check Pattern %d", request_.gait.gait);
  request_.request_id = ++change_gait_id;
  request_.timeout = request->timeout();
  action_request_pub->publish(request_);
  RCLCPP_INFO(get_logger(), "get check Pattern timeout seconds: %d", request->timeout());
  if (request->timeout() > 0) {
    patternRespond_queue->set_time_out(request->timeout());
  } else {
    patternRespond_queue->set_time_out(35);
  }
  while (true) {
    RCLCPP_INFO(get_logger(), "get check Pattern respond1");
    std::shared_ptr<std::shared_ptr<cyberdogapp::CheckoutPattern_respond>> respond =
      patternRespond_queue->wait_and_pop();
    RCLCPP_INFO(get_logger(), "get check Pattern respond2");
    if (respond == NULL) {
      RCLCPP_INFO(get_logger(), "get check Pattern timeout");
      return;
    }

    if (respond->get()->request_id() != request_.request_id) {
      RCLCPP_INFO(get_logger(), "get check Pattern invalid id");
      continue;
    }

    respond->get()->mutable_patternstamped()->CopyFrom(request->patternstamped());
    writer->Write(*(respond->get()));
    if (!respond->get()->is_feedback()) {
      RCLCPP_INFO(get_logger(), "get check Pattern result");
      patternRespond_queue->empty();
      return;
    }
  }
}

void Cyberdog_app::publishMode(
  const ::cyberdogapp::CheckoutMode_request * request,
  ::grpc::ServerWriter<::cyberdogapp::CheckoutMode_respond> * writer)
{
  RCLCPP_ERROR(get_logger(), "publishMode");
  motion_msgs::msg::Mode mode_out;
  mode_out.control_mode = request->next_mode().mode().control_mode();
  mode_out.mode_type = request->next_mode().mode().mode_type();
  motion_msgs::msg::ActionRequest request_;
  request_.type = 1;
  request_.mode = mode_out;
  request_.request_id = ++change_mode_id;
  request_.timeout = request->timeout();
  action_request_pub->publish(request_);
  RCLCPP_INFO(get_logger(), "get check mode timeout seconds: %d", request->timeout());
  if (request->timeout() > 0) {
    modeRespond_queue->set_time_out(request->timeout());
  } else {
    modeRespond_queue->set_time_out(35);
  }
  while (true) {
    RCLCPP_INFO(get_logger(), "get check mode respond1");
    std::shared_ptr<std::shared_ptr<::cyberdogapp::CheckoutMode_respond>> respond =
      modeRespond_queue->wait_and_pop();
    RCLCPP_INFO(get_logger(), "get check mode respond2");
    if (respond == NULL) {
      RCLCPP_INFO(get_logger(), "get check mode timeout");
      is_mode_check = false;
      return;
    }

    if (respond->get()->request_id() != request_.request_id) {
      RCLCPP_INFO(get_logger(), "get check mode invalid id");
      continue;
    }

    respond->get()->mutable_next_mode()->CopyFrom(request->next_mode().mode());
    writer->Write(*(respond->get()));
    if (!respond->get()->is_feedback()) {
      RCLCPP_INFO(get_logger(), "get check mode result");
      modeRespond_queue->empty();
      is_mode_check = false;
      return;
    }
  }
}

void Cyberdog_app::publishMotion(const SE3VelocityCMD_T & decissage_out)
{
  decision_pub->publish(decissage_out);
}

void Cyberdog_app::callCameraService(
  int command, std::string args,
  ::cyberdogapp::CameraService_respond * response)
{
  std::chrono::seconds timeout(5);
  auto request = std::make_shared<interaction_msgs::srv::CameraService::Request>();
  request->command = command;
  request->args = args;

  if (!camera_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "callCameraService server not avalible");
    return;
  }
  auto future_result = camera_client_->async_send_request(request);
  std::future_status status = future_result.wait_for(timeout);

  if (status == std::future_status::ready) {
    if (future_result.get()->result ==
      interaction_msgs::srv::CameraService_Response::RESULT_SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "Succeed call camera services.");
    } else {
      RCLCPP_INFO(get_logger(), "failed call camera services.");
    }
  } else {
    RCLCPP_INFO(get_logger(), "Failed to call camera services.");
  }
  response->set_result(future_result.get()->result);
  response->set_msg(future_result.get()->msg);
}
void Cyberdog_app::requestVoice(
  const ::cyberdogapp::Voiceprint_Request * request,
  ::cyberdogapp::Voiceprint_Response * respond)
{
  std::chrono::seconds timeout(15);
  auto request_ = std::make_shared<interaction_msgs::srv::Voiceprint::Request>();
  request_->info.header = returnHeader();
  request_->info.user.id = request->info().user().id();
  request_->info.ask = request->info().ask();
  if (!voice_print_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "requestVoice server not avalible");
    return;
  }
  auto future_result = voice_print_client_->async_send_request(request_);
  std::future_status status = future_result.wait_for(timeout);
  if (status == std::future_status::ready) {
    if (future_result.get()->accept == true) {
      RCLCPP_INFO(get_logger(), "Succeed changed requestVoice.");
    } else {
      RCLCPP_INFO(get_logger(), "Failed to changed requestVoice.");
    }
  } else {
    RCLCPP_INFO(get_logger(), "Failed to check requestVoice.");
  }
  respond->set_accept(future_result.get()->accept);
  respond->set_ask(request->info().ask());
}
void Cyberdog_app::subscribeActionResult(const motion_msgs::msg::ActionRespond::SharedPtr msg)
{
  if (msg->type == motion_msgs::msg::ActionRespond::CHECKOUT_MODE) {
    std::shared_ptr<::cyberdogapp::CheckoutMode_respond> respond =
      std::make_shared<::cyberdogapp::CheckoutMode_respond>();
    respond->set_is_feedback(false);
    respond->set_err_code(msg->err_code);
    respond->set_succeed(msg->succeed);
    respond->set_err_state(msg->err_state);
    respond->set_request_id(msg->request_id);
    modeRespond_queue->push(respond);
    RCLCPP_INFO(get_logger(), "get checkout mode result");
  } else if (msg->type == motion_msgs::msg::ActionRespond::CHECKOUT_PATTERN) {
    std::shared_ptr<::cyberdogapp::CheckoutPattern_respond> respond =
      std::make_shared<::cyberdogapp::CheckoutPattern_respond>();
    respond->set_is_feedback(false);
    respond->set_err_code(msg->err_code);
    respond->set_succeed(msg->succeed);
    respond->set_request_id(msg->request_id);
    patternRespond_queue->push(respond);
    RCLCPP_INFO(get_logger(), "get checkout pattern result");
  } else if (msg->type == motion_msgs::msg::ActionRespond::EXTMONORDER) {
    auto respond = std::make_shared<::cyberdogapp::ExtMonOrder_Respond>();
    respond->set_err_code(msg->err_code);
    respond->set_succeed(msg->succeed);
    respond->set_request_id(msg->request_id);
    respond->set_is_feedback(false);
    extMonOrderRespond_queue->push(respond);
  }
}

std::string Cyberdog_app::getDogIp(const string str, const string & split)
{
  string result;
  int pos = str.find(split);
  if (pos != -1) {
    result = str.substr(pos + split.size(), str.size());
  }
  return result;
}

std::string Cyberdog_app::getPhoneIp(const string str, const string & split)
{
  string result;
  int pos = str.find(split);
  if (pos != -1) {
    result = str.substr(0, pos);
  }
  return result;
}
void Cyberdog_app::subscribeIp(const std_msgs::msg::String::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get ip :%s", msg->data.c_str());
  RCLCPP_INFO(get_logger(), "old phoneip is:%s", (*server_ip).c_str());
  app_disconnected = false;
  local_ip = getDogIp(msg->data, ":");
  RCLCPP_INFO(get_logger(), "local_ip ip :%s", local_ip.c_str());
  std::string phoneIp = getPhoneIp(msg->data, ":");
  RCLCPP_INFO(get_logger(), "phoneIp ip :%s", phoneIp.c_str());
  if (*server_ip != phoneIp) {
    server_ip = std::make_shared<std::string>(phoneIp);
    destroyGrpc();
    createGrpc();
  }
}
void Cyberdog_app::destroyGrpcServer()
{
  if (server_ != nullptr) {
    RCLCPP_INFO(get_logger(), "close server");
    server_->Shutdown();
    RCLCPP_INFO(get_logger(), "join server");
    app_server_thread_->join();
    server_ = nullptr;
  }
}

void Cyberdog_app::destroyGrpc()
{
  can_process_messages = false;
  if (app_stub != nullptr) {
    app_stub = nullptr;
  }
  net_checker.pause();
}

void Cyberdog_app::createGrpc()
{
  RCLCPP_INFO(get_logger(), "Create server");
  if (server_ == nullptr) {
    app_server_thread_ = std::make_shared<std::thread>(&Cyberdog_app::RunServer, this);
  }
  RCLCPP_INFO(get_logger(), "Create client");
  grpc::string ip = *server_ip + std::string(":8980");
  can_process_messages = false;
  heartbeat_err_cnt = 0;
  net_checker.set_ip(*server_ip);
  RCLCPP_INFO(get_logger(), "before channel");
  auto channel_ = grpc::CreateChannel(ip, grpc::InsecureChannelCredentials());
  RCLCPP_INFO(get_logger(), "after channel");
  app_stub = std::make_shared<Cyberdog_App_Client>(channel_);
  RCLCPP_INFO(get_logger(), "end channel");
  can_process_messages = true;
  if (app_disconnected) {
    destroyGrpc();
  }
}

void Cyberdog_app::set_bms(const ception_msgs::msg::Bms::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get bms :%d", msg->batt_soc);
  if (can_process_messages && app_stub) {
    app_stub->SetBms(msg);
  }
}

void Cyberdog_app::set_status(const ControlState_T::SharedPtr msg)
{
  static int status_filter;
  if (status_filter++ % 6) {
    return;
  }
  RCLCPP_INFO(get_logger(), "set_status");
  if (can_process_messages && app_stub) {
    app_stub->SetStatus(msg);
  }
}

void Cyberdog_app::set_rssi(const std_msgs::msg::String::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "set_rssi");
  if (can_process_messages && app_stub) {
    app_stub->set_rssi(msg);
  }
}
// body
void Cyberdog_app::set_body_select(const interaction_msgs::msg::BodyInfo::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "set_body_select");
  if (can_process_messages && app_stub) {
    app_stub->set_BodySelect(msg);
  }
}

// tracking_result
void Cyberdog_app::set_tracking(const interaction_msgs::msg::BodyInfo::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "set_tracking");
  if (can_process_messages && app_stub) {
    app_stub->set_Tracking(msg);
  }
}

void Cyberdog_app::setFollowRegion(
  const ::cyberdogapp::BodyRegion_Request * request,
  ::cyberdogapp::BodyRegion_Respond * respond)
{
  std::chrono::seconds timeout(15);
  auto request_ = std::make_shared<interaction_msgs::srv::BodyRegion::Request>();
  request_->roi.x_offset = request->roi().x_offset();
  request_->roi.y_offset = request->roi().y_offset();
  request_->roi.height = request->roi().height();
  request_->roi.width = request->roi().width();
  request_->roi.do_rectify = request->roi().do_rectify();

  if (!tracking_object_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "setFollowRegion server not avalible");
    return;
  }
  auto future_result = tracking_object_client_->async_send_request(request_);
  std::future_status status = future_result.wait_for(timeout);

  if (status == std::future_status::ready) {
    if (future_result.get()->success == true) {
      RCLCPP_INFO(get_logger(), "Succeed changed tracking object.");
    } else {
      RCLCPP_INFO(get_logger(), "Failed to changed tracking object.");
    }
  } else {
    RCLCPP_INFO(get_logger(), "Failed to check tracking object.");
  }
  respond->set_success(future_result.get()->success);
}

void Cyberdog_app::requestFaceManager(
  const ::cyberdogapp::FaceManager_Request * request,
  ::cyberdogapp::FaceManager_Response * respond)
{
  std::chrono::seconds timeout(15);
  auto request_ = std::make_shared<interaction_msgs::srv::FaceManager::Request>();
  request_->command = request->command();
  request_->args = request->args();
  RCLCPP_INFO(get_logger(), "requestFaceManager.");

  if (!face_manager_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "requestFaceManager server not avalible");
    return;
  }
  auto future_result = face_manager_client_->async_send_request(request_);
  std::future_status status = future_result.wait_for(timeout);
  if (status == std::future_status::ready) {
    if (future_result.get()->result ==
      interaction_msgs::srv::FaceManager::Response::RESULT_SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "Succeed changed requestFaceManager.");
      std::vector<interaction_msgs::msg::CompressedImage> face_images =
        future_result.get()->face_images;
      int cnt = face_images.size();
      for (int i = 0; i < cnt; ++i) {
        ::cyberdogapp::CompressedImage * img = respond->add_face_images();
        ::cyberdogapp::Header header;
        ::cyberdogapp::Timestamp time;

        // construct format
        img->set_format(face_images[i].format);

        // construct header
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
    } else {
      RCLCPP_INFO(get_logger(), "Failed to changed requestFaceManager.");
    }
  } else {
    RCLCPP_INFO(get_logger(), "Failed to check requestFaceManager.");
  }
  respond->set_command(request_->command);
  respond->set_result(future_result.get()->result);
  respond->set_msg(future_result.get()->msg);
}
void Cyberdog_app::sendAiToken(
  const ::cyberdogapp::TokenPass_Request * request,
  ::cyberdogapp::TokenPass_Response * respond)
{
  std::chrono::seconds timeout(15);
  auto request_ = std::make_shared<interaction_msgs::srv::TokenPass::Request>();
  request_->ask = request->ask();
  request_->info.header = returnHeader();
  request_->info.token = request->info().token();
  request_->info.token = request->info().token();
  request_->info.token_refresh = request->info().token_refresh();
  request_->info.expire_in = request->info().expire_in();
  request_->vol = request->vol();
  request_->info.token_md5 = request->info().token_md5();
  request_->info.token_refresh_md5 = request->info().token_refresh_md5();
  if (!ai_token_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "sendAiToken server not avalible");
    return;
  }
  auto future_result = ai_token_client_->async_send_request(request_);
  std::future_status status = future_result.wait_for(timeout);
  if (status == std::future_status::ready) {
    RCLCPP_INFO(get_logger(), "Succeed changed aitoken.");
  } else {
    RCLCPP_INFO(get_logger(), "Failed to check aitoken.");
  }
  respond->set_flage(future_result.get()->flage);
  respond->set_divice_id(future_result.get()->divice_id);
  respond->set_vol(future_result.get()->vol);
}
void Cyberdog_app::setNavPosition(
  const ::cyberdogapp::Target_Request * request,
  ::cyberdogapp::Target_Response * respond)
{
  std::chrono::seconds timeout(15);
  auto request_ = std::make_shared<automation_msgs::srv::Target::Request>();
  request_->header = returnHeader();
  request_->info.map_load_time.sec = request->info().map_load_time().sec();
  request_->info.map_load_time.nanosec = request->info().map_load_time().nanosec();
  request_->info.resolution = request->info().resolution();
  request_->info.width = request->info().width();
  request_->info.height = request->info().height();

  request_->info.origin.position.x = request->info().origin().position().x();
  request_->info.origin.position.y = request->info().origin().position().y();
  request_->info.origin.position.z = request->info().origin().position().z();

  request_->info.origin.orientation.x = request->info().origin().orientation().x();
  request_->info.origin.orientation.y = request->info().origin().orientation().y();
  request_->info.origin.orientation.z = request->info().origin().orientation().z();
  request_->info.origin.orientation.w = request->info().origin().orientation().w();

  request_->target_x = request->target_x();
  request_->target_y = request->target_y();
  if (!nav_position_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "setNavPosition server not avalible");
    return;
  }
  auto future_result = nav_position_client_->async_send_request(request_);
  std::future_status status = future_result.wait_for(timeout);
  if (status == std::future_status::ready) {
    RCLCPP_INFO(get_logger(), "Succeed changed setNavPosition.");
    respond->set_success(future_result.get()->success);
  } else {
    respond->set_success(false);
    RCLCPP_INFO(get_logger(), "Failed to check setNavPosition.");
  }
}

void Cyberdog_app::subscribeVoiceprintResult(
  const interaction_msgs::msg::VoiceprintResult::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribeVoiceprintResult");
  if (can_process_messages && app_stub) {
    app_stub->subscribeVoiceprintResult(msg);
  }
}

void Cyberdog_app::subscribeFaceResult(const interaction_msgs::msg::FaceResult::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribeFaceResult");
  if (can_process_messages && app_stub) {
    app_stub->subscribeFaceResult(msg);
  }
}

void Cyberdog_app::subscribePosition(const SE3VelocityCMD_T::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribePosition");
  if (msg->velocity.frameid.id != motion_msgs::msg::Frameid::ODOM_FRAME) {
    return;
  }
  if (can_process_messages && app_stub) {
    app_stub->subscribePosition(msg);
  }
}

void Cyberdog_app::subscribeTrackingStatus(
  const automation_msgs::msg::TrackingStatus::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "subscribeTrackingStatus");
  if (can_process_messages && app_stub) {
    app_stub->subscribeTrackingStatus(msg);
  }
}

void Cyberdog_app::subscribeNavStatus(const automation_msgs::msg::Caution::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribeNavStatus");
  if (can_process_messages && app_stub) {
    app_stub->subscribeNavStatus(msg);
  }
}

void Cyberdog_app::subscribeMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get set_map");
  if (can_process_messages && app_stub) {
    app_stub->subscribeMap(msg);
  }
}

void Cyberdog_app::subscribeOdomOut(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  static int odom_out_filter;
  if (odom_out_filter++ % 100) {
    return;
  }
  RCLCPP_INFO(get_logger(), "get subscribeOdomOut");
  if (can_process_messages && app_stub) {
    app_stub->subscribeOdomOut(msg);
  }
}

void Cyberdog_app::subscribeObstacleDetection(const ception_msgs::msg::Around::SharedPtr msg)
{
  static int obstacle_filter;
  if (obstacle_filter++ % 10) {
    return;
  }
  RCLCPP_INFO(get_logger(), "get subscribeObstacleDetection");
  if (can_process_messages && app_stub) {
    app_stub->subscribeObstacleDetection(msg);
  }
}

void Cyberdog_app::subscribeDogPose(const motion_msgs::msg::SE3Pose::SharedPtr msg)
{
  static int dogpose_filter;
  if (dogpose_filter++ % 10) {
    return;
  }
  RCLCPP_INFO(get_logger(), "get subscribeDogPose");
  if (can_process_messages && app_stub) {
    app_stub->subscribeDogPose(msg);
  }
}

void Cyberdog_app::subscribeGpsScene(const motion_msgs::msg::Scene::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribeGpsScene");
  if (can_process_messages && app_stub) {
    app_stub->subscribeGpsScene(msg);
  }
}
void Cyberdog_app::subscribeRemoteEvent(const ception_msgs::msg::BtRemoteEvent::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribeRemoteEvent");
  if (can_process_messages && app_stub) {
    app_stub->subscribeRemoteEvent(msg);
  }
}

void Cyberdog_app::subscribePath(const nav_msgs::msg::Path::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "get subscribePath");
  if (can_process_messages && app_stub) {
    app_stub->subscribePath(msg);
  }
}

std_msgs::msg::Header Cyberdog_app::returnHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "odom";
  msg.stamp = this->get_clock()->now();
  RCLCPP_INFO(get_logger(), "returnHeader  %d %d", msg.stamp.sec, msg.stamp.nanosec);
  return msg;
}

#define OFFSET_WALK "speed_offset_walk"
#define OFFSET_TROT "speed_offset_trot"
#define OFFSET_FLY_TROT "speed_offset_fly_trot"
#define OFFSET_SLOW_TROT "speed_offset_slow_trot"

void Cyberdog_app::getOffsetData(
  const ::cyberdogapp::OffsetRequest * request,
  ::grpc::ServerWriter<::cyberdogapp::OffsetCalibationData> * writer)
{
  control_parameter_request_lcmt request_;
  switch (request->gait()) {
    case cyberdogapp::OffsetRequest::WALK:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_WALK, sizeof(request_.name) - 1);
      break;
    case cyberdogapp::OffsetRequest::TROT:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_TROT, sizeof(request_.name) - 1);
      break;
    case cyberdogapp::OffsetRequest::FLY_TROT:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_FLY_TROT, sizeof(request_.name) - 1);
      break;
    case cyberdogapp::OffsetRequest::SLOW_TROT:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_SLOW_TROT, sizeof(request_.name) - 1);
      break;
    default:
      RCLCPP_INFO(get_logger(), "Error at gait type");
      return;
  }
  RCLCPP_INFO(get_logger(), "getOffsetData %s", request_.name);
  request_.requestNumber = requestNumber++;
  request_.parameterKind = 3;
  request_.requestKind = 4;

  offset_request->publish("interface_request", &request_);

  wait_for_offset_data = true;
  std::shared_ptr<std::shared_ptr<cyberdogapp::OffsetCalibationData>> respond =
    offsetCalibationData_queue->wait_and_pop();
  if (respond == NULL) {
    RCLCPP_INFO(get_logger(), "get getOffsetData timeout");
    wait_for_offset_data = false;
    return;
  }
  writer->Write(*(respond->get()));
  wait_for_offset_data = false;
  offsetCalibationData_queue->clear();
}

void Cyberdog_app::setOffsetData(
  const ::cyberdogapp::OffsetCalibationData * request,
  ::grpc::ServerWriter<::cyberdogapp::OffsetRequest_result> * writer)
{
  control_parameter_request_lcmt request_;
  std::vector<double> vec;
  switch (request->gait()) {
    case cyberdogapp::OffsetRequest::WALK:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_WALK, sizeof(request_.name) - 1);
      break;
    case cyberdogapp::OffsetRequest::TROT:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_TROT, sizeof(request_.name) - 1);
      break;
    case cyberdogapp::OffsetRequest::FLY_TROT:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_FLY_TROT, sizeof(request_.name) - 1);
      break;
    case cyberdogapp::OffsetRequest::SLOW_TROT:
      strncpy(reinterpret_cast<char *>(request_.name), OFFSET_SLOW_TROT, sizeof(request_.name) - 1);
      break;
    default:
      RCLCPP_INFO(get_logger(), "Error at gait type");
      return;
  }
  request_.requestNumber = requestNumber++;

  vec.push_back(request->x_offset());
  vec.push_back(request->y_offset());
  vec.push_back(request->yaw_offset());
  RCLCPP_INFO(
    get_logger(), "setOffsetData: %s, %lf, %lf, %lf",
    request_.name,
    request->x_offset(),
    request->y_offset(),
    request->yaw_offset());

  memcpy(request_.value, vec.data(), sizeof(vec));
  request_.parameterKind = 3;
  request_.requestKind = 5;

  offset_request->publish("interface_request", &request_);
  wait_for_set_data_result = true;
  std::shared_ptr<std::shared_ptr<cyberdogapp::OffsetRequest_result>> respond =
    offsetRequestResult_queue->wait_and_pop();
  if (respond == NULL) {
    RCLCPP_INFO(get_logger(), "get setOffsetData timeout");
    wait_for_set_data_result = false;
    return;
  }
  writer->Write(*(respond->get()));
  wait_for_set_data_result = false;
  offsetRequestResult_queue->clear();
}
void Cyberdog_app::setExtmonOrder(
  const ::cyberdogapp::ExtMonOrder_Request * request,
  ::grpc::ServerWriter< ::cyberdogapp::ExtMonOrder_Respond> * writer)
{
  motion_msgs::msg::ActionRequest request_;
  motion_msgs::msg::MonOrder order;
  order.id = request->order().id();
  order.para = request->order().para();
  request_.type = motion_msgs::msg::ActionRequest::EXTMONORDER;
  request_.order = order;
  request_.request_id = ++ext_mon_id;
  request_.timeout = request->timeout();
  action_request_pub->publish(request_);
  RCLCPP_INFO(get_logger(), "get check extmonorder timeout seconds: %d", request->timeout());
  if (request->timeout() > 0) {
    extMonOrderRespond_queue->set_time_out(request->timeout());
  } else {
    extMonOrderRespond_queue->set_time_out(35);
  }
  while (true) {
    std::shared_ptr<std::shared_ptr<cyberdogapp::ExtMonOrder_Respond>> respond =
      extMonOrderRespond_queue->wait_and_pop();
    if (respond == NULL) {
      RCLCPP_INFO(get_logger(), "get check extmonorder timeout");
      return;
    }

    if (respond->get()->request_id() != request_.request_id) {
      RCLCPP_INFO(get_logger(), "get check extmonorder invalid id");
      continue;
    }

    respond->get()->mutable_order()->CopyFrom(request->order());
    writer->Write(*(respond->get()));
    if (!respond->get()->is_feedback()) {
      RCLCPP_INFO(get_logger(), "get check extmonorder result");
      extMonOrderRespond_queue->empty();
      return;
    }
  }
}

void Cyberdog_app::disconnect(
  const ::cyberdogapp::Disconnect * request,
  ::cyberdogapp::Result * respond)
{
  app_disconnected = true;
  destroyGrpc();
  server_ip = std::make_shared<std::string>("0.0.0.0");
  RCLCPP_INFO(get_logger(), "disconnect server ip :%s", (*server_ip).c_str());
}

void Cyberdog_app::setBtRemoteCmd(
  const ::cyberdogapp::BtRemoteCommand_Request * request,
  ::cyberdogapp::BtRemoteCommand_Respond * respond)
{
  std::chrono::seconds timeout(15);
  auto request_ = std::make_shared<ception_msgs::srv::BtRemoteCommand::Request>();
  request_->command = request->command();
  request_->address = request->address();

  if (!remote_manager_client_->wait_for_service()) {
    RCLCPP_INFO(get_logger(), "request remote server not avalible");
    return;
  }
  auto future_result = remote_manager_client_->async_send_request(request_);
  std::future_status status = future_result.wait_for(timeout);
  if (status == std::future_status::ready) {
    if (future_result.get()->success == true) {
      RCLCPP_INFO(get_logger(), "Succeed to send remote command.");
    } else {
      RCLCPP_INFO(get_logger(), "Failed to send remote command.");
    }
  } else {
    RCLCPP_INFO(get_logger(), "Failed to send remote command.");
  }

  respond->set_success(future_result.get()->success);
}

void Cyberdog_app::setBodyPara(const Parameters_T & para)
{
  para_pub->publish(para);
}

std::string Cyberdog_app::getLcmUrl(int ttl)
{
  return "udpm://239.255.76.67:7668?ttl=" + std::to_string(ttl);
}
bool Cyberdog_app::isMatch(const char * source, const char * dest)
{
  if (strncmp(source, dest, strlen(source)) == 0) {
    return true;
  } else {
    return false;
  }
}
void Cyberdog_app::receive_lcm_data(
  const lcm::ReceiveBuffer * rbuf,
  const std::string & chan,
  const control_parameter_respones_lcmt * lcm_data)
{
  RCLCPP_INFO(
    get_logger(), "receive_lcm_data, %d, %d, %d", lcm_data->requestKind, wait_for_offset_data,
    wait_for_set_data_result);
  if (lcm_data->requestKind == 4 && wait_for_offset_data) {
    // wait for read data;
    double v[3];
    RCLCPP_INFO(get_logger(), "receive_lcm_data, read result");
    auto respond = std::make_shared<::cyberdogapp::OffsetCalibationData>();
    respond->set_result(cyberdogapp::OffsetCalibationData::SUCCESS);

    if (isMatch(OFFSET_WALK, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::WALK);
    } else if (isMatch(OFFSET_TROT, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::TROT);
    } else if (isMatch(OFFSET_FLY_TROT, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::FLY_TROT);
    } else if (isMatch(OFFSET_SLOW_TROT, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::SLOW_TROT);
    } else {
      RCLCPP_INFO(get_logger(), "receive_lcm_data, Unknow gait:%s", lcm_data->name);
      respond->set_result(cyberdogapp::OffsetCalibationData::FAILED);
    }

    RCLCPP_INFO(get_logger(), "receive_lcm_data, gait:%d", respond->gait());
    memcpy(v, lcm_data->value, sizeof(v));
    respond->set_x_offset(v[0]);
    respond->set_y_offset(v[1]);
    respond->set_yaw_offset(v[2]);

    RCLCPP_INFO(
      get_logger(), "receive_lcm_data, gait:%lf, %lf, %lf ",
      respond->x_offset(),
      respond->y_offset(),
      respond->yaw_offset());
    offsetCalibationData_queue->push(respond);
  } else if (lcm_data->requestKind == 5 && wait_for_set_data_result) {
    // wait for set result
    RCLCPP_INFO(get_logger(), "receive_lcm_data, write result");
    auto respond = std::make_shared<::cyberdogapp::OffsetRequest_result>();
    respond->set_result(cyberdogapp::OffsetRequest_result::SUCCESS);
    if (isMatch(OFFSET_WALK, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::WALK);
    } else if (isMatch(OFFSET_TROT, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::TROT);
    } else if (isMatch(OFFSET_FLY_TROT, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::FLY_TROT);
    } else if (isMatch(OFFSET_SLOW_TROT, CON_TO_CHAR(lcm_data->name))) {
      respond->set_gait(cyberdogapp::OffsetRequest::SLOW_TROT);
    } else {
      RCLCPP_INFO(get_logger(), "receive_lcm_data, Unknow gait:%s", lcm_data->name);
      respond->set_result(cyberdogapp::OffsetRequest_result::FAILED);
    }
    offsetRequestResult_queue->push(respond);
  }
}
}  // namespace cyberdog_cyberdog_app
