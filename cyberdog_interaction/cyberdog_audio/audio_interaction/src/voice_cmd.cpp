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

#include <thread>
#include <string>
#include <utility>
#include <memory>

#include "audio_interaction/voice_cmd.hpp"

namespace cyberdog_audio
{
VoiceCmd::VoiceCmd()
: cyberdog_utils::LifecycleNode("voice_cmd")
{
  this->declare_parameter("back_distance", -1.0);
  this->declare_parameter("turn_angle", 6.28);
  RCLCPP_INFO(get_logger(), "Creating voice cmd.");
}

VoiceCmd::~VoiceCmd()
{
  RCLCPP_INFO(get_logger(), "Destroying");
}

cyberdog_utils::CallbackReturn VoiceCmd::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Configuring");

  loop_rate_ = 20;
  timeout_ = std::chrono::seconds(1);

  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  net_status_pub_ = this->create_publisher<std_msgs::msg::Int8>(
    "network_status",
    rclcpp::SystemDefaultsQoS());
  ai_switch_pub_ = this->create_publisher<std_msgs::msg::Int8>(
    "ai_switch",
    rclcpp::SystemDefaultsQoS());
  token_ready_pub_ = this->create_publisher<std_msgs::msg::Int8>(
    "token_get",
    rclcpp::SystemDefaultsQoS());
  ai_order_sub_ = this->create_subscription<std_msgs::msg::Int8>(
    "robot_order",
    rclcpp::SystemDefaultsQoS(), std::bind(
      &VoiceCmd::SubscripAiorder, this,
      std::placeholders::_1));

  token_server_ = this->create_service<TokenPassT>(
    "token_update",
    std::bind(
      &VoiceCmd::check_app_order, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  play_server_ = std::make_unique<PlayServer>(
    this->shared_from_this(), "audio_play",
    std::bind(&VoiceCmd::check_play_request, this));

  auto options0 = rclcpp::NodeOptions().arguments(
    {"--ros-args",
      "-r", std::string("__node:=") + get_name() + "_client_node0",
      "--"});

  play_client_node_ = std::make_shared<rclcpp::Node>("_", options0);
  play_client_ = rclcpp_action::create_client<PlayorderT>(play_client_node_, "audio_play");

  auto options1 = rclcpp::NodeOptions().arguments(
    {"--ros-args",
      "-r", std::string("__node:=") + get_name() + "_client_node1",
      "--"});

  ExtMon_client_node_ = std::make_shared<rclcpp::Node>("_", options1);
  ExtMon_client_ = rclcpp_action::create_client<MonorderT>(ExtMon_client_node_, "exe_monorder");

  wake_led_client_ = this->create_client<SensorNodeT>(
    "cyberdog_led", rmw_qos_profile_system_default,
    callback_group_);
  ask_assistant_client_ = this->create_client<AssistantT>(
    "ai_switch",
    rmw_qos_profile_system_default,
    callback_group_);

  timer_net_ = this->create_wall_timer(
    std::chrono::seconds(2),
    std::bind(&VoiceCmd::PublishNetStatus, this));

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn VoiceCmd::on_activate(const rclcpp_lifecycle::State &)
{
  int ai_status, pub_status, volume_value;
  RCLCPP_INFO(get_logger(), "Activating");
  player_init();
  sleep(10);
  play_server_->activate();
  net_status_pub_->on_activate();
  ai_switch_pub_->on_activate();
  token_ready_pub_->on_activate();
  std::string device_id = getDeviceId();

  const auto token = toml::parse(TOKEN_FILE);
  std::string title_toml = toml::find<std::string>(token, "title");
  RCLCPP_INFO(get_logger(), "toml title: %s", title_toml.c_str());
  std::int64_t expireIn_toml = toml::find<std::int64_t>(token, "token_expireIn");
  RCLCPP_INFO(get_logger(), "toml token_expireIn: %ld", expireIn_toml);
  std::string token_a_toml = toml::find<std::string>(token, "token_access");
  RCLCPP_INFO(get_logger(), "toml token_access: %s", token_a_toml.c_str());
  std::string token_r_toml = toml::find<std::string>(token, "token_fresh");
  RCLCPP_INFO(get_logger(), "toml token_fresh: %s", token_r_toml.c_str());

  const toml::value toml_t{
    {"title", title_toml},
    {"token_access", token_a_toml},
    {"token_fresh", token_r_toml},
    {"token_expireIn", expireIn_toml},
    {"token_deviceid", device_id}
  };
  std::ofstream ofs(TOKEN_FILE, std::ofstream::out);
  ofs << toml_t;
  ofs.close();
  PublishTokenReady(DEVICE_ID_READY);
  ai_status = get_ai_require_status();
  if (fac_test_flage(FAC_TEST_FILE)) {
    RCLCPP_INFO(get_logger(), "factory force xiaoai on");
    pub_status = AI_ONLINE_ON;
  } else {
    if (ai_status == -1) {
      pub_status = AI_OFF;
    } else if (ai_status == AI_OFFLINE_ON) {
      pub_status = AI_OFFLINE_ON;
    } else if (ai_status == AI_ONLINE_ON) {
      pub_status = AI_ONLINE_ON;
    } else {
      pub_status = AI_OFF;
    }
  }

  volume_value = volume_check();
  if (volume_value == -1) {
    volume_set(7);
  } else {
    volume_set(volume_value);
  }
  PublishAiSwitch(pub_status);
  set_ai_require_status(pub_status);

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn VoiceCmd::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactivating");
  net_status_pub_->on_deactivate();
  ai_switch_pub_->on_deactivate();
  token_ready_pub_->on_deactivate();
  play_server_->deactivate();

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn VoiceCmd::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Cleaning up");
  net_status_pub_.reset();
  ai_switch_pub_.reset();
  token_ready_pub_.reset();
  token_server_.reset();
  play_server_.reset();
  ExtMon_client_.reset();
  ExtMon_client_node_.reset();
  play_client_.reset();
  play_client_node_.reset();
  timer_net_->reset();

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn VoiceCmd::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Shutting down");

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

int VoiceCmd::ExecuteCMD(const char * cmd, char * result)
{
  int iRet = -1;
  char buf_ps[CMD_RESULT_BUF_SIZE];
  char ps[CMD_RESULT_BUF_SIZE] = {0};
  FILE * ptr;

  snprintf(ps, sizeof(ps), "%s", cmd);

  if ((ptr = popen(ps, "r")) != NULL) {
    while (fgets(buf_ps, sizeof(buf_ps), ptr) != NULL) {
      strncat(result, buf_ps, CMD_RESULT_BUF_SIZE);
      if (strlen(result) > CMD_RESULT_BUF_SIZE) {
        break;
      }
    }
    pclose(ptr);
    ptr = NULL;
    iRet = 0;
  } else {
    RCLCPP_INFO(get_logger(), "popen %s error\n", ps);
    iRet = -1;
  }
  return iRet;
}

int VoiceCmd::Detectwifi()
{
  int wifi_status = -1;
  char result[CMD_RESULT_BUF_SIZE] = {0};
  ExecuteCMD("iw wlan0 link | grep signal", result);
  std::string s(result);
  if (s.length() == 0) {
    RCLCPP_INFO(get_logger(), "wifi signal NULL");
    wifi_status = -1;
  } else {
    RCLCPP_INFO(get_logger(), "length: %ld", s.length());
    s.erase(0, 9);
    s = s.substr(0, s.length() - 5);
    RCLCPP_INFO(get_logger(), "wifi signal: %s", s.c_str());
    int i = atoi(s.c_str());
    RCLCPP_INFO(get_logger(), "wifi signal: %d", i);
    if (i > -70) {
      RCLCPP_INFO(get_logger(), "strong wifi signal: %d", i);
      wifi_status = 1;
    } else if (i >= -80 && i <= -70) {
      RCLCPP_INFO(get_logger(), "average wifi signal: %d", i);
      wifi_status = 1;
    } else if (i < -80) {
      RCLCPP_INFO(get_logger(), "weak wifi signal: %d", i);
      wifi_status = -1;
    }
  }
  return wifi_status;
}

std::string VoiceCmd::getDeviceId()
{
  // 对mac地址做md5
  std::string deviceId;
#ifdef __linux__

  struct ifreq ifr_mac;
  struct ifconf ifc;
  char mac_addr[32] = {0};
  char buf[1024];
  bool success = false;

  int sock_mac = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock_mac == -1) {
    perror("open socket failed/n");
    exit(-1);
  }

  ifc.ifc_len = sizeof(buf);
  ifc.ifc_buf = buf;
  if (ioctl(sock_mac, SIOCGIFCONF, &ifc) == -1) {
    perror("socket operation failed/n");
    exit(-1);
  }

  struct ifreq * it = ifc.ifc_req;
  const struct ifreq * const end = it + (ifc.ifc_len / sizeof(struct ifreq));

  for (; it != end; ++it) {
    memset(&ifr_mac, 0, sizeof(ifr_mac));
    snprintf(ifr_mac.ifr_name, sizeof(ifr_mac.ifr_name), "%s", it->ifr_name);
    std::cout << "try to use " << ifr_mac.ifr_name << " to calculate device id" << std::endl;
    if (ioctl(sock_mac, SIOCGIFFLAGS, &ifr_mac) == 0) {
      if (!(ifr_mac.ifr_flags & IFF_LOOPBACK)) {
        if (ioctl(sock_mac, SIOCGIFHWADDR, &ifr_mac) == 0) {
          success = true;
          break;
        }
      }
    } else {
      perror("socket operation failed/n");
      exit(-1);
    }
  }

  if (!success) {
    perror("get deviceid from mac failed/n");
    exit(-1);
  }

  snprintf(
    mac_addr,
    sizeof(mac_addr),
    "%02x%02x%02x%02x%02x%02x",
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[0],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[1],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[2],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[3],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[4],
    (unsigned char)ifr_mac.ifr_hwaddr.sa_data[5]);

  close(sock_mac);
  std::cout << "get mac addr: " << mac_addr << std::endl;
  deviceId = md5_cal(mac_addr);

#else
  deviceId = "windows_device_id";

#endif  // !#ifndef _MSC_VER
  printf("get device id:%s\n", deviceId.c_str());
  return deviceId;
}

void VoiceCmd::PublishNetStatus()
{
  auto msg = std::make_unique<std_msgs::msg::Int8>();
  msg->data = Detectwifi();
  if (!net_status_pub_->is_activated()) {
    // RCLCPP_INFO(get_logger(), "Publisher is inactive. Net status can't be published.");
  } else {
    // RCLCPP_INFO(get_logger(), "Publisher is active, publishing net status: %d.", msg->data);
  }
  net_status_pub_->publish(std::move(msg));
}

void VoiceCmd::PublishAiSwitch(int order)
{
  auto msg = std::make_unique<std_msgs::msg::Int8>();
  msg->data = order;
  if (!net_status_pub_->is_activated()) {
    RCLCPP_INFO(
      get_logger(), "Publisher is currently inactive. Messages are not published.");
  } else {
    RCLCPP_INFO(
      get_logger(), "Publisher is active, publishing xiaoai button ask: %d.", msg->data);
  }
  ai_switch_pub_->publish(std::move(msg));
  ai_status_temp = order;
  RCLCPP_INFO(get_logger(), "ai_status_temp: %d", ai_status_temp);
}

void VoiceCmd::PublishTokenReady(int order)
{
  auto msg = std::make_unique<std_msgs::msg::Int8>();
  msg->data = order;
  if (!token_ready_pub_->is_activated()) {
    RCLCPP_INFO(
      get_logger(), "Publisher is active, Messages are not published.");
  } else {
    RCLCPP_INFO(
      get_logger(), "Publishing token ready messenge to audio_assistant: %d.", msg->data);
  }
  token_ready_pub_->publish(std::move(msg));
}

void VoiceCmd::SubscripAiorder(const std_msgs::msg::Int8::SharedPtr order)
{
  int id = 0;
  double param = 0.0;
  switch (order->data) {
    case WAKE_UP_SIGNAL:
      RCLCPP_INFO(get_logger(), "WAKE STOP.");
      RCLCPP_INFO(get_logger(), "listen.");
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_play_goal(WAKE_UP_SIGNAL);
        send_wake_led_request(std::chrono::seconds(1), WAKE_UP_SIGNAL);
      }
      break;
    case ORDER_STAND_UP:
      RCLCPP_INFO(get_logger(), "STAND.");
      id = ORDER_STAND_UP;
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_extmon_goal(id, param);
      }
      break;
    case ORDER_PROSTRATE:
      RCLCPP_INFO(get_logger(), "DOWN.");
      id = ORDER_PROSTRATE;
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_extmon_goal(id, param);
      }
      break;
    case ORDER_COME_HERE:
      RCLCPP_INFO(get_logger(), "COME.");
      break;
    case ORDER_STEP_BACK:
      RCLCPP_INFO(get_logger(), "BACK.");
      id = ORDER_STEP_BACK;
      param = get_parameter("back_distance").as_double();
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_extmon_goal(id, param);
      }
      break;
    case ORDER_TURN_AROUND:
      RCLCPP_INFO(get_logger(), "TURN_ROUND.");
      id = ORDER_TURN_AROUND;
      param = get_parameter("turn_angle").as_double();
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_extmon_goal(id, param);
      }
      break;
    case ORDER_HI_FIVE:
      RCLCPP_INFO(get_logger(), "HIGH_FIVE.");
      id = ORDER_HI_FIVE;
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_extmon_goal(id, param);
      }
      break;
    case ORDER_DANCE:
      RCLCPP_INFO(get_logger(), "DANCE.");
      id = ORDER_DANCE;
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_extmon_goal(id, param);
      }
      break;
    case TALK_FEEDBACK_SIGNAL:
      RCLCPP_INFO(get_logger(), "101: talk feedback.");
      if (get_ai_require_status() == AI_ONLINE_ON || get_ai_require_status() == AI_OFFLINE_ON) {
        send_wake_led_request(std::chrono::seconds(1), TALK_FEEDBACK_SIGNAL);
      }
    case 102:
      break;
    case 103:
      break;
    case 104:
      break;
    case 110:
      RCLCPP_INFO(get_logger(), "110: xiaoai_online:token error.");
      // send_play_goal(124);
      break;
    case 111:
      RCLCPP_INFO(get_logger(), "111: xiaoai_online restart succeed.");
      // send_play_goal(125);
      break;
    default:
      RCLCPP_ERROR(get_logger(), "ERROR ORDER.");
  }
}

std_msgs::msg::Header VoiceCmd::returnMotionHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "ext_order";
  msg.stamp = this->get_clock()->now();
  action_start_time_ = msg.stamp;

  return msg;
}

std_msgs::msg::Header VoiceCmd::returnPlayHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "audio_play";
  msg.stamp = this->get_clock()->now();
  return msg;
}

std_msgs::msg::Header VoiceCmd::returnExtMonHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "ExeMonOrder";
  msg.stamp = this->get_clock()->now();
  return msg;
}

std_msgs::msg::Header VoiceCmd::returnSensorNodeTHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "rear_led_detection";
  msg.stamp = this->get_clock()->now();
  return msg;
}

std_msgs::msg::Header VoiceCmd::returnBmsReqHeader()
{
  std_msgs::msg::Header msg;
  msg.frame_id = "CheckPower";
  msg.stamp = this->get_clock()->now();
  return msg;
}

void VoiceCmd::check_play_request()
{
  auto goal = play_server_->get_current_goal();
  auto feedback = std::make_shared<PlayorderT::Feedback>();
  auto result = std::make_shared<PlayorderT::Result>();
  auto order_frame = goal->order.header.frame_id;
  // auto order_time = goal->order.header.stamp;
  // auto order_user = goal->order.user;
  auto order_name = goal->order.name.id;
  rclcpp::WallRate r(std::chrono::seconds(1));
  int volume_memory = volume_get();
  RCLCPP_INFO(get_logger(), "volume_get: %d.", volume_memory);
  bool new_goal(false);
  RCLCPP_INFO(
    get_logger(), "Got order goal %d from %s.", goal->order.name.id,
    rclcpp::get_c_string(goal->order.header.frame_id));

  if (!play_server_ || !play_server_->is_server_active()) {
    RCLCPP_DEBUG(get_logger(), "Playback server inactive. Stopping.");
    result->result.succeed = false;
    result->result.header = returnPlayHeader();
    result->result.name.id = goal->order.name.id;
    result->result.error = interaction_msgs::msg::AudioResult::SERVER_INACTIVE;
    play_server_->succeeded_current(result);
    return;
  }

  if (order_name == 8 || order_name == 9) {
    volume_set(7);
  }
  new_goal = true;
  RCLCPP_INFO(
    get_logger(), "Received playback order %d, the song while to be played or rejected.",
    static_cast<int>(order_name));
  while (rclcpp::ok()) {
    /* if (play_server_->is_cancel_requested())
    {
      RCLCPP_INFO(get_logger(), "Canceling playback goal.");
      play_server_->terminate_all();
      return;
    } */

    if (play_server_->is_preempt_requested()) {
      RCLCPP_INFO(get_logger(), "Preempting the playback order.");
      play_server_->terminate_pending_goal();
    }

    if (new_goal) {
      new_goal = false;
      RCLCPP_INFO(
        get_logger(),
        "Received a new play order.");

      play_end = false;
      player_handle(order_name);

      while (!play_end) {
        feedback->feed.header = returnPlayHeader();
        feedback->feed.name.id = goal->order.name.id;
        feedback->feed.status = interaction_msgs::msg::AudioFeedback::ACCEPT;
        // feedback->feed.status = cyberdog_interfaces::msg::AudioFeedback::REJECT;
        // feedback->feed.status = cyberdog_interfaces::msg::AudioFeedback::ERROR;
        play_server_->publish_feedback(feedback);
        RCLCPP_INFO(
          get_logger(),
          "Play order %d executing", order_name);
        r.sleep();
      }
      result->result.succeed = true;
      result->result.header = returnPlayHeader();
      result->result.name.id = goal->order.name.id;
      result->result.error = interaction_msgs::msg::AudioResult::NORMAL;
      play_server_->succeeded_current(result);
      RCLCPP_INFO(
        get_logger(),
        "Playorder %d executed succeed.", order_name);
      RCLCPP_INFO(
        get_logger(),
        "Playorder %d executed join.", order_name);
      /* RCLCPP_INFO(
        get_logger(),
        "Playorder %d executed failed.", order_name); */
      if (order_name == 8 || order_name == 9) {
        volume_set(volume_memory);
      }
      return;
    }
  }
}

void VoiceCmd::check_app_order(
  const std::shared_ptr<rmw_request_id_t> request_header_,
  const std::shared_ptr<TokenPassT::Request> request_,
  std::shared_ptr<TokenPassT::Response> response_)
{
  RCLCPP_INFO(get_logger(), "Token check and update entry.");
  int ai_pub_status;
  builtin_interfaces::msg::Time temp_time;
  (void)request_header_;
  std::string did_temp;
  if (request_->ask == TokenPassT::Request::ASK_TOKEN) {
    auto token_t = request_->info.token;
    auto token_md5_t = request_->info.token_md5;
    auto token_refresh_t = request_->info.token_refresh;
    auto token_refresh_md5_t = request_->info.token_refresh_md5;
    auto token_expire_in_t = request_->info.expire_in;

    auto token_md5_calcu = md5_cal(token_t);
    auto token_refresh_md5_calcu = md5_cal(token_refresh_t);

    RCLCPP_INFO(get_logger(), "token: %s", token_t.c_str());
    RCLCPP_INFO(get_logger(), "token_md5: %s", token_md5_t.c_str());
    RCLCPP_INFO(get_logger(), "token_md5_cal: %s", token_md5_calcu.c_str());
    RCLCPP_INFO(get_logger(), "token_ref: %s", token_refresh_t.c_str());
    RCLCPP_INFO(get_logger(), "token_ref_md5: %s", token_refresh_md5_t.c_str());
    RCLCPP_INFO(get_logger(), "token_ref_md5_cal: %s", token_refresh_md5_calcu.c_str());

    if (token_md5_t == token_md5_calcu && token_refresh_md5_t == token_refresh_md5_calcu) {
      RCLCPP_INFO(
        get_logger(), "the MD5 values of the token and fresh token are successfully verified.");
      response_->flage = TokenPassT::Response::TOKEN_SUCCEED;
      const toml::value data{
        {"title", "offline data for audio node"},
        {"token_access", token_t},
        {"token_fresh", token_refresh_t},
        {"token_expireIn", token_expire_in_t}
      };
      std::ofstream ofs(TOKEN_FILE, std::ofstream::out);
      ofs << data;
      ofs.close();
      PublishTokenReady(TOKEN_READY);
    } else {
      RCLCPP_INFO(
        get_logger(),
        "the MD5 values of the token and fresh token are failed to verify.");
      response_->flage = TokenPassT::Response::TOKEN_FAILED;
    }
  } else if (request_->ask == TokenPassT::Request::ASK_DEVICE_ID) {
    did_temp = response_->divice_id = getDeviceId();
    if (did_temp.length() == 0) {
      response_->flage = TokenPassT::Response::DID_FAILED;
    }
    response_->flage = TokenPassT::Response::DID_SUCCEED;
  } else if (request_->ask == TokenPassT::Request::ASK_XIAOAI_OFF) {
    ai_pub_status = AI_OFF;
    RCLCPP_INFO(
      get_logger(), "recive xiaoai off ask from app, the ask will pub to audio assitant: %d.",
      ai_pub_status);
    PublishAiSwitch(ai_pub_status);
    set_ai_require_status(AI_OFF);
    response_->flage = TokenPassT::Response::XIAOAI_OFF_SUCCEED;
  } else if (request_->ask == TokenPassT::Request::ASK_XIAOAI_ON) {
    ai_pub_status = AI_ONLINE_ON;
    RCLCPP_INFO(
      get_logger(), "recive xiaoai on ask from app, the ask will pub to audio assitant: %d.",
      ai_pub_status);
    PublishAiSwitch(ai_pub_status);
    set_ai_require_status(AI_ONLINE_ON);
    response_->flage = TokenPassT::Response::XIAOAI_ON_SUCCEED;
  } else if (request_->ask == TokenPassT::Request::ASK_SET_VOLUME) {
    response_->flage = TokenPassT::Response::SET_VOLUME_SUCCEED;
    volume_set(request_->vol);
    RCLCPP_INFO(get_logger(), "recive set volume ask from app.");
  } else if (request_->ask == TokenPassT::Request::ASK_GET_VOLUME) {
    response_->flage = TokenPassT::Response::GET_VOLUME_SUCCEED;
    response_->vol = volume_get();
    RCLCPP_INFO(get_logger(), "recive get volume ask from app: %d.", response_->vol);
  } else if (request_->ask == TokenPassT::Request::ASK_XIAOAI_SWITCH_STATUS) {
    RCLCPP_INFO(get_logger(), "recive get xiaoai status from app.");
    RCLCPP_INFO(get_logger(), "ai_status_temp: %d.", ai_status_temp);
    if (ai_status_temp == AI_ONLINE_ON) {
      response_->flage = TokenPassT::Response::XIAOAI_ONLINE_ON;
      RCLCPP_INFO(get_logger(), "response XIAOAI_ONLINE_ON to app: %d.", AI_ONLINE_ON);
    } else if (ai_status_temp == AI_OFFLINE_ON) {
      response_->flage = TokenPassT::Response::XIAOAI_OFFLINE_ON;
      RCLCPP_INFO(get_logger(), "response XIAOAI_OFFLINE_ON to app: %d.", AI_OFFLINE_ON);
    } else if (ai_status_temp == AI_OFF || ai_status_temp == -1) {
      response_->flage = TokenPassT::Response::XIAOAI_OFF;
      RCLCPP_INFO(get_logger(), "response AI_OFFLINE_ON to app: %d.", AI_OFF);
    }
  } else {
    RCLCPP_ERROR(get_logger(), "recive unknow ask from app.");
  }
}

void VoiceCmd::extmon_goal_response_callback(
  std::shared_future<GoalHandleMonorderT::SharedPtr> future)
{
  auto goal_handle = future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
  } else {
    RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
  }
}

void VoiceCmd::play_goal_response_callback(
  std::shared_future<GoalHandlePlayorderT::SharedPtr> future)
{
  auto goal_handle = future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
  } else {
    RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
  }
}

void VoiceCmd::play_feedback_callback(
  GoalHandlePlayorderT::SharedPtr,
  const std::shared_ptr<const PlayorderT::Feedback> feedback)
{
  RCLCPP_INFO(
    get_logger(),
    "feedback header frame is: %s, time src: %d, time namesec is %d",
    feedback->feed.header.frame_id.c_str(),
    feedback->feed.header.stamp.sec,
    feedback->feed.header.stamp.nanosec);
  RCLCPP_INFO(this->get_logger(), "this feedback order id is: %d", feedback->feed.name.id);
}

void VoiceCmd::extmon_result_callback(const GoalHandleMonorderT::WrappedResult & result)
{
  RCLCPP_INFO(this->get_logger(), "Result received");

  if (result.result->succeed) {
    RCLCPP_INFO(
      this->get_logger(), "Goal was execute successfully, error code is: %d",
      result.result->err_code);
  } else {
    RCLCPP_ERROR(
      this->get_logger(), "Goal has failed to execute, error code is: %d",
      result.result->err_code);
  }
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      RCLCPP_INFO(this->get_logger(), "Goal was succeed.");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
      break;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
      break;
    default:
      RCLCPP_ERROR(this->get_logger(), "Unknown result code");
      break;
  }
}

void VoiceCmd::play_result_callback(const GoalHandlePlayorderT::WrappedResult & result)
{
  RCLCPP_INFO(this->get_logger(), "Result received");

  if (result.result->result.succeed) {
    RCLCPP_INFO(
      this->get_logger(), "Goal was execute successfully, error code is: %d",
      result.result->result.error);
  } else {
    RCLCPP_ERROR(
      this->get_logger(), "Goal has failed to execute, error code is: %d",
      result.result->result.error);
  }
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      RCLCPP_INFO(this->get_logger(), "Goal was succeed.");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
      break;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
      break;
    default:
      RCLCPP_ERROR(this->get_logger(), "Unknown result code");
      break;
  }
}

void VoiceCmd::send_play_goal(int order)
{
  if (!play_client_) {
    RCLCPP_ERROR(this->get_logger(), "Action client not initialized");
  }

  if (!play_client_->wait_for_action_server(timeout_)) {
    RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
    return;
  }
  auto goal_msg = PlayorderT::Goal();
  goal_msg.order.header = returnPlayHeader();
  goal_msg.order.user.id = interaction_msgs::msg::AudioUser::DEFAULT;
  goal_msg.order.name.id = order;
  RCLCPP_ERROR(this->get_logger(), "song name id: %d", goal_msg.order.name.id);

  play_client_->async_send_goal(goal_msg);
}

void VoiceCmd::send_extmon_goal(int id, double param)
{
  if (!ExtMon_client_) {
    RCLCPP_ERROR(this->get_logger(), "Action client not initialized");
  }

  if (!ExtMon_client_->wait_for_action_server(timeout_)) {
    RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
    return;
  }
  auto goal_msg = MonorderT::Goal();
  goal_msg.orderstamped.timestamp = this->get_clock()->now();
  switch (id) {
    case ORDER_STAND_UP:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_STAND_UP;
      goal_msg.orderstamped.para = param;
      break;
    case ORDER_PROSTRATE:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_PROSTRATE;
      goal_msg.orderstamped.para = param;
      break;
    case ORDER_COME_HERE:
      break;
    case ORDER_STEP_BACK:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_STEP_BACK;
      goal_msg.orderstamped.para = param;
      break;
    case ORDER_TURN_AROUND:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_TURN_AROUND;
      goal_msg.orderstamped.para = param;
      break;
    case ORDER_HI_FIVE:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_HI_FIVE;
      goal_msg.orderstamped.para = param;
      break;
    case ORDER_DANCE:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_DANCE;
      goal_msg.orderstamped.para = param;
      break;
    default:
      goal_msg.orderstamped.id = motion_msgs::msg::MonOrder::MONO_ORDER_NULL;
      goal_msg.orderstamped.para = 0.0;
  }
  RCLCPP_INFO(this->get_logger(), "Sending goal");

  ExtMon_client_->async_send_goal(goal_msg);
}


void VoiceCmd::send_wake_led_request(const std::chrono::seconds timeout, int order)
{
  auto request = std::make_shared<SensorNodeT::Request>();
  switch (order) {
    case WAKE_UP_SIGNAL:
      {
        request->command = SensorNodeT::Request::HEAD_LED_SKYBLUE_BREATH;
        request->priority = SensorNodeT::Request::TYPE_FUNCTION;
        request->timeout = BLINK_CLIENT_TIMEOUT;
        break;
      }
    case TALK_FEEDBACK_SIGNAL:
      {
        request->command = SensorNodeT::Request::HEAD_LED_SKYBLUE_ON;
        request->priority = SensorNodeT::Request::TYPE_FUNCTION;
        request->timeout = BLINK_CLIENT_TIMEOUT;
        break;
      }
    default:
      {
        request->command = SensorNodeT::Request::HEAD_LED_DARKBLUE_ON;
        request->priority = SensorNodeT::Request::TYPE_EFFECTS;
        request->timeout = BLINK_CLIENT_DEFAULT_TIMEOUT;
        break;
      }
  }
  request->clientid = BLINK_CLIENT_ID;
  auto future_return = wake_led_client_->async_send_request(request);
  std::future_status status = future_return.wait_for(timeout);
  if (status == std::future_status::ready) {
    if (future_return.get()->success) {
      RCLCPP_INFO(
        get_logger(),
        "wake up led blinks successfully.");
    } else {
      RCLCPP_WARN(
        get_logger(),
        "wake up led blinks failed.");
    }
  } else if (status == std::future_status::timeout) {
    RCLCPP_INFO(
      get_logger(),
      "Timeout.");
  } else {
    RCLCPP_INFO(
      get_logger(),
      "Invalid.");
  }
}

int VoiceCmd::ask_assistant_switch(const std::chrono::seconds timeout, int ask)
{
  auto request = std::make_shared<AssistantT::Request>();
  if (ask == ASK_XIAOAI_OFF) {
    request->ask = AssistantT::Request::ASK_XIAOAI_OFF;
  } else if (ask == ASK_XIAOAI_ONLINE_ON) {
    request->ask = AssistantT::Request::ASK_XIAOAI_ONLINE_ON;
  } else if (ask == ASK_XIAOAI_OFFLINE_ON) {
    request->ask = AssistantT::Request::ASK_XIAOAI_OFFLINE_ON;
  } else if (ask == ASK_XIAOAI_STATUS) {
    request->ask = AssistantT::Request::ASK_XIAOAI_STATUS;
  } else {
    RCLCPP_ERROR(
      get_logger(),
      "Invalid ask, return.");
    return 2;
  }
  auto future_return = ask_assistant_client_->async_send_request(request);
  std::future_status status = future_return.wait_for(timeout);
  if (status == std::future_status::ready) {
    if (future_return.get()->flag == AssistantT::Response::XIAOAI_OFF_SUCCEED) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai switch off succeed.");
      return 0;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_OFF_FAILED) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai switch off failed.");
      return -1;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_ONLINE_ON_SUCCEED) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai online on succeed.");
      return 0;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_ONLINE_ON_FAILED) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai online on failed.");
      return -1;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_OFFLINE_ON_SUCCEED) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai offline on succeed.");
      return 0;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_OFFLINE_ON_FAILED) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai offline on failed.");
      return -1;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_OFF) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai status: XIAOAI_OFF.");
      return 0;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_ONLINE_ON) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai status: XIAOAI_ONLINE_ON.");
      return 0;
    } else if (future_return.get()->flag == AssistantT::Response::XIAOAI_OFFLINE_ON) {
      RCLCPP_INFO(
        get_logger(),
        "xiaoai status: XIAOAI_OFFLINE_ON.");
      return 0;
    } else {
      RCLCPP_WARN(
        get_logger(),
        "unknown flag.");
      return -1;
    }
  } else if (status == std::future_status::timeout) {
    RCLCPP_INFO(
      get_logger(),
      "Timeout.");
    return -1;
  } else {
    RCLCPP_INFO(
      get_logger(),
      "Invalid.");
    return -1;
  }
}

void VoiceCmd::set_ai_require_status(int status)
{
  const auto status_f = toml::parse(AI_STATUS_FILE);
  std::string title_toml = toml::find<std::string>(status_f, "title");
  RCLCPP_INFO(get_logger(), "toml title: %s", title_toml.c_str());

  const toml::value toml_t{
    {"title", title_toml},
    {"xiaoai_status", status},
  };
  std::ofstream ofs(AI_STATUS_FILE, std::ofstream::out);
  ofs << toml_t;
  ofs.close();
}

int VoiceCmd::get_ai_require_status()
{
  const auto status_f = toml::parse(AI_STATUS_FILE);

  std::string title_toml = toml::find<std::string>(status_f, "title");
  RCLCPP_INFO(get_logger(), "toml title: %s", title_toml.c_str());
  std::int64_t ai_status_toml = toml::find<std::int64_t>(status_f, "xiaoai_status");
  RCLCPP_INFO(get_logger(), "toml : %ld", ai_status_toml);

  return ai_status_toml;
}

void VoiceCmd::volume_set(int vol)
{
  int volume;
  if (vol == 100) {
    volume = 0;
  } else {
    volume = vol;
  }
  player_->SetVolume(vol_value[volume]);
  const auto volume_toml = toml::parse(VOLUME_FILE);
  const toml::value toml_t{
    {"title", "volume status"},
    {"volume", volume},
  };
  std::ofstream ofs(VOLUME_FILE, std::ofstream::out);
  ofs << toml_t;
  ofs.close();
}

int VoiceCmd::volume_get()
{
  int gears;
  int vol = player_->GetVolume();
  if (vol == 0) {
    gears = 100;
    return gears;
  } else {
    for (int i = 0; i < 10; i++) {
      if (vol == vol_value[i + 1]) {
        gears = i + 1;
        return gears;
      }
    }
  }
  return 0;
}

int64_t VoiceCmd::volume_check()
{
  const auto volume_toml = toml::parse(VOLUME_FILE);

  std::int64_t volume_value = toml::find<std::int64_t>(volume_toml, "volume");
  RCLCPP_INFO(get_logger(), "volume_value(toml) : %ld", volume_value);

  return volume_value;
}

void VoiceCmd::player_init()
{
  player_ =
    std::make_shared<AudioPlayer>(2, std::bind(&VoiceCmd::play_callback, this), AUDIO_GROUP);
}
void VoiceCmd::player_handle(int order_name)
{
  std::string dir = WAV_DIR + std::to_string(order_name) + ".wav";
  player_->AddPlay(dir.c_str());
}
void VoiceCmd::play_callback()
{
  play_end = true;
  RCLCPP_INFO(get_logger(), "play_callback\n");
}

int VoiceCmd::get_ai_status()
{
  return ai_status_temp;
}

bool VoiceCmd::fac_test_flage(const std::string & name)
{
  return access(name.c_str(), F_OK) != -1;
}

}  // namespace cyberdog_audio

// RCLCPP_COMPONENTS_REGISTER_NODE(cyberdog_voicecmd::VoiceCmdActionClient)
