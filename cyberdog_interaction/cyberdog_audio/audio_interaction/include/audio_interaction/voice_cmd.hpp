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

#ifndef AUDIO_INTERACTION__VOICE_CMD_HPP_
#define AUDIO_INTERACTION__VOICE_CMD_HPP_

#include <unistd.h>

#include <fstream>
#include <memory>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <utility>
#include <mutex>
#include <condition_variable>

#include "net/if.h"
#include "sys/ioctl.h"
#include "netinet/in.h"

#include "audio_interaction/md5.hpp"
#include "toml11/toml.hpp"

#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int8.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"
#include "cyberdog_utils/action_server.hpp"
#include "cyberdog_utils/Enums.hpp"

#include "motion_msgs/msg/mon_order.hpp"

#include "ception_msgs/srv/check_power.hpp"
#include "interaction_msgs/srv/token_pass.hpp"
#include "interaction_msgs/msg/token.hpp"
#include "interaction_msgs/srv/ask_assistant.hpp"

#include "ception_msgs/srv/sensor_detection_node.hpp"

#include "interaction_msgs/msg/touch.hpp"

#include "interaction_msgs/action/audio_play.hpp"
#include "motion_msgs/action/ext_mon_order.hpp"
#include "action_msgs/msg/goal_status.hpp"

#include "audio_base/audio_player.hpp"

#define TOKEN_FILE "/opt/ros2/cyberdog/data/token.toml"
#define AI_STATUS_FILE "/opt/ros2/cyberdog/data/ai_status.toml"
#define VOLUME_FILE "/opt/ros2/cyberdog/data/volume.toml"
#define WAV_DIR "/opt/ros2/cyberdog/data/wav/"
#define FAC_TEST_FILE "/opt/ros2/cyberdog/ai_conf/ai_off"
#define WAKE_UP_SIGNAL                  100
#define TALK_FEEDBACK_SIGNAL            101
#define BLINK_CLIENT_ID                 3
#define BLINK_CLIENT_PRIORITY           2
#define BLINK_CLIENT_TIMEOUT            10000000000
#define BLINK_CLIENT_DEFAULT_TIMEOUT    100
#define CMD_RESULT_BUF_SIZE             1024
#define DEFAULT_SDL_CHANNEL             2
#define AUDIO_GROUP                     1

namespace cyberdog_audio
{
class VoiceCmd : public cyberdog_utils::LifecycleNode
{
public:
  VoiceCmd();
  ~VoiceCmd();
  std::string getDeviceId();
  using MonorderT = motion_msgs::action::ExtMonOrder;
  using GoalHandleMonorderT = rclcpp_action::ClientGoalHandle<MonorderT>;

  using LifecycleNode_T = cyberdog_utils::LifecycleNode;
  using PlayorderT = interaction_msgs::action::AudioPlay;
  using PlayServer = cyberdog_utils::ActionServer<PlayorderT, LifecycleNode_T>;
  using GoalHandlePlayorderT = rclcpp_action::ClientGoalHandle<PlayorderT>;


  using TokenPassT = interaction_msgs::srv::TokenPass;
  using AssistantT = interaction_msgs::srv::AskAssistant;
  using SensorNodeT = ception_msgs::srv::SensorDetectionNode;
  using player_ptr = std::shared_ptr<AudioPlayer>;

  std::shared_ptr<std::string> file_name_ptr_;

protected:
  enum audio_user
  {
    USERNULL = 0,
    STARTER,
    POWER,
    TOUCH,
    CAMERA,
    APP,
    BLUETOOTH,
    WIFI,
    XIAOAI
  };

  enum ai_switch_ask
  {
    ASK_XIAOAI_OFF = 0,
    ASK_XIAOAI_ONLINE_ON,
    ASK_XIAOAI_OFFLINE_ON,
    ASK_XIAOAI_STATUS
  };

  enum ai_switch
  {
    AI_OFF =  0,
    AI_ONLINE_ON,
    AI_OFFLINE_ON,
  };

  enum token_state
  {
    TOKEN_READY = 1,
    DEVICE_ID_READY
  };

  enum order_name
  {
    ORDER_NULL = 0,
    ORDER_STAND_UP,
    ORDER_PROSTRATE,
    ORDER_COME_HERE,
    ORDER_STEP_BACK,
    ORDER_TURN_AROUND,
    ORDER_HI_FIVE,
    ORDER_DANCE
  };

  std::array<int, 11> vol_value = {0, 15, 30, 45, 60, 75, 90, 100, 110, 120, 128};

  bool play_end = false;
  bool voice_end = false;
  int ai_status_temp = -1;
  player_ptr player_;

  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

  /* Function */
  void PublishActionOrder();
  void PublishOwnerStatus();
  void PublishNetStatus();
  void PublishTokenReady(int order);
  void PublishAiSwitch(int order);

  // void SubscripTestOrder(const std_msgs::msg::Int8::SharedPtr order);
  void SubscripAiorder(const std_msgs::msg::Int8::SharedPtr order);
  std_msgs::msg::Header returnMotionHeader();
  std_msgs::msg::Header returnPlayHeader();
  std_msgs::msg::Header returnExtMonHeader();
  std_msgs::msg::Header returnSensorNodeTHeader();
  std_msgs::msg::Header returnBmsReqHeader();

  void check_play_request();

  void check_app_order(
    const std::shared_ptr<rmw_request_id_t> request_header_,
    const std::shared_ptr<TokenPassT::Request> request_,
    std::shared_ptr<TokenPassT::Response> response_);
  void send_extmon_goal(int order, double param);
  void extmon_goal_response_callback(std::shared_future<GoalHandleMonorderT::SharedPtr> future);
  void extmon_feedback_callback(
    GoalHandleMonorderT::SharedPtr,
    const std::shared_ptr<const MonorderT::Feedback> feedback);
  void extmon_result_callback(const GoalHandleMonorderT::WrappedResult & result);
  void send_wake_led_request(const std::chrono::seconds timeout, int order);

  void send_play_goal(int order);
  void play_goal_response_callback(std::shared_future<GoalHandlePlayorderT::SharedPtr> future);
  void play_feedback_callback(
    GoalHandlePlayorderT::SharedPtr,
    const std::shared_ptr<const PlayorderT::Feedback> feedback);
  void play_result_callback(const GoalHandlePlayorderT::WrappedResult & result);

  void player_init();
  void player_handle(int id);
  void play_callback();
  void volume_set(int vol);
  int volume_get();
  int64_t volume_check();
  int get_ai_status();
  int Detectwifi();
  int ExecuteCMD(const char * cmd, char * result);
  int get_ai_require_status();
  void set_ai_require_status(int status);
  int ask_assistant_switch(const std::chrono::seconds timeout, int ask);
  bool fac_test_flage(const std::string & name);

private:
  /* Clock */
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};
  builtin_interfaces::msg::Time action_start_time_;
  rclcpp::TimerBase::SharedPtr timer_net_;

  /* Para */
  std::chrono::seconds timeout_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  /* Subscriber */
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr ai_order_sub_;

  /* Publisher */
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Int8>::SharedPtr net_status_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Int8>::SharedPtr token_ready_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Int8>::SharedPtr ai_switch_pub_;

  /* Service */
  rclcpp::Service<TokenPassT>::SharedPtr token_server_;
  rclcpp::Client<SensorNodeT>::SharedPtr wake_led_client_;
  rclcpp::Client<AssistantT>::SharedPtr ask_assistant_client_;

  /* Action */
  std::unique_ptr<PlayServer> play_server_;
  std::shared_ptr<PlayorderT> orderback_;
  int loop_rate_;

  rclcpp::Node::SharedPtr play_client_node_;
  rclcpp_action::Client<PlayorderT>::SharedPtr play_client_;

  rclcpp::Node::SharedPtr ExtMon_client_node_;
  rclcpp_action::Client<MonorderT>::SharedPtr ExtMon_client_;

/* TEMP */
  rclcpp::TimerBase::SharedPtr timer_temp_;
  rclcpp::Node::SharedPtr temp_node_;

  /* Thread */
  std::shared_ptr<std::thread> play_thread_;
  std::shared_ptr<std::thread> wake_play_thread_;
  std::shared_ptr<std::thread> order_play_thread_;
};  // class VoiceCmd
}  // namespace cyberdog_audio

#endif  // AUDIO_INTERACTION__VOICE_CMD_HPP_
