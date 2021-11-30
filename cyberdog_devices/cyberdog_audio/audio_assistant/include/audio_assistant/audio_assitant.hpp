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

// #include <stdio.h>
// #include <stdlib.h>
// #include <cstdlib>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <sys/types.h>
// #include <unistd.h>

// using namespace std;
// int ai_vpm_engine_setup(void);

#ifndef AUDIO_ASSISTANT__AUDIO_ASSITANT_HPP_
#define AUDIO_ASSISTANT__AUDIO_ASSITANT_HPP_

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
#include <fstream>
#include <condition_variable>

#include "net/if.h"
#include "sys/ioctl.h"
#include "netinet/in.h"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int8.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"
#include "interaction_msgs/srv/ask_assistant.hpp"

/*ai engine*/
#include "xiaoai_sdk/aivs/DigestUtils.h"
#include "xiaoai_sdk/aivs/AllEnums.h"

void updateToken2Aivs(std::string token_access, std::string token_refresh);
int set_ai_recorde_switch(bool on);
int set_ai_keywordengine_switch(bool on);
int set_aivs_onlie_switch(bool on);
void set_useTestDeviceId(bool useTestDeviceId);

#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus
#include <alsa/asoundlib.h>
#ifdef __cplusplus
}
#endif  // __cplusplus

#define TOKEN_FILE "/opt/ros2/cyberdog/data/token.toml"

namespace cyberdog_audio
{
class AudioAssistant : public cyberdog_utils::LifecycleNode
{
public:
  AudioAssistant();
  ~AudioAssistant();

protected:
  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

  /*raw data recorde task*/
  std::shared_ptr<std::thread> threadMainCapture;
  /*wakeup task*/
  std::shared_ptr<std::thread> threadAiWakeup;
  /*native asr task*/
  std::shared_ptr<std::thread> threadNativeAsr;
  /*set up online sdk*/
  std::shared_ptr<std::thread> threadAiOnline;
  /*set tts handler task*/
  std::shared_ptr<std::thread> threadTts;
  /*set up wdog*/
  std::shared_ptr<std::thread> threadWdog;

  void AiMainCaptureTask(void);
  void AiNativeAsrTask(void);
  void AiWakeupTask(void);
  void AiOnlineTask(void);
  void AiTtsTask(void);
  void AiWdogTask(void);

  void enableAudioAssistantThread(void);
  void disableAudioAssistantThread(void);

  /*transfer/receive data*/
  void pub_robot_orders();
  void update_network_status(const std_msgs::msg::Int8::SharedPtr status);
  void update_token_status(const std_msgs::msg::Int8::SharedPtr status);
  void update_aiswitch_status(const std_msgs::msg::Int8::SharedPtr status);
  void check_switch_ask(
    const std::shared_ptr<rmw_request_id_t> request_header_,
    const std::shared_ptr<interaction_msgs::srv::AskAssistant::Request> request_,
    std::shared_ptr<interaction_msgs::srv::AskAssistant::Response> response_);
  void Wdog(void);

  int setWdogRunStatus(bool on);
  bool getWdogRunStatus(void);

  int mActionOrder;
  int mNetWorkChangedFrom;
  int mNetWorkChangedTo;
  static bool mWdogRunStatus;

private:
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  /* Subscriber network status*/
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr network_status;
  /* Publisher motion order*/
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Int8>::SharedPtr robot_order;
  /* Subscriber token status*/
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr token_status;
  /* Subscriber ai switch status*/
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr aiswitch_status;
  /* Service: ai_switch */
  rclcpp::Service<interaction_msgs::srv::AskAssistant>::SharedPtr switch_server_;
};  // class AudioAssistant
}  // namespace cyberdog_audio

#endif  // AUDIO_ASSISTANT__AUDIO_ASSITANT_HPP_
