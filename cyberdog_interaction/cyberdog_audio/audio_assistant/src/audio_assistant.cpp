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
#include <memory>
#include <utility>

#include "toml11/toml.hpp"
#include "audio_assistant/audio_assitant.hpp"
#include "audio_assistant/combiner.hpp"
#include "audio_base/debug/ai_debugger.hpp"
#include "audio_assistant/audio_token.hpp"

void ai_vpm_task(void);
void ai_native_asr_task();
void ai_online_task(void);
void ai_wdog_task(void);
int aivseMsgHandler(void);
int set_network_status(int status);
void ttsHandlerTask(void);

int setMainCaptureRunStatus(bool on);
int setAiWakeupRunStatus(bool on);
int setNativeAsrRunStatus(bool on);
int setAiOnlineRunStatus(bool on);
int setAivsTtsRunStatus(bool on);
int set_audiodata_silence_mode(bool);

namespace cyberdog_audio
{
bool AudioAssistant::mWdogRunStatus = true;

AudioAssistant::AudioAssistant()
: cyberdog_utils::LifecycleNode("audio_assistant")
{
  RCLCPP_INFO(get_logger(), "Creating AudioAssistant.");
}

AudioAssistant::~AudioAssistant()
{
  RCLCPP_INFO(get_logger(), "Destroying AudioAssistant");
}

cyberdog_utils::CallbackReturn AudioAssistant::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "AudioAssistant configuring");

  std::int64_t audioAssitantForceEnable = 0;
  std::int64_t rawDumpEnable = 0;
  std::int64_t asrDumpEnable = 0;
  std::int64_t silenceMode = 0;
  std::int64_t useTesttoken = 0;
  mNetWorkChangedFrom = -1;
  mNetWorkChangedTo = -1;

  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  /*get action order and network status*/
  robot_order = this->create_publisher<std_msgs::msg::Int8>(
    "robot_order",
    rclcpp::SystemDefaultsQoS());
  network_status = this->create_subscription<std_msgs::msg::Int8>(
    "network_status",
    rclcpp::SystemDefaultsQoS(),
    std::bind(&AudioAssistant::update_network_status, this, std::placeholders::_1));
  /*for token update*/
  token_status = this->create_subscription<std_msgs::msg::Int8>(
    "token_get",
    rclcpp::SystemDefaultsQoS(),
    std::bind(&AudioAssistant::update_token_status, this, std::placeholders::_1));
  /*for ai module switch update*/
  aiswitch_status = this->create_subscription<std_msgs::msg::Int8>(
    "ai_switch",
    rclcpp::SystemDefaultsQoS(),
    std::bind(&AudioAssistant::update_aiswitch_status, this, std::placeholders::_1));

  switch_server_ = this->create_service<interaction_msgs::srv::AskAssistant>(
    "ai_switch",
    std::bind(
      &AudioAssistant::check_switch_ask, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  /*debug config*/
  const auto data = toml::parse("/opt/ros2/cyberdog/data/audio_debug.toml");
  auto audioDebugger = std::make_shared<std::audioDebugger>();

  audioAssitantForceEnable = toml::find<std::int64_t>(data, "config_audio_force_enable");
  audioDebugger->setConfig(AUDIO_DEBUG_CONFIG_AS_FORCE_ENABLE, audioAssitantForceEnable);

  rawDumpEnable = toml::find<std::int64_t>(data, "config_audio_dump_raw");
  audioDebugger->setConfig(AUDIO_DEBUG_CONFIG_DUMPRAW, rawDumpEnable);

  asrDumpEnable = toml::find<std::int64_t>(data, "config_audio_dump_asr");
  audioDebugger->setConfig(AUDIO_DEBUG_CONFIG_DUMPASR, asrDumpEnable);

  silenceMode = toml::find<std::int64_t>(data, "config_audio_silence");
  audioDebugger->setConfig(AUDIO_DEBUG_CONFIG_SILENCE, silenceMode);

  useTesttoken = toml::find<std::int64_t>(data, "config_audio_testtoken");
  audioDebugger->setConfig(AUDIO_DEBUG_CONFIG_TESTTOKEN, useTesttoken);
  set_useTestDeviceId(useTesttoken);
  RCLCPP_INFO(
    get_logger(),
    "audioconfig raw[%d],asr[%d],silenceMode[%d],useTesttoken[%d]",
    rawDumpEnable, asrDumpEnable, silenceMode, useTesttoken);

  /*for token access*/
  auto audioToken = std::make_shared<std::audioToken>();
  audioToken->updateToken((useTesttoken == 0) ? false : true);
  updateToken2Aivs(*(audioToken->mToken_access), *(audioToken->mToken_refresh));

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn AudioAssistant::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "AudioAssistant activating");

  robot_order->on_activate();

  setWdogRunStatus(true);
  setAivsTtsRunStatus(true);
  setAiOnlineRunStatus(true);
  setNativeAsrRunStatus(true);
  setAiWakeupRunStatus(true);
  setMainCaptureRunStatus(true);

  /*raw data recorde task*/
  threadMainCapture = std::make_shared<std::thread>(&AudioAssistant::AiMainCaptureTask, this);
  threadMainCapture->detach();
  /*wakeup task*/
  threadAiWakeup = std::make_shared<std::thread>(&AudioAssistant::AiWakeupTask, this);
  threadAiWakeup->detach();
  /*native asr task*/
  threadNativeAsr = std::make_shared<std::thread>(&AudioAssistant::AiNativeAsrTask, this);
  threadNativeAsr->detach();
  /*setup online sdk*/
  threadAiOnline = std::make_shared<std::thread>(&AudioAssistant::AiOnlineTask, this);
  threadAiOnline->detach();
  /*setup tts handler task*/
  threadTts = std::make_shared<std::thread>(&AudioAssistant::AiTtsTask, this);
  threadTts->detach();
  /*setup audio assistant wdog*/
  threadWdog = std::make_shared<std::thread>(&AudioAssistant::AiWdogTask, this);
  threadWdog->detach();

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn AudioAssistant::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "AudioAssistant deactivating");
  robot_order->on_deactivate();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn AudioAssistant::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "AudioAssistant cleaningup");
  switch_server_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn AudioAssistant::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "AudioAssistant shutting down");
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

/*task loop*/
void AudioAssistant::AiMainCaptureTask(void)
{
  recorder_work_loop();
}

void AudioAssistant::AiWakeupTask(void)
{
  ai_vpm_task();
}

void AudioAssistant::AiNativeAsrTask(void)
{
  ai_native_asr_task();
}

void AudioAssistant::AiOnlineTask(void)
{
  RCLCPP_INFO(
    get_logger(),
    "AudioAssistant AiOnlineTask starting.");
  ai_online_task();
}

void AudioAssistant::AiTtsTask(void)
{
  RCLCPP_INFO(
    get_logger(),
    "AudioAssistant AiTtsTask starting.");
  ttsHandlerTask();
}

void AudioAssistant::AiWdogTask(void)
{
  RCLCPP_INFO(
    get_logger(),
    "AudioAssistant Wdog starting.");
  while (true) {
    if (getWdogRunStatus() == false) {
      usleep(1000 * 500);
      continue;
    }
    mActionOrder = -1;
    mActionOrder = aivseMsgHandler();

    if (mActionOrder > 0) {
      RCLCPP_INFO(
        get_logger(),
        "AudioAssistant pub_robot_orders[%d]", mActionOrder);
      pub_robot_orders();
    }
  }
}

void AudioAssistant::enableAudioAssistantThread(void)
{
  RCLCPP_INFO(
    get_logger(),
    "AudioAssistant enableAudioAssistantThread()");

  set_audiodata_silence_mode(false);
}

void AudioAssistant::disableAudioAssistantThread(void)
{
  auto audioDebugger = std::make_shared<std::audioDebugger>();

  if (audioDebugger->getConfig(AUDIO_DEBUG_CONFIG_AS_FORCE_ENABLE)) {
    RCLCPP_INFO(get_logger(), "AudioAssistant force enable, ignore this disable cmd");
    return;
  } else {
    RCLCPP_INFO(get_logger(), "AudioAssistant disable");
    set_audiodata_silence_mode(true);
  }
}

void AudioAssistant::pub_robot_orders()
{
  auto msg = std::make_unique<std_msgs::msg::Int8>();
  msg->data = mActionOrder;

  if (!robot_order->is_activated()) {
    RCLCPP_ERROR(
      get_logger(),
      "Lifecycle publisher is currently inactive. Messages are not published.");
  } else {
    RCLCPP_INFO(
      get_logger(),
      "Publish audio orders[%d]", msg->data);
    robot_order->publish(std::move(msg));
  }
}

void AudioAssistant::update_network_status(const std_msgs::msg::Int8::SharedPtr status)
{
  RCLCPP_INFO(get_logger(), "Receive network status %d.", status->data);
  if (status->data == 1 || status->data == -1) {
    /*The orig mNetWorkChangedFrom value is -1*/
    mNetWorkChangedFrom = mNetWorkChangedTo;
    mNetWorkChangedTo = status->data;
    if (mNetWorkChangedTo != mNetWorkChangedFrom) {
      RCLCPP_INFO(
        get_logger(), "mNetWorkChangedFrom[%d], mNetWorkChangedTo[%d]",
        mNetWorkChangedFrom, mNetWorkChangedTo);
      set_network_status(status->data);
    }

  } else {
    RCLCPP_ERROR(get_logger(), "Receive network status invalid valeu %d.", status->data);
  }
}

void AudioAssistant::update_token_status(const std_msgs::msg::Int8::SharedPtr status)
{
  RCLCPP_INFO(get_logger(), "Receive token/deviceid status %d.", status->data);
  if (status->data == 1) {
    RCLCPP_INFO(get_logger(), "NewToken is Ready!!!");

    auto audioToken = std::make_shared<std::audioToken>();
    audioToken->updateToken(false);        /*false: do not use testtoken*/
    updateToken2Aivs(*(audioToken->mToken_access), *(audioToken->mToken_refresh));

    set_network_status(1);
  } else if (status->data == 2) {
    // device id
  } else {
    RCLCPP_ERROR(get_logger(), "Receive token status invalid valeu %d.", status->data);
  }
}

void AudioAssistant::update_aiswitch_status(const std_msgs::msg::Int8::SharedPtr status)
{
  RCLCPP_INFO(get_logger(), "Receive aiswitch status %d.", status->data);
  auto audioDebugger = std::make_shared<std::audioDebugger>();

  if (audioDebugger->getConfig(AUDIO_DEBUG_CONFIG_AS_FORCE_ENABLE)) {
    RCLCPP_INFO(get_logger(), "AudioAssistant force enable, ignore aiswitch event");
    return;
  }

  switch (status->data) {
    case 0x0000:
      disableAudioAssistantThread();
      break;
    case 0x0001:
      enableAudioAssistantThread();
      break;
    default:
      break;
  }
}

void AudioAssistant::check_switch_ask(
  const std::shared_ptr<rmw_request_id_t> request_header_,
  const std::shared_ptr<interaction_msgs::srv::AskAssistant::Request> request_,
  std::shared_ptr<interaction_msgs::srv::AskAssistant::Response> response_)
{
  RCLCPP_INFO(get_logger(), "switch check and update entry.");
  (void)request_header_;
  if (request_->ask == interaction_msgs::srv::AskAssistant::Request::ASK_XIAOAI_OFF) {
    // add logic

    // add result
    response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_OFF_SUCCEED;
    // response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_OFF_FAILED;

  } else if (request_->ask == interaction_msgs::srv::AskAssistant::Request::ASK_XIAOAI_ONLINE_ON) {
    // add logic

    // add result
    response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_ONLINE_ON_SUCCEED;
    // response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_ONLINE_ON_FAILED;
  } else if (request_->ask == interaction_msgs::srv::AskAssistant::Request::ASK_XIAOAI_OFFLINE_ON) {
    // add logic

    // add result
    response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_OFFLINE_ON_SUCCEED;
    // response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_OFFLINE_ON_FAILED;
  } else if (request_->ask == interaction_msgs::srv::AskAssistant::Request::ASK_XIAOAI_STATUS) {
    // add logic

    // add result
    response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_OFF;
    // response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_ONLINE_ON;
    // response_->flag = interaction_msgs::srv::AskAssistant::Response::XIAOAI_OFFLINE_ON;
  } else {
    RCLCPP_ERROR(get_logger(), "recive unknow ask.");
  }
}

int AudioAssistant::setWdogRunStatus(bool on)
{
  RCLCPP_INFO(get_logger(), "in setWdogRunStatus()");
  mWdogRunStatus = on;
  return static_cast<int>(mWdogRunStatus);
}

bool AudioAssistant::getWdogRunStatus()
{
  RCLCPP_INFO(get_logger(), "in getWdogRunStatus()");
  return mWdogRunStatus;
}

}  // namespace cyberdog_audio

// RCLCPP_COMPONENTS_REGISTER_NODE(cyberdog_AudioAssistant::AudioAssistantActionClient)
