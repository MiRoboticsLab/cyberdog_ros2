// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd.
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

#define LOG_TAG "NCSClient"
#include <memory>
#include "camera_service/ncs_client.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

static const char * AUDIO_SERVER_NAME = "audio_play";
static const char * LED_SERVER_NAME = "cyberdog_led";

static int gSoundAudioArrays[SoundTypeNone] =
{
  [SoundShutter] = 8,
  [SoundRecordStart] = 122,
  [SoundRecording] = 9,
  [SoundLiveStart] = 118,
  [SoundFaceAddStart] = 103,
  [SoundFaceNumWarn] = 113,
  [SoundFacePoseWarn] = 115,
  [SoundFaceDistWarn] = 114,
  [SoundFaceAddEnd] = 109,
  [SoundFaceAddFailed] = 108,
  [SoundFaceDetect] = 5,
};

NCSClient & NCSClient::getInstance()
{
  static NCSClient s_instance;

  return s_instance;
}

NCSClient::NCSClient()
{
  CAM_INFO("Create");

  m_playerClient =
    CameraManager::getInstance()->create_action_client<AudioPlayT>(AUDIO_SERVER_NAME);
  m_ledClient = CameraManager::getInstance()->create_service_client<LedServiceT>(LED_SERVER_NAME);
}

NCSClient::~NCSClient()
{
  CAM_INFO("Destroy");
}

bool NCSClient::play(SoundType type)
{
  bool ret = true;

  if (gSoundAudioArrays[type] > 0) {
    ret = sendGoal(gSoundAudioArrays[type]);
  }

  return ret;
}

bool NCSClient::requestLed(bool on)
{
  bool ret = true;

  CAM_INFO("Request led on/off: %d", on);
  auto request = std::make_shared<LedServiceT::Request>();
  if (on) {
    request->command = LedServiceT::Request::HEAD_LED_SKYBLUE_ON;
  } else {
    request->command = LedServiceT::Request::HEAD_LED_DARKBLUE_ON;
  }
  request->clientid = 0;

  if (!m_ledClient->wait_for_service(std::chrono::seconds(1))) {
    CAM_ERR("Waiting for led service timeout.");
    return false;
  }

  m_ledClient->async_send_request(request);

  return ret;
}

bool NCSClient::sendGoal(int audio_id)
{
  using namespace std::placeholders;

  // sound playing should not block service call, so we just wait for a short time.
  if (!m_playerClient->wait_for_action_server(std::chrono::seconds(1))) {
    CAM_ERR("Audio server not available after waiting, id: %d", audio_id);
    return false;
  }

  auto goal_msg = AudioPlayT::Goal();
  goal_msg.order.name.id = audio_id;
  goal_msg.order.user.id = 4;  // camera

  CAM_INFO("Send audio play request %d", audio_id);

  auto send_goal_options = rclcpp_action::Client<AudioPlayT>::SendGoalOptions();
  send_goal_options.goal_response_callback =
    std::bind(&NCSClient::goal_response_callback, this, _1);
  send_goal_options.feedback_callback =
    std::bind(&NCSClient::feedback_callback, this, _1, _2);
  send_goal_options.result_callback =
    std::bind(&NCSClient::result_callback, this, _1);
  m_playerClient->async_send_goal(goal_msg, send_goal_options);

  return true;
}

void NCSClient::goal_response_callback(std::shared_future<GoalHandleAudio::SharedPtr> future)
{
  auto goal_handle = future.get();
  if (!goal_handle) {
    CAM_INFO("Goal was rejected by server");
  } else {
    CAM_INFO("Goal accepted by server, waiting for result");
  }
}

void NCSClient::feedback_callback(
  GoalHandleAudio::SharedPtr,
  const std::shared_ptr<const AudioPlayT::Feedback> feedback)
{
  CAM_DEBUG("status: %u", feedback->feed.status);
}

void NCSClient::result_callback(const GoalHandleAudio::WrappedResult & result)
{
  CAM_INFO("result: %u", result.result->result.error);
}

}  // namespace cyberdog_camera
