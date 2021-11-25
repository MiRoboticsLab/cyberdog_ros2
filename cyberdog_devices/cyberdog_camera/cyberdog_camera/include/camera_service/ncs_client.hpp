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

#ifndef CAMERA_SERVICE__NCS_CLIENT_HPP_
#define CAMERA_SERVICE__NCS_CLIENT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>
#include "./ros2_service.hpp"

namespace cyberdog_camera
{

enum SoundType
{
  SoundShutter = 0,
  SoundRecordStart,
  SoundRecording,
  SoundLiveStart,
  SoundFaceAddStart,
  SoundFaceNumWarn,
  SoundFacePoseWarn,
  SoundFaceDistWarn,
  SoundFaceAddEnd,
  SoundFaceAddFailed,
  SoundFaceDetect,
  SoundTypeNone,
};

class NCSClient
{
public:
  using GoalHandleAudio = rclcpp_action::ClientGoalHandle<AudioPlayT>;

  static NCSClient & getInstance();
  bool play(SoundType type);
  bool requestLed(bool on);

private:
  NCSClient();
  ~NCSClient();

  bool sendGoal(int audio_id);
  void goal_response_callback(std::shared_future<GoalHandleAudio::SharedPtr> future);
  void feedback_callback(
    GoalHandleAudio::SharedPtr,
    const std::shared_ptr<const AudioPlayT::Feedback> feedback);
  void result_callback(const GoalHandleAudio::WrappedResult & result);

  rclcpp_action::Client<AudioPlayT>::SharedPtr m_playerClient;
  rclcpp::Client<LedServiceT>::SharedPtr m_ledClient;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_SERVICE__NCS_CLIENT_HPP_
