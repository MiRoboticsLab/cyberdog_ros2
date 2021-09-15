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

#ifndef CYBERDOG_UTILS__ENUMS_HPP_
#define CYBERDOG_UTILS__ENUMS_HPP_

#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include "lifecycle_msgs/msg/state.hpp"
#include "lifecycle_msgs/msg/transition.hpp"

namespace cyberdog_utils
{

enum BTStatus
{
  SUCCEEDED = 0,
  FAILED = 1,
  CANCELED = 2
};

enum NodeCheckType
{
  CHECK_TO_START = 1,
  CHECK_TO_PAUSE = 2,
  CHECK_TO_CLEANUP = 3
};

static std::map<std::uint8_t, std::string> transition_label_map_ =
{
  {lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE, std::string("Configure...")},
  {lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP, std::string("Clean up...")},
  {lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE, std::string("Activate...")},
  {lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE, std::string("Deactivate...")},
  {lifecycle_msgs::msg::Transition::TRANSITION_UNCONFIGURED_SHUTDOWN, std::string("Shut down...")}
};
static std::unordered_map<std::uint8_t, std::uint8_t> transition_state_map_ =
{
  {lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE,
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE},
  {lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP,
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED},
  {lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE,
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE},
  {lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE,
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE},
  {lifecycle_msgs::msg::Transition::TRANSITION_UNCONFIGURED_SHUTDOWN,
    lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED}
};


// const string
static const char dictator_[] = "dictator";

enum GaitChangePriority
{
  LOCK_DETECT  = 255,
  MODE_TRIG    = 254,
  GAIT_TRIG    = 253,
  ORDER_REQ    = 252
};

}  // namespace cyberdog_utils

#endif  // CYBERDOG_UTILS__ENUMS_HPP_
