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

#ifndef CYBERDOG_MOTION_PLUGINS__ACTION__TIMESTAMP_CHECK_HPP_
#define CYBERDOG_MOTION_PLUGINS__ACTION__TIMESTAMP_CHECK_HPP_

#include <chrono>
#include <string>

#include "behaviortree_cpp_v3/action_node.h"

namespace cyberdog_motion_plugins
{

class TimeStampCheck : public BT::ActionNodeBase
{
public:
  TimeStampCheck(
    const std::string & name,
    const BT::NodeConfiguration & conf);

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::int64_t>(
        last_ts_str_,
        std::chrono::system_clock::now().time_since_epoch().count(),
        "Timestamp stored before"),
      BT::InputPort<std::int64_t>(
        last_ts_str_,
        std::chrono::system_clock::now().time_since_epoch().count(),
        "Timestamp name to check")
    };
  }

private:
  BT::NodeStatus tick() override;

  inline static std::string tree_name_;
  inline static std::string last_ts_str_;
  inline static std::string new_ts_str_;
};
}  // namespace cyberdog_motion_plugins

#endif  // CYBERDOG_MOTION_PLUGINS__ACTION__TIMESTAMP_CHECK_HPP_
