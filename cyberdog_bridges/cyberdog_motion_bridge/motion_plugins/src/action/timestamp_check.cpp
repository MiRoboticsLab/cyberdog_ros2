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

#include <chrono>
#include <string>

#include "cyberdog_motion_plugins/action/timestamp_check.hpp"

namespace cyberdog_motion_plugins
{
TimeStampCheck::TimeStampCheck(
  const std::string & name,
  const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(name, conf)
{
  getInput("Tree_Name", tree_name_);
  last_ts_str_ = tree_name_ + std::string("_last_trigger_time");
  new_ts_str_ = tree_name_ + std::string("_new_trigger_time");
}

inline BT::NodeStatus TimeStampCheck::tick()
{
  auto rtn_ = BT::NodeStatus::FAILURE;

  int64_t last_trigger_time_;
  int64_t new_trigger_time_;

  if (!getInput(last_ts_str_, last_trigger_time_).has_value() ||
    !getInput(new_ts_str_, new_trigger_time_).has_value())
  {
    std::cout << "Got trigger timestamp failed" << std::endl;
    return rtn_;
  }

  return (new_trigger_time_ > last_trigger_time_) ?
         BT::NodeStatus::SUCCESS :
         BT::NodeStatus::FAILURE;
}
}  // namespace cyberdog_motion_plugins

#include "behaviortree_cpp_v3/bt_factory.h"
BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<cyberdog_motion_plugins::TimeStampCheck>("TimeStampCheck");
}
