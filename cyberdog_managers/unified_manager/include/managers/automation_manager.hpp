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

#ifndef MANAGERS__AUTOMATION_MANAGER_HPP_
#define MANAGERS__AUTOMATION_MANAGER_HPP_

// C++ headers
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
// ROS headers
#include "manager_utils/cascade_manager.hpp"
#include "motion_msgs/msg/mode.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cyberdog
{
namespace manager
{

using CallbackReturn_T = cyberdog_utils::CallbackReturn;
using Mode_T = motion_msgs::msg::Mode;

class AutomationManager : public manager::CascadeManager
{
public:
  explicit AutomationManager(const std::string node_list);
  ~AutomationManager();

  void reset_nodes(const Mode_T & mode);

protected:
  CallbackReturn_T on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_shutdown(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_error(const rclcpp_lifecycle::State &) override;

private:
  const std::string explor_map_node_list_name_ = "explor_map_nodes_l";
  const std::string explor_nav_node_list_name_ = "explor_nav_nodes_l";
  const std::string track_node_list_name_ = "tracking_nodes_l";
};
}  // namespace manager
}  // namespace cyberdog

#endif  // MANAGERS__AUTOMATION_MANAGER_HPP_
