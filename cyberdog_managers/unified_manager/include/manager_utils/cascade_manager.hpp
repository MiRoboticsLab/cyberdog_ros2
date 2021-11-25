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

#ifndef MANAGER_UTILS__CASCADE_MANAGER_HPP_
#define MANAGER_UTILS__CASCADE_MANAGER_HPP_

// C++ headers
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// C++17 headers not support uncrustify yet
#include "string_view"

#include "cyberdog_utils/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cyberdog
{
namespace manager
{

using CallbackReturn_T = cyberdog_utils::CallbackReturn;

enum Manager_Type
{
  SINGLE_MANAGER = 0,
  SINGLE_LIST = 1,
  MULTI_LIST = 2
};

enum State_Req
{
  IS_ACTIVE = lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE,
  IS_DEACTIVE = lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE
};

enum ChainState
{
  STATE_NULL   = 0,
  ALL_ACTIVE   = 1,
  PART_ACTIVE  = 2,
  ALL_DEACTIVE = 3,
  PART_DEACTIVE = 4
};

static std::unordered_map<uint8_t, std::string> state_map_{
  {lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE, std::string("activated")},
  {lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE, std::string("deactivated")}
};

class CascadeManager : public cyberdog_utils::LifecycleNode
{
public:
  explicit CascadeManager(const std::string node_name, const std::string node_list_name);
  ~CascadeManager();

  uint8_t manager_type;
  uint8_t chainnodes_state_;

protected:
  bool manager_configure(const std::string node_list_name = "");
  bool manager_activate();
  bool manager_deactivate();
  bool manager_cleanup();
  bool manager_shutdown();
  bool manager_error();

// common funcs
  void message_info(std::string_view log);
  void message_warn(std::string_view log);
  void message_error(std::string_view log);

private:
/// Variables
// Parameters
  std::string node_list_name_;
  std::map<std::string, uint8_t> node_map_;
  std::unordered_set<std::string> node_name_set_;
  int timeout_manager_;

/// Threads
  bool thread_flag_;
  std::unique_ptr<std::thread> sub_node_checking;

// Subscriber for node's topic
  rclcpp::Subscription<cascade_lifecycle_msgs::msg::State>::SharedPtr node_states_;

/// Functions
// Subscription callback
  void node_state_callback(const cascade_lifecycle_msgs::msg::State::SharedPtr msg);
// common funcs
  void node_status_checking(const State_Req req_type);
};
}  // namespace manager
}  // namespace cyberdog

#endif  // MANAGER_UTILS__CASCADE_MANAGER_HPP_
