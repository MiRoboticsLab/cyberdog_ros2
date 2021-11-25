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

#ifndef MANAGERS__CEPTION_MANAGER_HPP_
#define MANAGERS__CEPTION_MANAGER_HPP_

// C++ headers
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// ception msgs
#include "ception_msgs/msg/bms.hpp"
// interaction msgs
#include "interaction_msgs/action/audio_play.hpp"
// motion msgs
#include "motion_msgs/msg/safety.hpp"
// ROS headers
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "manager_utils/cascade_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
// Other headers
#include "toml11/toml.hpp"

namespace cyberdog
{
namespace manager
{

using AudioPlay_T = interaction_msgs::action::AudioPlay;
using BMS_T = ception_msgs::msg::Bms;
using CallbackReturn_T = cyberdog_utils::CallbackReturn;
using Safety_T = motion_msgs::msg::Safety;

class CeptionManager : public manager::CascadeManager
{
public:
  CeptionManager();
  ~CeptionManager();

protected:
  CallbackReturn_T on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_shutdown(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_error(const rclcpp_lifecycle::State &) override;

private:
// Callbacks functions
  void bms_callback(const BMS_T::SharedPtr msg);

// Publisher
  void publish_guard(const Safety_T & msg);

// Internal functions
  void safe_guard_daemon();

/// Variables
// parameters<int> rate
  inline static int rate_tik_;

// parameters<int> value
  inline static int soc_limit_;

// Action client
  rclcpp_action::Client<AudioPlay_T>::SharedPtr audio_play_client_;

// Internal variables
  bool thread_flag_;
  Safety_T guard_info_;

// Threads ptr
  std::unique_ptr<std::thread> guard_daemon_thread_;

// Subscriber
  rclcpp::Subscription<BMS_T>::SharedPtr bms_sub_;

// Publisher
  rclcpp_lifecycle::LifecyclePublisher<Safety_T>::SharedPtr guard_pub_;

// Package directories
  std::string local_params_dir;

// global variables
  std::string param_file;
  bool ext_file_exist;
};
}  // namespace manager
}  // namespace cyberdog

#endif  // MANAGERS__CEPTION_MANAGER_HPP_
