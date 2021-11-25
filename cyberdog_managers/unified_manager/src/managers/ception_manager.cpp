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

#include <experimental/filesystem>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "managers/ception_manager.hpp"

namespace cyberdog
{
namespace manager
{

CeptionManager::CeptionManager()
: manager::CascadeManager("ception_manager", "ception_nodes_l")
{
  this->declare_parameter("rate_tik_hz", 2);
  this->declare_parameter("soc_limit_perc", 10);

  // Get package share dir
  #ifdef PACKAGE_NAME
  auto local_share_dir = ament_index_cpp::get_package_share_directory(PACKAGE_NAME);
  local_params_dir = local_share_dir + std::string("/params/");
  #endif
  param_file = local_params_dir + std::string("params.toml");

  ext_file_exist = std::experimental::filesystem::exists(param_file);
  if (!ext_file_exist) {
    message_warn(std::string("[ExtFile] File [") + param_file + std::string("] is not exist."));
  }
  // else {
  //   auto data = toml::parse(param_file);
  //   const auto& audio_file_id = toml::find(data, "AudioFileID");
  //   const auto low_btr_id = toml::find<int32_t>(audio_file_id, "low_battery");
  // }
}

CeptionManager::~CeptionManager()
{
  if (get_current_state().id() ==
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
  {
    on_deactivate(get_current_state());
    on_cleanup(get_current_state());
  }
  message_info(this->get_name() + std::string(" lifecycle destroyed."));
}

CallbackReturn_T CeptionManager::on_configure(const rclcpp_lifecycle::State &)
{
  message_info(this->get_name() + std::string(" onconfiguring"));

  rate_tik_ = this->get_parameter("rate_tik_hz").as_int();
  soc_limit_ = this->get_parameter("soc_limit_perc").as_int();

  if (manager_configure()) {
    message_info(std::string("Manager internal configured."));
  }

  bms_sub_ = this->create_subscription<BMS_T>(
    "bms_recv", rclcpp::SystemDefaultsQoS(),
    std::bind(&CeptionManager::bms_callback, this, std::placeholders::_1));

  guard_pub_ = this->create_publisher<Safety_T>(
    "safe_guard", rclcpp::SystemDefaultsQoS());

  audio_play_client_ = rclcpp_action::create_client<AudioPlay_T>(this, "audio_play");

  message_info(get_name() + std::string(" configured."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T CeptionManager::on_activate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" onactivating"));

  if (manager_activate()) {
    message_info(std::string("Manager internal activated."));
  }

  thread_flag_ = true;
  guard_info_.status = Safety_T::NORMAL;

  guard_pub_->on_activate();
  guard_daemon_thread_ = std::make_unique<std::thread>(&CeptionManager::safe_guard_daemon, this);

  message_info(get_name() + std::string(" activated."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T CeptionManager::on_deactivate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" deactivating..."));

  thread_flag_ = false;
  guard_pub_->on_deactivate();

  if (manager_deactivate()) {
    message_info(std::string("Manager internal deactivated."));
  }

  guard_daemon_thread_->join();

  message_info(get_name() + std::string(" deactivated."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T CeptionManager::on_cleanup(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" cleaning up..."));

  if (manager_cleanup()) {
    message_info(std::string("Manager internal cleaned up."));
  }

  audio_play_client_.reset();
  bms_sub_.reset();
  guard_pub_.reset();
  guard_daemon_thread_.reset();

  message_info(get_name() + std::string(" completely cleaned up."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T CeptionManager::on_shutdown(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" shutting down."));

  if (manager_shutdown()) {
    message_info(std::string("Manager internal shut down."));
  }

  message_info(get_name() + std::string(" shut down."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T CeptionManager::on_error(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" error raising..."));

  thread_flag_ = false;
  guard_pub_->on_deactivate();
  guard_daemon_thread_->join();

  if (manager_error()) {
    message_info(std::string("Manager internal error processed."));
  }

  audio_play_client_.reset();
  bms_sub_.reset();
  guard_pub_.reset();
  guard_daemon_thread_.reset();


  message_info(get_name() + std::string(" error processed."));
  return CallbackReturn_T::SUCCESS;
}

void CeptionManager::bms_callback(const BMS_T::SharedPtr msg)
{
  auto guard_temp = std::make_unique<Safety_T>();
  if (msg->batt_soc <= soc_limit_) {
    guard_temp->status = Safety_T::LOW_BTR;
  } else if (msg->batt_soc > soc_limit_ && msg->batt_soc <= 100) {
    guard_temp->status = Safety_T::NORMAL;
  }

  // TBD: bit check before assignment
  guard_info_ = *guard_temp;
}

void CeptionManager::safe_guard_daemon()
{
  rclcpp::WallRate rate_hold_(rate_tik_);

  while (rclcpp::ok() && thread_flag_) {
    guard_pub_->publish(guard_info_);

    rate_hold_.sleep();
  }
}
}  // namespace manager
}  // namespace cyberdog
