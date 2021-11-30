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

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "managers/automation_manager.hpp"

namespace cyberdog
{
namespace manager
{

AutomationManager::AutomationManager(const std::string node_list)
: manager::CascadeManager("automation_manager", node_list)
{
  const std::vector<std::string> cascade_nodes_names = {};
  this->declare_parameter(explor_map_node_list_name_, cascade_nodes_names);
  this->declare_parameter(explor_nav_node_list_name_, cascade_nodes_names);
  this->declare_parameter(track_node_list_name_, cascade_nodes_names);
}

AutomationManager::~AutomationManager()
{
  if (get_current_state().id() ==
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
  {
    on_deactivate(get_current_state());
    on_cleanup(get_current_state());
  }
  message_info(this->get_name() + std::string(" lifecycle destroyed."));
}

void AutomationManager::reset_nodes(const Mode_T & mode)
{
  switch (mode.control_mode) {
    case Mode_T::MODE_EXPLOR:
      {
        std::string explor_node_list_name_;
        if (mode.mode_type == Mode_T::EXPLR_NAV_AB) {
          explor_node_list_name_ = explor_nav_node_list_name_;
        } else {
          explor_node_list_name_ = explor_map_node_list_name_;
        }
        manager_configure(explor_node_list_name_);
        message_info(std::string("Reset explor nodes"));
        break;
      }
    case Mode_T::MODE_TRACK:
      {
        manager_configure(track_node_list_name_);
        message_info(std::string("Reset track nodes"));
        break;
      }
    default:
      {
        message_error(std::string("Unknown control_mode"));
        break;
      }
  }
}

CallbackReturn_T AutomationManager::on_configure(const rclcpp_lifecycle::State &)
{
  message_info(this->get_name() + std::string(" onconfiguring"));

  message_info(get_name() + std::string(" configured."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T AutomationManager::on_activate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" onactivating"));

  if (manager_activate()) {
    message_info(std::string("Manager internal activated."));
  }

  message_info(get_name() + std::string(" activated."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T AutomationManager::on_deactivate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" deactivating..."));

  if (manager_deactivate()) {
    message_info(std::string("Manager internal deactivated."));
  }

  message_info(get_name() + std::string(" deactivated."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T AutomationManager::on_cleanup(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" cleaning up..."));

  if (manager_cleanup()) {
    message_info(std::string("Manager internal cleaned up."));
  }

  message_info(get_name() + std::string(" completely cleaned up."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T AutomationManager::on_shutdown(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" shutting down."));

  if (manager_shutdown()) {
    message_info(std::string("Manager internal shut down."));
  }

  message_info(get_name() + std::string(" shut down."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T AutomationManager::on_error(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" error raising..."));

  if (manager_error()) {
    message_info(std::string("Manager internal error processed."));
  }

  message_info(get_name() + std::string(" error processed."));
  return CallbackReturn_T::SUCCESS;
}
}  // namespace manager
}  // namespace cyberdog
