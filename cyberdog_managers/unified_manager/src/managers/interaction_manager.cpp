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

#include "managers/interaction_manager.hpp"

namespace cyberdog
{
namespace manager
{

InteractionManager::InteractionManager()
: manager::CascadeManager("interaction_manager", "interaction_nodes_l")
{}

InteractionManager::~InteractionManager()
{
  if (get_current_state().id() ==
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
  {
    on_deactivate(get_current_state());
    on_cleanup(get_current_state());
  }
  message_info(this->get_name() + std::string(" lifecycle destroyed."));
}

CallbackReturn_T InteractionManager::on_configure(const rclcpp_lifecycle::State &)
{
  message_info(this->get_name() + std::string(" onconfiguring"));

  if (manager_configure()) {
    message_info(std::string("Manager internal configured."));
  }

  message_info(get_name() + std::string(" configured."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T InteractionManager::on_activate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" onactivating"));

  if (manager_activate()) {
    message_info(std::string("Manager internal activated."));
  }

  message_info(get_name() + std::string(" activated."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T InteractionManager::on_deactivate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" deactivating..."));

  if (manager_deactivate()) {
    message_info(std::string("Manager internal deactivated."));
  }

  message_info(get_name() + std::string(" deactivated."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T InteractionManager::on_cleanup(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" cleaning up..."));

  if (manager_cleanup()) {
    message_info(std::string("Manager internal cleaned up."));
  }

  message_info(get_name() + std::string(" completely cleaned up."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T InteractionManager::on_shutdown(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" shutting down."));

  if (manager_shutdown()) {
    message_info(std::string("Manager internal shut down."));
  }

  message_info(get_name() + std::string(" shut down."));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T InteractionManager::on_error(const rclcpp_lifecycle::State &)
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
