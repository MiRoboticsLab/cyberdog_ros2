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

#ifndef MANAGERS__INTERACTION_MANAGER_HPP_
#define MANAGERS__INTERACTION_MANAGER_HPP_

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
#include "rclcpp/rclcpp.hpp"

namespace cyberdog
{
namespace manager
{

using CallbackReturn_T = cyberdog_utils::CallbackReturn;

class InteractionManager : public manager::CascadeManager
{
public:
  InteractionManager();
  ~InteractionManager();

protected:
  CallbackReturn_T on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_shutdown(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_error(const rclcpp_lifecycle::State &) override;

private:
};
}  // namespace manager
}  // namespace cyberdog

#endif  // MANAGERS__INTERACTION_MANAGER_HPP_
