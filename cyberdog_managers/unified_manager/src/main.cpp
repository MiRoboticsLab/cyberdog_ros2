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

#include <exception>
#include <memory>
#include <string>

#include "managers/ception_manager.hpp"
#include "managers/interaction_manager.hpp"
#include "managers/motion_manager.hpp"

int main(int argc, char ** argv)
{
  try {
    rclcpp::init(argc, argv);
  } catch (rclcpp::ParameterTypeException &) {
    std::cerr << std::string("Parameter type exception catched.") << std::endl;
  }
  auto node_motion = std::make_shared<cyberdog::manager::MotionManager>();
  auto node_cept = std::make_shared<cyberdog::manager::CeptionManager>();
  auto node_inter = std::make_shared<cyberdog::manager::InteractionManager>();
  bool started = false;

  while (!started) {
    started = node_motion->auto_check(cyberdog_utils::CHECK_TO_START);
    started &= node_cept->auto_check(cyberdog_utils::CHECK_TO_START);
    started &= node_inter->auto_check(cyberdog_utils::CHECK_TO_START);
  }
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(node_motion->get_node_base_interface());
  exec_.add_node(node_cept->get_node_base_interface());
  exec_.add_node(node_inter->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();

  return 0;
}
