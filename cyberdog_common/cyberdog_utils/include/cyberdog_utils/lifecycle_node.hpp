// Copyright 2020 Intelligent Robotics Lab
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

#ifndef CYBERDOG_UTILS__LIFECYCLE_NODE_HPP_
#define CYBERDOG_UTILS__LIFECYCLE_NODE_HPP_

#include <set>
#include <map>
#include <string>

#include "rclcpp/macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/node_options.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/visibility_control.h"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"

#include "lifecycle_msgs/msg/state.hpp"
#include "cascade_lifecycle_msgs/msg/activation.hpp"
#include "cascade_lifecycle_msgs/msg/state.hpp"
#include "cyberdog_utils/Enums.hpp"

#define ANSI_COLOR_RESET    "\x1b[0m"
#define ANSI_COLOR_BLUE     "\x1b[34m"

namespace cyberdog_utils
{
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using lifecycle_msgs::msg::State;

class LifecycleNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  /// Create a new lifecycle node with the specified name.
  /**
   * \param[in] node_name Name of the node.
   * \param[in] namespace_ Namespace of the node.
   * \param[in] options Additional options to control creation of the node.
   */
  RCLCPP_LIFECYCLE_PUBLIC
  explicit LifecycleNode(
    const std::string & node_name,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /// Create a node based on the node name and a rclcpp::Context.
  /**
   * \param[in] node_name Name of the node.
   * \param[in] namespace_ Namespace of the node.
   * \param[in] options Additional options to control creation of the node.
   */
  RCLCPP_LIFECYCLE_PUBLIC
  LifecycleNode(
    const std::string & node_name,
    const std::string & namespace_,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  void add_activation(const std::string & node_name);
  void remove_activation(const std::string & node_name);
  void remove_activation_pub(const std::string & node_name);
  void clear_activation();

  const std::set<std::string> & get_activators() const {return activators_;}
  const std::set<std::string> & get_activations() const {return activations_;}
  const std::map<std::string, uint8_t> & get_activators_state() const
  {
    return activators_state_;
  }
  /// Auto check when node required.
  /**
   * \param[in] check_type : CHECK_TO_START to active,
   *                         CHECK_TO_PAUSE to pause,
   *                         CHECK_TO_SHUTDOWN to shutdown
   * \param[out] bool : result of checking.
   */
  bool auto_check(uint8_t check_type);

private:
  CallbackReturn
  on_configure_internal(const rclcpp_lifecycle::State & previous_state);

  CallbackReturn
  on_cleanup_internal(const rclcpp_lifecycle::State & previous_state);

  CallbackReturn
  on_shutdown_internal(const rclcpp_lifecycle::State & previous_state);

  CallbackReturn
  on_activate_internal(const rclcpp_lifecycle::State & previous_state);

  CallbackReturn
  on_deactivate_internal(const rclcpp_lifecycle::State & previous_state);

  CallbackReturn
  on_error_internal(const rclcpp_lifecycle::State & previous_state);

  rclcpp_lifecycle::LifecyclePublisher<cascade_lifecycle_msgs::msg::State>::SharedPtr states_pub_;
  rclcpp_lifecycle::LifecyclePublisher<cascade_lifecycle_msgs::msg::Activation>::SharedPtr
    activations_pub_;

  rclcpp::Subscription<cascade_lifecycle_msgs::msg::Activation>::SharedPtr activations_sub_;
  rclcpp::Subscription<cascade_lifecycle_msgs::msg::State>::SharedPtr states_sub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::set<std::string> activators_;
  std::set<std::string> activations_;
  std::map<std::string, uint8_t> activators_state_;
  bool governed;

  void activations_callback(const cascade_lifecycle_msgs::msg::Activation::SharedPtr msg);
  void states_callback(const cascade_lifecycle_msgs::msg::State::SharedPtr msg);
  void update_state(const uint8_t state = lifecycle_msgs::msg::Transition::TRANSITION_CREATE);
  void timer_callback();
  void message(const std::string & msg)
  {
    RCLCPP_INFO(get_logger(), ANSI_COLOR_BLUE "\33[1m%s\33[0m" ANSI_COLOR_RESET, msg.c_str());
  }
};

}  // namespace cyberdog_utils

#endif  // CYBERDOG_UTILS__LIFECYCLE_NODE_HPP_
