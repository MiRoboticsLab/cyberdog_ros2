// Copyright 2020 Intelligent Robotics Lab
// Copyright 2021 Homalozoa, Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
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

#include <string>
#include <set>

#include "cyberdog_utils/lifecycle_node.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"

#include "rclcpp/rclcpp.hpp"

namespace cyberdog_utils  // Modified from rclcpp_cascade_lifecycle
{

using namespace std::chrono_literals;

LifecycleNode::LifecycleNode(
  const std::string & node_name,
  const rclcpp::NodeOptions & options)
: LifecycleNode(
    node_name,
    "",
    options)
{}

LifecycleNode::LifecycleNode(
  const std::string & node_name,
  const std::string & namespace_,
  const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode(
    node_name,
    namespace_,
    options),
  governed(false)
{
  using std::placeholders::_1;
  using namespace std::chrono_literals;

  activations_pub_ = create_publisher<cascade_lifecycle_msgs::msg::Activation>(
    "cascade_lifecycle_activations",
    rclcpp::QoS(1000).keep_all().transient_local().reliable());

  states_pub_ = create_publisher<cascade_lifecycle_msgs::msg::State>(
    "cascade_lifecycle_states", rclcpp::QoS(100));

  activations_sub_ = create_subscription<cascade_lifecycle_msgs::msg::Activation>(
    "cascade_lifecycle_activations",
    rclcpp::QoS(1000).keep_all().transient_local().reliable(),
    std::bind(&LifecycleNode::activations_callback, this, _1));

  states_sub_ = create_subscription<cascade_lifecycle_msgs::msg::State>(
    "cascade_lifecycle_states",
    rclcpp::QoS(100),
    std::bind(&LifecycleNode::states_callback, this, _1));

  timer_ = create_wall_timer(
    500ms,
    std::bind(&LifecycleNode::timer_callback, this));

  activations_pub_->on_activate();
  states_pub_->on_activate();

  register_on_configure(
    std::bind(
      &LifecycleNode::on_configure_internal,
      this, std::placeholders::_1));

  register_on_cleanup(
    std::bind(
      &LifecycleNode::on_cleanup_internal,
      this, std::placeholders::_1));

  register_on_shutdown(
    std::bind(
      &LifecycleNode::on_shutdown_internal,
      this, std::placeholders::_1));

  register_on_activate(
    std::bind(
      &LifecycleNode::on_activate_internal,
      this, std::placeholders::_1));

  register_on_deactivate(
    std::bind(
      &LifecycleNode::on_deactivate_internal,
      this, std::placeholders::_1));

  register_on_error(
    std::bind(
      &LifecycleNode::on_error_internal,
      this, std::placeholders::_1));
}

void
LifecycleNode::activations_callback(
  const cascade_lifecycle_msgs::msg::Activation::SharedPtr msg)
{
  switch (msg->operation_type) {
    case cascade_lifecycle_msgs::msg::Activation::ADD:
      if (msg->activator == dictator_ && msg->activation == get_name()) {
        governed = true;
        message(std::string("Node ") + get_name() + std::string(" is governed by dictator."));
      } else if (msg->activation == get_name()) {
        activators_.insert(msg->activator);
        if (activators_state_.find(msg->activator) == activators_state_.end()) {
          activators_state_[msg->activator] = lifecycle_msgs::msg::State::PRIMARY_STATE_UNKNOWN;
        }
      }
      break;
    case cascade_lifecycle_msgs::msg::Activation::REMOVE:
      if (msg->activator == dictator_ && msg->activation == get_name()) {
        governed = false;
        message(std::string("Node ") + get_name() + std::string(" is liberated by dictator."));
      } else if (msg->activation == get_name() &&  // NOLINT
        activators_.find(msg->activator) != activators_.end())
      {
        uint8_t remover_state = activators_state_[msg->activator];

        activators_.erase(msg->activator);

        if (activators_state_.find(msg->activator) != activators_state_.end()) {
          activators_state_.erase(msg->activator);
        }

        if (remover_state == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
          bool any_other_activator = false;
          for (const auto & activator : activators_state_) {
            any_other_activator = any_other_activator ||
              activator.second == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
          }

          if (!any_other_activator) {
            trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
          }
        }
      }
      break;
  }
}

void
LifecycleNode::states_callback(const cascade_lifecycle_msgs::msg::State::SharedPtr msg)
{
  if (activators_state_.find(msg->node_name) != activators_state_.end() && !governed) {
    if (activators_state_[msg->node_name] != msg->state) {
      activators_state_[msg->node_name] = msg->state;
      update_state();
    }
  }

  if (msg->node_name == dictator_ && governed) {
    update_state(msg->state);
  }
}

void
LifecycleNode::add_activation(const std::string & node_name)
{
  if (node_name != get_name()) {
    cascade_lifecycle_msgs::msg::Activation msg;
    msg.operation_type = cascade_lifecycle_msgs::msg::Activation::ADD;
    msg.activator = get_name();
    msg.activation = node_name;

    activations_.insert(node_name);
    activations_pub_->publish(msg);
  } else {
    RCLCPP_WARN(get_logger(), "Trying to set an auto activation");
  }
}

void
LifecycleNode::remove_activation(const std::string & node_name)
{
  if (node_name != get_name()) {
    cascade_lifecycle_msgs::msg::Activation msg;
    msg.operation_type = cascade_lifecycle_msgs::msg::Activation::REMOVE;
    msg.activator = get_name();
    msg.activation = node_name;

    activations_.erase(node_name);
    activations_pub_->publish(msg);
  } else {
    RCLCPP_WARN(get_logger(), "Trying to remove & erase an auto activation");
  }
}

void
LifecycleNode::remove_activation_pub(const std::string & node_name)
{
  if (node_name != get_name()) {
    cascade_lifecycle_msgs::msg::Activation msg;
    msg.operation_type = cascade_lifecycle_msgs::msg::Activation::REMOVE;
    msg.activator = get_name();
    msg.activation = node_name;

    activations_pub_->publish(msg);
  } else {
    RCLCPP_WARN(get_logger(), "Trying to remove an auto activation");
  }
}

void
LifecycleNode::clear_activation()
{
  for (const auto & activation : activations_) {
    remove_activation_pub(activation);
  }

  activations_.clear();
}

CallbackReturn
LifecycleNode::on_configure_internal(
  const rclcpp_lifecycle::State & previous_state)
{
  cascade_lifecycle_msgs::msg::State msg;

  auto ret = on_configure(previous_state);

  if (ret == CallbackReturn::SUCCESS) {
    cascade_lifecycle_msgs::msg::State msg;
    msg.state = lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE;
    msg.node_name = get_name();

    states_pub_->publish(msg);
  }

  return ret;
}

CallbackReturn
LifecycleNode::on_cleanup_internal(
  const rclcpp_lifecycle::State & previous_state)
{
  cascade_lifecycle_msgs::msg::State msg;

  auto ret = on_cleanup(previous_state);

  if (ret == CallbackReturn::SUCCESS) {
    cascade_lifecycle_msgs::msg::State msg;
    msg.state = lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED;
    msg.node_name = get_name();

    states_pub_->publish(msg);
  }

  return ret;
}

CallbackReturn
LifecycleNode::on_shutdown_internal(
  const rclcpp_lifecycle::State & previous_state)
{
  cascade_lifecycle_msgs::msg::State msg;

  auto ret = on_shutdown(previous_state);

  if (ret == CallbackReturn::SUCCESS) {
    cascade_lifecycle_msgs::msg::State msg;
    msg.state = lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED;
    msg.node_name = get_name();

    states_pub_->publish(msg);
  }

  return ret;
}

CallbackReturn
LifecycleNode::on_activate_internal(
  const rclcpp_lifecycle::State & previous_state)
{
  cascade_lifecycle_msgs::msg::State msg;

  auto ret = on_activate(previous_state);

  if (ret == CallbackReturn::SUCCESS) {
    cascade_lifecycle_msgs::msg::State msg;
    msg.state = lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
    msg.node_name = get_name();

    states_pub_->publish(msg);
  }

  return ret;
}

CallbackReturn
LifecycleNode::on_deactivate_internal(
  const rclcpp_lifecycle::State & previous_state)
{
  cascade_lifecycle_msgs::msg::State msg;

  auto ret = on_deactivate(previous_state);

  if (ret == CallbackReturn::SUCCESS) {
    cascade_lifecycle_msgs::msg::State msg;
    msg.state = lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE;
    msg.node_name = get_name();

    states_pub_->publish(msg);
  }

  return ret;
}

CallbackReturn
LifecycleNode::on_error_internal(
  const rclcpp_lifecycle::State & previous_state)
{
  cascade_lifecycle_msgs::msg::State msg;

  auto ret = on_error(previous_state);

  if (ret == CallbackReturn::SUCCESS) {
    cascade_lifecycle_msgs::msg::State msg;
    msg.state = lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED;
    msg.node_name = get_name();

    states_pub_->publish(msg);
  }

  return ret;
}

void
LifecycleNode::update_state(const uint8_t state)
{
  bool parent_inactive = false;
  bool parent_active = false;
  auto last_state_id = get_current_state().id();

  if (state == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    parent_active = true;
    parent_inactive = false;
  } else if (state == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
    parent_active = false;
    parent_inactive = true;
  } else {
    for (const auto & activator : activators_state_) {
      parent_inactive = parent_inactive ||
        activator.second == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE;
      parent_active = parent_active ||
        activator.second == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
    }
  }

  switch (last_state_id) {
    case lifecycle_msgs::msg::State::PRIMARY_STATE_UNKNOWN:
      if (parent_active || parent_inactive) {
        trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
        message(
          std::string(
            transition_label_map_[lifecycle_msgs::msg::Transition::
            TRANSITION_CONFIGURE]) + get_name());
      }
      break;

    case lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED:
      if (parent_active || parent_inactive) {
        trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
        message(
          std::string(
            transition_label_map_[lifecycle_msgs::msg::Transition::
            TRANSITION_CONFIGURE]) + get_name());
      }
      break;

    case lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE:
      if (parent_active) {
        trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
        message(
          std::string(
            transition_label_map_[lifecycle_msgs::msg::Transition::
            TRANSITION_ACTIVATE]) + get_name());
      }
      break;

    case lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE:
      if (!parent_active && parent_inactive) {
        trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
        message(
          std::string(
            transition_label_map_[lifecycle_msgs::msg::Transition::
            TRANSITION_DEACTIVATE]) + get_name());
      }
      break;

    case lifecycle_msgs::msg::State::PRIMARY_STATE_FINALIZED:
      break;
  }
}

void
LifecycleNode::timer_callback()
{
  auto nodes = this->get_node_graph_interface()->get_node_names();
  std::string ns = get_namespace();
  if (ns != std::string("/")) {
    ns = ns + std::string("/");
  }

  std::set<std::string>::iterator it = activators_.begin();
  while (it != activators_.end()) {
    const auto & node_name = *it;
    if (std::find(nodes.begin(), nodes.end(), ns + node_name) == nodes.end()) {
      RCLCPP_DEBUG(
        get_logger(), "Activator %s is not longer present, removing from activators",
        node_name.c_str());
      it = activators_.erase(it);

      if (get_current_state().id() == activators_state_[node_name]) {
        update_state();
      }
      activators_state_.erase(node_name);
    } else {
      it++;
    }
  }

  cascade_lifecycle_msgs::msg::State msg;
  msg.state = get_current_state().id();
  msg.node_name = get_name();

  states_pub_->publish(msg);

  update_state();
}

bool
LifecycleNode::auto_check(uint8_t check_type)
{
  message(std::string("Auto checking ") + get_name());
  bool rtn_ = false;

  if (check_type == CHECK_TO_START) {
    message(std::string("Auto start..."));
    switch (get_current_state().id()) {
      case lifecycle_msgs::msg::State::PRIMARY_STATE_UNKNOWN:
      case lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED:
        {
          message(
            transition_label_map_[lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE] +
            get_name());
          if (transition_state_map_[lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE] !=
            trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE).id())
          {
            check_type = CHECK_TO_CLEANUP;
            break;
          }
        }
      // fall through
      case lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE:
        {
          message(
            transition_label_map_[lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE] +
            get_name());
          if (transition_state_map_[lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE] !=
            trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE).id())
          {
            check_type = CHECK_TO_CLEANUP;
            break;
          }
        }
      // fall through
      case lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE:
        {
          message(std::string("Bringup ") + get_name() + std::string(" succeed."));
          rtn_ = true;
          break;
        }
      default:
        break;
    }
  }

  if (check_type == CHECK_TO_PAUSE) {
    message(std::string("Auto pause..."));

    if (get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
      message(get_name() + std::string(" already inactive."));
      rtn_ = true;
    } else if (get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED) {
      message(
        transition_label_map_[lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE] +
        get_name());
      if (transition_state_map_[lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE] !=
        trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE).id())
      {
        check_type = CHECK_TO_CLEANUP;
      } else {
        rtn_ = true;
        message(get_name() + std::string(" pause succeed."));
      }
    } else if (get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
      message(
        get_name() +
        transition_label_map_[lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE]);
      if (transition_state_map_[lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE] !=
        trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE).id())
      {
        check_type = CHECK_TO_CLEANUP;
      } else {
        rtn_ = true;
        message(get_name() + std::string(" pause succeed."));
      }
    } else {
      message(std::string("Unknown state to pause."));
    }
  }

  if (check_type == CHECK_TO_CLEANUP) {
    message(std::string("Auto cleanup..."));
    switch (get_current_state().id()) {
      case lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE:
        {
          message(
            get_name() +
            transition_label_map_[lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE]);
          if (transition_state_map_[lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE] !=
            trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE).id())
          {
            break;
          }
        }
      // fall through
      case lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE:
        {
          message(
            get_name() +
            transition_label_map_[lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP]);
          if (transition_state_map_[lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP] !=
            trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CLEANUP).id())
          {
            break;
          }
        }
      // fall through
      case lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED:
        {
          message(get_name() + std::string(" cleanup succeed."));
          rtn_ = true;
          break;
        }
      default:
        break;
    }
  }

  return rtn_;
}

}  // namespace cyberdog_utils
