// Copyright 2019 Intelligent Robotics Lab
// Copyright 2021 Homalozoa, Beijing Xiaomi Mobile Software Co., Ltd.
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
#include <vector>
#include <regex>
#include <iostream>
#include <memory>


#include "cyberdog_utils/lifecycle_node.hpp"

#include "rclcpp/rclcpp.hpp"

#include "gtest/gtest.h"

class TestNode : public cyberdog_utils::LifecycleNode
{
public:
  explicit TestNode(
    const std::string & name,
    const std::string & ns = "")
  : LifecycleNode(name, ns),
    my_state_(lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED)
  {
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &)
  {
    my_state_ = lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE;

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &)
  {
    my_state_ = lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &)
  {
    my_state_ = lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE;

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  uint8_t get_my_state() {return my_state_;}

private:
  uint8_t my_state_;
};


TEST(cyberdog_utils, activations_managing_basic)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>("node_C");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());
  executor.add_node(node_c->get_node_base_interface());

  node_a->add_activation("node_B");
  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 2u);
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_EQ(node_b->get_activators().size(), 1u);
  ASSERT_TRUE(node_c->get_activations().empty());
  ASSERT_EQ(node_c->get_activators().size(), 1u);
}

TEST(cyberdog_utils, activations_managing_late_joining)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");
  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 2u);
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_EQ(node_b->get_activators().size(), 1u);

  node_b = nullptr;

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 3.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  auto node_b2 = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>("node_C");
  executor.add_node(node_c->get_node_base_interface());
  executor.add_node(node_b2->get_node_base_interface());

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 2u);
  ASSERT_TRUE(node_b2->get_activations().empty());
  ASSERT_EQ(node_b2->get_activators().size(), 1u);
  ASSERT_TRUE(node_c->get_activations().empty());
  ASSERT_EQ(node_c->get_activators().size(), 1u);
}

TEST(cyberdog_utils, activations_chained)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_EQ(node_b->get_activators().size(), 1u);

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

TEST(cyberdog_utils, multiple_activations_chained)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>("node_C");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());
  executor.add_node(node_c->get_node_base_interface());

  node_a->add_activation("node_C");
  node_b->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_TRUE(node_b->get_activators().empty());
  ASSERT_EQ(node_b->get_activations().size(), 1u);
  ASSERT_TRUE(node_c->get_activations().empty());
  ASSERT_EQ(node_c->get_activators().size(), 2u);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->remove_activation("node_C");
  node_b->remove_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
}

TEST(cyberdog_utils, fast_change)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_FALSE(node_b->get_activators().empty());
  ASSERT_EQ(node_b->get_activations().size(), 0u);
  ASSERT_FALSE(node_b->get_activators_state().empty());

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
}

TEST(cyberdog_utils, activators_disappearance)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_FALSE(node_b->get_activators().empty());
  ASSERT_EQ(node_b->get_activations().size(), 0u);
  ASSERT_FALSE(node_b->get_activators_state().empty());

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a = nullptr;

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 3.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_TRUE(node_b->get_activators().empty());
  ASSERT_TRUE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activations().size(), 0u);
}

TEST(cyberdog_utils, activators_disappearance_inter)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>("node_A");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>("node_B");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>("node_C");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());
  executor.add_node(node_c->get_node_base_interface());

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }


  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(
    node_c->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->remove_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
    node_c->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->remove_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->remove_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->remove_activation("node_B");
  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);


  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_TRUE(node_a->get_activations().empty());
  ASSERT_TRUE(node_b->get_activators().empty());
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_TRUE(node_c->get_activators().empty());
  ASSERT_TRUE(node_c->get_activations().empty());


  node_a->add_activation("node_C");
  node_b->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);


  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

TEST(cyberdog_utils, inheritance)
{
  auto node_1 = std::make_shared<TestNode>("node_1");
  auto node_2 = std::make_shared<TestNode>("node_2");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_1->get_node_base_interface());
  executor.add_node(node_2->get_node_base_interface());

  node_1->add_activation("node_2");
  node_1->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_1->now();
    while ((node_1->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_1->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_1->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_1->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_1->now();
    while ((node_1->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_1->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_2->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_1->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_2->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_1->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_1->now();
    while ((node_1->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_1->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_1->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

TEST(cyberdog_utils, activations_managing_basic_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_C",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());
  executor.add_node(node_c->get_node_base_interface());

  node_a->add_activation("node_B");
  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 2u);
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_EQ(node_b->get_activators().size(), 1u);
  ASSERT_TRUE(node_c->get_activations().empty());
  ASSERT_EQ(node_c->get_activators().size(), 1u);
}

TEST(cyberdog_utils, activations_managing_late_joining_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");
  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 2u);
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_EQ(node_b->get_activators().size(), 1u);

  node_b = nullptr;

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 3.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  auto node_b2 = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_C",
    "test_ns");
  executor.add_node(node_c->get_node_base_interface());
  executor.add_node(node_b2->get_node_base_interface());

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 2u);
  ASSERT_TRUE(node_b2->get_activations().empty());
  ASSERT_EQ(node_b2->get_activators().size(), 1u);
  ASSERT_TRUE(node_c->get_activations().empty());
  ASSERT_EQ(node_c->get_activators().size(), 1u);
}

TEST(cyberdog_utils, activations_chained_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_EQ(node_b->get_activators().size(), 1u);

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators_state().empty());
  ASSERT_FALSE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activators_state().size(), 1u);
  ASSERT_EQ(node_b->get_activators_state().begin()->first, "node_A");
  ASSERT_EQ(
    node_b->get_activators_state().begin()->second,
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

TEST(cyberdog_utils, multiple_activations_chained_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_C",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());
  executor.add_node(node_c->get_node_base_interface());

  node_a->add_activation("node_C");
  node_b->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_TRUE(node_b->get_activators().empty());
  ASSERT_EQ(node_b->get_activations().size(), 1u);
  ASSERT_TRUE(node_c->get_activations().empty());
  ASSERT_EQ(node_c->get_activators().size(), 2u);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->remove_activation("node_C");
  node_b->remove_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
}

TEST(cyberdog_utils, fast_change_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_FALSE(node_b->get_activators().empty());
  ASSERT_EQ(node_b->get_activations().size(), 0u);
  ASSERT_FALSE(node_b->get_activators_state().empty());

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
}

TEST(cyberdog_utils, activators_disappearance_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_EQ(node_a->get_activations().size(), 1u);
  ASSERT_FALSE(node_b->get_activators().empty());
  ASSERT_EQ(node_b->get_activations().size(), 0u);
  ASSERT_FALSE(node_b->get_activators_state().empty());

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a = nullptr;

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 3.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_TRUE(node_b->get_activators().empty());
  ASSERT_TRUE(node_b->get_activators_state().empty());
  ASSERT_EQ(node_b->get_activations().size(), 0u);
}

TEST(cyberdog_utils, activators_disappearance_inter_with_namespace)
{
  auto node_a = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_A",
    "test_ns");
  auto node_b = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_B",
    "test_ns");
  auto node_c = std::make_shared<cyberdog_utils::LifecycleNode>(
    "node_C",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_a->get_node_base_interface());
  executor.add_node(node_b->get_node_base_interface());
  executor.add_node(node_c->get_node_base_interface());

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_a->now();
    while ((node_a->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }


  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(
    node_b->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(
    node_c->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->remove_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(
    node_c->get_current_state().id(),
    lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->remove_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->add_activation("node_B");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->remove_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_c->remove_activation("node_B");
  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);


  ASSERT_TRUE(node_a->get_activators().empty());
  ASSERT_TRUE(node_a->get_activations().empty());
  ASSERT_TRUE(node_b->get_activators().empty());
  ASSERT_TRUE(node_b->get_activations().empty());
  ASSERT_TRUE(node_c->get_activators().empty());
  ASSERT_TRUE(node_c->get_activations().empty());


  node_a->add_activation("node_C");
  node_b->add_activation("node_C");

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);


  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);
  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_a->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_c->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_b->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_b->now();
    while ((node_b->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_a->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_b->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_c->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

TEST(cyberdog_utils, inheritance_with_namespace)
{
  auto node_1 = std::make_shared<TestNode>(
    "node_1",
    "test_ns");
  auto node_2 = std::make_shared<TestNode>(
    "node_2",
    "test_ns");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node_1->get_node_base_interface());
  executor.add_node(node_2->get_node_base_interface());

  node_1->add_activation("node_2");
  node_1->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);

  {
    rclcpp::Rate rate(10);
    auto start = node_1->now();
    while ((node_1->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_1->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_1->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  node_1->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_1->now();
    while ((node_1->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_1->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_2->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_1->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  ASSERT_EQ(node_2->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  node_1->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);

  {
    rclcpp::Rate rate(10);
    auto start = node_1->now();
    while ((node_1->now() - start).seconds() < 1.0) {
      executor.spin_some();
      rate.sleep();
    }
  }

  ASSERT_EQ(node_1->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_current_state().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_1->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node_2->get_my_state(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
