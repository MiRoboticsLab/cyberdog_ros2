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

// C++17 headers not support uncrustify yet
#include "string_view"

#include "manager_utils/cascade_manager.hpp"

namespace cyberdog
{
namespace manager
{

CascadeManager::CascadeManager(const std::string node_name, const std::string node_list_name)
: cyberdog_utils::LifecycleNode(node_name),
  node_list_name_(node_list_name)
{
  message_info(std::string("Creating ") + this->get_name());

  // Declare this node's parameters
  if (node_list_name_ == std::string("")) {
    manager_type = SINGLE_MANAGER;
  } else if (node_list_name_ == std::string("multi")) {
    manager_type = MULTI_LIST;
    this->declare_parameter("timeout_manager_s", 10);
  } else {
    manager_type = SINGLE_LIST;
    const std::vector<std::string> cascade_nodes_names = {};
    this->declare_parameter(node_list_name_, cascade_nodes_names);
    this->declare_parameter("timeout_manager_s", 10);
  }

  message_info(this->get_name() + std::string(" created."));
}

CascadeManager::~CascadeManager()
{}

bool CascadeManager::manager_configure(const std::string node_list_name)
{
  bool rtn_(false);
  if (manager_type != SINGLE_MANAGER) {
    if (chainnodes_state_ != STATE_NULL) {
      thread_flag_ = false;
      sub_node_checking->join();
    }
    chainnodes_state_ = STATE_NULL;
    thread_flag_ = false;
    node_map_.clear();
    this->clear_activation();
    auto node_name_list = manager_type == SINGLE_LIST ?
      this->get_parameter(node_list_name_).as_string_array() :
      this->get_parameter(node_list_name).as_string_array();

    if (node_name_list.size() != 0) {
      for (auto & node_name : node_name_list) {
        if (node_map_.find(node_name) == node_map_.end()) {
          node_map_.insert(
            std::pair<std::string, uint8_t>(
              node_name,
              lifecycle_msgs::msg::State::PRIMARY_STATE_UNKNOWN));
          message_info(
            std::string("Add ") +
            node_name +
            std::string(" to node chain list of ") +
            this->get_name());
        }
      }
      node_name_set_.clear();
      node_name_set_.insert(begin(node_name_list), end(node_name_list));
    }

    timeout_manager_ = this->get_parameter("timeout_manager_s").as_int();

    node_states_ = this->create_subscription<cascade_lifecycle_msgs::msg::State>(
      "cascade_lifecycle_states",
      rclcpp::QoS(100),
      std::bind(&CascadeManager::node_state_callback, this, std::placeholders::_1));
    rtn_ = true;
  }
  return rtn_;
}

bool CascadeManager::manager_activate()
{
  bool rtn_(false);
  if (manager_type != SINGLE_MANAGER) {
    for (auto & node : node_map_) {
      this->add_activation(static_cast<std::string>(node.first));
    }
    if (chainnodes_state_ != STATE_NULL) {
      thread_flag_ = false;
      sub_node_checking->join();
    }
    thread_flag_ = true;
    sub_node_checking = std::make_unique<std::thread>(
      &CascadeManager::node_status_checking, this, IS_ACTIVE);
    rtn_ = true;
  }
  return rtn_;
}

bool CascadeManager::manager_deactivate()
{
  bool rtn_(false);
  if (manager_type != SINGLE_MANAGER) {
    thread_flag_ = false;
    sub_node_checking->join();
    thread_flag_ = true;
    sub_node_checking = std::make_unique<std::thread>(
      &CascadeManager::node_status_checking, this, IS_DEACTIVE);
    rtn_ = true;
  }
  return rtn_;
}

bool CascadeManager::manager_cleanup()
{
  bool rtn_(false);
  thread_flag_ = false;
  if (manager_type != SINGLE_MANAGER) {
    sub_node_checking->join();
    node_map_.clear();
    node_states_.reset();
    sub_node_checking.reset();
    rtn_ = true;
  }
  return rtn_;
}

bool CascadeManager::manager_shutdown()
{
  bool rtn_(false);
  thread_flag_ = false;
  if (manager_type != SINGLE_MANAGER) {
    sub_node_checking->join();
    node_map_.clear();
    node_states_.reset();
    sub_node_checking.reset();
    rtn_ = true;
  }
  return rtn_;
}

bool CascadeManager::manager_error()
{
  bool rtn_(false);
  thread_flag_ = false;
  if (manager_type != SINGLE_MANAGER) {
    sub_node_checking->join();
    node_map_.clear();
    node_states_.reset();
    sub_node_checking.reset();
    rtn_ = true;
  }
  return rtn_;
}

void CascadeManager::node_state_callback(const cascade_lifecycle_msgs::msg::State::SharedPtr msg)
{
  if (node_name_set_.find(msg->node_name) != node_name_set_.end()) {
    node_map_[msg->node_name] = msg->state;
  }
}

void CascadeManager::node_status_checking(const State_Req req_type)
{
  auto activations = this->get_activations();
  rclcpp::WallRate activating_rate(4);
  rclcpp::Time start_time(this->get_clock()->now());
  std::set<std::string> active_set;
  auto timeout_count = (this->get_clock()->now() - start_time).seconds();

  while (rclcpp::ok() && thread_flag_ &&
    (this->get_clock()->now() - start_time <= std::chrono::seconds(timeout_manager_)))
  {
    timeout_count = (this->get_clock()->now() - start_time).seconds();

    if (active_set.size() == node_map_.size()) {
      break;
    }
    if (node_map_.size() == 0) {
      break;
    }

    for (auto & activation : activations) {
      if (node_map_[activation] == req_type &&
        active_set.find(activation) == active_set.end())
      {
        message_info(
          std::string("Node [") +
          activation +
          std::string("] is ") +
          state_map_[req_type]);
        active_set.insert(activation);
      }
    }
    activating_rate.sleep();
  }

  uint8_t unchanged_count_(0);
  if (activations.size() != 0) {
    for (auto & activation : activations) {
      if (node_map_[activation] != req_type) {
        unchanged_count_ += 1;
        message_error(
          std::string("After ") +
          std::to_string(timeout_count) +
          std::string(" seconds. Node ") +
          activation +
          std::string(" is still not ") +
          state_map_[req_type]);
      }
    }
  }
  if (unchanged_count_ == 0) {
    chainnodes_state_ = (req_type == IS_ACTIVE) ? ALL_ACTIVE : ALL_DEACTIVE;
    message_info(
      std::string("All node/nodes is/are ") +
      state_map_[req_type]);
  } else {
    chainnodes_state_ = (req_type == IS_ACTIVE) ? PART_ACTIVE : PART_DEACTIVE;
  }
}

void CascadeManager::message_info(std::string_view log)
{
  RCLCPP_INFO_STREAM(this->get_logger(), log);
}

void CascadeManager::message_warn(std::string_view log)
{
  RCLCPP_INFO_STREAM(this->get_logger(), log);
}

void CascadeManager::message_error(std::string_view log)
{
  RCLCPP_INFO_STREAM(this->get_logger(), log);
}

}  // namespace manager
}  // namespace cyberdog
