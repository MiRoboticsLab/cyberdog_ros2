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

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <automation_msgs/srv/target.hpp>
#include <automation_msgs/srv/restricted_area.hpp>
#include <motion_msgs/msg/mode.hpp>
#include <motion_msgs/msg/gait.hpp>
#include <motion_msgs/msg/action_request.hpp>
#include <motion_msgs/msg/action_respond.hpp>
#include <automation_msgs/srv/nav_mode.hpp>
#include <motion_msgs/action/change_mode.hpp>
#include <motion_msgs/action/change_gait.hpp>
#include <motion_msgs/action/ext_mon_order.hpp>
#include <motion_msgs/msg/mon_order.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include "std_msgs/msg/int32.hpp"
#include "cyberdog_utils/Enums.hpp"


using namespace std::chrono_literals;
using CheckMode_T = motion_msgs::action::ChangeMode;
using SubMode_T = automation_msgs::srv::NavMode_Request;
using Gait_T = motion_msgs::msg::Gait;
using CheckGait_T = motion_msgs::action::ChangeGait;
using CheckEXTMONORDER_T = motion_msgs::action::ExtMonOrder;
using EXTMONORDER_T = motion_msgs::msg::MonOrder;
class ActionClient : public rclcpp::Node
{
public:
  ActionClient()
  : Node("ActionClient")
  {
    auto callback = [this](motion_msgs::msg::ActionRequest::SharedPtr msg)
      {
        Command = msg->type;
        mode_ = msg->mode;
        gait_ = msg->gait;
        order_ = msg->order;
        request_id_ = msg->request_id;
        timeout = msg->timeout;
        dog_thread = std::make_shared<std::thread>(&ActionClient::dog_command_callback, this);
        dog_thread->detach();
      };
    dog_sub_ = this->create_subscription<motion_msgs::msg::ActionRequest>(
      "cyberdog_action",
      rclcpp::SystemDefaultsQoS(), callback);
    result_pub_ = this->create_publisher<motion_msgs::msg::ActionRespond>(
      "cyberdog_action_result",
      rclcpp::SystemDefaultsQoS());
  }

  rclcpp::Subscription<motion_msgs::msg::ActionRequest>::SharedPtr dog_sub_;
  rclcpp::Publisher<motion_msgs::msg::ActionRespond>::SharedPtr result_pub_;
  std::shared_ptr<std::thread> dog_thread;
  int Command;
  motion_msgs::msg::Mode mode_;
  motion_msgs::msg::Gait gait_;
  motion_msgs::msg::MonOrder order_;
  int request_id_;
  int timeout;

private:
  void dog_command_callback()
  {
    RCLCPP_INFO(get_logger(), "Dog command get %d.", Command);
    if (Command == motion_msgs::msg::ActionRequest::CHECKOUT_MODE) {
      auto mode_goal = CheckMode_T::Goal();
      motion_msgs::msg::Mode mode;
      mode.timestamp = this->get_clock()->now();
      mode.control_mode = mode_.control_mode;
      mode.mode_type = mode_.mode_type;
      mode_goal.modestamped = mode;
      auto mode_client_ = rclcpp_action::create_client<CheckMode_T>(this, "checkout_mode");
      auto mode_goal_handle = mode_client_->async_send_goal(mode_goal);
      auto mode_result = mode_client_->async_get_result(mode_goal_handle.get());
      motion_msgs::msg::ActionRespond respond;
      respond.type = motion_msgs::msg::ActionRequest::CHECKOUT_MODE;
      respond.succeed = false;
      respond.request_id = request_id_;
      mode_result.wait_for(std::chrono::seconds(timeout > 0 ? timeout : 30));
      if (mode_goal_handle.get()->is_result_aware()) {
        respond.succeed = mode_result.get().result->succeed;
        respond.err_code = mode_result.get().result->err_code;
        respond.err_state = mode_result.get().result->err_state;

        if (mode_result.get().result->succeed) {
          RCLCPP_INFO(get_logger(), "Result of mode_checkout: success");
        } else {
          RCLCPP_INFO(get_logger(), "Result of mode_checkout: failed");
        }
      } else {
        RCLCPP_INFO(get_logger(), "Result of mode_checkout: failed unaware");
      }
      result_pub_->publish(respond);
      RCLCPP_INFO(get_logger(), "mode end");
    } else if (Command == motion_msgs::msg::ActionRequest::CHECKOUT_PATTERN) {
      // checkout pattern
      Gait_T goal_pattern;
      goal_pattern.gait = gait_.gait;
      goal_pattern.timestamp = this->get_clock()->now();
      auto pattern_goal = CheckGait_T::Goal();
      pattern_goal.motivation = cyberdog_utils::GAIT_TRIG;
      pattern_goal.gaitstamped = goal_pattern;
      RCLCPP_INFO(get_logger(), "checkout_pattern: %d", goal_pattern.gait);
      auto pattern_client_ = rclcpp_action::create_client<CheckGait_T>(this, "checkout_gait");
      auto pattern_goal_handle = pattern_client_->async_send_goal(pattern_goal);
      auto pattern_result = pattern_client_->async_get_result(pattern_goal_handle.get());
      motion_msgs::msg::ActionRespond respond;
      respond.type = motion_msgs::msg::ActionRequest::CHECKOUT_PATTERN;
      respond.succeed = false;
      respond.request_id = request_id_;
      pattern_result.wait_for(std::chrono::seconds(timeout > 0 ? timeout : 30));
      if (pattern_goal_handle.get()->is_result_aware()) {
        respond.succeed = pattern_result.get().result->succeed;
        respond.err_code = pattern_result.get().result->err_code;
        respond.err_gait = pattern_result.get().result->err_gait;
        if (pattern_result.get().result->succeed) {
          RCLCPP_INFO(get_logger(), "Result of checkout_pattern: success");
        } else {
          RCLCPP_INFO(get_logger(), "Result of checkout_pattern: failed");
        }
      } else {
        RCLCPP_INFO(get_logger(), "Result of checkout_pattern: failed unaware");
      }
      result_pub_->publish(respond);
      RCLCPP_INFO(get_logger(), "pattern end");
    } else if (Command == motion_msgs::msg::ActionRequest::EXTMONORDER) {
      auto order_goal = CheckEXTMONORDER_T::Goal();
      order_goal.orderstamped = order_;
      order_goal.orderstamped.timestamp = this->get_clock()->now();
      RCLCPP_INFO(get_logger(), "EXTMONORDER");
      RCLCPP_INFO(get_logger(), "setExtmonOrder id: %d", order_goal.orderstamped.id);
      auto order_client = rclcpp_action::create_client<CheckEXTMONORDER_T>(this, "exe_monorder");
      auto order_goal_handle = order_client->async_send_goal(order_goal);
      auto order_result = order_client->async_get_result(order_goal_handle.get());
      motion_msgs::msg::ActionRespond respond;
      respond.type = motion_msgs::msg::ActionRequest::EXTMONORDER;
      respond.succeed = false;
      respond.request_id = request_id_;
      order_result.wait_for(std::chrono::seconds(timeout > 0 ? timeout : 30));
      if (order_goal_handle.get()->is_result_aware()) {
        respond.succeed = order_result.get().result->succeed;
        respond.err_code = order_result.get().result->err_code;
        if (order_result.get().result->succeed) {
          RCLCPP_INFO(get_logger(), "Result of EXTMONORDER: success");
        } else {
          RCLCPP_INFO(get_logger(), "Result of EXTMONORDER: failed");
        }
      } else {
        RCLCPP_INFO(get_logger(), "Result of EXTMONORDER: failed unaware");
      }
      result_pub_->publish(respond);
      RCLCPP_INFO(get_logger(), "EXTMONORDER end");
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ActionClient>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
