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

#include <lcm/lcm-cpp.hpp>
#include <string>
#include <memory>
#include "bms_common.hpp"
#include "rclcpp/rclcpp.hpp"
#include "lcm_translate_msgs/bms_request_lcmt.hpp"
#include "lcm_translate_msgs/bms_response_lcmt.hpp"

using std::placeholders::_1;

class Bms : public rclcpp::Node
{
public:
  Bms()
  : Node("bms_ctrl")
  {
    buzze_sub = this->create_subscription<ception_msgs::msg::Bms>(
      "buzze", 10, std::bind(&Bms::set_buzze, this, _1));

    power_supply_sub = this->create_subscription<ception_msgs::msg::Bms>(
      "power_supply", 10, std::bind(&Bms::set_power_supply, this, _1));

    disable_charge_sub = this->create_subscription<ception_msgs::msg::Bms>(
      "disable_charge", 10, std::bind(&Bms::set_disable_charge, this, _1));
  }

private:
  void set_buzze(const ception_msgs::msg::Bms::SharedPtr msg);
  void set_power_supply(const ception_msgs::msg::Bms::SharedPtr msg);
  void set_disable_charge(const ception_msgs::msg::Bms::SharedPtr msg);
  rclcpp::Subscription<ception_msgs::msg::Bms>::SharedPtr buzze_sub;
  rclcpp::Subscription<ception_msgs::msg::Bms>::SharedPtr power_supply_sub;
  rclcpp::Subscription<ception_msgs::msg::Bms>::SharedPtr disable_charge_sub;
  lcm::LCM bms_request;
  bms_request_lcmt bms_lcm_data;
};

void Bms::set_buzze(const ception_msgs::msg::Bms::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "bms set buzz :%d", msg->buzze);

  bms_lcm_data.buzz = msg->buzze;
  bms_request.publish("bms_command", &bms_lcm_data);
}

void Bms::set_power_supply(const ception_msgs::msg::Bms::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "bms set buzz :%d", msg->power_supply);

  bms_lcm_data.power_supply = msg->power_supply;

  bms_request.publish("bms_command", &bms_lcm_data);
}

void Bms::set_disable_charge(const ception_msgs::msg::Bms::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "bms set charge enable:%d", msg->disable_charge);

  bms_lcm_data.charge_enable = msg->disable_charge;

  bms_request.publish("bms_command", &bms_lcm_data);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<Bms>());

  rclcpp::shutdown();
  return 0;
}
