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
#include <string>
#include <cstring>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "bms_common.hpp"

int mode;
using std::placeholders::_1;

rclcpp::Node::SharedPtr node_handle = nullptr;

class Bms_Test : public rclcpp::Node
{
public:
  Bms_Test()
  : Node("test")
  {
    bms_test_pub = this->create_publisher<cyberdog_interfaces::msg::Bms>("buzze", 10);

    while (true) {
      printf("Input buzze status\n");
      cin >> mode;
      printf("buzze  mode:%d\n", mode);
      msg.buzze = mode;

      if (mode != 0) {
        bms_test_pub->publish(msg);
      }
    }
  }

private:
  rclcpp::Publisher<cyberdog_interfaces::msg::Bms>::SharedPtr bms_test_pub;
  cyberdog_interfaces::msg::Bms msg;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  node_handle = rclcpp::Node::make_shared("bms_test_node");

  rclcpp::spin(std::make_shared<Bms_Test>());

  rclcpp::shutdown();
  return 0;
}
