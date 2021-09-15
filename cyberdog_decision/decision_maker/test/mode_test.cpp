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

#include "rclcpp/rclcpp.hpp"
#include "automation_msgs/srv/nav_mode.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"

class Test_Node : public rclcpp::Node
{
  using NavMode_T = automation_msgs::srv::NavMode;

public:
  Test_Node()
  : Node("test_node")
  {
    test_nav_mode_server_ = this->create_service<NavMode_T>(
      "nav_mode",
      std::bind(
        &Test_Node::test_nav_mode_callback, this, std::placeholders::_1,
        std::placeholders::_2, std::placeholders::_3));
  }

private:
  void test_nav_mode_callback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<NavMode_T::Request> request,
    std::shared_ptr<NavMode_T::Response> response)
  {
    (void)request_header;
    (void)request;
    response->success = true;
  }
  rclcpp::Service<NavMode_T>::SharedPtr test_nav_mode_server_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node_test = std::make_shared<Test_Node>();
  auto node_cascade0 = std::make_shared<cyberdog_utils::LifecycleNode>("interactive");
  auto node_cascade1 = std::make_shared<cyberdog_utils::LifecycleNode>("rtabmap");
  auto node_cascade2 = std::make_shared<cyberdog_utils::LifecycleNode>("move_base_node");
  auto node_cascade3 = std::make_shared<cyberdog_utils::LifecycleNode>("tracking");
  rclcpp::executors::MultiThreadedExecutor exec_;

  exec_.add_node(node_test->get_node_base_interface());
  exec_.add_node(node_cascade0->get_node_base_interface());
  exec_.add_node(node_cascade1->get_node_base_interface());
  exec_.add_node(node_cascade2->get_node_base_interface());
  exec_.add_node(node_cascade3->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();

  return 0;
}
