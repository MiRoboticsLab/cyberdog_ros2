// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd.
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
#include "camera_service/main_camera_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto camera_node = std::make_shared<cyberdog_camera::CameraServerNode>();
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(camera_node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();

  return 0;
}
