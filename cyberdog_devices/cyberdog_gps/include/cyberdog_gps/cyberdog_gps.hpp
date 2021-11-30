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

#ifndef CYBERDOG_GPS__CYBERDOG_GPS_HPP_
#define CYBERDOG_GPS__CYBERDOG_GPS_HPP_

#include <memory>

#include "pluginlib/class_loader.hpp"
#include "ception_base/ception_base.hpp"
#include "ception_msgs/msg/gps_payload.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"

namespace CyberdogGpsDriver
{
class GpsDriver : public cyberdog_utils::LifecycleNode
{
public:
  GpsDriver();
  ~GpsDriver();

protected:
  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  std::shared_ptr<pluginlib::ClassLoader<ception_base::Cyberdog_GPS>> classloader_;
  std::shared_ptr<ception_base::Cyberdog_GPS> gps_;
  std::shared_ptr<ception_base::Cyberdog_GPS_payload> payload_;

  void payload_callback(std::shared_ptr<ception_base::Cyberdog_GPS_payload> payload);
  rclcpp_lifecycle::LifecyclePublisher<ception_msgs::msg::GpsPayload>::SharedPtr publisher_;
};
}  // namespace CyberdogGpsDriver

#endif  // CYBERDOG_GPS__CYBERDOG_GPS_HPP_
