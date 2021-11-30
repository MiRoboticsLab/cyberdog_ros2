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
#include <utility>

#include "cyberdog_gps/cyberdog_gps.hpp"

namespace CyberdogGpsDriver
{
GpsDriver::GpsDriver()
: cyberdog_utils::LifecycleNode("GpsDriver")
{
  RCLCPP_INFO(get_logger(), "Creating GpsDriver.");
}

GpsDriver::~GpsDriver()
{
  RCLCPP_INFO(get_logger(), "Destroying GpsDriver");
}

cyberdog_utils::CallbackReturn GpsDriver::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Configuring");
  publisher_ = this->create_publisher<ception_msgs::msg::GpsPayload>(
    "gps_raw",
    rclcpp::SystemDefaultsQoS());
  gps_ = nullptr;
  classloader_ = nullptr;
  classloader_ = std::make_shared<pluginlib::ClassLoader<ception_base::Cyberdog_GPS>>(
    "ception_base", "ception_base::Cyberdog_GPS");
  gps_ = classloader_->createSharedInstance("bcmgps_plugin::Cyberdog_BCMGPS");
  gps_->SetPayloadCallback(std::bind(&GpsDriver::payload_callback, this, std::placeholders::_1));
  gps_->Open();
  RCLCPP_INFO(get_logger(), "Configuring,success");
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsDriver::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Activaing");
  gps_->Start();
  publisher_->on_activate();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsDriver::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactiving");
  gps_->Stop();
  publisher_->on_deactivate();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsDriver::on_cleanup(const rclcpp_lifecycle::State &)
{
  gps_->Close();
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsDriver::on_shutdown(const rclcpp_lifecycle::State &)
{
  gps_->Close();
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void GpsDriver::payload_callback(std::shared_ptr<ception_base::Cyberdog_GPS_payload> payload)
{
  payload_ = payload;
  auto message = ception_msgs::msg::GpsPayload();
  message.itow = payload->iTOW;
  message.fix_type = payload->fixType;
  message.num_sv = payload->numSV;
  message.lon = payload->lon;
  message.lat = payload->lat;
  publisher_->publish(std::move(message));
}
}  // namespace CyberdogGpsDriver
