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

#include <termios.h>    // termios, TCSANOW, ECHO, ICANON
#include <unistd.h>     // STDIN_FILENO
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <syslog.h>
#include <math.h>

#include <memory>
#include <utility>
#include <string>
#include <fstream>
#include <thread>
#include <algorithm>
#include <cctype>
#include <vector>
#include <map>

#include "sys/stat.h"
#include "rclcpp/rclcpp.hpp"

#include "std_msgs/msg/string.hpp"
#include "motion_msgs/msg/scene.hpp"
#include "ception_msgs/srv/gps_scene_node.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"

#include "ception_base/ception_base.hpp"
#include "pluginlib/class_loader.hpp"

namespace SceneDetection
{
typedef enum
{
  UNSET,
  INDOOR,
  OUTDOOR
} detection_scene;

class GpsPubNode : public cyberdog_utils::LifecycleNode
{
public:
  GpsPubNode();
  ~GpsPubNode();

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

  void gps_data_receiver_callback(void);
  void payload_callback(std::shared_ptr<ception_base::Cyberdog_GPS_payload> payload);

  void handle_service(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Request> request,
    const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Response> response);

  // message struct define here
  motion_msgs::msg::Scene message;
  rclcpp::TimerBase::SharedPtr timer_;

  /* Service */
  rclcpp::Service<ception_msgs::srv::GpsSceneNode>::SharedPtr scene_detection_cmd_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp_lifecycle::LifecyclePublisher<motion_msgs::msg::Scene>::SharedPtr publisher_;
};  // class GpsPubNode
}  // namespace SceneDetection

namespace SceneDetection
{

GpsPubNode::GpsPubNode()
: cyberdog_utils::LifecycleNode("GpsPubNode")
{
  RCLCPP_INFO(get_logger(), "Creating GpsPubNode.");
}

GpsPubNode::~GpsPubNode()
{
  RCLCPP_INFO(get_logger(), "Destroying GpsPubNode");
}

cyberdog_utils::CallbackReturn GpsPubNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Configuring");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  publisher_ = this->create_publisher<motion_msgs::msg::Scene>(
    "SceneDetection",
    rclcpp::SystemDefaultsQoS());
  scene_detection_cmd_server_ = this->create_service<ception_msgs::srv::GpsSceneNode>(
    "SceneDetection",
    std::bind(
      &GpsPubNode::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);
  message = motion_msgs::msg::Scene();

  classloader_ = std::make_shared<pluginlib::ClassLoader<ception_base::Cyberdog_GPS>>(
    "ception_base", "ception_base::Cyberdog_GPS");
  gps_ = classloader_->createSharedInstance("bcmgps_plugin::Cyberdog_BCMGPS");
  gps_->SetPayloadCallback(
    std::bind(
      &GpsPubNode::payload_callback, this, std::placeholders::_1
  ));
  gps_->Open();

  RCLCPP_INFO(get_logger(), "Configuring,success");
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Activaing");
  gps_->Start();
  publisher_->on_activate();
  timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&SceneDetection::GpsPubNode::gps_data_receiver_callback, this));
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactiving");
  gps_->Stop();
  publisher_->on_deactivate();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  gps_->Close();
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn GpsPubNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  gps_->Close();
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void GpsPubNode::gps_data_receiver_callback()
{
  auto message = motion_msgs::msg::Scene();
  detection_scene environment = UNSET;
  if (payload_ != nullptr) {
    if (payload_->fixType == 2 || payload_->fixType == 3) {
      environment = OUTDOOR;
    } else {
      environment = INDOOR;
    }
    message.lat = payload_->lat;
    message.lon = payload_->lon;
    message.if_danger = false;
  } else {
    message.lat = 0;
    message.lon = 0;
    message.if_danger = false;
  }
  message.type = environment;
  publisher_->publish(std::move(message));
}

void GpsPubNode::payload_callback(std::shared_ptr<ception_base::Cyberdog_GPS_payload> payload)
{
  payload_ = payload;
}

void GpsPubNode::handle_service(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Request> request,
  const std::shared_ptr<ception_msgs::srv::GpsSceneNode::Response> response)
{
  (void)request_header;
  RCLCPP_INFO(get_logger(), "request: %d", request->command);
  response->success = true;

  switch (request->command) {
    case ception_msgs::srv::GpsSceneNode::Request::GPS_START:
      gps_->Start();
      break;
    case ception_msgs::srv::GpsSceneNode::Request::GPS_STOP:
      gps_->Stop();
      break;
    default:
      response->success = false;
      break;
  }
}
}  // namespace SceneDetection

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor exec_;
  auto node = std::make_shared<SceneDetection::GpsPubNode>();
  exec_.add_node(node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();
  return 0;
}
