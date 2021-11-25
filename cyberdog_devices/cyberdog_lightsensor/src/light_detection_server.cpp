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

#include <inttypes.h>
#include <memory>
#include <thread>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "ception_msgs/msg/light.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"
#include "cyberdog_utils/can/can_utils.hpp"
#include "cyberdog_utils/sensor/sensor_utils.hpp"

#ifndef UNUSED_VAR
#define UNUSED_VAR(var)  ((void)(var));
#endif

using namespace std::chrono_literals;

namespace light_detection
{
class LightDetPubNode : public cyberdog_utils::LifecycleNode
{
public:
  LightDetPubNode();
  ~LightDetPubNode();

protected:
  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  void heard_attack_checker_callback(void);
  void can_data_receiver_callback(void);
  void can_timestamp_receiver_callback(void);
  void handle_service(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> request,
    const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Response> response);
  std::thread can_data_receiver_thread;
  std::thread can_timestamp_receiver_thread;
  // message struct define here
  ception_msgs::msg::Light message;
  sensor_data_converter * publish_sensor_data = nullptr;
  bool isthreadrunning;
  cyberdog_utils::can_dev_operation * can_dev_op = nullptr;
  cyberdog_utils::can_dev_operation * can_tim_op = nullptr;
  cyberdog_utils::cyberdog_sensor * sensor_obj = nullptr;
  rclcpp::TimerBase::SharedPtr timer_;
  /* Service */
  uint64_t light_enable_count;
  rclcpp::Service<ception_msgs::srv::SensorDetectionNode>::SharedPtr light_cmd_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp_lifecycle::LifecyclePublisher<ception_msgs::msg::Light>::SharedPtr publisher_;
};
}  // namespace light_detection

namespace light_detection
{

LightDetPubNode::LightDetPubNode()
: cyberdog_utils::LifecycleNode("LightDetPubNode")
{
  RCLCPP_INFO(get_logger(), "Creating LightDetPubNode.");
}

LightDetPubNode::~LightDetPubNode()
{
  RCLCPP_INFO(get_logger(), "Destroying LightDetPubNode");
}

cyberdog_utils::CallbackReturn LightDetPubNode::on_configure(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Configuring");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  publisher_ = this->create_publisher<ception_msgs::msg::Light>(
    "LightDetection",
    rclcpp::SystemDefaultsQoS());
  light_cmd_server_ = this->create_service<ception_msgs::srv::SensorDetectionNode>(
    "light_detection",
    std::bind(
      &LightDetPubNode::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  message = ception_msgs::msg::Light();

  can_dev_op = new cyberdog_utils::can_dev_operation();
  can_tim_op = new cyberdog_utils::can_dev_operation();
  sensor_obj = new cyberdog_utils::cyberdog_sensor(1);
  sensor_obj->set_convert_sensortype(1 << SENSOR_TYPE_LIGHT);
  sensor_obj->find_sensor_filter_stdid();
  publish_sensor_data =
    reinterpret_cast<sensor_data_converter *>(malloc(sizeof(sensor_data_converter)));
  memset(publish_sensor_data, 0, sizeof(sensor_data_converter));
  for (int i = 0; i < SENSOR_TYPE_MAX; i++) {
    publish_sensor_data->sensor_data[i].sensor_type = i;
  }
  light_enable_count = 0;
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LightDetPubNode::on_activate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Activaing");
  publisher_->on_activate();
  isthreadrunning = true;
  can_data_receiver_thread = std::thread(
    &light_detection::LightDetPubNode::can_data_receiver_callback, this);
  can_timestamp_receiver_thread = std::thread(
    &light_detection::LightDetPubNode::can_timestamp_receiver_callback, this);
  // send stm32 timestamp sync message here to get mapping between stm32 & NV board
  can_frame stm32_head_timesync;
  stm32_head_timesync.can_id = 0x630;
  can_tim_op->send_can_message(stm32_head_timesync);

  timer_ = this->create_wall_timer(
    1000ms, std::bind(&LightDetPubNode::heard_attack_checker_callback, this));

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LightDetPubNode::on_deactivate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Deactiving");
  publisher_->on_deactivate();
  isthreadrunning = false;
  can_data_receiver_thread.join();
  can_timestamp_receiver_thread.join();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LightDetPubNode::on_cleanup(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LightDetPubNode::on_shutdown(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void LightDetPubNode::heard_attack_checker_callback()
{
  // RCLCPP_INFO(
  //    get_logger(), "\n @@@heard_attack_checker_callback: light, %d", light_enable_count);
  if (light_enable_count > 0) {
    if (sensor_obj->sensor_data_heart_attack_checker(SENSOR_TYPE_LIGHT)) {
      // record the checker set time and power off/on when checker return true
    }
  }
}

void LightDetPubNode::can_data_receiver_callback()
{
  while (isthreadrunning) {
    if (can_dev_op->wait_for_can_data() == 0) {
      // RCLCPP_INFO(
      //  get_logger(), "\n ###can data received!");

      if (sensor_obj->convert_sensor_data(publish_sensor_data, &can_dev_op->recv_frame)) {
        auto lux_time = std::chrono::nanoseconds(
          publish_sensor_data->sensor_data[SENSOR_TYPE_LIGHT].timestamp);
        builtin_interfaces::msg::Time lux_stamp;
        lux_stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(lux_time).count();
        lux_stamp.nanosec =
          (lux_time - std::chrono::duration_cast<std::chrono::seconds>(lux_time)).count();

        message.illuminance_info.header.stamp = lux_stamp;
        message.illuminance_info.illuminance =
          publish_sensor_data->sensor_data[SENSOR_TYPE_LIGHT].sensor_data_t.vec.data[0];

        // RCLCPP_INFO(
        //  get_logger(), "\n @@@light sensor data received: %f", message.lux);

        sensor_obj->log_prev_sensor_data(
          publish_sensor_data->sensor_data[SENSOR_TYPE_LIGHT],
          SENSOR_TYPE_LIGHT);
        publisher_->publish(std::move(message));
      }
    }
  }
}

void LightDetPubNode::can_timestamp_receiver_callback()
{
  int ret = 0;
  while (isthreadrunning) {
    if (can_tim_op->wait_for_can_data() == 0) {
      // RCLCPP_INFO(
      //  get_logger(), "\n ###can data received!");

      if (sensor_obj->convert_timestamp_data(&can_tim_op->recv_frame)) {
        RCLCPP_INFO(
          get_logger(), "\n @@@sensor timestamp received!");
        // reset enable status when receiving this kind of message
        // since the stm32 will reboot due to heart attack check or other reasons
        can_frame light_enable_cmd;
        light_enable_cmd.can_id = 0x031;

        if (light_enable_count > 0) {
          ret = can_dev_op->send_can_message(light_enable_cmd);
        }
      }
      UNUSED_VAR(ret);
    }
  }
}

void LightDetPubNode::handle_service(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> request,
  const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Response> response)
{
  (void)request_header;
  RCLCPP_INFO(
    get_logger(),
    "request: %d", request->command);

  response->success = true;
  int ret = 0;
  sensor_obj->previous_sensor_data[SENSOR_TYPE_LIGHT].timestamp = sensor_obj->get_current_nv_time();

  can_frame light_cmd;

  switch (request->command) {
    case ception_msgs::srv::SensorDetectionNode::Request::ENABLE_ALL:
      if (light_enable_count == 0) {
        light_cmd.can_id = 0x031;
        ret = can_dev_op->send_can_message(light_cmd);
      }
      light_enable_count++;
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::DISABLE_ALL:
      if (light_enable_count > 0) {
        light_enable_count--;
      }
      if (light_enable_count == 0) {
        light_cmd.can_id = 0x030;
        ret = can_dev_op->send_can_message(light_cmd);
      }
      break;
    default:
      response->success = false;
      break;
  }

  if (ret != 0) {
    response->success = false;
  }

  response->clientcount = this->count_subscribers("LightDetection");
}
}  // namespace light_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<light_detection::LightDetPubNode>();
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();
  return 0;
}
