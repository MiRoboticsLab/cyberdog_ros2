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
#include "ception_msgs/msg/around.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"
#include "cyberdog_utils/can/can_utils.hpp"
#include "cyberdog_utils/sensor/sensor_utils.hpp"

// #include <condition_variable>
// #include <mutex>
// using namespace std::chrono_literals; //timer related namespace

/**
 * A small convenience function for converting a thread ID to a string
 **/
// std::string string_thread_id()
// {
//   auto hashed = std::hash<std::thread::id>()(std::this_thread::get_id());
//   return std::to_string(hashed);
// }

#ifndef UNUSED_VAR
#define UNUSED_VAR(var)  ((void)(var));
#endif

using namespace std::chrono_literals;

namespace obstacle_detection
{
class ObDetPubNode : public cyberdog_utils::LifecycleNode
{
public:
  ObDetPubNode();
  ~ObDetPubNode();

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
  ception_msgs::msg::Around message;
  sensor_data_converter * publish_sensor_data = nullptr;
  bool isthreadrunning;
  cyberdog_utils::can_dev_operation * can_dev_op = nullptr;
  cyberdog_utils::can_dev_operation * can_tim_op = nullptr;
  cyberdog_utils::cyberdog_sensor * sensor_obj = nullptr;
  rclcpp::TimerBase::SharedPtr timer_;
  /* Service */
  uint64_t front_enable_count;
  uint64_t back_enable_count;
  rclcpp::Service<ception_msgs::srv::SensorDetectionNode>::SharedPtr obstacle_cmd_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp_lifecycle::LifecyclePublisher<ception_msgs::msg::Around>::SharedPtr publisher_;
};
}  // namespace obstacle_detection

namespace obstacle_detection
{

ObDetPubNode::ObDetPubNode()
: cyberdog_utils::LifecycleNode("ObDetPubNode")
{
  RCLCPP_INFO(get_logger(), "Creating ObDetPubNode.");
}

ObDetPubNode::~ObDetPubNode()
{
  RCLCPP_INFO(get_logger(), "Destroying ObDetPubNode");
}

cyberdog_utils::CallbackReturn ObDetPubNode::on_configure(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Configuring");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  publisher_ = this->create_publisher<ception_msgs::msg::Around>(
    "ObstacleDetection",
    rclcpp::SystemDefaultsQoS());
  obstacle_cmd_server_ = this->create_service<ception_msgs::srv::SensorDetectionNode>(
    "obstacle_detection",
    std::bind(
      &ObDetPubNode::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  message = ception_msgs::msg::Around();

  can_dev_op = new cyberdog_utils::can_dev_operation();
  can_tim_op = new cyberdog_utils::can_dev_operation();
  sensor_obj = new cyberdog_utils::cyberdog_sensor(2);
  sensor_obj->set_convert_sensortype(1 << SENSOR_TYPE_PROXIMITY_HEAD);
  sensor_obj->set_convert_sensortype(1 << SENSOR_TYPE_PROXIMITY_REAR);
  sensor_obj->find_sensor_filter_stdid();
  publish_sensor_data =
    reinterpret_cast<sensor_data_converter *>(malloc(sizeof(sensor_data_converter)));
  memset(publish_sensor_data, 0, sizeof(sensor_data_converter));
  for (int i = 0; i < SENSOR_TYPE_MAX; i++) {
    publish_sensor_data->sensor_data[i].sensor_type = i;
  }
  front_enable_count = 0;
  back_enable_count = 0;
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn ObDetPubNode::on_activate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Activaing");
  publisher_->on_activate();
  isthreadrunning = true;
  can_data_receiver_thread = std::thread(
    &obstacle_detection::ObDetPubNode::can_data_receiver_callback, this);
  can_timestamp_receiver_thread = std::thread(
    &obstacle_detection::ObDetPubNode::can_timestamp_receiver_callback, this);
  // send stm32 timestamp sync message here to get mapping between stm32 & NV board
  can_frame stm32_head_timesync;
  can_frame stm32_rear_timesync;
  stm32_head_timesync.can_id = 0x630;
  stm32_rear_timesync.can_id = 0x600;
  can_tim_op->send_can_message(stm32_head_timesync);
  can_tim_op->send_can_message(stm32_rear_timesync);

  timer_ = this->create_wall_timer(
    1000ms, std::bind(&ObDetPubNode::heard_attack_checker_callback, this));

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn ObDetPubNode::on_deactivate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Deactiving");
  publisher_->on_deactivate();
  isthreadrunning = false;
  can_data_receiver_thread.join();
  can_timestamp_receiver_thread.join();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn ObDetPubNode::on_cleanup(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn ObDetPubNode::on_shutdown(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void ObDetPubNode::heard_attack_checker_callback()
{
  // RCLCPP_INFO(
  //   get_logger(), "\n @@@heard_attack_checker_callback: front, %d, back, %d", front_enable_count,
  //   back_enable_count);
  if (front_enable_count > 0) {
    if (sensor_obj->sensor_data_heart_attack_checker(SENSOR_TYPE_PROXIMITY_HEAD)) {
      // record the checker set time and power off/on when checker return true
    }
  }
}

void ObDetPubNode::can_data_receiver_callback()
{
  while (isthreadrunning) {
    if (can_dev_op->wait_for_can_data() == 0) {
      // RCLCPP_INFO(
      //   get_logger(), "\n ###can data received!");

      if (sensor_obj->convert_sensor_data(publish_sensor_data, &can_dev_op->recv_frame)) {
        // RCLCPP_INFO(
        //   get_logger(), "\n @@@sensor data received!");

        if (sensor_obj->check_sensor_data_valid(publish_sensor_data, SENSOR_TYPE_PROXIMITY_HEAD)) {
          auto front_ultrasound_time =
            std::chrono::nanoseconds(
            publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_HEAD].timestamp);
          builtin_interfaces::msg::Time front_ultrasound_stamp;
          front_ultrasound_stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(
            front_ultrasound_time).count();
          front_ultrasound_stamp.nanosec =
            (front_ultrasound_time -
            std::chrono::duration_cast<std::chrono::seconds>(front_ultrasound_time)).count();

          message.front_distance.range_info.header.stamp = front_ultrasound_stamp;
          message.front_distance.range_info.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
          message.front_distance.range_info.range =
            publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_HEAD].sensor_data_t.vec.data[0] /
            1000.0f;
          message.front_distance.range_info.max_range = 2.0f;
          message.front_distance.range_info.min_range = 0.2f;

          auto back_ultrasound_time =
            std::chrono::nanoseconds(
            publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_REAR].timestamp);
          builtin_interfaces::msg::Time back_ultrasound_stamp;
          back_ultrasound_stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(
            back_ultrasound_time).count();
          back_ultrasound_stamp.nanosec =
            (back_ultrasound_time -
            std::chrono::duration_cast<std::chrono::seconds>(back_ultrasound_time)).count();

          message.back_distance.range_info.header.stamp = back_ultrasound_stamp;
          message.back_distance.range_info.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
          message.back_distance.range_info.range =
            publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_REAR].sensor_data_t.vec.data[0] /
            1000.0f;
          message.back_distance.range_info.max_range = 2.0f;
          message.back_distance.range_info.min_range = 0.2f;
          // Prep display message
        } else {
          auto front_ultrasound_time =
            std::chrono::nanoseconds(
            publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_HEAD].timestamp);
          builtin_interfaces::msg::Time front_ultrasound_stamp;
          front_ultrasound_stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(
            front_ultrasound_time).count();
          front_ultrasound_stamp.nanosec =
            (front_ultrasound_time -
            std::chrono::duration_cast<std::chrono::seconds>(front_ultrasound_time)).count();

          message.front_distance.range_info.header.stamp = front_ultrasound_stamp;
          message.front_distance.range_info.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
          message.front_distance.range_info.range = 0.2f;
          message.front_distance.range_info.max_range = 2.0f;
          message.front_distance.range_info.min_range = 0.2f;

          auto back_ultrasound_time =
            std::chrono::nanoseconds(
            publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_REAR].timestamp);
          builtin_interfaces::msg::Time back_ultrasound_stamp;
          back_ultrasound_stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(
            back_ultrasound_time).count();
          back_ultrasound_stamp.nanosec =
            (back_ultrasound_time -
            std::chrono::duration_cast<std::chrono::seconds>(back_ultrasound_time)).count();

          message.back_distance.range_info.header.stamp = back_ultrasound_stamp;
          message.back_distance.range_info.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
          message.back_distance.range_info.range = 0.2f;
          message.back_distance.range_info.max_range = 2.0f;
          message.back_distance.range_info.min_range = 0.2f;
        }

        sensor_obj->log_prev_sensor_data(
          publish_sensor_data->sensor_data[SENSOR_TYPE_PROXIMITY_HEAD],
          SENSOR_TYPE_PROXIMITY_HEAD);
        /*
        RCLCPP_INFO(
          get_logger(), "\n<<THREAD>> ObDetPubNode Notifying subscriber '%f, %f'",
          message.front_distance.range_info.range, message.back_distance.range_info.range);
       */
        publisher_->publish(std::move(message));
      }
    }
  }
}

void ObDetPubNode::can_timestamp_receiver_callback()
{
  int ret = 0;
  while (isthreadrunning) {
    if (can_tim_op->wait_for_can_data() == 0) {
      // RCLCPP_INFO(
      //   get_logger(), "\n ###can data received!");

      if (sensor_obj->convert_timestamp_data(&can_tim_op->recv_frame)) {
        RCLCPP_INFO(
          get_logger(), "\n @@@sensor timestamp received!");

        // reset enable status when receiving this kind of message
        // since the stm32 will reboot due to heart attack check or other reasons
        can_frame commend_frame;

        if (front_enable_count > 0) {
          commend_frame.can_id = 0x051;
          ret = can_dev_op->send_can_message(commend_frame);
        }
        if (back_enable_count > 0) {
          commend_frame.can_id = 0x071;
          ret = can_dev_op->send_can_message(commend_frame);
        }
        UNUSED_VAR(ret);
      }
    }
  }
}

void ObDetPubNode::handle_service(
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
  sensor_obj->previous_sensor_data[SENSOR_TYPE_PROXIMITY_HEAD].timestamp =
    sensor_obj->get_current_nv_time();

  can_frame commend_frame;

  switch (request->command) {
    case ception_msgs::srv::SensorDetectionNode::Request::ENABLE_ALL:
      if (front_enable_count == 0) {
        commend_frame.can_id = 0x051;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      if (back_enable_count == 0) {
        commend_frame.can_id = 0x071;
        ret = can_dev_op->send_can_message(commend_frame);
      }

      front_enable_count++;
      back_enable_count++;
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::DISABLE_ALL:
      if (front_enable_count > 0) {
        front_enable_count--;
      }
      if (back_enable_count > 0) {
        back_enable_count--;
      }
      if (front_enable_count == 0) {
        commend_frame.can_id = 0x050;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      if (back_enable_count == 0) {
        commend_frame.can_id = 0x070;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::ENABLE_FROUNT:
      if (front_enable_count == 0) {
        commend_frame.can_id = 0x051;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      front_enable_count++;
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::ENABLE_BACK:
      if (back_enable_count == 0) {
        commend_frame.can_id = 0x071;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      back_enable_count++;
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::DISABLE_FROUNT:
      if (front_enable_count > 0) {
        front_enable_count--;
      }
      if (front_enable_count == 0) {
        commend_frame.can_id = 0x050;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::DISABLE_BACK:
      if (back_enable_count > 0) {
        back_enable_count--;
      }
      if (back_enable_count == 0) {
        commend_frame.can_id = 0x070;
        ret = can_dev_op->send_can_message(commend_frame);
      }
      break;
    default:
      response->success = false;
      break;
  }

  if (ret != 0) {
    response->success = false;
  }

  response->clientcount = this->count_subscribers("ObstacleDetection");
}
}  // namespace obstacle_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<obstacle_detection::ObDetPubNode>();
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();
  return 0;
}
