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
#include "ception_msgs/msg/body_state.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"
#include "cyberdog_utils/can/can_utils.hpp"
#include "cyberdog_utils/sensor/sensor_utils.hpp"

#ifndef UNUSED_VAR
#define UNUSED_VAR(var)  ((void)(var));
#endif

using namespace std::chrono_literals;

namespace cyberdog_body_state
{
class BodyStatePubNode : public cyberdog_utils::LifecycleNode
{
public:
  BodyStatePubNode();
  ~BodyStatePubNode();

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
  ception_msgs::msg::BodyState message;
  sensor_data_converter * publish_sensor_data = nullptr;
  bool isthreadrunning;
  cyberdog_utils::can_dev_operation * can_dev_op = nullptr;
  cyberdog_utils::can_dev_operation * can_tim_op = nullptr;
  cyberdog_utils::cyberdog_sensor * sensor_obj = nullptr;
  rclcpp::TimerBase::SharedPtr timer_;
  /* Service */
  uint64_t rotv_enable_count;
  uint64_t speed_vec_enable_count;
  rclcpp::Service<ception_msgs::srv::SensorDetectionNode>::SharedPtr bodystate_cmd_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp_lifecycle::LifecyclePublisher<ception_msgs::msg::BodyState>::SharedPtr publisher_;
};
}  // namespace cyberdog_body_state

namespace cyberdog_body_state
{

BodyStatePubNode::BodyStatePubNode()
: cyberdog_utils::LifecycleNode("BodyStatePubNode")
{
  RCLCPP_INFO(get_logger(), "Creating BodyStatePubNode.");
}

BodyStatePubNode::~BodyStatePubNode()
{
  RCLCPP_INFO(get_logger(), "Destroying BodyStatePubNode");
}

cyberdog_utils::CallbackReturn BodyStatePubNode::on_configure(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Configuring");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  publisher_ = this->create_publisher<ception_msgs::msg::BodyState>(
    "BodyState",
    rclcpp::SystemDefaultsQoS());
  bodystate_cmd_server_ = this->create_service<ception_msgs::srv::SensorDetectionNode>(
    "cyberdog_body_state",
    std::bind(
      &BodyStatePubNode::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  message = ception_msgs::msg::BodyState();

  can_dev_op = new cyberdog_utils::can_dev_operation();
  can_tim_op = new cyberdog_utils::can_dev_operation();
  sensor_obj = new cyberdog_utils::cyberdog_sensor(2);
  sensor_obj->set_convert_sensortype(1 << SENSOR_TYPE_ROTATION_VECTOR);
  sensor_obj->set_convert_sensortype(1 << SENSOR_TYPE_SPEED_VECTOR);
  sensor_obj->find_sensor_filter_stdid();
  publish_sensor_data =
    reinterpret_cast<sensor_data_converter *>((malloc(sizeof(sensor_data_converter))));
  memset(publish_sensor_data, 0, sizeof(sensor_data_converter));
  for (int i = 0; i < SENSOR_TYPE_MAX; i++) {
    publish_sensor_data->sensor_data[i].sensor_type = i;
  }
  rotv_enable_count = 0;
  speed_vec_enable_count = 0;
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn BodyStatePubNode::on_activate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Activaing");
  publisher_->on_activate();
  isthreadrunning = true;
  can_data_receiver_thread = std::thread(
    &cyberdog_body_state::BodyStatePubNode::can_data_receiver_callback, this);
  can_timestamp_receiver_thread = std::thread(
    &cyberdog_body_state::BodyStatePubNode::can_timestamp_receiver_callback, this);
  // send stm32 timestamp sync message here to get mapping between stm32 & NV board
  can_frame stm32_head_timesync;
  stm32_head_timesync.can_id = 0x630;
  can_frame stm32_bot_timesync;
  stm32_bot_timesync.can_id = 0x620;
  can_tim_op->send_can_message(stm32_head_timesync);
  can_tim_op->send_can_message(stm32_bot_timesync);

  timer_ = this->create_wall_timer(
    1000ms, std::bind(&BodyStatePubNode::heard_attack_checker_callback, this));

  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn BodyStatePubNode::on_deactivate(
  const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Deactiving");
  publisher_->on_deactivate();
  isthreadrunning = false;
  can_data_receiver_thread.join();
  can_timestamp_receiver_thread.join();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn BodyStatePubNode::on_cleanup(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn BodyStatePubNode::on_shutdown(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void BodyStatePubNode::heard_attack_checker_callback()
{
  // RCLCPP_INFO(
  //    get_logger(), "\n @@@heard_attack_checker_callback: front, %d, back, %d",
  //    front_enable_count, back_enable_count);

  if (rotv_enable_count > 0) {
    if (sensor_obj->sensor_data_heart_attack_checker(SENSOR_TYPE_ROTATION_VECTOR)) {
      // record the checker set time and power off/on when checker return true
    }
  }
  if (speed_vec_enable_count > 0) {
    if (sensor_obj->sensor_data_heart_attack_checker(SENSOR_TYPE_SPEED_VECTOR)) {
      // record the checker set time and power off/on when checker return true
    }
  }
}

void BodyStatePubNode::can_data_receiver_callback()
{
  while (isthreadrunning) {
    if (can_dev_op->wait_for_can_data() == 0) {
      // RCLCPP_INFO(
      // get_logger(), "\n ###can data received!");

      if (sensor_obj->convert_sensor_data(publish_sensor_data, &can_dev_op->recv_frame)) {
        // RCLCPP_INFO(
        //  get_logger(), "\n @@@sensor data received!");
        if (((can_dev_op->recv_frame.can_id & 0x00F0) >> SENSOR_TYPE_BIT_SHIFT) ==
          SENSOR_TYPE_ROTATION_VECTOR)
        {
          auto posequat_time =
            std::chrono::nanoseconds(
            publish_sensor_data->sensor_data[SENSOR_TYPE_ROTATION_VECTOR].timestamp);
          builtin_interfaces::msg::Time posequat_stamp;
          posequat_stamp.sec =
            std::chrono::duration_cast<std::chrono::seconds>(posequat_time).count();
          posequat_stamp.nanosec =
            (posequat_time -
            std::chrono::duration_cast<std::chrono::seconds>(posequat_time)).count();

          message.header_q4.stamp = posequat_stamp;
          message.posequat.x =
            publish_sensor_data->sensor_data[SENSOR_TYPE_ROTATION_VECTOR].sensor_data_t.vec.data[0
            ];
          message.posequat.y =
            publish_sensor_data->sensor_data[SENSOR_TYPE_ROTATION_VECTOR].sensor_data_t.vec.data[1
            ];
          message.posequat.z =
            publish_sensor_data->sensor_data[SENSOR_TYPE_ROTATION_VECTOR].sensor_data_t.vec.data[2
            ];
          message.posequat.w =
            publish_sensor_data->sensor_data[SENSOR_TYPE_ROTATION_VECTOR].sensor_data_t.vec.data[3
            ];

          sensor_obj->log_prev_sensor_data(
            publish_sensor_data->sensor_data[
              SENSOR_TYPE_ROTATION_VECTOR], SENSOR_TYPE_ROTATION_VECTOR);
        } else if (((can_dev_op->recv_frame.can_id & 0x00F0) >> SENSOR_TYPE_BIT_SHIFT) ==  // NOLINT
          SENSOR_TYPE_SPEED_VECTOR)
        {
          auto speed_vec_time =
            std::chrono::nanoseconds(
            publish_sensor_data->sensor_data[SENSOR_TYPE_SPEED_VECTOR].timestamp);
          builtin_interfaces::msg::Time speed_vec_stamp;
          speed_vec_stamp.sec =
            std::chrono::duration_cast<std::chrono::seconds>(speed_vec_time).count();
          speed_vec_stamp.nanosec =
            (speed_vec_time -
            std::chrono::duration_cast<std::chrono::seconds>(speed_vec_time)).count();

          message.header_v3.stamp = speed_vec_stamp;
          message.speed_vector.vector.x =
            publish_sensor_data->sensor_data[SENSOR_TYPE_SPEED_VECTOR].sensor_data_t.vec.data[0];
          message.speed_vector.vector.y =
            publish_sensor_data->sensor_data[SENSOR_TYPE_SPEED_VECTOR].sensor_data_t.vec.data[1];
          message.speed_vector.vector.z =
            publish_sensor_data->sensor_data[SENSOR_TYPE_SPEED_VECTOR].sensor_data_t.vec.data[2];

          sensor_obj->log_prev_sensor_data(
            publish_sensor_data->sensor_data[SENSOR_TYPE_SPEED_VECTOR], SENSOR_TYPE_SPEED_VECTOR);
        }

        publisher_->publish(std::move(message));
      }
    }
  }
}

void BodyStatePubNode::can_timestamp_receiver_callback()
{
  int ret = 0;
  while (isthreadrunning) {
    if (can_tim_op->wait_for_can_data() == 0) {
      // RCLCPP_INFO(
      //  get_logger(), "###can data received!");

      if (sensor_obj->convert_timestamp_data(&can_tim_op->recv_frame)) {
        RCLCPP_INFO(
          get_logger(), "@@@BodyStatePubNode, sensor timestamp received! 0x%x",
          can_tim_op->recv_frame.can_id);

        // reset enable status when receiving this kind of message
        // since the stm32 will reboot due to heart attack check or other reasons
        can_frame command_frame;
        if (rotv_enable_count > 0) {
          command_frame.can_id = 0x0D1;
          ret = can_tim_op->send_can_message(command_frame);
        }
        if (speed_vec_enable_count > 0) {
          command_frame.can_id = 0x0E1;
          ret = can_tim_op->send_can_message(command_frame);
        }
      }
      UNUSED_VAR(ret);
    }
  }
}

void BodyStatePubNode::handle_service(
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
  sensor_obj->previous_sensor_data[SENSOR_TYPE_ROTATION_VECTOR].timestamp =
    sensor_obj->get_current_nv_time();

  can_frame command_frame;

  switch (request->command) {
    case ception_msgs::srv::SensorDetectionNode::Request::ENABLE_ROTATION_VECTOR:
      if (rotv_enable_count == 0) {
        command_frame.can_id = 0x0D1;
        ret = can_dev_op->send_can_message(command_frame);
      }
      rotv_enable_count++;
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::ENABLE_SPEED_VECTOR:
      if (speed_vec_enable_count == 0) {
        command_frame.can_id = 0x0E1;
        ret = can_dev_op->send_can_message(command_frame);
      }
      speed_vec_enable_count++;
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::DISABLE_ROTATION_VECTOR:
      if (rotv_enable_count > 0) {
        rotv_enable_count--;
      }
      if (rotv_enable_count == 0) {
        command_frame.can_id = 0x0D0;
        ret = can_dev_op->send_can_message(command_frame);
      }

      break;
    case ception_msgs::srv::SensorDetectionNode::Request::DISABLE_SPEED_VECTOR:
      if (speed_vec_enable_count > 0) {
        speed_vec_enable_count--;
      }
      if (speed_vec_enable_count == 0) {
        command_frame.can_id = 0x0E0;
        ret = can_dev_op->send_can_message(command_frame);
      }
      break;
    default:
      response->success = false;
      break;
  }
  if (ret != 0) {
    response->success = false;
  }

  response->clientcount = this->count_subscribers("BodyState");
}
}  // namespace cyberdog_body_state

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyberdog_body_state::BodyStatePubNode>();
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();
  return 0;
}
