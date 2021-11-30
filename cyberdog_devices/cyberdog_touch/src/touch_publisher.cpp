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
#include <sys/time.h>

#include <memory>
#include <thread>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "interaction_msgs/msg/touch.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
#include "cyberdog_utils/lifecycle_node.hpp"
#include "cyberdog_utils/can/can_utils.hpp"
#include "cyberdog_utils/sensor/sensor_utils.hpp"
#include "TouchSensorHandler.hpp"

#ifndef UNUSED_VAR
#define UNUSED_VAR(var)  ((void)(var));
#endif

#define GESTURE_XM_ADDR                              300
#define LPWG_SINGLETAP_DETECTED                      0x01
#define LPWG_DOUBLETAP_DETECTED                      0x03
#define LPWG_TOUCHANDHOLD_DETECTED                   0x07
#define LPWG_CIRCLE_DETECTED                         0x08
#define LPWG_TRIANGLE_DETECTED                       0x09
#define LPWG_VEE_DETECTED                            0x0A
#define LPWG_UNICODE_DETECTED                        0x0B
#define LPWG_SWIPE_DETECTED                          0x0D
#define LPWG_SWIPE_DETECTED_UP_CONTINU               0x0E
#define LPWG_SWIPE_DETECTED_DOWN_CONTINU             0x0F
#define LPWG_SWIPE_DETECTED_LEFT_CONTINU             0x10
#define LPWG_SWIPE_DETECTED_RIGHT_CONTINU            0x11

#define LPWG_SWIPE_FINGER_NUM_MASK                   0xF0
#define LPWG_SWIPE_FINGER_UP_DOWN_DIR_MASK           0x03
#define LPWG_SWIPE_FINGER_LEFT_RIGHT_DIR_MASK        0x0C
#define LPWG_SWIPE_ID_SINGLE                         0x40
#define LPWG_SWIPE_ID_DOUBLE                         0x80
#define LPWG_SWIPE_UP                                0x01
#define LPWG_SWIPE_DOWN                              0x02
#define LPWG_SWIPE_LEFT                              0x04
#define LPWG_SWIPE_RIGHT                             0x08

namespace cyberdog_touch
{
class TouchPubNode : public cyberdog_utils::LifecycleNode
{
public:
  TouchPubNode();
  ~TouchPubNode();

protected:
  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  void touch_receiver_callback(void);
  std::thread touch_receiver_thread;
  // message struct define here
  interaction_msgs::msg::Touch message;
  bool isthreadrunning;
  TouchSensorHandler * mTouchSensorHandler;
  rclcpp_lifecycle::LifecyclePublisher<interaction_msgs::msg::Touch>::SharedPtr publisher_;
};
}  // namespace cyberdog_touch

namespace cyberdog_touch
{

TouchPubNode::TouchPubNode()
: cyberdog_utils::LifecycleNode("TouchPubNode")
{
  RCLCPP_INFO(get_logger(), "Creating TouchPubNode.");
}

TouchPubNode::~TouchPubNode()
{
  RCLCPP_INFO(get_logger(), "Destroying TouchPubNode");
}

cyberdog_utils::CallbackReturn TouchPubNode::on_configure(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Configuring");

  publisher_ = this->create_publisher<interaction_msgs::msg::Touch>(
    "TouchState",
    rclcpp::SystemDefaultsQoS());

  message = interaction_msgs::msg::Touch();
  mTouchSensorHandler = new TouchSensorHandler();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn TouchPubNode::on_activate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Activaing");
  publisher_->on_activate();
  isthreadrunning = true;
  touch_receiver_thread = std::thread(&cyberdog_touch::TouchPubNode::touch_receiver_callback, this);
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn TouchPubNode::on_deactivate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Deactiving");
  publisher_->on_deactivate();
  isthreadrunning = false;
  touch_receiver_thread.join();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn TouchPubNode::on_cleanup(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn TouchPubNode::on_shutdown(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  publisher_.reset();
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void TouchPubNode::touch_receiver_callback()
{
  int ret = -1;
  int ret_count, count = 1;
  struct input_event * touch_event;
  if (mTouchSensorHandler == nullptr) {
    RCLCPP_INFO(
      get_logger(), "\n @@@touch sensor handler create failed!");
    return;
  }

  touch_event =
    (struct input_event *)malloc(sizeof(struct input_event) * count);
  while (isthreadrunning) {
    ret_count = mTouchSensorHandler->pollTouchEvents(touch_event, count);
    if (ret_count > 0) {
      if ((touch_event->type == EV_KEY)) {
        ret = touch_event->code - GESTURE_XM_ADDR;
        message.touch_state = ret;
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        message.timestamp = ts.tv_sec * 1000000000 + ts.tv_nsec;
        RCLCPP_INFO(
          get_logger(), "\n @@@touch sensor data received: 0x%x", message.touch_state);
        publisher_->publish(std::move(message));
      }
    }
  }
  free(touch_event);
}
}  // namespace cyberdog_touch

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyberdog_touch::TouchPubNode>();
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();
  return 0;
}
