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
#include <stdio.h>

#include <condition_variable>
#include <deque>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include "cyberdog_utils/lifecycle_node.hpp"
#include "cyberdog_utils/can/can_utils.hpp"
#include "cyberdog_utils/sensor/sensor_utils.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
#include "interaction_msgs/msg/led.hpp"
#include "std_msgs/msg/string.hpp"

#include "rclcpp/rclcpp.hpp"

#define TIMEOUT_MAX 40000000000

#ifndef UNUSED_VAR
#define UNUSED_VAR(var)  ((void)(var));
#endif

typedef struct
{
  uint64_t timeout;
  uint64_t finishTimestamp;
  uint64_t arriveTimestamp;
  uint8_t priority;
  uint8_t clientId;
  uint8_t ledId;
} led_command;

namespace cyberdog_led
{
class LedDetNode : public cyberdog_utils::LifecycleNode
{
public:
  LedDetNode();
  ~LedDetNode();

protected:
  /* Lifecycle stages */
  cyberdog_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  cyberdog_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  bool check_command_available(
    const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> request);
  void handle_service(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> request,
    const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Response> response);
  // message struct define here
  bool isthreadrunning;
  cyberdog_utils::can_dev_operation * can_dev_op = nullptr;
  /* Service */
  rclcpp::Service<ception_msgs::srv::SensorDetectionNode>::SharedPtr led_cmd_server_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  static void start_led_thread(LedDetNode * ledNode);
  void refresh_led_status(void);
  void update_led_command(uint8_t ledId);
  uint64_t get_current_nv_time();

  std::deque<led_command> alarm_q;
  std::deque<led_command> function_q;
  std::deque<led_command> effect_q;

  led_command temp_command;

  bool mIsEnabled = false;
  bool mIsNewRequest = false;
  std::atomic_bool mStopThread;
  std::condition_variable mWaitCV;
  std::mutex mRunMutex;
  std::thread mRunThread;
};
}  // namespace cyberdog_led

namespace cyberdog_led
{

LedDetNode::LedDetNode()
: cyberdog_utils::LifecycleNode("LedDetNode")
{
  RCLCPP_INFO(get_logger(), "Creating LedDetNode.");
}

LedDetNode::~LedDetNode()
{
  RCLCPP_INFO(get_logger(), "Destroying LedDetNode");
}

cyberdog_utils::CallbackReturn LedDetNode::on_configure(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Configuring");
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  led_cmd_server_ = this->create_service<ception_msgs::srv::SensorDetectionNode>(
    "cyberdog_led",
    std::bind(
      &LedDetNode::handle_service, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3),
    rmw_qos_profile_default, callback_group_);

  can_dev_op = new cyberdog_utils::can_dev_operation();
  mStopThread = false;
  mRunThread = std::thread(start_led_thread, this);
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LedDetNode::on_activate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Activaing");
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LedDetNode::on_deactivate(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  RCLCPP_INFO(get_logger(), "Deactiving");
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LedDetNode::on_cleanup(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

cyberdog_utils::CallbackReturn LedDetNode::on_shutdown(const rclcpp_lifecycle::State & state)
{
  UNUSED_VAR(state);
  return cyberdog_utils::CallbackReturn::SUCCESS;
}

void LedDetNode::update_led_command(uint8_t ledId)
{
  int ret = 0;
  can_frame headled_cmd;
  headled_cmd.can_id = 0x740;
  memset(headled_cmd.data, 0, sizeof(headled_cmd.data));

  can_frame rearled_cmd;
  rearled_cmd.can_id = 0x790;
  memset(rearled_cmd.data, 0, sizeof(rearled_cmd.data));

  switch (ledId) {
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_OFF:
      rearled_cmd.data[0] = 0x0;  // rearled_disable_cmd:{'7', '9', '0', '#', '0', '0'}
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_ON:
      rearled_cmd.data[0] = 0x1;  // rearled_red_on_cmd:{'7', '9', '0', '#', '0', '1'};
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_BREATH:
      rearled_cmd.data[0] = 0x2;  // rearled_red_breath_cmd:{'7', '9', '0', '#', '0', '2'}
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_BLINK:
      rearled_cmd.data[0] = 0x3;  // rearled_red_blink_cmd:{'7', '9', '0', '#', '0', '3'}
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_GREEN_ON:
      rearled_cmd.data[0] = 0x1;  // rearled_red_blink_cmd:{'7', '9', '0', '#', '0', '1'}
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_GREEN_BREATH:
      rearled_cmd.data[0] = 0x2;  // rearled_green_breath_cmd:{'7', '9', '0', '#', '0', '2'}
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_GREEN_BLINK:
      rearled_cmd.data[0] = 0x3;  // rearled_green_blink_cmd:{'7', '9', '0', '#', '0', '3'}
      ret = can_dev_op->send_can_message(rearled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_OFF:
      headled_cmd.data[0] = 0x0;  // headled_disable_cmd: {'7', '4', '0', '#', '0', '0'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_POWER_ON:
      headled_cmd.data[0] = 0x1;  // headled_power_on_cmd: {'7', '4', '0', '#', '0', '1'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_POWER_OFF:
      headled_cmd.data[0] = 0x2;  // headled_power_off_cmd: {'7', '4', '0', '#', '0', '2'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_ON:
      headled_cmd.data[0] = 0x3;  // headled_darkblue_on_cmd: {'7', '4', '0', '#', '0', '3'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_SKYBLUE_ON:
      headled_cmd.data[0] = 0x4;  // headled_skyblue_on_cmd: {'7', '4', '0', '#', '0', '4'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_ORANGE_ON:
      headled_cmd.data[0] = 0x5;  // headled_orange_on_cmd: {'7', '4', '0', '#', '0', '5'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_RED_ON:
      headled_cmd.data[0] = 0x6;  // headled_red_on_cmd: {'7', '4', '0', '#', '0', '6'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_BREATH:
      headled_cmd.data[0] = 0x7;  // headled_darkblue_breath_cmd: {'7', '4', '0', '#', '0', '7'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_SKYBLUE_BREATH:
      headled_cmd.data[0] = 0x8;  // headled_skyblue_breath_cmd: {'7', '4', '0', '#', '0', '8'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_BLINK:
      headled_cmd.data[0] = 0x9;  // headled_darkblue_blink_cmd: {'7', '4', '0', '#', '0', '9'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_ORANGE_BLINK:
      headled_cmd.data[0] = 0xA;  // headled_orange_blink_cmd: {'7', '4', '0', '#', '0', 'A'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    case ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_RED_BLINK:
      headled_cmd.data[0] = 0xA;  // headled_red_blink_cmd: {'7', '4', '0', '#', '0', 'B'}
      ret = can_dev_op->send_can_message(headled_cmd);
      break;
    default:
      break;
  }

  UNUSED_VAR(ret);
}

uint64_t LedDetNode::get_current_nv_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

void LedDetNode::refresh_led_status()
{
  std::unique_lock<std::mutex> runLock(mRunMutex);

  while (!mStopThread) {
    if (!mIsEnabled) {
      RCLCPP_INFO(
        get_logger(),
        "led-thread: waiting for notify in disable!");
      mWaitCV.wait(
        runLock, [&] {
          return mIsEnabled || mStopThread;
        });
    } else {
      RCLCPP_INFO(
        get_logger(),
        "led-thread: start process led cmd!");

      led_command current_command;
      if (alarm_q.size() != 0) {
        current_command = alarm_q.front();
      } else if (function_q.size() != 0) {
        current_command = function_q.front();
      } else if (effect_q.size() != 0) {
        current_command = effect_q.front();
      } else {
        // there is no current command to respond updata to default directly
        RCLCPP_INFO(
          get_logger(),
          "led-thread: no command, will update default led!");
        mIsEnabled = false;
        if (!mIsNewRequest) {
          // no command in timeout condition
          RCLCPP_INFO(
            get_logger(),
            "led-thread: no more command to process,exit!");
          update_led_command(ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_DARKBLUE_ON);
          update_led_command(ception_msgs::srv::SensorDetectionNode::Request::REAR_LED_RED_ON);
          continue;
        }
      }

      RCLCPP_INFO(
        get_logger(),
        "led-thread: current command = %u, clientid = %u, timeout = %ld, priority = %u",
        current_command.ledId, current_command.clientId, current_command.timeout,
        current_command.priority);

      update_led_command(current_command.ledId);

      RCLCPP_INFO(
        get_logger(),
        "led-thread: wait for timeout: %ld!", current_command.timeout);
      if (current_command.timeout !=
        ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
      {
        mWaitCV.wait_for(runLock, std::chrono::nanoseconds(current_command.timeout));
      } else {
        mWaitCV.wait(runLock);
      }

      // check if any command is timed out
      uint64_t current_time = get_current_nv_time();
      std::deque<led_command>::iterator iter;
      for (iter = alarm_q.begin(); iter != alarm_q.end(); ) {
        if (iter->finishTimestamp <= current_time) {
          RCLCPP_INFO(
            get_logger(),
            "led-thread: alarm_q item erasing, info(%u, %u, %u, %ld)!",
            iter->clientId, iter->ledId, iter->priority, iter->timeout);
          iter = alarm_q.erase(iter);
        } else {
          iter++;
        }
      }

      for (iter = function_q.begin(); iter != function_q.end(); ) {
        if (iter->finishTimestamp <= current_time) {
          RCLCPP_INFO(
            get_logger(),
            "led-thread: function_q item erasing, info(%u, %u, %u, %ld)!",
            iter->clientId, iter->ledId, iter->priority, iter->timeout);
          iter = function_q.erase(iter);
        } else {
          iter++;
        }
      }

      for (iter = effect_q.begin(); iter != effect_q.end(); ) {
        if (iter->finishTimestamp <= current_time) {
          RCLCPP_INFO(
            get_logger(),
            "led-thread: effect_q item erasing, info(%u, %u, %u, %ld)!",
            iter->clientId, iter->ledId, iter->priority, iter->timeout);
          iter = effect_q.erase(iter);
        } else {
          iter++;
        }
      }

      // update timeout value of every Q
      for (uint32_t i = 0; i < alarm_q.size(); i++) {
        if (alarm_q.at(i).timeout !=
          ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
        {
          alarm_q.at(i).timeout = alarm_q.at(i).finishTimestamp - current_time;
        }
        RCLCPP_INFO(
          get_logger(),
          "led-thread: timeout update,alarm_q(%d), timeout(%ld)!", i, alarm_q.at(i).timeout);
      }
      for (uint32_t i = 0; i < function_q.size(); i++) {
        if (function_q.at(i).timeout !=
          ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
        {
          function_q.at(i).timeout = function_q.at(i).finishTimestamp - current_time;
        }
        RCLCPP_INFO(
          get_logger(),
          "led-thread: timeout update,function_q(%d), timeout(%ld)!", i, function_q.at(i).timeout);
      }
      for (uint32_t i = 0; i < effect_q.size(); i++) {
        if (effect_q.at(i).timeout !=
          ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
        {
          effect_q.at(i).timeout = effect_q.at(i).finishTimestamp - current_time;
        }
        RCLCPP_INFO(
          get_logger(),
          "led-thread: timeout update,effect_q(%d), timeout(%ld)!", i, effect_q.at(i).timeout);
      }
    }
  }
}

void LedDetNode::start_led_thread(LedDetNode * ledNode)
{
  ledNode->refresh_led_status();
}

bool LedDetNode::check_command_available(
  const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> request)
{
  if (request->clientid < 1 || request->clientid > 3) {return false;}
  if (request->priority < 1 || request->priority > 3) {return false;}
  return true;
}

void LedDetNode::handle_service(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Request> request,
  const std::shared_ptr<ception_msgs::srv::SensorDetectionNode::Response> response)
{
  (void)request_header;
  RCLCPP_INFO(
    get_logger(),
    "led-request: command = %u, clientid = %u, timeout = %ld, priority = %u",
    request->command, request->clientid, request->timeout, request->priority);

  response->success = true;
  // check command avaiable
  if (!check_command_available(request)) {
    RCLCPP_INFO(
      get_logger(),
      "led-request: not available!");

    response->success = false;
    return;
  }

  // fill the struct and decice which queue to insert
  std::unique_lock<std::mutex> runLock(mRunMutex);
  // notify led refresh thread if this is a head led command
  if (request->command >= ception_msgs::srv::SensorDetectionNode::Request::HEAD_LED_OFF) {
    temp_command.arriveTimestamp = get_current_nv_time();
    temp_command.clientId = request->clientid;
    temp_command.ledId = request->command;
    temp_command.priority = request->priority;
    temp_command.timeout = request->timeout;
    if (request->timeout == ception_msgs::srv::SensorDetectionNode::Request::COMMAND_OFF) {
      temp_command.finishTimestamp = temp_command.arriveTimestamp;
    } else if (request->timeout ==  // NOLINT
      ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON)
    {
      temp_command.finishTimestamp = request->timeout;
    } else {
      if (request->timeout > TIMEOUT_MAX) {
        temp_command.finishTimestamp = TIMEOUT_MAX + temp_command.arriveTimestamp;
      }
      temp_command.finishTimestamp = request->timeout + temp_command.arriveTimestamp;
    }
    mIsEnabled = true;

    // insert current command to corrsponding Q
    if (temp_command.timeout == ception_msgs::srv::SensorDetectionNode::Request::COMMAND_OFF) {
      // find if any ALWAYSON Req is set with the same client and same ledid
      std::deque<led_command>::iterator iter;
      switch (temp_command.priority) {
        case ception_msgs::srv::SensorDetectionNode::Request::TYPE_EFFECTS:
          for (iter = effect_q.begin(); iter != effect_q.end(); ) {
            if (iter->timeout ==
              ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON &&
              iter->clientId == temp_command.clientId &&
              iter->ledId == temp_command.ledId &&
              iter->priority == temp_command.priority)
            {
              RCLCPP_INFO(
                get_logger(),
                "led-request: effect_q item erasing, info(%u, %u, %u, %ld)!",
                iter->clientId, iter->ledId, iter->priority, iter->timeout);
              iter = effect_q.erase(iter);
            } else {
              iter++;
            }
          }
          break;
        case ception_msgs::srv::SensorDetectionNode::Request::TYPE_FUNCTION:
          for (iter = function_q.begin(); iter != function_q.end(); ) {
            if (iter->timeout ==
              ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON &&
              iter->clientId == temp_command.clientId &&
              iter->ledId == temp_command.ledId &&
              iter->priority == temp_command.priority)
            {
              RCLCPP_INFO(
                get_logger(),
                "led-request: function_q item erasing, info(%u, %u, %u, %ld)!",
                iter->clientId, iter->ledId, iter->priority, iter->timeout);
              iter = function_q.erase(iter);
            } else {
              iter++;
            }
          }
          break;
        case ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM:
          for (iter = alarm_q.begin(); iter != alarm_q.end(); ) {
            if (iter->timeout ==
              ception_msgs::srv::SensorDetectionNode::Request::COMMAND_ALWAYSON &&
              iter->clientId == temp_command.clientId &&
              iter->ledId == temp_command.ledId &&
              iter->priority == temp_command.priority)
            {
              RCLCPP_INFO(
                get_logger(),
                "led-request: alarm_q item erasing, info(%u, %u, %u, %ld)!",
                iter->clientId, iter->ledId, iter->priority, iter->timeout);
              iter = alarm_q.erase(iter);
            } else {
              iter++;
            }
          }
          break;
        default:
          // do nothing
          break;
      }
    } else {
      switch (temp_command.priority) {
        case ception_msgs::srv::SensorDetectionNode::Request::TYPE_ALARM:
          alarm_q.push_front(temp_command);
          break;
        case ception_msgs::srv::SensorDetectionNode::Request::TYPE_FUNCTION:
          function_q.push_front(temp_command);
          break;
        case ception_msgs::srv::SensorDetectionNode::Request::TYPE_EFFECTS:
          effect_q.push_front(temp_command);
          break;
        default:
          break;
      }
    }
    mWaitCV.notify_all();
  } else {
    update_led_command(request->command);
  }

  RCLCPP_INFO(
    get_logger(),
    "led-request: notify done, finish time: %lu!", temp_command.finishTimestamp);
}
}  // namespace cyberdog_led

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cyberdog_led::LedDetNode>();
  rclcpp::executors::MultiThreadedExecutor exec_;
  exec_.add_node(node->get_node_base_interface());
  exec_.spin();
  rclcpp::shutdown();
  return 0;
}
