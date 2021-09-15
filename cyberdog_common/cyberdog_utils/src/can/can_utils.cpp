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

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "linux/can.h"
#include "linux/can/raw.h"
#include "net/if.h"
#include "sys/fcntl.h"
#include "sys/ioctl.h"
#include "sys/socket.h"
#include "sys/types.h"
#include "sys/unistd.h"

#include "cyberdog_utils/can/can_utils.hpp"
#include "rclcpp/logger.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cyberdog_utils
{

can_dev_operation::can_dev_operation()
{
  interface_ = "can0";

  try {
    receiver_ = std::make_unique<drivers::socketcan::SocketCanReceiver>(interface_);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(
      rclcpp::get_logger(interface_), "can0 receiver creat error!");
    return;
  }

  try {
    sender_ = std::make_unique<drivers::socketcan::SocketCanSender>(interface_);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(
      rclcpp::get_logger(interface_), "can0 sender creat error!");
    return;
  }
}

can_dev_operation::~can_dev_operation()
{
}

int can_dev_operation::wait_for_can_data()
{
  int ret = -1;
  drivers::socketcan::CanId receive_id{};

  try {
    receive_id = receiver_->receive(recv_frame.data, std::chrono::nanoseconds(-1));
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(
      rclcpp::get_logger(interface_),
      "Error receiving CAN message: %s - %s",
      interface_.c_str(), ex.what());
  }

  if (receive_id.frame_type() == drivers::socketcan::FrameType::DATA &&
    !receive_id.is_extended())
  {
    recv_frame.can_id = receive_id.get();
    recv_frame.can_dlc = receive_id.length();
    ret = 0;
  }

  return ret;
}

int can_dev_operation::send_can_message(struct can_frame cmd_frame)
{
  int ret = 0;
  drivers::socketcan::CanId send_id = drivers::socketcan::CanId(
    cmd_frame.can_id,
    drivers::socketcan::FrameType::DATA,
    drivers::socketcan::StandardFrame);
  cmd_frame.can_dlc = 8;

  try {
    sender_->send(cmd_frame.data, cmd_frame.can_dlc, send_id, std::chrono::nanoseconds(-1));
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(
      rclcpp::get_logger(interface_),
      "Error sending CAN message: %s - %s, errno: %d",
      interface_.c_str(), ex.what(), errno);
  }

  return ret;
}

}  // namespace cyberdog_utils
