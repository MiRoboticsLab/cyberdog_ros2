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
#include <algorithm>

#include "linux/can.h"
#include "linux/can/raw.h"
#include "net/if.h"
#include "sys/fcntl.h"
#include "sys/ioctl.h"
#include "sys/socket.h"
#include "sys/types.h"
#include "sys/unistd.h"

#include "cyberdog_utils/can/can_utils.hpp"

#define C_END "\033[m"
#define C_RED "\033[0;32;31m"
#define C_YELLOW "\033[1;33m"

namespace cyberdog_utils
{
// can_rx_dev ////////////////////////////////////////////////////////////////////////////////////
can_rx_dev::can_rx_dev(
  const std::string & interface,
  const std::string & name,
  can_std_callback rx_callback,
  int64_t timeout)
{
  init(interface, name, false, rx_callback, nullptr, timeout);
}

can_rx_dev::can_rx_dev(
  const std::string & interface,
  const std::string & name,
  can_fd_callback rx_callback,
  int64_t timeout)
{
  init(interface, name, true, nullptr, rx_callback, timeout);
}

can_rx_dev::~can_rx_dev()
{
  ready_ = false;
  isthreadrunning_ = false;
  if (main_T_ != nullptr) {main_T_->join();}
  main_T_ = nullptr;
  receiver_ = nullptr;
}

void can_rx_dev::set_filter(const struct can_filter filter[], size_t s)
{
  if (receiver_ != nullptr) {receiver_->set_filter(filter, s);}
}

void can_rx_dev::init(
  const std::string & interface,
  const std::string & name,
  bool canfd_on,
  can_std_callback std_callback,
  can_fd_callback fd_callback,
  int64_t timeout)
{
  name_ = name;
  ready_ = false;
  canfd_ = canfd_on;
  timeout_ = timeout;
  is_timeout_ = false;
  interface_ = interface;
  can_std_callback_ = std_callback;
  can_fd_callback_ = fd_callback;
  try {
    receiver_ = std::make_unique<drivers::socketcan::SocketCanReceiver>(interface_);
    receiver_->enable_canfd(canfd_);
    isthreadrunning_ = true;
    main_T_ = std::make_unique<std::thread>(std::bind(&can_rx_dev::main_recv_func, this));
  } catch (const std::exception & ex) {
    printf(
      C_RED "[CAN_RX][ERROR][%s] %s receiver creat error! %s\n" C_END,
      name_.c_str(), interface_.c_str(), ex.what());
    return;
  }
  ready_ = true;
}

bool can_rx_dev::wait_for_can_data()
{
  drivers::socketcan::CanId receive_id{};
  std::string can_type;
  if (canfd_) {
    can_type = "FD";
    rx_fd_frame_ = std::make_shared<struct canfd_frame>();
  } else {
    can_type = "STD";
    rx_std_frame_ = std::make_shared<struct can_frame>();
  }

  if (receiver_ != nullptr) {
    try {
      bool result = canfd_ ? receiver_->receive(rx_fd_frame_, std::chrono::nanoseconds(timeout_)) :
        receiver_->receive(rx_std_frame_, std::chrono::nanoseconds(timeout_));
      if (result) {is_timeout_ = false;}
      return result;
    } catch (const std::exception & ex) {
      if (ex.what()[0] == '$') {
        is_timeout_ = true;
        return false;
      }
      printf(
        C_RED "[CAN_RX %s][ERROR][%s] Error receiving CAN %s message: %s - %s\n" C_END,
        can_type.c_str(), name_.c_str(), can_type.c_str(), interface_.c_str(), ex.what());
    }
  } else {
    isthreadrunning_ = false;
    printf(
      C_RED "[CAN_RX %s][ERROR][%s] Error receiving CAN %s message: %s, "
      "no receiver init\n" C_END,
      can_type.c_str(), name_.c_str(), can_type.c_str(), interface_.c_str());
  }
  return false;
}

void can_rx_dev::main_recv_func()
{
  printf("[CAN_RX][INFO][%s] Start recv thread: %s\n", interface_.c_str(), name_.c_str());
  while (isthreadrunning_ && ready_) {
    if (wait_for_can_data()) {
      if (!canfd_ && can_std_callback_ != nullptr) {
        can_std_callback_(rx_std_frame_);
      } else if (canfd_ && can_fd_callback_ != nullptr) {can_fd_callback_(rx_fd_frame_);}
    }
  }
  printf("[CAN_RX][INFO][%s] Exit recv thread: %s\n", interface_.c_str(), name_.c_str());
}


// can_tx_dev ////////////////////////////////////////////////////////////////////////////////////
can_tx_dev::can_tx_dev(
  const std::string & interface,
  const std::string & name,
  bool extended_frame,
  bool canfd_on,
  int64_t timeout)
{
  ready_ = false;
  name_ = name;
  timeout_ = timeout;
  is_timeout_ = false;
  canfd_on_ = canfd_on;
  interface_ = interface;
  extended_frame_ = extended_frame;
  try {
    sender_ = std::make_unique<drivers::socketcan::SocketCanSender>(interface_);
    sender_->enable_canfd(canfd_on);
  } catch (const std::exception & ex) {
    printf(
      C_RED "[CAN_TX][ERROR][%s] %s sender creat error! %s\n" C_END,
      name_.c_str(), interface_.c_str(), ex.what());
    return;
  }
  ready_ = true;
}

can_tx_dev::~can_tx_dev()
{
  ready_ = false;
  sender_ = nullptr;
}

bool can_tx_dev::send_can_message(struct can_frame & tx_frame)
{
  return send_can_message(&tx_frame, nullptr);
}

bool can_tx_dev::send_can_message(struct canfd_frame & tx_frame)
{
  return send_can_message(nullptr, &tx_frame);
}

bool can_tx_dev::send_can_message(struct can_frame * std_frame, struct canfd_frame * fd_frame)
{
  bool self_fd = (fd_frame != nullptr);
  std::string can_type = self_fd ? "FD" : "STD";
  if (canfd_on_ != self_fd) {
    printf(
      C_RED "[CAN_TX %s][ERROR][%s] Error sending CAN message: %s - "
      "Not support can/canfd frame mixed\n" C_END,
      can_type.c_str(), name_.c_str(), interface_.c_str());
    return false;
  }
  canid_t * canid = self_fd ? &fd_frame->can_id : &std_frame->can_id;

  bool result = true;
  drivers::socketcan::CanId send_id = extended_frame_ ?
    drivers::socketcan::CanId(
    *canid,
    drivers::socketcan::FrameType::DATA,
    drivers::socketcan::ExtendedFrame) :
    drivers::socketcan::CanId(
    *canid,
    drivers::socketcan::FrameType::DATA,
    drivers::socketcan::StandardFrame);

  if (sender_ != nullptr) {
    try {
      if (self_fd) {
        sender_->send_fd(
          fd_frame->data, (fd_frame->len == 0) ? 64 : fd_frame->len, send_id,
          std::chrono::nanoseconds(timeout_));
        is_timeout_ = false;
      } else {
        sender_->send(
          std_frame->data, (std_frame->can_dlc == 0) ? 8 : std_frame->can_dlc, send_id,
          std::chrono::nanoseconds(timeout_));
        is_timeout_ = false;
      }
    } catch (const std::exception & ex) {
      if (ex.what()[0] == '$') {
        is_timeout_ = true;
        return false;
      }
      result = false;
      printf(
        C_RED "[CAN_TX %s][ERROR][%s] Error sending CAN message: %s - %s\n" C_END,
        can_type.c_str(), name_.c_str(), interface_.c_str(), ex.what());
    }
  } else {
    result = false;
    printf(
      C_RED "[CAN_TX %s][ERROR][%s] Error sending CAN message: %s - No device\n" C_END,
      can_type.c_str(), name_.c_str(), interface_.c_str());
  }
  return result;
}

// can_dev ///////////////////////////////////////////////////////////////////////////////////////
can_dev::can_dev(
  const std::string & interface,
  const std::string & name,
  bool extended_frame,
  can_std_callback rx_callback,
  int64_t timeout)
{
  name_ = name;
  send_only_ = false;
  tx_op_ = std::make_unique<can_tx_dev>(interface, name_, extended_frame, false, timeout);
  rx_op_ = std::make_unique<can_rx_dev>(interface, name_, rx_callback, timeout);
}

can_dev::can_dev(
  const std::string & interface,
  const std::string & name,
  bool extended_frame,
  can_fd_callback rx_callback,
  int64_t timeout)
{
  name_ = name;
  send_only_ = false;
  tx_op_ = std::make_unique<can_tx_dev>(interface, name_, extended_frame, true, timeout);
  rx_op_ = std::make_unique<can_rx_dev>(interface, name_, rx_callback, timeout);
}

can_dev::can_dev(
  const std::string & interface,
  const std::string & name,
  bool extended_frame,
  bool canfd_on,
  int64_t timeout)
{
  name_ = name;
  send_only_ = true;
  tx_op_ = std::make_unique<can_tx_dev>(interface, name_, extended_frame, canfd_on, timeout);
}

can_dev::~can_dev()
{
  rx_op_ = nullptr;
  tx_op_ = nullptr;
}

bool can_dev::send_can_message(struct can_frame & tx_frame)
{
  if (tx_op_ != nullptr) {return tx_op_->send_can_message(tx_frame);}
  return false;
}

bool can_dev::send_can_message(struct canfd_frame & tx_frame)
{
  if (tx_op_ != nullptr) {return tx_op_->send_can_message(tx_frame);}
  return false;
}

void can_dev::set_filter(const struct can_filter filter[], size_t s)
{
  if (rx_op_ != nullptr) {rx_op_->set_filter(filter, s);}
}

}  // namespace cyberdog_utils
