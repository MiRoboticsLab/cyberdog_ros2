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

#ifndef PROTOCOL__CAN__CAN_UTILS_HPP_
#define PROTOCOL__CAN__CAN_UTILS_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <functional>

#include "linux/can.h"
#include "linux/can/raw.h"
#include "linux/can/error.h"

#include "socket_can_receiver.hpp"
#include "socket_can_sender.hpp"

#define C_END "\033[m"
#define C_RED "\033[0;32;31m"
#define C_YELLOW "\033[1;33m"

namespace cyberdog
{
namespace common
{
using can_std_callback = std::function<void (std::shared_ptr<struct can_frame> recv_frame)>;
using can_fd_callback = std::function<void (std::shared_ptr<struct canfd_frame> recv_frame)>;

// CanRxDev //////////////////////////////////////////////////////////////////////////////////////
class CanRxDev
{
public:
  explicit CanRxDev(
    const std::string & interface,
    const std::string & name,
    can_std_callback recv_callback,
    int64_t nano_timeout = -1)
  {
    init(interface, name, false, recv_callback, nullptr, nano_timeout);
  }
  explicit CanRxDev(
    const std::string & interface,
    const std::string & name,
    can_fd_callback recv_callback,
    int64_t nano_timeout = -1)
  {
    init(interface, name, true, nullptr, recv_callback, nano_timeout);
  }
  ~CanRxDev()
  {
    ready_ = false;
    isthreadrunning_ = false;
    if (main_T_ != nullptr) {main_T_->join();}
    main_T_ = nullptr;
    receiver_ = nullptr;
  }

  bool is_ready() {return ready_;}
  bool is_timeout() {return is_timeout_;}
  void set_filter(const struct can_filter filter[], size_t s)
  {
    if (receiver_ != nullptr) {receiver_->set_filter(filter, s);}
  }

#ifdef COMMON_PROTOCOL_TEST
  bool testing_setcandata(can_frame frame)
  {
    if (can_std_callback_ != nullptr) {
      can_std_callback_(std::make_shared<struct can_frame>(frame));
      return true;
    }
    return false;
  }
  bool testing_setcandata(canfd_frame frame)
  {
    if (can_fd_callback_ != nullptr) {
      can_fd_callback_(std::make_shared<struct canfd_frame>(frame));
      return true;
    }
    return false;
  }
#endif

private:
  bool ready_;
  bool canfd_;
  bool is_timeout_;
  bool extended_frame_;
  bool isthreadrunning_;
  int64_t nano_timeout_;
  std::string name_;
  std::string interface_;
  can_std_callback can_std_callback_;
  can_fd_callback can_fd_callback_;
  std::unique_ptr<std::thread> main_T_;
  std::shared_ptr<struct can_frame> rx_std_frame_;
  std::shared_ptr<struct canfd_frame> rx_fd_frame_;
  std::unique_ptr<cyberdog::common::SocketCanReceiver> receiver_;

  void init(
    const std::string & interface,
    const std::string & name,
    bool canfd_on,
    can_std_callback std_callback,
    can_fd_callback fd_callback,
    int64_t nano_timeout)
  {
    name_ = name;
    ready_ = false;
    canfd_ = canfd_on;
    nano_timeout_ = nano_timeout;
    is_timeout_ = false;
    interface_ = interface;
    can_std_callback_ = std_callback;
    can_fd_callback_ = fd_callback;
    try {
      receiver_ = std::make_unique<cyberdog::common::SocketCanReceiver>(interface_);
      receiver_->enable_canfd(canfd_);
      isthreadrunning_ = true;
      main_T_ = std::make_unique<std::thread>(std::bind(&CanRxDev::main_recv_func, this));
    } catch (const std::exception & ex) {
      printf(
        C_RED "[CAN_RX][ERROR][%s] %s receiver creat error! %s\n" C_END,
        name_.c_str(), interface_.c_str(), ex.what());
      return;
    }
    ready_ = true;
  }
  bool wait_for_can_data()
  {
    cyberdog::common::CanId receive_id{};
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
        bool result =
          canfd_ ? receiver_->receive(rx_fd_frame_, std::chrono::nanoseconds(nano_timeout_)) :
          receiver_->receive(rx_std_frame_, std::chrono::nanoseconds(nano_timeout_));
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
  void main_recv_func()
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
};  // class CanRxDev

// CanTxDev //////////////////////////////////////////////////////////////////////////////////////
class CanTxDev
{
public:
  explicit CanTxDev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    bool canfd_on,
    int64_t nano_timeout = -1)
  {
    ready_ = false;
    name_ = name;
    nano_timeout_ = nano_timeout;
    is_timeout_ = false;
    canfd_on_ = canfd_on;
    interface_ = interface;
    extended_frame_ = extended_frame;
    try {
      sender_ = std::make_unique<cyberdog::common::SocketCanSender>(interface_);
      sender_->enable_canfd(canfd_on);
    } catch (const std::exception & ex) {
      printf(
        C_RED "[CAN_TX][ERROR][%s] %s sender creat error! %s\n" C_END,
        name_.c_str(), interface_.c_str(), ex.what());
      return;
    }
    ready_ = true;
  }
  ~CanTxDev()
  {
    ready_ = false;
    sender_ = nullptr;
  }

  bool send_can_message(struct can_frame & tx_frame)
  {
    return send_can_message(&tx_frame, nullptr);
  }
  bool send_can_message(struct canfd_frame & tx_frame)
  {
    return send_can_message(nullptr, &tx_frame);
  }
  bool is_ready() {return ready_;}
  bool is_timeout() {return is_timeout_;}

private:
  bool ready_;
  bool canfd_on_;
  bool is_timeout_;
  bool extended_frame_;
  int64_t nano_timeout_;
  std::string name_;
  std::string interface_;
  std::unique_ptr<cyberdog::common::SocketCanSender> sender_;

  bool send_can_message(struct can_frame * std_frame, struct canfd_frame * fd_frame)
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
    cyberdog::common::CanId send_id = extended_frame_ ?
      cyberdog::common::CanId(
      *canid,
      cyberdog::common::FrameType::DATA,
      cyberdog::common::ExtendedFrame) :
      cyberdog::common::CanId(
      *canid,
      cyberdog::common::FrameType::DATA,
      cyberdog::common::StandardFrame);

    if (sender_ != nullptr) {
      try {
        if (self_fd) {
          sender_->send_fd(
            fd_frame->data, (fd_frame->len == 0) ? 64 : fd_frame->len, send_id,
            std::chrono::nanoseconds(nano_timeout_));
          is_timeout_ = false;
        } else {
          sender_->send(
            std_frame->data, (std_frame->can_dlc == 0) ? 8 : std_frame->can_dlc, send_id,
            std::chrono::nanoseconds(nano_timeout_));
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
};  // class CanTxDev

// CanDev ////////////////////////////////////////////////////////////////////////////////////////
class CanDev
{
public:
  explicit CanDev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    can_std_callback recv_callback,
    int64_t nano_timeout = -1)
  {
    name_ = name;
    send_only_ = false;
    tx_op_ = std::make_unique<CanTxDev>(interface, name_, extended_frame, false, nano_timeout);
    rx_op_ = std::make_unique<CanRxDev>(interface, name_, recv_callback, nano_timeout);
  }
  explicit CanDev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    can_fd_callback recv_callback,
    int64_t nano_timeout = -1)
  {
    name_ = name;
    send_only_ = false;
    tx_op_ = std::make_unique<CanTxDev>(interface, name_, extended_frame, true, nano_timeout);
    rx_op_ = std::make_unique<CanRxDev>(interface, name_, recv_callback, nano_timeout);
  }
  explicit CanDev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    bool canfd_on,
    int64_t nano_timeout = -1)
  {
    name_ = name;
    send_only_ = true;
    tx_op_ = std::make_unique<CanTxDev>(interface, name_, extended_frame, canfd_on, nano_timeout);
  }
  ~CanDev()
  {
    rx_op_ = nullptr;
    tx_op_ = nullptr;
  }

  bool send_can_message(struct can_frame & tx_frame)
  {
#ifdef COMMON_PROTOCOL_TEST
    if (rx_op_ != nullptr) {return rx_op_->testing_setcandata(tx_frame);}
    return false;
#else
    if (tx_op_ != nullptr) {return tx_op_->send_can_message(tx_frame);}
    return false;
#endif
  }

  bool send_can_message(struct canfd_frame & tx_frame)
  {
#ifdef COMMON_PROTOCOL_TEST
    if (rx_op_ != nullptr) {return rx_op_->testing_setcandata(tx_frame);}
    return false;
#else
    if (tx_op_ != nullptr) {return tx_op_->send_can_message(tx_frame);}
    return false;
#endif
  }
  void set_filter(const struct can_filter filter[], size_t s)
  {
    if (rx_op_ != nullptr) {rx_op_->set_filter(filter, s);}
  }
  bool is_send_only() {return send_only_;}
  bool is_ready()
  {
    bool result = (tx_op_ == nullptr) ? false : tx_op_->is_ready();
    if (send_only_) {
      return result;
    } else {
      return result && ((rx_op_ == nullptr) ? false : rx_op_->is_ready());
    }
  }
  bool is_rx_timeout() {return rx_op_->is_timeout();}
  bool is_tx_timeout() {return tx_op_->is_timeout();}

private:
  bool send_only_;
  std::string name_;
  std::unique_ptr<CanRxDev> rx_op_;
  std::unique_ptr<CanTxDev> tx_op_;
};  // class CanDev

}  // namespace common
}  // namespace cyberdog

#endif  // PROTOCOL__CAN__CAN_UTILS_HPP_
