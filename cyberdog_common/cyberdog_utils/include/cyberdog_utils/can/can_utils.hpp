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

#ifndef CYBERDOG_UTILS__CAN__CAN_UTILS_HPP_
#define CYBERDOG_UTILS__CAN__CAN_UTILS_HPP_

#include <ctype.h>
#include <errno.h>
#include <libgen.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <memory>
#include <string>
#include <functional>
#include <thread>
#include <vector>

#include "net/if.h"
#include "sys/ioctl.h"
#include "sys/select.h"
#include "sys/socket.h"
#include "sys/time.h"
#include "sys/types.h"
#include "sys/uio.h"
#include "sys/unistd.h"

#include "linux/can.h"
#include "linux/can/raw.h"
#include "linux/can/error.h"

#include "cyberdog_utils/can/can_proto.h"

#include "cyberdog_utils/can/socket_can_receiver.hpp"
#include "cyberdog_utils/can/socket_can_sender.hpp"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

namespace cyberdog_utils
{
using can_std_callback = std::function<void (std::shared_ptr<struct can_frame> recv_frame)>;
using can_fd_callback = std::function<void (std::shared_ptr<struct canfd_frame> recv_frame)>;

// can_rx_dev ////////////////////////////////////////////////////////////////////////////////////
class can_rx_dev
{
public:
  explicit can_rx_dev(
    const std::string & interface,
    const std::string & name,
    can_std_callback recv_callback,
    int64_t timeout = -1);
  explicit can_rx_dev(
    const std::string & interface,
    const std::string & name,
    can_fd_callback recv_callback,
    int64_t timeout = -1);
  ~can_rx_dev();

  bool is_ready() {return ready_;}
  bool is_timeout() {return is_timeout_;}
  void set_filter(const struct can_filter filter[], size_t s);

private:
  bool ready_;
  bool canfd_;
  bool is_timeout_;
  bool extended_frame_;
  bool isthreadrunning_;
  int64_t timeout_;
  std::string name_;
  std::string interface_;
  can_std_callback can_std_callback_;
  can_fd_callback can_fd_callback_;
  std::unique_ptr<std::thread> main_T_;
  std::shared_ptr<struct can_frame> rx_std_frame_;
  std::shared_ptr<struct canfd_frame> rx_fd_frame_;
  std::unique_ptr<drivers::socketcan::SocketCanReceiver> receiver_;

  void init(
    const std::string & interface,
    const std::string & name,
    bool canfd_on,
    can_std_callback std_c,
    can_fd_callback fd_c,
    int64_t timeout);
  bool wait_for_can_data();
  void main_recv_func();
};

// can_tx_dev ////////////////////////////////////////////////////////////////////////////////////
class can_tx_dev
{
public:
  explicit can_tx_dev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    bool canfd_on,
    int64_t timeout = -1);
  ~can_tx_dev();

  bool send_can_message(struct can_frame & tx_frame);
  bool send_can_message(struct canfd_frame & tx_frame);
  bool is_ready() {return ready_;}
  bool is_timeout() {return is_timeout_;}

private:
  bool ready_;
  bool canfd_on_;
  bool is_timeout_;
  bool extended_frame_;
  int64_t timeout_;
  std::string name_;
  std::string interface_;
  std::unique_ptr<drivers::socketcan::SocketCanSender> sender_;

  bool send_can_message(struct can_frame * std_frame, struct canfd_frame * fd_frame);
};

// can_dev ///////////////////////////////////////////////////////////////////////////////////////
class can_dev
{
public:
  explicit can_dev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    can_std_callback recv_callback,
    int64_t timeout = -1);
  explicit can_dev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    can_fd_callback recv_callback,
    int64_t timeout = -1);
  explicit can_dev(
    const std::string & interface,
    const std::string & name,
    bool extended_frame,
    bool canfd_on,
    int64_t timeout = -1);
  ~can_dev();

  bool send_can_message(struct can_frame & tx_frame);
  bool send_can_message(struct canfd_frame & tx_frame);
  void set_filter(const struct can_filter filter[], size_t s);
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
  std::unique_ptr<can_rx_dev> rx_op_;
  std::unique_ptr<can_tx_dev> tx_op_;
};

}  // namespace cyberdog_utils

#endif  // CYBERDOG_UTILS__CAN__CAN_UTILS_HPP_
