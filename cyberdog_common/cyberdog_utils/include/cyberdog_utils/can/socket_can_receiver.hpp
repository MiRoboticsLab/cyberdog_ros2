// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
// Copyright 2021 the Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Co-developed by Tier IV, Inc. and Apex.AI, Inc.
/// \file
/// \brief This file defines a class a socket sender

#ifndef CYBERDOG_UTILS__CAN__SOCKET_CAN_RECEIVER_HPP_
#define CYBERDOG_UTILS__CAN__SOCKET_CAN_RECEIVER_HPP_

#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

#include <array>
#include <chrono>
#include <cstring>
#include <string>
#include <memory>

#include "cyberdog_utils/can/visibility_control.hpp"
#include "cyberdog_utils/can/socket_can_id.hpp"

namespace drivers
{
namespace socketcan
{

/// Simple RAII wrapper around a raw CAN receiver
class SOCKETCAN_PUBLIC SocketCanReceiver
{
public:
  /// Constructor
  explicit SocketCanReceiver(const std::string & interface = "can0");
  /// Destructor
  ~SocketCanReceiver() noexcept;

  bool receive(
    std::shared_ptr<struct can_frame> rx_frame,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());
  bool receive(
    std::shared_ptr<struct canfd_frame> rx_frame,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  void enable_canfd(bool enable = true)
  {
    int canfd_on = enable ? 1 : 0;
    if (m_first_init == false) {
      m_canfd_state = canfd_on;
      m_first_init = true;
    } else if (m_canfd_state != canfd_on) {
      throw std::logic_error{"Can't mix can and canfd device by logic"};
    }
    setsockopt(m_file_descriptor, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_on, sizeof(canfd_on));
  }

  void set_filter(const struct can_filter filter[], size_t s)
  {
    setsockopt(m_file_descriptor, SOL_CAN_RAW, CAN_RAW_FILTER, filter, s);
  }

private:
  // Wait for file descriptor to be available to send data via select()
  SOCKETCAN_LOCAL void wait(const std::chrono::nanoseconds timeout) const;

  inline static bool m_first_init;
  inline static int8_t m_canfd_state;
  int32_t m_file_descriptor;
};  // class SocketCanReceiver

}  // namespace socketcan
}  // namespace drivers

#endif  // CYBERDOG_UTILS__CAN__SOCKET_CAN_RECEIVER_HPP_
