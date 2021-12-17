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

#ifndef PROTOCOL__CAN__SOCKET_CAN_RECEIVER_HPP_
#define PROTOCOL__CAN__SOCKET_CAN_RECEIVER_HPP_

#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <unistd.h>  // for close()

#include <chrono>
#include <cstring>
#include <string>
#include <memory>

#include "socket_can_id.hpp"
#include "socket_can_common.hpp"
#include "visibility_control.hpp"

namespace cyberdog
{
namespace common
{
/// Simple RAII wrapper around a raw CAN receiver
class SOCKETCAN_PUBLIC SocketCanReceiver
{
public:
  /// Constructor
  explicit SocketCanReceiver(const std::string & interface = "can0")
  : m_file_descriptor{bind_can_socket(interface)} {}
  /// Destructor
  ~SocketCanReceiver() noexcept
  {
    // Can't do anything on error; in fact generally shouldn't on close() error
    (void)close(m_file_descriptor);
  }

  bool receive(
    std::shared_ptr<struct can_frame> rx_frame,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero())
  {
    wait(timeout);
    // Read
    struct canfd_frame frame;
    const auto nbytes = read(m_file_descriptor, &frame, sizeof(frame));
    // Checks
    if (nbytes < 0) {
      throw std::runtime_error{strerror(errno)};
    }
    if (static_cast<std::size_t>(nbytes) < CAN_MTU) {
      throw std::runtime_error{"read: incomplete CAN frame"};
    }
    if (static_cast<std::size_t>(nbytes) != CAN_MTU) {
      throw std::logic_error{"Message was wrong size"};
    }
    // Write
    auto receive_id = CanId{frame.can_id, frame.len};
    if (receive_id.frame_type() == cyberdog::common::FrameType::DATA) {
      rx_frame->can_id = receive_id.standard().get();
      rx_frame->can_dlc = receive_id.length();
      std::memcpy(rx_frame->data, frame.data, sizeof(rx_frame->data));
      return true;
    }
    return false;
  }
  bool receive(
    std::shared_ptr<struct canfd_frame> rx_frame,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero())
  {
    wait(timeout);
    // Read
    struct canfd_frame frame;
    const auto nbytes = read(m_file_descriptor, &frame, sizeof(frame));
    // Checks
    uint8_t except_len = CAN_MTU - 8 + frame.len;
    if (nbytes < 0) {
      throw std::runtime_error{strerror(errno)};
    }
    if (static_cast<std::size_t>(nbytes) < except_len) {
      throw std::runtime_error{"read: incomplete CAN frame"};
    }
    // Write
    auto receive_id = CanId{frame.can_id, frame.len};
    if (receive_id.frame_type() == cyberdog::common::FrameType::DATA) {
      rx_frame->can_id = receive_id.standard().get();
      rx_frame->len = receive_id.length();
      std::memcpy(rx_frame->data, frame.data, sizeof(rx_frame->data));
      return true;
    }
    return false;
  }

  void enable_canfd(bool enable = true)
  {
    int canfd_on = enable ? 1 : 0;
    if (m_first_init == false) {
      m_canfd_state = canfd_on;
      m_first_init = true;
    } else if (m_canfd_state != canfd_on) {
      throw std::logic_error{"Can't mix can and canfd protocol by logic"};
    }
    setsockopt(m_file_descriptor, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &canfd_on, sizeof(canfd_on));
  }

  void set_filter(const struct can_filter filter[], size_t s)
  {
    setsockopt(m_file_descriptor, SOL_CAN_RAW, CAN_RAW_FILTER, filter, s);
  }

private:
  // Wait for file descriptor to be available to send data via select()
  SOCKETCAN_LOCAL void wait(const std::chrono::nanoseconds timeout) const
  {
    if (decltype(timeout)::zero() < timeout) {
      auto c_timeout = to_timeval(timeout);
      auto read_set = single_set(m_file_descriptor);
      // Wait
      if (0 == select(m_file_descriptor + 1, &read_set, NULL, NULL, &c_timeout)) {
        throw SocketCanTimeout{"$CAN Receive Timeout"};
      }
      //lint --e{9130, 1924, 9123, 9125, 1924, 9126} NOLINT
      if (!FD_ISSET(m_file_descriptor, &read_set)) {
        throw SocketCanTimeout{"$CAN Receive timeout"};
      }
    } else {
      auto read_set = single_set(m_file_descriptor);
      // Wait
      if (0 == select(m_file_descriptor + 1, &read_set, NULL, NULL, NULL)) {
        throw SocketCanTimeout{"$CAN Receive Timeout"};
      }
      //lint --e{9130, 1924, 9123, 9125, 1924, 9126} NOLINT
      if (!FD_ISSET(m_file_descriptor, &read_set)) {
        throw SocketCanTimeout{"$CAN Receive timeout"};
      }
    }
  }

  inline static bool m_first_init;
  inline static int8_t m_canfd_state;
  int32_t m_file_descriptor;
};  // class SocketCanReceiver

}  // namespace common
}  // namespace cyberdog

#endif  // PROTOCOL__CAN__SOCKET_CAN_RECEIVER_HPP_
