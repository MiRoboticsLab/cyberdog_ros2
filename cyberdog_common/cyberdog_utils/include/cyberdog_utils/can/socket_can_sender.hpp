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

#ifndef CYBERDOG_UTILS__CAN__SOCKET_CAN_SENDER_HPP_
#define CYBERDOG_UTILS__CAN__SOCKET_CAN_SENDER_HPP_

#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

#include <chrono>
#include <string>

#include "cyberdog_utils/can/visibility_control.hpp"
#include "cyberdog_utils/can/socket_can_id.hpp"

namespace drivers
{
namespace socketcan
{

/// Simple RAII wrapper around a raw CAN sender
class SOCKETCAN_PUBLIC SocketCanSender
{
public:
  /// Constructor
  explicit SocketCanSender(
    const std::string & interface = "can0",
    const CanId & default_id = CanId{});
  /// Destructor
  ~SocketCanSender() noexcept;

  /// Send raw data with the default id
  /// \param[in] data A pointer to the beginning of the data to send
  /// \param[in] timeout Maximum duration to wait for file descriptor to be free for write. Negative
  ///                    durations are treated the same as zero timeout
  /// \param[in] length The amount of data to send starting from the data pointer
  /// \throw std::domain_error If length is > 8
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error on other errors
  void send(
    const void * const data,
    const std::size_t length,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const;
  /// Send raw data with an explicit CAN id
  /// \param[in] data A pointer to the beginning of the data to send
  /// \param[in] timeout Maximum duration to wait for file descriptor to be free for write. Negative
  ///                    durations are treated the same as zero timeout
  /// \param[in] id The id field for the CAN frame
  /// \param[in] length The amount of data to send starting from the data pointer
  /// \throw std::domain_error If length is > 8
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error on other errors
  void send(
    const void * const data,
    const std::size_t length,
    const CanId id,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const;
  /// Send typed data with the default id
  /// \tparam Type of data to send, must be 8 bytes or smaller
  /// \param[in] data The data to send
  /// \param[in] timeout Maximum duration to wait for file descriptor to be free for write. Negative
  ///                    durations are treated the same as zero timeout
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error on other errors
  template<typename T, typename = std::enable_if_t<!std::is_pointer<T>::value>>
  void send(
    const T & data,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const
  {
    send(data, m_default_id, timeout);
  }

  /// Send typed data with an explicit CAN Id
  /// \tparam Type of data to send, must be 8 bytes or smaller
  /// \param[in] data The data to send
  /// \param[in] timeout Maximum duration to wait for file descriptor to be free for write. Negative
  ///                    durations are treated the same as zero timeout
  /// \param[in] id The id field for the CAN frame
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error on other errors
  template<typename T, typename = std::enable_if_t<!std::is_pointer<T>::value>>
  void send(
    const T & data,
    const CanId id,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const
  {
    static_assert(sizeof(data) <= MAX_DATA_LENGTH, "Data type too large for CAN");
    //lint -e586 I have to use reinterpret cast because I'm operating on bytes, see below NOLINT
    send_impl(reinterpret_cast<const char *>(&data), sizeof(data), id, timeout);
    // reinterpret_cast to byte, or (unsigned) char is well defined;
    // all pointers can implicitly convert to void *
  }

  /// Send raw data with an explicit CAN id via fd can
  /// \param[in] data A pointer to the beginning of the data to send
  /// \param[in] timeout Maximum duration to wait for file descriptor to be free for write. Negative
  ///                    durations are treated the same as zero timeout
  /// \param[in] id The id field for the CAN frame
  /// \param[in] length The amount of data to send starting from the data pointer
  /// \throw std::domain_error If length is > 64
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error on other errors
  void send_fd(
    const void * const data,
    const std::size_t length,
    const CanId id,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const;

  /// Get the default CAN id
  CanId default_id() const noexcept;

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

private:
  // Underlying implementation of sending, data is assumed to be of an appropriate length
  void send_impl(
    const void * const data,
    const std::size_t length,
    const CanId id,
    const std::chrono::nanoseconds timeout) const;
  // Underlying implementation of can_fd sending, data is assumed to be of an appropriate length
  void send_fd_impl(
    const void * const data,
    const std::size_t length,
    const CanId id,
    const std::chrono::nanoseconds timeout) const;
  // Wait for file descriptor to be available to send data via select()
  SOCKETCAN_LOCAL void wait(const std::chrono::nanoseconds timeout) const;

  inline static bool m_first_init;
  inline static int8_t m_canfd_state;
  int32_t m_file_descriptor{};
  CanId m_default_id;
};  // class SocketCanSender

}  // namespace socketcan
}  // namespace drivers

#endif  // CYBERDOG_UTILS__CAN__SOCKET_CAN_SENDER_HPP_
