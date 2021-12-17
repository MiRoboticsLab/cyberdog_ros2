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

#ifndef PROTOCOL__CAN__SOCKET_CAN_SENDER_HPP_
#define PROTOCOL__CAN__SOCKET_CAN_SENDER_HPP_

#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <unistd.h>  // for close()

#include <chrono>
#include <string>
#include <stdexcept>

#include "socket_can_id.hpp"
#include "socket_can_common.hpp"
#include "visibility_control.hpp"


namespace cyberdog
{
namespace common
{

/// Simple RAII wrapper around a raw CAN sender
class SOCKETCAN_PUBLIC SocketCanSender
{
public:
  /// Constructor
  explicit SocketCanSender(
    const std::string & interface = "can0",
    const CanId & default_id = CanId{})
  : m_file_descriptor{bind_can_socket(interface)},
    m_default_id{default_id} {}
  /// Destructor
  ~SocketCanSender() noexcept
  {
    (void)close(m_file_descriptor);
    // I'm destructing--there's not much else I can do on an error
  }

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
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const
  {
    send(data, length, m_default_id, timeout);
  }
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
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const
  {
    if (length > MAX_DATA_LENGTH) {
      throw std::domain_error{"Size is too large to send via CAN"};
    }
    send_impl(data, length, id, timeout);
  }
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
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const
  {
    if (length > MAX_DATA_LENGTH) {
      throw std::domain_error{"Size is too large to send via CAN"};
    }
    send_fd_impl(data, length, id, timeout);
  }

  /// Get the default CAN id
  CanId default_id() const noexcept
  {
    return m_default_id;
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

private:
  // Underlying implementation of sending, data is assumed to be of an appropriate length
  void send_impl(
    const void * const data,
    const std::size_t length,
    const CanId id,
    const std::chrono::nanoseconds timeout) const
  {
    // Use select call on positive timeout
    wait(timeout);
    // Actually send the data
    struct can_frame data_frame;
    data_frame.can_id = id.get();
    // User facing functions do check
    data_frame.can_dlc = static_cast<decltype(data_frame.can_dlc)>(length);
    // lint -e{586} NOLINT data_frame is a stack variable; guaranteed not to overlap
    (void)std::memcpy(static_cast<void *>(&data_frame.data[0U]), data, length);
    if (write(m_file_descriptor, &data_frame, static_cast<int>(CAN_MTU)) !=
      static_cast<int>(CAN_MTU))
    {
      throw std::runtime_error{strerror(errno)};
    }
  }
  // Underlying implementation of can_fd sending, data is assumed to be of an appropriate length
  void send_fd_impl(
    const void * const data,
    const std::size_t length,
    const CanId id,
    const std::chrono::nanoseconds timeout) const
  {
    // Use select call on positive timeout
    wait(timeout);
    // Actually send the data
    struct canfd_frame data_frame;
    data_frame.can_id = id.get();
    // User facing functions do check
    data_frame.len = static_cast<decltype(data_frame.len)>(length);
    // lint -e{586} NOLINT data_frame is a stack variable; guaranteed not to overlap
    std::memcpy(static_cast<void *>(&data_frame.data[0U]), data, length);
    uint8_t real_len = CAN_MTU - 8 + data_frame.len;
    auto bytes_sent = write(m_file_descriptor, &data_frame, static_cast<int>(real_len));
    if (bytes_sent != static_cast<int>(real_len)) {
      throw std::runtime_error{strerror(errno)};
    }
  }
  // Wait for file descriptor to be available to send data via select()
  SOCKETCAN_LOCAL void wait(const std::chrono::nanoseconds timeout) const
  {
    if (decltype(timeout)::zero() < timeout) {
      auto c_timeout = to_timeval(timeout);
      auto write_set = single_set(m_file_descriptor);
      // Wait
      if (0 == select(m_file_descriptor + 1, NULL, &write_set, NULL, &c_timeout)) {
        throw SocketCanTimeout{"$CAN Send Timeout"};
      }
      // lint --e{9130, 9123, 9125, 1924, 9126} NOLINT
      if (!FD_ISSET(m_file_descriptor, &write_set)) {
        throw SocketCanTimeout{"$CAN Send timeout"};
      }
    } else {
      // do nothing
      // auto write_set = single_set(m_file_descriptor);
    }
  }

  inline static bool m_first_init;
  inline static int8_t m_canfd_state;
  int32_t m_file_descriptor{};
  CanId m_default_id;
};  // class SocketCanSender

}  // namespace common
}  // namespace cyberdog

#endif  // PROTOCOL__CAN__SOCKET_CAN_SENDER_HPP_
