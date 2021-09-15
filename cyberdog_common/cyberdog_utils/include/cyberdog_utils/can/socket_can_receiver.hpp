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

#include <array>
#include <chrono>
#include <cstring>
#include <string>

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

  /// Receive CAN data
  /// \param[out] data A buffer to be written with data bytes. Must be at least 8 bytes in size
  /// \param[in] timeout Maximum duration to wait for data on the file descriptor. Negative
  ///                    durations are treated the same as zero timeout
  /// \return The CanId for the received can_frame, with length appropriately populated
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error on other errors
  CanId receive(
    void * const data,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const;
  /// Receive typed CAN data. Slightly less efficient than untyped interface; has extra copy and
  /// branches
  /// \tparam Type of data to receive, must be 8 bytes or smaller
  /// \param[out] data A buffer to be written with data bytes. Must be at least 8 bytes in size
  /// \param[in] timeout Maximum duration to wait for data on the file descriptor. Negative
  ///                    durations are treated the same as zero timeout
  /// \return The CanId for the received can_frame, with length appropriately populated
  /// \throw SocketCanTimeout On timeout
  /// \throw std::runtime_error If received data would not fit into provided type
  /// \throw std::runtime_error on other errors
  template<typename T, typename = std::enable_if_t<!std::is_pointer<T>::value>>
  CanId receive(
    T & data,
    const std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) const
  {
    static_assert(sizeof(data) <= MAX_DATA_LENGTH, "Data type too large for CAN");
    std::array<uint8_t, MAX_DATA_LENGTH> data_raw{};
    const auto ret = receive(&data_raw[0U], timeout);
    if (ret.length() != sizeof(data)) {
      throw std::runtime_error{"Received CAN data is of size incompatible with provided type!"};
    }
    (void)std::memcpy(&data, &data_raw[0U], ret.length());
    return ret;
  }

private:
  // Wait for file descriptor to be available to send data via select()
  SOCKETCAN_LOCAL void wait(const std::chrono::nanoseconds timeout) const;

  int32_t m_file_descriptor;
};  // class SocketCanReceiver

}  // namespace socketcan
}  // namespace drivers

#endif  // CYBERDOG_UTILS__CAN__SOCKET_CAN_RECEIVER_HPP_
