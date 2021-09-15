// Copyright 2019 the Autoware Foundation
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

#include "cyberdog_utils/can/socket_can_common.hpp"
#include "cyberdog_utils/can/socket_can_receiver.hpp"

#include <unistd.h>  // for close()
#include <sys/select.h>
#include <sys/socket.h>
#include <linux/can.h>

#include <cstring>
#include <string>

namespace drivers
{
namespace socketcan
{

////////////////////////////////////////////////////////////////////////////////
SocketCanReceiver::SocketCanReceiver(const std::string & interface)
: m_file_descriptor{bind_can_socket(interface)}
{
}

////////////////////////////////////////////////////////////////////////////////
SocketCanReceiver::~SocketCanReceiver() noexcept
{
  // Can't do anything on error; in fact generally shouldn't on close() error
  (void)close(m_file_descriptor);
}

////////////////////////////////////////////////////////////////////////////////
void SocketCanReceiver::wait(const std::chrono::nanoseconds timeout) const
{
  if (decltype(timeout)::zero() < timeout) {
    auto c_timeout = to_timeval(timeout);
    auto read_set = single_set(m_file_descriptor);
    // Wait
    if (0 == select(m_file_descriptor + 1, &read_set, NULL, NULL, &c_timeout)) {
      throw SocketCanTimeout{"CAN Receive Timeout"};
    }
    //lint --e{9130, 1924, 9123, 9125, 1924, 9126} NOLINT
    if (!FD_ISSET(m_file_descriptor, &read_set)) {
      throw SocketCanTimeout{"CAN Receive timeout"};
    }
  } else {
    auto read_set = single_set(m_file_descriptor);
    // Wait
    if (0 == select(m_file_descriptor + 1, &read_set, NULL, NULL, NULL)) {
      throw SocketCanTimeout{"CAN Receive Timeout"};
    }
    //lint --e{9130, 1924, 9123, 9125, 1924, 9126} NOLINT
    if (!FD_ISSET(m_file_descriptor, &read_set)) {
      throw SocketCanTimeout{"CAN Receive timeout"};
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
CanId SocketCanReceiver::receive(void * const data, const std::chrono::nanoseconds timeout) const
{
  wait(timeout);
  // Read
  struct can_frame frame;
  const auto nbytes = read(m_file_descriptor, &frame, sizeof(frame));
  // Checks
  if (nbytes < 0) {
    throw std::runtime_error{strerror(errno)};
  }
  if (static_cast<std::size_t>(nbytes) < sizeof(frame)) {
    throw std::runtime_error{"read: incomplete CAN frame"};
  }
  if (static_cast<std::size_t>(nbytes) != sizeof(frame)) {
    throw std::logic_error{"Message was wrong size"};
  }
  // Write
  const auto data_length = static_cast<CanId::LengthT>(frame.can_dlc);
  (void)std::memcpy(data, static_cast<void *>(&frame.data[0U]), data_length);
  return CanId{frame.can_id, data_length};
}

}  // namespace socketcan
}  // namespace drivers
