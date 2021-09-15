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

#ifndef CYBERDOG_UTILS__CAN__SOCKET_CAN_COMMON_HPP_
#define CYBERDOG_UTILS__CAN__SOCKET_CAN_COMMON_HPP_

#include <sys/select.h>
#include <sys/time.h>

#include <chrono>
#include <string>

namespace drivers
{
namespace socketcan
{

/// Bind a non-blocking CAN_RAW socket to the given interface
/// \param[in] interface The name of the interface to bind, must be smaller than IFNAMSIZ
/// \return The file descriptor bound to the given interface
/// \throw std::runtime_error If one of socket(), fnctl(), ioctl(), bind() failed
/// \throw std::domain_error If the provided interface name is too long
int32_t bind_can_socket(const std::string & interface);
/// Convert std::chrono duration to timeval (with microsecond resolution)
struct timeval to_timeval(const std::chrono::nanoseconds timeout) noexcept;
/// Create a fd_set for use with select() that only contains the specified file descriptor
fd_set single_set(int32_t file_descriptor) noexcept;

}  // namespace socketcan
}  // namespace drivers

#endif  // CYBERDOG_UTILS__CAN__SOCKET_CAN_COMMON_HPP_
