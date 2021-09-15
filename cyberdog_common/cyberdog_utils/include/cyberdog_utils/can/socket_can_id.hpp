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
#ifndef CYBERDOG_UTILS__CAN__SOCKET_CAN_ID_HPP_
#define CYBERDOG_UTILS__CAN__SOCKET_CAN_ID_HPP_

#include <stdexcept>

#include "cyberdog_utils/can/visibility_control.hpp"

namespace drivers
{
namespace socketcan
{

constexpr std::size_t MAX_DATA_LENGTH = 64U;
/// Special error for timeout
class SOCKETCAN_PUBLIC SocketCanTimeout : public std::runtime_error
{
public:
  explicit SocketCanTimeout(const char * const what)
  : runtime_error{what} {}
};  // class SocketCanTimeout

enum class FrameType : uint32_t
{
  DATA,
  ERROR,
  REMOTE
  // SocketCan doesn't support Overload frame directly?
};  // enum class FrameType

/// Tag for standard frame
struct StandardFrame_ {};
//lint -e{1502} NOLINT It's a tag
constexpr StandardFrame_ StandardFrame;
/// Tag for extended frame
struct ExtendedFrame_ {};
//lint -e{1502} NOLINT It's a tag
constexpr ExtendedFrame_ ExtendedFrame;

/// A wrapper around can_id_t to make it a little more C++-y
/// WARNING: I'm assuming the 0th bit is the MSB aka the leftmost bit
class SOCKETCAN_PUBLIC CanId
{
public:
  using IdT = uint32_t;
  using LengthT = uint32_t;
  // Default constructor: standard data frame with id 0
  CanId() = default;
  /// Directly set id, blindly taking whatever bytes are given
  explicit CanId(const IdT raw_id, const LengthT data_length = 0U);
  /// Sets ID
  /// \throw std::domain_error if id would get truncated
  CanId(const IdT id, FrameType type, StandardFrame_);
  /// Sets ID
  /// \throw std::domain_error if id would get truncated
  CanId(const IdT id, FrameType type, ExtendedFrame_);

  /// Sets bit 31 to 0
  CanId & standard() noexcept;
  /// Sets bit 31 to 1
  CanId & extended() noexcept;
  /// Sets bit 29 to 1, and bit 30 to 0
  CanId & error_frame() noexcept;
  /// Sets bit 29 to 0, and bit 30 to 1
  CanId & remote_frame() noexcept;
  /// Clears bits 29 and 30 (sets to 0)
  CanId & data_frame() noexcept;
  /// Sets the type accordingly
  CanId & frame_type(const FrameType type);
  /// Sets leading bits
  /// \throw std::domain_error If id would get truncated, 11 bits for Standard, 29 bits for Extended
  CanId & identifier(const IdT id);

  /// Get just the can_id bits
  IdT identifier() const noexcept;
  /// Get the whole id value
  IdT get() const noexcept;
  /// Check if frame is extended
  bool is_extended() const noexcept;
  /// Check frame type
  /// \throw std::domain_error If bits are in an inconsistent state
  FrameType frame_type() const;
  /// Get the length of the data; only nonzero on received data
  LengthT length() const noexcept;

private:
  SOCKETCAN_LOCAL CanId(const IdT id, FrameType type, bool is_extended);

  IdT m_id{};
  LengthT m_data_length{};
};  // class CanId
}  // namespace socketcan
}  // namespace drivers

#endif  // CYBERDOG_UTILS__CAN__SOCKET_CAN_ID_HPP_
