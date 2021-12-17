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
#ifndef PROTOCOL__CAN__SOCKET_CAN_ID_HPP_
#define PROTOCOL__CAN__SOCKET_CAN_ID_HPP_

#include <linux/can.h>
#include <utility>
#include <stdexcept>

#include "visibility_control.hpp"

namespace cyberdog
{
namespace common
{
using IdT = uint32_t;
using LengthT = uint32_t;

constexpr std::size_t MAX_DATA_LENGTH = 64U;
//lint -e{9006} NOLINT false positive: this expression is compile time evaluated
static_assert(
  MAX_DATA_LENGTH == sizeof(std::declval<struct canfd_frame>().data),
  "Unexpected CAN frame data size");
static_assert(std::is_same<IdT, canid_t>::value, "Underlying type of CanId is incorrect");
constexpr IdT EXTENDED_MASK = CAN_EFF_FLAG;
constexpr IdT REMOTE_MASK = CAN_RTR_FLAG;
constexpr IdT ERROR_MASK = CAN_ERR_FLAG;
constexpr IdT EXTENDED_ID_MASK = CAN_EFF_MASK;
constexpr IdT STANDARD_ID_MASK = CAN_SFF_MASK;

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
  // Default constructor: standard data frame with id 0
  CanId() = default;
  /// Directly set id, blindly taking whatever bytes are given
  explicit CanId(const IdT raw_id, const LengthT data_length = 0U)
  : m_id{raw_id},
    m_data_length{data_length}
  {
    (void)frame_type();  // just to throw
  }
  /// Sets ID
  /// \throw std::domain_error if id would get truncated
  CanId(const IdT id, FrameType type, StandardFrame_)
  : CanId{id, type, false} {}
  /// Sets ID
  /// \throw std::domain_error if id would get truncated
  CanId(const IdT id, FrameType type, ExtendedFrame_)
  : CanId{id, type, true} {}

  /// Sets bit 31 to 0
  CanId & standard() noexcept
  {
    //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
    m_id = m_id & (~EXTENDED_MASK);
    return *this;
  }
  /// Sets bit 31 to 1
  CanId & extended() noexcept
  {
    m_id = m_id | EXTENDED_MASK;
    return *this;
  }
  /// Sets bit 29 to 1, and bit 30 to 0
  CanId & error_frame() noexcept
  {
    //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
    m_id = m_id & (~REMOTE_MASK);
    m_id = m_id | ERROR_MASK;
    return *this;
  }
  /// Sets bit 29 to 0, and bit 30 to 1
  CanId & remote_frame() noexcept
  {
    //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
    m_id = m_id & (~ERROR_MASK);
    m_id = m_id | REMOTE_MASK;
    return *this;
  }

  /// Clears bits 29 and 30 (sets to 0)
  CanId & data_frame() noexcept
  {
    //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
    m_id = m_id & (~ERROR_MASK);
    //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
    m_id = m_id & (~REMOTE_MASK);
    return *this;
  }
  /// Sets the type accordingly
  CanId & frame_type(const FrameType type)
  {
    switch (type) {
      case FrameType::DATA:
        (void)data_frame();
        break;
      case FrameType::ERROR:
        (void)error_frame();
        break;
      case FrameType::REMOTE:
        (void)remote_frame();
        break;
      default:
        throw std::logic_error{"CanId: No such type"};
    }
    return *this;
  }
  /// Sets leading bits
  /// \throw std::domain_error If id would get truncated, 11 bits for Standard, 29 bits for Extended
  CanId & identifier(const IdT id)
  {
    // Can specification: http://esd.cs.ucr.edu/webres/can20.pdf
    // says "The 7 most significant bits cannot all be recessive (value of 1)", pg 11
    constexpr auto MAX_EXTENDED = 0x1FBF'FFFFU;
    constexpr auto MAX_STANDARD = 0x07EFU;
    static_assert(MAX_EXTENDED <= EXTENDED_ID_MASK, "Max extended id value is wrong");
    static_assert(MAX_STANDARD <= STANDARD_ID_MASK, "Max extended id value is wrong");
    const auto max_id = is_extended() ? MAX_EXTENDED : MAX_STANDARD;
    if (max_id < id) {
      throw std::domain_error{"CanId would be truncated!"};
    }
    // Clear and set
    //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
    m_id = m_id & (~EXTENDED_ID_MASK);  // clear ALL ID bits, not just standard bits
    m_id = m_id | id;
    return *this;
  }

  /// Get just the can_id bits
  IdT identifier() const noexcept
  {
    const auto mask = is_extended() ? EXTENDED_ID_MASK : STANDARD_ID_MASK;
    return m_id & mask;
  }
  /// Get the whole id value
  IdT get() const noexcept
  {
    return m_id;
  }
  /// Check if frame is extended
  bool is_extended() const noexcept
  {
    return (m_id & EXTENDED_MASK) == EXTENDED_MASK;
  }
  /// Check frame type
  /// \throw std::domain_error If bits are in an inconsistent state
  FrameType frame_type() const
  {
    const auto is_error = (m_id & ERROR_MASK) == ERROR_MASK;
    const auto is_remote = (m_id & REMOTE_MASK) == REMOTE_MASK;
    if (is_error && is_remote) {
      throw std::domain_error{"CanId has both bits 29 and 30 set! Inconsistent!"};
    }

    if (is_error) {
      return FrameType::ERROR;
    }
    if (is_remote) {
      return FrameType::REMOTE;
    }
    return FrameType::DATA;
  }
  /// Get the length of the data; only nonzero on received data
  LengthT length() const noexcept
  {
    return m_data_length;
  }

private:
  SOCKETCAN_LOCAL CanId(const IdT id, FrameType type, bool is_extended)
  {
    // Set extended bit
    if (is_extended) {
      (void)extended();
    }
    (void)frame_type(type);
    (void)identifier(id);
  }

  IdT m_id{};
  LengthT m_data_length{};
};  // class CanId
}  // namespace common
}  // namespace cyberdog

#endif  // PROTOCOL__CAN__SOCKET_CAN_ID_HPP_
