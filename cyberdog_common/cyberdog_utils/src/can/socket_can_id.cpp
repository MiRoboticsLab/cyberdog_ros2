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

#include <linux/can.h>  // for CAN typedef so I can static_assert it

#include <utility>

#include "cyberdog_utils/can/socket_can_id.hpp"

namespace drivers
{
namespace socketcan
{

//lint -e{9006} NOLINT false positive: this expression is compile time evaluated
static_assert(
  MAX_DATA_LENGTH == sizeof(std::declval<struct canfd_frame>().data),
  "Unexpected CAN frame data size");
static_assert(std::is_same<CanId::IdT, canid_t>::value, "Underlying type of CanId is incorrect");
constexpr CanId::IdT EXTENDED_MASK = CAN_EFF_FLAG;
constexpr CanId::IdT REMOTE_MASK = CAN_RTR_FLAG;
constexpr CanId::IdT ERROR_MASK = CAN_ERR_FLAG;
constexpr CanId::IdT EXTENDED_ID_MASK = CAN_EFF_MASK;
constexpr CanId::IdT STANDARD_ID_MASK = CAN_SFF_MASK;

////////////////////////////////////////////////////////////////////////////////
CanId::CanId(const IdT raw_id, const LengthT data_length)
: m_id{raw_id},
  m_data_length{data_length}
{
  (void)frame_type();  // just to throw
}
CanId::CanId(const IdT id, FrameType type, StandardFrame_)
: CanId{id, type, false} {}

CanId::CanId(const IdT id, FrameType type, ExtendedFrame_)
: CanId{id, type, true} {}

////////////////////////////////////////////////////////////////////////////////
CanId::CanId(const IdT id, FrameType type, bool is_extended)
{
  // Set extended bit
  if (is_extended) {
    (void)extended();
  }
  (void)frame_type(type);
  (void)identifier(id);
}

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::standard() noexcept
{
  //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
  m_id = m_id & (~EXTENDED_MASK);
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::extended() noexcept
{
  m_id = m_id | EXTENDED_MASK;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::error_frame() noexcept
{
  //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
  m_id = m_id & (~REMOTE_MASK);
  m_id = m_id | ERROR_MASK;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::remote_frame() noexcept
{
  //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
  m_id = m_id & (~ERROR_MASK);
  m_id = m_id | REMOTE_MASK;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::data_frame() noexcept
{
  //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
  m_id = m_id & (~ERROR_MASK);
  //lint -e{9126} NOLINT false positive: underlying type is unsigned long, and same as m_id
  m_id = m_id & (~REMOTE_MASK);
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::frame_type(const FrameType type)
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

////////////////////////////////////////////////////////////////////////////////
CanId & CanId::identifier(const IdT id)
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
////////////////////////////////////////////////////////////////////////////////
CanId::IdT CanId::get() const noexcept
{
  return m_id;
}
////////////////////////////////////////////////////////////////////////////////
bool CanId::is_extended() const noexcept
{
  return (m_id & EXTENDED_MASK) == EXTENDED_MASK;
}
////////////////////////////////////////////////////////////////////////////////
CanId::IdT CanId::identifier() const noexcept
{
  const auto mask = is_extended() ? EXTENDED_ID_MASK : STANDARD_ID_MASK;
  return m_id & mask;
}
////////////////////////////////////////////////////////////////////////////////
CanId::LengthT CanId::length() const noexcept
{
  return m_data_length;
}
////////////////////////////////////////////////////////////////////////////////
FrameType CanId::frame_type() const
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

}  // namespace socketcan
}  // namespace drivers
